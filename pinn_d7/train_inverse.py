# -*- coding: utf-8 -*-
"""
Inverse-problem training: simultaneous reconstruction of
  • the embedding  l(ρ , m)   (via a conditional network LNetworkM)
  • the radial potential V(r) (via VNetwork)

The loss is a mixed policy:
    - even steps   → only V is updated
    - odd  steps   → only l is updated

The data driven term uses an interpolator Iapp(m) built from the
reference free-energy file supplied by the user.

This module intentionally mirrors the API of `train_conditional.py` so
that it can be launched from `scripts/run_inverse.py` without touching
the rest of the code-base.
"""
from __future__ import annotations

import time
import pathlib
import math
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from .train_conditional import build_rho_grid           # <- already defined there
from .models          import LNetworkM, VNetwork
from .losses          import dbi_energy_V


# default data-files ------------------------------------------------
_ref_files = {
    "hot":      "data/HotFree0_0_9.csv",
    "magnetic": "data/MagFree.csv",
}

def get_diagnostics_curve(potential: str = "hot", m_min: float = 0.0, m_max: float = 0.8):
    ref_file = _ref_files[potential]
    Fs  = np.loadtxt(ref_file, delimiter=",")
    sel = (Fs[:, 0] >= m_min) & (Fs[:, 0] <= m_max)
    return Fs[sel, :]

# ---------------------------------------------------------------------
# helper : 1-D linear interpolator wrapped for torch tensors
# ---------------------------------------------------------------------
def make_Iapp(Fs: np.ndarray,
              m_min: float,
              m_max: float,
              device: torch.device | str = "cpu"):
    """
    Build a *vectorised* torch-friendly interpolator Iapp(m).

    Outside the [m_min , m_max] range the interpolator saturates to the
    nearest tabulated value (same behaviour as NumPy’s `interp`).
    """
    
    m_tab = Fs[:, 0]
    F_tab = Fs[:, 1]

    def _I(m_tensor: torch.Tensor) -> torch.Tensor:
        m_np = m_tensor.detach().cpu().numpy()
        F_np = np.interp(m_np, m_tab, F_tab)
        return torch.from_numpy(F_np).to(m_tensor)

    return _I


# ---------------------------------------------------------------------
# training routine
# ---------------------------------------------------------------------
def train_inverse(
        ρ_max          : float               = 20.0,
        N_grid         : int                 = 2_000,
        cut            : float               = 5.0,
        m_min          : float               = 0.1,
        m_max          : float               = 0.6,
        batch_m        : int                 = 1,
        lr_L           : float               = 1e-4,
        lr_V           : float               = 5e-4,
        n_epochs       : int                 = 20_000,
        print_every    : int                 = 500,
        diagnostics_every: int               = 1000,
        checkpoint_every : int               = 1_000,
        checkpoint_dir : str | pathlib.Path  = "checkpoints_inverse",
        resume_from    : Optional[str | pathlib.Path] = None,
        device         : Optional[torch.device] = None,
        live_plot      : bool                = True,
        potential      : str                 = "hot",
        hidden         : int                 = 16,
        depth          : int                 = 2
) -> Tuple[nn.Module, nn.Module, List[Dict[str, object]]]:
    """
    Jointly train (l , V).

    Returns
    -------
    l_model , v_model , history
    """
    # –– device & dtype ----------------------------------------------------
    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    torch.set_default_dtype(torch.float64)

    # –– grids -------------------------------------------------------------
    ρ_grid = build_rho_grid(ρ_max, N_grid, cut, device)
    r_max  = math.sqrt(ρ_max**2 + m_max**2)
    r_grid = torch.linspace(math.sqrt(0.5), r_max, 201,
                            device=device).unsqueeze(1)

    # –– models & optimisers ----------------------------------------------
    l_model = LNetworkM(ρ_max, hidden=hidden, depth=depth).to(device)
    v_model = VNetwork(r_max, hidden=hidden, depth=depth).to(device)

    opt_L = torch.optim.Adam(l_model.parameters(), lr=lr_L)
    opt_V = torch.optim.Adam(v_model.parameters(), lr=lr_V)

    # –– interpolate target free-energy -----------------------------------
    
    Fs = get_diagnostics_curve(potential=potential, m_min=m_min, m_max=m_max)
    
    Iapp = make_Iapp(Fs, m_min, m_max, device)

    # –– resume (optional) -------------------------------------------------
    start_epoch = 0
    if resume_from is not None:
        ckpt_path = pathlib.Path(resume_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found.")
        ckpt = torch.load(ckpt_path, map_location=device)
        l_model.load_state_dict(ckpt["model_state"])
        v_model.load_state_dict(ckpt["v_model_state"])
        opt_L.load_state_dict(ckpt["optim_state_L"])
        opt_V.load_state_dict(ckpt["optim_state_V"])
        start_epoch = int(ckpt["epoch"])
        print(f"✔ Resumed from {ckpt_path} (epoch {start_epoch})")

    # –– I/O ----------------------------------------------------------------
    ckpt_root = pathlib.Path(checkpoint_dir)
    ckpt_root.mkdir(exist_ok=True)

    # diagnostics container
    history : List[Dict[str, object]] = []

    # live plot handle
    fig_diag = ax_diag = None
    if live_plot:
        plt.ion()

    # –– training loop ------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):

        # sample mini-batch of masses
        m_batch = (m_min + (m_max - m_min) *
                   torch.rand(batch_m, 1, device=device)).double()

        # ------------------------------------------------------------------
        # alternating optimisation
        # ------------------------------------------------------------------
        dbi = dbi_energy_V(l_model, v_model, ρ_grid, m_batch, reduce="none")

        if epoch % 2 == 0:                 # update V
            target = Iapp(m_batch[:, 0])
            loss   = 100.0 * ((dbi - target)**2).mean()
            opt_V.zero_grad()
            loss.backward()
            opt_V.step()
        else:                              # update l
            loss = dbi.mean()
            opt_L.zero_grad()
            loss.backward()
            opt_L.step()

        # ------------------------------------------------------------------
        # console log
        # ------------------------------------------------------------------
        if epoch % print_every == 0 or epoch == 1:
            print(f"[{epoch:6d}/{start_epoch + n_epochs}]  loss = {loss.item():.6e}")

        # ------------------------------------------------------------------
        # diagnostics
        # ------------------------------------------------------------------
        if epoch % diagnostics_every == 0 or epoch == 1:
            m_eval = torch.linspace(m_min, m_max, 60,
                                        device=device).unsqueeze(1)
            I_eval = dbi_energy_V(l_model, v_model,
                                    ρ_grid, m_eval, reduce="none")
            history.append({
                "epoch":  epoch,
                "loss":   loss.item(),
                "m_eval": m_eval.cpu(),
                "I_eval": I_eval.cpu(),
            })

            # ------------- live plotting (F(m)  +  V(r)) -------------------------
            if live_plot:
                # first time → create the figure with 2 sub-plots
                if fig_diag is None:
                    fig_diag, (ax_F, ax_V) = plt.subplots(1, 2, figsize=(9, 4))
                    fig_diag.tight_layout()

                # left panel :  F(m)
                ax_F.clear()
                ax_F.plot(m_eval.cpu().detach(), I_eval.cpu().detach(), lw=2, label="ANN")
                ax_F.plot(Fs[:, 0], Fs[:, 1], lw=2, label="ODE", linestyle=':', color='red')
        
                ax_F.set_xlabel("mass $m$")
                ax_F.set_ylabel("free energy $F(m)$")
                ax_F.set_title(f"DBI free energy  (epoch {epoch})")
                ax_F.grid(alpha=.4)

                # right panel :  V(r)
                ax_V.clear()
                with torch.no_grad():
                    V_rec = v_model(r_grid).cpu().squeeze()
                ax_V.plot(r_grid.cpu(), V_rec, lw=2, label="ANN")
                # optional: theoretical/target potential
                if potential.lower() == "hot":
                    ax_V.plot(r_grid.cpu(),
                        1.0 - 1.0 / (r_grid.cpu()**8) / 16.0,
                        ":", color="red", lw=1.5, label=r"$1-\frac{1}{16 r^{8}}$")
                else:
                    assert potential.lower() == "magnetic"
                    ax_V.plot(r_grid.cpu(),
                        torch.sqrt(1.0 + 1.0 / (r_grid.cpu()**4)),
                        ":", color="red", lw=1.5, label=r"$\sqrt{1+\frac{1}{r^{4}}}$")
                ax_V.set_xlabel("radial coordinate $r$")
                ax_V.set_ylabel("potential $V(r)$")
                ax_V.set_title("Recovered potential")
                ax_V.grid(alpha=.4)
                ax_V.legend(loc="best", fontsize=8)

                # draw / refresh
                fig_diag.canvas.draw()
                fig_diag.canvas.flush_events()
            # ------------- end of live plotting --------------------------------
        # ------------------------------------------------------------------
        # checkpoint
        # ------------------------------------------------------------------
        last_iter = epoch == start_epoch + n_epochs
        if epoch % checkpoint_every == 0 or last_iter:
            ckpt_file = ckpt_root / f"ckpt_epoch_{epoch:06d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state":     l_model.state_dict(),
                    "v_model_state":   v_model.state_dict(),
                    "optim_state_L":   opt_L.state_dict(),
                    "optim_state_V":   opt_V.state_dict(),
                    "loss":            loss.item(),
                },
                ckpt_file,
            )
            print(f"✔ Checkpoint saved to {ckpt_file}")

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining finished in {elapsed:.1f} min.")
    return l_model, v_model, history