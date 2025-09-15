# -*- coding: utf-8 -*-
"""
Conditional training routine for the probe–D7 DBI free energy functional.

This module exposes a single public function, ``train_conditional``, which
implements the batch‑training loop for the conditional network
:class:`pinn_d7.models.LNetworkM`.  It is intentionally self‑contained so that
it can be launched from ``scripts/run_mass_cond.py`` without modifying any
other part of the code‑base.
"""
from __future__ import annotations

import time
import pathlib
from typing import Optional, List, Dict, Tuple
import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np

from .models import LNetworkM
from .losses import dbi_energy_map_conditional


# ---------------------------------------------------------------------------
# ρ‑grid construction
# ---------------------------------------------------------------------------


def build_rho_grid(
    ρ_max: float = 20.0,
    N_grid: int = 2000,
    cut: float = 5.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Construct a non‑uniform collocation grid on :math:`\\rho \\in [0, \\rho_{max}]`.

    The interval :math:`[0, \\text{cut}]` is sampled densely (≈½ of the points)
    while :math:`[\\text{cut}, \\rho_{max}]` is sampled more coarsely.
    """
    N_dense = N_grid // 2
    N_coarse = N_grid - N_dense

    ρ_dense = torch.linspace(0.0, cut, N_dense, device=device)
    ρ_coarse = torch.linspace(cut, ρ_max, N_coarse, device=device)[1:]  # avoid duplicate “cut”
    return torch.cat([ρ_dense, ρ_coarse]).unsqueeze(1)  # (N,1)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def train_conditional(
    ρ_max: float = 20.0,
    N_grid: int = 2000,
    cut: float = 5.0,
    m_min: float = 0.0,
    m_max: float = 0.8,
    batch_m: int = 1,
    lr: float = 5e-5,
    n_epochs: int = 30_000,
    print_every: int = 500,
    diagnostics_every: int = 1000,
    checkpoint_every: int = 1_000,
    checkpoint_dir: str | pathlib.Path = "checkpoints_hot",
    resume_from: Optional[str | pathlib.Path] = None,
    device: Optional[torch.device] = None,
    live_plot: bool = False,
    potential: str = "hot",
    ref_file: str = 'data/HotFree0_0_9.csv',
    hidden: int = 16,
    depth: int = 2,
    seed: Optional[int] = None,
    init_strategy: str = 'physics_informed',
    mae_eval_fn = None,
    condensate_mae_eval_fn = None
) -> Tuple[nn.Module, List[Dict[str, object]], Optional[Dict[str, List]], Optional[Dict[str, List]]]:
    """
    Train the conditional network ``l(ρ, m)`` against the hot DBI energy.

    Parameters
    ----------
    ρ_max, N_grid, cut
        Parameters of the collocation grid.
    m_min, m_max
        Range of quark masses sampled during training.
    batch_m
        *Mini‑batch* size, i.e. how many different masses are used per SGD step.
    lr
        Learning rate.
    n_epochs
        Number of optimisation steps **after** any loaded checkpoint.
    print_every, diagnostics_every
        Console logging frequency and diagnostics sampling frequency.
    checkpoint_every, checkpoint_dir
        Checkpoint cadence and output directory.
    resume_from
        Path to a checkpoint to resume from.  If *None*, training starts fresh.
    device
        Torch device.  If *None*, chosen automatically.
    live_plot
        If True, open an interactive Matplotlib window (terminal mode) and
        refresh the F(m) diagnostic curve every ``diagnostics_every`` steps.

    Returns
    -------
    model
        Trained network.
    history
        List with one entry per *diagnostics* event containing:
        ``{"epoch", "loss", "m_eval", "F_eval"}``.
    """
    # –– device & dtype ------------------------------------------------------
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    torch.set_default_dtype(torch.float64)
    
    # –– set random seed if provided ------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # –– grid & model --------------------------------------------------------
    ρ_grid = build_rho_grid(ρ_max, N_grid, cut, device)
    model = LNetworkM(ρ_max, hidden=hidden, depth=depth).to(device)
    
    # Apply custom initialization strategy
    if init_strategy != 'physics_informed':  # physics_informed is default in __init__
        model.reinitialize_weights(init_strategy)
    
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # –– optional resume -----------------------------------------------------
    start_epoch = 0
    if resume_from is not None:
        ckpt_path = pathlib.Path(resume_from)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found.")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt["epoch"])
        print(f"✔ Resumed from {ckpt_path} (epoch {start_epoch})")

    # –– I/O setup -----------------------------------------------------------
    ckpt_root = pathlib.Path(checkpoint_dir)
    ckpt_root.mkdir(exist_ok=True)
    history: List[Dict[str, object]] = []
    
    # MAE tracking arrays
    mae_history = []
    mae_epochs = []
    
    # Condensate MAE tracking arrays
    condensate_mae_history = []
    condensate_mae_epochs = []
    
    Fs = np.loadtxt(ref_file, delimiter=",")
    sel = (Fs[:, 0] >= m_min) & (Fs[:, 0] <= m_max)
    Fs = Fs[sel, :]

    # live‑plot handle
    fig_diag = ax_diag = None
    if live_plot:
        plt.ion()  # interactive plotting for terminal sessions

    # –– training loop -------------------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):
        if epoch % 5 == 0:
            extremal = m_min if torch.rand(()) < 0.5 else m_max
            m_batch = torch.full((batch_m, 1), extremal,
                                 device=device, dtype=torch.float64)
        else:
            m_batch = (m_min + (m_max - m_min) *
                       torch.rand(batch_m, 1, device=device)).double()
        
        dbi_energy = dbi_energy_map_conditional.get(potential)
        loss = dbi_energy(model, ρ_grid, m_batch, reduce="mean")

        optim.zero_grad()
        loss.backward()
        optim.step()

        # ---- console logging ----------------------------------------------
        if epoch % print_every == 0 or epoch == 1:
            print(f"[{epoch:6d}/{start_epoch + n_epochs}] ⟨DBI⟩ = {loss.item():.6e}")

        # ---- diagnostics ---------------------------------------------------
        if epoch % diagnostics_every == 0 or epoch == 1:
            m_eval = torch.linspace(m_min, m_max, 60, device=device).unsqueeze(1)
            F_eval = dbi_energy(model, ρ_grid, m_eval, reduce="none").detach()
            history.append(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "m_eval": m_eval.cpu(),
                    "F_eval": F_eval.cpu(),
                }
            )

            # ---- live plotting -------------------------------------------------
            if live_plot:
                if fig_diag is None:
                    fig_diag, ax_diag = plt.subplots(figsize=(5, 4))
                ax_diag.clear()
                ax_diag.plot(m_eval.cpu(), F_eval.cpu(), lw=2)
                ax_diag.plot(Fs[:,0], Fs[:,1], label="ODE")
                ax_diag.set_xlabel("mass $m$")
                ax_diag.set_ylabel("free energy $F(m)$")
                ax_diag.set_title(f"DBI free energy  (epoch {epoch})")
                ax_diag.grid(alpha=.4)
                fig_diag.tight_layout()
                fig_diag.canvas.draw()
                fig_diag.canvas.flush_events()

        # ---- checkpoint ----------------------------------------------------
        last_iter = epoch == start_epoch + n_epochs
        if epoch % checkpoint_every == 0 or last_iter:
            ckpt_file = ckpt_root / f"ckpt_epoch_{epoch:06d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "loss": loss.item(),
                },
                ckpt_file,
            )
            print(f"✔ Checkpoint saved to {ckpt_file}")
            
            # ---- MAE evaluation at checkpoint --------------------------------
            if mae_eval_fn is not None:
                mae = mae_eval_fn(model, Fs, ρ_max, N_grid, cut, device, potential)
                mae_history.append(mae)
                mae_epochs.append(epoch)
            
            # ---- Condensate MAE evaluation at checkpoint ---------------------
            if condensate_mae_eval_fn is not None:
                condensate_mae = condensate_mae_eval_fn(model, ρ_max, N_grid, cut, m_min, m_max, device, potential)
                if condensate_mae is not None:
                    condensate_mae_history.append(condensate_mae)
                    condensate_mae_epochs.append(epoch)

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining finished in {elapsed:.1f} min.")
    
    # Add MAE history to the return
    mae_data = {"epochs": mae_epochs, "mae_values": mae_history} if mae_history else None
    condensate_mae_data = {"epochs": condensate_mae_epochs, "mae_values": condensate_mae_history} if condensate_mae_history else None
    return model, history, mae_data, condensate_mae_data


def evaluate_condensate(
    model: nn.Module,
    ρ_max=20.0,
    N_grid=2000,
    cut=5.0,
    m_min=0.0,
    m_max=1.0,
    potential: str = "magnetic",
    device: Optional[torch.device] = None,
    n_epochs: int = 30000,
    depth: int = 2,
    hidden: int = 16,
) -> torch.Tensor:
    """
    Evaluate the DBI condensate for a range of masses.

    Parameters
    ----------
    model
        The trained conditional network.
    ρ_grid
        Collocation grid (on device).
    m_eval
        Masses to evaluate the condensate at (on device).
    potential
        The type of potential, either "hot" or "magnetic".

    -------
    plots comparision of the condensate from ANN with the ODE result.
    -------
    """
    dbi_energy = dbi_energy_map_conditional.get(potential)
    
    m_eval = torch.linspace(m_min, m_max, 101, device=device).unsqueeze(1)
    m_eval.requires_grad_(True)
    
    
    ρ_grid = build_rho_grid(ρ_max, N_grid, cut, device)

    I_eval = dbi_energy(model, ρ_grid, m_eval, reduce="none")

    dIdm = torch.autograd.grad(I_eval.sum(), m_eval, create_graph=False)[0]

    cs = np.loadtxt("data/MagCondensate.csv", delimiter=",")

    plt.figure(figsize=(4,3.2))
    plt.plot(m_eval.cpu().detach()[:, 0], dIdm.cpu().detach(), '-', lw=2, label="ANN")
    plt.xlabel("mass $m$")
    plt.ylabel("$dF/dm$")
    plt.title(f"Fundamental condensate (magnetic)")
    plt.plot(cs[:, 0], cs[:, 1], 'x', lw=2, label="ODE", color='red')
    plt.grid(alpha=.4)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/c_vs_m_magnetic_{n_epochs}_D{depth}N{hidden}R{N_grid}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)   # display without blocking the script
    plt.pause(0.1)          # give the GUI event loop time to draw
    