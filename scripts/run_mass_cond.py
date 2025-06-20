#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conditional training of the probe D7-brane DBI action with mass parameter.

Parameters
----------
...
live_plot
    If True, open an interactive Matplotlib window (terminal mode) and
    refresh the F(m) diagnostic curve every ``diagnostics_every`` steps.
"""

# Work‑around for macOS duplicated OpenMP runtime issue
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Optional, Tuple, List, Dict
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def train_conditional(
    ρ_max: float = 20.0,
    N_grid: int = 2000,
    cut: float = 5.0,
    m_min: float = 0.0,
    m_max: float = 0.8,
    batch_m: int = 1,
    lr: float = 1e-5,
    n_epochs: int = 30_000,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
    device: Optional[torch.device] = None,
    live_plot: bool = False,
) -> Tuple[nn.Module, List[Dict[str, object]]]:

    history: List[Dict[str, object]] = []

    # live‑plot handle
    fig_diag = ax_diag = None
    if live_plot:
        plt.ion()  # interactive plotting for terminal sessions

    for epoch in range(1, n_epochs + 1):
        # ... training steps ...

        diagnostics_every = 1000  # example value

        if epoch % diagnostics_every == 0 or epoch == 1:
            # compute diagnostics
            m_eval = torch.linspace(m_min, m_max, 100, device=device)
            F_eval = torch.zeros_like(m_eval)  # placeholder for actual eval

            history.append({"epoch": epoch, "m_eval": m_eval, "F_eval": F_eval})

            # ---- live plotting -------------------------------------------------
            if live_plot:
                if fig_diag is None:
                    fig_diag, ax_diag = plt.subplots(figsize=(5, 4))
                ax_diag.clear()
                ax_diag.plot(m_eval.cpu(), F_eval.cpu(), lw=2)
                ax_diag.set_xlabel("mass $m$")
                ax_diag.set_ylabel("free energy $F(m)$")
                ax_diag.set_title(f"DBI free energy  (epoch {epoch})")
                ax_diag.grid(alpha=.4)
                fig_diag.tight_layout()
                fig_diag.canvas.draw()
                fig_diag.canvas.flush_events()

    return nn.Module(), history
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI launcher for conditional probe–D7 training (hot DBI potential).

Example
-------
$ python run_mass_cond.py --epochs 40000 --lr 5e-6 --batch 2 --m_min 0.0 --m_max 0.8 --live
"""

# ---------------------------------------------------------------------------
# ENV‑VAR WORKAROUND (must be *before* torch import)
# ---------------------------------------------------------------------------
import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OpenMP workaround for macOS

# ---------------------------------------------------------------------------
# stdlib imports
# ---------------------------------------------------------------------------
import argparse
import sys
import pathlib

# ---------------------------------------------------------------------------
# third‑party imports
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")  # interactive backend for terminal use
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# ensure the repository root is on sys.path so that `pinn_d7` is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pinn_d7.train_conditional import train_conditional


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_mass_cond.py",
        description="Conditional PINN training for l(ρ, m) with the hot DBI action.",
    )
    p.add_argument("--epochs", type=int, default=30_000, help="number of training epochs")
    p.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    p.add_argument("--batch", type=int, default=1, help="masses per SGD step")
    p.add_argument("--rho_max", type=float, default=20.0, help="UV cutoff ρ_max")
    p.add_argument("--N_grid", type=int, default=2000, help="total collocation points")
    p.add_argument("--cut", type=float, default=5.0, help="boundary between dense & coarse zones")
    p.add_argument("--m_min", type=float, default=0.0, help="minimum mass in training range")
    p.add_argument("--m_max", type=float, default=0.8, help="maximum mass in training range")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_hot",
        help="directory in which checkpoints are saved",
    )
    p.add_argument("--resume", type=str, default=None, help="resume from checkpoint file")
    p.add_argument("--plot", action="store_true", help="plot F(m) at the end")
    p.add_argument("--live", action="store_true", help="live‑update F(m) during training")
    p.add_argument("--wait", action="store_true",
                   help="wait for a key press before the script exits")
    p.add_argument("--ref_file", type=str, default='data/HotFree0_0_9.csv', help="reference file for F(m)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main entry‑point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Device & dtype -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # Training -----------------------------------------------------------------------
    model, history = train_conditional(
        ρ_max=args.rho_max,
        N_grid=args.N_grid,
        cut=args.cut,
        m_min=args.m_min,
        m_max=args.m_max,
        batch_m=args.batch,
        lr=args.lr,
        n_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        device=device,
        live_plot=args.live,
    )

    # Optional final plot ------------------------------------------------------------
    if args.plot and history:
        
        last = history[-1]
        Fs = np.loadtxt(args.ref_file, delimiter=",")
        m_eval_min = float(last["m_eval"][0])
        m_eval_max = float(last["m_eval"][-1])
        sel = (Fs[:, 0] >= m_eval_min) & (Fs[:, 0] <= m_eval_max)
        Fs = Fs[sel, :]
        
        plt.ioff()  # turn off interactive mode for final plot
        plt.figure(figsize=(4.5, 3))
        plt.plot(last["m_eval"].numpy(), last["F_eval"].numpy(), lw=2, label="ANN")
        plt.plot(Fs[:, 0], Fs[:, 1], lw=2, label="ODE", linestyle=':', color='red')
        plt.xlabel(r"$m$")
        plt.ylabel(r"$F(m)$")
        plt.title("Free energy vs mass (final)")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/F_vs_m_{args.epochs}_conditional.png", dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)  # let the GUI render
        print("Close the plot window or press Ctrl+C to exit.")

    # Optional pause ---------------------------------------------------------------
    if args.wait:
        try:
            input("Training complete. Press <Enter> to exit…")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()