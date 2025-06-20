#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI launcher for conditional probe–D7 training (hot DBI potential).
Example
-------
$ python run_mass_cond.py --epochs 40000 --lr 5e-6 --batch 2 --m_min 0.0 --m_max 0.8 --live
"""

import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OpenMP workaround for macOS
import argparse
import sys
import pathlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pinn_d7.train_conditional import train_conditional, evaluate_condensate

dafaFree = {
    'magnetic': "data/MagFree.csv",
    'hot': "data/HotFree0_0_9.csv",
}

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
    p.add_argument("--potential", type=str, default='hot', help="either 'hot' or 'magnetic' potential")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="directory in which checkpoints are saved",
    )
    p.add_argument("--resume", type=str, default=None, help="resume from checkpoint file")
    p.add_argument("--plot", action="store_true", help="plot F(m) at the end")
    p.add_argument("--live", action="store_true", help="live‑update F(m) during training")
    p.add_argument("--wait", action="store_true",
                   help="wait for a key press before the script exits")
    return p.parse_args()


# ---------------------------------------------------------------------------
# main entry‑point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Device & dtype -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    ref_file = dafaFree[args.potential]

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
        checkpoint_dir=f'{args.checkpoint_dir}_{args.potential}',
        resume_from=args.resume,
        device=device,
        live_plot=args.live,
        ref_file=ref_file,
        potential=args.potential
    )

    # Optional final plot ------------------------------------------------------------
    if args.plot and history:
        
        last = history[-1]
        
        Fs = np.loadtxt(ref_file, delimiter=",")
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
        plt.title(f"Free energy vs mass ({args.potential})")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/F_vs_m_{args.epochs}_{args.potential}_conditional.png", dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)  # let the GUI render
        print("Close the plot window or press Ctrl+C to exit.")
        
        if args.potential == "magnetic":
            evaluate_condensate(
                model=model,
                ρ_max=args.rho_max,
                N_grid=args.N_grid,
                cut=args.cut,
                m_min=args.m_min,
                m_max=args.m_max,
                device=device,
                potential=args.potential,
                n_epochs=args.epochs,
            )

    # Optional pause ---------------------------------------------------------------
    if args.wait:
        try:
            input("Training complete. Press <Enter> to exit…")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()