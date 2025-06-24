#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI launcher for the inverse-problem training in `pinn_d7.train_inverse`.

Example
-------
$ python run_inverse.py --epochs 40000 --lr_L 1e-4 --lr_V 5e-4 \
                        --m_min 0.1 --m_max 0.6 --plot
"""
import os as _os
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"     # macOS / OpenMP fix

import argparse
import sys
import pathlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ------------------------------------------------------------------
# repo import path
# ------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pinn_d7.train_inverse import train_inverse, get_diagnostics_curve


# ------------------------------------------------------------------
# argument parser
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        prog="run_inverse.py",
        description="Joint (l , V) training for the probe–D7 inverse problem."
    )
    p.add_argument("--epochs", type=int,   default=20_000, help="number of training epochs")
    p.add_argument("--lr_L",   type=float, default=1e-4,   help="learning rate for L-network")
    p.add_argument("--lr_V",   type=float, default=5e-4,   help="learning rate for V-network")
    p.add_argument("--batch",  type=int,   default=1,      help="masses per SGD step")
    p.add_argument("--rho_max", type=float, default=20.0,  help="UV cutoff ρ_max")
    p.add_argument("--N_grid",  type=int,   default=2000,  help="total collocation points")
    p.add_argument("--cut",     type=float, default=5.0,   help="boundary between dense & coarse zones")
    p.add_argument("--m_min",   type=float, default=0.0,   help="minimum mass")
    p.add_argument("--m_max",   type=float, default=0.8,   help="maximum mass")
    p.add_argument("--potential",     type=str,   default="hot", help="'hot' or 'magnetic' (chooses CSV)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_inverse",
                   help="directory in which checkpoints are saved")
    p.add_argument("--resume",   type=str, default=None,   help="resume from checkpoint file")
    p.add_argument("--plot",     action="store_true",      help="plot F(m) at the end")
    p.add_argument("--live",     action="store_true",      help="live-update plot during training")
    p.add_argument("--wait",     action="store_true",      help="wait for <Enter> before exit")
    return p.parse_args()


# ------------------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)

    # ref_file = _ref_files.get(args.potential.lower(), args.potential)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    l_model, v_model, history = train_inverse(
        ρ_max       = args.rho_max,
        N_grid      = args.N_grid,
        cut         = args.cut,
        m_min       = args.m_min,
        m_max       = args.m_max,
        batch_m     = args.batch,
        lr_L        = args.lr_L,
        lr_V        = args.lr_V,
        n_epochs    = args.epochs,
        checkpoint_dir = args.checkpoint_dir,
        resume_from    = args.resume,
        device         = device,
        live_plot      = args.live,
        potential= args.potential.lower()
    )

    # ------------------------------------------------------------------
    # final diagnostic plot  (always executed)
    # ------------------------------------------------------------------
    import math
    from pinn_d7.train_conditional import build_rho_grid
    from pinn_d7.losses            import dbi_energy_V

    # ----- evaluate F(m) on a dense mass grid -------------------------
    m_eval = torch.linspace(args.m_min, args.m_max, 101,
                            device=device).unsqueeze(1)
    ρ_grid = build_rho_grid(args.rho_max, args.N_grid, args.cut, device)
    I_eval = dbi_energy_V(l_model, v_model,
                        ρ_grid, m_eval, reduce="none").detach().cpu()

    # ----- radial grid and reconstructed V(r) -------------------------
    r_max  = math.sqrt(args.rho_max**2 + args.m_max**2)
    r_grid = torch.linspace(math.sqrt(0.5), r_max, 201,
                            device=device).unsqueeze(1)
    V_rec  = v_model(r_grid).detach().cpu().squeeze()

    # ----- reference free-energy curve --------------------------------
    Fs = get_diagnostics_curve(args.potential.lower(), m_min=args.m_min, m_max=args.m_max)

    # ----- figure -----------------------------------------------------
    plt.ioff()                                      # make sure we are non-interactive
    fig, (ax_F, ax_V) = plt.subplots(1, 2, figsize=(9, 4))

    # -- left panel : F(m) ---------------------------------------------
    ax_F.plot(m_eval.cpu(), I_eval, lw=2, label="ANN")
    ax_F.plot(Fs[:, 0], Fs[:, 1], ":", lw=2, label="target", color="red")
    ax_F.set_xlabel(r"$m$")
    ax_F.set_ylabel(r"$F(m)$")
    ax_F.set_title("Free energy vs mass")
    ax_F.grid(alpha=0.4)
    ax_F.legend()

    # -- right panel : V(r) --------------------------------------------
    ax_V.plot(r_grid.cpu(), V_rec, lw=2, label="ANN")
    # optional analytic reference
    ax_V.plot(r_grid.cpu(),
            1.0 - 1.0 / (r_grid.cpu() ** 8) / 16.0,
            ":", color="red", lw=1.2,
            label=r"$1-\dfrac{1}{16 r^{8}}$")
    ax_V.set_xlabel(r"radial coordinate $r$")
    ax_V.set_ylabel(r"potential $V(r)$")
    ax_V.set_title("Recovered potential")
    ax_V.grid(alpha=0.4)
    ax_V.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"plots/F_and_V_{args.epochs}_inverse.png",
                dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)           # give the GUI a chance to draw

    # ------------------------------------------------------------------
    if args.wait:
        try:
            input("Training complete. Press <Enter> to exit…")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()