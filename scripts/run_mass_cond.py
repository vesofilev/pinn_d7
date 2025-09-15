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
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pinn_d7.train_conditional import train_conditional, evaluate_condensate, build_rho_grid
from pinn_d7.losses import dbi_energy_map_conditional

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
    p.add_argument("--checkpoint_every", type=int, default=1000, help="epochs between checkpoints")
    p.add_argument("--hidden", type=int, default=16, help="hidden layer size")
    p.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible results")
    p.add_argument("--init_strategy", type=str, default='physics_informed', 
                   choices=['physics_informed', 'small_xavier', 'kaiming_small', 'uniform_small'],
                   help="weight initialization strategy")
    p.add_argument("--plot", action="store_true", help="plot F(m) at the end")
    p.add_argument("--live", action="store_true", help="live‑update F(m) during training")
    p.add_argument("--eval_mae", action="store_true", default=True, help="evaluate MAE against reference data")
    p.add_argument("--mae_linear", action="store_true", help="use linear scale for MAE plot instead of log scale")
    p.add_argument("--wait", action="store_true",
                   help="wait for a key press before the script exits")
    return p.parse_args()


def evaluate_mae_against_reference(model, Fs, ρ_max, N_grid, cut, device, potential):
    """Evaluate MAE between network predictions and reference data."""
    m_eval = torch.tensor(Fs[:, 0], dtype=torch.float64, device=device).unsqueeze(1)
    m_eval.requires_grad_(True)
    
    ρ_grid = build_rho_grid(ρ_max, N_grid, cut, device)
    dbi_energy = dbi_energy_map_conditional[potential]
    F_pred = dbi_energy(model, ρ_grid, m_eval, reduce="none").detach().cpu().numpy()
    
    mae = np.mean(np.abs(F_pred - Fs[:, 1]))
    print(f"MAE: {mae:.2e} (relative: {mae/np.mean(np.abs(Fs[:, 1]))*100:.2f}%)")
    return mae


def evaluate_condensate_mae_against_reference(model, ρ_max, N_grid, cut, m_min, m_max, device, potential):
    """Evaluate MAE between network condensate predictions and reference condensate data."""
    if potential != 'magnetic':
        return None  # Only evaluate condensate MAE for magnetic potential
    
    dbi_energy = dbi_energy_map_conditional[potential]
    
    m_eval = torch.linspace(m_min, m_max, 101, device=device).unsqueeze(1)
    m_eval.requires_grad_(True)
    
    ρ_grid = build_rho_grid(ρ_max, N_grid, cut, device)
    I_eval = dbi_energy(model, ρ_grid, m_eval, reduce="none")
    dIdm = torch.autograd.grad(I_eval.sum(), m_eval, create_graph=False)[0]
    
    # Load reference condensate data
    cs = np.loadtxt("data/MagCondensate.csv", delimiter=",")
    
    # Interpolate reference data to match evaluation points
    m_eval_np = m_eval.cpu().detach().numpy().flatten()
    dIdm_pred = dIdm.cpu().detach().numpy().flatten()
    
    # Filter reference data to overlap with evaluation range
    cs_filtered = cs[(cs[:, 0] >= m_min) & (cs[:, 0] <= m_max)]
    
    if len(cs_filtered) == 0:
        print("Warning: No reference condensate data in evaluation range")
        return None
    
    # Interpolate predicted values at reference points
    if len(m_eval_np) > 1:
        interp_func = interp1d(m_eval_np, dIdm_pred, kind='linear', bounds_error=False, fill_value='extrapolate')
        dIdm_interp = interp_func(cs_filtered[:, 0])
        
        mae_condensate = np.mean(np.abs(dIdm_interp - cs_filtered[:, 1]))
        print(f"Condensate MAE: {mae_condensate:.2e} (relative: {mae_condensate/np.mean(np.abs(cs_filtered[:, 1]))*100:.2f}%)")
        return mae_condensate
    
    return None


# ---------------------------------------------------------------------------
# main entry‑point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Device & dtype -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    
    ref_file = dafaFree[args.potential]

    # Start timing -------------------------------------------------------------------
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    # Training -----------------------------------------------------------------------
    model, history, mae_data, condensate_mae_data = train_conditional(
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
        checkpoint_every=args.checkpoint_every,
        device=device,
        live_plot=args.live,
        ref_file=ref_file,
        potential=args.potential,
        hidden=args.hidden,
        depth=args.depth,
        seed=args.seed,
        init_strategy=args.init_strategy,
        mae_eval_fn=evaluate_mae_against_reference,
        condensate_mae_eval_fn=evaluate_condensate_mae_against_reference
    )

    # End timing ---------------------------------------------------------------------
    cpu_end = time.process_time()
    wall_end = time.perf_counter()
    
    cpu_time = cpu_end - cpu_start
    wall_time = wall_end - wall_start

    # Load reference data for evaluation and plotting
    Fs = np.loadtxt(ref_file, delimiter=",")

    # Evaluate MAE against reference data -----------------------------------------------
    if args.eval_mae:
        evaluate_mae_against_reference(model, Fs, args.rho_max, args.N_grid, args.cut, device, args.potential)

    # Optional final plot ------------------------------------------------------------
    if args.plot and history:
        
        last = history[-1]
        
        m_eval_min = float(last["m_eval"][0])
        m_eval_max = float(last["m_eval"][-1])
        sel = (Fs[:, 0] >= m_eval_min) & (Fs[:, 0] <= m_eval_max)
        Fs_plot = Fs[sel, :]
        
        plt.ioff()  # turn off interactive mode for final plot
        plt.figure(figsize=(4.5, 3))
        plt.plot(last["m_eval"].numpy(), last["F_eval"].numpy(), lw=2, label="ANN")
        plt.plot(Fs_plot[:, 0], Fs_plot[:, 1], lw=2, label="ODE", linestyle=':', color='red')
        plt.xlabel(r"$m$")
        plt.ylabel(r"$F(m)$")
        plt.title(f"Free energy vs mass ({args.potential})")
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"plots/F_vs_m_{args.epochs}_{args.potential}_conditional_D{args.depth}N{args.hidden}R{args.N_grid}.png", dpi=300, bbox_inches="tight")
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
                depth=args.depth,
                hidden=args.hidden,
            )

            
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Total CPU time: {cpu_time:.3f} seconds ({cpu_time/60:.2f} minutes)")
    print(f"Total wall time: {wall_time:.3f} seconds ({wall_time/60:.2f} minutes)")
    print(f"CPU efficiency: {cpu_time/wall_time:.2f}x")
    print(f"{'='*60}")
    
    # Plot MAE vs epochs if MAE data is available ------------------------------------
    if mae_data is not None and mae_data["mae_values"]:
        plt.figure(figsize=(4.5, 3))
        
        if args.mae_linear:
            # Linear scale plot
            plt.plot(mae_data["epochs"], mae_data["mae_values"], 'o-', lw=2, markersize=4)
            plt.ylabel("MAE")
            scale_suffix = "_linear"
        else:
            # Logarithmic scale plot (default)
            plt.semilogy(mae_data["epochs"], mae_data["mae_values"], 'o-', lw=2, markersize=4)
            plt.ylabel("MAE (log scale)")
            plt.yscale('log')  # Explicitly set log scale for y-axis
            scale_suffix = "_log"
        
        plt.xlabel("Epoch")
        plt.title(f"MAE vs Epochs ({args.potential})")
        plt.grid(True, which="major", alpha=0.4)
        plt.grid(True, which="minor", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"plots/MAE_vs_epochs_{args.epochs}_{args.potential}_D{args.depth}N{args.hidden}R{args.N_grid}{scale_suffix}.png", dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)
        print(f"MAE plot ({'linear' if args.mae_linear else 'log'} scale) saved and displayed. Final MAE: {mae_data['mae_values'][-1]:.6e}")
    
    # Plot Condensate MAE vs epochs if data is available -------------------------
    if condensate_mae_data is not None and condensate_mae_data["mae_values"]:
        plt.figure(figsize=(4.5, 3))
        
        if args.mae_linear:
            # Linear scale plot
            plt.plot(condensate_mae_data["epochs"], condensate_mae_data["mae_values"], 'o-', lw=2, markersize=4, color='orange')
            plt.ylabel("Condensate MAE")
            scale_suffix = "_linear"
        else:
            # Logarithmic scale plot (default)
            plt.semilogy(condensate_mae_data["epochs"], condensate_mae_data["mae_values"], 'o-', lw=2, markersize=4, color='orange')
            plt.ylabel("Condensate MAE (log scale)")
            plt.yscale('log')  # Explicitly set log scale for y-axis
            scale_suffix = "_log"
        
        plt.xlabel("Epoch")
        plt.title(f"Condensate MAE vs Epochs ({args.potential})")
        plt.grid(True, which="major", alpha=0.4)
        plt.grid(True, which="minor", alpha=0.2)
        plt.tight_layout()
        plt.savefig(f"plots/Condensate_MAE_vs_epochs_{args.epochs}_{args.potential}_D{args.depth}N{args.hidden}R{args.N_grid}{scale_suffix}.png", dpi=300, bbox_inches="tight")
        plt.show(block=False)
        plt.pause(0.1)
        print(f"Condensate MAE plot ({'linear' if args.mae_linear else 'log'} scale) saved and displayed. Final Condensate MAE: {condensate_mae_data['mae_values'][-1]:.6e}")
    
    # Optional pause ---------------------------------------------------------------
    if args.wait:
        try:
            input("Training complete. Press <Enter> to exit…")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()