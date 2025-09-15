#!/usr/bin/env python3
"""
Train one network per supplied mass and plot all profiles together.

Example
-------
$ python run_mass_scan.py 0.05 0.1 0.2 0.3 --epochs 40000 --potential magnetic
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use a more stable backend for macOS - fallback hierarchy
def setup_matplotlib_backend():
    """Setup matplotlib backend with fallback options for macOS."""
    backends_to_try = ["Qt5Agg", "MacOSX", "TkAgg", "Agg"]
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            print(f"Using matplotlib backend: {backend}")
            break
        except (ImportError, RuntimeError):
            continue
    else:
        print("Warning: Could not set any preferred backend, using default")

setup_matplotlib_backend()
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pinn_d7.trainer import train_scan, get_default_device

dafaFree = {
    'magnetic': "data/MagFree.csv",
    'hot': "data/HotFree.csv",
    'hot-zoom': "data/HotFreeZoom.csv"
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("masses", nargs="+", type=float, help="list of bare masses m")
    ap.add_argument("--epochs", type=int, default=30_000, help="training epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    ap.add_argument("--potential", type=str, default="magnetic", help="either magnetic or hot")
    ap.add_argument("--sort", type=bool, default=False, help="either True or False")
    ap.add_argument("--skip", type=int, default=0, help="how many to skip")
    ap.add_argument("--no-display", action="store_true", help="save plots without displaying them")
    args = ap.parse_args()

    device = get_default_device()
    dtype  = torch.double if device.type == "mps" else torch.float64
    torch.set_default_dtype(dtype)
    
    results = train_scan(
        args.masses,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        potential=args.potential,
        sort=args.sort
    )

    plt.figure(figsize=(4.5, 3))
    Fs_net = []
    ms = []
    for m, ρ, L, F, dF in results[args.skip:]:
        plt.plot(ρ, L, lw=2, label=fr"$m={m:.3f}$")
        Fs_net.append(F)
        ms.append(m)
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$L(\rho)$")
    plt.title(r"Probe D7 profiles")
    if args.potential in ['hot', 'hot-zoom']:
        x_ball = np.linspace(0, 1/np.sqrt(2), 150)
        y_ball = np.array([np.sqrt(1/2 - x**2) for x in x_ball])
        plt.plot(x_ball, y_ball, '--', color='black')
        plt.plot(x_ball, -y_ball, '--', color='black')
        
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    # plt.gca().set_aspect(2)
    plt.savefig(f"plots/probe_d7_profiles_{args.epochs}_{args.potential}.png", dpi=300, bbox_inches="tight")
    
    if not args.no_display:
        plt.show(block=False)
        plt.pause(0.1) # let the GUI render
    else:
        plt.close()
        
    plt.figure(figsize=(4.5, 3))
    Fs = np.loadtxt(dafaFree[args.potential], delimiter=",")
    plt.xlabel(r"$m$")
    plt.ylabel(r"$F$")
    plt.title(r"Free energy vs mass")
    plt.plot(Fs[:,0], Fs[:,1], label="ODE")
    plt.plot(ms, Fs_net, 'x', label='ANN')
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    # plt.gca().set_aspect(2)
    plt.savefig(f"plots/F_vs_m_{args.epochs}_{args.potential}.png", dpi=300, bbox_inches="tight")
    
    if not args.no_display:
        plt.show(block=False)
        plt.pause(0.1)          # let the GUI render
        
        print("Finished all work. Plots saved and displayed.")
        print("Close the plot windows to exit, or press Ctrl+C...")

        try:
            # Use input() to wait for user action instead of infinite loop
            input("\nPress Enter to exit (or close plot windows)...")
        except KeyboardInterrupt:
            print("\nCtrl+C detected – closing figures and terminating.")
        finally:
            plt.close('all')
    else:
        plt.close()
        print("Finished all work. Plots saved to files (no display).")


if __name__ == "__main__":
    main()