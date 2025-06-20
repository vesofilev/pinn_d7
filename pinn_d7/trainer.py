import time
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .models import LNetwork
from .losses import dbi_energy_magnetic, dbi_energy_hot
from .grid import build_grid

from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR

__all__ = ["get_default_device", "train_for_mass", "train_scan"]

dbi_energy = {
    'magnetic': dbi_energy_magnetic,
    'hot': dbi_energy_hot,
    'hot-zoom': dbi_energy_hot
}


def get_default_device(allow_mps=False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and allow_mps:  # Apple-Silicon Metal backend
        return torch.device("mps")
    return torch.device("cpu")


def train_for_mass(
    m: float,
    ρ_max: float = 20.0,
    lr: float = 1e-4,
    n_epochs: int = 30_000,
    print_every: int = 1000,
    grid_kwargs=None,
    device: str = "cpu",
    model=None,
    potential='magnetic',
    sort=False
) -> Tuple[torch.Tensor, torch.Tensor, float, float, float, LNetwork]:
    """
    Train a single network for a given mass and return (ρ, L, best_loss, mean_loss, std_loss, model).
    """
    grid_kwargs = grid_kwargs or {}
    ρ_grid = build_grid(ρ_max=ρ_max, device=device, **grid_kwargs)

    model = model or LNetwork(ρ_max, m).to(device)
    # model = LNetwork(ρ_max, m).to(device)
    model.m = m
    # optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        # betas=(0.9, 0.9999),
        # weight_decay=1e-5
    )
    scheduler = StepLR(optim, step_size=n_epochs//2, gamma=0.5)
    # record losses each epoch
    losses = []

    for epoch in range(1, n_epochs + 1):
        optim.zero_grad()
        loss = dbi_energy[potential](model, ρ_grid)
        # record loss per epoch
        losses.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step()

        if epoch % print_every == 0 or epoch == 1 or epoch == n_epochs:
            print(f"[m={m:5.4f}]  {epoch:5d}/{n_epochs}  loss={loss.item():.6e}")

    with torch.no_grad():
        L = model(ρ_grid).cpu()

    # compute metrics for free energy estimate
    window = min(len(losses), 500)
    last_losses = losses[-window:]
    median_loss = float(np.median(last_losses))
    std_loss = float(np.std(last_losses))
    best_loss = float(np.min(losses))
    print(f"[m={m:5.4f}] final loss={best_loss} ± {std_loss/np.sqrt(window)}")
    return ρ_grid.cpu().squeeze(), L.squeeze(), best_loss, std_loss, model


def train_scan(masses: List[float], **kwargs) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Train one network per mass and return a dict: m → (ρ, L(ρ)).
    """
    results = []
    model = None
    if kwargs.get('sort', False):
        masses = sorted(masses, reverse=True)
    lr = kwargs['lr']
    for m in masses:
        st_time = time.time()
        print(f"\n=== Training for m = {m:.4f} ===")
        kwargs['model'] = model
        if model:
            kwargs['lr'] = lr/2
        ρ, L, F, dF, model = train_for_mass(m, **kwargs)
        results.append((m, ρ.detach().numpy(), L.detach().numpy(), F, dF))
        print(f"Training took {time.time() - st_time:.3f} seconds")
    return results