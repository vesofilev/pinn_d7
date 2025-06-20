import torch

__all__ = ["build_grid"]


def build_grid(ρ_min=0.0, ρ_max=20.0, N=2000, cut=5.0, dense_frac=0.5, device="cpu", dtype=torch.double):
    """
    Non-uniform grid combining a dense IR region and a coarse UV tail.
    Returns a tensor of shape (N,1).
    """
    N_dense = int(N * dense_frac)
    N_coarse = N - N_dense

    ρ_dense = torch.linspace(ρ_min, cut, N_dense, device=device, dtype=dtype)
    ρ_coarse = torch.linspace(cut, ρ_max, N_coarse + 1, device=device, dtype=dtype)[1:]  # drop duplicate cut
    return torch.cat([ρ_dense, ρ_coarse]).unsqueeze(1)