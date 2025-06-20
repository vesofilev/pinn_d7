import torch
import torch.nn as nn

__all__ = ["dbi_energy_magnetic", "dbi_energy_hot",
           "dbi_energy_hot_m", "dbi_energy_magnetic_m"]


def dbi_energy_magnetic(model, ρ_grid: torch.Tensor) -> torch.Tensor:
    r"""
    Regularised Euclidean DBI free-energy functional (Eq. (2.6)).

    Parameters
    ----------
    model : nn.Module
        Neural network giving L(ρ).
    ρ_grid : (N,1) tensor
        Collocation points.
    """
    ρ_grid = ρ_grid.requires_grad_(True)
    L = model(ρ_grid)

    L_prime, = torch.autograd.grad(
        L, ρ_grid,
        grad_outputs=torch.ones_like(L),
        create_graph=True
    )

    integrand = (
        ρ_grid ** 3 *
        torch.sqrt(1.0 + 1.0 / (ρ_grid ** 2 + L ** 2) ** 2) *
        torch.sqrt(1.0 + L_prime ** 2) -
        ρ_grid * torch.sqrt(ρ_grid ** 4 + 1.0)
    )

    f = integrand.squeeze(-1)
    dρ = ρ_grid[1:, 0] - ρ_grid[:-1, 0]
    return 0.5 * torch.sum((f[1:] + f[:-1]) * dρ)


def dbi_energy_hot(model, ρ_grid: torch.Tensor) -> torch.Tensor:
    """
    Regularised Euclidean DBI free-energy functional (Eq. (2.15)).
    r₀ = √2
    Parameters
    ----------
    model : nn.Module
        Neural network giving L(ρ).
    ρ_grid : (N,1) tensor
        Collocation points.
    """
    ρ_grid = ρ_grid.requires_grad_(True)

    l = model(ρ_grid)                     # l(ρ)
    l_prime = torch.autograd.grad(
                    l, ρ_grid,
                    grad_outputs=torch.ones_like(l),
                    create_graph=True)[0]        # dl/dρ
    
    # if (1. - (1. / (ρ_grid[0] ** 2 + l[0] ** 2) ** 4)/16) > 0:
    #     clamped_term = (1. - (1. / (ρ_grid[0] ** 2 + l[0] ** 2) ** 4)/16) * 1e6
    # else:
    clamped_term = torch.clamp(1. - (1. / (ρ_grid ** 2 + l ** 2) ** 4)/16, min=0.0)
    integrand = ρ_grid ** 3 * clamped_term * torch.sqrt(1. +  l_prime ** 2) - ρ_grid ** 3
         # (N,1)

    # trapezoidal rule : ∑ ½(f_i+1+f_i) Δu
    f = integrand.squeeze(-1)                    # (N,)
    dρ = ρ_grid[1:, 0] - ρ_grid[:-1, 0]
    return torch.sum(0.5 * (f[1:] + f[:-1]) * dρ)


def dbi_energy_hot_m(model: nn.Module,
               ρ_grid: torch.Tensor,
               m_batch: torch.Tensor,
               reduce: str = "mean") -> torch.Tensor:
    """
    ρ_grid : (N,1)  collocation grid (on device)
    m_batch: (B,1)  masses for the current mini-batch
    reduce : "mean" → scalar      (for back-prop)
             "none" → (B,) tensor (diagnostics)
    """
    assert reduce in ("mean", "none")
    B, N = m_batch.size(0), ρ_grid.size(0)

    # Broadcast the grid to all masses
    ρ = ρ_grid.repeat(B, 1, 1).reshape(B * N, 1).clone().detach()
    ρ.requires_grad_(True)
    m = m_batch.repeat_interleave(N, dim=0)

    # Forward + first derivative
    l  = model(ρ, m)
    lʹ = torch.autograd.grad(l, ρ,
                             grad_outputs=torch.ones_like(l),
                             create_graph=True)[0]

    # DBI integrand  f(ρ , m)
    clamped_term = torch.clamp(1. - (1. / (ρ ** 2 + l ** 2) ** 4)/16, min=0.0)    
    integrand = ρ**3 * (clamped_term * torch.sqrt(1.0 + lʹ**2) - 1)

    # Trapezoidal integration for every mass
    f  = integrand.view(B, N)                           # (B,N)
    dρ = (ρ_grid[1:, 0] - ρ_grid[:-1, 0]).view(1, N-1) # (1,N-1)
    I  = torch.sum(0.5 * (f[:, 1:] + f[:, :-1]) * dρ, dim=1)  # (B,)

    return I.mean() if reduce == "mean" else I     


def dbi_energy_magnetic_m(model: nn.Module,
               ρ_grid: torch.Tensor,
               m_batch: torch.Tensor,
               reduce: str = "mean") -> torch.Tensor:
    """
    ρ_grid : (N,1)  collocation grid (on device)
    m_batch: (B,1)  masses for the current mini-batch
    reduce : "mean" → scalar      (for back-prop)
             "none" → (B,) tensor (diagnostics)
    """
    assert reduce in ("mean", "none")
    B, N = m_batch.size(0), ρ_grid.size(0)

    # Broadcast the grid to all masses
    ρ = ρ_grid.repeat(B, 1, 1).reshape(B * N, 1).clone().detach()
    ρ.requires_grad_(True)
    m = m_batch.repeat_interleave(N, dim=0)

    # Forward + first derivative
    l  = model(ρ, m)
    lʹ = torch.autograd.grad(l, ρ,
                             grad_outputs=torch.ones_like(l),
                             create_graph=True)[0]

    # DBI integrand  f(ρ , m)
    integrand = ρ**3 * torch.sqrt(1.0 + 1.0/(ρ**2 + l**2)**2) * torch.sqrt(1.0 + lʹ**2) \
              - ρ * torch.sqrt(ρ**4 + 1.0)

    # Trapezoidal integration for every mass
    f  = integrand.view(B, N)                           # (B,N)
    dρ = (ρ_grid[1:, 0] - ρ_grid[:-1, 0]).view(1, N-1) # (1,N-1)
    I  = torch.sum(0.5 * (f[:, 1:] + f[:, :-1]) * dρ, dim=1)  # (B,)

    return I.mean() if reduce == "mean" else I          # () or (B,)


dbi_energy_map ={
    "hot": dbi_energy_hot,
    "magnetic": dbi_energy_magnetic
}

dbi_energy_map_conditional ={
    "hot": dbi_energy_hot_m,
    "magnetic": dbi_energy_magnetic_m
}