import torch
import torch.nn as nn
import math

__all__ = ["LNetwork"]


class LNetwork(nn.Module):
    r"""
    Neural ansatz for the D7-brane profile L(ρ).

    Parameters
    ----------
    ρ_max : float
        UV cutoff imposed on ρ.
    m : float
        Bare quark mass (sets the UV boundary condition L(ρ_max)=m).
    hidden : int
        Width of each hidden layer.
    depth : int
        Number of hidden layers (≥1).
    """
    def __init__(self, ρ_max: float, m: float, hidden: int = 10, depth: int = 2):
        super().__init__()
        self.m = float(m)
        self.ρ_max = float(ρ_max)

        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]

        self.core = nn.Sequential(*layers)

    # ------------------------------------------------------------
    def forward(self, ρ: torch.Tensor) -> torch.Tensor:
        """Return L(ρ) with the UV boundary condition built in."""
        return self.m + (self.ρ_max - ρ) * self.core(ρ)
    

class LNetworkM(nn.Module):
    """
    Architecture:  l(ρ , m) = m + (ρ_max - ρ) · g([ρ , m]; W)
    – UV BC  l(ρ_max , m) = m is hard-wired.
    – g(·) is a small fully-connected network that vanishes at ρ=ρ_max.
    """
    def __init__(self, ρ_max: float):
        super().__init__()
        self.ρ_max = ρ_max
        self.core = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 1)
        ).double()

    def forward(self, ρ: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        ρ : (B·N,1),  m : (B·N,1)
        returns l(ρ , m) with correct UV boundary.
        """
        x = torch.cat([ρ, m], dim=1)
        g = self.core(x)
        return m + (self.ρ_max - ρ) * g
    
class VNetwork(nn.Module):
    def __init__(self, u_max: float):
        super().__init__()
        self.u_max = u_max
        self.u_min = 1/math.sqrt(2)
        self.core = nn.Sequential(
            nn.Linear(1, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1),
        ).double()
        final_linear = self.core[-1]
        nn.init.constant_(final_linear.bias, 1.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = self.core(u)
        g_max = self.core(torch.tensor([[self.u_max]], device=u.device, dtype=u.dtype))
        g_min = self.core(torch.tensor([[self.u_min]], device=u.device, dtype=u.dtype))
        return ((g - g_min) / (g_max - g_min + 1e-8))