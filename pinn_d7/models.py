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
    def __init__(self, ρ_max: float, hidden: int = 16, depth: int = 2):
        super().__init__()
        self.ρ_max = ρ_max
        
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        
        self.core = nn.Sequential(*layers).double()
        
        # Physics-informed initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better PINN convergence."""
        for module in self.core:
            if isinstance(module, nn.Linear):
                # Use smaller initialization for PINNs (scaled Xavier)
                fan_in = module.weight.size(1)
                fan_out = module.weight.size(0)
                std = math.sqrt(2.0 / (fan_in + fan_out)) * 0.5  # Scale down by 0.5
                nn.init.normal_(module.weight, 0.0, std)
                
                # Small bias initialization
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Make final layer output small initially (physics prior: small deviations)
        final_layer = self.core[-1]
        nn.init.normal_(final_layer.weight, 0.0, 0.01)  # Very small initial output
        if final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, 0.0)
    
    def reinitialize_weights(self, strategy='physics_informed'):
        """Reinitialize weights with different strategies for testing."""
        if strategy == 'physics_informed':
            self._initialize_weights()
        elif strategy == 'small_xavier':
            for module in self.core:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        elif strategy == 'kaiming_small':
            for module in self.core:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='tanh')
                    module.weight.data *= 0.3  # Scale down
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        elif strategy == 'uniform_small':
            for module in self.core:
                if isinstance(module, nn.Linear):
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, -0.01, 0.01)

    def forward(self, ρ: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        ρ : (B·N,1),  m : (B·N,1)
        returns l(ρ , m) with correct UV boundary.
        """
        x = torch.cat([ρ, m], dim=1)
        g = self.core(x)
        return m + (self.ρ_max - ρ) * g
    
class VNetwork(nn.Module):
    def __init__(self, u_max: float, hidden: int = 10, depth: int = 2):
        super().__init__()
        self.u_max = u_max
        self.u_min = 1/math.sqrt(2)
        
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        
        self.core = nn.Sequential(*layers).double()
        final_linear = self.core[-1]
        nn.init.constant_(final_linear.bias, 1.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        g = self.core(u)
        g_max = self.core(torch.tensor([[self.u_max]], device=u.device, dtype=u.dtype))
        g_min = self.core(torch.tensor([[self.u_min]], device=u.device, dtype=u.dtype))
        return ((g - g_min) / (g_max - g_min + 1e-8))