from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .layers import RMSNorm

Tensor = torch.Tensor


class HebbianUpdater(nn.Module):
    """
    Online Hebbian fast-weight adapter.

    The module keeps a small per-layer fast-weight matrix that updates during
    inference, giving the model short-term adaptive memory without backprop.
    """

    def __init__(
        self,
        d_model: int,
        plastic_dim: int = 48,
        learning_rate: float = 0.02, # Tuned for Normalized Updates
        decay: float = 0.001,
        clamp_value: float = 1.0,    # Stronger clipping against explosion
    ) -> None:
        super().__init__()
        self.plastic_dim = plastic_dim
        self.learning_rate = learning_rate
        self.decay = decay
        self.clamp_value = clamp_value

        self.norm = RMSNorm(d_model)
        if plastic_dim > 0:
            self.pre_proj = nn.Linear(d_model, plastic_dim, bias=False)
            self.post_proj = nn.Linear(d_model, plastic_dim, bias=False)
            self.out_proj = nn.Linear(plastic_dim, d_model, bias=False)
        else:
            self.pre_proj = self.post_proj = self.out_proj = None
            
        self.mod_proj = nn.Linear(d_model, 1)
        self.output_scale = nn.Parameter(torch.full((d_model,), 0.1))

    def update_state(
        self,
        fast_weights: torch.Tensor,
        pre_activations: torch.Tensor,
        post_activations: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        new_state = fast_weights.clone()
        keep = 1.0 - self.decay

        for t in range(pre_activations.size(1)):
            pre_t = pre_activations[:, t]
            post_t = post_activations[:, t]
            mod_t = modulation[:, t].mean(dim=-1, keepdim=True).unsqueeze(-1)

            hebb = torch.einsum("bi,bj->bij", post_t, pre_t)
            oja_norm = post_t.pow(2).mean(dim=-1, keepdim=True).unsqueeze(-1)
            delta = self.learning_rate * mod_t * (hebb - oja_norm * new_state)
            new_state = (keep * new_state + delta).clamp(-self.clamp_value, self.clamp_value)

        return new_state

    def forward(
        self,
        x: Tensor,
        fast_weights: Optional[Tensor] = None,
        update: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        batch = x.size(0)
        residual = x
        x_norm = self.norm(x)
        if self.plastic_dim <= 0 or self.pre_proj is None:
            return residual, x.new_zeros(batch, 1, 1) # Dummy state for 0-dim

        pre = torch.tanh(self.pre_proj(x_norm))
        post = torch.tanh(self.post_proj(x_norm))
        modulation = torch.sigmoid(self.mod_proj(x_norm))

        if fast_weights is None:
            fast_weights = x.new_zeros(batch, self.plastic_dim, self.plastic_dim)

        adapted = torch.einsum("btd,bdh->bth", pre, fast_weights)
        
        # Stability Guard: L2 Normalize to prevent gradient explosion through recurrent loops
        adapted = torch.nn.functional.normalize(adapted, p=2.0, dim=-1)
        
        plastic_delta = torch.tanh(self.out_proj(adapted)) * self.output_scale

        next_state = fast_weights
        if update:
            with torch.no_grad():
                next_state = self.update_state(
                    fast_weights=fast_weights,
                    pre_activations=pre.detach(),
                    post_activations=post.detach(),
                    modulation=modulation.detach(),
                )

        return residual + plastic_delta, next_state


__all__ = ["HebbianUpdater"]
