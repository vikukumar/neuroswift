from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

Tensor = torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.fc1(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.fc2(x)
        return self.dropout(x)


class LinearSSM(nn.Module):
    """
    Linear-complexity state-space mixing block.

    This compact reference layer uses a recurrent selective scan to mix tokens
    in O(sequence_length) time instead of quadratic attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expansion: int = 2,
        conv_kernel: int = 4,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner_dim = d_model * expansion
        self.dt_min = dt_min

        self.norm = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.inner_dim * 2)
        self.conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=conv_kernel,
            groups=self.inner_dim,
            padding=conv_kernel - 1,
            bias=True,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.bc_proj = nn.Linear(self.inner_dim, 2 * d_state)
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        self.A_log = nn.Parameter(torch.empty(self.inner_dim, d_state))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            base = torch.arange(1, self.d_state + 1, dtype=torch.float32).log()
            self.A_log.copy_(base.unsqueeze(0).repeat(self.inner_dim, 1))

    def _scan(
        self,
        u: Tensor,
        delta: Tensor,
        b_t: Tensor,
        c_t: Tensor,
        state: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        batch, seq_len, inner_dim = u.shape
        if state is None:
            state = u.new_zeros(batch, inner_dim, self.d_state)

        A = -torch.exp(self.A_log).unsqueeze(0)
        running_state = state
        outputs = []

        for t in range(seq_len):
            dt_t = delta[:, t].unsqueeze(-1)
            decay = torch.exp(A * dt_t)
            drive = dt_t * u[:, t].unsqueeze(-1) * b_t[:, t].unsqueeze(1)
            running_state = decay * running_state + drive
            y_t = (running_state * c_t[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1), running_state

    def forward(
        self,
        x: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        residual = x
        seq_len = x.size(1)

        x = self.norm(x)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)

        u = rearrange(u, "b t d -> b d t")
        u = self.conv(u)[..., :seq_len]
        u = rearrange(u, "b d t -> b t d")
        u = F.silu(u)

        delta = F.softplus(self.dt_proj(u)) + self.dt_min
        b_t, c_t = self.bc_proj(u).chunk(2, dim=-1)
        b_t = torch.tanh(b_t)
        c_t = torch.tanh(c_t)

        ssm_out, next_state = self._scan(u, delta, b_t, c_t, state)
        mixed = (ssm_out + u * self.D) * torch.sigmoid(gate)
        out = residual + self.dropout(self.out_proj(mixed))
        return out, next_state


class SparseMoE(nn.Module):
    """
    CPU-friendly sparse MoE.

    Routes each token to top-2 experts out of N experts, executing only those
    experts to reduce FLOPs during inference.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden: int = 256,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")

        aligned_hidden = int(math.ceil(expert_hidden / 64.0) * 64)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self.norm = RMSNorm(d_model)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.context_router = nn.Linear(d_model, num_experts, bias=False)
        self.context_gate = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, aligned_hidden, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        residual = x
        batch, seq_len, d_model = x.shape

        normalized = self.norm(x)
        prefix_steps = torch.arange(1, seq_len + 1, device=x.device, dtype=x.dtype).view(1, seq_len, 1)
        prefix_summary = normalized.cumsum(dim=1) / prefix_steps
        conditioned = normalized * torch.sigmoid(self.context_gate(prefix_summary))

        tokens = conditioned.reshape(batch * seq_len, d_model)
        router_logits = self.router(tokens) + self.context_router(prefix_summary).reshape(
            batch * seq_len,
            self.num_experts,
        )
        router_probs = torch.softmax(router_logits, dim=-1)

        top_weights, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_output = torch.zeros_like(tokens)
        token_ids = torch.arange(tokens.size(0), device=x.device)

        assigned_tokens = token_ids.repeat_interleave(self.top_k)
        assigned_experts = top_indices.reshape(-1)
        assigned_weights = top_weights.reshape(-1)

        order = torch.argsort(assigned_experts)
        assigned_tokens = assigned_tokens[order]
        assigned_experts = assigned_experts[order]
        assigned_weights = assigned_weights[order]

        capacity = max(
            self.top_k,
            int(self.capacity_factor * tokens.size(0) / self.num_experts),
        )

        unique_experts, counts = torch.unique_consecutive(assigned_experts, return_counts=True)
        cursor = 0
        for expert_id_tensor, count_tensor in zip(unique_experts, counts):
            expert_id = int(expert_id_tensor.item())
            count = int(count_tensor.item())
            segment = slice(cursor, cursor + count)

            expert_token_ids = assigned_tokens[segment]
            expert_token_weights = assigned_weights[segment]

            if expert_token_ids.numel() > capacity:
                keep = torch.topk(expert_token_weights, k=capacity, sorted=False).indices
                expert_token_ids = expert_token_ids[keep]
                expert_token_weights = expert_token_weights[keep]

            expert_input = tokens.index_select(0, expert_token_ids).contiguous()
            expert_output = self.experts[expert_id](expert_input)
            flat_output.index_add_(
                0,
                expert_token_ids,
                expert_output * expert_token_weights.unsqueeze(-1),
            )
            cursor += count

        importance = router_probs.sum(dim=0)
        load = F.one_hot(top_indices, num_classes=self.num_experts).sum(dim=1).float().sum(dim=0)
        total_tokens = float(tokens.size(0))
        aux_loss = self.num_experts * torch.sum(
            (importance / total_tokens) * (load / total_tokens)
        )

        out = residual + self.dropout(flat_output.view(batch, seq_len, d_model))
        return out, aux_loss


__all__ = ["RMSNorm", "LinearSSM", "SparseMoE"]
