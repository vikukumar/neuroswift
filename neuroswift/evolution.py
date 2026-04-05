"""
neuroswift.evolution
====================
The Evolution Engine for Progressive Model Growth.

This module provides the 'WeightExpansionEngine' which takes an existing 
NeuroSwiftLM and 'grows' its weights into a larger architecture (d_model, n_layers, etc.) 
while preserving the knowledge it has already acquired.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .model import NeuroSwiftLM, NeuroSwiftConfig

logger = logging.getLogger(__name__)

class WeightEvolutionEngine:
    """
    Expert logic for mapping weights from a small model to a large model.
    Supports d_model expansion (padding) and n_layers expansion (stacking).
    """

    @staticmethod
    def evolve(old_model: NeuroSwiftLM, new_config: NeuroSwiftConfig) -> NeuroSwiftLM:
        """
        Takes an existing model and configures a new 'Grown' model with 
        inherited weights.
        """
        from .model import NeuroSwiftLM # avoid circular import
        
        logger.info(f"Evolving model: L{old_model.config.n_layers}D{old_model.config.d_model} -> L{new_config.n_layers}D{new_config.d_model}")
        
        new_model = NeuroSwiftLM(new_config).to(old_model.token_embedding.weight.device)
        
        # 1. Map Embeddings (vocab_size must match, or be larger)
        WeightEvolutionEngine._copy_embedding(old_model.token_embedding, new_model.token_embedding)
        
        # 2. Map LM Head
        WeightEvolutionEngine._copy_linear(old_model.lm_head, new_model.lm_head)
        
        # 3. Map Blocks
        # If we have more layers, we repeat layers or use identity stacking
        old_blocks = old_model.blocks
        new_blocks = new_model.blocks
        
        for i in range(len(new_blocks)):
            # Strategy: Cycle through old layers if we have more new layers
            old_idx = i % len(old_blocks)
            WeightEvolutionEngine._copy_block(old_blocks[old_idx], new_blocks[i])
            
        # 4. Map Final Norm
        WeightEvolutionEngine._copy_norm(old_model.final_norm, new_model.final_norm)
        
        return new_model

    @staticmethod
    def _copy_embedding(old_m: nn.Embedding, new_m: nn.Embedding):
        with torch.no_grad():
            v, d = old_m.weight.shape
            nv, nd = new_m.weight.shape
            # Copy common part
            new_m.weight[:v, :d].copy_(old_m.weight)

    @staticmethod
    def _copy_linear(old_m: nn.Linear, new_m: nn.Linear):
        if not hasattr(old_m, "weight"): return
        with torch.no_grad():
            oh, ow = old_m.weight.shape
            nh, nw = new_m.weight.shape
            # Copy common part
            new_m.weight[:oh, :ow].copy_(old_m.weight)
            if old_m.bias is not None and new_m.bias is not None:
                new_m.bias[:oh].copy_(old_m.bias)

    @staticmethod
    def _copy_norm(old_m: nn.Module, new_m: nn.Module):
        if hasattr(old_m, "weight"):
            with torch.no_grad():
                d = old_m.weight.shape[0]
                new_m.weight[:d].copy_(old_m.weight)

    @staticmethod
    def _copy_block(old_b: nn.Module, new_b: nn.Module):
        """Recursively copy sub-module weights with dimension handling."""
        with torch.no_grad():
            # LinearSSM
            WeightEvolutionEngine._copy_linear_ssm(old_b.ssm, new_b.ssm)
            # SparseMoE
            WeightEvolutionEngine._copy_moe(old_b.moe, new_b.moe)
            # Plasticity
            WeightEvolutionEngine._copy_plasticity(old_b.plasticity, new_b.plasticity)

    @staticmethod
    def _copy_linear_ssm(old_m: nn.Module, new_m: nn.Module):
        # LinearSSM usually has x_proj, dt_proj, out_proj
        for name in ["x_proj", "dt_proj", "out_proj"]:
            if hasattr(old_m, name) and hasattr(new_m, name):
                WeightEvolutionEngine._copy_linear(getattr(old_m, name), getattr(new_m, name))
        if hasattr(old_m, "conv1d") and hasattr(new_m, "conv1d"):
            oc = old_m.conv1d.weight.shape
            nc = new_m.conv1d.weight.shape
            new_m.conv1d.weight[:oc[0], :oc[1], :oc[2]].copy_(old_m.conv1d.weight)

    @staticmethod
    def _copy_moe(old_m: nn.Module, new_m: nn.Module):
        # Router
        WeightEvolutionEngine._copy_linear(old_m.gate, new_m.gate)
        # Experts (if they match in count)
        n_old = len(old_m.experts)
        n_new = len(new_m.experts)
        for i in range(n_new):
            old_idx = i % n_old
            # Each expert is a Sequential [Linear, Act, Linear, Dropout]
            # Expert format in SparseMoE: self.experts = nn.ModuleList([nn.Sequential(...)])
            old_exp = old_m.experts[old_idx]
            new_exp = new_m.experts[i]
            # Map the linear layers inside the Sequential
            for sub_old, sub_new in zip(old_exp, new_exp):
                if isinstance(sub_old, nn.Linear):
                    WeightEvolutionEngine._copy_linear(sub_old, sub_new)

    @staticmethod
    def _copy_plasticity(old_m: nn.Module, new_m: nn.Module):
        # q_proj, k_proj, v_proj
        for name in ["query", "key", "value"]:
            if hasattr(old_m, name) and hasattr(new_m, name):
                WeightEvolutionEngine._copy_linear(getattr(old_m, name), getattr(new_m, name))
