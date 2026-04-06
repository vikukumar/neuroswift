from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from torch import nn

from .layers import (
    DynamicDepthGate,
    LinearAttentionAnchor,
    LinearSSM,
    MLALinearAttention,
    MultiTokenHead,
    RMSNorm,
    SelectiveSSD,
    SparseMoE,
    auto_device,
)
from .plasticity import HebbianUpdater

Tensor = torch.Tensor


@dataclass
class NeuroSwiftConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    expansion: int = 2
    conv_kernel: int = 4
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 256
    plastic_dim: int = 48
    dropout: float = 0.1
    aux_loss_scale: float = 1e-2
    attn_interval: int = 0  # Reverted default to protection legacy models
    ternary_mode: bool = False  # Enable BitNet-style MatMul-free execution
    latent_dim: int = 64  # Compression for MLA anchors
    
    # NeuroSwift v1 Features
    version: str = "v1"
    label_smoothing: float = 0.0
    adaptive_dropout: bool = False
    stability_module: bool = False
    dynamic_top_k: bool = True
    expansion_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_type"] = "neuroswift"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NeuroSwiftConfig":
        normalized = dict(payload)
        normalized.pop("model_type", None)
        # Legacy failsafe: If fields are missing (V1 model), use backward-compatible defaults
        if "attn_interval" not in normalized:
            normalized["attn_interval"] = 0
        if "ternary_mode" not in normalized:
            normalized["ternary_mode"] = False
        if "latent_dim" not in normalized:
            normalized["latent_dim"] = 64
        if "version" not in normalized:
            normalized["version"] = "v0"
        if "label_smoothing" not in normalized:
            normalized["label_smoothing"] = 0.0
        if "adaptive_dropout" not in normalized:
            normalized["adaptive_dropout"] = False
        if "stability_module" not in normalized:
            normalized["stability_module"] = False
        if "dynamic_top_k" not in normalized:
            normalized["dynamic_top_k"] = False
        if "expansion_mode" not in normalized:
            normalized["expansion_mode"] = False
        return cls(**normalized)


class NeuroSwiftBlock(nn.Module):
    def __init__(self, config: NeuroSwiftConfig) -> None:
        super().__init__()
        # Upgrade to SelectiveSSD (Mamba-2 style) for best O(N) sequence logic
        self.ssm = SelectiveSSD(
            d_model=config.d_model,
            d_state=config.d_state,
            expansion=config.expansion,
            conv_kernel=config.conv_kernel,
            dropout=config.dropout,
        )
        self.moe = SparseMoE(
            d_model=config.d_model,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_hidden=config.expert_hidden,
            dropout=config.dropout,
            dynamic_top_k=config.dynamic_top_k,
        )
        self.plasticity = HebbianUpdater(
            d_model=config.d_model,
            plastic_dim=config.plastic_dim,
        )
        # Upgrade to MLALinearAttention (DeepSeek style) for reasoning IQ
        self.attn = (
            MLALinearAttention(
                d_model=config.d_model, 
                latent_dim=config.latent_dim,
                dropout=config.dropout,
            )
            if config.attn_interval > 0
            else None
        )
        self.thinking_gate = DynamicDepthGate(config.d_model)

        self.expansion_mode = getattr(config, "expansion_mode", False)
        if self.expansion_mode:
            self.post_norm = RMSNorm(config.d_model)
            self.mixing_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor] = None,
        ssm_external_state: Optional[Tensor] = None,
        plastic_state: Optional[Tensor] = None,
        update_plasticity: bool = True,
        use_attn: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # 0. Thinking Gate (DDS) - Predict intensity for this logic unit
        thinking_intensity = self.thinking_gate(x) # [B, T, 1]

        if use_attn and self.attn is not None:
            x_attn, next_ssm_state = self.attn(x, state=ssm_state)
            x = x + thinking_intensity * (x_attn - x)
        else:
            x_ssm, next_ssm_state = self.ssm(
                x, 
                state=ssm_state, 
                external_state=ssm_external_state
            )
            # SSM always updates for state continuity, but intensity controls depth
            x = x + thinking_intensity * (x_ssm - x)

        x_moe, aux_loss = self.moe(x)
        x = x + thinking_intensity * (x_moe - x)

        x_plastic, next_plastic_state = self.plasticity(
            x,
            fast_weights=plastic_state,
            update=update_plasticity,
        )
        x = x + thinking_intensity * (x_plastic - x)

        if self.expansion_mode:
            # Underfitting guard: Extra feature mixing layer (Optional)
            xm = self.post_norm(x)
            x = x + 0.1 * self.mixing_proj(xm)

        return x, next_ssm_state, next_plastic_state, aux_loss


class NeuroSwiftLM(nn.Module):
    def __init__(self, config: NeuroSwiftConfig, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_norm = RMSNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([NeuroSwiftBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.mtp_head = MultiTokenHead(config.d_model, config.vocab_size, n_tokens=1)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # Omega-Mode Stability: Kaiming Normal for 1M models
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_in")
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        ssm_states: Optional[list[Optional[Tensor]]] = None,
        ssm_external_states: Optional[list[Optional[Tensor]]] = None,
        plastic_states: Optional[list[Optional[Tensor]]] = None,
        update_plasticity: Optional[bool] = None,
    ) -> dict[str, Any]:
        if update_plasticity is None:
            update_plasticity = not self.training

        device = input_ids.device
        # Note: Thread management is handled once at process initialization for Absolute Speed.

        x = self.token_embedding(input_ids)
        x = self.emb_norm(x)
        x = self.dropout(x)

        if ssm_states is None:
            ssm_states = [None] * len(self.blocks)
        if ssm_external_states is None:
            ssm_external_states = [None] * len(self.blocks)
        if plastic_states is None:
            plastic_states = [None] * len(self.blocks)

        next_ssm_states: list[Tensor] = []
        next_plastic_states: list[Tensor] = []
        aux_losses = []

        # AMP context: use autocast on CUDA, no-op on CPU
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda" and torch.cuda.is_available()
            else torch.autocast(device_type="cpu", enabled=False)
        )

        with amp_ctx:
            for idx, block in enumerate(self.blocks):
                if self.use_checkpoint and self.training:
                    from torch.utils.checkpoint import checkpoint as ckpt
                    def _block_fn(x_in, ssm_s, plastic_s):
                        return block(x_in, ssm_state=ssm_s, plastic_state=plastic_s, update_plasticity=update_plasticity)
                    x, next_ssm, next_plastic, aux_loss = ckpt(
                        _block_fn, x, ssm_states[idx], plastic_states[idx],
                        use_reentrant=False,
                    )
                else:
                    use_attn = (self.config.attn_interval > 0 and (idx + 1) % self.config.attn_interval == 0)
                    x, next_ssm, next_plastic, aux_loss = block(
                        x,
                        ssm_state=ssm_states[idx],
                        ssm_external_state=ssm_external_states[idx],
                        plastic_state=plastic_states[idx],
                        update_plasticity=update_plasticity,
                        use_attn=use_attn,
                    )
                next_ssm_states.append(next_ssm.detach())
                next_plastic_states.append(next_plastic.detach())
                aux_losses.append(aux_loss)

            x = self.final_norm(x)
            logits = self.lm_head(x)
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_tensor(0.0)

        out: dict[str, Any] = {
            "logits": logits,
            "mtp_logits": None,
            "aux_loss": aux_loss,
            "ssm_states": next_ssm_states,
            "plastic_states": next_plastic_states,
        }

        if targets is not None:
            # Standard next-token CE loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                label_smoothing=self.config.label_smoothing
            )
            
            # Multi-Token Prediction (MTP) Loss - Disabled for 30 steps/sec Hyperdrive
            mtp_loss = 0.0
            # mtp_logits = self.mtp_head(x).float()
            # out["mtp_logits"] = mtp_logits
            # if targets.size(1) > 1:
            #     mtp_targets = targets[:, 1:]
            #     mtp_loss = F.cross_entropy(
            #         mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)),
            #         mtp_targets.reshape(-1)
            #     )
            
            # Combined Loss: Standard + 0.1*MTP + Aux (Titan v10 Stability)
            out["loss"] = ce_loss + 0.1 * mtp_loss + self.config.aux_loss_scale * aux_loss

        return out

    @staticmethod
    def _apply_repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
        if penalty <= 1.0 or generated.numel() == 0:
            return logits

        adjusted = logits.clone()
        for batch_idx in range(generated.size(0)):
            token_ids = torch.unique(generated[batch_idx])
            batch_logits = adjusted[batch_idx, token_ids]
            adjusted[batch_idx, token_ids] = torch.where(
                batch_logits < 0,
                batch_logits * penalty,
                batch_logits / penalty,
            )
        return adjusted

    @staticmethod
    def _sample_from_logits(
        logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tensor:
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        filtered = logits / max(temperature, 1e-5)

        if top_k > 0 and top_k < filtered.size(-1):
            top_values, _ = torch.topk(filtered, top_k, dim=-1)
            kth = top_values[:, -1].unsqueeze(-1)
            filtered = torch.where(filtered < kth, torch.full_like(filtered, float("-inf")), filtered)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            filtered = torch.full_like(filtered, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)

        probs = torch.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        adapt_during_generation: bool = False,
        ssm_external_states: Optional[list[Optional[Tensor]]] = None,
    ) -> Tensor:
        self.eval()

        outputs = self(
            input_ids,
            ssm_states=None,
            ssm_external_states=ssm_external_states,
            plastic_states=None,
            update_plasticity=True,
        )
        ssm_states = outputs["ssm_states"]
        plastic_states = outputs["plastic_states"]
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = outputs["logits"][:, -1]
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
            next_token = self._sample_from_logits(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break

            generated = torch.cat([generated, next_token], dim=1)
            outputs = self(
                next_token,
                ssm_states=ssm_states,
                ssm_external_states=ssm_external_states,
                plastic_states=plastic_states,
                update_plasticity=adapt_during_generation,
            )
            ssm_states = outputs["ssm_states"]
            plastic_states = outputs["plastic_states"]

        return generated

    def save_pretrained(self, save_directory: str | Path, vocab: Optional[dict[str, Any]] = None) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_dict = self.config.to_dict()
        (save_dir / "config.json").write_text(
            json.dumps(config_dict, indent=2),
            encoding="utf-8",
        )
        
        # Revert to Safetensors format
        from safetensors.torch import save_model as _save_safe
        save_path = save_dir / "model.safetensors"
        _save_safe(self, str(save_path), metadata={"format": "pt", "model_type": "neuroswift"})
        
        # Save vocab separately
        if vocab:
            (save_dir / "tokenizer_vocab.json").write_text(
                json.dumps(vocab, indent=2),
                encoding="utf-8"
            )
            
        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str | Path,
        device: str | torch.device | None = None,
        use_checkpoint: bool = False,
    ) -> "NeuroSwiftLM":
        if device is None:
            device = auto_device()
        save_dir = Path(save_directory)
        config_path = save_dir / "config.json"
        safe_path = save_dir / "model.safetensors"
        model_pt = save_dir / "model.pt"

        # Prioritize safetensors primary
        if safe_path.exists():
            config = NeuroSwiftConfig.from_dict(json.loads(config_path.read_text()))
            model = cls(config, use_checkpoint=use_checkpoint)
            from safetensors.torch import load_model as _load_safe
            _load_safe(model, safe_path, device=str(device))
            # Validation: Ensure weights are healthy
            if len(model.state_dict()) < 50:
                raise ValueError("Incomplete safetensors checkpoint.")
        elif model_pt.exists():
            # Support the temporary .pt bundle format for transition
            checkpoint = torch.load(model_pt, map_location=device)
            config = NeuroSwiftConfig.from_dict(checkpoint["config"])
            model = cls(config, use_checkpoint=use_checkpoint)
            model.load_state_dict(checkpoint["model_state"], strict=True)
        else:
            raise FileNotFoundError(f"No model found at {save_dir}")

        model.to(device)
        model.eval()
        return model

    def compile(self, backend: str = "inductor", mode: str = "reduce-overhead") -> nn.Module:
        """
        God-level optimization: torch.compile fusion.
        Requires Torch 2.0+ and a compatible toolchain (e.g., 'cl' on Windows, 'gcc' on Linux).
        """
        if not hasattr(torch, "compile"):
            logger.warning("torch.compile not available in this version of Torch. Falling back to eager.")
            return self
            
        # Windows-specific pre-flight for Inductor (requires MSVC cl.exe)
        if os.name == "nt" and backend == "inductor":
            import shutil
            if not shutil.which("cl"):
                logger.warning("[NeuroSwift] 'cl.exe' (MSVC) not found in PATH. Falling back to 'aot_eager' for Windows stability.")
                backend = "aot_eager"

        try:
            logger.info(f"Initializing 'Absolute Performance' Engine (backend={backend}) …")
            # Only pass 'mode' if backend is inductor (unsupported by aot_eager)
            compile_kwargs = {"backend": backend}
            if backend == "inductor":
                compile_kwargs["mode"] = mode
                
            return torch.compile(self, **compile_kwargs)
        except Exception as e:
            logger.warning(f"Compilation failed during initialization: {e}. Falling back to eager.")
            return self

    def save_partial(self, directory: str | Path, step: int, loss: float) -> None:
        """Save a resumable 'hot' checkpoint."""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {"step": step, "loss": loss, "config": self.config.to_dict()}
        (save_dir / "checkpoint_metadata.json").write_text(json.dumps(checkpoint, indent=2))
        self.save_pretrained(save_dir)

    @classmethod
    def from_partial(cls, directory: str | Path, device: str | torch.device = "cpu") -> tuple[Optional["NeuroSwiftLM"], int, float]:
        """Load from a partial checkpoint if it exists."""
        save_dir = Path(directory)
        meta_file = save_dir / "checkpoint_metadata.json"
        if not meta_file.exists():
            return None, 0, 0.0
        meta = json.loads(meta_file.read_text())
        model = cls.from_pretrained(save_dir, device=device)
        return model, meta["step"], meta["loss"]

    def evolve(self, new_config: NeuroSwiftConfig) -> "NeuroSwiftLM":
        """
        Transition current model to a new, larger config while keeping knowledge.
        God-Level Progressive Growth functionality.
        """
        from .evolution import WeightEvolutionEngine
        return WeightEvolutionEngine.evolve(self, new_config)

    @staticmethod
    def get_evolution_target(tokens_trained: int) -> Optional[dict[str, Any]]:
        """
        Suggest a next-level configuration based on trained dataset size.
        """
        # Thresholds (tokens)
        T_SMALL = 100_000   # 100k tokens
        T_MEDIUM = 1_000_000 # 1M tokens
        T_LARGE = 10_000_000 # 10M tokens
        T_HUGE = 100_000_000 # 100M tokens

        if tokens_trained < T_SMALL: return None
        if tokens_trained < T_MEDIUM:
            return {"d_model": 192, "n_layers": 6, "expert_hidden": 384}
        if tokens_trained < T_LARGE:
            return {"d_model": 256, "n_layers": 8, "expert_hidden": 512}
        if tokens_trained < T_HUGE:
            return {"d_model": 512, "n_layers": 12, "expert_hidden": 1024}
        return {"d_model": 768, "n_layers": 18, "expert_hidden": 1536}


__all__ = ["NeuroSwiftBlock", "NeuroSwiftConfig", "NeuroSwiftLM"]
