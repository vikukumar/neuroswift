"""
neuroswift.auto_trainer
=======================
World Auto-Training — continuous self-supervised incremental learning.

Features
--------
* Device-agnostic: auto-selects GPU (CUDA / MPS) if available, else CPU
* Adaptive batch sizing: smaller on CPU (4-16), larger on GPU (32-128)
* Watchdog loop: polls ``data_dir`` every ``interval`` seconds for new files
* Seen-file tracking: only trains on newly added data (no redundant re-processing)
* Plasticity warm-up after each incremental batch
* Crash-resilient: saves checkpoint every ``save_every`` steps, resumes on restart
* Compatible with both ``NeuroSwiftLM`` (text-only) and ``NeuroSwiftOmni``
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data_pipeline import TrainPair, run_pipeline
from .layers import auto_device
from .model import NeuroSwiftConfig, NeuroSwiftLM
from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[NeuroSwift AutoTrain] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Helpers (re-used from train_small_llm logic)
# ---------------------------------------------------------------------------


def _format_prompt(prompt: str) -> str:
    return f"user: {prompt}\nassistant:"


def _extract_pair(payload: dict[str, Any]) -> dict[str, str] | None:
    prompt = str(payload.get("prompt", "")).strip()
    response = str(payload.get("response", "")).strip()
    if prompt and response:
        return {"prompt": prompt, "response": response}
    messages = payload.get("messages")
    if isinstance(messages, list):
        users, assistants = [], []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            content = str(msg.get("content", "")).strip()
            if role == "user" and content:
                users.append(content)
            elif role == "assistant" and content:
                assistants.append(content)
        if users and assistants:
            return {"prompt": users[-1], "response": assistants[-1]}
    return None


def _build_dataset(
    pairs: list[TrainPair],
    tokenizer: WordTokenizer,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    input_rows, label_rows = [], []
    pad_id = tokenizer.stoi[tokenizer.pad_token]
    for pair in pairs:
        prompt_ids = tokenizer.encode(_format_prompt(pair.prompt), add_eos=False)
        response_ids = tokenizer.encode(pair.response, add_eos=True)
        full_ids = (prompt_ids + response_ids)[: seq_len + 1]
        if len(full_ids) < 2 or len(prompt_ids) >= len(full_ids):
            continue
        input_ids = full_ids[:-1]
        labels = [-100 if i < len(prompt_ids) else full_ids[i + 1] for i in range(len(input_ids))]
        if not any(l != -100 for l in labels):
            continue
        pad_len = seq_len - len(input_ids)
        input_rows.append(torch.tensor(input_ids + [pad_id] * pad_len, dtype=torch.long))
        label_rows.append(torch.tensor(labels + [-100] * pad_len, dtype=torch.long))
    if not input_rows:
        return None
    return torch.stack(input_rows), torch.stack(label_rows)


# ---------------------------------------------------------------------------
# AutoTrainer
# ---------------------------------------------------------------------------


class AutoTrainer:
    """
    Continuous self-supervised training loop for NeuroSwiftLM.

    Usage::

        trainer = AutoTrainer(data_dir="my_data", output_dir="artifacts/auto")
        trainer.run()          # blocks; Ctrl+C to stop

    Or one-shot::

        trainer.train_once()
    """

    def __init__(
        self,
        data_dir: str | Path,
        output_dir: str | Path = "artifacts/auto-trained",
        *,
        device: torch.device | None = None,
        epochs_per_cycle: int = 2,
        seq_len: int = 96,
        lr: float = 2e-3,
        weight_decay: float = 1e-2,
        max_examples: int = 4096,
        save_every: int = 50,          # save checkpoint every N batches
        interval: float = 300.0,       # polling interval in seconds (5 min default)
        min_pairs: int = 4,            # minimum new pairs to trigger a training cycle
        plasticity_warmup: bool = True,
        freeze_first_n_layers: int = 0,  # freeze first N blocks for faster incremental updates
        evolution: bool = False,       # Enable progressive model growth
        use_ui: bool = True,          # Use rich live dashboard
        verbose: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.device = device or auto_device()
        self.epochs_per_cycle = epochs_per_cycle
        self.seq_len = seq_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_examples = max_examples
        self.save_every = save_every
        self.interval = interval
        self.min_pairs = min_pairs
        self.plasticity_warmup = plasticity_warmup
        self.freeze_first_n_layers = freeze_first_n_layers
        self.evolution = evolution
        self.use_ui = use_ui
        self.verbose = verbose

        # Adaptive batch size
        if self.device.type == "cpu":
            self.batch_size = 8
        elif self.device.type == "mps":
            self.batch_size = 16
        else:  # CUDA
            self.batch_size = 64

        # State
        self.model: Optional[NeuroSwiftLM] = None
        self.tokenizer: Optional[WordTokenizer] = None
        self.optimizer: Optional[torch.optim.AdamW] = None
        self.seen_files: set[str] = set()
        self.global_step: int = 0
        self.cycle: int = 0
        self.total_tokens_trained: int = 0
        self.ui_context: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Bootstrap / resume
    # ------------------------------------------------------------------

    def _build_or_load_model(self, pairs: list[TrainPair]) -> None:
        """Build tokenizer + model from scratch or load from checkpoint."""
        ckpt_path = self.output_dir / "model.safetensors"
        tok_path = self.output_dir / "tokenizer.json"

        # Try to load partial model first (God-level resumption)
        partial_model, step, loss = NeuroSwiftLM.from_partial(self.output_dir, device=self.device)
        if partial_model is not None:
            self._log(f"Resuming from partial checkpoint at step {step} (loss={loss:.4f}) …")
            self.model = partial_model
            self.tokenizer = WordTokenizer.from_pretrained(self.output_dir)
            self.global_step = step
        elif ckpt_path.exists() and tok_path.exists():
            self._log("Resuming from existing full checkpoint …")
            self.tokenizer = WordTokenizer.from_pretrained(self.output_dir)
            self.model = NeuroSwiftLM.from_pretrained(self.output_dir, device=self.device)
        else:
            self._log("Building new model from scratch …")
            import random as _rnd
            texts = []
            for p in pairs:
                texts.append(_format_prompt(p.prompt))
                texts.append(p.response)
            # Cap vocab texts
            if len(texts) > 10_000:
                texts = _rnd.sample(texts, 10_000)
            self.tokenizer = WordTokenizer.from_texts(texts)

            # Adaptive model size
            if self.device.type == "cpu":
                config = NeuroSwiftConfig(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=128, n_layers=4, d_state=16,
                    expansion=2, conv_kernel=4, num_experts=8,
                    top_k=2, expert_hidden=256, plastic_dim=48,
                    dropout=0.05, aux_loss_scale=1e-2,
                )
            else:
                config = NeuroSwiftConfig(
                    vocab_size=self.tokenizer.vocab_size,
                    d_model=256, n_layers=8, d_state=32,
                    expansion=2, conv_kernel=4, num_experts=8,
                    top_k=2, expert_hidden=512, plastic_dim=64,
                    dropout=0.05, aux_loss_scale=1e-2,
                )
            self.model = NeuroSwiftLM(config).to(self.device)

        assert self.model is not None
        assert self.tokenizer is not None

        # Freeze first N layers for incremental efficiency
        if self.freeze_first_n_layers > 0:
            for idx, block in enumerate(self.model.blocks):
                if idx < self.freeze_first_n_layers:
                    for p in block.parameters():
                        p.requires_grad = False
            self._log(f"Frozen first {self.freeze_first_n_layers} blocks.")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(
            f"Model: {sum(p.numel() for p in self.model.parameters()):,} params "
            f"({trainable:,} trainable) on {self.device}"
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training logic
    # ------------------------------------------------------------------

    def _train_pairs(self, pairs: list[TrainPair]) -> float:
        """Train for configured epochs on given pairs. Returns mean loss."""
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.optimizer is not None

        result = _build_dataset(pairs, self.tokenizer, self.seq_len)
        if result is None:
            self._log("No usable examples built — skipping cycle.")
            return float("nan")

        inputs, targets = result
        
        # Use God-level MmapDataset for very large data (SSD-backed streaming)
        if len(pairs) > 5000:
            from .streaming import MmapDataset
            mmap_file = self.output_dir / f"stream_{self.cycle}.mmap"
            self._log(f"Dataset too large for RAM. Streaming from SSD: {mmap_file}")
            loader = DataLoader(
                MmapDataset.from_pairs(pairs, self.tokenizer, mmap_file, seq_len=self.seq_len),
                batch_size=self.batch_size,
                shuffle=False, # MmapDataset is usually used for sequential or worker-split streaming
            )
        else:
            loader = DataLoader(
                TensorDataset(inputs, targets),
                batch_size=self.batch_size,
                shuffle=True,
            )
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # Setup rich UI if enabled
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
        if self.use_ui:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            step_task = progress.add_task(f"Training Cycle {self.cycle}", total=len(loader) * self.epochs_per_cycle)
            progress.start()
        else:
            progress = None

        for epoch in range(self.epochs_per_cycle):
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                out = self.model(batch_inputs, targets=batch_targets, update_plasticity=False)
                loss = out["loss"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                self.global_step += 1
                self.total_tokens_trained += batch_inputs.numel()

                if progress:
                    progress.update(step_task, advance=1, description=f"Cycle {self.cycle} | Loss: {loss.item():.4f}")

                if self.global_step % self.save_every == 0:
                    self.model.save_partial(self.output_dir, self.global_step, loss.item())
                    self._checkpoint()

            self._log(f"  Epoch {epoch + 1}/{self.epochs_per_cycle} — loss={total_loss / max(n_batches, 1):.4f}")

        if progress:
            progress.stop()

        # Plasticity warmup pass
        if self.plasticity_warmup and pairs:
            self.model.eval()
            sample_text = _format_prompt(pairs[0].prompt)
            ids = torch.tensor([self.tokenizer.encode(sample_text)], dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.model(ids, update_plasticity=True)
            self._log("  Plasticity warmup complete.")

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # File watching
    # ------------------------------------------------------------------

    def _get_new_files(self) -> list[Path]:
        new_files: list[Path] = []
        for p in sorted(self.data_dir.rglob("*")):
            if p.is_file() and str(p) not in self.seen_files:
                new_files.append(p)
        return new_files

    def _ingest_new_files(self, new_files: list[Path]) -> list[TrainPair]:
        """Run data_pipeline on new files and mark them as seen."""
        from .data_pipeline import ingest_path, normalize_pair, filter_pair, deduplicate
        raw: list[TrainPair] = []
        for path in new_files:
            try:
                for pair in ingest_path(path):
                    raw.append(pair)
            except Exception as exc:
                self._log(f"Failed reading {path}: {exc}")
            self.seen_files.add(str(path))

        normed = [normalize_pair(p) for p in raw]
        filtered = [p for p in normed if filter_pair(p, min_response_words=1)]
        deduped = deduplicate(filtered, exact=True, near_dup=False)
        return deduped

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        assert self.model is not None
        assert self.tokenizer is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        summary_path = self.output_dir / "auto_train_state.json"
        summary_path.write_text(
            json.dumps(
                {
                    "global_step": self.global_step,
                    "cycle": self.cycle,
                    "seen_files": sorted(self.seen_files),
                    "device": str(self.device),
                    "total_tokens_trained": self.total_tokens_trained,
                    "evolution_enabled": self.evolution,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._log(f"Checkpoint saved (step {self.global_step}).")

    def _restore_state(self) -> None:
        state_path = self.output_dir / "auto_train_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.global_step = int(state.get("global_step", 0))
            self.cycle = int(state.get("cycle", 0))
            self.seen_files = set(state.get("seen_files", []))
            self.total_tokens_trained = int(state.get("total_tokens_trained", 0))
            self._log(f"Restored state: step={self.global_step}, cycle={self.cycle}, tokens={self.total_tokens_trained}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_once(self) -> float:
        """Run a single training cycle over all data in ``data_dir``."""
        if not self.data_dir.exists():
            self._log(f"Data directory not found: {self.data_dir}. Nothing to train on.")
            return float("nan")

        train_pairs, _, pipe_stats = run_pipeline(
            source=self.data_dir,
            max_total_pairs=self.max_examples,
            max_pairs_per_file=5000,
            dedup_exact=True,
            dedup_near=False,
            val_fraction=0.0,
            verbose=self.verbose,
        )
        if len(train_pairs) < self.min_pairs:
            self._log(f"Only {len(train_pairs)} pairs found (min={self.min_pairs}). Skipping.")
            return float("nan")

        self._log(f"Training on {len(train_pairs)} pairs from {self.data_dir} …")
        if self.model is None:
            self._build_or_load_model(train_pairs)

        loss = self._train_pairs(train_pairs)
        self._checkpoint()
        return loss

    def run(self, max_cycles: int = 0) -> None:
        """Continuous auto-training loop.

        Args:
            max_cycles: Stop after this many cycles (0 = run forever).
        """
        self._log(
            f"Starting Auto-Training | device={self.device} | "
            f"batch={self.batch_size} | interval={self.interval}s"
        )
        self._restore_state()

        if not self.data_dir.exists():
            self._log(f"Creating data directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initial bootstrap with all existing data
        train_pairs, _, _ = run_pipeline(
            source=self.data_dir,
            max_total_pairs=self.max_examples,
            max_pairs_per_file=5000,
            dedup_exact=True,
            dedup_near=False,
            val_fraction=0.0,
            verbose=self.verbose,
        )
        for path in sorted(self.data_dir.rglob("*")):
            if path.is_file():
                self.seen_files.add(str(path))

        if self.model is None:
            if train_pairs:
                self._log(f"Bootstrap: {len(train_pairs)} pairs found.")
                self._build_or_load_model(train_pairs)
                self._train_pairs(train_pairs)
                self._checkpoint()
            else:
                self._log("No initial data. Building minimal model; waiting for data …")
                from .data_pipeline import TrainPair
                dummy_pairs = [
                    TrainPair(prompt="hello", response="hi"),
                    TrainPair(prompt="who are you", response="i am neuroswift"),
                ] * 20
                self._build_or_load_model(dummy_pairs)

        try:
            while True:
                self.cycle += 1
                self._log(f"=== Cycle {self.cycle} | Watching {self.data_dir} ===")
                time.sleep(self.interval)

                new_files = self._get_new_files()
                if not new_files:
                    self._log("No new files detected.")
                    if max_cycles > 0 and self.cycle >= max_cycles:
                        break
                    continue

                self._log(f"Detected {len(new_files)} new file(s):")
                for f in new_files:
                    self._log(f"  + {f}")

                new_pairs = self._ingest_new_files(new_files)
                if len(new_pairs) < self.min_pairs:
                    self._log(f"Only {len(new_pairs)} new pairs extracted. Waiting for more data.")
                    if max_cycles > 0 and self.cycle >= max_cycles:
                        break
                    continue

                self._log(f"Training on {len(new_pairs)} new pairs …")
                loss = self._train_pairs(new_pairs)
                self._log(f"Cycle {self.cycle} done. Mean loss: {loss:.4f}")
                
                # Check for Evolution (Progressive Growth)
                if self.evolution:
                    target_config_dict = self.model.get_evolution_target(self.total_tokens_trained)
                    if target_config_dict:
                        # Check if it actually describes a larger model than current
                        current_c = self.model.config
                        if target_config_dict["d_model"] > current_c.d_model or target_config_dict["n_layers"] > current_c.n_layers:
                            self._log(f"!!! TRIGGERING EVOLUTION !!! Tokens Trained: {self.total_tokens_trained:,}")
                            new_c = NeuroSwiftConfig(
                                vocab_size=current_c.vocab_size,
                                d_model=target_config_dict["d_model"],
                                n_layers=target_config_dict["n_layers"],
                                d_state=current_c.d_state,
                                expert_hidden=target_config_dict["expert_hidden"],
                                # others stay same
                            )
                            self.model = self.model.evolve(new_c)
                            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                            self._checkpoint()

                self._checkpoint()

                if max_cycles > 0 and self.cycle >= max_cycles:
                    break

        except KeyboardInterrupt:
            self._log("Interrupted by user. Saving checkpoint …")
            self._checkpoint()

        self._log("Auto-training stopped.")

    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)


__all__ = ["AutoTrainer"]
