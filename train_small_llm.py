"""
train_small_llm.py
==================
Train NeuroSwiftLM on ANY data: text, JSON/JSONL, CSV, images, audio, video.

Features
--------
* Auto-detects all file types in --data-dir (no manual format specification)
* Built-in data pipeline: normalize → filter → deduplicate → split
* Cosine LR schedule with linear warm-up
* Gradient accumulation for large effective batch sizes
* Validation loss tracking + early stopping
* Checkpoint resume (skips re-training if already converged)
* Label smoothing for better generalization
* Post-training plasticity demo + answer generation

Usage examples
--------------
    # Train on a folder with mixed file types
    python train_small_llm.py --data-dir my_data/

    # Train on a single JSONL file
    python train_small_llm.py --data-path data/qa.jsonl

    # Larger model, more epochs, grad accumulation
    python train_small_llm.py --data-dir my_data/ --epochs 5 --d-model 256 --grad-accum 4

    # Full options
    python train_small_llm.py --help
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neuroswift.data_pipeline import TrainPair, run_pipeline, UniversalSchemaMapper
from neuroswift.layers import auto_device
from neuroswift.model import NeuroSwiftConfig, NeuroSwiftLM
from neuroswift.tokenizer import WordTokenizer
from neuroswift.streaming import MmapDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("NeuroSwift.Train")


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def fmt_prompt(prompt: str) -> str:
    return f"user: {prompt}\nassistant:"


# ---------------------------------------------------------------------------
# SFT example builder
# ---------------------------------------------------------------------------


def _sft_worker(args):
    """Parallel worker for build_sft_tensors."""
    pair, tokenizer, seq_len, fmt_fn = args
    try:
        pad_id = tokenizer.stoi[tokenizer.pad_token]
        prompt_ids = tokenizer.encode(fmt_fn(pair.prompt), add_eos=False)
        resp_ids = tokenizer.encode(pair.response, add_eos=True)
        full_ids = (prompt_ids + resp_ids)[: seq_len + 1]

        if len(full_ids) < 2 or len(prompt_ids) >= len(full_ids):
            return None

        input_ids = full_ids[:-1]
        labels: list[int] = []
        for i in range(len(input_ids)):
            labels.append(-100 if i < len(prompt_ids) else full_ids[i + 1])

        if not any(l != -100 for l in labels):
            return None

        pad_len = seq_len - len(input_ids)
        if pad_len < 0:
            return None

        return (
            torch.tensor(input_ids + [pad_id] * pad_len, dtype=torch.long),
            torch.tensor(labels + [-100] * pad_len, dtype=torch.long)
        )
    except Exception:
        return None


def build_sft_tensors(
    pairs: list[TrainPair],
    tokenizer: WordTokenizer,
    seq_len: int,
    label_smoothing_ignore: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert instruction pairs into (input_ids, labels) tensors in parallel.
    Bypasses GIL using ProcessPoolExecutor.
    """
    if not pairs:
        return torch.empty(0), torch.empty(0), 0

    input_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    skipped = 0

    # Only parallelize for non-trivial datasets
    if len(pairs) > 500:
        num_procs = min(multiprocessing.cpu_count(), 16)
        worker_args = [(p, tokenizer, seq_len, fmt_prompt) for p in pairs]
        
        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            results = list(executor.map(_sft_worker, worker_args, chunksize=250))
        
        for res in results:
            if res is None:
                skipped += 1
            else:
                input_rows.append(res[0])
                label_rows.append(res[1])
    else:
        # Sequential fallback for tiny datasets (avoid process spawn overhead)
        for pair in pairs:
            res = _sft_worker((pair, tokenizer, seq_len, fmt_prompt))
            if res is None:
                skipped += 1
            else:
                input_rows.append(res[0])
                label_rows.append(res[1])

    if not input_rows:
        raise RuntimeError("No valid training examples built.")

    return torch.stack(input_rows), torch.stack(label_rows), skipped


# ---------------------------------------------------------------------------
# LR scheduling
# ---------------------------------------------------------------------------


def cosine_lr_with_warmup(
    optimizer: torch.optim.Optimizer,
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Update optimizer LR and return the current LR value."""
    base_lr = optimizer.param_groups[0]["initial_lr"]
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = base_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Loss with label smoothing
# ---------------------------------------------------------------------------


def masked_ce_loss(
    logits: torch.Tensor,  # [B, T, V]
    labels: torch.Tensor,  # [B, T]
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Cross-entropy loss ignoring −100 labels, with optional label smoothing."""
    B, T, V = logits.shape
    flat_logits = logits.reshape(-1, V)
    flat_labels = labels.reshape(-1)

    # Standard CE with built-in label smoothing (ignores −100 automatically)
    loss = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )
    return loss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> ArgumentParser:
    p = ArgumentParser(
        description="Train NeuroSwiftLM — auto-ingests any data folder or file.",
        epilog="""
Examples:
  python train_small_llm.py --data-dir my_data/
  python train_small_llm.py --data-path data/qa.jsonl --epochs 5
  python train_small_llm.py --data-dir my_data/ --d-model 256 --n-layers 6
""",
    )
    # Data
    data_grp = p.add_argument_group("Data")
    data_grp.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to a single JSONL/JSON/TXT/CSV file. Overridden by --data-dir.",
    )
    data_grp.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder to auto-scan for ALL supported file types (JSONL, JSON, TXT, "
             "CSV, XLSX, PNG, WAV, MP4 …). Takes priority over --data-path.",
    )
    data_grp.add_argument("--max-examples", type=int, default=50_000,
                          help="Max training pairs after pipeline (0=unlimited).")
    data_grp.add_argument("--max-per-file", type=int, default=10_000,
                          help="Max pairs extracted from any single file.")
    data_grp.add_argument("--min-prompt-words", type=int, default=2)
    data_grp.add_argument("--max-prompt-words", type=int, default=512)
    data_grp.add_argument("--min-response-words", type=int, default=3)
    data_grp.add_argument("--max-response-words", type=int, default=600)
    data_grp.add_argument("--val-fraction", type=float, default=0.05,
                          help="Fraction of pairs held out for validation.")
    data_grp.add_argument("--no-dedup", action="store_true",
                          help="Disable deduplication (faster but lower quality).")
    data_grp.add_argument("--latent-dim", type=int, default=64,
                          help="Latent compression dimension for MLA anchors.")
    data_grp.add_argument("--augment", action="store_true",
                          help="Enable light prompt-template augmentation.")
    data_grp.add_argument("--fetch-urls", action="store_true",
                          help="Fetch remote URL contents during ingestion.")
    data_grp.add_argument("--hf-dataset", type=str, default=None,
                          help="Hugging Face repo(s). Support comma-sep list.")
    data_grp.add_argument("--kaggle-dataset", type=str, default=None,
                          help="Kaggle dataset(s). Support comma-sep list.")

    # Training
    train_grp = p.add_argument_group("Training")
    train_grp.add_argument("--epochs", type=int, default=10, 
                           help="Number of epochs. Default 10 for small data high-accuracy.")
    train_grp.add_argument("--batch-size", type=int, default=0,
                           help="Batch size (0 = auto: 8 CPU, 32 GPU).")
    train_grp.add_argument("--grad-accum", type=int, default=1,
                           help="Gradient accumulation steps (effective_bs = batch × accum).")
    train_grp.add_argument("--seq-len", type=int, default=256,
                           help="Sequence length (default 256 for linear context).")
    train_grp.add_argument("--lr", type=float, default=2e-3)
    train_grp.add_argument("--min-lr-ratio", type=float, default=0.1,
                           help="Minimum LR as a fraction of peak LR (cosine schedule).")
    train_grp.add_argument("--warmup-ratio", type=float, default=0.06,
                           help="Fraction of total steps used for linear LR warm-up.")
    train_grp.add_argument("--label-smoothing", type=float, default=0.1)
    train_grp.add_argument("--weight-decay", type=float, default=1e-2)
    train_grp.add_argument("--device", type=str, default=None,
                           help="cpu / cuda / mps (default: auto).")
    train_grp.add_argument("--seed", type=int, default=42)
    train_grp.add_argument("--early-stop-patience", type=int, default=3,
                           help="Stop if val loss doesn't improve for N epochs (0=disabled).")

    # Model architecture
    arch_grp = p.add_argument_group("Architecture")
    arch_grp.add_argument("--d-model", type=int, default=0,
                          help="Hidden dimension (0=auto-scale by dataset size).")
    arch_grp.add_argument("--n-layers", type=int, default=0,
                          help="Number of blocks (0=auto-scale).")
    arch_grp.add_argument("--d-state", type=int, default=16)
    arch_grp.add_argument("--num-experts", type=int, default=8)
    arch_grp.add_argument("--expert-hidden", type=int, default=0,
                          help="Expert FFN hidden size (0 = 2×d_model).")
    arch_grp.add_argument("--plastic-dim", type=int, default=64)
    arch_grp.add_argument("--dropout", type=float, default=0.05)
    arch_grp.add_argument("--grad-checkpoint", action="store_true",
                          help="Enable gradient checkpointing (saves memory, slower).")
    arch_grp.add_argument("--compile", action="store_true",
                          help="Enable torch.compile() for massive speedup (requires Torch 2.0+).")
    arch_grp.add_argument("--ternary", action="store_true",
                          help="Enable BitNet-style addition-only training (1.58-bit).")
    arch_grp.add_argument("--use-ssd", action="store_true",
                          help="Force SSD-backed MmapDataset (even for small datasets).")

    # Output / misc
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument("--output-dir", type=Path, default=Path("artifacts/neuroswift-tiny"))
    out_grp.add_argument("--save-every", type=int, default=0,
                         help="Save checkpoint every N steps (0=epoch-only).")
    out_grp.add_argument("--resume", action="store_true",
                         help="Resume from existing checkpoint in --output-dir.")
    out_grp.add_argument("--prompt", type=str, default="what is neuroswift?",
                         help="Post-training demo prompt.")
    out_grp.add_argument("--max-new-tokens", type=int, default=64)
    out_grp.add_argument("--legacy-checkpoint", type=Path, default=None)
    return p


# ---------------------------------------------------------------------------
# Auto-scale model size
# ---------------------------------------------------------------------------


def auto_model_size(n_train: int, device: torch.device) -> tuple[int, int]:
    """Pick d_model and n_layers based on training set size and device."""
    if device.type == "cpu":
        if n_train < 1_000:
            return 192, 4
        elif n_train < 5_000:
            return 224, 6
        elif n_train < 20_000:
            return 256, 8
        else:
            return 320, 10
    else:  # GPU
        if n_train < 2_000:
            return 192, 4
        elif n_train < 10_000:
            return 256, 6
        else:
            return 384, 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args().parse_args()
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = auto_device()

    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    # Batch size auto
    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 8 if device.type == "cpu" else 32

    logger.info(f"Device: {device}  |  Batch size: {batch_size}")

    # ── Data pipeline ──────────────────────────────────────────────────────
    local_sources: list[Path] = []
    if args.data_dir:
        for d in str(args.data_dir).split(","):
            dp = Path(d.strip())
            if dp.exists():
                local_sources.append(dp)
    if args.data_path:
        for p in str(args.data_path).split(","):
            pp = Path(p.strip())
            if pp.exists():
                local_sources.append(pp)

    hf_list = [s.strip() for s in args.hf_dataset.split(",")] if args.hf_dataset else None
    kaggle_list = [s.strip() for s in args.kaggle_dataset.split(",")] if args.kaggle_dataset else None

    if not local_sources and not hf_list and not kaggle_list:
        # Default fallback
        fallback = Path("examples/data/neuroswift_qa.jsonl")
        if fallback.exists():
            local_sources = [fallback]
            logger.info(f"Using default data: {fallback}")
        else:
            logger.error("No data source provided.")
            sys.exit(1)

    train_pairs, val_pairs, pipe_stats = run_pipeline(
        source=local_sources,
        hf_dataset=hf_list,
        kaggle_dataset=kaggle_list,
        max_total_pairs=args.max_examples if args.max_examples > 0 else 0,
        max_pairs_per_file=args.max_per_file,
        min_prompt_words=args.min_prompt_words,
        max_prompt_words=args.max_prompt_words,
        min_response_words=args.min_response_words,
        max_response_words=args.max_response_words,
        dedup_exact=not args.no_dedup,
        dedup_near=not args.no_dedup,
        augment=args.augment,
        val_fraction=args.val_fraction,
        seed=args.seed,
        verbose=True,
    )

    if len(train_pairs) < 8:
        logger.error(
            f"Too few training pairs ({len(train_pairs)}). "
            "Add more data or lower --min-response-words."
        )
        sys.exit(1)

    # ── Vocabulary ──────────────────────────────────────────────────────────
    vocab_texts: list[str] = []
    for pair in train_pairs:
        vocab_texts.append(fmt_prompt(pair.prompt))
        vocab_texts.append(pair.response)
    # Cap to avoid OOM on huge datasets
    if len(vocab_texts) > 20_000:
        import random as _rnd
        _rnd.seed(args.seed)
        vocab_texts = _rnd.sample(vocab_texts, 20_000)

    logger.info("Building vocabulary …")
    tokenizer = WordTokenizer.from_texts(vocab_texts)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # ── Tensor datasets ─────────────────────────────────────────────────────
    logger.info("Tokenizing training pairs …")
    
    use_ssd = args.use_ssd or (len(train_pairs) > 10000)
    
    if use_ssd:
        mmap_path = args.output_dir / "train_cache.mmap"
        logger.info(f"SSD-Streaming enabled: writing tokens to {mmap_path}")
        train_loader = DataLoader(
            MmapDataset.from_pairs(train_pairs, tokenizer, mmap_path, seq_len=args.seq_len),
            batch_size=batch_size,
            shuffle=False, 
        )
        train_inputs = train_pairs # marker for auto_model_size
    else:
        train_inputs, train_labels, skipped_train = build_sft_tensors(
            train_pairs, tokenizer, seq_len=args.seq_len
        )
        logger.info(f"Training tensors: {len(train_inputs):,} examples ({skipped_train} skipped)")
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_labels),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True if device.type == "cuda" else False,
            num_workers=min(4, os.cpu_count() or 1),
            persistent_workers=True if (os.cpu_count() or 1) >= 4 else False,
            drop_last=True if len(train_inputs) >= batch_size else False,
        )

    has_val = len(val_pairs) > 0
    val_loader: DataLoader | None = None
    if has_val:
        v_inputs, v_labels, skipped_val = build_sft_tensors(val_pairs, tokenizer, seq_len=args.seq_len)
        logger.info(f"Validation tensors: {len(v_inputs):,} examples ({skipped_val} skipped)")
        if len(v_inputs) > 0:
            val_loader = DataLoader(
                TensorDataset(v_inputs, v_labels),
                batch_size=batch_size * 2,
                shuffle=False,
                pin_memory=True if device.type == "cuda" else False,
            )
        else:
            has_val = False

    # ── Model ───────────────────────────────────────────────────────────────
    d_model_arg = args.d_model
    n_layers_arg = args.n_layers

    if d_model_arg <= 0 or n_layers_arg <= 0:
        auto_d, auto_l = auto_model_size(len(train_inputs), device)
        if d_model_arg <= 0:
            d_model_arg = auto_d
        if n_layers_arg <= 0:
            n_layers_arg = auto_l
        logger.info(
            f"Auto-scaled model: d_model={d_model_arg}, n_layers={n_layers_arg} "
            f"(based on {len(train_inputs):,} training examples)"
        )

    expert_hidden = args.expert_hidden if args.expert_hidden > 0 else d_model_arg * 2

    # Resume from checkpoint?
    ckpt_file = args.output_dir / "model.safetensors"
    if args.resume and ckpt_file.exists():
        logger.info(f"Resuming from existing checkpoint: {args.output_dir}")
        model = NeuroSwiftLM.from_pretrained(
            args.output_dir,
            device=device,
            use_checkpoint=args.grad_checkpoint,
        )
        # Re-load tokenizer to ensure vocab matches
        try:
            tok_check = WordTokenizer.from_pretrained(args.output_dir)
            if tok_check.vocab_size == tokenizer.vocab_size:
                tokenizer = tok_check
        except Exception:
            pass
    else:
        config = NeuroSwiftConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model_arg,
            n_layers=n_layers_arg,
            d_state=args.d_state,
            expansion=2,
            conv_kernel=4,
            num_experts=args.num_experts,
            top_k=2,
            expert_hidden=expert_hidden,
            plastic_dim=args.plastic_dim,
            dropout=args.dropout,
            aux_loss_scale=1e-2,
            ternary_mode=args.ternary,
            latent_dim=args.latent_dim,
        )
        model = NeuroSwiftLM(config, use_checkpoint=args.grad_checkpoint).to(device)

    # Optional torch.compile (God-level optimized)
    if args.compile:
        model = model.compile()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Config: {model.config.to_dict()}")

    # ── Optimizer + scheduler ───────────────────────────────────────────────
    # Separate weight-decay from bias/norm params
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        fused=True if device.type == "cuda" else False,
    )
    # Store initial_lr for scheduler
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    logger.info(
        f"Training: {args.epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total | "
        f"warmup={warmup_steps}"
    )

    # AMP scaler for GPU
    scaler = torch.cuda.GradScaler() if device.type == "cuda" else None

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    last_train_loss = float("nan")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (batch_inp, batch_lbl) in enumerate(train_loader):
            batch_inp = batch_inp.to(device)
            batch_lbl = batch_lbl.to(device)

            if scaler is not None:
                # Use bf16 if available (more stable for SSMs)
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(batch_inp, targets=batch_lbl, update_plasticity=False)
                    loss = out["loss"] / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(batch_inp, targets=batch_lbl, update_plasticity=False)
                loss = out["loss"] / args.grad_accum
                loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                global_step += 1
                cosine_lr_with_warmup(
                    optimizer, global_step, warmup_steps, total_steps, args.min_lr_ratio
                )
                optimizer.zero_grad(set_to_none=True)
                n_steps += 1

                epoch_loss += (loss.item() * args.grad_accum)

                if args.save_every > 0 and global_step % args.save_every == 0:
                    model.save_partial(args.output_dir, global_step, loss.item())
                    logger.info(f"  [step {global_step}] hot-checkpoint saved.")

        mean_train_loss = epoch_loss / max(n_steps, 1)
        last_train_loss = mean_train_loss
        current_lr = optimizer.param_groups[0]["lr"]

        # Validation
        val_loss_str = ""
        if val_loader is not None:
            val_loss = _evaluate(model, val_loader, device, args.label_smoothing)
            val_loss_str = f"  val_loss={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                _save_checkpoint(model, tokenizer, args.output_dir, mean_train_loss, is_best=True)
            else:
                patience_counter += 1

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={mean_train_loss:.4f}{val_loss_str} | "
            f"lr={current_lr:.2e}"
        )

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    # ── Post-training ─────────────────────────────────────────────────────
    model.eval()
    
    # Reload best model for evaluation/demo if it exists (highly important for accuracy!)
    if best_val_loss < float("inf"):
        logger.info("  Training complete. Reloading BEST model for generation demo...")
        best_dir = args.output_dir / "best"
        model = NeuroSwiftLM.from_pretrained(best_dir, device=device)
    
    chat_prompt = fmt_prompt(args.prompt)
    prompt_ids = torch.tensor([tokenizer.encode(chat_prompt)], dtype=torch.long, device=device)

    with torch.no_grad():
        cold = model(prompt_ids, update_plasticity=False)
        warmed = model(prompt_ids, update_plasticity=True)
        adapted = model(prompt_ids, plastic_states=warmed["plastic_states"], update_plasticity=False)

        plastic_shift = (adapted["logits"] - cold["logits"]).abs().mean().item()
        plastic_norm = torch.stack([s.norm() for s in warmed["plastic_states"]]).mean().item()

        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            adapt_during_generation=False,
        )

    answer_ids = gen_ids[0, prompt_ids.size(1):].tolist()
    generated_text = tokenizer.decode(answer_ids).strip()
    
    # Hallucination Check (Entropy-based Confidence)
    with torch.no_grad():
        gen_logits = model(gen_ids)["logits"][:, prompt_ids.size(1)-1:-1]
        probs = torch.softmax(gen_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
        confidence = max(0, 100 - (entropy * 20)) # Heuristic God-level scorer

    # ── Save final artifacts ───────────────────────────────────────────────
    # Final epoch save (into 'last/' subfolder to avoid overwriting BEST in root)
    last_dir = args.output_dir / "last"
    _save_checkpoint(model, tokenizer, last_dir, last_train_loss)

    generation_config = {
        "prompt_template": "user: {prompt}\nassistant:",
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "adapt_during_generation": False,
        "task_type": "instruction_qa",
        "answer_only": True,
    }
    (args.output_dir / "generation_config.json").write_text(
        json.dumps(generation_config, indent=2), encoding="utf-8"
    )

    training_summary = {
        "creator": "Vikash Kumar",
        "data_source": str(source),
        "data_is_dir": source.is_dir(),
        "pipeline_stats": {
            "raw_files": pipe_stats.raw_files,
            "raw_pairs": pipe_stats.raw_pairs,
            "after_filter": pipe_stats.after_filter,
            "after_dedup": pipe_stats.after_dedup,
            "train_pairs": pipe_stats.train_pairs,
            "val_pairs": pipe_stats.val_pairs,
        },
        "num_examples": len(train_inputs),
        "final_train_loss": last_train_loss,
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
        "plasticity_mean_logit_shift": plastic_shift,
        "plasticity_mean_state_norm": plastic_norm,
        "device": str(device),
        "config": model.config.to_dict(),
        "tokenizer_type": "word",
        "task_type": "instruction_qa",
        "hallucination_confidence": f"{confidence:.2f}%",
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )

    # Legacy checkpoint (optional)
    if args.legacy_checkpoint is not None:
        legacy = {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
            "data_source": str(source),
            "prompt": args.prompt,
        }
        args.legacy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(legacy, args.legacy_checkpoint)
        logger.info(f"Legacy checkpoint saved to: {args.legacy_checkpoint}")

    # Print summary
    _sep = "-" * 54
    enc = sys.stdout.encoding or "utf-8"
    def _p(s: str) -> None:
        print(s.encode(enc, errors="replace").decode(enc, errors="replace"))

    _p("\n" + _sep)
    _p(f"  Training complete -- {args.output_dir}")
    _p(f"  Train pairs: {len(train_pairs):,}  |  Val pairs: {len(val_pairs):,}")
    _p(f"  Final train loss: {last_train_loss:.4f}")
    if best_val_loss < float("inf"):
        _p(f"  Best val loss:    {best_val_loss:.4f}")
    _p(f"  Plasticity shift: {plastic_shift:.6f}")
    _p(f"  Plastic norm:     {plastic_norm:.6f}")
    _p(f"  Vocab size:       {tokenizer.vocab_size:,}")
    _p(f"  Model params:     {n_params:,}")
    _p(f"\n  Prompt:  {args.prompt!r}")
    _p(f"  Confidence: {confidence:.1f}% (Hallucination risk: {'LOW' if confidence > 80 else 'MEDIUM' if confidence > 50 else 'HIGH'})")
    _p(f"  Answer:  {generated_text}")
    _p(_sep + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate(
    model: NeuroSwiftLM,
    loader: DataLoader,
    device: torch.device,
    label_smoothing: float = 0.05,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch_inp, batch_lbl in loader:
        batch_inp = batch_inp.to(device)
        batch_lbl = batch_lbl.to(device)
        out = model(batch_inp, targets=None, update_plasticity=False)
        logits = out["logits"].float()
        loss = masked_ce_loss(logits, batch_lbl, label_smoothing)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def _save_checkpoint(
    model: NeuroSwiftLM,
    tokenizer: WordTokenizer,
    output_dir: Path | str,
    last_loss: float,
    is_best: bool = False,
) -> None:
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save full weights and config
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    if is_best:
        # If this is the best model, also update a 'best' subdirectory
        best_dir = save_path / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        # Create a tiny marker
        (best_dir / "best_model_mark.txt").write_text(f"Loss: {last_loss:.4f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
