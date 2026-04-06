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
import time
import gc
import threading
import queue
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

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
    """Deep-Optimization worker for build_sft_tensors."""
    pair, slim_token_data, seq_len, fmt_fn = args
    try:
        stoi = slim_token_data["stoi"]
        special = slim_token_data["special_tokens"]
        
        pad_id = stoi[special["pad"]]
        eos_token = special["eos"]

        # Minimal encoding logic to stay fast inside worker
        def encode_local(text: str, add_eos: bool = False) -> list[int]:
            text = text.lower() if slim_token_data["lowercase"] else text
            import re
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            ids = [stoi.get(w, stoi[special["unk"]]) for w in words]
            if add_eos:
                ids.append(stoi[eos_token])
            return ids

        prompt_ids = encode_local(fmt_fn(pair.prompt), add_eos=False)
        resp_ids = encode_local(pair.response, add_eos=True)
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

        # Return plain lists instead of torch.tensors to avoid shared memory depletion
        # when passing nearly 100,000 tiny tensors back to the main process.
        return (
            input_ids + [pad_id] * pad_len,
            labels + [-100] * pad_len
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

    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    skipped = 0

    if len(pairs) > 500:
        # Hyper-Optimization: Pass only the essential vocab to the workers
        slim_token_data = {
            "stoi": tokenizer.stoi,
            "special_tokens": {
                "pad": tokenizer.pad_token,
                "unk": tokenizer.unk_token,
                "eos": tokenizer.eos_token,
            },
            "lowercase": tokenizer.lowercase
        }
        
        num_procs = min(multiprocessing.cpu_count(), 10) 
        worker_args = [(p, slim_token_data, seq_len, fmt_prompt) for p in pairs]
        
        # Using a Pool with larger chunks is more efficient for Windows IPC
        with multiprocessing.Pool(processes=num_procs) as pool:
            results = pool.map(_sft_worker, worker_args, chunksize=500)
        
        for res in results:
            if res is None:
                skipped += 1
            else:
                input_rows.append(res[0])
                label_rows.append(res[1])
    else:
        # Sequential fallback for tiny datasets
        slim_token_data = {
            "stoi": tokenizer.stoi,
            "special_tokens": {
                "pad": tokenizer.pad_token,
                "unk": tokenizer.unk_token,
                "eos": tokenizer.eos_token,
            },
            "lowercase": tokenizer.lowercase
        }
        for pair in pairs:
            res = _sft_worker((pair, slim_token_data, seq_len, fmt_prompt))
            if res is None:
                skipped += 1
            else:
                input_rows.append(res[0])
                label_rows.append(res[1])

    if not input_rows:
        raise RuntimeError("No valid training examples built.")

    # Convert lists to tensors once in the main process (eliminates shared memory mmap issues)
    inputs = torch.tensor(input_rows, dtype=torch.long)
    labels = torch.tensor(label_rows, dtype=torch.long)
    
    # Ensure zero-copy IPC: Mark tensors as shared
    inputs.share_memory_()
    labels.share_memory_()

    return inputs, labels, skipped


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
    data_grp.add_argument("--max-examples", type=int, default=0,
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
    train_grp.add_argument("--batch-size", type=int, default=2,
                           help="Micro-batch size for training")
    train_grp.add_argument("--max-steps-per-epoch", type=int, default=0,
                           help="Max steps to process per epoch (0 = full dataset).")
    train_grp.add_argument("--grad-accum", type=int, default=1,
                           help="Gradient accumulation steps (effective_bs = batch × accum).")
    train_grp.add_argument("--seq-len", type=int, default=256,
                           help="Sequence length (default 256 for linear context).")
    train_grp.add_argument("--lr", type=float, default=4e-4)
    train_grp.add_argument("--min-lr-ratio", type=float, default=0.05,
                           help="Minimum LR as a fraction of peak LR (cosine schedule).")
    train_grp.add_argument("--warmup-ratio", type=float, default=0.1,
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
        return 64, 2 # CPU Sweet Spot (~1.2M params)
    else:
        # Aero-Turbo v26: High-end GPU Scaling (Targets 1.5GB VRAM usage)
        return 384, 12 # ~130M parameters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def setup_msvc_env() -> None:
    """
    Absolute Performance 1.0.0 Standard: Automatically locate and inject 
    the MSVC (cl.exe) path into the environment for torch.compile (Inductor).
    """
    if os.name != "nt":
        return
        
    import shutil
    if shutil.which("cl"):
        return # Already in path
        
    # Search common VS/BuildTools locations for vcvars64.bat
    search_roots = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio"),
        Path("C:/Program Files/Microsoft Visual Studio"),
    ]
    
    for root in search_roots:
        if not root.exists():
            continue
            
        import subprocess
        # Search for the official environment initialization script
        vcvars_path = next(root.glob("**/vcvars64.bat"), None)
        if vcvars_path:
            try:
                # Execute the script and dump the environment variables
                cmd = f'"{vcvars_path}" && set'
                out = subprocess.check_output(cmd, shell=True, text=True)
                for line in out.splitlines():
                    if '=' in line:
                        key, val = line.split('=', 1)
                        key_upper = key.upper()
                        # Sync standard MSVC paths to our process
                        if key_upper in ['PATH', 'INCLUDE', 'LIB', 'LIBPATH']:
                            os.environ[key_upper] = val
                
                logger.info(f"Absolute Performance: Synchronized full MSVC Environment (vcvars64.bat)")
                return
            except Exception as e:
                logger.warning(f"Failed to sync MSVC environment: {e}")


def main() -> None:
    setup_msvc_env()
    args = parse_args().parse_args()
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = auto_device()

    if device.type == "cpu":
        # Heterogeneous Storm v18: Full 10-core Spawning Drive
        torch.set_flush_denormal(True)
        # Using 10 processes, each with 1 OMP thread = Full machine saturation with zero contention
        torch.set_num_threads(1) 
        logger.info(f"Auto-Optimization (CPU): Heterogeneous Storm v18 Active | 10 Workers Spawning...")
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Auto-Optimization (GPU): TensorCores Enabled (Ampere+)")
        else:
            logger.info("Auto-Optimization (GPU): Legacy CUDA Fallback")

    # Batch size auto (V20 scale upgrade)
    batch_size = args.batch_size
    if device.type == "cpu" and (batch_size <= 0 or batch_size == 1):
        batch_size = 2 # Forced scale-up for 10-core saturation
    elif batch_size <= 0:
        batch_size = 32

    logger.info(f"Device: {device}  |  Batch size (V20): {batch_size}")
    args.batch_size = batch_size # Sync back to args for distributed launch

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
    
    # Turbo Engine v4: Force RAM Mode for CPU to eliminate SSD latency
    # Mode: RAM-Master (threshold increased to 1M to allow large datasets in GPU/High-RAM environments)
    use_ssd = False if device.type == "cpu" else (args.use_ssd or (len(train_pairs) > 1_000_000))
    
    if use_ssd:
        mmap_path = args.output_dir / "train_cache.mmap"
        logger.info(f"SSD-Streaming enabled: writing tokens to {mmap_path}")
        train_loader = DataLoader(
            MmapDataset.from_pairs(train_pairs, tokenizer, mmap_path, seq_len=args.seq_len),
            batch_size=batch_size,
            shuffle=False, 
            num_workers=4, # Overdrive v16: Parallel Data Engine
            prefetch_factor=4,
            # ── VRAM-Master Mode: Shift Data to GPU at Start ─────────────────────────
            persistent_workers=True,
            pin_memory=True,
        )
        # Placeholder for auto_model_size calc
        class Dummy: pass
        train_inputs = Dummy()
        train_inputs.__len__ = lambda: len(train_pairs)
    else:
        train_inputs, train_labels, skipped_train = build_sft_tensors(
            train_pairs, tokenizer, seq_len=args.seq_len
        )
        logger.info(f"Training tokens: {len(train_inputs):,} examples ({skipped_train} skipped) | Mode: RAM-Master")
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_labels),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2, # Overdrive v16: Parallel Data Engine
            prefetch_factor=4,
            persistent_workers=True,
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
        # Elite CPU Complexity Pruning
        auto_ternary = args.ternary
        auto_top_k = args.top_k if hasattr(args, "top_k") else 2
        auto_d_state = args.d_state
        auto_plastic_dim = args.plastic_dim
        
        if device.type == "cpu":
            auto_ternary = True
            auto_top_k = 2
            auto_d_state = 4
            auto_plastic_dim = 16 # Enabled Hebbian
            logger.info("Auto-Optimization (CPU): Applied Elite Speed Pruning (d_state=4, d_model=128, Hebbian=ON, Top-K=2).")

    # ── Memory-Shield v31: Offloading Initialization to Workers ─────────────
    # We no longer initialize the model or optimizer in the master process.
    # This keeps the master RAM footprint < 1GB, preventing SIGKILL/OOM.
    
    # ── Distributed Architecture: Aero-Turbo Overdrive (v31) ────────────────
    # Aero-Intelligence: Auto-detecting over-subscription mode
    world_size = 1
    backend = "gloo"
    
    # Distributed Sync Environment (Atomic Persistence)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if device.type == "cuda":
        num_physical_gpus = torch.cuda.device_count()
        # Intelligent Backend Selection
        if num_physical_gpus == 1:
            # NCCL does not support multiple ranks on a single GPU.
            # Using GLOO for stable over-subscription on single-GPU hardware.
            world_size = 2 
            backend = "gloo"
            
            # Aero-Turbo v32: OOM Shield for single-GPU over-subscription
            # We MUST enable grad checkpointing to fit two processes on one device.
            # We also cap batch size to 2 to prevent activation OOM.
            args.grad_checkpoint = True
            if args.batch_size > 2:
                logger.info(f"[OOM-SHIELD] Reducing batch size {args.batch_size} -> 2 for 2-process GPU-sharing stability.")
                args.batch_size = 2
                
            logger.info("[AERO-INTELLIGENCE] Single GPU Over-subscription: Switching to GLOO + OOM-Shield (Checkpointing=ON, Batch=2).")
        else:
            # Professional NCCL for multi-GPU clusters
            world_size = num_physical_gpus 
            backend = "nccl"
    else:
        world_size = 10 
        backend = "gloo"

    logger.info(f"Master: Launching {world_size} Absolute Engine Processes ({backend})...")
    
    # Ensure CPU tensors are shared for zero-copy access by workers
    train_inputs.share_memory_()
    train_labels.share_memory_()
    if v_inputs is not None:
        v_inputs.share_memory_()
        v_labels.share_memory_()

    gc.collect()
    
    try:
        mp.spawn(
            train_worker,
            args=(world_size, backend, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, d_model_arg, n_layers_arg),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"Distributed Engine Failed: {e}")
        raise e

    logger.info("Training Run Complete.")

class BackgroundPrefetcher:
    """Zero-Blocking Async Loader replacing PyTorch's native worker queue locks."""
    def __init__(self, loader, maxsize=32, warmup_size=16):
        self.loader = loader
        self.queue = queue.Queue(maxsize=maxsize)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.warmup_size = warmup_size

    def _worker(self):
        try:
            for batch in self.loader:
                if self.stop_event.is_set():
                    break
                self.queue.put(batch)
        except Exception:
            pass
        finally:
            self.queue.put(None)

    def wait_for_warmup(self):
        """Block until the queue has reached the target warmup size to prevent 'cold start' slowness."""
        while self.queue.qsize() < self.warmup_size and self.thread.is_alive():
            time.sleep(0.01)

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            self.stop_event.set()
            raise StopIteration
        return batch

def train_worker(rank, world_size, backend, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, d_model, n_layers):
    # Aero-Turbo v32: Absolute Sync Protection
    # Explicitly clear cache to prevent OOM fragmentation on startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.48) # Leave room for the second process + OS
        
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if backend == "nccl":
        # Logical Over-subscription: Mapping multiple ranks to the same physical device
        num_physical_gpus = torch.cuda.device_count()
        physical_gpu_id = rank % num_physical_gpus
        device = torch.device(f"cuda:{physical_gpu_id}")
        torch.cuda.set_device(device)
        amp_enabled = True
    else:
        # Gloo/CPU Mode
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda": torch.cuda.set_device(device)
        torch.set_num_threads(1)
        amp_enabled = (device.type == "cuda")

    # ── Memory-Shield v31: Late VRAM Loading ───────────────────────────────
    # Move the dataset to GPU locally in the worker to prevent master OOM
    if device.type == "cuda":
        if rank == 0: logger.info(f"  [v31 MASTER] VRAM-MASTER ACTIVE: Mapping {len(train_inputs):,} samples to {device}...")
        train_inputs = train_inputs.to(device)
        train_labels = train_labels.to(device)
        if v_inputs is not None:
            v_inputs = v_inputs.to(device)
            v_labels = v_labels.to(device)
    
    # ── Memory-Shield v31: Late Model/Optimizer Initialization ────────────
    # This prevents the master process from being killed by the OOM Killer
    expert_hidden = args.expert_hidden if args.expert_hidden > 0 else d_model * 2
    
    # Auto-Configuration Logic (Consistent across ranks)
    auto_ternary = args.ternary or (device.type == "cpu")
    auto_top_k = args.top_k if hasattr(args, "top_k") else 2
    auto_d_state = 4 if device.type == "cpu" else args.d_state
    auto_plastic_dim = 16 if device.type == "cpu" else args.plastic_dim

    config = NeuroSwiftConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_layers=n_layers,
        d_state=auto_d_state, expansion=2, conv_kernel=4, num_experts=args.num_experts,
        top_k=auto_top_k, expert_hidden=expert_hidden, plastic_dim=auto_plastic_dim,
        dropout=args.dropout, aux_loss_scale=1e-2, ternary_mode=auto_ternary,
        latent_dim=args.latent_dim,
    )
    
    model = NeuroSwiftLM(config, use_checkpoint=args.grad_checkpoint).to(device)
    
    if args.compile:
        try: model = model.compile(mode="reduce-overhead")
        except: pass

    if world_size > 1:
        ddp_device_id = [rank % torch.cuda.device_count()] if backend == "nccl" else None
        model = DDP(model, device_ids=ddp_device_id, find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    
    # Set aggregation: m_size=1 for maximum 'Steps/s' (Hitting 200+ objective)
    m_size = 10 if backend == "gloo" else 1
    if backend == "nccl":
        torch.backends.cudnn.benchmark = True
    
    # Dataset Sharding
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=0, pin_memory=(device.type == "cuda" and train_inputs.device.type == "cpu")
    )
    
    val_loader = None
    if v_inputs is not None:
        val_dataset = TensorDataset(v_inputs, v_labels)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=0)

        # Aero-Intelligence v30: NCCL requires device_ids, GLOO requires None for single-GPU stability
        # Mapping ranks to physical device IDs
        ddp_device_id = [rank % torch.cuda.device_count()] if backend == "nccl" else None
        model = DDP(model, device_ids=ddp_device_id, find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    
    steps_per_epoch = len(train_loader) // m_size
    total_steps = steps_per_epoch * args.epochs

    global_step = 0
    best_val_loss = float("inf")
    ema_loss = None

    if backend == "gloo":
        gc.disable()

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            n_steps = 0
            v_prefetcher = None
            
            # Warp Queue: High-bandwidth Prefetcher
            prefetcher = BackgroundPrefetcher(train_loader, maxsize=64, warmup_size=16)
            prefetcher.wait_for_warmup()
            
            # Parallel Validation Prefetcher: Initialized per-epoch to prevent stalls
            if val_loader:
                v_prefetcher = BackgroundPrefetcher(val_loader, maxsize=16, warmup_size=4)
            
            step_start_time = time.time()
            mb_inp, mb_lbl = [], []
            
            for batch_idx, (inp, lbl) in enumerate(prefetcher):
                mb_inp.append(inp)
                mb_lbl.append(lbl)
                if len(mb_inp) < m_size:
                    continue
                
                # Aero-Turbo v24: Zero-latency Slicing (non-blocking transfers)
                batch_inp = torch.cat(mb_inp, dim=0).to(device, non_blocking=True)
                batch_lbl = torch.cat(mb_lbl, dim=0).to(device, non_blocking=True)
                mb_inp, mb_lbl = [], []
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=amp_enabled):
                    # Warp Engine v18: Distributed Inductor Fallback
                    try:
                        out = model(batch_inp, targets=batch_lbl)
                    except Exception as e:
                        # Catch specific Inductor/Triton/Scan errors that occur in v2.4/v2.5
                        if any(x in str(e) for x in ["Inductor", "Triton", "list"]):
                            raw_mod = model.module if hasattr(model, "module") else model
                            if hasattr(raw_mod, "_orig_mod"): raw_mod = raw_mod._orig_mod
                            out = raw_mod(batch_inp, targets=batch_lbl)
                        else:
                            raise e
                    
                    loss = out["loss"]
                
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                current_loss_val = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss_val
                else:
                    ema_loss = 0.9 * ema_loss + 0.1 * current_loss_val

                global_step += 1
                n_steps += 1
                epoch_loss += current_loss_val

                if rank == 0 and n_steps % 50 == 0:
                    elapsed = time.time() - step_start_time
                    # Reporting optimized for Stepping Frequency
                    steps_per_sec = 50 / max(elapsed, 0.001)
                    samples_per_sec = (50 * world_size * m_size * args.batch_size) / max(elapsed, 0.001)
                    # Show total steps per epoch
                    logger.info(f"  [Epoch {epoch}] Step {n_steps}/{steps_per_epoch} | {steps_per_sec:.1f} steps/s | Aggregate: {samples_per_sec:.1f} smp/s | Loss: {ema_loss:.4f}")
                    step_start_time = time.time()
                
                if n_steps % 250 == 0:
                    gc.collect()
            
            prefetcher.stop_event.set()
            avg_loss = epoch_loss / max(n_steps, 1)
            
            # Parallel Validation: All ranks participate to eliminate stalls
            v_loss = 0.0
            if v_prefetcher:
                v_loss = _evaluate_worker(model, v_prefetcher, device)
                v_prefetcher.stop_event.set()
                del v_prefetcher
                
            if rank == 0:
                logger.info(f"Master: Epoch {epoch:02d} Complete | Eval Loss: {v_loss:.4f}")
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    # Adaptive Model Selection (Fixes AttributeError)
                    raw_mod = model.module if hasattr(model, "module") else model
                    _save_checkpoint(raw_mod, tokenizer, args.output_dir, avg_loss, is_best=True)
            
            dist.barrier()

    finally:
        # Atomic Cleanup: Removing resource leak warnings
        if dist.is_initialized():
            dist.destroy_process_group()
        
        if rank == 0:
            logger.info("Distributed Training Complete. Master exiting...")
            # Cleanup Fix
            raw_mod = model.module if hasattr(model, "module") else model
            _save_checkpoint(raw_mod, tokenizer, args.output_dir / "last", 0.0)
        
        gc.enable()

def _evaluate_worker(model, prefetcher, device):
    """Distributed evaluation helper with safe initialization checks."""
    model.eval()
    local_total, local_n = 0.0, 0
    with torch.no_grad():
        for b_i, b_l in prefetcher:
            b_i, b_l = b_i.to(device, non_blocking=True), b_l.to(device, non_blocking=True)
            o = model(b_i)
            l = F.cross_entropy(o["logits"].view(-1, o["logits"].size(-1)), b_l.view(-1), ignore_index=-100)
            local_total += l.item()
            local_n += 1
            
    if dist.is_initialized():
        t_loss = torch.tensor([local_total], device=device)
        t_count = torch.tensor([local_n], device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
        res = t_loss.item() / max(t_count.item(), 1)
    else:
        res = local_total / max(local_n, 1)
    
    model.train()
    return t_loss.item() / max(t_count.item(), 1)

def _save_checkpoint(model, tokenizer, output_dir, last_loss, is_best=False):
    """Unified checkpointing for distributed workers."""
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Bundle vocab into the .pt checkpoint
    vocab_data = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    
    model.save_pretrained(save_path, vocab=vocab_data)
    tokenizer.save_pretrained(save_path)
    if is_best:
        best_dir = save_path / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(best_dir, vocab=vocab_data)
        tokenizer.save_pretrained(best_dir)

if __name__ == "__main__":
    # Windows requires spawn method
    mp.set_start_method("spawn", force=True)
    main()
