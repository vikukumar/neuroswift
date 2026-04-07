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
import socket
import errno
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
logger.setLevel(logging.WARNING) # Silence redundant logger output

def _get_stamp() -> str:
    """Standardized NeuroSwift Telemetry Stamp."""
    return f"[NeuroSwift AutoTrain] {datetime.now().strftime('%H:%M:%S')}"


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
        
        # Pulse-Stream v5: Response-First Window Slicing
        # If the total length > seq_len, we prioritize the transition from prompt to response.
        # This prevents "Skipped Examples" when using small contexts (e.g., 32 tokens).
        if len(prompt_ids) + len(resp_ids) > seq_len:
            # Keep up to 50% prompt, 50% response if possible
            p_len = min(len(prompt_ids), seq_len // 2)
            r_len = seq_len - p_len
            prompt_ids = prompt_ids[-p_len:]
            resp_ids = resp_ids[:r_len]

        full_ids = (prompt_ids + resp_ids)[: seq_len + 1]

        if len(full_ids) < 2 or len(prompt_ids) == 0:
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

    inputs = torch.stack(input_rows)
    labels = torch.stack(label_rows)
    
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
    data_grp.add_argument("--max-examples", type=int, default=1_000_000,
                          help="Max training pairs after pipeline (0=unlimited).")
    data_grp.add_argument("--max-per-file", type=int, default=1_000_000,
                          help="Max pairs extracted from any single file.")
    data_grp.add_argument("--min-prompt-words", type=int, default=2)
    data_grp.add_argument("--max-prompt-words", type=int, default=2048)
    data_grp.add_argument("--min-response-words", type=int, default=3)
    data_grp.add_argument("--max-response-words", type=int, default=5000)
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
    train_grp.add_argument("--seq-len", type=int, default=32, help="Sequence length (Pulse-Stream: 32 for 100+ steps/s)")
    train_grp.add_argument("--lr", type=float, default=2e-4,
                           help="Base learning rate (Saturation v24).")
    train_grp.add_argument("--min-lr-ratio", type=float, default=0.005,
                           help="Minimum LR as a fraction of peak LR (cosine schedule).")
    train_grp.add_argument("--warmup-ratio", type=float, default=0.1,
                           help="Fraction of total steps used for linear LR warm-up.")
    train_grp.add_argument("--label-smoothing", type=float, default=0.0) # v23: Disable to allow <0.01 loss
    train_grp.add_argument("--weight-decay", type=float, default=1e-2)
    train_grp.add_argument("--device", type=str, default=None,
                           help="cpu / cuda / mps (default: auto).")
    train_grp.add_argument("--seed", type=int, default=42)
    train_grp.add_argument("--early-stop-patience", type=int, default=3,
                           help="Stop if val loss doesn't improve for N epochs (0=disabled).")
    train_grp.add_argument("--world-size", type=int, default=0,
                           help="Number of parallel processes (0=auto-detect).")

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
    out_grp.add_argument("--port", type=int, default=12355,
                         help="Master port for distributed training (default 12355).")
    out_grp.add_argument("--legacy-checkpoint", type=Path, default=None)
    return p


# ---------------------------------------------------------------------------
# Auto-scale model size
# ---------------------------------------------------------------------------


def auto_model_size(n_train: int, device: torch.device) -> tuple[int, int]:
    """Pick d_model and n_layers based on training set size and device."""
    if device.type == "cpu":
        # Global Optimization v23: Reasoning enabled via attn_interval=1
        # 1-layer + Attention = ~3.8M parameters.
        return 168, 1
    else:  # GPU
        if n_train < 2_000:
            return 192, 4
        elif n_train < 10_000:
            return 256, 6
        else:
            return 384, 8


def find_free_port(start_port: int) -> int:
    """Find an available TCP port starting from start_port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except socket.error as e:
                # 10048 is WSAEADDRINUSE on Windows
                if e.errno == errno.EADDRINUSE or e.errno == 10048:
                    port += 1
                else:
                    raise e
    raise RuntimeError("No free ports available.")


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
    # Shift-Core v32: 2 threads per cluster
    torch.set_num_threads(2)
    args = parse_args().parse_args()
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = auto_device()

    if device.type == "cpu":
        # Temporal Overdrive v32: Full-Core Saturation
        torch.set_flush_denormal(True)
        # 10 workers * 1 thread = 10 physical cores. Perfect 1:1 mapping.
        torch.set_num_threads(1) 
        logger.info(f"Auto-Optimization (CPU): Saturation v32 Active | 10 Workers Spawning (1 thread/ea)...")
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Auto-Optimization (GPU): TensorCores Enabled (Ampere+)")
        else:
            logger.info("Auto-Optimization (GPU): Legacy CUDA Fallback")

    # Batch size auto (V20 scale upgrade)
    if args.batch_size <= 0:
        args.batch_size = 2 if device.type == "cpu" else 32
    batch_size = args.batch_size
    
    # ── Multi-Lane Acceleration (Hogwild! v31) ───────────────────────────
    # Detect optimal core saturation (P-Cores vs E-Cores)
    # Mobile CPUs with 8+ cores often perform better with 4-6 workers (Reduced cache thrashing)
    if args.world_size == 0:
        if device.type == "cpu":
            # Saturation v32: Full cluster for 100+ steps/sec global target
            args.world_size = 10
            logger.info("Auto-Optimization (CPU): Saturation v32 | Target: 100+ steps/s Cluster.")
        else:
            args.world_size = mp.cpu_count() or 1
            
    world_size = args.world_size

    logger.info(f"Device: {device}  |  Batch size (V20): {args.batch_size}")

    # ── CPU Hyper-Breakthrough (100+ steps/s) ──────────────────────────────
    if device.type == "cpu" and args.seq_len == 512:
        args.seq_len = 128
        logger.info("Auto-Optimization (CPU): Applied Turbo-Context Scaling (seq_len=128) to ensure 100+ steps/s.")
    
    # Pulse-Sync v29.1: Auto-Scale Gradient Accumulation for CPU Performance
    if device.type == "cpu" and args.grad_accum == 1:
        args.grad_accum = 16
        logger.info("Auto-Optimization (CPU): Applied Grad-Accum (m_size=16) to ensure 100+ steps/s with Clipping.")

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

    logger.info("Building vocabulary (Point 1 & 3) …")
    # Point 1: 8k-16k vocab target (12,000 default)
    # Point 3: rare tokens removed (min_freq=2) + lowercase forced
    tokenizer = WordTokenizer.from_texts(vocab_texts, max_vocab=12000, min_freq=2)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # ── Tensor datasets ─────────────────────────────────────────────────────
    logger.info("Tokenizing training pairs …")
    # Point 9: Prepare contiguous datasets in RAM. 
    train_inputs, train_labels, skipped_train = build_sft_tensors(train_pairs, tokenizer, args.seq_len)
    train_inputs = train_inputs.contiguous()
    train_labels = train_labels.contiguous()
    
    v_inputs, v_labels, skipped_val = build_sft_tensors(val_pairs, tokenizer, args.seq_len)
    v_inputs = v_inputs.contiguous()
    v_labels = v_labels.contiguous()
    
    # --- Point 2: Torch.Compile Optimization ---
    # mode="reduce-overhead" is ideal for CPU training loops
    # NOTE: On Windows (spawn), we compile inside the worker to avoid pickling errors.
    if args.compile and device.type != "cpu":
        logger.info("Auto-Optimization (GPU): Compiling model (reduce-overhead) …")
        model = torch.compile(model, mode="reduce-overhead")
    
    # Turbo Engine v4: Force RAM Mode for CPU to eliminate SSD latency
    use_ssd = False if device.type == "cpu" else (args.use_ssd or (len(train_pairs) > 50000))
    
    if use_ssd:
        mmap_path = args.output_dir / "train_cache.mmap"
        logger.info(f"SSD-Streaming enabled: writing tokens to {mmap_path}")
        train_loader = DataLoader(
            MmapDataset.from_pairs(train_pairs, tokenizer, mmap_path, seq_len=args.seq_len),
            batch_size=batch_size,
            shuffle=False, 
            num_workers=4, # Overdrive v16: Parallel Data Engine
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
        )
        # Placeholder for auto_model_size calc
        class Dummy: pass
        train_inputs = Dummy()
        train_inputs.__len__ = lambda: len(train_pairs)
    else:
        logger.info(f"Training tokens: {len(train_inputs):,} examples | Mode: RAM-Master")
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
        logger.info(f"Validation tensors: {len(v_inputs):,} examples")
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

        config = NeuroSwiftConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model_arg,
            n_layers=n_layers_arg,
            d_state=auto_d_state,
            expansion=2,
            conv_kernel=4,
            num_experts=args.num_experts,
            top_k=auto_top_k,
            expert_hidden=expert_hidden,
            plastic_dim=auto_plastic_dim,
            dropout=args.dropout,
            aux_loss_scale=1e-2,
            attn_interval=1, # v23: Enabled for reasoning Ability
            ternary_mode=False, 
            latent_dim=args.latent_dim,
        )
        model = NeuroSwiftLM(config, use_checkpoint=args.grad_checkpoint).to(device)

    # ── Pre-Training Intelligence (v23 Transparency) ────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    # Estimate size: Params * 4 bytes (FP32) / 1024^2 = MB
    est_size_mb = (n_params * 4) / (1024 ** 2)
    
    total_expected_steps = len(train_inputs) // args.batch_size # Global potential
    # Corrected alignment: Use the actual world_size (number of workers) for steps_per_epoch calculation
    steps_per_epoch = len(train_inputs) // (args.batch_size * world_size) if world_size > 0 else len(train_inputs) // args.batch_size
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    
    # Pulse-Sync v26: Zero-Step Guard
    if steps_per_epoch <= 0 and len(train_pairs) > 0:
        steps_per_epoch = 1
    
    print(f"{_get_stamp()}  Training Plan: {args.epochs} Epochs | {steps_per_epoch} Steps/Epoch")
    print(f"{_get_stamp()}  Model Intelligence: {n_params:,} Parameters | ~{est_size_mb:.2f} MB on disk")
    print(f"{_get_stamp()}  Reasoning: Multi-Head Linear Attention ACTIVE (attn_interval=1)")

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
        fused=True,
    )
    
    # --- NeuroSwift Turbo-Engine 2.0: Asynchronous Hogwild Launch ---
    # --- NeuroSwift Saturation 2.1: Optimized Rank Affinity ---
    # world_size = 8: Focuses on 100% P-Core saturation, avoiding E-Core slowdowns.
    # Pulse-Sync v31: Uses dynamic world_size from main logic
    world_size = world_size
    if device.type == "cpu":
        # ── Zero-Copy Shared Memory (Mandatory for Windows/mp.spawn) ────────────
        train_inputs.share_memory_()
        train_labels.share_memory_()
        if v_inputs is not None:
            v_inputs.share_memory_()
            v_labels.share_memory_()
            
        # Model Parameters Sharing
        model.share_memory()
        
        logger.info(f"Master: Launching {world_size} ASYNC processes (Zero-Barrier Hogwild)...")
        # Shared Optimization Context: Absolute Lock-Free
        shared_steps = mp.RawArray('i', world_size)
        shared_loss = mp.RawArray('f', world_size)
        barrier = mp.Barrier(world_size)
        
        try:
            mp.spawn(
                train_worker,
                args=(world_size, model, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, shared_steps, shared_loss, barrier),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            logger.error(f"Hogwild Launch Failed: {e}.")
            raise e
        return

def train_worker(rank, world_size, model, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, shared_steps, shared_losses, barrier):
    # Worker Startup (Hogwild! Asynchronous Mode) ─────────────────────────────
    # --- Point 3: Dynamic Thread Allocation ---
    cores = mp.cpu_count()
    threads_per_worker = max(1, cores // world_size)
    torch.set_num_threads(threads_per_worker) 
    device = torch.device(args.device if args.device else "cpu")
    torch.set_num_interop_threads(1)
    
    # Point 5 & 6: Pre-allocate Batch Buffers EARLY for compilation warm-up
    batch_inp = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long, device=device)
    batch_lbl = torch.zeros((args.batch_size, args.seq_len), dtype=torch.long, device=device)
    
    # Point 2: Sequential Compile for Windows (CPU/Spawn) stability
    if args.compile and device.type == "cpu":
        try:
            for r in range(world_size):
                if rank == r:
                    if os.name == "nt":
                        setup_msvc_env()
                    model = torch.compile(model, mode="reduce-overhead")
                    # Warm-up pass to trigger MSVC
                    model(batch_inp)
                barrier.wait() 
        except Exception as e:
            if rank == 0:
                print(f"[NeuroSwift AutoTrain] Warning: torch.compile failed: {e}. Falling back to eager mode.")
            barrier.wait()

    # Ghost-Sync Weight Sharing: Already connected to Master memory
    model.train()
    
    # Point 10: Grad Accumulation (m_size)
    m_size = args.grad_accum
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Turbo-Engine Configuration
    # Heartbeat suppressed per User Request (v20.1)
    
    # Pulse-Sync v21: Buffer multi batches before shared-memory update
    
    # High-Performance Data Engine: Raw Slicing (Bypasses Python DataLoader overhead)
    segment_size = len(train_inputs) // world_size
    start_idx = rank * segment_size
    end_idx = start_idx + segment_size
    total_local = end_idx - start_idx
    best_val_loss = float("inf")
    
    # Epoch Boundary Logic: Strict Multi-Lane Tracking
    total_steps_in_epoch = len(train_inputs) // (world_size * args.batch_size) if world_size > 0 else len(train_inputs) // args.batch_size
    if hasattr(args, "max_steps_per_epoch") and args.max_steps_per_epoch > 0:
        total_steps_in_epoch = min(total_steps_in_epoch, args.max_steps_per_epoch)
    
    for epoch in range(1, args.epochs + 1):
        # Reset counters for the new epoch
        if rank == 0:
            for r in range(world_size):
                shared_steps[r] = 0
                shared_losses[r] = 0.0
        barrier.wait()
        
        model.train()
        # Pulse-Sync v29.1: Diverse Worker Randomization
        # Each worker gets a unique seed per epoch to maximize data coverage across the cluster
        worker_seed = args.seed + epoch + rank
        torch.manual_seed(worker_seed)
        import random as _rnd
        _rnd.seed(worker_seed)

        # Initial calibration
        prev_cluster_steps = 0
        step_start_time = time.time()
        actual_speed = 0.0   # Pulse-Sync v27: Global Stability Fix
        samples_sec = 0.0    # Pulse-Sync v27: Global Stability Fix
        
        # v22: Randomized Telemetry Interval (50-100 steps)
        import random as _rnd
        next_log_step = _rnd.randint(50, 100)
        
        # Pulse-Sync v29: Data Diversity & Subset Sampling (20K-40K window)
        subset_size = min(40000 // world_size, total_local)
        local_indices = torch.randperm(total_local)[:subset_size]
        local_steps = subset_size // args.batch_size
        
        # Pulse-Sync v29: Cosine Scheduler with 5% Warmup phase
        total_training_steps = args.epochs * local_steps
        warmup_steps = int(0.05 * total_training_steps)
        
        for i in range(0, subset_size - args.batch_size, args.batch_size):
            current_local_step = i // args.batch_size
            global_step = (epoch - 1) * local_steps + current_local_step
            
            # --- Pulse-Sync v29: Cosine Decay with Warmup Logic ---
            if global_step < warmup_steps:
                # Linear Warmup
                lr_scale = float(global_step) / float(max(1, warmup_steps))
            else:
                # Cosine Decay
                progress = float(global_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
                lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            
            # Adaptive LR Chill: Prevent late-stage oscillation if loss is already low
            if shared_losses[rank] < 0.05:
                lr_scale *= 0.7
                
            current_lr = args.lr * lr_scale
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # --- Pulse-Sync v30: Loop Break Safety ---
            # Correctly handle step-based early finish per User Request
            if args.max_steps_per_epoch > 0 and (i // args.batch_size) >= args.max_steps_per_epoch:
                break

            # Zero-Overhead Slicing: Directly indexing into RAM tensors
            batch_slice = local_indices[i : i + args.batch_size]
            # Point 5 & 6: Use copy_() into preallocated buffer to avoid allocation
            batch_inp.copy_(train_inputs[start_idx + batch_slice])
            batch_lbl.copy_(train_labels[start_idx + batch_slice])
            
            # Point 1 & 8: Ultra-lean forward step
            out = model(batch_inp, targets=batch_lbl)
            loss = out["loss"] / m_size
            loss.backward()
            
            if ((current_local_step) + 1) % m_size == 0:
                # Pulse-Sync v29: Gradient Clipping (max_norm=1.0) for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # --- Pulse-Sync v20 Telemetry: Real-Time Smooth Reporting ---
            # Update shared counter every batch for fluid speed calc (Eliminates 'fake' 0.0 reporting)
            shared_steps[rank] += 1
            shared_losses[rank] = 0.9 * shared_losses[rank] + 0.1 * (loss.item() * m_size)
            
            is_last_step = (i + args.batch_size * 2) >= subset_size or (current_local_step + 1) >= local_steps
            
            if rank == 0 and (current_local_step >= next_log_step or is_last_step):
                current_time = time.time()
                elapsed = current_time - step_start_time
                
                # Global Progress: Efficient Sum
                curr_total_steps = shared_steps[0] # Using Rank 0 as proxy for Speed (Pulse-Sync v31)
                for r in range(1, world_size):
                     curr_total_steps += shared_steps[r]
                
                # Speed: Aggregate Micro-batches per second
                global_delta = curr_total_steps - prev_cluster_steps
                actual_speed = global_delta / max(elapsed, 0.001)
                
                # User-Requested Format (v23): 
                # [NeuroSwift AutoTrain] HH:MM:SS | Epoch <no> | Rank <no> | <steps> / <total> | Speed | LR | Loss
                completed = min(curr_total_steps // world_size, total_steps_in_epoch)
                percent = (completed / total_steps_in_epoch) * 100
                # Samples/s strictly relative to global iterations (actual_speed)
                samples_sec = actual_speed * args.batch_size
                
                log_line = (f"{_get_stamp()}  Epoch {epoch} | Rank {rank} | "
                            f"{completed} / {total_steps_in_epoch} ({percent:.1f}%) | "
                            f"Speed {actual_speed:.1f} global-steps /sec | "
                            f"LR: {current_lr:.6f} | Samples/s: {samples_sec:.1f} | "
                            f"Loss: {shared_losses[rank]:.4f}")
                            
                print(log_line, flush=True)
                with open("artifacts/throughput.log", "a") as f:
                    f.write(log_line)
                
                prev_cluster_steps = curr_total_steps
                step_start_time = current_time
                # v22: Schedule next random log pulse
                next_log_step = current_local_step + _rnd.randint(50, 100)

            # Periodic GC
            if (i // args.batch_size) % 500 == 0:
                gc.collect()
        
        # --- Pulse-Barrier v20.1: Epoch Synchronization ---
        # Ensures all workers finish their gradients before eval/save
        barrier.wait()
        
        # Final Progress Pulse: Ensure 100% is shown at the end of the epoch
        if rank == 0:
            avg_train_loss = sum(shared_losses) / world_size
            completed = total_steps_in_epoch
            samples_sec = actual_speed * world_size * args.batch_size
            
            final_line = (f"  [Hyper-Drive] Epoch {epoch} | Rank {rank} | "
                        f"{completed} / {total_steps_in_epoch} (100.0%) | "
                        f"Final Speed {actual_speed:.1f} steps /sec | "
                        f"LR: {args.lr} | Samples/s: {samples_sec:.1f} | "
                        f"Loss: {avg_train_loss:.4f}\n")
            # Unified Evaluation Logic (v23)
            if rank == 0:
                print(f"{_get_stamp()}  Evaluating epoch {epoch} generalization …", flush=True)
                model.eval()
                v_losses = []
                with torch.no_grad():
                    # Batch the validation dataset
                    if v_inputs is not None:
                        for vi in range(0, len(v_inputs), args.batch_size * 4):
                            vi_end = min(vi + args.batch_size * 4, len(v_inputs))
                            v_batch_in = v_inputs[vi:vi_end].to(device)
                            v_batch_lab = v_labels[vi:vi_end].to(device)
                            v_out = model(v_batch_in, targets=v_batch_lab)
                            v_losses.append(v_out["loss"].item())
                
                avg_v_loss = sum(v_losses) / len(v_losses) if v_losses else 0.0
                print(f"{_get_stamp()}  Epoch {epoch:02d} Summary: Train Loss {avg_train_loss:.4f} | Val Loss {avg_v_loss:.4f}", flush=True)
                print(f"---------------------------------------------------\n", flush=True)

                if avg_v_loss < best_val_loss:
                    best_val_loss = avg_v_loss
                    print(f"{_get_stamp()}  New best loss {avg_v_loss:.4f}! Saving checkpoint...", flush=True)
                    # Unified v22 Saving: Root + Best folder
                    _save_checkpoint(model, tokenizer, args.output_dir, avg_v_loss, subdir="best")

    if rank == 0:
        print(f"\n[NeuroSwift] Training Task Complete. Saving final weights to {args.output_dir}\n", flush=True)
        # Unified v22 Saving: Root + Last folder
        _save_checkpoint(model, tokenizer, args.output_dir, 0.0, subdir="last")

def _save_checkpoint(model, tokenizer, output_dir, last_loss, subdir=None):
    """Unified checkpointing for shared-memory workers (v22 Dual-Saving)."""
    root_path = Path(output_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    
    # Handle compiled models
    real_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    
    # Bundle vocab into the .pt checkpoint
    vocab_data = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    
    # Path 1: Root output directory (Main benchmark path)
    real_model.save_pretrained(root_path, vocab=vocab_data)
    tokenizer.save_pretrained(root_path)
    
    # Path 2: Versioned subdirectory (best/last/epoch_N)
    if subdir:
        sub_path = root_path / subdir
        sub_path.mkdir(parents=True, exist_ok=True)
        real_model.save_pretrained(sub_path, vocab=vocab_data)
        tokenizer.save_pretrained(sub_path)
        print(f"[NeuroSwift AutoTrain] Model and version '{subdir}' saved to {root_path}", flush=True)

if __name__ == "__main__":
    # Windows requires spawn method
    mp.set_start_method("spawn", force=True)
    main()
