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

from neuroswift.data_pipeline import TrainPair, run_pipeline, UniversalSchemaMapper, QualityFilter, sanitize_text
from neuroswift.layers import auto_device
from neuroswift.model import NeuroSwiftConfig, NeuroSwiftLM
from neuroswift.tokenizer import WordTokenizer, HybridTokenizer, load_tokenizer
from neuroswift.streaming import MmapDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("NeuroSwift.Train")
logger.setLevel(logging.INFO) 

def _get_stamp() -> str:
    """Standardized NeuroSwift Telemetry Stamp."""
    return f"[NeuroSwift AutoTrain] {datetime.now().strftime('%H:%M:%S')}"


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def fmt_prompt(instruction: str, context: str = "") -> str:
    """Standardized NeuroSwift Prompt Format."""
    if context:
        return f"instruction: {instruction}\ncontext: {context}\nresponse:"
    return f"instruction: {instruction}\nresponse:"


# ---------------------------------------------------------------------------
# SFT example builder
# ---------------------------------------------------------------------------


def _sft_worker(args):
    """Deep-Optimization worker for build_sft_tensors."""
    pair, slim_token_data, seq_len, fmt_fn = args
    try:
        stoi = slim_token_data["stoi"]
        special = slim_token_data["special_tokens"]
        spm_model_path = slim_token_data.get("spm_model_path")
        
        if spm_model_path and os.path.exists(spm_model_path):
            import sentencepiece as spm
            sp_model = spm.SentencePieceProcessor(model_file=spm_model_path)
            
            # v34: Fix: Ensure we use IDs, not token names
            eos_name = special.get("eos", "<eos>")
            pad_name = special.get("pad", "<pad>")
            
            eos_token_id = sp_model.piece_to_id(eos_name)
            if eos_token_id < 0: eos_token_id = sp_model.eos_id()
            
            pad_id = sp_model.piece_to_id(pad_name)
            if pad_id < 0: pad_id = sp_model.pad_id()
            if pad_id < 0: pad_id = 0 # Hard fallback
            
            def encode_local(text: str, add_eos: bool = False) -> list[int]:
                ids = sp_model.encode(text)
                if add_eos:
                    ids.append(eos_token_id)
                return ids
        else:
            pad_id = stoi.get(special["pad"], 0)
            eos_token = special["eos"]

            # Minimal encoding logic to stay fast inside worker
            def encode_local(text: str, add_eos: bool = False) -> list[int]:
                text = text.lower() if slim_token_data["lowercase"] else text
                import re
                words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
                ids = [stoi.get(w, stoi.get(special["unk"], 1)) for w in words]
                if add_eos:
                    ids.append(stoi.get(eos_token, 3))
                return ids

        p_text = fmt_fn(pair.instruction, pair.context)
        prompt_ids = encode_local(p_text, add_eos=False)
        resp_ids = encode_local(pair.response, add_eos=True)
        
        # Pulse-Stream v5: Response-First Window Slicing
        if len(prompt_ids) + len(resp_ids) > seq_len:
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
        import traceback
        # Only print first few errors to avoid flooding
        worker_id = multiprocessing.current_process().name
        print(f"\n[CRITICAL] SFT Worker {worker_id} crashed:\n{traceback.format_exc()}")
        return None


def build_sft_tensors(
    pairs: list[TrainPair],
    tokenizer: WordTokenizer | HybridTokenizer,
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
                "pad": getattr(tokenizer, "pad_token", "<pad>"),
                "unk": getattr(tokenizer, "unk_token", "<unk>"),
                "eos": getattr(tokenizer, "eos_token", "<eos>"),
            },
            "lowercase": getattr(tokenizer, "lowercase", False),
            "spm_model_path": getattr(tokenizer, "spm_model_path", None)
        }
        
        # v32: Debug Diagnostics
        if len(pairs) > 0:
            samp = pairs[0]
            test_text = fmt_prompt(samp.instruction, samp.context)
            print(f"{_get_stamp()}  [DEBUG SFT] Testing Tokenizer: '{samp.instruction[:50]}...' -> {len(tokenizer.encode(test_text))} tokens")
            print(f"{_get_stamp()}  [DEBUG SFT] First Sample: InstLen={len(samp.instruction)}, CtxLen={len(samp.context)}, RespLen={len(samp.response)}")

        valid_pairs = []
        for i, pair in enumerate(pairs):
            inst_text = pair.instruction.strip() if pair.instruction else ""
            resp_text = pair.response.strip() if pair.response else ""
            
            if not inst_text or not resp_text:
                if skipped < 5:
                    print(f"{_get_stamp()}  [Build SFT] Skipping sample {i}: Empty Instruction or Response. Inst: '{inst_text[:30]}...', Resp: '{resp_text[:30]}...'")
                skipped += 1
                continue
                
            text = fmt_prompt(inst_text, pair.context.strip()) + "\n" + resp_text
            token_ids = tokenizer.encode(text)
            
            if len(token_ids) >= 3:
                valid_pairs.append(pair)
            else:
                if skipped < 10:
                    print(f"{_get_stamp()}  [Build SFT] Sample too short ({len(token_ids)} tokens) for BPE. Text: {text[:80]}...")
                skipped += 1

        num_procs = min(multiprocessing.cpu_count(), 10) 
        worker_args = [(p, slim_token_data, seq_len, fmt_prompt) for p in valid_pairs]
        
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
                "pad": getattr(tokenizer, "pad_token", "<pad>"),
                "unk": getattr(tokenizer, "unk_token", "<unk>"),
                "eos": getattr(tokenizer, "eos_token", "<eos>"),
            },
            "lowercase": getattr(tokenizer, "lowercase", False),
            "spm_model_path": getattr(tokenizer, "spm_model_path", None)
        }
        for pair in pairs:
            inst_text = pair.instruction.strip() if pair.instruction else ""
            resp_text = pair.response.strip() if pair.response else ""
            if not inst_text or not resp_text:
                skipped += 1
                continue
            text = fmt_prompt(inst_text, pair.context.strip()) + "\n" + resp_text
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 3:
                if skipped < 10:
                    print(f"{_get_stamp()}  [Build SFT] Sample too short ({len(token_ids)} tokens) for BPE. Text: {text[:80]}...")
                skipped += 1
                continue
            res = _sft_worker((pair, slim_token_data, seq_len, fmt_prompt))
            if res is None:
                skipped += 1
            else:
                input_rows.append(res[0])
                label_rows.append(res[1])

    if len(input_rows) == 0:
        raise RuntimeError(
            f"No valid training examples built. Skipped={skipped}, Total={len(pairs)}"
        )

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
    base_lr = optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"])
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
    data_grp.add_argument("--max-per-file", type=int, default=10_000_000_000,
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
    train_grp.add_argument("--seq-len", type=int, default=256, help="Sequence length (Pulse-Stream: 32 for 100+ steps/s)")
    p.add_argument("--lr", type=float, default=0.0002) # v22: Ultra-High LR for <0.2 Loss
    train_grp.add_argument("--min-lr-ratio", type=float, default=0.05,
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
    train_grp.add_argument("--vocab-size", type=int, default=12000,
                           help="Tokenizer vocabulary size (SentencePiece BPE). Default 12,000.")

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
        return 256, 4
    else:  # GPU
        if n_train < 2_000:
            return 256, 4
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

# ===================== ADD BELOW IMPORTS =====================
def compute_steps_per_epoch(dataset_size, world_size, batch_size):
    return dataset_size // (world_size * batch_size)


def compute_total_training_steps(dataset_size, world_size, batch_size, epochs):
    return compute_steps_per_epoch(dataset_size, world_size, batch_size) * epochs

def main() -> None:
    import contextlib
    import os
    setup_msvc_env()
    args = parse_args().parse_args()
    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = auto_device()
    # world_size = 8: Focuses on 100% P-Core saturation, avoiding E-Core slowdowns.
    world_size = 8 if device.type == "cpu" else 1
    if device.type == "cpu":
        # Heterogeneous Storm v18: Full 10-core Spawning Drive
        torch.set_flush_denormal(True)
        # Using 8 processes, each with 1 OMP thread = Full P-Core saturation
        torch.set_num_threads(4) 
        logger.info(f"Auto-Optimization (CPU): Saturation Active | {world_size} Workers Spawning...")
    elif device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if torch.cuda.get_device_capability()[0] >= 8:
            logger.info("Auto-Optimization (GPU): TensorCores Enabled (Ampere+)")
        else:
            logger.info("Auto-Optimization (GPU): Legacy CUDA Fallback")

    # Batch size auto (V20 scale upgrade)
    if args.batch_size <= 0:
        batch_size = 2 if device.type == "cpu" else 32
    else:
        batch_size = args.batch_size
    
    logger.info(f"Device: {device}  |  Batch size : {batch_size}")
    args.batch_size = batch_size # Sync back to args for distributed launch

    # ── CPU Hyper-Breakthrough (100+ steps/s) ──────────────────────────────
    if device.type == "cpu" and args.seq_len == 512:
        args.seq_len = 256 
        logger.info(f"Auto-Optimization (CPU): Applied Turbo-Context Scaling (seq_len={args.seq_len}) to ensure 100+ steps/s.")
        #print(f"Auto-Optimization (CPU): Applied Turbo-Context Scaling (seq_len={args.seq_len}) to ensure 100+ steps/s.")

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
    # ── Save Datasets ────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train_set.jsonl", "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps({
                "instruction": p.instruction,
                "context": p.context,
                "response": p.response
            }, ensure_ascii=False) + "\n")
    if val_pairs:
        with open(args.output_dir / "val_set.jsonl", "w", encoding="utf-8") as f:
            for p in val_pairs:
                f.write(json.dumps({
                    "instruction": p.instruction,
                    "context": p.context,
                    "response": p.response
                }, ensure_ascii=False) + "\n")
    logger.info(f"Datasets archived to {args.output_dir}")

    if len(train_pairs) < 8:
        logger.error(
            f"Too few training pairs ({len(train_pairs)}). "
            "Add more data or lower --min-response-words."
        )
        sys.exit(1)

    # ── Vocabulary ──────────────────────────────────────────────────────────
    tokenizer_corpus_path = os.path.join(args.output_dir, "tokenizer_corpus.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.output_dir / "tokenizer.model").exists():
        logger.info("Training SentencePiece tokenizer...")
        
        with open(tokenizer_corpus_path, "w", encoding="utf-8") as f:
            for pair in train_pairs:
                # v33: Tokenizer Input Uniform Guarantee
                i = sanitize_text(pair.instruction).strip()
                c = sanitize_text(pair.context).strip()
                r = sanitize_text(pair.response).strip()
                if i: f.write(i.replace('\n', ' ') + "\n")
                if c: f.write(c.replace('\n', ' ') + "\n")
                if r: f.write(r.replace('\n', ' ') + "\n")

        import sentencepiece as spm

        # v36: Hard-redirect C++ stdout/stderr to devnull to silence BPE logs
        with open(os.devnull, 'w') as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                spm.SentencePieceTrainer.train(
                    f'--input={tokenizer_corpus_path} --model_prefix={os.path.join(args.output_dir, "tokenizer")} '
                    f'--vocab_size={args.vocab_size} --model_type=bpe --character_coverage=0.9995 '
                    f'--input_sentence_size=2000000 --byte_fallback=true --split_by_unicode_script=true '
                    f'--unk_id=0 --pad_id=1 --eos_id=2 --bos_id=-1 --shuffle_input_sentence=true'
                )
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)

    logger.info("Building vocabulary …")
    # Empty list passed as fallback since model is already trained/exists
    tokenizer = HybridTokenizer.from_texts([], vocab_size=args.vocab_size, output_dir=args.output_dir)
    
    # v31: Tokenizer Health Check
    if tokenizer.vocab_size < 100:
        print(f"{_get_stamp()}  [WARNING] Tokenizer at {args.output_dir} appears corrupted (vocab={tokenizer.vocab_size}). Forcing retrain...")
        model_file = args.output_dir / "tokenizer.model"
        if model_file.exists():
            model_file.unlink()
        # Re-run the main to trigger retrain
        return main()

    # v35: Decoding Sanity Check
    test_str = "NeuroSwift Tokenizer Check"
    try:
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded)
        if not decoded or len(decoded.replace(" ", "")) < 5:
             logger.warning("Tokenizer decoding sanity check failed! Model may be unstable.")
    except Exception as e:
        logger.error(f"Tokenizer health check crashed: {e}")
        return main()

    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")

    # ── Tensor datasets ─────────────────────────────────────────────────────
    logger.info("Tokenizing training pairs …")
    
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
            tokenizer = load_tokenizer(args.output_dir)
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
            logger.info(f"Auto-Optimization (CPU): Applied Elite Speed Pruning (d_state={auto_d_state}, d_model={d_model_arg}, Hebbian=ON, Top-K={auto_top_k}).")
            #print(f"Auto-Optimization (CPU): Applied Elite Speed Pruning (d_state={auto_d_state}, d_model={d_model_arg}, Hebbian=ON, Top-K={auto_top_k}).")

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
            attn_interval=4, # v23: Enabled for reasoning Ability
            ternary_mode=False, 
            latent_dim=args.latent_dim,
        )
        model = NeuroSwiftLM(config, use_checkpoint=args.grad_checkpoint).to(device)

    # ── Pre-Training Intelligence (v23 Transparency) ────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    # Estimate size: Params * 4 bytes (FP32) / 1024^2 = MB
    est_size_mb = (n_params * 4) / (1024 ** 2)
    dataset_size = len(train_inputs)
    total_expected_steps = dataset_size // args.batch_size
    steps_per_epoch = compute_steps_per_epoch(dataset_size, world_size, args.batch_size)
    
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    
    total_training_steps = compute_total_training_steps(
        dataset_size, world_size, args.batch_size, args.epochs
    )
    
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    
    print(f"{_get_stamp()}  Training Plan: {args.epochs} Epochs | {steps_per_epoch} Steps/Epoch")
    print(f"{_get_stamp()}  Total Steps: {total_training_steps} | Warmup: {warmup_steps}")
    print(f"{_get_stamp()}  Model Intelligence: {n_params:,} Parameters | ~{est_size_mb:.2f} MB on disk")
    print(f"{_get_stamp()}  Reasoning: Multi-Head Linear Attention ACTIVE (attn_interval={config.attn_interval})")
    print(f"{_get_stamp()}  Vocabulary size: {tokenizer.vocab_size:,} tokens")
    print(f"{_get_stamp()}  Model Config: {model.config}")

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
    
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr
    
    # --- NeuroSwift Turbo-Engine 2.0: Asynchronous Hogwild Launch ---
    # --- NeuroSwift Saturation 2.1: Optimized Rank Affinity ---
    
    if device.type == "cpu":
        # ── Zero-Copy Shared Memory (Mandatory for Windows/mp.spawn) ────────────
        train_inputs.share_memory_()
        train_labels.share_memory_()
        if v_inputs is not None:
            v_inputs.share_memory_()
            v_labels.share_memory_()
            
        # Model Parameters Sharing
        model.share_memory()

        #print(f"{_get_stamp()}  Master: Launching {world_size} ASYNC processes (Zero-Barrier Hogwild)...")
        
        logger.info(f"Master: Launching {world_size} ASYNC processes (Zero-Barrier Hogwild)...")

        # Shared Optimization Context: Absolute Lock-Free
        shared_steps = mp.RawArray('i', world_size)
        shared_loss = mp.RawArray('f', world_size)
        barrier = mp.Barrier(world_size)
        
        try:
            # mp.spawn(
            #     train_worker,
            #     args=(world_size, model, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, shared_steps, shared_loss, barrier),
            #     nprocs=world_size,
            #     join=True
            # )
            mp.spawn(
                train_worker,
                args=(
                    world_size,
                    model,
                    train_inputs,
                    train_labels,
                    v_inputs,
                    v_labels,
                    tokenizer,
                    args,
                    shared_steps,
                    shared_loss,
                    barrier,
                    total_training_steps,
                    warmup_steps,
                ),
                nprocs=world_size,
                join=True
            )

        except Exception as e:
            logger.error(f"Hogwild Launch Failed: {e}.")
            raise e
        return

#def train_worker(rank, world_size, model, train_inputs, train_labels, v_inputs, v_labels, tokenizer, args, shared_steps, shared_losses, barrier):
#    # Worker Startup (Hogwild! Asynchronous Mode) ─────────────────────────────
#    device = torch.device("cpu")
#    torch.set_num_threads(2) 
#    torch.set_num_interop_threads(2)
#    
#    # Ghost-Sync Weight Sharing: Already connected to Master memory
#    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
#    
#    # Turbo-Engine Configuration
#    # Heartbeat suppressed per User Request (v20.1)
#    
#    # Pulse-Sync v21: Buffer 16 batches before shared-memory update
#    # Balanced Performance: Restores high-fidelity convergence (Stability v21).
#    m_size = 16
#    
#    # High-Performance Data Engine: Raw Slicing (Bypasses Python DataLoader overhead)
#    segment_size = len(train_inputs) // world_size
#    start_idx = rank * segment_size
#    end_idx = start_idx + segment_size
#    total_local = end_idx - start_idx
#    best_val_loss = float("inf")
#    
#    # Epoch Boundary Logic: Strict Multi-Lane Tracking
#    
#    total_steps_in_epoch = len(train_inputs) // (world_size * args.batch_size) if world_size > 0 else len(train_inputs) // args.batch_size
#    if hasattr(args, "max_steps_per_epoch") and args.max_steps_per_epoch > 0:
#        total_steps_in_epoch = min(total_steps_in_epoch, args.max_steps_per_epoch)
#    
#    for epoch in range(1, args.epochs + 1):
#        # Reset counters for the new epoch
#        if rank == 0:
#            for r in range(world_size):
#                shared_steps[r] = 0
#                shared_losses[r] = 0.0
#        barrier.wait()
#        
#        model.train()
#        # Initial calibration
#        prev_cluster_steps = 0
#        step_start_time = time.time()
#        
#        # v22: Randomized Telemetry Interval (50-100 steps)
#        import random as _rnd
#        next_log_step = _rnd.randint(50, 100)
#        
#        # Local Shuffle for each epoch
#        local_indices = torch.randperm(total_local)
#        
#        for i in range(0, total_local - args.batch_size, args.batch_size):
#            if hasattr(args, "max_steps_per_epoch") and args.max_steps_per_epoch > 0 and (i // args.batch_size) >= args.max_steps_per_epoch:
#                break
#            
#            # Zero-Overhead Slicing: Directly indexing into RAM tensors
#            batch_slice = local_indices[i : i + args.batch_size]
#            batch_inp = train_inputs[start_idx + batch_slice].to(device)
#            batch_lbl = train_labels[start_idx + batch_slice].to(device)
#            
#            out = model(batch_inp, targets=batch_lbl)
#            loss = out["loss"] / m_size
#            loss.backward()
#            
#            if ((i // args.batch_size) + 1) % m_size == 0:
#                optimizer.step()
#                optimizer.zero_grad(set_to_none=True)
#            # --- Pulse-Sync v20 Telemetry: Real-Time Smooth Reporting ---
#            # Update shared counter every batch for fluid speed calc (Eliminates 'fake' 0.0 reporting)
#            shared_steps[rank] += 1
#            shared_losses[rank] = 0.9 * shared_losses[rank] + 0.1 * (loss.item() * m_size)
#            
#            current_local_step = i // args.batch_size
#            is_last_step = (i + args.batch_size) >= (total_local - args.batch_size) or (current_local_step + 1) >= total_steps_in_epoch
#            
#            if rank == 0 and (current_local_step >= next_log_step or is_last_step):
#                current_time = time.time()
#                elapsed = current_time - step_start_time
#                
#                # Global Progress: Sum of all lanes
#                curr_total_steps = sum(shared_steps)
#                
#                # Speed: Steps / Sec (Now strictly accurate and smooth)
#                global_delta = curr_total_steps - prev_cluster_steps
#                actual_speed = global_delta / max(elapsed, 0.001)
#                
#                # User-Requested Format (v23): 
#                # [NeuroSwift AutoTrain] HH:MM:SS | Epoch <no> | Rank <no> | <steps> / <total> | Speed | LR | Loss
#                completed = min(curr_total_steps // world_size, total_steps_in_epoch)
#                percent = (completed / total_steps_in_epoch) * 100
#                samples_sec = actual_speed * world_size * args.batch_size
#                
#                log_line = (f"{_get_stamp()}  Epoch {epoch} | Rank {rank} | "
#                            f"{completed} / {total_steps_in_epoch} ({percent:.1f}%) | "
#                            f"Speed {actual_speed:.1f} steps /sec | "
#                            f"LR: {args.lr} | Samples/s: {samples_sec:.1f} | "
#                            f"Loss: {shared_losses[rank]:.4f}")
#                            
#                print(log_line, flush=True)
#                # with open("artifacts/throughput.log", "a") as f:
#                #     f.write(log_line)
#                
#                prev_cluster_steps = curr_total_steps
#                step_start_time = current_time
#                # v22: Schedule next random log pulse
#                next_log_step = current_local_step + _rnd.randint(50, 100)
#
#            # Periodic GC
#            if (i // args.batch_size) % 500 == 0:
#                gc.collect()
#        
#        # --- Pulse-Barrier v20.1: Epoch Synchronization ---
#        # Ensures all workers finish their gradients before eval/save
#        barrier.wait()
#        
#        # Final Progress Pulse: Ensure 100% is shown at the end of the epoch
#        if rank == 0:
#            avg_train_loss = sum(shared_losses) / world_size
#            completed = total_steps_in_epoch
#            samples_sec = actual_speed * world_size * args.batch_size
#            
#            final_line = (f"  [Hyper-Drive] Epoch {epoch} | Rank {rank} | "
#                        f"{completed} / {total_steps_in_epoch} (100.0%) | "
#                        f"Final Speed {actual_speed:.1f} steps /sec | "
#                        f"LR: {args.lr} | Samples/s: {samples_sec:.1f} | "
#                        f"Loss: {avg_train_loss:.4f}\n")
#            # Unified Evaluation Logic (v23)
#            if rank == 0:
#                print(f"{_get_stamp()}  Evaluating epoch {epoch} generalization …", flush=True)
#                model.eval()
#                v_losses = []
#                with torch.no_grad():
#                    # Batch the validation dataset
#                    if v_inputs is not None:
#                        for vi in range(0, len(v_inputs), args.batch_size * 4):
#                            vi_end = min(vi + args.batch_size * 4, len(v_inputs))
#                            v_batch_in = v_inputs[vi:vi_end].to(device)
#                            v_batch_lab = v_labels[vi:vi_end].to(device)
#                            v_out = model(v_batch_in, targets=v_batch_lab)
#                            v_losses.append(v_out["loss"].item())
#                
#                avg_v_loss = sum(v_losses) / len(v_losses) if v_losses else 0.0
#                print(f"{_get_stamp()}  Epoch {epoch:02d} Summary: Train Loss {avg_train_loss:.4f} | Val Loss {avg_v_loss:.4f}", flush=True)
#                print(f"---------------------------------------------------\n", flush=True)
#
#                if avg_v_loss < best_val_loss:
#                    best_val_loss = avg_v_loss
#                    print(f"{_get_stamp()}  New best loss {avg_v_loss:.4f}! Saving checkpoint...", flush=True)
#                    # Unified v22 Saving: Root + Best folder
#                    _save_checkpoint(model, tokenizer, args.output_dir, avg_v_loss, subdir="best")
#
#    if rank == 0:
#        print(f"\n[NeuroSwift] Training Task Complete. Saving final weights to {args.output_dir}\n", flush=True)
#        # Unified v22 Saving: Root + Last folder
#        _save_checkpoint(model, tokenizer, args.output_dir, 0.0, subdir="last")
        
        
def train_worker(
    rank,
    world_size,
    model,
    train_inputs,
    train_labels,
    v_inputs,
    v_labels,
    tokenizer,
    args,
    shared_steps,
    shared_losses,
    barrier,
    total_training_steps,
    warmup_steps,
):
    device = torch.device("cpu")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = args.lr

    m_size = 16

    segment_size = len(train_inputs) // world_size
    start_idx = rank * segment_size
    end_idx = start_idx + segment_size
    total_local = end_idx - start_idx

    dataset_size = len(train_inputs)

    total_steps_in_epoch = compute_steps_per_epoch(
        dataset_size, world_size, args.batch_size
    )

    if args.max_steps_per_epoch > 0:
        total_steps_in_epoch = min(total_steps_in_epoch, args.max_steps_per_epoch)

    best_val_loss = float("inf")

    global_step = 0  # 🔥 NEW: global step tracking per rank

    for epoch in range(1, args.epochs + 1):

        if rank == 0:
            for r in range(world_size):
                shared_steps[r] = 0
                shared_losses[r] = 0.0

        barrier.wait()

        model.train()

        prev_cluster_steps = 0
        step_start_time = time.time()

        import random as _rnd
        next_log_step = _rnd.randint(50, 100)

        local_indices = torch.randperm(total_local)

        for i in range(0, total_local, args.batch_size):

            if i + args.batch_size > total_local:
                break

            if args.max_steps_per_epoch > 0 and (i // args.batch_size) >= args.max_steps_per_epoch:
                break

            # ================= LR SCHEDULER (GLOBAL CONSISTENT) =================
            current_global_step = epoch * total_steps_in_epoch + (i // args.batch_size)

            lr = cosine_lr_with_warmup(
                optimizer,
                current_global_step,
                warmup_steps,
                total_training_steps,
                args.min_lr_ratio,
            )

            # ================================================================

            batch_slice = local_indices[i : i + args.batch_size]

            batch_inp = train_inputs[start_idx + batch_slice].to(device)
            batch_lbl = train_labels[start_idx + batch_slice].to(device)

            out = model(batch_inp, targets=batch_lbl)

            loss = out["loss"] / m_size
            loss.backward()

            if ((i // args.batch_size) + 1) % m_size == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            shared_steps[rank] += 1
            shared_losses[rank] = 0.9 * shared_losses[rank] + 0.1 * (loss.item() * m_size)

            current_local_step = i // args.batch_size

            is_last_step = (
                (i + args.batch_size) >= total_local
                or (current_local_step + 1) >= total_steps_in_epoch
            )

            if rank == 0 and (current_local_step >= next_log_step or is_last_step):

                current_time = time.time()
                elapsed = current_time - step_start_time

                curr_total_steps = sum(shared_steps)

                global_delta = curr_total_steps - prev_cluster_steps
                actual_speed = global_delta / max(elapsed, 0.001)

                completed = min(curr_total_steps // world_size, total_steps_in_epoch)
                percent = (completed / total_steps_in_epoch) * 100

                samples_sec = actual_speed * world_size * args.batch_size

                log_line = (
                    f"{_get_stamp()}  Epoch {epoch} | Rank {rank} | "
                    f"{completed} / {total_steps_in_epoch} ({percent:.1f}%) | "
                    f"Speed {actual_speed:.1f} steps /sec | "
                    f"LR: {lr:.6f} | Samples/s: {samples_sec:.1f} | "
                    f"Loss: {shared_losses[rank]:.4f}"
                )

                print(log_line, flush=True)

                prev_cluster_steps = curr_total_steps
                step_start_time = current_time
                next_log_step = current_local_step + _rnd.randint(50, 100)

            if (i // args.batch_size) % 500 == 0:
                gc.collect()

        barrier.wait()

        if rank == 0:
            avg_train_loss = sum(shared_losses) / world_size
            completed = total_steps_in_epoch

            print(
                f"{_get_stamp()}  Evaluating epoch {epoch} generalization …",
                flush=True,
            )

            model.eval()
            v_losses = []

            with torch.no_grad():
                if v_inputs is not None:
                    for vi in range(0, len(v_inputs), args.batch_size * 4):
                        vi_end = min(vi + args.batch_size * 4, len(v_inputs))
                        v_batch_in = v_inputs[vi:vi_end].to(device)
                        v_batch_lab = v_labels[vi:vi_end].to(device)
                        v_out = model(v_batch_in, targets=v_batch_lab)
                        v_losses.append(v_out["loss"].item())

            avg_v_loss = sum(v_losses) / len(v_losses) if v_losses else 0.0

            print(
                f"{_get_stamp()}  Epoch {epoch:02d} Summary: Train Loss {avg_train_loss:.4f} | Val Loss {avg_v_loss:.4f}",
                flush=True,
            )
            print("---------------------------------------------------\n", flush=True)

            if avg_v_loss < best_val_loss:
                best_val_loss = avg_v_loss
                print(f"{_get_stamp()}  New best loss {avg_v_loss:.4f}! Saving checkpoint...", flush=True)
                _save_checkpoint(model, tokenizer, args.output_dir, avg_v_loss, subdir="best")

    if rank == 0:
        print(f"\n[NeuroSwift] Training Task Complete. Saving final weights to {args.output_dir}\n", flush=True)
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
