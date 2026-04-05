"""
neuroswift.train_omni
=====================
Dedicated multimodal training script for NeuroSwiftOmni.

Supports:
  - All modalities: text, image, audio, video (any combination)
  - CPU + GPU auto-detection with AMP on CUDA
  - Gradient checkpointing for memory-limited hardware
  - Combined loss: text CE + image MSE + audio MSE + video MSE (weighted)
  - Auto-training flag for continuous incremental loop
  - Checkpoint saving via NeuroSwiftOmni.save_pretrained()
  - Integrated with data_pipeline for auto-cleaning and quality filtering

Usage::

    python -m neuroswift train-omni --data-dir my_data/ --epochs 5

Or as a script::

    python neuroswift/train_omni.py --data-dir my_data/ --epochs 5
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data_pipeline import TrainPair, run_pipeline
from .ingest import DatasetFolderReader, MultimodalSample
from .layers import auto_device
from .omni import NeuroSwiftOmni, NeuroSwiftOmniConfig
from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[NeuroSwift OmniTrain] %(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# ---------------------------------------------------------------------------
# Dataset / collation
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = "user: {prompt}\nassistant:"


class OmniDataset(Dataset):
    """
    Combines high-quality text pairs from run_pipeline with multimodal samples 
    from DatasetFolderReader.
    """
    def __init__(
        self,
        text_pairs: list[TrainPair],
        multimodal_samples: list[MultimodalSample],
    ) -> None:
        self.text_pairs = text_pairs
        self.mm_samples = multimodal_samples
        # Total size is max of both, or sum? sum is safer for coverage.
        self.total = len(text_pairs) + len(multimodal_samples)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> TrainPair | MultimodalSample:
        if idx < len(self.text_pairs):
            return self.text_pairs[idx]
        return self.mm_samples[idx - len(self.text_pairs)]


class OmniCollate:
    """Collate multimodal samples and text pairs into padded batch tensors."""

    def __init__(
        self,
        tokenizer: WordTokenizer,
        seq_len: int,
        image_size: int = 128,
        max_audio_samples: int = 8192,
        max_video_frames: int = 8,
        video_frame_size: int = 64,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.image_size = image_size
        self.max_audio_samples = max_audio_samples
        self.max_video_frames = max_video_frames
        self.video_frame_size = video_frame_size

    def __call__(self, batch_items: list[TrainPair | MultimodalSample]) -> dict[str, Any]:
        text_inputs, text_targets = [], []
        images, audios, videos = [], [], []
        pad_id = self.tokenizer.stoi[self.tokenizer.pad_token]

        for item in batch_items:
            # Case 1: Pure text pair from pipeline
            if isinstance(item, TrainPair):
                prompt_ids = self.tokenizer.encode(_PROMPT_TEMPLATE.format(prompt=item.prompt), add_eos=False)
                resp_ids = self.tokenizer.encode(item.response, add_eos=True)
                full_ids = (prompt_ids + resp_ids)[: self.seq_len + 1]
                if len(full_ids) >= 2:
                    inp = full_ids[:-1]
                    labels = [-100 if i < len(prompt_ids) else full_ids[i + 1] for i in range(len(inp))]
                    pad_len = self.seq_len - len(inp)
                    text_inputs.append(torch.tensor(inp + [pad_id] * pad_len, dtype=torch.long))
                    text_targets.append(torch.tensor(labels + [-100] * pad_len, dtype=torch.long))
                continue

            # Case 2: Multimodal sample
            sample = item
            if sample.text:
                # Use as descriptive context
                raw_text = sample.to_rag_text()[:400]
                prompt = _PROMPT_TEMPLATE.format(prompt=f"summarize {Path(sample.source).name}")
                resp = raw_text
                p_ids = self.tokenizer.encode(prompt, add_eos=False)
                r_ids = self.tokenizer.encode(resp, add_eos=True)
                f_ids = (p_ids + r_ids)[: self.seq_len + 1]
                if len(f_ids) >= 2:
                    inp = f_ids[:-1]
                    lbls = [-100 if i < len(p_ids) else f_ids[i + 1] for i in range(len(inp))]
                    pad_len = self.seq_len - len(inp)
                    text_inputs.append(torch.tensor(inp + [pad_id] * pad_len, dtype=torch.long))
                    text_targets.append(torch.tensor(lbls + [-100] * pad_len, dtype=torch.long))

            if sample.tensor is not None and sample.modality == "image":
                t = sample.tensor # [C, H, W]
                if t.ndim == 3 and t.shape[0] == 3:
                    t = F.interpolate(t.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False).squeeze(0)
                    images.append(t)

            if sample.tensor is not None and sample.modality == "audio":
                wav = sample.tensor.float()
                if wav.numel() > self.max_audio_samples:
                    wav = wav[: self.max_audio_samples]
                else:
                    wav = F.pad(wav, (0, self.max_audio_samples - wav.numel()))
                audios.append(wav)

            if sample.tensor is not None and sample.modality == "video":
                vid = sample.tensor # [F, C, H, W]
                vid = vid[: self.max_video_frames]
                if vid.ndim == 4:
                    F_count = vid.shape[0]
                    vid_flat = F.interpolate(vid, size=(self.video_frame_size, self.video_frame_size), mode="bilinear", align_corners=False)
                    if self.max_video_frames > F_count:
                        pad_f = self.max_video_frames - F_count
                        vid_flat = torch.cat([vid_flat, torch.zeros(pad_f, *vid_flat.shape[1:])], dim=0)
                    videos.append(vid_flat)

        batch: dict[str, Any] = {}
        if text_inputs:
            batch["text_input_ids"] = torch.stack(text_inputs)
            batch["text_targets"] = torch.stack(text_targets)
        if images:
            batch["image_tensors"] = torch.stack(images)
        if audios:
            batch["audio_tensors"] = torch.stack(audios)
        if videos:
            batch["video_tensors"] = torch.stack(videos)
        return batch


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_omni_loss(
    model: NeuroSwiftOmni,
    batch: dict[str, Any],
    text_weight: float = 1.0,
    image_weight: float = 0.5,
    audio_weight: float = 0.5,
    video_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Forward pass and compute weighted multimodal loss."""
    outputs = model(
        text_input_ids=batch.get("text_input_ids"),
        image_tensors=batch.get("image_tensors"),
        audio_tensors=batch.get("audio_tensors"),
        video_tensors=batch.get("video_tensors"),
        update_plasticity=False,
    )

    total_loss = outputs["aux_loss"] * model.config.aux_loss_scale
    breakdown: dict[str, float] = {"aux": outputs["aux_loss"].item()}

    # Text loss
    if "text_logits" in outputs and "text_targets" in batch:
        text_logits = outputs["text_logits"].float()
        text_targets = batch["text_targets"].to(text_logits.device)
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=0.05
        )
        total_loss = total_loss + text_weight * text_loss
        breakdown["text"] = text_loss.item()

    # Image reconstruction loss (MSE)
    if "image_tensors" in batch and "image_tensor" in outputs:
        target = batch["image_tensors"].to(outputs["image_tensor"].device)
        pred = outputs["image_tensor"]
        if pred.shape != target.shape:
             pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
        img_loss = F.mse_loss(pred, target)
        total_loss = total_loss + image_weight * img_loss
        breakdown["image"] = img_loss.item()

    # Audio reconstruction loss
    if "audio_tensors" in batch and "audio_tensor" in outputs:
        target = batch["audio_tensors"].to(outputs["audio_tensor"].device)
        pred = outputs["audio_tensor"]
        n = min(pred.size(-1), target.size(-1))
        aud_loss = F.mse_loss(pred[..., :n], target[..., :n])
        total_loss = total_loss + audio_weight * aud_loss
        breakdown["audio"] = aud_loss.item()

    # Video reconstruction loss
    if "video_tensors" in batch and "video_tensor" in outputs:
        target = batch["video_tensors"].to(outputs["video_tensor"].device)
        pred = outputs["video_tensor"]
        if pred.shape[2:] != target.shape[2:]:
            B, F_c, C, H, W = target.shape
            pred_flat = pred.reshape(B * F_c, C, *pred.shape[-2:])
            pred_flat = F.interpolate(pred_flat, size=(H, W), mode="bilinear", align_corners=False)
            pred = pred_flat.view(B, F_c, C, H, W)
        vid_loss = F.mse_loss(pred, target)
        total_loss = total_loss + video_weight * vid_loss
        breakdown["video"] = vid_loss.item()

    return total_loss, breakdown


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Train NeuroSwiftOmni on multimodal data.")
    p_data = parser.add_argument_group("Data & Ingestion")
    p_data.add_argument("--data-dir", type=Path, required=True,
                        help="Root folder containing multimodal training files.")
    p_data.add_argument("--val-fraction", type=float, default=0.05)
    p_data.add_argument("--max-examples", type=int, default=5000)
    p_data.add_argument("--max-per-file", type=int, default=1000)
    
    p_arch = parser.add_argument_group("Architecture")
    p_arch.add_argument("--output-dir", type=Path, default=Path("artifacts/neuroswift-omni"))
    p_arch.add_argument("--d-model", type=int, default=192)
    p_arch.add_argument("--n-layers", type=int, default=6)
    p_arch.add_argument("--num-experts", type=int, default=8)
    p_arch.add_argument("--image-size", type=int, default=128)
    p_arch.add_argument("--max-video-frames", type=int, default=8)
    p_arch.add_argument("--video-frame-size", type=int, default=64)

    p_train = parser.add_argument_group("Training")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=0)
    p_train.add_argument("--grad-accum", type=int, default=1)
    p_train.add_argument("--seq-len", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--grad-checkpoint", action="store_true")
    
    p_loss = parser.add_argument_group("Loss Weights")
    p_loss.add_argument("--text-weight", type=float, default=1.0)
    p_loss.add_argument("--image-weight", type=float, default=0.5)
    p_loss.add_argument("--audio-weight", type=float, default=0.5)
    p_loss.add_argument("--video-weight", type=float, default=0.5)

    p_misc = parser.add_argument_group("Misc")
    p_misc.add_argument("--save-every", type=int, default=100)
    p_misc.add_argument("--auto-train", action="store_true")
    p_misc.add_argument("--auto-interval", type=float, default=300.0)
    p_misc.add_argument("--prompt", type=str, default="describe this scene")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------


def train_omni(args: Namespace) -> None:
    torch.manual_seed(42)
    device = torch.device(args.device) if args.device else auto_device()
    logger.info(f"Device: {device}")

    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    batch_size = args.batch_size if args.batch_size > 0 else (8 if device.type == "cpu" else 32)
    logger.info(f"Batch size: {batch_size} (grad_accum={args.grad_accum})")

    # 1. Integrated Data Pipeline (Text)
    logger.info(f"Running data pipeline on: {args.data_dir}")
    train_pairs, _, pipe_stats = run_pipeline(
        source=args.data_dir,
        max_total_pairs=args.max_examples,
        max_pairs_per_file=args.max_per_file,
        val_fraction=args.val_fraction,
        verbose=True,
    )
    logger.info(f"High-quality text pairs: {len(train_pairs)}")

    # 2. Multimodal Ingestion (Media)
    reader = DatasetFolderReader(image_size=(args.image_size, args.image_size), max_video_frames=args.max_video_frames)
    mm_samples = reader.read_folder(args.data_dir)
    logger.info(f"Multimodal media samples: {len(mm_samples)}")

    # 3. Vocabulary
    texts = [p.prompt + " " + p.response for p in train_pairs[:1000]]
    texts += [s.text[:500] for s in mm_samples[:1000] if s.text]
    if not texts:
        texts = ["neuroswift omni multimodal learning"]
    tokenizer = WordTokenizer.from_texts(texts)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # 4. DataLoader
    dataset = OmniDataset(train_pairs, mm_samples)
    collator = OmniCollate(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        image_size=args.image_size,
        max_video_frames=args.max_video_frames,
        video_frame_size=args.video_frame_size
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, drop_last=True)

    # 5. Model
    config = NeuroSwiftOmniConfig(
        text_vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_experts=args.num_experts,
        image_size=args.image_size,
        max_video_frames=args.max_video_frames,
        video_frame_size=args.video_frame_size,
    )
    ckpt_file = args.output_dir / "model.safetensors"
    if ckpt_file.exists():
        logger.info("Resuming Omni checkpoint ...")
        model = NeuroSwiftOmni.from_pretrained(args.output_dir, device=device)
    else:
        model = NeuroSwiftOmni(config).to(device)

    # 6. Optimizer + Schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.cuda.GradScaler() if device.type == "cuda" else None
    
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.05)
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        optimizer.zero_grad(set_to_none=True)
        
        for i, batch in enumerate(loader):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss, _ = compute_omni_loss(model, batch, args.text_weight, args.image_weight, args.audio_weight, args.video_weight)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                loss, _ = compute_omni_loss(model, batch, args.text_weight, args.image_weight, args.audio_weight, args.video_weight)
                loss = loss / args.grad_accum
                loss.backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # Simple linear LR decay after warmup
                if global_step < warmup_steps:
                    curr_lr = args.lr * global_step / max(1, warmup_steps)
                else:
                    curr_lr = args.lr * (1.0 - (global_step - warmup_steps) / max(1, total_steps - warmup_steps))
                for pg in optimizer.param_groups: pg["lr"] = max(curr_lr, args.lr * 0.01)

                if global_step % args.save_every == 0:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    logger.info(f"  [step {global_step}] checkpoint")

            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            
        logger.info(f"Epoch {epoch:02d}/{args.epochs} | loss={epoch_loss/max(1,n_batches):.4f}")

    # Final Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Final model saved to: {args.output_dir}")

    # Demmo
    model.eval()
    logger.info(f"Demo generation for: {args.prompt}")
    p_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    try:
        art = model.generate_artifact(p_ids, args.prompt, modality="image", out_dir=args.output_dir / "demo")
        logger.info(art.summary())
    except Exception as e:
        logger.warning(f"Demo fail: {e}")

    if args.auto_train:
        from .auto_trainer import AutoTrainer
        trainer = AutoTrainer(data_dir=args.data_dir, output_dir=args.output_dir / "auto_loop", interval=args.auto_interval)
        trainer.run()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_omni(args)


if __name__ == "__main__":
    main()
