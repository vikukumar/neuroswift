# Architecture Notes

NeuroSwift is designed as the world's most powerful, all-in-one CPU+GPU multimodal
deep learning framework with linear-time processing, sparse conditional computation,
online plasticity, and real artifact generation.

Created by Vikash Kumar.

## Core Design Philosophy

> **CPU-first, GPU-accelerated** — every component runs at full quality on CPU,
> and gains additional acceleration on CUDA/MPS without any code changes.

## Component Overview

### 1. LinearSSM — Linear State-Space Mixer

`LinearSSM` replaces standard self-attention with a recurrent state-space scan:

- **Sequential mode (CPU)**: O(T) recurrent scan, memory-efficient, excellent throughput
- **Parallel mode (GPU)**: log-space associative prefix scan, dispatched automatically for sequences ≥ 32 tokens on CUDA/MPS
- Lower memory pressure vs. full attention
- Predictable latency scaling on long contexts

### 2. SparseMoE — Sparse Mixture of Experts

`SparseMoE` routes each token to the top-2 experts out of 8 total:

- Lower FLOPs per token at inference
- Context-aware routing via prefix-sum conditioning
- SiLU-gated expert MLPs (SwiGLU-style)
- CPU capacity control prevents expert overflow

### 3. HebbianUpdater — Online Plasticity

Per-layer fast-weight matrix updated online via Oja's rule:

- Short-term adaptive memory without backpropagation
- Modulation gate controls learning rate per token
- Clip + decay prevents unbounded fast-weight growth

### 4. CrossModalAttention — Modality Fusion

Lightweight multi-head cross attention between modality token streams:

- Allows text tokens to attend to image/audio/video context
- Residual connection with learnable output scale
- n_heads automatically adjusted to divide d_model

### 5. ImageCNNDecoder — Spatial Image Generation

Replaces prior flat-linear decoder with a full CNN upsampling pipeline:

- Linear projection → small spatial feature map (e.g. 4×4)
- 4× `ConvTranspose2d+SiLU` upsample stages → 64×64 → 128×128+
- Final 1×1 RGB conv + bilinear interpolation to exact target size
- Sigmoid output in [0, 1] range → saves directly as PNG

### 6. VideoCNNDecoder — Multi-Frame Video Generation

Per-frame conditioning via temporal MLP + shared `ImageCNNDecoder`:

- Linear → per-frame latent vectors → decoded independently
- Default: 8 frames × 64×64 pixels

### 7. ArtifactSaver — Real File Output

`ArtifactSaver` and `OmniArtifact` handle saving:

- **Image**: PNG via Pillow
- **Audio**: WAV via soundfile (22 050 Hz mono)
- **Video**: PNG frame directory + optional MP4 via imageio/ffmpeg
- JSON metadata sidecar per generation call

### 8. AutoTrainer — World Auto-Training

Continuous self-supervised incremental training loop:

- Polls `data_dir` every N seconds for new files
- Adaptive batch size: 8 (CPU), 16 (MPS), 64 (CUDA)
- Plasticity warm-up pass after each cycle
- Optional layer freezing for faster incremental updates
- Crash-resilient: checkpoint every N batches

## Block Composition

Each `NeuroSwiftBlock` (stacked N times):

1. `LinearSSM` — state-space sequence mixing
2. `SparseMoE` — sparse conditional FFN
3. `HebbianUpdater` — online fast-weight plasticity

## NeuroSwiftOmni Flow

```
[text tokens]   [image patches]  [audio chunks]  [video frames]
     ↓                ↓               ↓               ↓
 Embedding        Patch proj       Chunk proj     Frame encoder
     ↓                ↓               ↓               ↓
         ←────── Concatenate + modality bias ──────────→
                          ↓
                  N × NeuroSwiftBlock
                    (SSM · MoE · Plastic)
                          ↓
                   RMSNorm + mask
                          ↓
                 CrossModalAttention
               (text↔image↔audio↔video)
                          ↓
          ┌───────────────┼──────────────────┐
     Text head       Image CNN         Audio MLP
      (CE loss)   (DeConv↑, MSE)    (Linear, MSE)
                                          ↓
                                   Video CNN (per-frame)
```

## Device Strategy

| Device | Features |
|--------|----------|
| CPU    | Sequential SSM scan, float32, 8 threads max |
| MPS    | Parallel scan (Apple Silicon), batch=16 |  
| CUDA   | Parallel scan + AMP float16 + GradScaler |

## Design Goals

- **Fastest** linear-time architecture on any hardware
- **Automatic** understanding, learning, responding, generating
- **All-in-one**: text, image, audio, video from a single model
- **World Auto-Training**: continuously learns from new data without intervention
- **CPU-first**: no GPU required to run, GPU accelerates transparently
