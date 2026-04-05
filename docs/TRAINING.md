# Training Guide

NeuroSwift includes a unified CLI (`python -m neuroswift`) plus standalone scripts
for every training mode: text-only, full multimodal, and continuous world auto-training.

Created by Vikash Kumar.

---

## Quick Start

### Train text-only model (fastest)

```bash
python -m neuroswift train --data-path examples/data/neuroswift_qa.jsonl --epochs 3
```

### Train multimodal Omni model

```bash
python -m neuroswift train-omni --data-dir examples/data/ --epochs 5
```

### Chat with a trained model

```bash
python -m neuroswift chat --model artifacts/neuroswift-tiny
```

### Index a folder into RAG

```bash
python -m neuroswift index --source my_data/ --output artifacts/rag.json
```

### Generate a real image artifact

```bash
python -m neuroswift generate \
  --model artifacts/neuroswift-omni \
  --prompt "a glowing neural network" \
  --modality image
```

### Start World Auto-Training (continuous)

```bash
python -m neuroswift auto-train --data-dir my_data/ --interval 60
```

---

## Subcommand Reference

### `train` — Text-Only Training (NeuroSwiftLM)

```bash
python -m neuroswift train [options]

Options:
  --epochs INT          Training epochs (default: 2)
  --batch-size INT      Mini-batch size (default: 16)
  --seq-len INT         Max sequence length (default: 96)
  --lr FLOAT            Learning rate (default: 2e-3)
  --data-path PATH      JSONL file with prompt/response pairs
  --data-dir PATH       Folder to auto-ingest (overrides --data-path)
  --output-dir PATH     Checkpoint directory (default: artifacts/neuroswift-tiny)
  --max-examples INT    Max training examples (default: 2048)
  --fetch-urls          Fetch remote URLs during folder ingestion
```

---

### `train-omni` — Multimodal Training (NeuroSwiftOmni)

```bash
python -m neuroswift train-omni [options]

Options:
  --data-dir PATH       Root folder with multimodal files (required)
  --output-dir PATH     Checkpoint directory (default: artifacts/neuroswift-omni)
  --epochs INT          Training epochs (default: 3)
  --batch-size INT      0 = auto (8 CPU / 32 GPU)
  --seq-len INT         Max text sequence length (default: 128)
  --lr FLOAT            Learning rate (default: 1e-3)
  --device STR          cpu / cuda / mps (default: auto)
  --image-size INT      Image resolution (default: 128)
  --d-model INT         Model hidden dimension (default: 192)
  --n-layers INT        Number of blocks (default: 6)
  --text-weight FLOAT   Text loss weight (default: 1.0)
  --image-weight FLOAT  Image MSE weight (default: 0.5)
  --audio-weight FLOAT  Audio MSE weight (default: 0.5)
  --video-weight FLOAT  Video MSE weight (default: 0.5)
  --auto-train          After training, enter continuous auto-training
  --auto-interval FLOAT Seconds between auto-training cycles (default: 300)
```

**Supported file types in `--data-dir`:**

| Type | Extensions |
|------|-----------|
| Text | `.txt`, `.md`, `.rst` |
| JSON/JSONL | `.json`, `.jsonl` |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp` |
| Audio | `.wav`, `.flac`, `.ogg`, `.mp3` |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv` |
| Spreadsheet | `.csv`, `.tsv`, `.xlsx` |

---

### `chat` — Interactive REPL

```bash
python -m neuroswift chat --model artifacts/neuroswift-tiny

# Single-shot (non-interactive):
python -m neuroswift chat --model artifacts/neuroswift-tiny --prompt "what is NeuroSwift?"

# With folder indexing before chat:
python -m neuroswift chat --model artifacts/neuroswift-tiny --index-dir my_docs/
```

---

### `index` — RAG Indexing

```bash
python -m neuroswift index --source my_data/ --output artifacts/rag.json
python -m neuroswift index --source my_data/ --fetch-urls   # also fetch URLs
```

---

### `generate` — Artifact Generation

```bash
# Generate a PNG image
python -m neuroswift generate \
  --model artifacts/neuroswift-omni \
  --prompt "a sunset over mountains" \
  --modality image \
  --out-dir artifacts/generated

# Generate a WAV audio file
python -m neuroswift generate \
  --model artifacts/neuroswift-omni \
  --prompt "calm orchestral music" \
  --modality audio

# Generate video frames
python -m neuroswift generate \
  --model artifacts/neuroswift-omni \
  --prompt "water flowing" \
  --modality video \
  --fps 12

# Generate all modalities at once
python -m neuroswift generate \
  --model artifacts/neuroswift-omni \
  --prompt "a thunderstorm" \
  --modality all
```

Output files are saved under `artifacts/generated/`:
- `images/generated_image_<timestamp>.png`
- `audio/generated_audio_<timestamp>.wav`
- `video/generated_video_<timestamp>/frame_*.png` (+ `.mp4` if ffmpeg available)
- `metadata/artifact_<timestamp>.json`

---

### `auto-train` — World Auto-Training

```bash
# Run continuously (watch folder every 5 min)
python -m neuroswift auto-train --data-dir my_data/ --interval 300

# One-shot cycle and exit
python -m neuroswift auto-train --data-dir my_data/ --once

# Run 10 cycles then stop
python -m neuroswift auto-train --data-dir my_data/ --max-cycles 10 --interval 60

# With layer freezing for faster incremental updates
python -m neuroswift auto-train --data-dir my_data/ --freeze-layers 2
```

---

## Programmatic API

```python
import neuroswift as ns

# Auto-training
trainer = ns.AutoTrainer(data_dir="my_data/", output_dir="artifacts/auto")
trainer.run()          # continuous loop
loss = trainer.train_once()  # single cycle

# Generate artifacts
model = ns.NeuroSwiftOmni.from_pretrained("artifacts/neuroswift-omni")
tokenizer = ns.load_tokenizer("artifacts/neuroswift-omni")
ids = torch.tensor([tokenizer.encode("a galaxy")], dtype=torch.long)
artifact = model.generate_artifact(ids, "a galaxy", modality="image")
print(artifact.summary())

# Save tensors manually
ns.save_image(image_tensor)          # → PNG
ns.save_audio(audio_tensor)          # → WAV
ns.save_video_frames(video_tensor)   # → PNG frames + MP4

# Auto device detection
device = ns.auto_device()  # returns cuda / mps / cpu
```

---

## Legacy Scripts

The original standalone scripts remain fully functional:

```bash
python train_small_llm.py --epochs 3
python test_llm.py --checkpoint artifacts/neuroswift-tiny --prompt "hello"
```

---

## Windows End-to-End Run

```bat
run_all.bat
```

## Linux / macOS

```bash
bash run_all.sh
```
