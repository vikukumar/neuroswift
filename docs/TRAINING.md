# Training Guide

NeuroSwift includes a unified CLI (`python -m neuroswift`) plus standalone scripts
for every training mode: text-only, full multimodal, and continuous world auto-training.

Created by Vikash Kumar.

---

## 🚀 Quick Start - V3 Auto-Intelligence

### 1. Multi-Source Ingestion (Local + HF + Kaggle)

V3 Alpha (God-Mode) allows you to download and train on world-class datasets with a single command. You can mix local data with multiple remote sources using comma-separated strings.

```bash
# Mix local folder with multiple remote datasets
python -m neuroswift train --data-dir my_data --hf-dataset "fka/awesome-chatgpt-prompts,nomic-ai/gpt4all-j-prompt-generations" --epochs 2
```

> [!IMPORTANT]
> **Kaggle Auth**: Ensure `kaggle.json` is in `~/.kaggle/` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

### 2. Standard Training (Local Data)

```bash
# Auto-detect folder (PDF, CODE, YAML, JSONL, LOG)
python -m neuroswift train --data-dir my_data/ --epochs 3

# Single-file training
python -m neuroswift train --data-path data/qa.jsonl --epochs 3
```

---

## Subcommand Reference

### `train` — Text-Only Training (NeuroSwiftLM)

```bash
python -m neuroswift train [options]

Options:
  --hf-dataset STR      Hugging Face repo (e.g. 'fka/awesome-chatgpt-prompts')
  --kaggle-dataset STR  Kaggle dataset (e.g. 'user/dataset-name')
  --epochs INT          Training epochs (default: 2)
  --batch-size INT      Mini-batch size (0=auto)
  --seq-len INT         Max sequence length (default: 128)
  --lr FLOAT            Learning rate (default: 2e-3)
  --data-path PATH      JSONL/CSV/TXT file path
  --data-dir PATH       Folder to auto-ingest (overrides --data-path)
  --max-examples INT    Max training examples cap (default: 50000)
  --no-dedup            Disable SHA-256 fingerprinting (NOT RECOMMENDED)
```

### `train-omni` — Multimodal Training (NeuroSwiftOmni)

```bash
python -m neuroswift train-omni [options]

Options:
  --data-dir PATH       Root folder with multimodal files (required)
  --output-dir PATH     Checkpoint directory (default: artifacts/neuroswift-omni)
  --epochs INT          Training epochs (default: 3)
  --lr FLOAT            Learning rate (default: 1e-3)
  --device STR          cpu / cuda / mps (default: auto)
```

**Supported file types in `--data-dir` (V3 Omni-Ingestion):**

| Type | Extensions |
|------|-----------|
| **Remote** | **Hugging Face Hub, Kaggle API (Auto-Unzip)** |
| **Logic** | **20+ Languages (.py, .js, .c, .rs, .go, .cpp, etc.)** |
| **Docs** | **.pdf, .json, .jsonl, .yaml, .csv, .xlsx, .log, .txt, .md** |
| **Media** | **.png, .jpg, .webp, .wav, .mp3, .mp4, .mov** |

---

## 📊 Data Dominance (V3 Alpha)

V3 Alpha features **SHA-256 Fingerprinting**, fixing the 'Instruction Collision' bug that caused 98% data loss in previous versions. 

- **10x Data Density**: Reclaims high-quality content that was previously pruned.
- **Modality-Aware Filtering**: Rescues short mathematical and law answers from the quality filter.
- **Dynamic Context**: Supports sequence lengths up to 2048+ tokens for complex reasoning.

---

## Programmatic API

```python
import neuroswift as ns

# Run the God-level pipeline on a remote source
train, val, stats = ns.run_pipeline(
    source=Path("."), 
    hf_dataset="fka/awesome-chatgpt-prompts"
)

# Automated training cycle
trainer = ns.AutoTrainer(data_dir="my_data/", output_dir="artifacts/auto")
loss = trainer.train_once()
```

---

## Legacy Scripts

Standalone scripts remain fully functional for direct usage:

```bash
python train_small_llm.py --hf-dataset fka/awesome-chatgpt-prompts --epochs 2
python test_llm.py --model-dir artifacts/neuroswift-tiny --prompt "hello"
```
