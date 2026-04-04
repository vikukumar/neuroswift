# Training Guide

This repository includes a tiny training script for quickly validating the NeuroSwift architecture on CPU.

Created by Vikash Kumar.

## What The Script Does

`train_small_llm.py`:

- builds a character-level tokenizer
- creates a toy dataset from a small built-in corpus
- trains `NeuroSwiftLM` for a few epochs
- demonstrates plasticity after training
- saves a checkpoint for later inference

## Default Output

The default checkpoint path is:

```txt
artifacts/neuroswift_tiny.pt
```

## Train

```bash
python train_small_llm.py
```

## Train With Custom Settings

```bash
python train_small_llm.py --epochs 10 --batch-size 8 --seq-len 64 --stride 16 --lr 0.001
```

## Test A Saved Model

```bash
python test_llm.py --checkpoint artifacts/neuroswift_tiny.pt --prompt "neuroswift "
```

## End-To-End Run

### Windows

```bat
run_all.bat
```

### Linux / macOS

```bash
bash run_all.sh
```

## Notes

- the current training data is intentionally tiny
- generated text is for validation, not benchmark quality
- CPU thread count is set conservatively for local use
