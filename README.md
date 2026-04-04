# NeuroSwift

NeuroSwift is a CPU-optimized small language model workspace built around linear-complexity state space modeling, sparse mixture-of-experts routing, and online Hebbian plasticity.

Created by Vikash Kumar.

## Highlights

- Linear token mixing with a `LinearSSM` block instead of quadratic self-attention
- Sparse MoE routing that activates only 2 of 8 experts per token
- Online plasticity through a `HebbianUpdater` for short-term adaptive memory
- Tiny end-to-end training and testing scripts for CPU-first experimentation

## Project Layout

- `neuroswift/layers.py`: RMSNorm, `LinearSSM`, and `SparseMoE`
- `neuroswift/plasticity.py`: online Hebbian fast-weight updater
- `neuroswift/model.py`: `NeuroSwiftLM` model and config
- `train_small_llm.py`: tiny training pipeline and checkpoint saver
- `test_llm.py`: checkpoint loading, inference, and plasticity check
- `setup_env.sh` / `setup_env.bat`: environment setup
- `run_all.sh` / `run_all.bat`: one-command setup, train, and test

## Quick Start

### Windows

```bat
setup_env.bat
venv\Scripts\activate.bat
python train_small_llm.py
python test_llm.py
```

Or run the full flow:

```bat
run_all.bat
```

### Linux / macOS

```bash
bash setup_env.sh
source venv/bin/activate
python train_small_llm.py
python test_llm.py
```

Or run the full flow:

```bash
bash run_all.sh
```

## Package Build

Build source and wheel distributions locally:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

## Publish To PyPI

### Manual publish

```bash
python -m twine upload dist/*
```

### GitHub Actions publish

This repository includes a publish workflow at `.github/workflows/publish.yml`.

Typical release flow:

```bash
python scripts/bump_version.py bump patch
git add VERSION neuroswift/_version.py
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

The workflow will build the package and publish it when a `v*` tag is pushed.

Note: configure PyPI trusted publishing or repository secrets before using automated publish.

## Example Workflow

1. Create and activate the virtual environment.
2. Train the tiny model on the built-in toy corpus.
3. Save the checkpoint to `artifacts/neuroswift_tiny.pt`.
4. Load the checkpoint with `test_llm.py`.
5. Inspect generated text and plasticity-driven logit changes.

## Core Idea

NeuroSwift replaces attention-heavy token mixing with a recurrent state-space scan that scales linearly with sequence length. A sparse MoE block reduces CPU compute by executing only the most relevant experts. A plasticity layer updates small fast-weight memories during inference so the model can adapt to recent context without full backpropagation.

## Training

Default training uses a small character-level corpus embedded directly in `train_small_llm.py`. The script is intended as a simple research sandbox for architecture experiments rather than a production training pipeline.

## Inference and Plasticity

`test_llm.py` demonstrates two behaviors:

- standard text generation from a saved checkpoint
- online plasticity by comparing logits before and after a short adaptive update

## Documentation

- [Architecture Notes](docs/ARCHITECTURE.md)
- [Training Guide](docs/TRAINING.md)
- [Publishing Guide](docs/PUBLISHING.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Authors](AUTHORS.md)

## Status

This repository is a compact experimental workspace for research and prototyping.

## Author

Created by Vikash Kumar.
