#!/usr/bin/env bash
set -euo pipefail

bash ./setup_env.sh
source venv/bin/activate

python train_small_llm.py --data-path examples/data/neuroswift_corpus.txt --output-dir artifacts/neuroswift-tiny
python test_llm.py --model-dir artifacts/neuroswift-tiny --prompt "neuroswift "
