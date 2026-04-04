#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

"$PYTHON_BIN" -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
