#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="python3"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python 3.11+ is required." >&2; exit 1; }
if [ ! -x "$PWD/.venv/bin/python" ]; then "$PYTHON_BIN" -m venv .venv; fi
"$PWD/.venv/bin/python" -m pip install --upgrade pip
"$PWD/.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$PWD/.venv/bin/python" -m pip install -r requirements.txt
echo "Setup complete. Run ./run-local.sh to open EffectLab, then visit /train.html."
