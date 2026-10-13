#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR="${DATA_DIR:-$PWD/training-data}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
PYTHON_BIN="python3"
if [ -x "$PWD/.venv/bin/python" ]; then PYTHON_BIN="$PWD/.venv/bin/python"; fi
if ! "$PYTHON_BIN" -c "import torch, cv2, fastapi" >/dev/null 2>&1; then
  echo "Python dependencies are missing. Run ./setup-laptop.sh first." >&2
  exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "Training data directory not found: $DATA_DIR" >&2
  echo "Add sequence folders with reference.jpg, driving.mp4 and target.mp4." >&2
  exit 1
fi
mkdir -p "$PWD/training-audit" "$PWD/weights"
echo "Auditing aligned data..."
"$PYTHON_BIN" prepare_data.py --data "$DATA_DIR" --report "$PWD/training-audit" --min-valid-sequences 1
echo "Training the compact EffectLab model from random initialisation with automatic CPU/CUDA detection."
"$PYTHON_BIN" train_scratch.py --data "$DATA_DIR" --output "$PWD/weights/effectlab_scratch.pt" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --max-steps "$MAX_STEPS" --height 96 --width 64 --clip-frames 4 --base-channels 8
