#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="python3"
if [ -x "$PWD/.venv/bin/python" ]; then PYTHON_BIN="$PWD/.venv/bin/python"; fi
if ! "$PYTHON_BIN" -c "import torch, cv2, fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Python dependencies are missing. Run ./setup-laptop.sh first." >&2
  exit 1
fi
mkdir -p "$PWD/weights"
export EFFECTLAB_CHECKPOINT="${EFFECTLAB_CHECKPOINT:-$PWD/weights/effectlab_scratch.pt}"
export EFFECTLAB_TRAINING_ROOT="${EFFECTLAB_TRAINING_ROOT:-$PWD/training-data}"
export EFFECTLAB_WORK_ROOT="${EFFECTLAB_WORK_ROOT:-$HOME/.effectlab/work}"
echo
echo "EffectLab local UI: http://127.0.0.1:7860"
echo "Training page:       http://127.0.0.1:7860/train.html"
echo "No API key is needed. The engine listens only on this computer; keep this terminal open."
echo "Training and inference use this laptop; no hosted AI renderer is called."
echo "Press Ctrl+C to stop."
exec "$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 7860
