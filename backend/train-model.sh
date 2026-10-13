#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-$PWD/training-data}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed." >&2
  exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "Training data directory not found: $DATA_DIR" >&2
  echo "See README.md for the required aligned source/target video layout." >&2
  exit 1
fi

docker build -t effectlab-scratch .
docker run --rm --gpus all \
  -v "$DATA_DIR:/data:ro" \
  -v effectlab-model-weights:/app/weights \
  effectlab-scratch \
  python /app/train_scratch.py \
    --data /data \
    --output /app/weights/effectlab_scratch.pt \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"
