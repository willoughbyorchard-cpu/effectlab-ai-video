#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-$PWD/training-data}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
AUDIT_DIR="${AUDIT_DIR:-$PWD/training-audit}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed." >&2
  exit 1
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "Training data directory not found: $DATA_DIR" >&2
  echo "Each sequence must contain reference.jpg, driving.mp4 and a target.mp4 aligned to the same action and camera timing." >&2
  echo "Do not use the same source clip as its target: that trains reconstruction, not identity replacement." >&2
  exit 1
fi

mkdir -p "$AUDIT_DIR"
echo "Checking training sequence files, frame counts, frame rates and duration..."
docker build -t effectlab-scratch .
docker run --rm \
  -v "$DATA_DIR:/data:ro" \
  -v "$AUDIT_DIR:/audit" \
  effectlab-scratch \
  python /app/prepare_data.py --data /data --report /audit --min-valid-sequences 1

echo "Basic timing audit passed. Open the contact sheets in $AUDIT_DIR and manually confirm target/driving clips show the same action and camera movement."
echo "Starting training from random initialization. No pretrained weights are loaded."
docker run --rm --gpus all \
  -v "$DATA_DIR:/data:ro" \
  -v effectlab-model-weights:/app/weights \
  effectlab-scratch \
  python /app/train_scratch.py \
    --data /data \
    --output /app/weights/effectlab_scratch.pt \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"
