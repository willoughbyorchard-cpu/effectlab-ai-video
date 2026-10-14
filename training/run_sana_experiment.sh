#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_NONCOMMERCIAL_RESEARCH:-}" != "YES" ]]; then
  echo "STOP: the example dataset is non-commercial-research-only."
  echo "Read training/SANA_EXPERIMENT.md and set ALLOW_NONCOMMERCIAL_RESEARCH=YES only if your use is strictly non-commercial research."
  exit 2
fi

WORK_DIR="${EFFECTLAB_SANA_WORK_DIR:-$PWD/effectlab-sana-work}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "STOP: NVIDIA GPU tools are unavailable. This published training recipe needs an NVIDIA GPU environment."
  exit 2
fi
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$GPU_COUNT" -lt 8 ]]; then
  echo "STOP: found ${GPU_COUNT} NVIDIA GPU(s); the published recipe launches 8 processes and expects an 8-GPU setup."
  echo "No dataset or model weights were downloaded. Use an appropriate GPU machine or adapt the recipe with the model authors' guidance."
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "STOP: git is required."
  exit 2
fi
mkdir -p "$WORK_DIR"
if [[ ! -d "$WORK_DIR/Sana/.git" ]]; then
  git clone https://github.com/NVlabs/Sana.git "$WORK_DIR/Sana"
fi
cd "$WORK_DIR/Sana"
python -m pip install --upgrade huggingface_hub

hf download Efficient-Large-Model/SANA-Streaming-example-training-dataset \
  --repo-type dataset --local-dir data/sana_streaming_1k

DATA_DIR="data/sana_streaming_1k/data/example_data"
if [[ ! -f "$DATA_DIR/checksums.sha256" ]]; then
  echo "STOP: expected dataset checksum file not found at $DATA_DIR/checksums.sha256"
  exit 2
fi
(
  cd "$DATA_DIR"
  sha256sum -c checksums.sha256
)

hf download Efficient-Large-Model/SANA-Streaming \
  --include dit/sana_streaming_ar.pth --local-dir checkpoints/sana_streaming
hf download Efficient-Large-Model/SANA-Streaming_bidirectional \
  --include dit/sana_bidirectional_short.pth --local-dir checkpoints/sana_streaming_bidirectional

# Published recipe from NVlabs/Sana docs. This is a costly research run, not a free hosted job.
torchrun --nproc_per_node=8 --master_port=29500 \
  train_video_scripts/train_video_ivjoint_chunk.py \
  --config_path=configs/sana_streaming/train/sana_streaming_bidirectional_2b_720p.yaml
