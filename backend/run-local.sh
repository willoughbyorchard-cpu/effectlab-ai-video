#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Desktop or Docker Engine first." >&2
  exit 1
fi

echo "Building EffectLab Scratch from source (Docker will reuse cached layers)..."
docker build -t effectlab-scratch .

if [ -z "${EFFECTLAB_API_KEY:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    EFFECTLAB_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  else
    echo "Set EFFECTLAB_API_KEY to a long random secret before running." >&2
    exit 1
  fi
fi

echo
echo "EffectLab Scratch UI: http://127.0.0.1:7860"
echo "Private local API key (paste into the page settings):"
echo "$EFFECTLAB_API_KEY"
echo
echo "This is a from-scratch network, not a pretrained AI model."
echo "If no trained checkpoint exists, the page will explain that training is required."
echo "The app makes no hosted AI-inference calls and downloads no third-party AI weights."
echo "Press Ctrl+C to stop."

GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS+=(--gpus all)
else
  echo "No NVIDIA GPU detected; local inference will use CPU and may be very slow."
fi

docker run "${GPU_ARGS[@]}" --rm --name effectlab-scratch-local \
  -p 127.0.0.1:7860:7860 \
  -e EFFECTLAB_API_KEY="$EFFECTLAB_API_KEY" \
  -e EFFECTLAB_ALLOWED_ORIGINS="http://127.0.0.1:7860,http://localhost:7860" \
  -v effectlab-model-weights:/app/weights \
  -v effectlab-render-output:/tmp/effectlab \
  effectlab-scratch
