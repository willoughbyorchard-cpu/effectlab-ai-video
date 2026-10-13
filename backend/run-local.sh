#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed. Install Docker Desktop or Docker Engine first." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA driver / nvidia-smi was found on this machine." >&2
  echo "This Wan 2.2 Animate backend requires a suitable NVIDIA CUDA GPU; it will not run on integrated graphics or normal macOS hardware." >&2
  exit 1
fi
nvidia-smi
echo "Building the local EffectLab engine image (cached layers are reused)..."
docker build -t effectlab-engine .

if [ -z "$EFFECTLAB_API_KEY" ]; then
  if command -v python3 >/dev/null 2>&1; then
    EFFECTLAB_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  else
    echo "Set EFFECTLAB_API_KEY to a long random secret before running." >&2
    exit 1
  fi
fi

echo
echo "EffectLab will run locally at http://127.0.0.1:7860"
echo "Enter this private API key into the settings panel on that local page:"
echo "$EFFECTLAB_API_KEY"
echo
echo "Photos and videos uploaded to this app are processed by this machine's GPU. They are not sent to a public AI renderer."
echo "The first start downloads open model weights. Subsequent runs reuse the local Hugging Face cache."
echo "Press Ctrl+C to stop the local engine."
echo

docker run --gpus all --rm --name effectlab-local \
  -p 127.0.0.1:7860:7860 \
  -e EFFECTLAB_API_KEY="$EFFECTLAB_API_KEY" \
  -e EFFECTLAB_ALLOWED_ORIGINS="http://127.0.0.1:7860,http://localhost:7860" \
  -v effectlab-hf-cache:/root/.cache/huggingface \
  -v effectlab-comfy-model-links:/opt/ComfyUI/models \
  -v effectlab-render-output:/tmp/effectlab \
  effectlab-engine
