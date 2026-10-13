# EffectLab — Local Full-Body Engine

This version does not call a public AI-rendering website. The browser interface and generation API are served together from your own machine at http://127.0.0.1:7860. Uploaded photos and source clips are submitted to that local API and the Wan 2.2 Animate pipeline runs on your machine's NVIDIA GPU.

## First, check the hardware

- Linux or Windows with WSL2 and Docker, a working NVIDIA driver, and Docker GPU support.
- Wan 2.2 Animate is a 14B-parameter video model. Official deployments list very high GPU memory requirements; NVIDIA's support matrix lists 80 GB GPU memory as the minimum for Wan2.2-Animate-14B. The quantised ComfyUI pipeline here may use less, but it has not been validated on a 24 GB card. Integrated graphics and ordinary Mac hardware are not enough for this setup.
- Tens of GB of free disk space for model weights, ComfyUI, dependencies and cache.

Model reference: https://github.com/Wan-Video/Wan2.2 . NVIDIA deployment requirements: https://docs.nvidia.com/nim/visual-genai/latest/support-matrix.html

## Run locally

1. Clone or download this repository onto the NVIDIA GPU computer.
2. Open a terminal in the backend folder.
3. Run docker build -t effectlab-engine . (the launch script does this the first time automatically).
4. Run bash run-local.sh.
5. Keep the terminal open. Open http://127.0.0.1:7860 in your browser. Copy the private API key printed in the terminal into the Local GPU engine settings and save it.
6. Wait for model loading to complete. The health check in the settings will report readiness.

The script binds the service to 127.0.0.1 only; it is not exposed to your home network or the public internet. Do not change the port binding to 0.0.0.0 unless you understand the security implications.

## Fully local media use

The standalone local page does not load remote preview images or stock clips. Upload your own source MP4 in the optional source-video section for each effect and upload a full-body reference image. You may need to choose the source-video upload even if a trend card is selected. The local page contains the UI; there is no external inference/API call.

## What downloads externally?

At first build, Docker downloads open-source runtime code and Python packages. On the first model startup, the backend downloads the published model weights and the required detection/segmentation weights into a local cache. Those downloads are model files, not user photos or videos. Subsequent runs reuse the cached weights. Once everything is downloaded, rendering itself can run with the machine offline if all dependencies and weights are present.

This is self-hosted inference, not a model trained from scratch. The underlying pretrained Wan 2.2 Animate weights were released by the model authors. Reproducing their training process would require a large training dataset and far more compute. No user media is submitted to Hugging Face Spaces, an AI-rendering website, or an EffectLab cloud server in this local setup.

## API and security

- GET /health — model status and visible GPU.
- POST /api/jobs — local multipart input (reference photo, source clip, prompt, frame cap).
- GET /api/jobs/{job_id} — progress.
- GET /api/jobs/{job_id}/result — MP4 output.

The local launch script generates a random API key and binds the service to localhost. The key is only held in the current browser tab. Input files are removed from the temporary job directory after processing; generated MP4s stay in temporary local storage until you delete them or remove the Docker volume.

## Limitations

This code has passed Python/JavaScript syntax checks and is committed in the repository, but a real render has not been completed in this environment because it has no Docker runtime or GPU. The model pipeline may still need dependency or VRAM adjustments on a specific GPU. The GPU backend will reject startup if it cannot see a working NVIDIA GPU rather than downloading model weights onto an unsupported machine.
