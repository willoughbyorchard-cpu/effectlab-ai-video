# EffectLab Full-Body Engine

This folder contains the GPU backend that performs actual full-body character replacement for the EffectLab website. It tracks the source person's full-body pose, facial motion and mask, uses the reference character to condition Wan 2.2 Animate, and writes a generated MP4. It does not paste a cropped face on top of the source.

The public website runs on GitHub Pages, which cannot host a 14B video model. The backend runs separately on a GPU host.

## Hardware and setup

- Linux with Docker and NVIDIA Container Toolkit; launch with --gpus all.
- A GPU with 24 GB VRAM or more is a sensible starting target for this quantised model pipeline; exact needs vary and startup can still fail on incompatible drivers or dependencies.
- Several tens of GB of free disk for model downloads and cache.
- A public HTTPS address if the GitHub Pages website must connect to it.
- A long private EFFECTLAB_API_KEY; never commit this key into the website or repository.

The first start downloads model weights. This may take a considerable time. The weights are not included in the repository.

## Build

From the backend directory:

~~~bash
docker build -t effectlab-engine .
~~~

Run on the GPU host with a private key:

~~~bash
docker run --gpus all --rm -p 7860:7860 \
  -e EFFECTLAB_API_KEY='replace-with-a-long-random-secret' \
  -e EFFECTLAB_ALLOWED_ORIGINS='https://willoughbyorchard-cpu.github.io' \
  -v effectlab-hf-cache:/root/.cache/huggingface \
  effectlab-engine
~~~

Put the API behind an HTTPS reverse proxy before connecting the public website. Do not disable API key checks on an internet-facing GPU host.

## Connect to EffectLab

Open a template and expand Connect your own EffectLab GPU engine. Enter the server's HTTPS base URL and the same key configured as EFFECTLAB_API_KEY. The URL is kept in local browser storage; the key is kept in the current tab's session storage. Leaving the URL blank continues to use the shared public renderer fallback.

API:

- GET /health reports model loading and GPU readiness.
- POST /api/jobs accepts multipart image, video, frame_cap, prompt, and negative_prompt; requires Authorization: Bearer key.
- GET /api/jobs/{job_id} reports progress.
- GET /api/jobs/{job_id}/result returns the MP4.

The defaults limit videos to 12 seconds, the render cap to 150 frames, upload sizes to 100 MB video and 24 MB image, and accepted jobs per source IP to four per hour. Jobs are serialised on a single GPU worker.

## What this is — and is not

This service runs an inference pipeline around released Wan 2.2 Animate weights and ComfyUI's pose, face, video, and segmentation nodes. It is not a foundation model trained from scratch. Full-body video generation is computationally heavy. This repository does not include a GPU allocation, a deployed server, or private hosting credentials.

Source uploads are removed from the job folder after processing; generated results remain in temporary job storage until the container/volume is cleared. Only submit images and footage you have permission to process.
