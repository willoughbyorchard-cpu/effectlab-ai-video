# EffectLab Scratch — original AI video model

EffectLab Scratch is a first-party, train-from-random-initialisation video network. It does not load Wan, ComfyUI, Hugging Face checkpoints, pretrained feature extractors, or hosted AI inference APIs. The model architecture is in scratch_model.py; training code is in train_scratch.py; local inference and its API are in app.py.

IMPORTANT: this repository contains model code, not a trained model checkpoint. It cannot perform convincing full-body replacement until trained on suitable examples. Random weights are not a working AI.

## What the network does

- Encodes the entire reference photo into trainable appearance features.
- Encodes a short source-video clip using 3D convolutions to learn motion and scene context.
- Fuses reference appearance features into the temporal network at multiple resolutions.
- Decodes a replacement video clip, using temporal and image losses during training.

Every AI layer starts from random initialisation. The inference server checks checkpoint metadata and refuses to render if there is no trained EffectLab Scratch checkpoint.

## The hard part is the training data

To teach a model to replace the person rather than paste a face on top, training requires aligned examples of what a correct replacement looks like. Each training sequence folder needs:

~~~text
training-data/
  sequence_0001/
    reference.jpg   # full-body photo of the identity to render
    driving.mp4     # source person's movement and scene
    target.mp4      # desired person in that movement, aligned frame-for-frame
  sequence_0002/
    reference.jpg
    driving.mp4
    target.mp4
~~~

driving.mp4 and target.mp4 must show the same action and camera timing; the target video must show the intended replacement identity and be a real training label, not a renamed copy of the source. Include many identities, clothes, body shapes, camera angles, lighting conditions and movements to generalise. One or two examples only test the code; they do not train a reliable AI. You must have permission to use all training footage.

## Train your own checkpoint

Training is strongly recommended on a CUDA-capable NVIDIA GPU. CPU training is technically supported by PyTorch but is impractically slow for this video model.

1. Put aligned sequence folders in backend/training-data/, or set the DATA_DIR environment variable to your dataset path.
2. Open a terminal in backend/ and run ./train-model.sh.
3. Set EPOCHS=100 BATCH_SIZE=1 ./train-model.sh to change training settings. Training starts from random weights and saves effectlab_scratch.pt in the Docker volume effectlab-model-weights.
4. After training, run ./run-local.sh and open http://127.0.0.1:7860.
5. The launch script prints a private API key; paste it into the local page's engine settings. Check engine health before generating.

Training command implemented by the script:

~~~bash
docker run --rm --gpus all -v /path/to/training-data:/data:ro -v effectlab-model-weights:/app/weights effectlab-scratch python /app/train_scratch.py --data /data --output /app/weights/effectlab_scratch.pt --epochs 100 --batch-size 1
~~~

## Local generation

./run-local.sh builds the app image and serves the bundled UI and API at http://127.0.0.1:7860. The server binds to localhost only. It does not upload photos/videos to any public rendering website or external AI API. It loads only the checkpoint you trained yourself.

At build time Docker downloads the base runtime and ordinary software packages (PyTorch is the tensor/training framework, not a pretrained generator). No AI model weights are downloaded. After software dependencies and your model are available locally, inference itself works offline.

Inputs are limited to a 20 MB reference image, 80 MB source video, 8 seconds duration and 150 frames. Model resolution is 128×192 in this prototype, so output detail is low; it is not a finished commercial-quality system. Improve resolution/architecture and train on a substantial dataset before expecting convincing performance.

## API

- GET /health reports whether a trained checkpoint is loaded.
- POST /api/jobs accepts a reference image plus driving video.
- GET /api/jobs/{job_id} reports inference progress.
- GET /api/jobs/{job_id}/result returns the generated MP4.

Only localhost is exposed by the default launcher. The service uses a random API key and stores temporary render files locally.

## Current status

The original architecture, training loop, local inference API, local frontend and launch scripts are implemented. Python/JavaScript/shell syntax checks pass; the model smoke tests in backend/tests check tensor shapes, output range and gradient flow. A trained checkpoint and real full-body test video are not included because no aligned training dataset or GPU was available in this coding session. This is an honest first-party AI training project, not yet a working trained character replacement model.
