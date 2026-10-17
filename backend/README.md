# EffectLab Scratch — laptop training and local inference

EffectLab Scratch is an original neural network trained from random weights. It does not load a pretrained generator and does not call a hosted AI renderer. The project now includes a compact, low-resolution configuration intended to make CPU training attempts possible on ordinary laptops. That is a capability test, not a promise of good replacement quality.

## Windows setup

1. Install Python 3.11 or 3.12 from python.org.
2. Open PowerShell in the backend folder.
3. Run Set-ExecutionPolicy -Scope Process Bypass and then .\setup-laptop.ps1.
4. Run .\run-local.ps1 and keep the terminal open. No API key is required.
5. Open http://127.0.0.1:7860/train.html in your browser; the training page connects without a key.
6. Select a dataset ZIP or the full training-data folder, upload it, then start with 1 epoch and 20 steps.

No NVIDIA GPU or Docker is required by the native laptop path. The setup uses the CPU build of PyTorch. Mac/Linux users can run bash setup-laptop.sh, then bash run-local.sh. Both launchers bind the engine to 127.0.0.1; do not expose this development server directly to the internet.

## Expected data layout

Each training sequence folder must contain all three items:

    training-data/
      sequence_0001/
        reference.jpg
        driving.mp4
        target.mp4
      sequence_0002/
        reference.jpg
        driving.mp4
        target.mp4

reference.jpg is the identity photo, driving.mp4 is the original footage, and target.mp4 is the desired replacement already completed. Driving and target clips should show the same movement, background and camera position, with similar duration. Exact frame counts are not required: training maps target frames across relative clip progress. The automatic audit compares timing metadata and makes contact sheets, but a person must still inspect that the actions and camera framing genuinely match.

The Training page can upload one ZIP, a directory selection, or multiple individual files using sequence subfolders. Uploaded media is sent to the local engine URL configured on the page and saved under backend/training-data/; it is not uploaded to GitHub Pages. Training videos stay local and are excluded from Git. This repository now includes one explicitly experimental starter checkpoint in `backend/weights/effectlab_scratch.pt`; do not confuse it with a production-quality model.

## CPU-friendly model

The initial laptop configuration uses 64 by 96 pixel frames, four-frame clips and eight base channels. This is intentionally small to reduce memory use on a 16 GB laptop. Training can still be slow. The network is trained from random initialization and may produce distorted frames, learn colour correlations, or fail to replace the identity at all. Ten examples can exercise the pipeline, but cannot guarantee a functional general-purpose swap model. A good general-purpose video model from scratch requires far more data, compute and architecture development.

The training page defaults to 1 epoch and 20 steps to keep the first attempt bounded. Increase those settings only after the first run completes without memory or thermal issues. The saved checkpoint is `backend/weights/effectlab_scratch.pt`. A 1,270-step experimental checkpoint is included so the renderer can be tested immediately; the checkpoint documentation in `backend/weights/README.md` describes how it was trained and its limits.

## Manual command

After installing dependencies and adding data, the following runs the same small experiment:

    python train_scratch.py --data training-data --output weights/effectlab_scratch.pt --epochs 1 --max-steps 20 --height 96 --width 64 --clip-frames 4 --base-channels 8

The app reloads the checkpoint after training completes. Inference supports person replacement only; object replacement is not implemented. The model outputs low-resolution video and drops audio in the current renderer.


## Included experimental body-swap checkpoint

The repository includes a first-party checkpoint so the backend can load and render a rough demonstration without the user first running training. The selected checkpoint has 1,270 recorded optimizer steps and uses one short demo scene at 80 × 144 pixels. Evaluation on eight sampled clips from that same scene improved over the earlier checkpoint, but this is not a general-purpose or production-ready person replacement system. Review weights/README.md before use. GitHub Pages remains a static frontend and does not run model inference itself.


## Continuing a saved checkpoint

train_scratch.py supports --resume to continue from existing weights rather than resetting the model. The AdamW optimizer starts fresh, so use a smaller learning rate when fine-tuning. For example:

    python train_scratch.py --data training-data --resume weights/effectlab_scratch.pt --output weights/effectlab_scratch_next.pt --epochs 100 --max-steps 20 --height 112 --width 64 --clip-frames 2 --base-channels 4 --learning-rate 0.000003

When --resume is used, --max-steps is the number of additional updates. The optional --gradient-checkpointing flag trades speed for lower activation memory. In a 1 GB environment, increasing output dimensions substantially may still exceed memory limits.


## Optional cloud deployment

A Render Blueprint is provided at the repository root in render.yaml. In Render, create a new Blueprint from this GitHub repository and use the values it reads from that file. Supply a long random secret when asked for EFFECTLAB_ACCESS_TOKEN; do not place the real token in GitHub. The public website's hosted-engine field should point at the resulting Render URL, and the access token goes into the password field. The cloud backend requires the token for render-job creation/status/download, restricts browser origins, limits render requests, and disables the memory-heavy training API. The shared free instance may take time to wake after inactivity; it does not require this computer or iPad to start a local server.

This is an inference-only deployment path for the custom Scratch model, not fal.ai. It runs on CPU and may be slow or fail if the host's free memory limit is exceeded. It has not been declared deployed until the Render service is created, builds successfully, and passes its health check.
