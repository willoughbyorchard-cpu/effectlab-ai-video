# EffectLab Scratch — laptop training and local inference

EffectLab Scratch is an original neural network trained from random weights. It does not load a pretrained generator and does not call a hosted AI renderer. The project now includes a compact, low-resolution configuration intended to make CPU training attempts possible on ordinary laptops. That is a capability test, not a promise of good replacement quality.

## Windows setup

1. Install Python 3.11 or 3.12 from python.org.
2. Open PowerShell in the backend folder.
3. Run Set-ExecutionPolicy -Scope Process Bypass and then .\setup-laptop.ps1.
4. Run .\run-local.ps1. Keep the terminal open and copy the private API key it prints.
5. Open http://127.0.0.1:7860/train.html in your browser and paste the API key.
6. Select a dataset ZIP or the full training-data folder, upload it, then start with 1 epoch and 20 steps.

No NVIDIA GPU or Docker is required by the native laptop path. The setup uses the CPU build of PyTorch. Mac/Linux users can run bash setup-laptop.sh, then bash run-local.sh.

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

The Training page can upload one ZIP, a directory selection, or multiple individual files using sequence subfolders. Uploaded media is sent to the local engine URL configured on the page and saved under backend/training-data/; it is not uploaded to GitHub Pages. The files and model checkpoint are excluded from Git.

## CPU-friendly model

The initial laptop configuration uses 64 by 96 pixel frames, four-frame clips and eight base channels. This is intentionally small to reduce memory use on a 16 GB laptop. Training can still be slow. The network is trained from random initialization and may produce distorted frames, learn colour correlations, or fail to replace the identity at all. Ten examples can exercise the pipeline, but cannot guarantee a functional general-purpose swap model. A good general-purpose video model from scratch requires far more data, compute and architecture development.

The training page defaults to 1 epoch and 20 steps to keep the first attempt bounded. Increase those settings only after the first run completes without memory or thermal issues. The saved checkpoint is backend/weights/effectlab_scratch.pt.

## Manual command

After installing dependencies and adding data, the following runs the same small experiment:

    python train_scratch.py --data training-data --output weights/effectlab_scratch.pt --epochs 1 --max-steps 20 --height 96 --width 64 --clip-frames 4 --base-channels 8

The app reloads the checkpoint after training completes. Inference supports person replacement only; object replacement is not implemented. The model outputs low-resolution video and drops audio in the current renderer.
