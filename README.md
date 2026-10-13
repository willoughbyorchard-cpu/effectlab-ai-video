# EffectLab AI

EffectLab is a browser UI plus a self-hosted AI video experiment. It contains an original neural network, trained from random initialisation, rather than a downloaded pretrained video generator.

## Start here

- Open backend/README.md for laptop setup and dataset instructions.
- The main website is deployed from index.html; train.html is the training and dataset-upload page.
- On Windows, run .\\backend\\setup-laptop.ps1, then .\\backend\\run-local.ps1 from PowerShell.
- On Mac/Linux, run bash backend/setup-laptop.sh, then bash backend/run-local.sh.
- Visit http://127.0.0.1:7860/train.html to upload training examples to your local engine and start training.

The public GitHub Pages site is static and cannot train on GitHub's servers. The training page connects to the Python engine running on the user's own laptop. Files and the trained checkpoint stay on that computer. No pretrained model weights or third-party inference API are used.

## Important limitations

A compact CPU-friendly training path is implemented, but the repository does not include a trained checkpoint. The model starts from random weights and the first training runs are experiments. A completed training run does not guarantee a meaningful person replacement. The current inference path supports person replacement only, at low resolution, without preserving audio. Object replacement is not implemented.
