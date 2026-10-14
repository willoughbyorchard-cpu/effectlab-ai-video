# EffectLab paired-video training experiment

## What this experiment does

This is a reproducible **research prototype** for fine-tuning the public SANA-Streaming bidirectional video-to-video editing model on aligned source/edited clips. It is not a trained EffectLab model, and it is not a dedicated body-swap model. The public example set includes 1,000 pairs for background changes, local additions, local removals and local changes.

## Dataset and licence

Dataset: https://huggingface.co/datasets/Efficient-Large-Model/SANA-Streaming-example-training-dataset

**Non-commercial research only.** The dataset and derivatives must not be used in a paid EffectLab service unless separate rights are obtained from the relevant rights holders. Do not use it to launch or evaluate a commercial product. Use a separate dataset with explicit commercial permissions for that.

## Hardware reality

The published fine-tuning recipe uses `torchrun --nproc_per_node=8` and is intended for an 8-GPU setup. A GitHub Pages website cannot train the model. This launcher checks for eight NVIDIA GPUs and stops rather than pretending that a laptop or free single-GPU notebook can run the published recipe as-is. It downloads roughly 1.13 GB of example data plus model checkpoints, and training will need substantial additional storage and time.

## Run

From a Linux machine with NVIDIA drivers, a CUDA-compatible PyTorch environment and eight visible GPUs:

```bash
chmod +x training/run_sana_experiment.sh
ALLOW_NONCOMMERCIAL_RESEARCH=YES ./training/run_sana_experiment.sh
```

The script clones the official NVlabs/Sana repository into `effectlab-sana-work`, downloads the dataset and required checkpoints, verifies dataset checksums, then launches the published bidirectional fine-tuning recipe. The original EffectLab web app is not modified by this experiment.

## What it does not do

- It does not scrape YouTube or assume a video is free to train on just because it is public.
- It does not train a foundation model from scratch.
- It does not yet produce whole-body character replacement; this dataset is for general video editing, not a matched identity-replacement dataset.
- It does not deploy trained weights to the public website. That requires evaluation, model-serving infrastructure, and checking the licenses of the model, data, and output use.

For a character-replacement model, collect paired source/target clips with permission for model training and intended deployment, or use a pretrained model whose licence explicitly permits the intended use.