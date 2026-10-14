# EffectLab model-training foundation

This folder is the start of a training workflow for an EffectLab-owned video transformation model. It does **not** claim a trained model exists yet.

## Why paired clips matter

For video appearance/character replacement, a useful training pair needs to show the same action and scene in both the input and desired output. Random unrelated videos are not interchangeable training examples. Use footage you own, have explicit permission to use, or a dataset whose licence permits both model training and your intended deployment. Do not scrape private or copyrighted clips without permission, and do not train on a real person's likeness without appropriate consent.

## Dataset layout

Create a local dataset like this (do not commit large video files to GitHub):

```text
dataset/
  pairs/
    000001/
      source.mp4
      target.mp4
      rights.txt
    000002/
      source.mp4
      target.mp4
      rights.txt
```

`source.mp4` is the original clip; `target.mp4` is the matching transformed clip. The videos should be temporally aligned and show the same motion/camera/background, with only the intended appearance change. `rights.txt` should describe the permission/licence and consent basis for each pair.

## Validate the pairs

Install Python 3.10+ and OpenCV, then run:

```bash
pip install opencv-python
python training/prepare_pairs.py --dataset ./dataset/pairs --output ./dataset/manifest.jsonl
```

The validator writes a JSONL manifest only for pairs that pass basic checks. It does not download videos, bypass website restrictions, train weights, or prove that a pair is legally usable. Review the rights notes yourself.

## Training strategy

1. Start with a permissively licensed open-source video transformation/character-replacement base model, after checking its code, checkpoint and dataset licences separately.
2. Build a small, high-quality, consented and temporally aligned dataset; split by source/person into train and validation sets so near-identical frames cannot leak into validation.
3. Fine-tune with a supported training framework such as [Hugging Face finetrainers](https://github.com/huggingface/finetrainers), using a model/task it actually supports. Do not assume every model can be fine-tuned by the same script.
4. Evaluate held-out clips for identity consistency, motion preservation, flicker, edge artefacts, background preservation and failure cases. Keep the previous checkpoint and compare before replacing it.
5. Only deploy a checkpoint after testing quality, safety, licensing and inference cost.

## Hardware reality

A browser website on GitHub Pages cannot train or run a large video model by itself. Training and inference need a machine with a compatible GPU or a separately hosted compute service. Some video-model fine-tuning recipes require high VRAM; for example, the finetrainers support matrix reports model-specific requirements ranging from around 5 GB for some LTX-Video LoRA training configurations to 18 GB for CogVideoX-5B LoRA, under specific settings. These are not guarantees for this project. See the current [finetrainers requirements](https://github.com/huggingface/finetrainers).

## Current status

- [x] Dataset structure and manifest validator
- [ ] Add licensed/consented paired clips
- [ ] Choose a base model after checking its licence and actual replacement capability
- [ ] Run a small training experiment on available GPU hardware
- [ ] Evaluate held-out clips and iterate
- [ ] Connect the resulting inference service to EffectLab

**No video-generation API is called by these scripts and there is no per-generation provider charge.** Local/hosted compute may still have a cost.