# Experimental EffectLab Scratch checkpoint

`effectlab_scratch.pt` is a real checkpoint from EffectLab's own neural network, trained from random initialization (no pretrained generator and no hosted inference service).

- Model: `EffectLabScratchVideoAI-CompactV2`
- Recorded optimizer steps: 1,070
- Checkpoint configuration: 112 × 64 pixels, 2-frame clips, 4 base channels
- Device: CPU
- Most recent epoch mean training loss: about 0.106 (training metric only)
- Demonstrated result: a very low-resolution rough replacement on the one short source clip used in training; it is not a general-purpose body-swap model

## Important limitations

The first 70 steps were a small smoke test on unrelated dance clips. The next 1,000 steps fine-tuned that checkpoint on one public demo source/result/reference triplet. This is one short scene, not a diverse dataset. It has learned strong correlations specific to that scene; it has not demonstrated reliable replacement of an arbitrary person in arbitrary footage. At 64 × 112 output resolution faces, hair, hands and clothing details remain poor. Current renderer drops source audio.

The video/reference came from a public demo page, not a permissively licensed training corpus; redistribution and commercial-use rights for the demo assets have not been independently confirmed. Do not treat this checkpoint as cleared for commercial deployment. For a commercial product, replace these examples with content you own or are licensed to use, then train and evaluate on a broader, properly aligned dataset.

## How it loads

The local FastAPI backend automatically reads this file from `backend/weights/effectlab_scratch.pt` when `backend/run-local.ps1` or `backend/run-local.sh` starts. The browser site hosted by GitHub Pages is static: this file does not cause inference to run on GitHub Pages. The site still needs the local Python engine running on your computer (or a separately deployed GPU/CPU backend).
