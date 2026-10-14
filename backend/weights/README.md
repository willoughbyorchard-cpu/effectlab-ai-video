# Experimental EffectLab Scratch checkpoint

effectlab_scratch.pt is a checkpoint from EffectLab's own neural network, trained from random initialization. It does not use pretrained model weights or a hosted inference service.

- Model: EffectLabScratchVideoAI-CompactV2
- Recorded optimizer steps: **1,158**
- Checkpoint configuration: **64 × 112 pixels**, 2-frame clips, 4 base channels
- Device used: CPU
- Latest saved epoch training loss: 0.1413 (training metric only, not a quality score)
- Inference smoke test: 24 frames successfully rendered to a 64 × 112 MP4
- Selected checkpoint: chosen after comparing candidates, not simply because it had the most steps

## What was measured

On eight fixed sampled clips from the **same single demonstration sequence used for training**, the original 1,070-step checkpoint had combined pixel/temporal/edge loss 0.1466 and mean absolute pixel error 0.1293. The selected 1,158-step checkpoint measured 0.1386 and 0.1214 respectively (about 5.4% lower combined loss). These are same-scene measurements, not an independent test set and not evidence that arbitrary videos will work better.

## Important limitations

The initial 70 steps were a smoke test on unrelated dance clips. Training then used one short public demo source/result/reference triplet, followed by 88 additional optimizer updates and checkpoint comparisons. This is one scene, not a diverse dataset. The model can learn correlations specific to that scene and has **not** demonstrated reliable replacement of arbitrary people in arbitrary footage.

The active model remains low-resolution at 64 × 112 pixels. A higher-resolution training attempt at 128 × 224 was killed by the runtime's memory limit before saving a checkpoint, so no higher-resolution weights are included. Faces, hair, hands and clothing details may remain blurry or distorted. The renderer currently drops source audio. This is an experimental prototype, not a production-ready body-swap model.

The video/reference came from a public demo page, not a permissively licensed training corpus; redistribution and commercial-use rights for those demo assets have not been independently confirmed. Do not treat this checkpoint as cleared for commercial deployment. For a commercial product, replace those examples with content you own or are licensed to use, then train and evaluate on a broader, properly aligned dataset.

## How it loads

The local FastAPI backend reads this file from backend/weights/effectlab_scratch.pt when backend/run-local.ps1 or backend/run-local.sh starts. GitHub Pages serves static HTML and does not run model inference; actual generation still requires the local Python engine (or a separately deployed backend).
