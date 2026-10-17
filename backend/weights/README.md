# Experimental EffectLab Scratch checkpoint

The file effectlab_scratch.pt is a checkpoint from EffectLab's own neural network, trained from random initialization and fine-tuned without pretrained model weights or a hosted generator.

- Model: EffectLabScratchVideoAI-CompactV2
- Recorded optimizer steps: **1,270**
- Checkpoint configuration: **80 × 144 pixels**, 2-frame clips, 4 base channels
- Compute used for fine-tuning: CPU
- Latest saved epoch training loss: 0.1520 (training metric only, not a quality score)
- Inference smoke test: successfully rendered a short MP4 from the demonstration clip
- Checkpoint selection: chosen from several candidates using fixed sampled clips, not simply by picking the largest step count

## Evaluation performed

At the same 80 × 144 evaluation size and on eight fixed sampled clips from the **same single demonstration sequence used for training**, the published 1,158-step checkpoint measured combined pixel/temporal/edge loss 0.14627 and mean absolute pixel error (MAE) 0.12947. The selected 1,270-step checkpoint measured 0.14092 and 0.12417 respectively, reductions of about **3.7%** in combined loss and **4.1%** in MAE. These numbers are same-scene measurements—not an independent test set, user-study, or proof that the model generalises to arbitrary videos.

## Important limitations

This model learned from one short public demo source/result/reference triplet, initially after a small unrelated-video smoke test. It remains highly prone to scene-specific correlations and has **not** demonstrated reliable replacement of arbitrary people in arbitrary footage. The model's native output is still low resolution at 80 × 144 pixels; it may blur or distort faces, hands, hair, and clothing. Current rendering drops source audio. This is an experimental prototype, not a production-ready body-swap system.

A 128 × 224 training attempt exceeded the available memory limit; the current 80 × 144 checkpoint is the largest configuration successfully fine-tuned in this constrained environment.

The training video/reference came from a public demo page, not a permissively licensed training corpus; redistribution and commercial-use rights for those demo assets have not been independently confirmed. Do not treat this checkpoint as cleared for commercial deployment. For a commercial product, replace these examples with footage you own or have permission to use, then train and evaluate on a diverse, properly aligned dataset.

## How it loads

The FastAPI backend reads backend/weights/effectlab_scratch.pt automatically. GitHub Pages is a static frontend and does not run inference itself. The optional Render Blueprint at the repository root can host the Python model online; cloud mode requires an access token and disables dataset-training routes on the shared free service.
