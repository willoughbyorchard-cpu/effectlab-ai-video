# EffectLab AI

EffectLab is a browser UI plus an experimental first-party video model. Its default Scratch inference path uses a custom neural network trained from random initialization. The provided cloud configuration does **not** set a third-party inference key.

## Try the website

- Website: https://willoughbyorchard-cpu.github.io/effectlab-ai-video/
- Training/data guide: open backend/README.md.
- Local option: install with backend/setup-laptop.ps1 (Windows) or backend/setup-laptop.sh (Mac/Linux), then run the matching local launcher.
- Cloud option: render.yaml defines a Render web service. Create a Blueprint from this repository in Render and supply a private EFFECTLAB_ACCESS_TOKEN when prompted. The website points at the intended hosted URL; once the service is deployed and healthy, paste the same token into the password field. The public website itself is static, while Render runs the Python model.

Render's shared free service may go idle after inactivity and need time to wake, but it does not require you to start a server on your laptop or iPad. The first cloud deployment still needs to be created in your hosting account, and it is not considered live until the service passes its health check.

## Current model and limits

The included checkpoint records 1,270 optimizer steps and outputs 80 × 144 pixels. In an eight-clip evaluation from the same single demo scene used for training, it improved the combined pixel/temporal/edge loss by about 3.7% versus the previous checkpoint at the same output size. That is a narrow same-scene measurement, **not** evidence of reliable swaps on arbitrary videos. Faces, clothing, and movement can remain blurry or distorted; the output drops audio. Object replacement is not implemented.

The model has so far trained on one short public demonstration source/result/reference triplet, and rights for that demo material have not been independently confirmed. Do not treat this checkpoint as cleared for commercial use. For meaningful generalisation, use a diverse set of properly aligned pairs that you own or are licensed to use.

## Optional legacy provider mode

The backend still contains an optional fal.ai API path for experiments. It runs only when FAL_KEY is explicitly configured as a server-side secret. The Render Blueprint described above does not set FAL_KEY and is designed to use the project's own Scratch checkpoint instead. Never put API secrets in browser JavaScript or commit them to GitHub.
