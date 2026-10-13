# EffectLab AI

EffectLab is a browser UI plus a self-hosted AI video experiment. The current model implementation is EffectLab Scratch: an original temporal neural network defined in backend/scratch_model.py, with training in backend/train_scratch.py. It does not download or call a pretrained video-generation model.

## Start here

- Read backend/README.md first. It explains the paired training dataset and hardware needs.
- Use backend/train-model.sh to train from random initialisation on aligned replacement videos.
- Use backend/run-local.sh to run the local UI and inference API.

Important: the repository does not contain a trained checkpoint. Full-body replacement will not work until the model is trained on aligned driving/target examples. The code deliberately rejects missing weights instead of returning random output and pretending it is a successful video.

The locally bundled UI can run at http://127.0.0.1:7860. Source photos/videos are processed by that local service, not sent to a hosted AI inference website.
