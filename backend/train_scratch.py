"""Train EffectLab Scratch from random initialization using aligned swap examples.

Each training folder must contain:
  reference.jpg  - full-body photo of the person to render
  driving.mp4    - original video whose motion/background should be followed
  target.mp4     - desired replacement output, frame-aligned to driving.mp4

No pretrained checkpoints or AI feature extractors are loaded.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scratch_model import EffectLabScratchVideoAI

HEIGHT = 192
WIDTH = 128
CLIP_FRAMES = 8


def _read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


class AlignedSwapDataset(Dataset):
    def __init__(self, root: str, clip_frames: int = CLIP_FRAMES):
        self.root = Path(root)
        self.clip_frames = clip_frames
        self.items: list[tuple[Path, Path, Path, int]] = []
        for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
            ref = next((folder / n for n in ("reference.jpg", "reference.jpeg", "reference.png", "reference.webp") if (folder / n).is_file()), None)
            driving = next((folder / ("driving" + ext) for ext in (".mp4", ".mov", ".m4v", ".avi", ".webm") if (folder / ("driving" + ext)).is_file()), None)
            target = next((folder / ("target" + ext) for ext in (".mp4", ".mov", ".m4v", ".avi", ".webm") if (folder / ("target" + ext)).is_file()), None)
            if not ref or not driving or not target:
                continue
            dcap = cv2.VideoCapture(str(driving))
            tcap = cv2.VideoCapture(str(target))
            dn = int(dcap.get(cv2.CAP_PROP_FRAME_COUNT))
            tn = int(tcap.get(cv2.CAP_PROP_FRAME_COUNT))
            dcap.release()
            tcap.release()
            valid_frames = min(dn, tn)
            if valid_frames >= 2:
                self.items.append((ref, driving, target, valid_frames))
        if not self.items:
            raise RuntimeError(
                f"No aligned swap sequences found in {self.root}. Every subfolder needs reference.jpg, driving.mp4 and target.mp4."
            )

    def __len__(self) -> int:
        return max(len(self.items), 1) * 8

    def __getitem__(self, index: int):
        ref_path, drive_path, target_path, length = self.items[index % len(self.items)]
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref is None:
            raise RuntimeError(f"Could not read reference image: {ref_path}")
        ref = cv2.cvtColor(cv2.resize(ref, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)
        start = random.randint(0, max(0, length - self.clip_frames))
        drive_cap = cv2.VideoCapture(str(drive_path))
        target_cap = cv2.VideoCapture(str(target_path))
        driving_frames, target_frames = [], []
        for frame_idx in range(start, min(start + self.clip_frames, length)):
            frame_a = _read_frame(drive_cap, frame_idx)
            frame_b = _read_frame(target_cap, frame_idx)
            if frame_a is None or frame_b is None:
                break
            driving_frames.append(frame_a)
            target_frames.append(frame_b)
        drive_cap.release()
        target_cap.release()
        if not driving_frames:
            raise RuntimeError(f"Could not decode training clip in {drive_path}")
        while len(driving_frames) < self.clip_frames:
            driving_frames.append(driving_frames[-1].copy())
            target_frames.append(target_frames[-1].copy())

        def image_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)

        def clip_tensor(frames: list[np.ndarray]) -> torch.Tensor:
            stacked = torch.from_numpy(np.stack(frames).copy()).float().div_(255.0)
            return stacked.permute(3, 0, 1, 2).contiguous()

        return image_tensor(ref), clip_tensor(driving_frames), clip_tensor(target_frames)


def _gradient_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = predicted[..., :, 1:] - predicted[..., :, :-1]
    targ_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = predicted[..., 1:, :] - predicted[..., :-1, :]
    targ_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, targ_dx) + F.l1_loss(pred_dy, targ_dy)


def _loss(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pixel = F.l1_loss(predicted, target)
    if predicted.shape[2] > 1:
        pred_motion = predicted[:, :, 1:] - predicted[:, :, :-1]
        target_motion = target[:, :, 1:] - target[:, :, :-1]
        temporal = F.l1_loss(pred_motion, target_motion)
    else:
        temporal = pixel.new_zeros(())
    edges = _gradient_loss(predicted, target)
    total = pixel + 0.25 * temporal + 0.10 * edges
    return total, {
        "pixel": float(pixel.detach()),
        "temporal": float(temporal.detach()),
        "edges": float(edges.detach()),
        "total": float(total.detach()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EffectLab Scratch video AI from random initialisation.")
    parser.add_argument("--data", required=True, help="Folder containing aligned sequence subfolders.")
    parser.add_argument("--output", default="weights/effectlab_scratch.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional small smoke-train limit; does not make a production model.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Training on", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no CUDA GPU found. From-scratch video training on CPU will be extremely slow.")

    dataset = AlignedSwapDataset(args.data)
    print(f"Found {len(dataset.items)} aligned sequence(s); {len(dataset)} sampled clips per epoch.")
    print("Training from random initialisation. No pretrained weights or pretrained AI are loaded.")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    model = EffectLabScratchVideoAI().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=0.01)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    started = time.time()
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        samples = 0
        for reference, driving, target in loader:
            reference = reference.to(device, non_blocking=True)
            driving = driving.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(reference, driving)
            loss, parts = _loss(predicted, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}; check the source/target videos.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += parts["total"] * reference.shape[0]
            samples += reference.shape[0]
            step += 1
            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch}/{args.epochs} step={step} "
                    f"loss={parts['total']:.4f} pixel={parts['pixel']:.4f} "
                    f"temporal={parts['temporal']:.4f} edges={parts['edges']:.4f}",
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                break
        mean_loss = epoch_loss / max(samples, 1)
        print(f"epoch={epoch} mean_loss={mean_loss:.4f} elapsed_min={(time.time()-started)/60:.1f}", flush=True)
        checkpoint = {
            "model_name": "EffectLabScratchVideoAI",
            "model_state": model.state_dict(),
            "epoch": epoch,
            "steps": step,
            "mean_loss": mean_loss,
            "config": {"height": HEIGHT, "width": WIDTH, "clip_frames": CLIP_FRAMES, "base_channels": 32},
            "trained_from_random_initialisation": True,
            "pretrained_model": None,
        }
        temp_output = output.with_suffix(output.suffix + ".tmp")
        torch.save(checkpoint, temp_output)
        temp_output.replace(output)
        if args.max_steps and step >= args.max_steps:
            break

    print(f"Saved checkpoint to {output}")
    print("Note: a few smoke-test steps are not enough to learn realistic full-body replacement.")


if __name__ == "__main__":
    main()
