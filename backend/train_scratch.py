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
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from scratch_model import EffectLabScratchVideoAI

HEIGHT = 96
WIDTH = 64
CLIP_FRAMES = 4


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
        if not self.root.is_dir():
            raise RuntimeError(f"Training data folder does not exist: {self.root}")
        self.items: list[tuple[Path, Path, Path, int, int]] = []
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
            if dn >= 2 and tn >= 2:
                # Keep each duration separately; target frames are sampled across
                # its full timeline to tolerate small differences between takes.
                self.items.append((ref, driving, target, dn, tn))
        if not self.items:
            raise RuntimeError(
                f"No aligned swap sequences found in {self.root}. Every subfolder needs reference.jpg, driving.mp4 and target.mp4."
            )

    def __len__(self) -> int:
        return max(len(self.items), 1) * 8

    def __getitem__(self, index: int):
        ref_path, drive_path, target_path, driving_length, target_length = self.items[index % len(self.items)]
        ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref is None:
            raise RuntimeError(f"Could not read reference image: {ref_path}")
        ref = cv2.cvtColor(cv2.resize(ref, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)
        start = random.randint(0, max(0, driving_length - self.clip_frames))
        drive_cap = cv2.VideoCapture(str(drive_path))
        target_cap = cv2.VideoCapture(str(target_path))
        driving_frames, target_frames = [], []
        for frame_idx in range(start, min(start + self.clip_frames, driving_length)):
            target_idx = round(frame_idx * max(0, target_length - 1) / max(driving_length - 1, 1))
            frame_a = _read_frame(drive_cap, frame_idx)
            frame_b = _read_frame(target_cap, target_idx)
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
    parser.add_argument("--max-steps", type=int, default=0, help="Optional step limit for laptop experiments; does not make a production model.")
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--clip-frames", type=int, default=CLIP_FRAMES)
    parser.add_argument("--base-channels", type=int, default=8)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    global HEIGHT, WIDTH, CLIP_FRAMES
    if args.height < 32 or args.width < 32 or args.height % 8 or args.width % 8:
        parser.error("--height and --width must be at least 32 and divisible by 8")
    if args.clip_frames < 1 or args.clip_frames > 8:
        parser.error("--clip-frames must be between 1 and 8")
    if args.base_channels < 4 or args.base_channels > 32:
        parser.error("--base-channels must be between 4 and 32")
    HEIGHT, WIDTH, CLIP_FRAMES = args.height, args.width, args.clip_frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if device.type == "cuda":
        print("Training on", torch.cuda.get_device_name(0))
    else:
        print("CPU mode: using the compact low-resolution model. Training can still take a long time and quality is not guaranteed.")

    dataset = AlignedSwapDataset(args.data)
    print(f"Found {len(dataset.items)} aligned sequence(s); {len(dataset)} sampled clips per epoch.")
    print("Training from random initialisation. No pretrained weights or pretrained AI are loaded.")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    model = EffectLabScratchVideoAI(base_channels=args.base_channels).to(device)
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
            "model_name": "EffectLabScratchVideoAI-CompactV2",
            "model_state": model.state_dict(),
            "epoch": epoch,
            "steps": step,
            "mean_loss": mean_loss,
            "config": {"height": HEIGHT, "width": WIDTH, "clip_frames": CLIP_FRAMES, "base_channels": args.base_channels},
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
