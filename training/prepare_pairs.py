#!/usr/bin/env python3
"""Validate aligned source/target video pairs and write a JSONL manifest.

Expected layout: DATASET/<pair-id>/source.mp4, target.mp4, rights.txt
This is dataset preparation only; it does not train a model or download footage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Missing dependency: pip install opencv-python", file=sys.stderr)
    raise SystemExit(2)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def inspect_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"missing/invalid video metadata: {path}")
        return {
            "path": str(path), "fps": round(fps, 4), "frames": frames,
            "width": width, "height": height,
            "duration_seconds": round(frames / fps, 4),
        }
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Folder containing one directory per pair")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL manifest path")
    parser.add_argument("--duration-tolerance", type=float, default=0.15, help="Maximum source/target duration difference in seconds")
    parser.add_argument("--fps-tolerance", type=float, default=0.1, help="Maximum source/target FPS difference")
    args = parser.parse_args()

    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        print(f"Dataset directory does not exist: {root}", file=sys.stderr)
        return 2

    records = []
    errors = []
    pair_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not pair_dirs:
        print(f"No pair directories found in {root}", file=sys.stderr)
        return 2

    for pair_dir in pair_dirs:
        source_candidates = [p for p in pair_dir.iterdir() if p.is_file() and p.stem == "source" and p.suffix.lower() in VIDEO_EXTENSIONS]
        target_candidates = [p for p in pair_dir.iterdir() if p.is_file() and p.stem == "target" and p.suffix.lower() in VIDEO_EXTENSIONS]
        rights_path = pair_dir / "rights.txt"
        try:
            if len(source_candidates) != 1 or len(target_candidates) != 1:
                raise ValueError("expected exactly one source video and one target video")
            if not rights_path.is_file() or not rights_path.read_text(encoding="utf-8").strip():
                raise ValueError("missing/empty rights.txt; record permission/licence and consent basis")
            source = inspect_video(source_candidates[0])
            target = inspect_video(target_candidates[0])
            if abs(source["duration_seconds"] - target["duration_seconds"]) > args.duration_tolerance:
                raise ValueError("source and target durations differ too much; align the clips first")
            if abs(source["fps"] - target["fps"]) > args.fps_tolerance:
                raise ValueError("source and target FPS differ too much; standardise the frame rate first")
            records.append({
                "pair_id": pair_dir.name,
                "source": str(source_candidates[0].resolve()),
                "target": str(target_candidates[0].resolve()),
                "source_metadata": source,
                "target_metadata": target,
                "rights_note_file": str(rights_path.resolve()),
                "review_required": True,
            })
        except (OSError, ValueError) as exc:
            errors.append({"pair_id": pair_dir.name, "error": str(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Valid pairs: {len(records)} / {len(pair_dirs)}")
    print(f"Manifest: {args.output.resolve()}")
    if errors:
        print("Rejected pairs:", file=sys.stderr)
        for item in errors:
            print(f"- {item['pair_id']}: {item['error']}", file=sys.stderr)
    print("Note: metadata checks cannot prove temporal alignment, consent, or licence compliance.")
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())