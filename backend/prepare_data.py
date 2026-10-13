"""Audit aligned driving/target video pairs for EffectLab Scratch.

This tool uses ordinary image/video I/O only. It does not use an AI model.
For each sequence it verifies that reference.jpg, driving.mp4 and target.mp4
exist, compares timing metadata, and writes a side-by-side contact sheet so
the alignment and replacement can be checked by a human before training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REQUIRED_REFERENCE_NAMES = ("reference.jpg", "reference.jpeg", "reference.png", "reference.webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".webm")


def first_file(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def video_info(path: Path) -> dict[str, float | int | bool | str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return {"path": str(path), "opened": False, "frames": 0, "fps": 0.0, "duration": 0.0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    return {
        "path": str(path), "opened": True, "frames": frames, "fps": fps,
        "duration": duration, "width": width, "height": height,
    }


def sample_frame(path: Path, fraction: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count < 1:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, max(0, round((count - 1) * fraction))))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.resize(frame, (320, 480), interpolation=cv2.INTER_AREA)


def labeled(frame: np.ndarray | None, label: str) -> np.ndarray:
    if frame is None:
        frame = np.zeros((480, 320, 3), dtype=np.uint8)
        cv2.putText(frame, "READ ERROR", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (319, 40), (18, 18, 18), -1)
    cv2.putText(out, label[:32], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_contact_sheet(sequence: Path, driving: Path, target: Path, output: Path) -> None:
    parts: list[np.ndarray] = []
    for fraction, caption in ((0.05, "5%"), (0.5, "50%"), (0.95, "95%")):
        a = labeled(sample_frame(driving, fraction), "DRIVING " + caption)
        b = labeled(sample_frame(target, fraction), "TARGET " + caption)
        row = np.concatenate([a, b], axis=1)
        parts.append(row)
    sheet = np.concatenate(parts, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)




def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_sequence(folder: Path, report_dir: Path) -> dict:
    reference = first_file(folder, REQUIRED_REFERENCE_NAMES)
    driving = first_file(folder, tuple("driving" + ext for ext in VIDEO_EXTENSIONS))
    target = first_file(folder, tuple("target" + ext for ext in VIDEO_EXTENSIONS))
    record: dict = {
        "sequence": folder.name,
        "folder": str(folder),
        "ok": False,
        "errors": [],
        "warnings": [],
    }
    if reference is None:
        record["errors"].append("Missing reference.jpg (or reference.jpeg/png/webp).")
    if driving is None:
        record["errors"].append("Missing driving video (driving.mp4 recommended).")
    if target is None:
        record["errors"].append("Missing target video (target.mp4 recommended).")
    if record["errors"]:
        return record

    ref = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    if ref is None or ref.size == 0:
        record["errors"].append("Reference image could not be decoded.")
    else:
        record["reference"] = {
            "file": str(reference), "width": int(ref.shape[1]), "height": int(ref.shape[0]),
            "bytes": int(reference.stat().st_size),
        }
        if min(ref.shape[:2]) < 160:
            record["warnings"].append("Reference image is very small; use a clear head-to-toe photo.")

    d = video_info(driving)
    t = video_info(target)
    record["driving"] = d
    record["target"] = t
    if sha256_file(driving) == sha256_file(target):
        record["errors"].append(
            "Driving and target files are byte-for-byte identical. A copy of the source teaches reconstruction, not full-body replacement."
        )
    if not d["opened"] or not t["opened"]:
        record["errors"].append("At least one video could not be opened.")
        return record
    if d["frames"] < 8 or t["frames"] < 8:
        record["errors"].append("Each video needs at least eight decodable frames.")
    if abs(int(d["frames"]) - int(t["frames"])) > 1:
        record["errors"].append(
            f"Frame counts differ ({d['frames']} vs {t['frames']}). Align the target to the driving video."
        )
    fps_delta = abs(float(d["fps"]) - float(t["fps"]))
    if fps_delta > 0.2:
        record["errors"].append(
            f"Frame rates differ ({d['fps']:.3f} vs {t['fps']:.3f}). Use the same fps and timing."
        )
    duration_delta = abs(float(d["duration"]) - float(t["duration"]))
    if duration_delta > 0.08:
        record["errors"].append(
            f"Durations differ ({d['duration']:.3f}s vs {t['duration']:.3f}s). Synchronise both videos."
        )
    if d["frames"] > 3000 or t["frames"] > 3000:
        record["warnings"].append("Long training sequence; consider splitting it into shorter aligned clips.")
    if d.get("width", 0) < 128 or d.get("height", 0) < 192:
        record["warnings"].append("Driving video is low resolution.")
    report_image = report_dir / (folder.name + "-alignment.jpg")
    try:
        build_contact_sheet(folder, driving, target, report_image)
        record["contact_sheet"] = str(report_image)
    except Exception as exc:
        record["warnings"].append("Could not create contact sheet: " + str(exc))
    record["ok"] = not record["errors"]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and preview aligned EffectLab Scratch training pairs.")
    parser.add_argument("--data", required=True, help="Root directory with one subfolder per sequence.")
    parser.add_argument("--report", default="training-audit", help="Output folder for JSON/CSV/contact sheets.")
    parser.add_argument("--min-valid-sequences", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.data).resolve()
    report = Path(args.report).resolve()
    if not root.is_dir():
        parser.error(f"Data folder does not exist: {root}")
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
    if not sequences:
        parser.error(f"No sequence folders found inside {root}")
    report.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    results = [inspect_sequence(folder, report) for folder in sequences]
    valid = [result for result in results if result["ok"]]
    payload = {
        "root": str(root), "sequence_count": len(results),
        "valid_count": len(valid), "invalid_count": len(results) - len(valid),
        "model_training_ready": len(valid) >= args.min_valid_sequences,
        "sequences": results,
        "note": "This checks basic file/timing alignment only. Open each contact sheet and visually confirm the person, action and camera are frame-aligned. It does not train the network.",
    }
    (report / "audit.json").write_text(json.dumps(payload, indent=2))
    columns = ["sequence", "ok", "errors", "warnings", "contact_sheet", "driving_frames", "target_frames", "duration"]
    with (report / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "sequence": result["sequence"],
                "ok": result["ok"],
                "errors": " | ".join(result["errors"]),
                "warnings": " | ".join(result["warnings"]),
                "contact_sheet": result.get("contact_sheet", ""),
                "driving_frames": result.get("driving", {}).get("frames", ""),
                "target_frames": result.get("target", {}).get("frames", ""),
                "duration": result.get("driving", {}).get("duration", ""),
            })
    print(json.dumps({k: payload[k] for k in ("root", "sequence_count", "valid_count", "invalid_count", "model_training_ready")}, indent=2))
    print("Audit report:", report / "audit.json")
    print("Open every *-alignment.jpg and verify alignment before training.")
    if len(valid) < args.min_valid_sequences:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
