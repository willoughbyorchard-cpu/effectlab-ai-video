"""EffectLab Scratch local inference server.

This backend uses EffectLabScratchVideoAI, a model defined in scratch_model.py
and initialised from random weights. It downloads no model weights and calls no
hosted AI inference API. It refuses to generate until a trained checkpoint
created by train_scratch.py is present.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import tempfile
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from scratch_model import EffectLabScratchVideoAI

APP_TITLE = "EffectLab Scratch — local video AI"
BASE_DIR = Path(__file__).resolve().parent
STATIC_PAGE = BASE_DIR / "static" / "index.html"
CHECKPOINT = Path(os.environ.get("EFFECTLAB_CHECKPOINT", str(BASE_DIR / "weights" / "effectlab_scratch.pt")))
WORK_ROOT = Path(os.environ.get("EFFECTLAB_WORK_ROOT", str(Path(tempfile.gettempdir()) / "effectlab")))
TRAINING_ROOT = Path(os.environ.get("EFFECTLAB_TRAINING_ROOT", str(BASE_DIR / "training-data")))
TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_HEIGHT = 96
MODEL_WIDTH = 64
CLIP_FRAMES = 4
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 80 * 1024 * 1024
MAX_VIDEO_SECONDS = float(os.environ.get("EFFECTLAB_MAX_VIDEO_SECONDS", "8"))
MAX_FRAMES = 150
MAX_JOBS_PER_HOUR = 4

app = FastAPI(title=APP_TITLE, version="0.4.0")
_origins = [
    part.strip()
    for part in os.environ.get(
        "EFFECTLAB_ALLOWED_ORIGINS",
        "http://127.0.0.1:7860,http://localhost:7860,https://willoughbyorchard-cpu.github.io",
    ).split(",")
    if part.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL: EffectLabScratchVideoAI | None = None
STATE: dict[str, Any] = {
    "ready": False,
    "loading": True,
    "error": None,
    "checkpoint": str(CHECKPOINT),
    "device": str(DEVICE),
}
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.RLock()
RATE_LOCK = threading.Lock()
REQUESTS_BY_IP: dict[str, list[float]] = {}
GPU_QUEUE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="effectlab-scratch")
TRAINING_QUEUE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="effectlab-training")
TRAINING_JOBS: dict[str, dict[str, Any]] = {}
TRAINING_JOBS_LOCK = threading.RLock()
MAX_TRAINING_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_TRAINING_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
MAX_TRAINING_FILES = 1000


def _require_auth(authorization: str | None) -> None:
    """No user-entered key is needed for the loopback-only local engine."""
    # The provided launch scripts bind uvicorn to 127.0.0.1 only. Browser-origin
    # requests are separately restricted to EFFECTLAB_ALLOWED_ORIGINS below.
    # Do not expose this development server directly to a public network.
    return


@app.middleware("http")
async def restrict_browser_origins(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin and origin not in _origins:
        return JSONResponse(status_code=403, content={"detail": "Origin not allowed."})
    return await call_next(request)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with RATE_LOCK:
        recent = [stamp for stamp in REQUESTS_BY_IP.get(ip, []) if now - stamp < 3600]
        if len(recent) >= MAX_JOBS_PER_HOUR:
            raise HTTPException(status_code=429, detail="Local render limit reached. Try again later.")
        recent.append(now)
        REQUESTS_BY_IP[ip] = recent


def _job_update(
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    error: str | None = None,
    result: str | None = None,
) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = max(0.0, min(1.0, float(progress)))
        if message is not None:
            job["message"] = message
        if error is not None:
            job["error"] = error[:1500]
        if result is not None:
            job["result_path"] = result
        job["updated_at"] = time.time()


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "status_url": "/api/jobs/" + job["id"],
        "result_url": "/api/jobs/" + job["id"] + "/result" if job["status"] == "complete" else None,
    }


def _load_model() -> None:
    global MODEL, MODEL_HEIGHT, MODEL_WIDTH, CLIP_FRAMES
    try:
        if not CHECKPOINT.is_file():
            raise RuntimeError(
                "EffectLab Scratch has no trained model checkpoint yet. "
                "No outside AI is used: first create a paired dataset and run train_scratch.py. "
                "The network will not return random noise and call it a result."
            )
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != "EffectLabScratchVideoAI-CompactV2":
            raise RuntimeError("Checkpoint is not an EffectLab Scratch model.")
        if checkpoint.get("pretrained_model") is not None or not checkpoint.get("trained_from_random_initialisation"):
            raise RuntimeError("Only a Scratch checkpoint trained from random initialization is accepted.")
        config = checkpoint.get("config", {})
        MODEL_HEIGHT = int(config.get("height", 96))
        MODEL_WIDTH = int(config.get("width", 64))
        CLIP_FRAMES = int(config.get("clip_frames", 4))
        base_channels = int(config.get("base_channels", 8))
        model = EffectLabScratchVideoAI(base_channels=base_channels)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(DEVICE)
        model.eval()
        MODEL = model
        STATE.update(
            ready=True,
            loading=False,
            error=None,
            checkpoint=str(CHECKPOINT),
            device=str(DEVICE),
            trained_steps=int(checkpoint.get("steps", 0)),
            trained_epochs=int(checkpoint.get("epoch", 0)),
            training_loss=checkpoint.get("mean_loss"),
        )
        print(
            "[EffectLab Scratch] loaded first-party checkpoint:",
            f"steps={STATE['trained_steps']}, epochs={STATE['trained_epochs']}, device={DEVICE}",
            flush=True,
        )
    except Exception as exc:
        STATE.update(ready=False, loading=False, error=f"{type(exc).__name__}: {exc}")
        print("[EffectLab Scratch] model not ready:", STATE["error"], flush=True)


@app.on_event("startup")
def on_startup() -> None:
    cv2.setNumThreads(1)
    threading.Thread(target=_load_model, daemon=True, name="scratch-model-loader").start()


@app.get("/", include_in_schema=False)
def serve_local_app() -> FileResponse:
    if not STATIC_PAGE.is_file():
        raise HTTPException(status_code=404, detail="Local EffectLab interface not found at backend/static/index.html.")
    return FileResponse(STATIC_PAGE, media_type="text/html; charset=utf-8")


@app.get("/train.html", include_in_schema=False)
def serve_training_page() -> FileResponse:
    page = BASE_DIR / "static" / "train.html"
    if not page.is_file():
        raise HTTPException(status_code=404, detail="Training page was not found in backend/static.")
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "ready": STATE["ready"],
        "loading": STATE["loading"],
        "error": STATE["error"] if not STATE["ready"] else None,
        "device": STATE["device"],
        "checkpoint": STATE["checkpoint"],
        "trained_steps": STATE.get("trained_steps", 0),
        "trained_epochs": STATE.get("trained_epochs", 0),
        "training_loss": STATE.get("training_loss"),
        "pretrained_ai_used": False,
        "external_inference_used": False,
        "model_height": MODEL_HEIGHT,
        "model_width": MODEL_WIDTH,
        "clip_frames": CLIP_FRAMES,
    }


def _read_video(path: Path, frame_cap: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("OpenCV could not open the uploaded source video.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if not np.isfinite(fps) or fps < 1 or fps > 120:
        fps = 24.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        cap.release()
        raise RuntimeError("The source video contains no readable frames.")
    duration = count / fps
    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise RuntimeError(f"Keep source videos at or under {MAX_VIDEO_SECONDS:g} seconds.")
    if count > frame_cap:
        # Sample evenly to stay inside the configured inference budget.
        indices = np.linspace(0, count - 1, frame_cap).round().astype(int).tolist()
        frames: list[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA))
        if frames and duration > 0:
            fps = len(frames) / duration
    else:
        frames = []
        while len(frames) < frame_cap:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError("No decodable frames were found in the source video.")
    return frames, fps


def _reference_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Reference image could not be decoded.")
    rgb = cv2.cvtColor(cv2.resize(bgr, (MODEL_WIDTH, MODEL_HEIGHT)), cv2.COLOR_BGR2RGB)
    array = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _render_video(job_id: str, reference_path: Path, source_path: Path, frame_cap: int) -> None:
    job_dir = WORK_ROOT / job_id
    result_path = job_dir / "result.mp4"
    try:
        if MODEL is None or not STATE["ready"]:
            raise RuntimeError(STATE.get("error") or "The Scratch model checkpoint is not loaded.")
        _job_update(job_id, status="processing", progress=0.03, message="Decoding source clip locally")
        frames, fps = _read_video(source_path, frame_cap)
        reference = _reference_to_tensor(reference_path, DEVICE)
        outputs: list[np.ndarray] = []
        total = len(frames)
        for offset in range(0, total, CLIP_FRAMES):
            chunk = frames[offset:offset + CLIP_FRAMES]
            actual = len(chunk)
            while len(chunk) < CLIP_FRAMES:
                chunk.append(chunk[-1].copy())
            array = np.stack(chunk).astype(np.float32) / 255.0
            driving = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                prediction = MODEL(reference, driving)
            prediction = prediction[0, :, :actual].permute(1, 2, 3, 0).detach().cpu().numpy()
            prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)
            outputs.extend([frame for frame in prediction])
            pct = 0.05 + 0.85 * min(1.0, len(outputs) / total)
            _job_update(
                job_id,
                progress=pct,
                message=f"EffectLab Scratch is generating frames locally ({min(len(outputs), total)}/{total})",
            )
        _job_update(job_id, progress=0.93, message="Encoding the local MP4")
        writer = cv2.VideoWriter(
            str(result_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (MODEL_WIDTH, MODEL_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not start the MP4 encoder. Check OpenCV/FFmpeg installation.")
        try:
            for rgb in outputs:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
        if not result_path.is_file() or result_path.stat().st_size < 1000:
            raise RuntimeError("The video encoder did not produce a valid output.")
        _job_update(
            job_id, status="complete", progress=1.0,
            message="Finished by the locally trained EffectLab Scratch network",
            result=str(result_path),
        )
    except Exception as exc:
        _job_update(job_id, status="failed", message="Local generation failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        for path in (reference_path, source_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/jobs")
async def create_job(
    request: Request,
    image: UploadFile = File(...),
    video: UploadFile = File(...),
    frame_cap: int = Form(120),
    prompt: str = Form("Replace the full person while retaining the source motion and background."),
    negative_prompt: str = Form(""),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    _rate_limit(request)
    if not STATE["ready"] or MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.get("error") or "The locally trained EffectLab Scratch checkpoint is not ready.",
        )
    with JOBS_LOCK:
        active = sum(1 for job in JOBS.values() if job["status"] in {"queued", "processing"})
        if active >= 2:
            raise HTTPException(status_code=429, detail="The local GPU is busy. Try again when the current render finishes.")
    frame_cap = max(CLIP_FRAMES, min(int(frame_cap), MAX_FRAMES))
    image_name = Path(image.filename or "reference.jpg").name
    suffix = Path(image_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    video_name = Path(video.filename or "source.mp4").name
    video_suffix = Path(video_name).suffix.lower()
    if video_suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload an MP4, MOV, AVI or WebM source video.")

    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    video_bytes = await video.read(MAX_VIDEO_BYTES + 1)
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Reference image must be non-empty and smaller than 20 MB.")
    if not video_bytes or len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Source video must be non-empty and smaller than 80 MB.")
    decoded_reference = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if decoded_reference is None:
        raise HTTPException(status_code=415, detail="Reference file is not a supported image.")

    job_id = uuid.uuid4().hex
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    image_path = job_dir / ("reference" + suffix)
    video_path = job_dir / ("source" + video_suffix)
    image_path.write_bytes(image_bytes)
    video_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(video_path))
    opened = cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if not opened or count <= 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=415, detail="Source video could not be decoded.")
    duration = count / (fps if fps > 0 else 24.0)
    if duration > MAX_VIDEO_SECONDS:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"Keep source videos at or under {MAX_VIDEO_SECONDS:g} seconds.")

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Waiting for the local model",
        "error": None,
        "result_path": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    GPU_QUEUE.submit(_render_video, job_id, image_path, video_path, frame_cap)
    return _public_job(job)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Local render job not found.")
        return _public_job(job)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, authorization: str | None = Header(default=None)) -> FileResponse:
    _require_auth(authorization)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Local render job not found.")
        if job["status"] != "complete" or not job.get("result_path"):
            raise HTTPException(status_code=409, detail="The local AI has not finished rendering this video.")
        path = Path(job["result_path"])
    if not path.is_file():
        raise HTTPException(status_code=410, detail="The local output file has expired.")
    return FileResponse(path, media_type="video/mp4", filename="effectlab-scratch-full-body.mp4")

# ---------------------- local training page/API ----------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def _has_triplet(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    names = {p.name.lower() for p in folder.iterdir() if p.is_file()}
    has_ref = any("reference" + ext in names for ext in IMAGE_EXTENSIONS)
    has_driving = any("driving" + ext in names for ext in VIDEO_EXTENSIONS)
    has_target = any("target" + ext in names for ext in VIDEO_EXTENSIONS)
    return has_ref and has_driving and has_target


def _complete_sequence_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((child for child in root.iterdir() if child.is_dir() and _has_triplet(child)), key=lambda p: p.name)


def _safe_relative_path(raw: str) -> Path:
    normalised = (raw or "").replace("\\", "/")
    rel = PurePosixPath(normalised)
    if not normalised or rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("Unsafe or empty upload path.")
    if any(":" in part for part in rel.parts):
        raise ValueError("Drive-qualified paths are not allowed in dataset uploads.")
    return Path(*rel.parts)


def _group_flat_sequence_files(root: Path) -> None:
    pattern = re.compile(r"^(?P<sequence>.+?)[_-](?P<kind>reference|driving|target)(?P<ext>\.[^.]+)$", re.IGNORECASE)
    for item in list(root.iterdir()) if root.is_dir() else []:
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        kind = match.group("kind").lower()
        extension = match.group("ext").lower()
        if kind == "reference" and extension not in IMAGE_EXTENSIONS:
            continue
        if kind in {"driving", "target"} and extension not in VIDEO_EXTENSIONS:
            continue
        seq_name = re.sub(r"[^A-Za-z0-9_-]+", "_", match.group("sequence")).strip("_") or "sequence"
        seq_dir = root / seq_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        destination = seq_dir / (kind + extension)
        if not destination.exists():
            shutil.move(str(item), str(destination))


def _find_dataset_root(payload_root: Path) -> Path | None:
    _group_flat_sequence_files(payload_root)
    if _complete_sequence_dirs(payload_root):
        return payload_root
    if _has_triplet(payload_root):
        seq = payload_root / "sequence_0001"
        seq.mkdir(parents=True, exist_ok=True)
        for item in list(payload_root.iterdir()):
            if item == seq:
                continue
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                kind = item.stem.lower()
                if kind in {"reference", "driving", "target"}:
                    shutil.move(str(item), str(seq / item.name))
        return payload_root if _complete_sequence_dirs(payload_root) else None
    for current, dirs, _files in os.walk(payload_root):
        candidate = Path(current)
        if _complete_sequence_dirs(candidate):
            return candidate
    return None


def _public_training_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "error": job.get("error"),
        "logs": list(job.get("logs", []))[-45:],
        "sequences": job.get("sequences", 0),
        "updated_at": job.get("updated_at", job.get("created_at")),
    }


def _training_job_update(job_id: str, **values: Any) -> None:
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            return
        log_line = values.pop("log_line", None)
        if log_line:
            job.setdefault("logs", []).append(str(log_line)[:1000])
            job["logs"] = job["logs"][-100:]
        job.update(values)
        job["updated_at"] = time.time()


@app.get("/api/training/dataset")
def training_dataset_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    sequences = _complete_sequence_dirs(TRAINING_ROOT)
    return {
        "sequences": len(sequences),
        "names": [p.name for p in sequences],
        "dataset_root": str(TRAINING_ROOT),
        "local_only": True,
    }


@app.post("/api/training/upload")
async def upload_training_dataset(
    files: list[UploadFile] = File(default=[]),
    dataset_zip: UploadFile | None = File(default=None),
    paths: str = Form(default=""),
    replace_existing: bool = Form(default=False),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    if not files and dataset_zip is None:
        raise HTTPException(status_code=400, detail="Select a training-data folder, files or a ZIP.")
    if len(files) > MAX_TRAINING_FILES:
        raise HTTPException(status_code=413, detail=f"Select no more than {MAX_TRAINING_FILES} files at once.")
    temp_root = WORK_ROOT / ("training-upload-" + uuid.uuid4().hex)
    temp_root.mkdir(parents=True, exist_ok=False)
    payload_roots: list[Path] = []
    total_bytes = 0
    files_written = 0
    try:
        if dataset_zip is not None:
            zip_path = temp_root / "dataset.zip"
            with zip_path.open("wb") as out:
                while True:
                    chunk = await dataset_zip.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_TRAINING_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Training upload exceeds 2 GB.")
                    out.write(chunk)
            if not zipfile.is_zipfile(zip_path):
                raise HTTPException(status_code=415, detail="The selected dataset file is not a valid ZIP archive.")
            unpacked = temp_root / "unzipped"
            unpacked.mkdir()
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    infos = [info for info in archive.infolist() if not info.is_dir()]
                    if len(infos) > MAX_TRAINING_FILES:
                        raise HTTPException(status_code=413, detail="ZIP contains too many files.")
                    expanded = sum(max(0, info.file_size) for info in infos)
                    if expanded > MAX_TRAINING_EXPANDED_BYTES:
                        raise HTTPException(status_code=413, detail="Expanded ZIP data exceeds 4 GB.")
                    for info in infos:
                        relative = _safe_relative_path(info.filename)
                        if stat.S_ISLNK(info.external_attr >> 16):
                            raise HTTPException(status_code=400, detail="ZIP symlinks are not allowed.")
                        dest = unpacked / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info, "r") as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                        files_written += 1
            except (zipfile.BadZipFile, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"Could not safely extract dataset ZIP: {exc}") from exc
            payload_roots.append(unpacked)

        if files:
            try:
                path_list = json.loads(paths) if paths else []
                if not isinstance(path_list, list):
                    path_list = []
            except json.JSONDecodeError:
                path_list = []
            files_root = temp_root / "selected-files"
            files_root.mkdir()
            seen: set[str] = set()
            for idx, uploaded in enumerate(files):
                raw_name = path_list[idx] if idx < len(path_list) and isinstance(path_list[idx], str) else (uploaded.filename or "")
                try:
                    relative = _safe_relative_path(raw_name)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid upload file path: {exc}") from exc
                rel_key = relative.as_posix().lower()
                if rel_key in seen:
                    raise HTTPException(status_code=400, detail="Duplicate relative file paths detected in the upload.")
                seen.add(rel_key)
                dest = files_root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("wb") as out:
                    while True:
                        chunk = await uploaded.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > MAX_TRAINING_UPLOAD_BYTES:
                            raise HTTPException(status_code=413, detail="Training upload exceeds 2 GB.")
                        out.write(chunk)
                files_written += 1
            payload_roots.append(files_root)

        dataset_sources: list[Path] = []
        for payload_root in payload_roots:
            found = _find_dataset_root(payload_root)
            if found:
                dataset_sources.append(found)
        if not dataset_sources:
            raise HTTPException(
                status_code=400,
                detail="No complete sequences found. Each sequence folder needs reference.jpg, driving.mp4 and target.mp4. ZIPs or folder uploads must preserve sequence subfolders.",
            )

        if replace_existing:
            for child in TRAINING_ROOT.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
        for source_root in dataset_sources:
            for seq in _complete_sequence_dirs(source_root):
                shutil.copytree(seq, TRAINING_ROOT / seq.name, dirs_exist_ok=True)

        sequences = _complete_sequence_dirs(TRAINING_ROOT)
        if not sequences:
            raise HTTPException(status_code=400, detail="Uploaded files did not create any complete training sequences.")
        return {
            "ok": True,
            "sequences": len(sequences),
            "files_written": files_written,
            "sequence_names": [seq.name for seq in sequences],
            "dataset_root": str(TRAINING_ROOT),
            "message": "Training files were saved on this computer, ready for audit and training.",
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _run_training_job(job_id: str, epochs: int, max_steps: int) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        _training_job_update(job_id, status="processing", progress=0.01, message="Auditing videos, frame counts and timing", log_line="Starting local dataset audit.")
        audit_dir = WORK_ROOT / "training-audit"
        audit = subprocess.run(
            [sys.executable, str(BASE_DIR / "prepare_data.py"), "--data", str(TRAINING_ROOT), "--report", str(audit_dir), "--min-valid-sequences", "1"],
            cwd=str(BASE_DIR), env=env, capture_output=True, text=True, timeout=900,
        )
        for line in (audit.stdout + "\n" + audit.stderr).splitlines():
            if line.strip():
                _training_job_update(job_id, log_line=line.strip())
        if audit.returncode != 0:
            raise RuntimeError("Training-data audit failed. Review the log: clips must be synchronised and target.mp4 must show the replacement.")
        sequence_count = len(_complete_sequence_dirs(TRAINING_ROOT))
        _training_job_update(job_id, progress=0.08, message=f"Dataset audit passed for {sequence_count} sequence(s). Starting CPU-friendly training.", sequences=sequence_count)
        command = [
            sys.executable, str(BASE_DIR / "train_scratch.py"),
            "--data", str(TRAINING_ROOT), "--output", str(CHECKPOINT),
            "--epochs", str(epochs), "--batch-size", "1", "--max-steps", str(max_steps),
            "--height", "96", "--width", "64", "--clip-frames", "4", "--base-channels", "8",
        ]
        proc = subprocess.Popen(
            command, cwd=str(BASE_DIR), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                clean = line.rstrip()
                if clean:
                    step_match = re.search(r"\bstep=(\d+)", clean)
                    progress = None
                    if step_match:
                        progress = min(0.94, 0.08 + 0.84 * int(step_match.group(1)) / max(max_steps, 1))
                    _training_job_update(job_id, status="processing", message=clean[:500], progress=progress, log_line=clean)
        code = proc.wait()
        if code != 0:
            raise RuntimeError(f"Training process exited with code {code}. Inspect the training log above.")
        if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size < 1000:
            raise RuntimeError("Training exited without producing a model checkpoint.")
        _training_job_update(job_id, status="processing", progress=0.96, message="Training finished. Loading the new checkpoint into EffectLab.")
        _load_model()
        if not STATE.get("ready"):
            raise RuntimeError("Checkpoint was saved but the local renderer could not load it: " + str(STATE.get("error")))
        _training_job_update(job_id, status="complete", progress=1.0, message="Training run finished and the checkpoint is loaded. Test a swap; quality is not guaranteed.", log_line=f"Saved checkpoint: {CHECKPOINT}")
    except Exception as exc:
        _training_job_update(job_id, status="failed", message="Training failed", error=f"{type(exc).__name__}: {exc}", log_line=f"ERROR: {type(exc).__name__}: {exc}")


@app.post("/api/training/start")
def start_training(
    epochs: int = Form(1),
    max_steps: int = Form(20),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    if epochs < 1 or epochs > 20:
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 20.")
    if max_steps < 1 or max_steps > 2000:
        raise HTTPException(status_code=400, detail="Maximum steps must be between 1 and 2000.")
    sequences = _complete_sequence_dirs(TRAINING_ROOT)
    if not sequences:
        raise HTTPException(status_code=400, detail="Upload a complete training dataset first.")
    with TRAINING_JOBS_LOCK:
        if any(job["status"] in {"queued", "processing"} for job in TRAINING_JOBS.values()):
            raise HTTPException(status_code=409, detail="A training run is already active.")
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id, "status": "queued", "progress": 0.0,
            "message": "Queued on this computer", "error": None,
            "logs": [], "sequences": len(sequences),
            "created_at": time.time(), "updated_at": time.time(),
        }
        TRAINING_JOBS[job_id] = job
    TRAINING_QUEUE.submit(_run_training_job, job_id, epochs, max_steps)
    return _public_training_job(job)


@app.get("/api/training/jobs/{job_id}")
def training_job_status(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Local training job not found.")
        return _public_training_job(job)


