"""EffectLab Scratch local inference server.

This backend uses EffectLabScratchVideoAI, a model defined in scratch_model.py
and initialised from random weights. It downloads no model weights and calls no
hosted AI inference API. It refuses to generate until a trained checkpoint
created by train_scratch.py is present.
"""
from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from scratch_model import EffectLabScratchVideoAI

APP_TITLE = "EffectLab Scratch — local video AI"
BASE_DIR = Path(__file__).resolve().parent
STATIC_PAGE = BASE_DIR / "static" / "index.html"
CHECKPOINT = Path(os.environ.get("EFFECTLAB_CHECKPOINT", "/app/weights/effectlab_scratch.pt"))
WORK_ROOT = Path(os.environ.get("EFFECTLAB_WORK_ROOT", "/tmp/effectlab"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_HEIGHT = 192
MODEL_WIDTH = 128
CLIP_FRAMES = 8
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 80 * 1024 * 1024
MAX_VIDEO_SECONDS = float(os.environ.get("EFFECTLAB_MAX_VIDEO_SECONDS", "8"))
MAX_FRAMES = 150
MAX_JOBS_PER_HOUR = 4

app = FastAPI(title=APP_TITLE, version="0.3.0")
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


def _require_auth(authorization: str | None) -> None:
    secret = os.environ.get("EFFECTLAB_API_KEY", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Local engine API key is not configured.")
    given = ""
    if authorization and authorization.lower().startswith("bearer "):
        given = authorization[7:].strip()
    if not given or not secrets.compare_digest(secret, given):
        raise HTTPException(status_code=401, detail="Wrong or missing local EffectLab API key.")


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
    global MODEL
    try:
        if not CHECKPOINT.is_file():
            raise RuntimeError(
                "EffectLab Scratch has no trained model checkpoint yet. "
                "No outside AI is used: first create a paired dataset and run train_scratch.py. "
                "The network will not return random noise and call it a result."
            )
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != "EffectLabScratchVideoAI":
            raise RuntimeError("Checkpoint is not an EffectLab Scratch model.")
        if checkpoint.get("pretrained_model") is not None or not checkpoint.get("trained_from_random_initialisation"):
            raise RuntimeError("Only a Scratch checkpoint trained from random initialization is accepted.")
        model = EffectLabScratchVideoAI()
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
