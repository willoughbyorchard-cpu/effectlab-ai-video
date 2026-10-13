# EffectLab's own API service for full-body replacement.
# The generative weights are open models; this file owns the request, tracking,
# segmentation, sampling, output, and job lifecycle around them.
from __future__ import annotations

import asyncio
import os
import random
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

# ComfyUI must configure its allocator before torch is imported.
COMFYUI_PATH = Path(os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")).resolve()
os.environ.setdefault("COMFYUI_PATH", str(COMFYUI_PATH))
if "--disable-cuda-malloc" not in sys.argv:
    sys.argv.append("--disable-cuda-malloc")

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from PIL import Image

APP_TITLE = "EffectLab Full-Body Engine"
WORK_ROOT = Path(os.environ.get("EFFECTLAB_WORK_ROOT", "/tmp/effectlab"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_POSITIVE_PROMPT = "Replace the main person with the full character from the reference image, including the entire body and outfit. Preserve source movement, camera, background, scene lighting and framing. One clear full-body character."
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
MAX_IMAGE_BYTES = 24 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_DURATION_SECONDS = float(os.environ.get("EFFECTLAB_MAX_VIDEO_SECONDS", "12"))
MAX_QUEUED_JOBS = int(os.environ.get("EFFECTLAB_MAX_QUEUED_JOBS", "3"))
JOBS_PER_HOUR = int(os.environ.get("EFFECTLAB_JOBS_PER_HOUR", "4"))

PIPELINE: dict[str, Any] = {}
STATE: dict[str, Any] = {"ready": False, "loading": True, "error": None}
JOBS: dict[str, dict[str, Any]] = {}
JOB_LOCK = threading.RLock()
RATE_LOCK = threading.Lock()
REQUESTS_BY_IP: dict[str, list[float]] = {}
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="effectlab-gpu")

app = FastAPI(title=APP_TITLE, version="0.2.0")
_allowed_origins = [
    value.strip()
    for value in os.environ.get(
        "EFFECTLAB_ALLOWED_ORIGINS",
        "https://willoughbyorchard-cpu.github.io,http://localhost:8000,http://127.0.0.1:5500",
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def get_value_at_index(obj: Sequence | Mapping, index: int) -> Any:
    try:
        return obj[index]
    except (KeyError, TypeError, IndexError):
        if isinstance(obj, Mapping) and "result" in obj:
            return obj["result"][index]
        raise


def _require_auth(authorization: str | None) -> None:
    expected = os.environ.get("EFFECTLAB_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="The GPU server is not configured: set EFFECTLAB_API_KEY before accepting render jobs.",
        )
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing EffectLab engine API key.")


def _update_job(job_id: str, *, status: str | None = None,
                progress: float | None = None, message: str | None = None,
                error: str | None = None, result_path: str | None = None) -> None:
    with JOB_LOCK:
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
        if result_path is not None:
            job["result_path"] = result_path
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
        "status_url": f"/api/jobs/{job['id']}",
        "result_url": f"/api/jobs/{job['id']}/result" if job["status"] == "complete" else None,
    }


def _check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    with RATE_LOCK:
        recent = [t for t in REQUESTS_BY_IP.get(ip, []) if now - t < 3600]
        if len(recent) >= JOBS_PER_HOUR:
            raise HTTPException(
                status_code=429,
                detail=f"Render limit reached ({JOBS_PER_HOUR} jobs per IP per hour). Try again later.",
            )
        recent.append(now)
        REQUESTS_BY_IP[ip] = recent


def _download_model(repo_id: str, filename: str, target_dir: str,
                    target_name: str | None = None) -> str:
    target_name = target_name or os.path.basename(filename)
    target = Path(target_dir) / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return str(target)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    try:
        target.symlink_to(cached)
    except OSError:
        shutil.copyfile(cached, target)
    return str(target)


def _download_models() -> None:
    models = COMFYUI_PATH / "models"
    files = [
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("Kijai/WanVideo_comfy_fp8_scaled", "Wan22Animate/Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors", "diffusion_models"),
        ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors", "loras"),
        ("Kijai/sam2-safetensors", "sam2.1_hiera_base_plus.safetensors", "sam2"),
        ("Wan-AI/Wan2.2-Animate-14B", "process_checkpoint/det/yolov10m.onnx", "detection"),
        ("JunkyByte/easy_ViTPose", "onnx/wholebody/vitpose-l-wholebody.onnx", "detection"),
    ]
    for repo, name, folder in files:
        _download_model(repo, name, str(models / folder))


def _bootstrap_comfyui_runtime() -> None:
    if not COMFYUI_PATH.is_dir():
        raise RuntimeError(
            f"ComfyUI was not found at {COMFYUI_PATH}. Build the supplied Docker image or set COMFYUI_PATH."
        )
    sys.path.insert(0, str(COMFYUI_PATH))
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import cuda_malloc
    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ["OCL_SET_SVM_SIZE"] = "262144"


def _import_custom_nodes() -> None:
    sys.path.insert(0, str(COMFYUI_PATH))
    import execution
    import server
    from nodes import init_extra_nodes
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    loop.run_until_complete(init_extra_nodes())
    loop.close()


def _initialize_pipeline() -> None:
    if PIPELINE:
        return
    _bootstrap_comfyui_runtime()
    _import_custom_nodes()
    from nodes import (
        CLIPLoader, CLIPTextEncode, CLIPVisionEncode, CLIPVisionLoader,
        ImageScale, KSampler, LoadImage, LoraLoaderModelOnly,
        NODE_CLASS_MAPPINGS, VAEDecode, VAELoader, UNETLoader,
    )
    import torch

    clip = CLIPLoader().load_clip(
        clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        type="wan", device="default",
    )
    clip_vision = CLIPVisionLoader().load_clip(clip_name="clip_vision_h.safetensors")
    vae = VAELoader().load_vae(vae_name="wan_2.1_vae.safetensors")
    unet = UNETLoader().load_unet(
        unet_name="Wan2_2-Animate-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        weight_dtype="default",
    )
    lora = LoraLoaderModelOnly()
    model = lora.load_lora_model_only(
        lora_name="lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        strength_model=1, model=get_value_at_index(unet, 0),
    )
    model = lora.load_lora_model_only(
        lora_name="WanAnimate_relight_lora_fp16.safetensors",
        strength_model=1, model=get_value_at_index(model, 0),
    )
    model = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]().patch(
        shift=8, model=get_value_at_index(model, 0)
    )
    sam2_model = NODE_CLASS_MAPPINGS["DownloadAndLoadSAM2Model"]().loadmodel(
        model="sam2.1_hiera_base_plus.safetensors",
        segmentor="video", device="cuda", precision="fp16",
    )
    detection_model = NODE_CLASS_MAPPINGS["OnnxDetectionModelLoader"]().loadmodel(
        vitpose_model="vitpose-l-wholebody.onnx",
        yolo_model="yolov10m.onnx", onnx_device="CUDAExecutionProvider",
    )
    PIPELINE.update({
        "clip": clip, "clip_vision": clip_vision, "vae": vae, "model": model,
        "sam2_model": sam2_model, "detection_model": detection_model, "torch": torch,
        "node_classes": {
            "LoadImage": LoadImage, "CLIPTextEncode": CLIPTextEncode,
            "CLIPVisionEncode": CLIPVisionEncode, "ImageScale": ImageScale,
            "KSampler": KSampler, "VAEDecode": VAEDecode,
            "NODE_CLASS_MAPPINGS": NODE_CLASS_MAPPINGS,
        },
    })


def _prepared(node: Any) -> Any:
    # Several current ComfyUI nodes require graph executor hidden context.
    return type(node).PREPARE_CLASS_CLONE(None)


def _run_generation(job_id: str, image_path: str, video_path: str,
                    frame_cap: int, prompt: str, negative_prompt: str) -> None:
    staged_image = None
    job_dir = WORK_ROOT / job_id
    try:
        if not STATE["ready"]:
            raise RuntimeError("The model is not ready yet. Check /health and retry.")
        import folder_paths
        nc = PIPELINE["node_classes"]
        mapping = nc["NODE_CLASS_MAPPINGS"]
        node = lambda name: mapping[name]()

        loadimage = nc["LoadImage"]()
        cliptextencode = nc["CLIPTextEncode"]()
        clipvisionencode = nc["CLIPVisionEncode"]()
        imagescale = nc["ImageScale"]()
        ksampler = nc["KSampler"]()
        vaedecode = nc["VAEDecode"]()
        vhs = node("VHS_LoadVideoPath")
        getimagesize = node("GetImageSize")
        poseface = node("PoseAndFaceDetection")
        drawpose = node("DrawViTPose")
        samseg = node("Sam2Segmentation")
        growmask = node("GrowMaskWithBlur")
        blockmask = node("BlockifyMask")
        drawmask = node("DrawMaskOnImage")
        animate = node("WanAnimateToVideo")
        trimlatent = node("TrimVideoLatent")
        imagebatch = node("ImageFromBatch")
        createvideo = node("CreateVideo")
        savevideo = node("SaveVideo")

        width, height = 480, 832
        _update_job(job_id, status="processing", progress=0.04, message="Preparing the reference image and source video")
        input_dir = Path(folder_paths.get_input_directory())
        input_dir.mkdir(parents=True, exist_ok=True)
        staged_image = input_dir / f"effectlab_{job_id}{Path(image_path).suffix or '.jpg'}"
        shutil.copyfile(image_path, staged_image)

        with PIPELINE["torch"].inference_mode():
            image_result = loadimage.load_image(image=staged_image.name)
            video_result = vhs.load_video(
                video=video_path, force_rate=0, custom_width=width, custom_height=height,
                frame_load_cap=frame_cap, skip_first_frames=0, select_every_nth=1,
                format="AnimateDiff", unique_id=random.randint(1, 2**63),
            )
            _update_job(job_id, progress=0.12, message="Encoding character and motion prompts")
            positive = cliptextencode.encode(text=prompt or DEFAULT_POSITIVE_PROMPT, clip=get_value_at_index(PIPELINE["clip"], 0))
            negative = cliptextencode.encode(text=negative_prompt or DEFAULT_NEGATIVE_PROMPT, clip=get_value_at_index(PIPELINE["clip"], 0))
            reference_scaled = imagescale.upscale(
                upscale_method="lanczos", width=width, height=height, crop="center",
                image=get_value_at_index(image_result, 0),
            )
            reference_vision = clipvisionencode.encode(
                crop="none", clip_vision=get_value_at_index(PIPELINE["clip_vision"], 0),
                image=get_value_at_index(reference_scaled, 0),
            )
            video_scaled = imagescale.upscale(
                upscale_method="lanczos", width=width, height=height, crop="center",
                image=get_value_at_index(video_result, 0),
            )
            video_size = _prepared(getimagesize).EXECUTE_NORMALIZED(image=get_value_at_index(video_scaled, 0))
            _update_job(job_id, progress=0.23, message="Tracking the original person, face, and body pose")
            pose_face_result = poseface.process(
                width=get_value_at_index(video_size, 0), height=get_value_at_index(video_size, 1),
                face_padding=0, model=get_value_at_index(PIPELINE["detection_model"], 0),
                images=get_value_at_index(video_scaled, 0),
            )
            pose_drawn = drawpose.process(
                width=get_value_at_index(video_size, 0), height=get_value_at_index(video_size, 1),
                retarget_padding=16, body_stick_width=-1, hand_stick_width=-1,
                draw_head="True", pose_data=get_value_at_index(pose_face_result, 0),
            )
            _update_job(job_id, progress=0.34, message="Building a temporal full-body mask")
            segmentation = samseg.segment(
                keep_model_loaded=True, individual_objects=False,
                sam2_model=get_value_at_index(PIPELINE["sam2_model"], 0),
                image=get_value_at_index(video_scaled, 0),
                bboxes=get_value_at_index(pose_face_result, 3),
            )
            mask = growmask.expand_mask(
                expand=10, incremental_expandrate=0, tapered_corners=True,
                flip_input=False, blur_radius=0, lerp_alpha=1, decay_factor=1,
                fill_holes=False, mask=get_value_at_index(segmentation, 0),
            )
            mask = blockmask.process(block_size=32, device="cpu", masks=get_value_at_index(mask, 0))
            background = drawmask.apply(
                color="0, 0, 0", device="cpu",
                image=get_value_at_index(video_scaled, 0), mask=get_value_at_index(mask, 0),
            )
            _update_job(job_id, progress=0.42, message="Preparing Wan Animate full-body conditioning")
            conditioning = _prepared(animate).EXECUTE_NORMALIZED(
                width=width, height=height, length=get_value_at_index(video_result, 1),
                batch_size=1, continue_motion_max_frames=5, video_frame_offset=0,
                positive=get_value_at_index(positive, 0), negative=get_value_at_index(negative, 0),
                vae=get_value_at_index(PIPELINE["vae"], 0),
                clip_vision_output=get_value_at_index(reference_vision, 0),
                reference_image=get_value_at_index(reference_scaled, 0),
                face_video=get_value_at_index(pose_face_result, 1),
                pose_video=get_value_at_index(pose_drawn, 0),
                background_video=get_value_at_index(background, 0),
                character_mask=get_value_at_index(mask, 0),
            )
            _update_job(job_id, progress=0.50, message="Generating the complete replacement character — this is the slow stage")
            sampled = ksampler.sample(
                seed=random.randint(1, 2**63), steps=6, cfg=1, sampler_name="euler",
                scheduler="simple", denoise=1, model=get_value_at_index(PIPELINE["model"], 0),
                positive=get_value_at_index(conditioning, 0),
                negative=get_value_at_index(conditioning, 1),
                latent_image=get_value_at_index(conditioning, 2),
            )
            trimmed = _prepared(trimlatent).EXECUTE_NORMALIZED(
                trim_amount=get_value_at_index(conditioning, 3), samples=get_value_at_index(sampled, 0),
            )
            decoded = vaedecode.decode(samples=get_value_at_index(trimmed, 0), vae=get_value_at_index(PIPELINE["vae"], 0))
            frames = _prepared(imagebatch).EXECUTE_NORMALIZED(
                batch_index=get_value_at_index(conditioning, 4), length=4096, image=get_value_at_index(decoded, 0),
            )
            _update_job(job_id, progress=0.88, message="Encoding the generated frames into MP4")
            video_out = _prepared(createvideo).EXECUTE_NORMALIZED(
                fps=30, bit_depth="auto", color_space="sRGB", images=get_value_at_index(frames, 0),
            )
            output_dir = Path(folder_paths.get_output_directory()) / "video"
            output_dir.mkdir(parents=True, exist_ok=True)
            before = set(output_dir.iterdir())
            _prepared(savevideo).EXECUTE_NORMALIZED(
                filename_prefix=f"effectlab/{job_id}", format="auto", codec="auto",
                video=get_value_at_index(video_out, 0),
            )
            created = [p for p in output_dir.iterdir() if p not in before and p.is_file()]
            if not created:
                created = sorted((p for p in output_dir.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)[:1]
            if not created:
                raise RuntimeError("Wan Animate finished but no MP4 file was written.")
            result_path = job_dir / "result.mp4"
            shutil.copy2(max(created, key=lambda p: p.stat().st_mtime), result_path)
        _update_job(job_id, status="complete", progress=1.0, message="Full-body replacement finished", result_path=str(result_path))
    except Exception as exc:
        _update_job(job_id, status="failed", message="Generation failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if staged_image and staged_image.exists():
            try:
                staged_image.unlink()
            except OSError:
                pass
        for temporary in job_dir.iterdir() if job_dir.exists() else []:
            if temporary.name == "result.mp4":
                continue
            try:
                if temporary.is_file() or temporary.is_symlink():
                    temporary.unlink(missing_ok=True)
                elif temporary.is_dir():
                    shutil.rmtree(temporary, ignore_errors=True)
            except OSError:
                pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_models() -> None:
    try:
        if not COMFYUI_PATH.exists():
            raise RuntimeError(f"ComfyUI is missing at {COMFYUI_PATH}. Use backend/Dockerfile to build this service.")
        _download_models()
        _initialize_pipeline()
        STATE.update(ready=True, loading=False, error=None)
        print("[EffectLab] Full-body engine is ready.")
    except Exception as exc:
        STATE.update(ready=False, loading=False, error=f"{type(exc).__name__}: {exc}")
        print(f"[EffectLab] Engine startup failed: {STATE['error']}", flush=True)


@app.on_event("startup")
def start_model_loader() -> None:
    threading.Thread(target=_load_models, daemon=True, name="effectlab-model-bootstrap").start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "service": APP_TITLE,
        "ready": bool(STATE["ready"]),
        "loading": bool(STATE["loading"]),
        "error": STATE["error"] if not STATE["ready"] else None,
        "gpu": _gpu_status(),
    }


def _gpu_status() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1),
            }
        return {"available": False, "name": None}
    except Exception:
        return {"available": False, "name": None}


@app.post("/api/jobs")
async def create_job(
    request: Request,
    image: UploadFile = File(...),
    video: UploadFile = File(...),
    frame_cap: int = Form(120),
    prompt: str = Form(DEFAULT_POSITIVE_PROMPT),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_auth(authorization)
    _check_rate_limit(request)
    if not STATE["ready"]:
        reason = STATE["error"] or "Model weights are still loading. Check /health and retry."
        raise HTTPException(status_code=503, detail=reason)
    with JOB_LOCK:
        active = sum(1 for job in JOBS.values() if job["status"] in {"queued", "processing"})
        if active >= MAX_QUEUED_JOBS:
            raise HTTPException(status_code=429, detail="EffectLab is at its render capacity. Try again shortly.")
    frame_cap = max(17, min(int(frame_cap), 150))
    image_name = Path(image.filename or "reference.jpg").name
    image_suffix = Path(image_name).suffix.lower()
    if image_suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        image_suffix = ".jpg"
    video_name = Path(video.filename or "source.mp4").name
    video_suffix = Path(video_name).suffix.lower()
    if video_suffix not in {".mp4", ".webm", ".mov", ".m4v"}:
        raise HTTPException(status_code=415, detail="Upload an MP4, WebM or MOV source video.")
    image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
    video_bytes = await video.read(MAX_VIDEO_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Reference image exceeds 24 MB.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail="Source video exceeds 100 MB.")
    if not image_bytes or not video_bytes:
        raise HTTPException(status_code=400, detail="Both a full-body reference image and a source video are required.")
    try:
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as im:
            im.verify()
    except Exception:
        raise HTTPException(status_code=415, detail="The reference file is not a supported image.")
    job_id = uuid.uuid4().hex
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    image_path = job_dir / ("reference" + image_suffix)
    video_path = job_dir / ("source" + video_suffix)
    image_path.write_bytes(image_bytes)
    video_path.write_bytes(video_bytes)
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=20,
        )
        duration = float(probe.stdout.strip())
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=415, detail="The source video could not be read. Try a standard MP4 file.")
    if probe.returncode != 0 or duration <= 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=415, detail="The source video could not be decoded.")
    if duration > MAX_DURATION_SECONDS:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=f"Keep the source clip under {MAX_DURATION_SECONDS:g} seconds for this first version.")
    with JOB_LOCK:
        JOBS[job_id] = {
            "id": job_id, "status": "queued", "progress": 0.0,
            "message": "Waiting for the GPU", "error": None, "result_path": None,
            "created_at": time.time(), "updated_at": time.time(),
            "directory": str(job_dir),
        }
    GPU_EXECUTOR.submit(_run_generation, job_id, str(image_path), str(video_path), frame_cap, prompt, negative_prompt)
    return _public_job(JOBS[job_id])


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    _require_auth(authorization)
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Render job not found or the server restarted.")
        return _public_job(job)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str, authorization: str | None = Header(default=None)) -> FileResponse:
    _require_auth(authorization)
    with JOB_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Render job not found or the server restarted.")
        if job["status"] != "complete" or not job.get("result_path"):
            raise HTTPException(status_code=409, detail="The video has not finished generating.")
        result = Path(job["result_path"])
    if not result.is_file():
        raise HTTPException(status_code=410, detail="This output has expired. Generate the video again.")
    return FileResponse(result, media_type="video/mp4", filename="effectlab-full-body-swap.mp4")
