"""Kaggle's GPU-only half of the local bridge, with two isolated T4 clip workers."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from behaviorsense.models.rgb_hybrid import MODEL_ID, validate_bundle
from behaviorsense.models.visual_context import VALIDATED_MODEL_SHA256, file_sha256
from behaviorsense.service.chunk_uploads import UploadStore, install_chunk_upload_routes
from behaviorsense.service.clip_workers import ClipJob, ClipQueueFull, ClipWorker, ClipWorkerPool, clip_response
from behaviorsense.service.upload_limits import upload_health
from behaviorsense.video import _child_device_config, child_env


@dataclass(frozen=True)
class ServingConfig:
    bundle: Path
    rtmo: Path
    rtdetr: Path
    max_frames: int = 36000
    max_upload_mb: int = 100
    token: str = ""


def isolated_events(config, worker, video, metadata, directory):
    """Keep the GPU reservation until this child exits, including on disconnect."""
    directory = Path(directory)
    gallery_file = None
    gallery = metadata.get("gallery", "")
    if gallery:
        if not isinstance(gallery, str):
            raise ValueError("Gallery metadata must be JSON text.")
        document = json.loads(gallery)
        if not isinstance(document, dict) or not isinstance(document.get("residents", {}), dict):
            raise ValueError("Gallery must contain a residents object.")
        if document.get("residents"):
            gallery_file = directory / "gallery.json"
            gallery_file.write_text(gallery, encoding="utf-8")
    payload = {"bundle": str(config.bundle), "rtmo": str(config.rtmo), "rtdetr": str(config.rtdetr),
        "video": str(video), "result": str(directory / "result.json"), "max_frames": config.max_frames,
        "smoke_only": video is None,
        "use_osnet": bool(gallery_file or metadata.get("enrol", "").strip()),
        "gallery_file": str(gallery_file) if gallery_file else None}
    job_file = directory / "worker.json"
    job_file.write_text(json.dumps(payload), encoding="utf-8")
    environment, _, _ = _child_device_config(child_env(), worker.device, worker.device)
    command = [sys.executable, "-u", "-m", "behaviorsense.service.rgb_hybrid_worker", "--config", str(job_file)]
    proc = None
    with (directory / "worker.log").open("w", encoding="utf-8") as log, \
            (directory / "worker.log").open("r", encoding="utf-8") as reader:
        try:
            proc = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            changed = heartbeat = time.monotonic()
            last = {"step": "loading_models"}
            while proc.poll() is None:
                line = reader.readline()
                if line:
                    changed = time.monotonic()
                    try:
                        event = json.loads(line)
                        if isinstance(event, dict) and "step" in event:
                            last = event
                    except ValueError:
                        pass
                if time.monotonic() - heartbeat >= 5:
                    yield json.dumps({"event": "heartbeat", **last, "device": worker.device}) + "\n"
                    heartbeat = time.monotonic()
                if time.monotonic() - changed > 240:
                    raise RuntimeError("The GPU worker stopped reporting progress for four minutes.")
                if not line:
                    time.sleep(.1)
            if proc.returncode:
                log.flush()
                lines = (directory / "worker.log").read_text(encoding="utf-8", errors="replace").splitlines()
                raise RuntimeError("GPU worker failed: " + (lines[-1] if lines else f"exit {proc.returncode}"))
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            for stage in result["stages"]:
                yield json.dumps({"event": "stage", "stage": stage}) + "\n"
            yield json.dumps({"event": "result", "payload": result}, allow_nan=False) + "\n"
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def create_app(config, *, workers=None, process=isolated_events, validate=True, upload_directory=None):
    """Construct the service; injectable CPU workers exercise real admission/routes in tests."""
    if config.max_frames < 30 or config.max_upload_mb < 1:
        raise ValueError("Invalid serving limits")
    if validate:
        validate_bundle(config.bundle)
        if not config.rtmo.is_file():
            raise ValueError("Attach behavioursense-ww with weights/rtmo-l.onnx.")
        if not config.rtdetr.is_file() or file_sha256(config.rtdetr) != VALIDATED_MODEL_SHA256:
            raise ValueError("Attach the validated rtdetr-l.onnx from behavioursense-ww.")
    gpu_names = []
    if workers is None:
        import torch
        from behaviorsense.models.ensemble import torch_supports_device
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Select GPU T4 x2 in Kaggle Settings; two visible GPUs are required.")
        workers = []
        for index in range(2):
            supported, detail = torch_supports_device(index)
            if not supported:
                raise RuntimeError(detail)
            with torch.cuda.device(index):
                # Check both actual kernels before advertising readiness.
                torch.zeros(1, device=f"cuda:{index}").add_(1).cpu()
            gpu_names.append(torch.cuda.get_device_name(index))
            workers.append(ClipWorker(f"cuda:{index}", f"cuda:{index}", None))
    pool = ClipWorkerPool(workers, max_waiting=2)
    store = UploadStore(limit_mb=config.max_upload_mb, directory=upload_directory)

    @asynccontextmanager
    async def lifespan(app):
        yield
        store.close()

    app = FastAPI(title="BehaviorSense RGB hybrid — GPU stages", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-BS-Token", "X-BS-Upload-Token"])
    app.state.pool, app.state.upload_store = pool, store

    def auth(token):
        if config.token and token != config.token:
            raise HTTPException(401, "bad or missing X-BS-Token")

    @app.get("/health")
    def health():
        return {"service": "behaviorsense", "ready": True, "error": None,
            "gpu": gpu_names[0] if gpu_names else "test worker", "gpu_devices": gpu_names,
            "streams": ["context", "temporal", "rgb"], "n_classes": 22, "task_type": "multilabel",
            "model_id": MODEL_ID, "rgb_precision": "float32", "sample_fps": 20,
            "pose_available": True, "video": True, "auth": bool(config.token), "agent4": None,
            "reporting": "local_backend_only", "model_loading": "isolated child per upload",
            "runtime_benchmark_verified": False, "max_frames": config.max_frames,
            **upload_health(config.max_upload_mb), **pool.snapshot()}

    def start(video, metadata, cleanup):
        try:
            ticket = pool.submit()
        except ClipQueueFull as exc:
            cleanup()
            raise HTTPException(429, str(exc)) from exc
        try:
            directory = Path(tempfile.mkdtemp(prefix="bs-rgb-worker-", dir=upload_directory))
        except BaseException:
            pool.release(ticket)
            cleanup()
            raise

        def finish():
            try:
                cleanup()
            finally:
                shutil.rmtree(directory, ignore_errors=True)

        job = ClipJob(pool, ticket, lambda worker: process(config, worker, video, metadata, directory),
                      finish, max_frames=config.max_frames)
        job.start()
        return clip_response(job)

    async def complete(upload, request):
        if request.query_params.get("stages", "12") != "12":
            raise HTTPException(422, "This notebook serves stages 1–2. Use web/local_backend.py for reports.")
        return start(upload.path, upload.metadata, lambda: None)

    install_chunk_upload_routes(app, store, auth, complete)

    @app.post("/video")
    async def video(file: UploadFile = File(...), enrol: str = Form(""), gallery: str = Form(""),
                    stages: str = "12", x_bs_token: str = Header(default="")):
        auth(x_bs_token)
        if stages != "12":
            raise HTTPException(422, "Use the local backend for stages 3–4.")
        directory = Path(tempfile.mkdtemp(prefix="bs-rgb-upload-", dir=upload_directory))
        path, size = directory / "upload.mp4", 0
        try:
            with path.open("wb") as target:
                while data := await file.read(1024 * 1024):
                    size += len(data)
                    if size > config.max_upload_mb << 20:
                        raise HTTPException(413, f"Upload exceeds {config.max_upload_mb} MB.")
                    target.write(data)
            if not size:
                raise HTTPException(422, "Empty video upload.")
            return start(path, {"enrol": enrol, "gallery": gallery},
                         lambda: shutil.rmtree(directory, ignore_errors=True))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            await file.close()

    return app
