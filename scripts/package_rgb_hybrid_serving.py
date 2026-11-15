"""Create the local T4 x2 notebook and private upload bundle; never contacts Kaggle."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from behaviorsense.models.rgb_hybrid import BUNDLE_SCHEMA, FROZEN_FILES, MODEL_ID
from behaviorsense.models.rgb_action import PREPROCESSING_SHA256

SOURCES = {
    "context_seed2.pt": "artifacts/training-tv-20261007/completion-audit/bs-mlfull-20261007-ctx-all-a00/candidates/seed2/best_multilabel.pt",
    "temporal_asl_seed1.pt": "artifacts/temporal-multilabel-20261011/output/candidates/asl-seed1/best_multilabel.pt",
    "rgb_head.pt": "artifacts/rgb-runtime-20261014/submit-compact/output/pilot/best_rgb_multilabel.pt",
    "blend_thresholds.json": "artifacts/rgb-improve-20261015/fixed-updated-blend/candidate_thresholds.json",
    "rgb_thresholds.json": "artifacts/rgb-runtime-20261014/submit-compact/output/pilot/thresholds.json",
    "swin3d_t-7615ae03.pth": "artifacts/pipeline-rebuild-20261012/assets/swin3d_t-7615ae03.pth",
    "osnet_ain_x1_0_msmt17.pth": "weights/osnet_ain_x1_0_msmt17.pth",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def notebook_document():
    cells = []

    def cell(kind, body):
        body = body.strip() + "\n"
        if kind == "code":
            ast.parse(body)
        value = {"cell_type": kind, "metadata": {}, "source": body,
                 "id": hashlib.sha256(body.encode()).hexdigest()[:12]}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """
# 08 — Saved RGB hybrid on Kaggle (ONLINE / T4 ×2)

Upload this notebook yourself. Select **GPU T4 x2** and **Internet On**, then attach:

1. **[penguin069/behaviorsense-rgb-hybrid-serving](https://www.kaggle.com/datasets/penguin069/behaviorsense-rgb-hybrid-serving)**.
   It contains the runtime source, three frozen heads,
   thresholds, Swin3D-T and optional OSNet weights.
2. **`krishnagupta02468/behavioursense-ww`**, containing `weights/rtmo-l.onnx` and `rtdetr-l.onnx`.

No Qwen model, old Toyota ADL runs, training videos, annotations, feature caches,
or separate `behaviorsense-code` dataset is required. This notebook never loads an LLM.
Agents 3–4 and OpenRouter credentials stay in `web/local_backend.py` on your machine.

The frozen model combines 37.5% context seed 2, 37.5% temporal ASL seed 1 and 25%
RGB, preserving its class thresholds and TV-plus-drinking decision rule. It is an
offline multilabel model; multiple activities may appear at the same time.

**T4 portability:** Swin runs in FP32 because T4 has no native BF16 support.
The original BF16 benchmark reached 69.21% mAP and 28.12% pair F1 and failed its
pair quality gate. Those figures do not measure this new FP32/track adapter.
Both GPUs receive a real forward-pass smoke check below; this is not an accuracy evaluation.

Each active upload runs in its own process on one T4. Two uploads can run concurrently;
GPU memory is not pooled. Defaults: 100 MB per file and 30 minutes at 20 FPS.
Videos below 20 FPS are refused, and only complete windows with sufficient visible
pose are classified. Decisions occupy disjoint 0.75-second center bins; gaps and
unscored edges are preserved. No training, threshold fitting or test scoring occurs.
""")
    cell("code", r'''
# Locate and verify the runtime packaged with the model. Standard library only.
import hashlib, json, os, pathlib, sys
INPUT = pathlib.Path("/kaggle/input")
BUNDLE_DIR = None  # Optional explicit path if multiple serving bundles are attached.
mounts = sorted((INPUT / "datasets").glob("*/*")) + sorted(p for p in INPUT.iterdir() if p.is_dir())
if BUNDLE_DIR is None:
    candidates = sorted({base / prefix for base in mounts
        for prefix in ("", "bundle", "behaviorsense-rgb-hybrid-serving-20261016")
        if (base / prefix / "hybrid_manifest.json").is_file()})
    assert len(candidates) == 1, (
        "Attach the RGB hybrid serving bundle (hybrid_manifest.json must be at its root), "
        "or set BUNDLE_DIR explicitly. Found: " + str(candidates))
    BUNDLE_DIR = candidates[0]
BUNDLE_DIR = pathlib.Path(BUNDLE_DIR).resolve()
source_manifest = json.loads((BUNDLE_DIR / "source_manifest.json").read_text())
SOURCE = BUNDLE_DIR / "runtime_src"
for relative, expected in source_manifest["files"].items():
    path = (SOURCE / relative).resolve()
    assert path.is_relative_to(SOURCE.resolve()), "Source path escapes the bundle"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Source checksum differs: " + relative
sys.path.insert(0, str(SOURCE))
print("Serving bundle:", BUNDLE_DIR)
print("Verified runtime files:", len(source_manifest["files"]))
''')
    cell("code", r'''
# Install the CUDA 12.8 Torch runtime and CUDA 12 ONNX Runtime before importing models.
import importlib.metadata, subprocess
def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""
if not (version("torch").startswith("2.11.") and "+cu128" in version("torch")
        and version("torchvision").startswith("0.26.") and "+cu128" in version("torchvision")):
    assert "torch" not in sys.modules, "Restart the Kaggle session before replacing its Torch runtime."
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
        "torch==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "rtmlib"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn", "nest_asyncio", "python-multipart", "pydantic>=2",
    "opencv-python-headless>=4.13,<4.15", "scipy", "PyYAML", "tqdm", "flatbuffers", "protobuf", "packaging"], check=True)
# Reinstall last: CPU and GPU ORT distributions share the same Python package directory.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
    "onnxruntime-gpu==1.26.0"], check=True)
import torch, torchvision, onnxruntime as ort
assert torch.cuda.device_count() >= 2, "Select GPU T4 x2 in Kaggle Settings, then start a fresh session."
assert "CUDAExecutionProvider" in ort.get_available_providers(), "ONNX Runtime CUDA is unavailable."
print("Torch", torch.__version__, "torchvision", torchvision.__version__, "ORT", ort.__version__)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
''')
    cell("code", r'''
# Resolve detector assets. No old ADL runs or language-model dataset is read.
from behaviorsense.models.rgb_hybrid import validate_bundle
from behaviorsense.models.visual_context import VALIDATED_MODEL_SHA256, file_sha256
from behaviorsense.service.rgb_hybrid_backend import ServingConfig, create_app, isolated_events
from behaviorsense.service.clip_workers import ClipWorker

def asset(filename, expected=None):
    candidates = sorted({base / prefix / filename for base in mounts
        for prefix in ("", "weights", "EmotionSense-Extended/weights")
        if (base / prefix / filename).is_file()})
    if expected:
        candidates = [path for path in candidates if file_sha256(path) == expected]
    assert candidates, "Attach behavioursense-ww containing the validated " + filename
    return candidates[0]

manifest = validate_bundle(BUNDLE_DIR)
config = ServingConfig(bundle=BUNDLE_DIR, rtmo=asset("rtmo-l.onnx"),
    rtdetr=asset("rtdetr-l.onnx", VALIDATED_MODEL_SHA256),
    max_frames=int(os.environ.get("BS_MAX_FRAMES", "36000")),
    max_upload_mb=int(os.environ.get("BS_MAX_UPLOAD_MB", "100")), token=os.environ.get("BS_TOKEN", ""))
print("Model:", manifest["model_id"])
print("Class decisions: frozen sigmoid thresholds; RGB encoder: FP32")
print("RTMO:", config.rtmo)
print("RT-DETR:", config.rtdetr)
''')
    cell("code", r'''
# Exercise RTMO, RT-DETR, Swin and all three heads on BOTH T4s before starting the service.
# Synthetic pixels only: no benchmark arrays, fitting or threshold changes.
import tempfile
smoke_receipts = []
for gpu in range(2):
    worker = ClipWorker(f"cuda:{gpu}", f"cuda:{gpu}", None)
    with tempfile.TemporaryDirectory(prefix="rgb-smoke-") as directory:
        for raw_event in isolated_events(config, worker, None, {}, directory):
            event = json.loads(raw_event)
            if event["event"] == "heartbeat":
                print(f"GPU {gpu}:", event.get("step"))
            if event["event"] == "result":
                receipt = event["payload"]
                assert receipt.get("smoke_passed"), receipt
                smoke_receipts.append(receipt)
                print(f"GPU {gpu} smoke passed:", receipt["gpu"], receipt["rgb_precision"])
assert len(smoke_receipts) == 2
app = create_app(config)
print("Ready for two simultaneous clips. Reports run through your local backend.")
''')
    cell("code", r'''
# Start the API and print the address to pass to your LOCAL backend.
import re, threading, time, urllib.request, uvicorn
CF = pathlib.Path("/kaggle/working/cloudflared")
if not CF.is_file():
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/download/2024.12.2/cloudflared-linux-amd64", CF)
    CF.chmod(0o755)
server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning"))
server_thread = threading.Thread(target=server.run, daemon=True)
server_thread.start()
for _ in range(100):
    if server.started:
        break
    if not server_thread.is_alive():
        raise RuntimeError("The API server exited during startup.")
    time.sleep(.1)
assert server.started, "The API server did not start."
log_path = pathlib.Path("/kaggle/working/rgb-cloudflared.log")
with log_path.open("w") as log:
    tunnel = subprocess.Popen([str(CF), "tunnel", "--url", "http://localhost:8000", "--no-autoupdate"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        URL = None
        for _ in range(120):
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text())
            if match:
                URL = match.group(0)
                break
            assert tunnel.poll() is None, "Cloudflare exited; inspect rgb-cloudflared.log."
            time.sleep(.5)
        assert URL, "No tunnel address appeared; inspect rgb-cloudflared.log."
        print("Run this command on your Windows machine:")
        print("python web/local_backend.py --kaggle " + URL + (" --token YOUR_BS_TOKEN" if config.token else ""))
        print("Connect the web page to http://127.0.0.1:8899 and leave this cell running.")
        print("No language-model dataset or API key is used by this notebook.")
        while tunnel.poll() is None and server_thread.is_alive():
            time.sleep(10)
    finally:
        server.should_exit = True
        if tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
''')
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


def build(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts/rgb-serving-20261016"):
        raise ValueError("The private serving package must stay inside its ignored artifact directory.")
    bundle = output / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    files = {}
    packaged_paths = set()
    for name, relative in SOURCES.items():
        source = ROOT / relative
        digest = sha(source)
        if name in FROZEN_FILES and digest != FROZEN_FILES[name]:
            raise ValueError("Frozen file changed: " + name)
        shutil.copy2(source, bundle / name)
        packaged_paths.add(bundle / name)
        files[name] = {"sha256": digest, "bytes": source.stat().st_size}
    source_files = {}
    for source in sorted((ROOT / "src/behaviorsense").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(ROOT / "src")
        target = bundle / "runtime_src" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        ast.parse(source.read_text(encoding="utf-8-sig"))
        shutil.copy2(source, target)
        packaged_paths.add(target)
        source_files[relative.as_posix()] = sha(target)
    for name in ("torchvision-LICENSE.txt", "Video-Swin-Transformer-LICENSE.txt"):
        shutil.copy2(ROOT / "artifacts/pipeline-rebuild-20261012/assets" / name, bundle / name)
        packaged_paths.add(bundle / name)
    shutil.copy2(ROOT / "LICENSE", bundle / "BehaviorSense-LICENSE.txt")
    packaged_paths.add(bundle / "BehaviorSense-LICENSE.txt")
    (bundle / "source_manifest.json").write_text(json.dumps({"files": source_files}, indent=2) + "\n", encoding="utf-8")
    packaged_paths.add(bundle / "source_manifest.json")
    manifest = {"schema": BUNDLE_SCHEMA, "model_id": MODEL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": files,
        "preprocessing_sha256": PREPROCESSING_SHA256, "rgb_serving_precision": "float32",
        "benchmark_rgb_precision": "bfloat16", "runtime_benchmark_verified": False,
        "sample_fps": 20, "thresholds_refit": False, "external_dataset": "krishnagupta02468/behavioursense-ww",
        "required_external_files": ["rtmo-l.onnx", "rtdetr-l.onnx"], "visibility": "private"}
    (bundle / "hybrid_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    packaged_paths.add(bundle / "hybrid_manifest.json")
    (bundle / "README.md").write_text(
        "# BehaviorSense RGB hybrid serving inputs\n\n"
        "Keep this dataset private. Attach it to 08_serve_rgb_hybrid_t4x2.ipynb together with "
        "krishnagupta02468/behavioursense-ww. Select GPU T4 x2 and Internet On.\n\n"
        "This package includes runtime code, three frozen activity heads, frozen thresholds, "
        "Swin3D-T and OSNet weights. The other dataset supplies RTMO and RT-DETR. "
        "No language-model dataset, training data or local credentials are included.\n\n"
        "The T4 encoder uses FP32. The original BF16 benchmark is not an accuracy benchmark "
        "for this serving port. Run the notebook's two GPU smoke checks before serving.\n",
        encoding="utf-8")
    packaged_paths.add(bundle / "README.md")
    document = notebook_document()
    notebook = ROOT / "notebooks/08_serve_rgb_hybrid_t4x2.ipynb"
    notebook.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    archive_path = output / "behaviorsense-rgb-hybrid-serving-20261016.zip"
    hashes = {}
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(packaged_paths):
            relative = path.relative_to(bundle).as_posix()
            hashes[relative] = sha(path)
            archive.write(path, relative)
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in hashes.items():
            with archive.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise RuntimeError("Archive verification failed: " + name)
    receipt = {"status": "packaged_and_byte_verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive_path.relative_to(ROOT).as_posix(), "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha(archive_path), "file_count": len(hashes),
        "notebook": notebook.relative_to(ROOT).as_posix(), "notebook_sha256": sha(notebook),
        "remote_notebook_modified": False, "uploaded_to_kaggle": False,
        "gpu_smoke_test_run_locally": False, "files": hashes}
    (output / "package_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "archive", "archive_bytes", "file_count", "notebook")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/rgb-serving-20261016")
    build(parser.parse_args().out)
