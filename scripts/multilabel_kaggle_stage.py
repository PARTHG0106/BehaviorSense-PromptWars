"""Offline, finite stages of the private full-coverage multilabel experiment.

The local launcher embeds a frozen source archive and a job description. No Kaggle
credentials are needed or accepted inside a GPU session.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_summary(manifest):
    # Full per-video/window provenance remains in candidate/manifest.json. Keeping
    # it in every coordinator transition would repeatedly duplicate millions of IDs.
    fields = ("format", "model", "train", "validation", "convergence", "checkpoint_selection",
              "full_eligible_window_coverage_verified", "test_loaded_during_training", "execution")
    return {key: manifest[key] for key in fields if key in manifest}


def cache_directory():
    # Keep reconstructible arrays outside saved Kaggle output artifacts.
    cache = Path("/kaggle/temp") / "multilabel_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def dataset_root(owner, slug, inputs=Path("/kaggle/input")):
    choices = [inputs / slug, inputs / "datasets" / owner / slug]
    found = [path for path in choices if path.is_dir()]
    if not found:
        found = [path for path in inputs.rglob(slug) if path.is_dir()]
    found = sorted(set(path.resolve() for path in found))
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one mounted dataset {slug}, got {found}")
    return found[0]


def asset(root, name):
    found = sorted(root.rglob(name))
    if not found:
        raise FileNotFoundError(f"{name} missing from {root}")
    # The weights dataset has duplicate RT-DETR files. Accept only identical copies.
    if len(found) > 1 and len({sha256(path) for path in found}) != 1:
        raise ValueError(f"Conflicting copies of {name} in the mounted dataset")
    return found[0]


def mounted_sources(job, inputs=Path("/kaggle/input")):
    expected = set(job.get("sources", []))
    found = {}
    for path in inputs.rglob("stage_result.json"):
        result = read_json(path)
        identity = result.get("job_id")
        if identity not in expected:
            continue
        if identity in found:
            raise ValueError(f"Ambiguous mounted output for {identity}")
        if result.get("source_sha256") != job["source_sha256"]:
            raise ValueError(f"Source snapshot mismatch in {identity}")
        if result.get("status") not in ("complete", "resumable"):
            raise ValueError(f"Input stage {identity} did not finish safely")
        found[identity] = (path.parent, result)
    if set(found) != expected:
        raise FileNotFoundError(f"Missing input stages: {sorted(expected - set(found))}")
    return found


def bootstrap(job, repo, work):
    def run(args):
        print("RUN", " ".join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), cwd=repo, check=True)

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"The verified Blackwell image uses Python 3.13, got {sys.version}")
    runtime_root = dataset_root(job["owner"], job["runtime_dataset"])
    lock = read_json(runtime_root / "runtime-lock.json")
    if lock != job["runtime_lock"]:
        raise ValueError("Mounted runtime lock differs from the frozen experiment")
    for name, expected in lock["files"].items():
        if sha256(runtime_root / name) != expected:
            raise ValueError(f"Runtime wheel checksum mismatch: {name}")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"])
    run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--force-reinstall", *[runtime_root / name for name in lock["files"]]])
    import torch
    import numpy as np
    import onnxruntime as ort
    import cv2
    import rtmlib  # noqa: F401
    import yaml  # noqa: F401
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU training/extraction is forbidden")
    gpu = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in gpu or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(f"Expected the authorized RTX PRO 6000 Blackwell, got {gpu}")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("Torch build lacks native Blackwell support")
    value = torch.ones((64, 64), device="cuda")
    if not bool(torch.isfinite(value @ value).all()):
        raise RuntimeError("CUDA arithmetic check failed")
    ort.preload_dlls()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime GPU provider is missing")
    report = {"gpu": gpu, "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "onnxruntime": ort.__version__, "opencv": cv2.__version__,
              "numpy": np.__version__, "runtime_lock": lock}
    write_json(work / "environment.json", report)
    print(json.dumps(report), flush=True)
    return run


def execute(job, repo, work):
    began = time.monotonic()
    result = {"job_id": job["job_id"], "stage": job["stage"],
              "source_sha256": job["source_sha256"], "status": "running"}

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(time.monotonic() - began, 2), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)

    try:
        status("preflight")
        run = bootstrap(job, repo, work)
        sources = mounted_sources(job)
        weights = dataset_root(job["owner"], "behavioursense-ww")
        annotations = asset(dataset_root(job["owner"], "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("Annotations changed from the reviewed full-data experiment")
        stage = job["stage"]
        if stage == "preflight":
            import numpy as np
            from extract_multilabel_windows import load_detector, require_cuda_session, validate_pose
            from behaviorsense.models.visual_context import RTDETRContextDetector
            body = load_detector(asset(weights, "rtmo-l.onnx"))
            frame = np.zeros((640, 640, 3), np.uint8)
            points, scores = validate_pose(*body(frame))
            require_cuda_session(body)
            detector = RTDETRContextDetector(asset(weights, "rtdetr-l.onnx"))
            # A blank frame is for the GPU contract, not a quality or accuracy score.
            feature = detector.features(frame, np.zeros((2, 17, 3), np.float32))
            if not np.isfinite(feature).all():
                raise RuntimeError("Non-finite visual features in the GPU preflight")
            status("complete", status="complete", pose_shape=list(points.shape),
                   context_shape=list(feature.shape), accuracy_evaluation=False)
        elif stage == "extract":
            subset = job["subset"]
            out = work / "shards"
            resumes = []
            for root, previous in sources.values():
                if previous["stage"] == "extract":
                    if (previous.get("subset") != subset
                            or previous.get("partition_index") != job["partition_index"]
                            or previous.get("partition_count") != job["partition_count"]):
                        raise ValueError("Extraction resume mounts must be from the same subset/partition")
                    resumes += ["--resume-from", root / "shards"]
            status("extract", subset=subset, partition_index=job["partition_index"],
                   partition_count=job["partition_count"], windows_per_video=None)
            run([sys.executable, "scripts/extract_multilabel_windows.py",
                 "--videos-root", dataset_root(job["owner"], "rgb-untrimmed"),
                 "--annotations", annotations, "--rtmo-model", asset(weights, "rtmo-l.onnx"),
                 "--context-model", asset(weights, "rtdetr-l.onnx"), "--out", out,
                 "--subset", subset, "--selection", "full", "--candidate-stride", "15",
                 "--decode-mode", "sequential", "--partition-count", job["partition_count"],
                 "--partition-index", job["partition_index"],
                 "--stop-after-seconds", job["stage_seconds"], *resumes])
            manifests = list(out.glob("manifest_*.json"))
            if len(manifests) != 1:
                raise ValueError("Extraction must emit one partition manifest")
            manifest = read_json(manifests[0])
            complete = bool(manifest["scope_complete"])
            status("complete" if complete else "resume_required",
                   status="complete" if complete else "resumable",
                   scope_complete=complete, completed_videos=manifest["completed_videos"],
                   expected_videos=manifest["scope_expected_videos"], windows=manifest["windows"],
                   config_fingerprint=manifest["fingerprint"])
        elif stage == "train":
            out = work / "candidate"
            shard_patterns = [root / "shards" / "*.npz" for root, value in sources.values()
                              if value["stage"] == "extract" and value.get("subset") == "training"]
            if not shard_patterns or any(value.get("subset") == "testing" for _, value in sources.values()):
                raise ValueError("Training requires fitting shards and forbids test mounts")
            resume = []
            for root, value in sources.values():
                if value["stage"] == "train":
                    if value.get("model") != job["model"] or value.get("seed") != job["seed"]:
                        raise ValueError("Training resume model/seed mismatch")
                    if resume:
                        raise ValueError("Only the latest training checkpoint may be mounted")
                    shutil.copytree(root / "candidate", out)
                    resume = ["--resume", out / "last_multilabel.pt"]
            cache = cache_directory()
            status("train", model=job["model"], seed=job["seed"])
            args = [sys.executable, "scripts/train_adl_multilabel.py", "train",
                    "--train-shards", *shard_patterns, "--annotations", annotations,
                    "--out", out, "--model", job["model"], "--seed", job["seed"],
                    "--val-subjects", "p04", "p15", "--epochs", "150", "--min-epochs", "40",
                    "--patience", "25", "--scheduler", "plateau", "--lr-patience", "5",
                    "--lr-factor", "0.5", "--min-lr", "0.000001", "--convergence-lr-reductions", "2",
                    "--lr", "0.0003", "--batch-size", "256", "--workers", "2", "--device", "cuda",
                    "--require-full-coverage", "--require-convergence", "--cache-dir", cache,
                    "--stop-after-seconds", job["stage_seconds"], *resume]
            if job["model"] == "context-fusion":
                args.append("--require-context")
            run(args)
            inference = out / "best_multilabel.pt"
            manifest = read_json(out / "manifest.json")
            if inference.is_file():
                status("complete", status="complete", manifest=training_summary(manifest),
                       checkpoint_sha256=sha256(inference))
            else:
                if not (out / "last_multilabel.pt").is_file():
                    raise RuntimeError("Interrupted training did not save a resumable checkpoint")
                convergence = manifest.get("convergence", {})
                exhausted = (bool(convergence.get("budget_exhausted", False))
                             or "budget_exhausted" in str(convergence.get("status", ""))
                             or (convergence.get("completed_epochs", -1) >=
                                 convergence.get("epoch_budget", float("inf"))))
                status("needs_review" if exhausted else "resume_required",
                       status="needs_review" if exhausted else "resumable", manifest=training_summary(manifest))
        elif stage == "evaluate":
            chosen = job["selection"]
            root, trained = sources[chosen["job_id"]]
            checkpoint = root / "candidate" / "best_multilabel.pt"
            if sha256(checkpoint) != chosen["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after validation selection")
            tests = [r / "shards" / "*.npz" for r, value in sources.values()
                     if value["stage"] == "extract" and value.get("subset") == "testing"]
            if not tests:
                raise ValueError("No official test shards")
            write_json(work / "frozen_selection.json", chosen)
            status("evaluate_test", checkpoint_selection_on_test=False)
            run([sys.executable, "scripts/train_adl_multilabel.py", "evaluate-test",
                 "--checkpoint", checkpoint, "--test-shards", *tests, "--annotations", annotations,
                 "--device", "cuda", "--cache-dir", cache_directory(),
                 "--require-full-coverage", "--out", work / "test"])
            report = read_json(work / "test" / "test_metrics.json")
            summary = {key: report[key] for key in ("metrics", "checkpoint_sha256", "thresholds_refit",
                       "checkpoint_selection_on_test", "split", "training_convergence") if key in report}
            status("complete", status="complete", test=summary,
                   serving_model_replaced=False)
        else:
            raise ValueError(f"Unknown stage {stage}")
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
