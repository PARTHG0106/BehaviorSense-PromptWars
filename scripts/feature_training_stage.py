"""Train/evaluate frozen extracted features without mounting detector assets.

This separately checksummed orchestration companion leaves the original source
archive, trainer, models, annotations, and feature provenance checks unchanged.
Only the extraction-only mounts and dependency setup are omitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from multilabel_kaggle_stage import (asset, cache_directory, dataset_root,
                                    mounted_sources, read_json, sha256,
                                    training_summary, write_json)


def bootstrap(job, repo, work):
    """Verify the pinned image's Torch runtime; no package installation is needed."""
    def run(args):
        print("RUN", " ".join(map(str, args)), flush=True)
        subprocess.run(list(map(str, args)), cwd=repo, check=True)

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"The verified Blackwell image uses Python 3.13, got {sys.version}")
    import torch
    import numpy as np

    core = {"python": ".".join(map(str, sys.version_info[:2])),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "numpy": str(np.__version__)}
    if job.get("expected_core_runtime") is not None and job["expected_core_runtime"] != core:
        raise ValueError(f"Core runtime differs from the reviewed image: {core}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU training/evaluation is forbidden")
    gpu = torch.cuda.get_device_name(0)
    if "RTX PRO 6000" not in gpu or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError(f"Expected the authorized RTX PRO 6000 Blackwell, got {gpu}")
    if "sm_120" not in torch.cuda.get_arch_list():
        raise RuntimeError("Torch build lacks native Blackwell support")
    value = torch.ones((64, 64), device="cuda")
    if not bool(torch.isfinite(value @ value).all()):
        raise RuntimeError("CUDA arithmetic check failed")
    report = {"gpu": gpu, "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
              "python": sys.version, "torch": core["torch"], "cuda": core["cuda"],
              "numpy": core["numpy"], "core_runtime": core,
              "dependency_profile": "frozen_features_torch_numpy",
              "extraction_runtime_installed": False,
              "source_sha256": job["source_sha256"],
              "feature_runner_sha256": job["feature_runner_sha256"]}
    write_json(work / "environment.json", report)
    print(json.dumps(report), flush=True)
    return run


def execute_features(job, repo, work):
    began = time.monotonic()
    result = {"job_id": job["job_id"], "stage": job["stage"],
              "source_sha256": job["source_sha256"],
              "feature_runner_sha256": job["feature_runner_sha256"], "status": "running"}

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(time.monotonic() - began, 2), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)

    try:
        if job["stage"] not in ("train", "evaluate"):
            raise ValueError("Feature orchestration only supports train/evaluate stages")
        status("preflight")
        run = bootstrap(job, repo, work)
        sources = mounted_sources(job)
        annotations = asset(dataset_root(job["owner"], "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("Annotations changed from the reviewed full-data experiment")
        stage = job["stage"]
        # Preserve the original stage's trainer arguments and completion policy.
        if stage == "train":
            out = work / "candidate"
            shard_patterns = [root / "shards" / "*.npz" for root, value in sources.values()
                              if value["stage"] == "extract" and value.get("subset") == "training"]
            if not shard_patterns or any(value.get("subset") == "testing" for _, value in sources.values()):
                raise ValueError("Training requires fitting shards and forbids test mounts")
            resume = []
            for root, value in sources.values():
                if value["stage"] == "train":
                    if value.get("status") != "resumable":
                        raise ValueError("Training resume requires a safely resumable stage result")
                    if value.get("model") != job["model"] or value.get("seed") != job["seed"]:
                        raise ValueError("Training resume model/seed mismatch")
                    if resume:
                        raise ValueError("Only the latest training checkpoint may be mounted")
                    if not (root / "candidate" / "last_multilabel.pt").is_file():
                        raise ValueError("Training resume source has no last_multilabel.pt checkpoint")
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
            if trained.get("stage") not in ("train", "train_group") or trained.get("status") != "complete":
                raise ValueError("Test evaluation requires the selected completed training stage")
            if trained["stage"] == "train_group":
                seed, model = chosen.get("seed"), chosen.get("model")
                if type(seed) is not int or seed not in (0, 1, 2) or model not in ("skeleton", "context-fusion"):
                    raise ValueError("Selected group candidate has an invalid model/seed")
                candidates = trained.get("candidates")
                if not isinstance(candidates, list):
                    raise ValueError("Selected training group has no candidate inventory")
                matches = [candidate for candidate in candidates if isinstance(candidate, dict)
                           and candidate.get("model") == model
                           and type(candidate.get("seed")) is int and candidate["seed"] == seed]
                if len(matches) != 1 or matches[0].get("status") != "complete":
                    raise ValueError("Selected group candidate must identify one completed model/seed")
                relative = chosen.get("checkpoint_relative_path")
                if relative != f"candidates/seed{seed}/best_multilabel.pt":
                    raise ValueError("Selected group checkpoint path is not canonical")
                candidate = matches[0]
                if (candidate.get("checkpoint_relative_path") != relative
                        or candidate.get("checkpoint_sha256") != chosen["checkpoint_sha256"]):
                    raise ValueError("Selected group candidate path/hash differs from frozen selection")
                checkpoint = root / relative
                if not checkpoint.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Selected group checkpoint path escapes mounted output")
            else:
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
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
