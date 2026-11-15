"""Run three frozen-feature training seeds sequentially in one GPU session.

This orchestration companion does not change the frozen trainer or its scientific
configuration. Only one root stage_result.json is emitted so mounted output
discovery remains unambiguous across resumable group attempts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
import time

from feature_training_stage import bootstrap
from multilabel_kaggle_stage import (asset, cache_directory, dataset_root,
                                    mounted_sources, read_json, sha256,
                                    training_summary, write_json)


SESSION_SECONDS = 40500
SEEDS = [0, 1, 2]


def candidate_result(out, model, seed):
    """Read the trainer's own completion evidence, never just a checkpoint name."""
    manifest = read_json(out / "manifest.json")
    training = read_json(out / "training_status.json")
    if manifest.get("model") != model:
        raise ValueError(f"Candidate model mismatch for seed {seed}")
    selection = manifest.get("checkpoint_selection", {})
    if selection.get("training_seed") != seed:
        raise ValueError(f"Candidate training seed mismatch for seed {seed}")
    convergence = manifest.get("convergence", {})
    reported = training.get("convergence", {})
    complete = (training.get("status") == "complete"
                and training.get("inference_checkpoint_exported") is True
                and convergence.get("converged") is True
                and reported.get("converged") is True)
    result = {"seed": seed, "model": model,
              "manifest": training_summary(manifest),
              "training_status": training.get("status")}
    if complete:
        checkpoint = out / "best_multilabel.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Completed seed {seed} has no inference checkpoint")
        result.update(status="complete", checkpoint_sha256=sha256(checkpoint),
                      checkpoint_relative_path=f"candidates/seed{seed}/best_multilabel.pt")
        return result
    if not (out / "last_multilabel.pt").is_file():
        raise RuntimeError(f"Interrupted seed {seed} did not save a resumable checkpoint")
    exhausted = (bool(convergence.get("budget_exhausted", False))
                 or "budget_exhausted" in str(convergence.get("status", ""))
                 or "budget_exhausted" in str(training.get("status", ""))
                 or convergence.get("completed_epochs", -1) >= convergence.get("epoch_budget", 150))
    if training.get("status") not in ("resumable_session_pause",
                                       "epoch_budget_exhausted_without_convergence"):
        raise ValueError(f"Inconsistent training completion evidence for seed {seed}")
    result.update(status="needs_review" if exhausted else "resumable",
                  resume_checkpoint_sha256=sha256(out / "last_multilabel.pt"),
                  resume_checkpoint_relative_path=f"candidates/seed{seed}/last_multilabel.pt")
    return result


def restore_group(job, sources, work, candidates):
    previous = [(root, value) for root, value in sources.values()
                if value.get("stage") == "train_group"]
    if len(previous) > 1:
        raise ValueError("Only the latest group attempt may be mounted")
    if not previous:
        return
    root, value = previous[0]
    for key in ("source_sha256", "model", "seeds", "seed_runner_sha256", "feature_runner_sha256",
                "expected_core_runtime"):
        if value.get(key) != job.get(key):
            raise ValueError(f"Group resume {key} mismatch")
    entries = value.get("candidates", [])
    if (len(entries) != len(SEEDS)
            or [entry.get("seed") for entry in entries] != SEEDS):
        raise ValueError("Group resume must report each seed exactly once in order")
    for entry in entries:
        seed = entry["seed"]
        if entry.get("model") != job["model"]:
            raise ValueError(f"Group resume candidate model mismatch for seed {seed}")
        state = entry.get("status")
        if state == "pending":
            continue
        if state not in ("complete", "resumable", "needs_review"):
            raise ValueError(f"Unsafe group resume candidate status for seed {seed}")
        source = root / "candidates" / f"seed{seed}"
        verified = candidate_result(source, job["model"], seed)
        if verified["status"] != state:
            raise ValueError(f"Group resume candidate status mismatch for seed {seed}")
        hash_key = "checkpoint_sha256" if state == "complete" else "resume_checkpoint_sha256"
        path_key = "checkpoint_relative_path" if state == "complete" else "resume_checkpoint_relative_path"
        if entry.get(hash_key) != verified[hash_key] or entry.get(path_key) != verified[path_key]:
            raise ValueError(f"Group resume checkpoint mismatch for seed {seed}")
        # Preserve complete inference artifacts and exact partial training state.
        # Older incidental nested summaries are deliberately excluded.
        shutil.copytree(source, work / "candidates" / f"seed{seed}",
                        ignore=shutil.ignore_patterns("stage_result.json"))
        candidates[seed] = verified


def execute_seed_group(job, repo, work):
    began = time.monotonic()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    candidates = [{"seed": seed, "model": job.get("model"), "status": "pending"} for seed in SEEDS]
    result = {key: job.get(key) for key in ("job_id", "stage", "source_sha256", "model", "seeds",
                                          "seed_runner_sha256", "feature_runner_sha256",
                                          "expected_core_runtime")}
    result.update(status="running", candidates=candidates)

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(time.monotonic() - began, 2), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)

    try:
        if job.get("stage") != "train_group":
            raise ValueError("Seed orchestration only supports train_group")
        if job.get("model") not in ("skeleton", "context-fusion") or job.get("seeds") != SEEDS:
            raise ValueError("Seed group requires skeleton/context-fusion and seeds [0, 1, 2]")
        budget = float(job.get("stage_seconds", SESSION_SECONDS))
        if not math.isfinite(budget) or budget <= 0 or budget > SESSION_SECONDS:
            raise ValueError(f"Global session budget must be positive and at most {SESSION_SECONDS} seconds")
        result["stage_seconds"] = budget
        status("preflight")
        run = bootstrap(job, repo, work)
        sources = mounted_sources(job)
        if any(value.get("stage") not in ("extract", "train_group") for _, value in sources.values()):
            raise ValueError("Group training accepts extraction and latest group sources only")
        shards = [root / "shards" / "*.npz" for root, value in sources.values()
                  if value.get("stage") == "extract" and value.get("subset") == "training"]
        if not shards or any(value.get("subset") == "testing" for _, value in sources.values()):
            raise ValueError("Training requires fitting shards and forbids test mounts")
        annotations = asset(dataset_root(job["owner"], "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("Annotations changed from the reviewed full-data experiment")
        restore_group(job, sources, work, candidates)
        cache_root = cache_directory() / job["model"]
        for seed in SEEDS:
            if candidates[seed]["status"] in ("complete", "needs_review"):
                continue
            remaining = budget - (time.monotonic() - began)
            if remaining <= 0:
                break
            out = work / "candidates" / f"seed{seed}"
            cache = cache_root / f"seed{seed}"
            cache.mkdir(parents=True, exist_ok=True)
            resume = (["--resume", out / "last_multilabel.pt"]
                      if candidates[seed]["status"] == "resumable" else [])
            status("train", active_seed=seed, remaining_seconds=round(remaining, 2))
            # All scientific arguments match feature_training_stage exactly.
            args = [sys.executable, "scripts/train_adl_multilabel.py", "train",
                    "--train-shards", *shards, "--annotations", annotations,
                    "--out", out, "--model", job["model"], "--seed", seed,
                    "--val-subjects", "p04", "p15", "--epochs", "150", "--min-epochs", "40",
                    "--patience", "25", "--scheduler", "plateau", "--lr-patience", "5",
                    "--lr-factor", "0.5", "--min-lr", "0.000001", "--convergence-lr-reductions", "2",
                    "--lr", "0.0003", "--batch-size", "256", "--workers", "2", "--device", "cuda",
                    "--require-full-coverage", "--require-convergence", "--cache-dir", cache,
                    "--stop-after-seconds", remaining, *resume]
            if job["model"] == "context-fusion":
                args.append("--require-context")
            run(args)
            candidates[seed] = candidate_result(out, job["model"], seed)
            status("seed_finished", active_seed=seed)
            if candidates[seed]["status"] == "resumable":
                break
        overall = ("complete" if all(item["status"] == "complete" for item in candidates)
                   else "needs_review" if any(item["status"] == "needs_review" for item in candidates)
                   else "resumable")
        result.pop("active_seed", None)
        status("complete" if overall == "complete" else "needs_review" if overall == "needs_review"
               else "resume_required", status=overall,
               completed_seeds=sum(item["status"] == "complete" for item in candidates))
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
