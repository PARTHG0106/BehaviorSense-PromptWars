"""One finite temporal-fusion experiment; never resumes or submits another job."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from multilabel_kaggle_stage import asset, dataset_root, read_json, sha256, write_json

SESSION_SECONDS = 40500
FINAL_RESERVE_SECONDS = 900
CANDIDATES = [{"loss": loss, "seed": seed} for loss in ("bce", "asl") for seed in (0, 1, 2)]
FIT_SUBJECTS = ["p03", "p06", "p07", "p09", "p12", "p13", "p17", "p19", "p25"]
VALIDATION_SUBJECTS = ["p04", "p15"]


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_source_receipt(expected, actual, extraction_hash):
    if canonical_sha256(actual) != expected["receipt_sha256"]:
        raise ValueError(f"Extraction receipt changed for {expected['ref']}")
    if (actual.get("job_id") != expected["ref"] or actual.get("stage") != "extract"
            or actual.get("status") != "complete" or actual.get("scope_complete") is not True
            or actual.get("source_sha256") != extraction_hash
            or actual.get("subset") != expected["subset"]
            or actual.get("completed_videos") != expected["expected_videos"]
            or actual.get("expected_videos") != expected["expected_videos"]
            or actual.get("partition_count") != expected["partition_count"]
            or actual.get("config_fingerprint") != expected["config_fingerprint"]):
        raise ValueError(f"Invalid completed extraction receipt for {expected['ref']}")
    actual_indices = actual.get("partition_indices", [actual.get("partition_index")])
    if actual_indices != expected["partition_indices"]:
        raise ValueError(f"Extraction partition mismatch for {expected['ref']}")


def feature_sources(job, subset, inputs=Path("/kaggle/input")):
    """Inspect only receipts and then the requested subset's feature files."""
    expected = {entry["ref"]: entry for entry in job["feature_sources"] if entry["subset"] == subset}
    if not expected:
        raise ValueError(f"No {subset} feature sources were frozen")
    partitions = [i for entry in expected.values() for i in entry["partition_indices"]]
    partition_count, videos = (8, 351) if subset == "training" else (4, 185)
    if (sorted(partitions) != list(range(partition_count))
            or sum(entry["expected_videos"] for entry in expected.values()) != videos):
        raise ValueError(f"Incomplete or overlapping {subset} extraction inventory")
    found = {}
    for path in Path(inputs).rglob("stage_result.json"):
        receipt = read_json(path)
        identity = receipt.get("job_id")
        if identity not in expected:
            continue
        if identity in found:
            raise ValueError(f"Duplicate mounted extraction {identity}")
        verify_source_receipt(expected[identity], receipt, job["extraction_source_sha256"])
        found[identity] = path.parent
    if set(found) != set(expected):
        raise FileNotFoundError(f"Missing {subset} extraction mounts: {sorted(set(expected) - set(found))}")
    inventory, patterns, names = [], [], set()
    for identity in sorted(found):
        root = found[identity]
        shards = sorted((root / "shards").glob("*.npz"))
        if len(shards) != expected[identity]["expected_videos"]:
            raise ValueError(f"Incomplete shard files in {identity}")
        patterns.append(root / "shards" / "*.npz")
        for shard in shards:
            if shard.name in names:
                raise ValueError(f"Duplicate feature shard {shard.name}")
            names.add(shard.name)
            inventory.append({"source": identity, "name": shard.name,
                              "bytes": shard.stat().st_size, "sha256": sha256(shard)})
    return patterns, {"subset": subset, "n_videos": videos, "files": inventory,
                      "sha256": canonical_sha256(inventory),
                      "extraction_source_sha256": job["extraction_source_sha256"]}


def bootstrap(job, repo, work):
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("The pinned Blackwell runtime requires Python 3.13")
    import torch
    import numpy as np
    core = {"python": "3.13", "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "numpy": str(np.__version__)}
    if core != job["expected_core_runtime"]:
        raise ValueError(f"Unexpected pinned runtime: {core}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    gpu = torch.cuda.get_device_name(0)
    if ("RTX PRO 6000" not in gpu or torch.cuda.get_device_capability(0) != (12, 0)
            or "sm_120" not in torch.cuda.get_arch_list()):
        raise RuntimeError(f"Expected native RTX PRO 6000 Blackwell, got {gpu}")
    value = torch.ones((64, 64), device="cuda")
    if not bool(torch.isfinite(value @ value).all()):
        raise RuntimeError("CUDA arithmetic preflight failed")
    environment = {"gpu": gpu, "gpu_capability": [12, 0],
        "native_sm_120": True, "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "core_runtime": core, "source_sha256": job["source_sha256"],
        "extraction_source_sha256": job["extraction_source_sha256"]}
    write_json(work / "environment.json", environment)
    print(json.dumps(environment), flush=True)

    def run(arguments, timeout):
        print("RUN", " ".join(map(str, arguments)), flush=True)
        subprocess.run(list(map(str, arguments)), cwd=repo, check=True, timeout=timeout)
    return run


def candidate_result(out, candidate):
    manifest, state = read_json(out / "manifest.json"), read_json(out / "training_status.json")
    selection = manifest.get("checkpoint_selection", {})
    if (manifest.get("loss") != candidate["loss"] or manifest.get("seed") != candidate["seed"]
            or manifest.get("model") != "temporal-fusion"
            or selection.get("training_seed") != candidate["seed"]):
        raise ValueError("Trainer candidate identity mismatch")
    converged = (state.get("status") == "complete"
                 and state.get("inference_checkpoint_exported") is True
                 and manifest.get("convergence", {}).get("converged") is True
                 and state.get("convergence", {}).get("converged") is True)
    result = {**candidate, "status": "complete" if converged else "incomplete",
              "training_status": state.get("status"), "convergence": manifest.get("convergence", {})}
    if not converged:
        if (out / "last_multilabel.pt").is_file():
            result["last_checkpoint_sha256"] = sha256(out / "last_multilabel.pt")
        return result
    if (manifest.get("full_eligible_window_coverage_verified") is not True
            or manifest.get("test_loaded_during_training") is not False
            or selection.get("source_split") != "validation"
            or selection.get("test_metrics_consulted") is not False
            or selection.get("threshold_source_split") != "validation"
            or not selection.get("validation_comparison_fingerprint")):
        raise ValueError("Training selection/coverage evidence is invalid")
    for split, subjects, windows, videos in (("train", FIT_SUBJECTS, 217090, 274),
                                            ("validation", VALIDATION_SUBJECTS, 63220, 77)):
        report = manifest.get(split, {})
        if (report.get("subjects") != subjects or report.get("n_windows") != windows
                or report.get("n_videos") != videos):
            raise ValueError(f"Unexpected {split} coverage or subjects")
    metrics = read_json(out / "validation_metrics.json")
    pair = metrics.get("tv_drinking_colabeled_windows", {})
    scores = {"validation_macro_ap": metrics.get("macro_ap"), "pair_f1": pair.get("f1"),
              "pair_precision": pair.get("precision")}
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("Candidate validation scores must be finite")
    if not math.isclose(selection.get("best_score", -1), scores["validation_macro_ap"], abs_tol=1e-8):
        raise ValueError("Validation metrics differ from checkpoint selection score")
    relative = f"candidates/{candidate['loss']}-seed{candidate['seed']}/best_multilabel.pt"
    result.update(scores, checkpoint_relative_path=relative,
        checkpoint_sha256=sha256(out / "best_multilabel.pt"),
        thresholds_sha256=sha256(out / "thresholds.json"),
        validation_metrics_sha256=sha256(out / "validation_metrics.json"),
        validation_predictions_sha256=sha256(out / "validation_logits.npz"),
        validation_comparison_fingerprint=selection["validation_comparison_fingerprint"])
    return result


def select_candidate(job, candidates, features):
    if ([{key: item[key] for key in ("loss", "seed")} for item in candidates] != CANDIDATES
            or any(item.get("status") != "complete" for item in candidates)):
        raise ValueError("All six prescribed candidates must converge before selection")
    if len({item["validation_comparison_fingerprint"] for item in candidates}) != 1:
        raise ValueError("Candidates used different validation rows")
    baseline = job["baseline"]
    passing = [item for item in candidates
               if item["validation_macro_ap"] > baseline["validation_macro_ap"]
               and item["pair_f1"] >= baseline["pair_f1"]
               and item["pair_precision"] >= baseline["pair_precision"]]
    selected = max(passing or candidates, key=lambda item: item["validation_macro_ap"])
    return {**selected, "job_id": job["job_id"], "source_split": "validation",
            "gate_passed": bool(passing), "n_gate_passing": len(passing), "baseline": baseline,
            "selection_rule": "highest validation mAP among candidates with strict mAP improvement and nondecreasing pair F1/precision; otherwise highest mAP research candidate",
            "threshold_source_split": "validation", "test_metrics_consulted": False,
            "test_benchmark_previously_inspected": True, "serving_model_replaced": False,
            "source_sha256": job["source_sha256"], "annotation_sha256": job["annotation_sha256"],
            "training_features_sha256": features["sha256"]}


def training_allowance(budget, elapsed, remaining_candidates):
    return max(0., (budget - elapsed - FINAL_RESERVE_SECONDS) / remaining_candidates)


def execute(job, repo, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    candidates = [{**item, "status": "pending"} for item in CANDIDATES]
    result = {"job_id": job["job_id"], "stage": "temporal_single_session",
              "source_sha256": job["source_sha256"], "extraction_source_sha256": job["extraction_source_sha256"],
              "status": "running", "candidates": candidates, "automatic_continuation": False,
              "test_features_loaded": False, "serving_model_replaced": False}

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(time.monotonic() - began, 2), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "candidates"}), flush=True)

    try:
        budget = float(job["session_seconds"])
        if not math.isfinite(budget) or not FINAL_RESERVE_SECONDS < budget <= SESSION_SECONDS:
            raise ValueError("Invalid single-session time budget")
        if job.get("candidates") != CANDIDATES:
            raise ValueError("Candidate schedule differs from six prespecified loss/seed combinations")
        if (work / "frozen_selection.json").exists() or (work / "candidates").exists():
            raise ValueError("This single-session experiment never resumes existing training output")
        status("preflight")
        run = bootstrap(job, repo, work)
        annotations = asset(dataset_root(job["owner"], "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("Annotation hash changed")
        shards, features = feature_sources(job, "training")
        write_json(work / "training_feature_inventory.json", features)
        cache = Path("/kaggle/temp/temporal_cache")
        for index, candidate in enumerate(CANDIDATES):
            elapsed = time.monotonic() - began
            allowance = training_allowance(budget, elapsed, len(CANDIDATES) - index)
            if allowance < 60:
                status("incomplete_budget", status="incomplete", reason="No full candidate allocation remains")
                return result
            out = work / "candidates" / f"{candidate['loss']}-seed{candidate['seed']}"
            status("train", active_candidate=candidate, allowance_seconds=round(allowance, 2))
            args = [sys.executable, "scripts/train_multilabel_temporal.py", "train",
                    "--train-shards", *shards, "--annotations", annotations, "--out", out,
                    "--loss", candidate["loss"], "--seed", candidate["seed"], "--epochs", "80",
                    "--min-epochs", "40", "--patience", "20", "--batch-size", "512", "--workers", "4",
                    "--context-radius", "6", "--neighbor-dropout", "0.1", "--device", "cuda",
                    "--require-full-coverage", "--cache-dir", cache, "--stop-after-seconds", allowance]
            try:
                run(args, timeout=min(allowance + 120, budget - elapsed - 600))
            except subprocess.TimeoutExpired:
                candidates[index] = {**candidate, "status": "incomplete", "training_status": "process_timeout"}
                status("incomplete_budget", status="incomplete", reason="Candidate exceeded its finite time allocation")
                return result
            candidates[index] = candidate_result(out, candidate)
            if candidates[index]["status"] != "complete":
                status("incomplete_training", status="incomplete", reason="A candidate did not converge; no selection or test evaluation")
                return result
            status("candidate_complete")
        selection = select_candidate(job, candidates, features)
        write_json(work / "frozen_selection.json", selection)
        result["selection"] = selection
        result.pop("active_candidate", None)
        status("selection_frozen")
        if any(item["subset"] == "testing" for item in job["feature_sources"]):
            # Test files are first accessed after all six candidates converge and this receipt exists.
            tests, test_features = feature_sources(job, "testing")
            write_json(work / "test_feature_inventory.json", test_features)
            result["test_features_loaded"] = True
            checkpoint = work / selection["checkpoint_relative_path"]
            if sha256(checkpoint) != selection["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed after freezing")
            remaining = budget - (time.monotonic() - began)
            if remaining < 60:
                status("evaluation_budget_exhausted", status="incomplete", training_complete=True)
                return result
            status("evaluate_reused_test_benchmark")
            run([sys.executable, "scripts/train_multilabel_temporal.py", "evaluate-test",
                 "--checkpoint", checkpoint, "--test-shards", *tests, "--annotations", annotations,
                 "--out", work / "test", "--device", "cuda", "--require-full-coverage",
                 "--cache-dir", cache / "test"], timeout=remaining)
            report = read_json(work / "test" / "test_metrics.json")
            if (report.get("checkpoint_sha256") != selection["checkpoint_sha256"]
                    or report.get("thresholds_refit") is not False
                    or report.get("checkpoint_selection_on_test") is not False):
                raise ValueError("Test evaluation did not preserve frozen selection")
            summary_fields = ("metrics", "split", "checkpoint_sha256", "thresholds_refit",
                              "checkpoint_selection_on_test", "full_eligible_window_coverage_verified",
                              "model", "training_convergence")
            result["test"] = {key: report[key] for key in summary_fields if key in report}
            result["test_report_relative_path"] = "test/test_metrics.json"
            result["test_report_sha256"] = sha256(work / "test" / "test_metrics.json")
        status("complete", status="complete", training_complete=True, all_six_converged=True,
               improvement_gate_passed=selection["gate_passed"], exploratory_test_benchmark=True)
        return result
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
