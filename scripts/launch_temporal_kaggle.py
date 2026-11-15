"""Freeze and commit one private, finite Kaggle temporal-fusion experiment."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

from launch_multilabel_kaggle import (ROOT, TRAINING_IMAGE, EXPECTED_CORE_RUNTIME,
                                    authenticated_api, api_document, coordinator_lock)
from multilabel_kaggle_stage import read_json, sha256, write_json
from temporal_kaggle_stage import (CANDIDATES, SESSION_SECONDS, canonical_sha256,
                                   verify_source_receipt)

DEFAULT_DIRECTORY = ROOT / "artifacts/temporal-multilabel-20261011"
DEFAULT_BASELINE = ROOT / "artifacts/training-tv-20261007/full-run"
SLUG = "bs-temporal-multilabel-20261011"
SCRIPTS = ("scripts/train_adl.py", "scripts/train_adl_multilabel.py",
           "scripts/train_multilabel_temporal.py", "scripts/multilabel_kaggle_stage.py",
           "scripts/temporal_kaggle_stage.py")


def source_bundle(root=ROOT):
    paths = sorted((root / "src").rglob("*.py")) + [root / name for name in SCRIPTS]
    paths.append(root / "configs/taxonomy.yaml")
    secrets = [value.encode() for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN")
               if (value := os.environ.get(name))]
    for path in root.glob("kaggle*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig")).get("key")
            if value:
                secrets.append(str(value).encode())
        except (ValueError, UnicodeError):
            pass
    manifest, stream = {}, io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            relative, data = path.relative_to(root).as_posix(), path.read_bytes()
            if path.suffix not in (".py", ".yaml") or any(secret in data for secret in secrets):
                raise ValueError(f"Unsafe source bundle input: {relative}")
            if path.suffix == ".py":
                compile(data, relative, "exec")
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
            manifest[relative] = hashlib.sha256(data).hexdigest()
    raw = stream.getvalue()
    return base64.b64encode(raw).decode("ascii"), {"sha256": hashlib.sha256(raw).hexdigest(),
        "files": manifest, "contains_credentials": False, "contains_dataset": False}


def extraction_sources(baseline, include_test=True):
    entries = []
    subsets = ("training", "testing") if include_test else ("training",)
    for subset in subsets:
        for stage in baseline["stages"]:
            if stage.get("stage") != "extract" or stage.get("subset") != subset:
                continue
            receipt = stage.get("result", {})
            if stage.get("complete") is not True:
                raise ValueError(f"Baseline extraction incomplete: {stage['key']}")
            entry = {"ref": receipt["job_id"], "subset": subset,
                     "expected_videos": receipt["expected_videos"],
                     "partition_count": receipt["partition_count"],
                     "partition_indices": receipt.get("partition_indices", [receipt.get("partition_index")]),
                     "config_fingerprint": receipt["config_fingerprint"],
                     "receipt_sha256": canonical_sha256(receipt)}
            verify_source_receipt(entry, receipt, baseline["source_sha256"])
            entries.append(entry)
        selected = [item for item in entries if item["subset"] == subset]
        partitions, videos, sources = (8, 351, 5) if subset == "training" else (4, 185, 4)
        if (len(selected) != sources or sum(item["expected_videos"] for item in selected) != videos
                or sorted(i for item in selected for i in item["partition_indices"]) != list(range(partitions))):
            raise ValueError(f"Expected full {subset} inventory from the frozen baseline")
    return entries


def runtime(bundle, manifest, job):
    return f'''# Private source-only, single-session experiment. No credentials are embedded.
import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
WORK = Path("/kaggle/working")
REPO = WORK / "temporal_source"
REPO.mkdir(parents=True, exist_ok=True)
MANIFEST = {manifest!r}
JOB = {job!r}
raw = base64.b64decode({bundle!r}, validate=True)
assert hashlib.sha256(raw).hexdigest() == MANIFEST["sha256"] == JOB["source_sha256"]
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    assert set(archive.namelist()) == set(MANIFEST["files"])
    for name in archive.namelist():
        if not (REPO / name).resolve().is_relative_to(REPO.resolve()):
            raise ValueError("Source archive path escapes repository")
        data = archive.read(name)
        assert hashlib.sha256(data).hexdigest() == MANIFEST["files"][name]
        target = REPO / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
(WORK / "source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
(WORK / "job.json").write_text(json.dumps(JOB, indent=2))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
os.environ["PYTHONPATH"] = str(REPO / "src")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
from temporal_kaggle_stage import execute
execute(JOB, REPO, WORK)
'''


def build(directory, owner, baseline_directory=DEFAULT_BASELINE, include_test=True, root=ROOT):
    directory, baseline_directory = Path(directory), Path(baseline_directory)
    if (directory / "plan.json").exists():
        raise FileExistsError("A frozen experiment already exists; never rebuild over it")
    baseline = read_json(baseline_directory / "plan.json")
    frozen = read_json(baseline_directory / "frozen_selection.json")
    if frozen != baseline.get("selection"):
        raise ValueError("Baseline selection is not frozen consistently")
    relative = frozen["checkpoint_relative_path"]
    if relative != f"candidates/seed{frozen['seed']}/best_multilabel.pt":
        raise ValueError("Invalid frozen baseline checkpoint path")
    validation = (baseline_directory.parent / "completion-audit" / frozen["job_id"].split("/")[-1]
                  / Path(relative).parent / "validation_metrics.json")
    metrics = read_json(validation)
    if abs(metrics["macro_ap"] - frozen["validation_macro_ap"]) > 1e-10:
        raise ValueError("Baseline validation scores changed")
    pair = metrics["tv_drinking_colabeled_windows"]
    bundle, manifest = source_bundle(root)
    job = {"job_id": f"{owner}/{SLUG}", "owner": owner,
           "stage": "temporal_single_session", "session_seconds": SESSION_SECONDS,
           "source_sha256": manifest["sha256"], "extraction_source_sha256": baseline["source_sha256"],
           "annotation_sha256": baseline["annotation_sha256"],
           "expected_core_runtime": EXPECTED_CORE_RUNTIME, "candidates": CANDIDATES,
           "feature_sources": extraction_sources(baseline, include_test),
           "baseline": {"model": frozen["model"], "seed": frozen["seed"],
                        "checkpoint_sha256": frozen["checkpoint_sha256"],
                        "validation_macro_ap": metrics["macro_ap"], "pair_f1": pair["f1"],
                        "pair_precision": pair["precision"], "validation_metrics_sha256": sha256(validation)},
           "test_benchmark_previously_inspected": True, "automatic_continuation": False,
           "serving_model_replaced": False}
    body = runtime(bundle, manifest, job)
    compile(body, "training.py", "exec")
    metadata = {"id": job["job_id"], "title": SLUG.replace("-", " ").title(),
                "code_file": "training.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": False, "machine_shape": "NvidiaRtxPro6000",
                "docker_image": TRAINING_IMAGE, "docker_image_pinning_type": "original",
                "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "dataset_sources": [f"{owner}/behaviorsense-toyota-meta"],
                "kernel_sources": [item["ref"] for item in job["feature_sources"]], "model_sources": []}
    submission = directory / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    (submission / "training.py").write_text(body, encoding="utf-8")
    write_json(submission / "kernel-metadata.json", metadata)
    write_json(submission / "job.json", job)
    write_json(directory / "source_manifest.json", manifest)
    (directory / "source.b64").write_text(bundle, encoding="ascii")
    plan = {"schema_version": 1, "status": "prepared", "job": job, "submission": "submission",
            "timeout_seconds": 43200, "single_session": True, "automatic_continuation": False,
            "source_manifest": manifest,
            "submission_hashes": {name: sha256(submission / name)
                                  for name in ("training.py", "kernel-metadata.json", "job.json")}}
    write_json(directory / "plan.json", plan)
    return plan


def verify_submission(directory, plan):
    submission = Path(directory) / plan["submission"]
    for name, expected in plan["submission_hashes"].items():
        if sha256(submission / name) != expected:
            raise ValueError(f"Frozen submission changed: {name}")
    if read_json(submission / "job.json") != plan["job"]:
        raise ValueError("Frozen job differs from plan")
    metadata = read_json(submission / "kernel-metadata.json")
    if (metadata.get("is_private") is not True or metadata.get("machine_shape") != "NvidiaRtxPro6000"
            or metadata.get("enable_internet") is not False or plan.get("timeout_seconds") != 43200):
        raise ValueError("Unsafe runtime/submission settings")
    return submission


def submission_http_error(exc):
    """Keep bounded server diagnostics, never request data or raw response bodies."""
    try:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    except Exception:
        return {}
    if type(status_code) is not int or not 100 <= status_code <= 599:
        return {}
    result = {"status_code": status_code}

    def safe_value(value, limit):
        if type(value) is int:
            return value if -999999 <= value <= 999999 else None
        if not isinstance(value, str):
            return None
        for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
            if secret := os.environ.get(name):
                value = value.replace(secret, "[REDACTED]")
        return " ".join(value.split())[:limit]

    try:
        headers = {}
        for name, value in response.headers.items():
            if isinstance(name, str) and name.lower() in (
                    "x-request-id", "x-correlation-id", "x-kaggle-request-id", "x-goog-request-id"):
                value = safe_value(value, 256)
                if value is not None:
                    headers[name.lower()] = value
        if headers:
            result["request_ids"] = headers
    except Exception:
        pass
    try:
        body = response.json()
        if isinstance(body, dict):
            # Some HTTP clients expose the API error under an `error` object.
            fields = {}
            for source in (body, body.get("error")):
                if not isinstance(source, dict):
                    continue
                for name, limit in (("status", 128), ("code", 128), ("message", 1024)):
                    value = safe_value(source.get(name), limit)
                    if value is not None and name not in fields:
                        fields[name] = value
            if fields:
                result["response"] = fields
    except Exception:
        # HTML, malformed JSON, or an unexpected client response remains ambiguous.
        pass
    return result


def push(api, directory):
    directory = Path(directory)
    with coordinator_lock(directory):
        plan = read_json(directory / "plan.json")
        if plan.get("status") != "prepared" or plan.get("submission_attempt"):
            raise ValueError("Submission already attempted. Reconcile its remote identity before any new commit; automatic resubmission is forbidden")
        submission = verify_submission(directory, plan)
        plan.update(status="submitting", submission_attempt={"ref": plan["job"]["job_id"],
                                                            "state": "intent_journaled"})
        write_json(directory / "plan.json", plan)
        try:
            receipt = api_document(api.kernels_push(str(submission), timeout="43200", acc="NvidiaRtxPro6000"))
        except Exception as exc:
            plan.update(status="submission_ambiguous")
            plan["submission_attempt"].update(state="ambiguous", error=f"{type(exc).__name__}: {exc}")
            if diagnostic := submission_http_error(exc):
                plan["submission_attempt"]["http_error"] = diagnostic
            write_json(directory / "plan.json", plan)
            raise
        write_json(directory / "submission_receipt.json", receipt)
        attempt = plan["submission_attempt"]
        attempt["receipt"] = receipt
        invalid = [key for key in ("invalidDatasetSources", "invalidCompetitionSources", "invalidKernelSources",
                                  "invalidModelSources") if receipt.get(key)]
        if receipt.get("error") or invalid:
            plan["status"], attempt["state"] = "submission_rejected", "rejected"
            write_json(directory / "plan.json", plan)
            raise RuntimeError(f"Kaggle rejected commit: {receipt.get('error') or invalid}")
        actual = str(receipt.get("ref", "")).removeprefix("/code/")
        if (actual != plan["job"]["job_id"] or not receipt.get("versionNumber")
                or not receipt.get("kernelId") or not receipt.get("url")):
            plan["status"], attempt["state"] = "submission_ambiguous", "ambiguous"
            write_json(directory / "plan.json", plan)
            raise RuntimeError("Kaggle returned an incomplete or unexpected commit identity; do not resubmit")
        plan["status"], attempt["state"] = "submitted", "accepted"
        write_json(directory / "plan.json", plan)
        return receipt


def status(api, directory):
    directory = Path(directory)
    with coordinator_lock(directory):
        plan = read_json(directory / "plan.json")
        remote = api_document(api.kernels_status(plan["job"]["job_id"]))
        plan["remote_status"] = remote
        if remote.get("status") == "RUNNING":
            plan["status"] = "running"
        elif remote.get("status") == "COMPLETE":
            plan["status"] = "remote_complete_outputs_pending_audit"
        write_json(directory / "plan.json", plan)
        return {"job_id": plan["job"]["job_id"], "remote": remote, "local_status": plan["status"]}


def outputs(api, directory, logs_only=False):
    directory = Path(directory)
    plan = read_json(directory / "plan.json")
    output = directory / "output"
    pattern = r"(?!)" if logs_only else r"(^|/)(stage_result|environment|frozen_selection|training_feature_inventory|test_feature_inventory|manifest|training_status|history|validation_metrics|validation_by_subject|thresholds|test_metrics)\.json$|(^|/)(best_multilabel\.pt|validation_logits\.npz|test_logits\.npz)$"
    paths, _ = api.kernels_output(plan["job"]["job_id"], str(output), file_pattern=pattern,
                                  force=True, quiet=True)
    if logs_only:
        return {"downloaded": paths}
    result = read_json(output / "stage_result.json")
    environment = read_json(output / "environment.json")
    if (result.get("job_id") != plan["job"]["job_id"]
            or result.get("source_sha256") != plan["job"]["source_sha256"]
            or environment.get("source_sha256") != plan["job"]["source_sha256"]
            or "RTX PRO 6000" not in environment.get("gpu", "")
            or environment.get("gpu_capability") != [12, 0]
            or environment.get("native_sm_120") is not True
            or environment.get("core_runtime") != plan["job"]["expected_core_runtime"]):
        raise ValueError("Downloaded job identity or GPU runtime proof differs")
    for candidate in result.get("candidates", []):
        if candidate.get("status") == "complete":
            expected_relative = f"candidates/{candidate['loss']}-seed{candidate['seed']}/best_multilabel.pt"
            if (candidate.get("checkpoint_relative_path") != expected_relative
                    or sha256(output / expected_relative) != candidate["checkpoint_sha256"]):
                raise ValueError("Downloaded checkpoint does not match its completion receipt")
    audit = {"status": result["status"], "gpu_verified": True,
             "all_six_converged": result.get("all_six_converged", False),
             "improvement_gate_passed": result.get("improvement_gate_passed"),
             "source_sha256": result["source_sha256"], "output_result_sha256": sha256(output / "stage_result.json")}
    write_json(directory / "output_audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "push", "status", "logs", "output"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--credentials", type=Path, default=ROOT / "kaggle-krishna.json")
    parser.add_argument("--baseline-directory", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--without-test", action="store_true", help="Build training/validation only")
    args = parser.parse_args()
    if args.action == "build":
        owner = read_json(args.credentials)["username"]
        result = build(args.directory, owner, args.baseline_directory, not args.without_test)
        print(json.dumps({"status": result["status"], "job_id": result["job"]["job_id"],
                          "source_sha256": result["job"]["source_sha256"]}, indent=2))
        return
    api, owner = authenticated_api(args.credentials)
    if args.action in ("push", "status"):
        plan = read_json(args.directory / "plan.json")
        if owner != plan["job"]["owner"]:
            raise ValueError("Authenticated credentials owner differs from the frozen job owner")
    if args.action == "push":
        result = push(api, args.directory)
    elif args.action == "status":
        result = status(api, args.directory)
    else:
        result = outputs(api, args.directory, logs_only=args.action == "logs")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
