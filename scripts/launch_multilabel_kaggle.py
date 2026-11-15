"""Build a private, self-contained Kaggle experiment and optionally submit it.

Only Python source and taxonomy facts enter the notebook. Credentials are read locally
for the Kaggle API and are never included in the notebook or its source archive.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import time
import math
import re
import sys
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
SLUG = "behaviorsense-multilabel-tv-drinking-20261007"
TITLE = "BehaviorSense Multilabel TV Drinking 20261007"
TRAINING_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
FULL_PREFIX = "bs-mlfull-20261007"
RUNTIME_DATASET = "behaviorsense-runtime-py313-20261007"
EXPECTED_CORE_RUNTIME = {"python": "3.13", "torch": "2.11.0+cu128", "cuda": "12.8", "numpy": "2.1.3"}
SCRIPTS = (
    "scripts/prepare_skeletons.py",
    "scripts/train_adl.py",
    "scripts/train_adl_multilabel.py",
    "scripts/extract_multilabel_windows.py",
    "scripts/multilabel_kaggle_stage.py",
)


def authenticated_api(credentials: Path):
    data = json.loads(credentials.read_text(encoding="utf-8-sig"))
    if not data.get("username") or not data.get("key"):
        raise ValueError("Kaggle credentials must contain username and key")
    os.environ["KAGGLE_USERNAME"] = data["username"]
    os.environ["KAGGLE_KEY"] = data["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api, data["username"]


def source_bundle() -> tuple[str, dict]:
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / name for name in SCRIPTS]
    paths += [ROOT / "configs/taxonomy.yaml"]
    archive = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            if "__pycache__" in path.parts:
                continue
            if path.suffix not in (".py", ".yaml"):
                raise ValueError(f"Unexpected bundle file: {relative}")
            body = path.read_bytes()
            for name in ("KAGGLE_KEY", "KAGGLE_API_TOKEN"):
                secret = os.environ.get(name, "")
                if secret and secret.encode() in body:
                    raise ValueError(f"Credential value detected in source file {relative}")
            manifest[relative] = hashlib.sha256(body).hexdigest()
            info = zipfile.ZipInfo(relative, date_time=(2026, 10, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(info, body)
    body = archive.getvalue()
    return base64.b64encode(body).decode("ascii"), {
        "sha256": hashlib.sha256(body).hexdigest(), "files": manifest,
        "contains_credentials": False,
    }


def write_json(path, value):
    temporary = Path(path).with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def api_document(value):
    return value if isinstance(value, dict) else json.loads(str(value))


class OutputPendingError(RuntimeError):
    """Kaggle has completed a job but has not published its result file yet."""


def retryable_error(exc):
    if isinstance(exc, OutputPendingError):
        return True
    import requests
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        return getattr(exc.response, "status_code", None) in (408, 429, 500, 502, 503, 504)
    return False


def planned_stages():
    stages = [{"key": "preflight", "stage": "preflight"}]
    stages += [{"key": f"xtr{i:02d}", "stage": "extract", "subset": "training",
                "partition_index": i, "partition_count": 8} for i in range(8)]
    stages += [{"key": f"{name}{seed}", "stage": "train", "model": model, "seed": seed}
               for name, model in (("sk", "skeleton"), ("ctx", "context-fusion")) for seed in (0, 1, 2)]
    stages += [{"key": "select", "stage": "select"}]
    stages += [{"key": f"xte{i:02d}", "stage": "extract", "subset": "testing",
                "partition_index": i, "partition_count": 4} for i in range(4)]
    stages += [{"key": "test", "stage": "evaluate"}]
    return [{**stage, "attempts": []} for stage in stages]


def build_plan(directory, owner, runtime_dir, annotations):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "plan.json").exists():
        raise FileExistsError("A frozen plan already exists; use run/status to continue it or a new directory")
    bundle, manifest = source_bundle()
    lock = read_json(runtime_dir / "runtime-lock.json")
    if not any("cp313" in name and name.startswith("onnxruntime_gpu-") for name in lock["files"]):
        raise ValueError("Missing Python 3.13 GPU runtime wheel")
    for name, expected in lock["files"].items():
        if hashlib.sha256((runtime_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Runtime wheel hash mismatch: {name}")
    plan = {"schema_version": 1, "owner": owner, "status": "ready", "next_stage": 0,
            "active": None, "source_sha256": manifest["sha256"], "source_manifest": manifest,
            "runtime_dataset": RUNTIME_DATASET, "runtime_lock": lock,
            "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
            "stages": planned_stages(), "selection": None, "stage_seconds": 33000,
            "protocol": {"selection": "all annotation-eligible windows", "window_cap": None,
                         "sample_fps": 20, "window_frames": 30, "stride": 15,
                         "validation_subjects": ["p04", "p15"], "seeds": [0, 1, 2],
                         "epochs": 150, "min_epochs": 40, "patience": 25,
                         "trained_classes": 15, "test_used_for_selection": False,
                         "gpu": "NvidiaRtxPro6000", "parallel_gpu_jobs": 1,
                         "serving_model_replaced": False}}
    (directory / "source.b64").write_text(bundle, encoding="ascii")
    feature_runner = (ROOT / "scripts/feature_training_stage.py").read_bytes()
    (directory / "feature_training_stage.py").write_bytes(feature_runner)
    plan["feature_runner_sha256"] = hashlib.sha256(feature_runner).hexdigest()
    plan["expected_core_runtime"] = dict(EXPECTED_CORE_RUNTIME)
    write_json(directory / "plan.json", plan)
    return plan


def usable_attempts(stage):
    """Only validated output artifacts may be mounted; failed attempts stay in history."""
    return [attempt for attempt in stage.get("attempts", [])
            if attempt.get("result", {}).get("status") in ("complete", "resumable")]


def stage_sources(plan, stage):
    """Mount only the stages needed by this phase; test data never enters fitting."""
    sources = []
    for previous in plan["stages"]:
        if previous is stage or previous["key"] == stage["key"]:
            if stage["stage"] == "extract":
                sources.extend(item["ref"] for item in usable_attempts(previous))
            elif stage["stage"] in ("train", "train_group"):
                resumable = [item for item in usable_attempts(previous)
                             if item["result"]["status"] == "resumable"]
                if resumable:
                    sources.append(resumable[-1]["ref"])
            continue
        wanted_subset = "training" if stage["stage"] in ("train", "train_group") else (
            "testing" if stage["stage"] == "evaluate" else None)
        if wanted_subset and previous["stage"] == "extract" and previous["subset"] == wanted_subset:
            if not previous.get("complete"):
                raise ValueError(f"Required extraction stage {previous['key']} is incomplete")
            artifacts = usable_attempts(previous)
            if not artifacts or artifacts[-1]["result"]["status"] != "complete":
                raise ValueError(f"Required extraction stage {previous['key']} has no validated complete output")
            sources.extend(item["ref"] for item in artifacts)
    if stage["stage"] == "evaluate":
        if not plan.get("selection"):
            raise ValueError("Test cannot run before validation-only selection")
        sources.append(plan["selection"]["job_id"])
    return sources


def runtime(bundle, manifest, job, grouped_runner=None, feature_runner=None, seed_runner=None):
    body = f'''# Private, frozen full-coverage candidate. No credentials are embedded.
import base64, hashlib, io, json, os, sys, zipfile
from pathlib import Path
WORK = Path("/kaggle/working")
REPO = WORK / "multilabel_source"
REPO.mkdir(parents=True, exist_ok=True)
BUNDLE = {bundle!r}
MANIFEST = {manifest!r}
JOB = {job!r}
raw = base64.b64decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == MANIFEST["sha256"] == JOB["source_sha256"]
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for name in archive.namelist():
        if not (REPO / name).resolve().is_relative_to(REPO.resolve()):
            raise RuntimeError("Archive path escapes source directory")
    archive.extractall(REPO)
(WORK / "source_manifest.json").write_text(json.dumps(MANIFEST, indent=2))
(WORK / "job.json").write_text(json.dumps(JOB, indent=2))
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
os.environ["PYTHONPATH"] = str(REPO / "src")
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
'''
    if feature_runner is not None:
        if grouped_runner is not None or job.get("stage") not in ("train", "evaluate", "train_group"):
            raise ValueError("Feature-only orchestration is restricted to training and evaluation")
        encoded = base64.b64encode(feature_runner).decode("ascii")
        body += f'''
runner = base64.b64decode({encoded!r})
assert hashlib.sha256(runner).hexdigest() == JOB["feature_runner_sha256"]
(REPO / "scripts" / "feature_training_stage.py").write_bytes(runner)
(WORK / "orchestration_manifest.json").write_text(json.dumps({{
    "sha256": JOB["feature_runner_sha256"], "frozen_source_sha256": MANIFEST["sha256"],
    "stage": JOB["stage"], "mode": "features_only", "scientific_source_modified": False
}}, indent=2))
'''
        if job["stage"] == "train_group":
            if seed_runner is None:
                raise ValueError("Grouped training requires its frozen seed runner")
            seed_encoded = base64.b64encode(seed_runner).decode("ascii")
            return body + f'''
seed_runner = base64.b64decode({seed_encoded!r})
assert hashlib.sha256(seed_runner).hexdigest() == JOB["seed_runner_sha256"]
(REPO / "scripts" / "seed_training_stage.py").write_bytes(seed_runner)
(WORK / "seed_orchestration_manifest.json").write_text(json.dumps({{
    "sha256": JOB["seed_runner_sha256"], "feature_runner_sha256": JOB["feature_runner_sha256"],
    "frozen_source_sha256": MANIFEST["sha256"], "model": JOB["model"], "seeds": JOB["seeds"]
}}, indent=2))
from seed_training_stage import execute_seed_group
execute_seed_group(JOB, REPO, WORK)
'''
        return body + "from feature_training_stage import execute_features\nexecute_features(JOB, REPO, WORK)\n"
    if grouped_runner is None:
        if job.get("partition_indices"):
            raise ValueError("Grouped extraction requires its frozen orchestration companion")
        return body + "from multilabel_kaggle_stage import execute\nexecute(JOB, REPO, WORK)\n"
    if job.get("stage") != "extract" or not job.get("partition_indices"):
        raise ValueError("An orchestration companion is only valid for grouped extraction")
    encoded = base64.b64encode(grouped_runner).decode("ascii")
    return body + f'''
runner = base64.b64decode({encoded!r})
assert hashlib.sha256(runner).hexdigest() == JOB["group_runner_sha256"]
(REPO / "scripts" / "grouped_multilabel_extract.py").write_bytes(runner)
(WORK / "orchestration_manifest.json").write_text(json.dumps({{
    "sha256": JOB["group_runner_sha256"], "frozen_source_sha256": MANIFEST["sha256"],
    "partition_indices": JOB["partition_indices"], "extractor_modified": False
}}, indent=2))
from grouped_multilabel_extract import execute_group
execute_group(JOB, REPO, WORK)
'''


def prepare_submission(directory, plan, stage):
    number = len(stage["attempts"])
    slug = f"{FULL_PREFIX}-{stage['key']}-a{number:02d}"
    ref = f"{plan['owner']}/{slug}"
    job = {key: plan[key] for key in ("owner", "source_sha256", "runtime_dataset", "runtime_lock",
                                     "annotation_sha256", "stage_seconds")}
    job.update({key: value for key, value in stage.items() if key not in ("attempts", "complete", "result")})
    job.update(job_id=ref, sources=stage_sources(plan, stage))
    if stage["stage"] == "evaluate":
        job["selection"] = plan["selection"]
    bundle = (directory / "source.b64").read_text(encoding="ascii")
    if hashlib.sha256(base64.b64decode(bundle, validate=True)).hexdigest() != plan["source_sha256"]:
        raise ValueError("Frozen source bundle changed; refusing a GPU submission")
    grouped_runner = None
    if stage.get("partition_indices"):
        grouped_runner = (directory / "grouped_multilabel_extract.py").read_bytes()
        expected_hash = plan["group_runner_sha256"]
        if hashlib.sha256(grouped_runner).hexdigest() != expected_hash:
            raise ValueError("Frozen grouped-extraction runner changed")
        job["group_runner_sha256"] = expected_hash
    feature_runner = None
    if stage["stage"] in ("train", "evaluate", "train_group") and plan.get("feature_runner_sha256"):
        feature_runner = (directory / "feature_training_stage.py").read_bytes()
        if hashlib.sha256(feature_runner).hexdigest() != plan["feature_runner_sha256"]:
            raise ValueError("Frozen feature-training runner changed")
        job["feature_runner_sha256"] = plan["feature_runner_sha256"]
        job["expected_core_runtime"] = plan["expected_core_runtime"]
    seed_runner = None
    if stage["stage"] == "train_group":
        if feature_runner is None:
            raise ValueError("Grouped training requires its frozen feature runtime")
        seed_runner = (directory / "seed_training_stage.py").read_bytes()
        if hashlib.sha256(seed_runner).hexdigest() != plan["seed_runner_sha256"]:
            raise ValueError("Frozen seed-training runner changed")
        job["seed_runner_sha256"] = plan["seed_runner_sha256"]
    body = runtime(bundle, plan["source_manifest"], job, grouped_runner, feature_runner, seed_runner)
    compile(body, "training.py", "exec")
    out = directory / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "training.py").write_text(body, encoding="utf-8")
    dataset_names = (("behaviorsense-toyota-meta",) if feature_runner is not None else
                     (RUNTIME_DATASET, "behaviorsense-toyota-meta", "behavioursense-ww"))
    datasets = [f"{plan['owner']}/{name}" for name in dataset_names]
    if stage["stage"] == "extract":
        datasets.append(f"{plan['owner']}/rgb-untrimmed")
    # On first creation Kaggle derives the actual slug from the title, even if id
    # names a different slug. Keep these exactly aligned for durable dependencies.
    metadata = {"id": ref, "title": slug.replace("-", " ").title(),
                "code_file": "training.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "machine_shape": "NvidiaRtxPro6000", "docker_image": TRAINING_IMAGE,
                "docker_image_pinning_type": "original", "competition_sources": ["arc-prize-2026-arc-agi-3"],
                "dataset_sources": datasets, "kernel_sources": job["sources"], "model_sources": []}
    write_json(out / "kernel-metadata.json", metadata)
    write_json(out / "job.json", job)
    return ref, out


def select_candidate(plan):
    candidates = []
    for stage in plan["stages"]:
        if stage["stage"] not in ("train", "train_group"):
            continue
        if not stage.get("complete"):
            raise ValueError("All six prespecified model/seed candidates must finish before selection")
        if stage["stage"] == "train":
            candidates.append({"model": stage["model"], "seed": stage["seed"], "result": stage["result"]})
        else:
            for item in stage["result"].get("candidates", []):
                if item.get("status") != "complete" or item.get("model") != stage["model"]:
                    raise ValueError("Grouped candidate is incomplete or belongs to another model")
                candidates.append({"model": item["model"], "seed": item["seed"],
                                   "result": {**item, "job_id": stage["result"]["job_id"]}})
    expected = {(model, seed) for model in ("skeleton", "context-fusion") for seed in (0, 1, 2)}
    if len(candidates) != 6 or {(c["model"], c["seed"]) for c in candidates} != expected:
        raise ValueError("Selection requires exactly the six prespecified model/seed candidates")
    comparison = set()
    records = []
    for stage in candidates:
        result = stage["result"]
        manifest = result["manifest"]
        convergence = manifest["convergence"]
        if not convergence.get("converged", False):
            raise ValueError(f"Candidate {stage['model']}/{stage['seed']} has not established convergence")
        selected = manifest["checkpoint_selection"]
        if selected.get("source_split") != "validation" or selected.get("test_metrics_consulted") is not False:
            raise ValueError("Candidate selection must be exclusively validation based")
        score = float(selected["best_score"])
        if not math.isfinite(score):
            raise ValueError("Candidate validation score is not finite")
        comparison.add(selected["validation_comparison_fingerprint"])
        records.append({"job_id": result["job_id"], "model": stage["model"], "seed": stage["seed"],
                        "validation_macro_ap": score, "checkpoint_sha256": result["checkpoint_sha256"],
                        "checkpoint_relative_path": result.get("checkpoint_relative_path", "candidate/best_multilabel.pt")})
    if len(comparison) != 1:
        raise ValueError("Candidates do not share the same validation targets and split")
    best = sorted(records, key=lambda item: (-item["validation_macro_ap"], item["model"], item["seed"]))[0]
    by_model = {}
    for model in ("skeleton", "context-fusion"):
        scores = [item["validation_macro_ap"] for item in records if item["model"] == model]
        mean = sum(scores) / len(scores)
        by_model[model] = {"seeds": len(scores), "validation_macro_ap_mean": mean,
                           "validation_macro_ap_population_std": math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))}
    return {**best, "criterion": "highest validation macro AP across six prespecified candidates",
            "source_split": "validation", "test_consulted": False, "candidates": records,
            "model_seed_summaries": by_model,
            "validation_comparison_fingerprint": comparison.pop()}


def advance(api, directory, plan):
    """One API transition, persisted before another job can be submitted."""
    if plan["next_stage"] >= len(plan["stages"]):
        plan.update(status="complete", active=None)
        write_json(directory / "plan.json", plan)
        return False
    stage = plan["stages"][plan["next_stage"]]
    if stage["stage"] == "train_group":
        from parallel_training_control import advance_parallel
        return advance_parallel(api, directory, plan)
    if stage["stage"] == "select":
        plan["selection"] = select_candidate(plan)
        write_json(directory / "frozen_selection.json", plan["selection"])
        stage["complete"] = True
        plan["next_stage"] += 1
        write_json(directory / "plan.json", plan)
        return True
    if plan["active"]:
        response = api_document(api.kernels_status(plan["active"]))
        plan["remote_status"] = response
        state = response["status"]
        if state in ("QUEUED", "RUNNING", "INITIALIZING"):
            write_json(directory / "plan.json", plan)
            return False
        if state != "COMPLETE":
            raise RuntimeError(f"{plan['active']}: {state}: {response.get('failureMessage')}")
        out = directory / plan["active"].split("/")[1] / "output"
        out.mkdir(parents=True, exist_ok=True)
        paths, token = api.kernels_output(plan["active"], str(out), file_pattern=r"(^|/)stage_result\.json$",
                                          force=True, quiet=True)
        matches = list(out.rglob("stage_result.json"))
        if not matches:
            raise OutputPendingError("Completed remote stage has not published stage_result.json yet")
        if len(matches) != 1:
            raise RuntimeError("Completed remote stage has no unambiguous stage_result.json")
        result = read_json(matches[0])
        if result["job_id"] != plan["active"] or result["source_sha256"] != plan["source_sha256"]:
            raise ValueError("Completed stage identity/source mismatch")
        if (stage["stage"] in ("train", "evaluate") and plan.get("feature_runner_sha256")
                and result.get("feature_runner_sha256") != plan["feature_runner_sha256"]):
            raise ValueError("Feature-training result does not match the frozen orchestration")
        if stage.get("partition_indices"):
            if (result.get("partition_indices") != stage["partition_indices"]
                    or result.get("partition_count") != stage["partition_count"]
                    or result.get("subset") != stage["subset"]
                    or result.get("orchestration_sha256") != plan["group_runner_sha256"]):
                raise ValueError("Grouped extraction result does not match the reviewed scope/runner")
            if result.get("status") == "complete" and (
                    result.get("scope_complete") is not True
                    or result.get("completed_videos") != stage["expected_videos"]
                    or result.get("expected_videos") != stage["expected_videos"]):
                raise ValueError("Grouped extraction has not completed every expected video")
        stage["attempts"][-1]["result"] = result
        stage["result"] = result
        if result["status"] == "complete":
            stage["complete"] = True
            plan["next_stage"] += 1
        elif result["status"] != "resumable":
            raise RuntimeError(f"Stage requires review: {result.get('error', result.get('phase'))}")
        elif len(stage["attempts"]) >= 16:
            raise RuntimeError("Stage exceeded 16 resume sessions; inspect progress before further compute")
        plan["active"] = None
        write_json(directory / "plan.json", plan)
        print(f"{stage['key']}: {result['status']}", flush=True)
        return True
    ref, submission = prepare_submission(directory, plan, stage)
    # Journal before submission: an uncertain network response must never duplicate a job.
    plan.update(status="submitting", active=ref)
    stage["attempts"].append({"ref": ref, "submitted": False})
    write_json(directory / "plan.json", plan)
    response = api_document(api.kernels_push(str(submission), timeout="43200", acc="NvidiaRtxPro6000"))
    if response.get("error"):
        stage["attempts"][-1]["rejection"] = response
        # A confirmed rejection is distinct from an uncertain network response.
        # Remove it from the attempt list so it cannot become a nonexistent input mount.
        stage.setdefault("rejected_submissions", []).append(stage["attempts"].pop())
        plan["active"] = None
        write_json(directory / "plan.json", plan)
        raise RuntimeError(f"Kaggle submission rejected: {response['error']}")
    actual_ref = str(response.get("ref", "")).removeprefix("/code/")
    if actual_ref != ref:
        plan["actual_submission_ref"] = actual_ref
        raise RuntimeError(f"Kaggle changed the requested kernel identity to {actual_ref}")
    for name in ("invalidDatasetSources", "invalidCompetitionSources", "invalidKernelSources", "invalidModelSources"):
        if response.get(name):
            raise RuntimeError(f"Kaggle rejected required input sources: {name}={response[name]}")
    stage["attempts"][-1].update(submitted=True, response=response)
    plan["status"] = "running"
    write_json(directory / "plan.json", plan)
    print(f"Submitted private stage https://www.kaggle.com/code/{ref}", flush=True)
    return False


@contextmanager
def coordinator_lock(directory):
    """OS releases the lock on process death; a stale PID file cannot strand a plan."""
    lockfile = (directory / "coordinator.lock").open("a+b")
    try:
        lockfile.seek(0)
        if os.name == "nt":
            import msvcrt
            if lockfile.read(1) == b"":
                lockfile.write(b"0"); lockfile.flush()
            lockfile.seek(0)
            msvcrt.locking(lockfile.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        lockfile.close()


def run_coordinator(api, directory, plan, *, poll_seconds=60, once=False, sleep=time.sleep):
    failures = 0
    while True:
        try:
            changed = advance(api, directory, plan)
        except Exception as exc:
            failures += 1
            if once or not retryable_error(exc) or failures > 10:
                raise
            delay = min(300, max(30, poll_seconds) * 2 ** min(failures - 1, 3))
            plan.update(status="reconnecting", retry_count=failures,
                        retry_error=f"{type(exc).__name__}: {exc}")
            write_json(directory / "plan.json", plan)
            print(f"Temporary Kaggle error; retrying the same journaled stage in {delay}s: {exc}", flush=True)
            sleep(delay)
            continue
        if failures:
            failures = 0
            plan.pop("retry_count", None)
            plan.pop("retry_error", None)
            if plan["status"] == "reconnecting":
                plan["status"] = "running" if plan.get("active") else "ready"
            write_json(directory / "plan.json", plan)
        if once or plan["status"] == "complete":
            return
        if not changed:
            sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "push", "run", "status", "logs", "output"))
    parser.add_argument("--credentials", type=Path, default=ROOT / "kaggle-krishna.json")
    parser.add_argument("--directory", type=Path, default=ROOT / "artifacts/training-tv-20261007/full-run")
    parser.add_argument("--runtime-directory", type=Path, default=ROOT / "artifacts/training-tv-20261007/runtime-py313")
    parser.add_argument("--annotations", type=Path, default=ROOT / "artifacts/training-tv-20261007/data/smarthome_CS_51.json")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error("poll-seconds must be at least 30")
    api, owner = authenticated_api(args.credentials)
    directory = args.directory.resolve()
    if args.action == "build":
        plan = build_plan(directory, owner, args.runtime_directory, args.annotations)
        print(f"Built full plan: {len(plan['stages'])} stages, six candidates, one Blackwell job at a time; no window cap")
        return
    plan = read_json(directory / "plan.json")
    if plan["owner"] != owner:
        raise ValueError("Credentials belong to a different experiment owner")
    if args.action in ("run", "push"):
        with coordinator_lock(directory):
            write_json(directory / "coordinator.json", {"pid": os.getpid(), "directory": str(directory)})
            try:
                run_coordinator(api, directory, plan, poll_seconds=args.poll_seconds,
                                once=args.action == "push")
            except Exception as exc:
                plan.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
                write_json(directory / "plan.json", plan)
                raise
        return
    active = plan.get("active")
    active_jobs = plan.get("active_jobs") or ([active] if active else [])
    if args.action == "status":
        print(json.dumps({key: plan.get(key) for key in ("status", "active", "active_jobs", "next_stage", "error", "parallel_errors", "selection")}, indent=2))
        for ref in active_jobs:
            print(ref, api.kernels_status(ref))
    elif not active:
        print("No active stage. Inspect plan.json for completed stage references.")
    elif args.action == "logs":
        if args.follow:
            for event in api.kernels_logs_stream(active):
                print(event.get("data", ""), end="", flush=True)
        else:
            raw = api.kernels_logs(active)
            print("".join(item.get("data", "") for item in json.loads(raw)) if raw else "No persisted logs yet")
    else:
        output = directory / active.split("/")[1] / "output"
        output.mkdir(parents=True, exist_ok=True)
        print(api.kernels_output(active, str(output), force=True, quiet=True))


if __name__ == "__main__":
    # Parallel control imports these helpers; keep exception identities identical
    # when this file is launched as a script instead of imported as a module.
    sys.modules["launch_multilabel_kaggle"] = sys.modules[__name__]
    main()
