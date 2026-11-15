"""One RGB experiment with real-video throughput and quality gates before expansion."""
from __future__ import annotations

import glob
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from multilabel_kaggle_stage import asset, dataset_root, read_json, sha256, write_json
from temporal_kaggle_stage import bootstrap as core_bootstrap, feature_sources, FIT_SUBJECTS, VALIDATION_SUBJECTS


GATE_CRITERIA = {"macro_ap_gain_over_matched_control": 0.03, "macro_ap_min": 0.6379390777,
                 "pair_f1_min": 0.30, "pair_precision_min": 0.30, "pair_recall_min": 0.30,
                 "subjects": {"p04": {"pair_f1_min": 0.21808510638, "macro_ap_min": 0.59126160643},
                              "p15": {"pair_f1_min": 0.03861003861, "macro_ap_min": 0.64502818523}}}
PROFILE_PROTOCOL = "native-border-profile.v2"
PREFETCH_PROTOCOL = "native-prefetch-profile.v3"
PROFILE_PROTOCOLS = (PROFILE_PROTOCOL, PREFETCH_PROTOCOL)


def validate_job(job):
    protocol = job.get("rgb_protocol")
    if protocol not in (None, *PROFILE_PROTOCOLS):
        raise ValueError("Unknown frozen RGB experiment protocol")
    encoders = ["swin3d_t"] if protocol in PROFILE_PROTOCOLS else ["swin3d_b", "swin3d_t"]
    if job.get("prefetch_batches", 0) != (2 if protocol == PREFETCH_PROTOCOL else 0):
        raise ValueError("Unexpected frozen prefetch queue bound")
    if protocol in PROFILE_PROTOCOLS and job.get("opencv_threads") != 1:
        raise ValueError("Profile protocol requires the fixed one-thread OpenCV runtime")
    if (job.get("session_seconds") != 40500 or job.get("pilot_seconds") != 4500
            or job.get("probe_seconds") != 900 or job.get("automatic_continuation") is not False
            or job.get("encoder_preference") != encoders
            or job.get("throughput_safety_factor") != 1.25
            or job.get("full_head_and_export_reserve") != 3600
            or job.get("pilot_head_reserve") != 600 or job.get("batch_size") != 8):
        raise ValueError("Unexpected finite RGB experiment protocol")
    if job.get("gate_criteria") != GATE_CRITERIA:
        raise ValueError("RGB validation gates differ from the frozen material-improvement requirements")
    for key in ("source_sha256", "extraction_source_sha256", "annotation_sha256", "assets_manifest_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(job.get(key, ""))):
            raise ValueError(f"Invalid frozen provenance hash: {key}")
    if job["source_sha256"] == job["extraction_source_sha256"]:
        raise ValueError("New RGB source must not impersonate the original extraction source")
    records = job["video_inventory"]
    by_id = {item["video_id"]: item for item in records}
    if len(by_id) != len(records) or len(records) != 536:
        raise ValueError("RGB inventory must identify each of 536 videos exactly once")
    for item in records:
        if (not re.fullmatch(r"P\d{2}T\d{2}C\d{2}", str(item["video_id"]))
                or item["subject"] != "p" + item["video_id"][1:3]
                or type(item["n_windows"]) is not int or item["n_windows"] < 1
                or item["source_filename"] != item["video_id"] + ".mp4"
                or Path(item["shard_filename"]).name != item["shard_filename"]
                or any(not re.fullmatch(r"[0-9a-f]{64}", str(item[key]))
                       for key in ("source_sha256", "shard_sha256"))):
            raise ValueError("Invalid video/shard identity in frozen RGB inventory")
    for subset, videos, windows in (("training", 351, 280310), ("testing", 185, 153537)):
        selected = [item for item in records if item["subset"] == subset]
        if len(selected) != videos or sum(item["n_windows"] for item in selected) != windows:
            raise ValueError(f"Incomplete {subset} inventory")
    pilot, probes = job["pilot"]["pilot_video_ids"], job["pilot"]["probe_video_ids"]
    if (len(pilot) != (60 if "coverage" in job else 36) or len(set(pilot)) != len(pilot)
            or not set(pilot) <= set(by_id)
            or len(probes) != 3 or len(set(probes)) != 3 or not set(probes) <= set(pilot)):
        raise ValueError("Invalid whole-video pilot/probe inventory")
    if any(by_id[vid]["subject"] not in FIT_SUBJECTS or by_id[vid]["subset"] != "training" for vid in pilot):
        raise ValueError("Pilot selection must contain fitting videos only")
    if "coverage" not in job and any(sum(by_id[vid]["subject"] == subject for vid in pilot) != 4 for subject in FIT_SUBJECTS):
        raise ValueError("Pilot must contain exactly four videos per fitting subject")
    if any(not 100 <= by_id[vid]["n_windows"] <= 1000 for vid in probes):
        raise ValueError("Probe videos must have 100–1000 retained windows")
    if len({by_id[vid]["subject"] for vid in probes}) != 3:
        raise ValueError("Probe videos must cover three different fitting subjects")
    validation = [item for item in records if item["subject"] in VALIDATION_SUBJECTS]
    if (len(validation) != 77 or sum(item["n_windows"] for item in validation) != 63220
            or any(item["subset"] != "training" for item in validation)):
        raise ValueError("Pilot must include the complete frozen validation split")
    if "coverage" in job:
        from rgb_coverage_contract import validate_coverage_contract
        validate_coverage_contract(job)
    return by_id


def prepare_assets(job, repo, work):
    run = core_bootstrap(job, repo, work)
    root = dataset_root(job["owner"], job["asset_dataset_slug"])
    manifest_path = asset(root, "asset-manifest.json")
    if sha256(manifest_path) != job["assets_manifest_sha256"]:
        raise ValueError("Mounted RGB asset manifest differs from frozen job")
    manifest = read_json(manifest_path)
    if manifest != job["assets"]:
        raise ValueError("Mounted encoder assets differ from frozen asset contract")
    for name, expected in manifest["files"].items():
        path = mounted_asset(root, name, manifest)
        if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Mounted RGB asset checksum mismatch: {name}")
    try:
        version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        version = None
    if version != manifest["torchvision"]:
        # Kaggle removes '+' from mounted filenames. Pip requires the original
        # PEP 427 version spelling, so restore only this verified wheel's name.
        wheel = work / "runtime_wheels" / manifest["torchvision_wheel"]
        wheel.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mounted_asset(root, manifest["torchvision_wheel"], manifest), wheel)
        run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall",
             wheel], timeout=180)
    import torchvision
    if torchvision.__version__ != manifest["torchvision"]:
        raise RuntimeError("The matched offline torchvision runtime did not load")
    environment = read_json(work / "environment.json")
    environment.update(torchvision=torchvision.__version__, assets_manifest_sha256=job["assets_manifest_sha256"])
    write_json(work / "environment.json", environment)
    print(json.dumps(environment), flush=True)
    return root, run


def mounted_asset(root, name, manifest):
    """Resolve the sole observed Kaggle '+' filename normalization for the wheel."""
    if name != manifest["torchvision_wheel"]:
        return asset(root, name)
    candidates = [path for spelling in (name, name.replace("+", ""))
                  for path in Path(root).rglob(spelling) if path.is_file()]
    if len(candidates) != 1:
        raise ValueError("Missing or ambiguous matched torchvision wheel mount")
    return candidates[0]


def bind_shards(job, subset, work):
    patterns, inventory = feature_sources(job, subset)
    write_json(work / f"{subset}_pose_feature_inventory.json", inventory)
    by_name = {}
    for pattern in patterns:
        for name in glob.glob(str(pattern)):
            path = Path(name)
            if path.name in by_name:
                raise ValueError("Duplicate mounted shard filename")
            by_name[path.name] = path
    bound = {}
    for record in job["video_inventory"]:
        if record["subset"] == subset:
            path = by_name.get(record["shard_filename"])
            if path is None or sha256(path) != record["shard_sha256"]:
                raise ValueError(f"Wrong source shard for {record['video_id']}")
            bound[record["video_id"]] = path
    return patterns, bound


def bind_videos(job, subset):
    root = dataset_root(job["owner"], "rgb-untrimmed")
    found = {}
    expected = {record["source_filename"]: record["video_id"] for record in job["video_inventory"]
                if record["subset"] == subset}
    for path in root.rglob("*.mp4"):
        if path.name in expected:
            vid = expected[path.name]
            if vid in found:
                raise ValueError(f"Ambiguous RGB source video {vid}")
            found[vid] = path
    if set(found) != set(expected.values()):
        raise ValueError(f"Raw RGB mount does not contain all expected {subset} videos")
    return found


def projected_budget(job, reports, elapsed, pilot_ids, by_id, reused_reports=None):
    if "coverage" in job:
        from rgb_coverage_budget import projected_coverage_budget
        return projected_coverage_budget(job, reports, elapsed, pilot_ids, by_id, reused_reports)
    if not reports or any(r.get("reused") or r.get("status") != "complete" or r["n_windows"] <= 0 for r in reports):
        raise ValueError("Throughput requires fresh complete real-video extraction reports")
    if any(not math.isfinite(r["elapsed_seconds"]) or r["elapsed_seconds"] <= 0 for r in reports):
        raise ValueError("Throughput timings must be positive and finite")
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Elapsed experiment time must be finite and nonnegative")
    if (len({r["video_id"] for r in reports}) != len(reports)
            or any(r["video_id"] not in job["pilot"]["probe_video_ids"]
                   or r["n_windows"] != by_id[r["video_id"]]["n_windows"] for r in reports)):
        raise ValueError("Throughput reports must describe distinct prescribed complete probe videos")
    seconds_per_window = max(r["elapsed_seconds"] / r["n_windows"] for r in reports) * job["throughput_safety_factor"]
    completed = {r["video_id"] for r in reports}
    full_windows = sum(item["n_windows"] for vid, item in by_id.items() if vid not in completed)
    pilot_windows = sum(by_id[vid]["n_windows"] for vid in pilot_ids if vid not in completed)
    total = (elapsed + full_windows * seconds_per_window + job["full_head_and_export_reserve"]
             + job["pilot_head_reserve"])
    pilot = elapsed + pilot_windows * seconds_per_window + job["pilot_head_reserve"]
    return {"seconds_per_window_conservative": seconds_per_window,
            "projected_job_seconds": total, "projected_pilot_seconds": pilot,
            "job_fits": total <= job["session_seconds"], "pilot_fits": pilot <= job["pilot_seconds"],
            "passed": total <= job["session_seconds"] and pilot <= job["pilot_seconds"],
            "estimator": "1.25 times slowest fresh whole-probe-video elapsed seconds per retained window"}


def validate_report(report, record, encoder_info):
    if (report.get("status") != "complete" or report.get("video_id") != record["video_id"]
            or report.get("subject") != record["subject"] or report.get("n_windows") != record["n_windows"]
            or report.get("source_video_sha256") != record["source_sha256"]
            or report.get("source_shard_sha256") != record["shard_sha256"]
            or report.get("encoder_sha256") != encoder_info["weights_sha256"]
            or report.get("preprocessing_sha256") != encoder_info["preprocessing_sha256"]):
        raise ValueError(f"RGB extraction identity/coverage differs for {record['video_id']}")


def verify_full_selection(directory):
    selection = read_json(directory / "frozen_selection.json")
    if type(selection.get("gate_passed")) is not bool:
        raise ValueError("Full training must record the material validation gate decision")
    if not selection["gate_passed"]:
        return selection
    if selection.get("checkpoint_relative_path") != "best_rgb_multilabel.pt":
        raise ValueError("Full checkpoint path is not canonical")
    if (sha256(directory / "best_rgb_multilabel.pt") != selection.get("checkpoint_sha256")
            or sha256(directory / "thresholds.json") != selection.get("thresholds_sha256")):
        raise ValueError("Full checkpoint/threshold hashes differ from frozen selection")
    return selection


def training_status(directory):
    path = directory / "training_status.json"
    return read_json(path).get("status") if path.is_file() else None


def _receipt(path):
    """A parent may observe a receipt before the child's atomic replace."""
    try:
        value = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _terminate_process_group(process):
    # The worker and every trainer it launches share this new POSIX group.
    # SIGKILL also stops opaque decoder/CUDA calls that cannot check deadlines.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def coverage_probe_receipts_valid(job, work, choice):
    """Do not extend the parent deadline from an incomplete coverage probe."""
    if "coverage" not in job:
        return True
    if not choice:
        return False
    work = Path(work)
    cache = _receipt(work / "cache_reuse.json")
    sparse = _receipt(work / "sparse_probe_report.json")
    if cache is None or sparse is None:
        return False
    expected = {r["video_id"] for r in job["coverage"]["cache_reuse"]["records"]}
    probes = job["coverage"]["probe_video_ids"]
    if (cache.get("status") != "complete" or cache.get("verified") is not True
            or cache.get("kernel_source") != job["coverage"]["cache_reuse"]["kernel_source"]
            or cache.get("verified_source_files") != 113 or cache.get("installed_files") != 110
            or cache.get("test_features_loaded") is not False
            or cache.get("fresh_probe_video_ids") != sorted(probes)
            or choice.get("fresh_probe_video_ids") != probes
            or sparse.get("status") != "complete" or sparse.get("job_id") != job["job_id"]
            or sparse.get("source_sha256") != job["source_sha256"]
            or sparse.get("probe_video_ids") != job["coverage"]["sparse_probe_video_ids"]
            or sparse.get("fresh_complete_videos") != 2 or sparse.get("all_retained_windows_processed") is not True):
        return False
    records = cache.get("records", [])
    if (len(records) != 113 or {r.get("video_id") for r in records} != expected
            or {r.get("video_id") for r in records if r.get("installed") is True} != expected - set(probes)
            or any(r.get("source_sha256_verified") is not True or r.get("array_validation_verified") is not True for r in records)):
        return False
    stamp = sparse.get("completed_elapsed_seconds")
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 <= stamp <= choice.get("completed_elapsed_seconds", -1) <= job["probe_seconds"]):
        return False
    return (choice.get("cache_reuse_sha256") == sha256(work / "cache_reuse.json")
            and choice.get("sparse_probe_report_sha256") == sha256(work / "sparse_probe_report.json"))


def supervise(job, repo, work, started_at=None):
    """Enforce wall-clock limits outside the one GPU worker and its children."""
    if sys.platform != "linux":
        raise RuntimeError("The Kaggle RGB supervisor requires Linux process-group termination")
    validate_job(job)
    repo, work = Path(repo).resolve(), Path(work).resolve()
    began = time.monotonic() if started_at is None else started_at
    if not math.isfinite(began) or began < 0 or began > time.monotonic():
        raise ValueError("Invalid RGB supervisor start clock")
    if read_json(work / "job.json") != job:
        raise ValueError("Supervisor job differs from the frozen worker job")
    marker = work / "supervision_started.json"
    if marker.exists() or (work / "execution_started.json").exists():
        raise ValueError("RGB experiment cannot resume a supervised execution")
    write_json(marker, {"job_id": job["job_id"], "source_sha256": job["source_sha256"]})
    worker = (
        "import sys; from pathlib import Path; "
        "sys.path[:0]=[str(Path(sys.argv[2])/'scripts'), str(Path(sys.argv[2])/'src')]; "
        "from rgb_kaggle_stage import execute; from multilabel_kaggle_stage import read_json; "
        "execute(read_json(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), started_at=float(sys.argv[4]))"
    )
    process = None
    terminated = False
    probe_passed = pilot_passed = False

    def timely(value, limit):
        stamp = value.get("completed_elapsed_seconds") if value else None
        return (value is not None and value.get("job_id") == job["job_id"]
                and value.get("source_sha256") == job["source_sha256"]
                and type(stamp) in (int, float) and math.isfinite(stamp)
                and 0 <= stamp <= limit and stamp <= time.monotonic() - began)

    def deadline_result(phase, limit):
        previous = _receipt(work / "stage_result.json") or {}
        if previous.get("job_id") != job["job_id"] or previous.get("source_sha256") != job["source_sha256"]:
            previous = {}
        for key in ("test", "test_report_sha256"):
            previous.pop(key, None)
        result = {**previous, "job_id": job["job_id"], "source_sha256": job["source_sha256"],
            "extraction_source_sha256": job["extraction_source_sha256"], "stage": "rgb_single_session",
            "status": phase, "phase": phase, "reason": "Parent supervisor terminated the worker process group at its fixed deadline",
            "elapsed_seconds": time.monotonic() - began, "supervisor_deadline_seconds": limit,
            "hard_timeout": True, "automatic_continuation": False, "serving_model_replaced": False,
            "test_features_loaded": previous.get("test_features_loaded") is True}
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)
        return result

    try:
        while True:
            if not probe_passed:
                choice = _receipt(work / "encoder_selection.json")
                probe_passed = (timely(choice, job["probe_seconds"])
                                and choice.get("projection", {}).get("passed") is True
                                and coverage_probe_receipts_valid(job, work, choice))
            if probe_passed and not pilot_passed:
                receipt = _receipt(work / "pilot_passed.json")
                if timely(receipt, job["pilot_seconds"]):
                    gate = _receipt(work / "pilot" / "gate.json")
                    pilot_passed = (gate is not None and gate.get("passed") is True
                                    and gate.get("criteria") == job["gate_criteria"]
                                    and sha256(work / "pilot" / "gate.json") == receipt.get("gate_sha256"))
            phase, limit = (("throughput_rejected", job["probe_seconds"]) if not probe_passed else
                            ("pilot_budget_exhausted", job["pilot_seconds"]) if not pilot_passed else
                            ("session_budget_exhausted", job["session_seconds"]))
            remaining = began + limit - time.monotonic()
            if remaining <= 0:
                if process is not None:
                    _terminate_process_group(process)
                    terminated = True
                return deadline_result(phase, limit)
            if process is None:
                process = subprocess.Popen([sys.executable, "-u", "-c", worker,
                    str(work / "job.json"), str(repo), str(work), str(began)],
                    cwd=repo, start_new_session=True)
            code = process.poll()
            if code is not None:
                if code != 0:
                    raise subprocess.CalledProcessError(code, process.args)
                result = _receipt(work / "stage_result.json")
                if (result is None or result.get("job_id") != job["job_id"]
                        or result.get("source_sha256") != job["source_sha256"]
                        or result.get("status") not in {"complete", "throughput_rejected", "quality_rejected",
                            "pilot_budget_exhausted", "session_budget_exhausted", "incomplete_epoch_limit"}
                        or (result.get("status") == "complete" and not pilot_passed)):
                    raise RuntimeError("RGB worker exited without a valid terminal result")
                return result
            time.sleep(min(.25, remaining))
    finally:
        if process is not None and not terminated:
            _terminate_process_group(process)


def execute(job, repo, work, started_at=None):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    began = time.monotonic() if started_at is None else started_at
    result = {"job_id": job["job_id"], "source_sha256": job["source_sha256"],
              "extraction_source_sha256": job["extraction_source_sha256"], "stage": "rgb_single_session",
              "status": "running", "automatic_continuation": False, "test_features_loaded": False,
              "serving_model_replaced": False}
    reports_by_arch = {}

    def elapsed():
        return time.monotonic() - began

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(elapsed(), 3), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)

    def stop(phase, reason):
        status(phase, status=phase, reason=reason)
        return result

    try:
        by_id = validate_job(job)
        if any((work / name).exists() for name in ("execution_started.json", "rgb_features", "pilot", "full", "frozen_selection.json")):
            raise ValueError("RGB experiment cannot resume or overwrite previous execution output")
        write_json(work / "execution_started.json", {"job_id": job["job_id"], "source_sha256": job["source_sha256"]})
        status("preflight")
        asset_root, run = prepare_assets(job, repo, work)
        annotations = asset(dataset_root(job["owner"], "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("RGB annotations differ from the frozen source")
        train_patterns, train_shards = bind_shards(job, "training", work)
        raw_videos = bind_videos(job, "training")
        from behaviorsense.models.rgb_action import load_rgb_encoder
        from extract_rgb_action_features import extract_video_features
        import torch
        if job.get("rgb_protocol") in PROFILE_PROTOCOLS:
            import cv2
            cv2.setNumThreads(job["opencv_threads"])
        pilot_fit = job["pilot"]["pilot_video_ids"]
        validation_ids = sorted(vid for vid, item in by_id.items() if item["subject"] in VALIDATION_SUBJECTS)
        pilot_ids = sorted(pilot_fit + validation_ids)
        if len(validation_ids) != 77:
            raise ValueError("Pilot must include all 77 held-out validation videos")
        write_json(work / "pilot_video_ids.json", pilot_fit)
        write_json(work / "gate_config.json", job["gate_criteria"])
        if "coverage" in job:
            write_json(work / "coverage_execution.json", {
                "job_id": job["job_id"], "source_sha256": job["source_sha256"],
                "coverage_schema": job["coverage"]["schema"], "pilot_video_ids": pilot_fit,
                "core_probe_video_ids": job["pilot"]["probe_video_ids"],
                "sparse_probe_video_ids": job["coverage"]["sparse_probe_video_ids"],
                "folds": job["coverage"]["folds"], "test_features_loaded": False,
                "phase_seconds": {k: job[k] for k in ("probe_seconds", "pilot_seconds", "session_seconds")}})
        encoder = encoder_info = choice = None
        reused_reports = {}

        def extract_one(vid, deadline, shards, *, profile=None, fresh=False):
            output = work / "rgb_features" / encoder_info["arch"] / f"{vid}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            if "coverage" in job:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Coverage extraction reached its phase deadline")
                existing = reports_by_arch.get(encoder_info["arch"], {}).get(vid)
                if fresh and output.exists():
                    raise ValueError("Fresh coverage probe cannot use an existing output")
                if existing is not None and not fresh:
                    if sha256(output) != existing["output_sha256"]:
                        raise ValueError("Verified coverage feature changed before reuse: " + vid)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Coverage cache hash reached its phase deadline")
                    return existing
            measured_began = time.monotonic()
            report = extract_video_features(raw_videos[vid], shards[vid], output, encoder, encoder_info,
                batch_size=job["batch_size"], deadline=deadline,
                **({"prefetch_batches": 2} if job.get("rgb_protocol") == PREFETCH_PROTOCOL else {}),
                **({"profile": profile} if profile is not None else {}))
            validate_report(report, by_id[vid], encoder_info)
            if fresh and report.get("reused") is not False:
                raise ValueError("Fresh sparse probe returned a reused measurement")
            if profile is not None:
                report = {**report, "elapsed_seconds": max(report["elapsed_seconds"], time.monotonic() - measured_began),
                          "coverage_timings": profile.summary()}
            reports_by_arch.setdefault(encoder_info["arch"], {})[vid] = report
            write_json(work / f"extraction_{encoder_info['arch']}.json", list(reports_by_arch[encoder_info["arch"]].values()))
            return report

        probes = []
        for arch_index, arch in enumerate(job["encoder_preference"]):
            if elapsed() >= job["probe_seconds"] - 30:
                break
            # A failed Base allocation may leave tensors in reference cycles.
            # Release it before Tiny uses the same, still-finite probe allowance.
            encoder = None
            gc.collect()
            torch.cuda.empty_cache()
            # Reserve time for the second encoder only in the original protocol.
            # The explicit Tiny-only A/B protocol gets the shared remainder.
            remaining_probe = job["probe_seconds"] - elapsed()
            encoder_deadline = began + job["probe_seconds"]
            if arch_index == 0 and len(job["encoder_preference"]) > 1:
                encoder_deadline = time.monotonic() + remaining_probe / 2
            specification = job["assets"]["encoders"][arch]
            fresh = []
            status("throughput_probe", encoder=arch)
            try:
                encoder, encoder_info = load_rgb_encoder(asset(asset_root, specification["filename"]),
                    specification["sha256"], arch, device="cuda")
                if (encoder_info.get("preprocessing") != job["assets"]["preprocessing"]
                        or encoder_info.get("preprocessing_sha256") != job["assets"]["preprocessing_sha256"]):
                    raise ValueError("Loaded encoder preprocessing differs from the frozen asset contract")
                if "coverage" in job:
                    from rgb_cache_reuse import prepare_reused_features
                    status("verify_reusable_features", encoder=arch)
                    reused_reports = prepare_reused_features(job, work, train_shards, encoder_info, encoder_deadline)
                    for vid, report in reused_reports.items():
                        validate_report(report, by_id[vid], encoder_info)
                    reports_by_arch.setdefault(arch, {}).update(reused_reports)
                    status("throughput_probe", encoder=arch, verified_cache_videos=len(reused_reports))
                profile_evidence = None
                if job.get("rgb_protocol") in PROFILE_PROTOCOLS:
                    from rgb_profile_stage import profile_probe_variants

                    def run_variant(variant, vid, output, deadline, profile):
                        return extract_video_features(raw_videos[vid], train_shards[vid], output, encoder, encoder_info,
                            batch_size=job["batch_size"], deadline=deadline, profile=profile,
                            spatial_backend="reference" if variant == "baseline" and job["rgb_protocol"] == PROFILE_PROTOCOL else "native",
                            prefetch_batches=job.get("prefetch_batches", 0) if variant == "optimized" else 0)

                    profile_evidence = profile_probe_variants(job, work, encoder_info, run_variant,
                        started_at=began, deadline=encoder_deadline)
                    if profile_evidence.get("status") == "incomplete_budget":
                        raise TimeoutError("A/B comparison exhausted the shared probe allowance")
                    if (profile_evidence.get("status") != "complete"
                            or profile_evidence.get("all_three_equivalence_checks_passed") is not True):
                        raise ValueError("RGB A/B equivalence must pass before forecasting optimized extraction")
                    fresh = profile_evidence["optimized_reports"]
                    if ([report.get("video_id") for report in fresh] != job["pilot"]["probe_video_ids"]
                            or any(report.get("reused") is not False for report in fresh)):
                        raise ValueError("A/B profiling did not return the three fresh prescribed optimized probes")
                    if "coverage" in job:
                        timed = [r for r in profile_evidence["runs"] if r.get("variant") == "optimized"]
                        if ([r["video_id"] for r in timed] != job["pilot"]["probe_video_ids"]):
                            raise ValueError("Coverage probe timings differ from complete optimized reports")
                        fresh = [{**report, "elapsed_seconds": max(report["elapsed_seconds"], timing["elapsed_wall_seconds"]),
                                  "coverage_timings": timing["timings"]}
                                 for report, timing in zip(fresh, timed)]
                    for report in fresh:
                        validate_report(report, by_id[report["video_id"]], encoder_info)
                        reports_by_arch.setdefault(arch, {})[report["video_id"]] = report
                    write_json(work / f"extraction_{arch}.json", list(reports_by_arch[arch].values()))
                else:
                    for vid in job["pilot"]["probe_video_ids"]:
                        fresh.append(extract_one(vid, encoder_deadline, train_shards))
                if "coverage" in job:
                    from rgb_profile_stage import TimingProfile
                    sparse_reports = []
                    sparse_path = work / "sparse_probe_report.json"
                    write_json(sparse_path, {"status": "running", "job_id": job["job_id"],
                        "source_sha256": job["source_sha256"],
                        "probe_video_ids": job["coverage"]["sparse_probe_video_ids"], "reports": sparse_reports})
                    for vid in job["coverage"]["sparse_probe_video_ids"]:
                        profile = TimingProfile(encoder_info.get("device", "cuda"))
                        report = extract_one(vid, encoder_deadline, train_shards, profile=profile, fresh=True)
                        sparse_reports.append(report)
                        write_json(sparse_path, {"status": "running", "job_id": job["job_id"],
                            "source_sha256": job["source_sha256"],
                            "probe_video_ids": job["coverage"]["sparse_probe_video_ids"], "reports": sparse_reports})
                    if time.monotonic() >= encoder_deadline:
                        raise TimeoutError("Sparse probes exhausted the shared probe deadline")
                    fresh.extend(sparse_reports)
                    write_json(sparse_path, {"status": "complete", "job_id": job["job_id"],
                        "source_sha256": job["source_sha256"],
                        "probe_video_ids": job["coverage"]["sparse_probe_video_ids"],
                        "reports": sparse_reports, "completed_elapsed_seconds": elapsed(),
                        "fresh_complete_videos": 2, "all_retained_windows_processed": True})
            except TimeoutError:
                probes.append({"arch": arch, "status": "probe_deadline", "complete_videos": len(fresh), "videos": fresh})
                write_json(work / "probe_report.json", probes)
                continue
            except Exception as exc:
                if not isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
                    raise
                probes.append({"arch": arch, "status": "cuda_out_of_memory",
                               "complete_videos": len(fresh), "videos": fresh})
                write_json(work / "probe_report.json", probes)
                continue
            projection = projected_budget(job, fresh, elapsed(), pilot_ids, by_id, reused_reports)
            passed = (projection["passed"] and len(fresh) == (5 if "coverage" in job else 3)
                      and elapsed() <= job["probe_seconds"] and time.monotonic() <= encoder_deadline)
            probes.append({"arch": arch, "status": "passed" if passed else "rejected_projection",
                           "videos": fresh, "projection": projection})
            if profile_evidence is not None:
                probes[-1].update(rgb_protocol=job["rgb_protocol"],
                    profile_report_relative_path=f"profiling/{arch}/profile_report.json",
                    profile_report_sha256=sha256(work / "profiling" / arch / "profile_report.json"),
                    all_three_equivalence_checks_passed=True)
            write_json(work / "probe_report.json", probes)
            if passed:
                choice = {"arch": arch, "encoder_info": encoder_info, "projection": projection,
                          "job_id": job["job_id"], "source_sha256": job["source_sha256"],
                          "completed_elapsed_seconds": elapsed(), "selection_used_test_metrics": False}
                if profile_evidence is not None:
                    choice.update(rgb_protocol=job["rgb_protocol"],
                        profile_report_relative_path=probes[-1]["profile_report_relative_path"],
                        profile_report_sha256=probes[-1]["profile_report_sha256"])
                if "coverage" in job:
                    choice.update(cache_reuse_sha256=sha256(work / "cache_reuse.json"),
                                  sparse_probe_report_sha256=sha256(work / "sparse_probe_report.json"),
                                  fresh_probe_video_ids=job["coverage"]["probe_video_ids"])
                write_json(work / "encoder_selection.json", choice)
                break
        if choice is None:
            return stop("throughput_rejected", "No prescribed encoder passed the finite whole-video throughput forecast")
        features_dir = work / "rgb_features" / choice["arch"]
        status("pilot_extract", encoder=choice["arch"], total_pilot_videos=len(pilot_ids))
        try:
            for vid in pilot_ids:
                extract_one(vid, began + job["pilot_seconds"] - job["pilot_head_reserve"], train_shards)
                status("pilot_extract", completed_pilot_videos=sum(v in reports_by_arch[choice["arch"]] for v in pilot_ids))
        except TimeoutError:
            return stop("pilot_budget_exhausted", "Whole pilot extraction did not finish within its reserved time")
        remaining = job["pilot_seconds"] - elapsed() - 30
        if remaining <= 30:
            return stop("pilot_budget_exhausted", "No time remains for the matched pilot-head comparison")
        torch.cuda.empty_cache()
        common = ["--features-dir", features_dir, "--shards", *train_patterns, "--annotations", annotations,
                  "--pilot-video-ids", work / "pilot_video_ids.json", "--gate-config", work / "gate_config.json", "--device", "cuda"]
        status("pilot_train")
        try:
            run([sys.executable, "scripts/train_rgb_multilabel.py", "train", *common,
                 "--out", work / "pilot", "--scope", "pilot", "--stop-after-seconds", remaining], timeout=remaining + 20)
        except subprocess.TimeoutExpired:
            return stop("pilot_budget_exhausted", "Matched pilot heads exceeded the fixed phase deadline")
        gate = read_json(work / "pilot" / "gate.json")
        if type(gate.get("passed")) is not bool:
            raise ValueError("Pilot must export an explicit boolean quality gate")
        result["pilot_gate"] = gate
        result["pilot_gate_sha256"] = sha256(work / "pilot" / "gate.json")
        if not gate["passed"]:
            incomplete = training_status(work / "pilot")
            if incomplete == "incomplete_budget":
                return stop("pilot_budget_exhausted", "Matched pilot heads reached their fixed phase deadline")
            if incomplete == "incomplete_epoch_limit":
                result["training_scope"] = "pilot"
                return stop(incomplete, "Matched pilot heads exhausted their epoch budget without a validation plateau")
            return stop("quality_rejected", gate.get("reason", "Pilot did not meet the material validation criteria"))
        if elapsed() > job["pilot_seconds"]:
            return stop("pilot_budget_exhausted", "Pilot finished after its allowed deadline")
        write_json(work / "pilot_passed.json", {"job_id": job["job_id"],
            "source_sha256": job["source_sha256"], "completed_elapsed_seconds": elapsed(),
            "gate_sha256": result["pilot_gate_sha256"]})
        status("full_training_extract")
        rate = choice["projection"]["seconds_per_window_conservative"]
        test_windows = sum(item["n_windows"] for item in by_id.values() if item["subset"] == "testing")
        test_reserve = choice["projection"].get("projected_test_seconds", rate * test_windows)
        # Preserve the forecast test cost and the entire head/export reserve.
        # Previously completed pilot caches are verified/reused by the extractor.
        full_extract_deadline = began + job["session_seconds"] - test_reserve - job["full_head_and_export_reserve"]
        try:
            for vid in sorted(train_shards):
                extract_one(vid, full_extract_deadline, train_shards)
                status("full_training_extract", completed_training_videos=len(reports_by_arch[choice["arch"]]))
        except TimeoutError:
            return stop("session_budget_exhausted", "Full fitting feature coverage did not finish before the deadline")
        head_budget = min(job["full_head_and_export_reserve"] - 900.,
                          job["session_seconds"] - elapsed() - test_reserve - 900)
        if head_budget < 120:
            return stop("session_budget_exhausted", "Insufficient projected time for full training and held-out evaluation")
        torch.cuda.empty_cache()
        status("full_train")
        try:
            run([sys.executable, "scripts/train_rgb_multilabel.py", "train", *common,
                 "--out", work / "full", "--scope", "full", "--stop-after-seconds", head_budget], timeout=head_budget + 30)
        except subprocess.TimeoutExpired:
            return stop("session_budget_exhausted", "Full head exceeded its finite time allocation")
        selection = verify_full_selection(work / "full")
        if not selection["gate_passed"]:
            incomplete = training_status(work / "full")
            if incomplete == "incomplete_budget":
                return stop("session_budget_exhausted", "Full head reached its finite time allocation")
            if incomplete == "incomplete_epoch_limit":
                result["training_scope"] = "full"
                return stop(incomplete, "Full head exhausted its epoch budget without a validation plateau")
            result["full_gate"] = selection.get("gate", selection.get("gate_summary", {}))
            return stop("quality_rejected", "Full-data candidate did not preserve the material validation improvement")
        selection = {**selection, "checkpoint_relative_path": "full/best_rgb_multilabel.pt",
                     "source_sha256": job["source_sha256"], "annotation_sha256": job["annotation_sha256"],
                     "encoder_selection": choice, "test_benchmark_previously_inspected": True,
                     "checkpoint_selection_on_test": False, "serving_model_replaced": False}
        write_json(work / "frozen_selection.json", selection)
        result["selection"] = selection
        status("selection_frozen")
        test_patterns, test_shards = bind_shards(job, "testing", work)
        raw_videos.update(bind_videos(job, "testing"))
        result["test_features_loaded"] = True
        try:
            for vid in sorted(test_shards):
                extract_one(vid, began + job["session_seconds"] - 600, test_shards)
                status("test_extract", completed_test_videos=sum(v in reports_by_arch[choice["arch"]] for v in test_shards))
        except TimeoutError:
            return stop("session_budget_exhausted", "Test feature extraction did not finish; no partial test score exported")
        remaining = job["session_seconds"] - elapsed() - 30
        if remaining <= 30:
            return stop("session_budget_exhausted", "No time remains for full held-out evaluation")
        checkpoint = work / selection["checkpoint_relative_path"]
        if sha256(checkpoint) != selection["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed before test evaluation")
        torch.cuda.empty_cache()
        status("evaluate_reused_test_benchmark")
        try:
            run([sys.executable, "scripts/train_rgb_multilabel.py", "evaluate-test", "--features-dir", features_dir,
                 "--shards", *test_patterns, "--annotations", annotations, "--checkpoint", checkpoint,
                 "--out", work / "test", "--device", "cuda", "--stop-after-seconds", remaining], timeout=remaining + 20)
        except subprocess.TimeoutExpired:
            return stop("session_budget_exhausted", "Held-out evaluation exceeded its finite time allocation")
        if training_status(work / "test") == "incomplete_budget":
            return stop("session_budget_exhausted", "Held-out evaluation reached its finite time allocation; no test score exported")
        test = read_json(work / "test" / "test_metrics.json")
        if (test.get("checkpoint_sha256") != selection["checkpoint_sha256"]
                or test.get("thresholds_refit") is not False or test.get("checkpoint_selection_on_test") is not False
                or test.get("full_eligible_window_coverage_verified") is not True):
            raise ValueError("Test evaluation did not preserve frozen model/threshold selection")
        if elapsed() > job["session_seconds"]:
            return stop("session_budget_exhausted", "Test evaluation finished after the finite session deadline")
        result["test"] = {key: test[key] for key in ("metrics", "split", "checkpoint_sha256", "thresholds_refit",
                           "checkpoint_selection_on_test", "full_eligible_window_coverage_verified") if key in test}
        result["test_report_sha256"] = sha256(work / "test" / "test_metrics.json")
        status("complete", status="complete", pilot_gate_passed=True, full_gate_passed=True,
               training_videos=351, test_videos=185, single_session=True)
        return result
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
