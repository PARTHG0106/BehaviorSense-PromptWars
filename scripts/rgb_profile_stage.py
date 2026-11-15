"""Bounded A/B instrumentation inside one RGB job's existing throughput probe.

This module never starts a worker, GPU session, trainer or continuation. Complete
optimized probe reports become eligible for the existing throughput forecast only
after their source clocks and features agree with the baseline implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time

from multilabel_kaggle_stage import write_json
from rgb_kaggle_stage import validate_job, validate_report


FEATURE_RTOL = 1e-3
FEATURE_ATOL = 1e-3
IDENTITY_FIELDS = ("schema", "video_id", "subject", "n_windows", "arch", "feature_dim",
    "source_video_sha256", "source_shard_sha256", "encoder_sha256", "encoder_info",
    "preprocessing_sha256", "source_fps", "source_frame_count", "source_width", "source_height", "views")


class TimingProfile:
    """Thread-safe CPU wall times and explicitly synchronized CUDA event times.

    CPU contexts can surround decode, crop, packing, validation and serialization.
    CUDA contexts should surround transfer, preprocessing, forward and readback in
    the actual encoder. CUDA imports and device access occur only on cuda() use.
    Nested/concurrent stages may overlap; their totals are not an elapsed-time sum.
    """

    def __init__(self, device=None, *, clock=None, cuda_module=None):
        self.device = device
        self._clock = clock or time.perf_counter
        self._cuda = cuda_module
        self._stages = {}
        self._lock = threading.Lock()

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
            raise ValueError("Timing stage names must be short lowercase identifiers")

    def _record(self, name, wall, cuda_seconds, failed, mode):
        if (not math.isfinite(wall) or wall < 0
                or (cuda_seconds is not None and (not math.isfinite(cuda_seconds) or cuda_seconds < 0))):
            raise ValueError("Profile timings must be finite and nonnegative")
        with self._lock:
            value = self._stages.setdefault(name, {"calls": 0, "failed_calls": 0,
                "wall_seconds": 0., "cuda_seconds": 0. if cuda_seconds is not None else None, "timing_mode": mode})
            if value["timing_mode"] != mode:
                raise ValueError("A profile stage cannot mix CPU and CUDA timing modes")
            value["calls"] += 1
            value["failed_calls"] += int(failed)
            value["wall_seconds"] += wall
            if cuda_seconds is not None:
                value["cuda_seconds"] = (value["cuda_seconds"] or 0.) + cuda_seconds

    @contextmanager
    def cpu(self, name):
        self._name(name)
        began = self._clock()
        failed = True
        try:
            yield
            failed = False
        finally:
            self._record(name, self._clock() - began, None, failed, "cpu_wall_clock")

    @contextmanager
    def cuda(self, name):
        self._name(name)
        if self._cuda is None:
            import torch
            self._cuda = torch.cuda
        if not self._cuda.is_available():
            raise RuntimeError("CUDA profiling requires the actual CUDA runtime")
        device = self.device if self.device is not None else "cuda"
        # Synchronize the same device and record both events on its current stream.
        # This is instrumentation overhead, charged to the surrounding probe wall time.
        with self._cuda.device(device):
            self._cuda.synchronize(device)
            start = self._cuda.Event(enable_timing=True)
            end = self._cuda.Event(enable_timing=True)
            start.record()
            began = self._clock()
            failed = True
            try:
                yield
                failed = False
            finally:
                cuda_seconds = None
                if not failed:
                    end.record()
                    self._cuda.synchronize(device)
                    cuda_seconds = float(start.elapsed_time(end)) / 1000.
                self._record(name, self._clock() - began, cuda_seconds, failed,
                             "cuda_events_and_synchronized_wall_clock")

    def summary(self):
        with self._lock:
            stages = {name: dict(value) for name, value in sorted(self._stages.items())}
        return {"schema": "behaviorsense.rgb_stage_timings.v1", "stages": stages,
            "cuda_timing": "events on current device stream; device synchronized before and after each successful CUDA stage",
            "totals_may_overlap": True, "profiling_overhead_included_in_probe_wall_time": True}


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("RGB A/B profiling exhausted its existing probe allowance")


def _file_sha256(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            _deadline(deadline)
            digest.update(block)
    return digest.hexdigest()


def compare_probe_outputs(baseline, optimized, record, encoder_info, deadline):
    """Require original row clocks/crops exactly and float16 features within 1e-3."""
    import numpy as np
    from extract_rgb_action_features import SCHEMA, _array_sha256

    _deadline(deadline)
    loaded = []
    for report in (baseline, optimized):
        validate_report(report, record, encoder_info)
        if report.get("reused") is not False:
            raise ValueError("A/B profiling requires fresh complete-video outputs")
        path = Path(report["output_path"])
        if _file_sha256(path, deadline) != report.get("output_sha256"):
            raise ValueError("Profile output hash changed before A/B verification")
        with np.load(path, allow_pickle=False) as archive:
            if not {"metadata", "features"} <= set(archive.files):
                raise ValueError("Profile output lacks its original extraction identity/features")
            metadata = json.loads(str(archive["metadata"]))
            if (metadata.get("schema") != SCHEMA or metadata.get("status") != "complete"
                    or any(key not in metadata for key in IDENTITY_FIELDS)
                    or metadata.get("encoder_info") != encoder_info):
                raise ValueError("Profile output encoder/source-clock identity is incomplete")
            for key in ("video_id", "subject", "n_windows", "source_video_sha256", "source_shard_sha256",
                        "encoder_sha256", "preprocessing_sha256"):
                if metadata.get(key) != report.get(key):
                    raise ValueError("Profile output metadata differs from its complete-video report")
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files if key != "metadata"}
        if metadata.get("array_sha256") != {key: _array_sha256(value) for key, value in arrays.items()}:
            raise ValueError("Profile output array content differs from its recorded hashes")
        features = arrays["features"]
        if (features.dtype != np.float16
                or features.shape != (record["n_windows"], 2, encoder_info["feature_dim"])
                or not np.isfinite(features).all()):
            raise ValueError("Profile output lacks complete finite float16 features")
        loaded.append((metadata, arrays))
        _deadline(deadline)
    (left_meta, left), (right_meta, right) = loaded
    if any(left_meta[key] != right_meta[key] for key in IDENTITY_FIELDS):
        raise ValueError("A/B implementations changed encoder, source or preprocessing identity")
    if set(left) != set(right):
        raise ValueError("A/B implementations exported different source-clock arrays")
    required = {"starts", "video_id", "subject", "video_ids", "subjects", "center_source_frames",
                "source_frames_rgb", "boundary_duplicate_flags", "person_rois", "roi_observed", "source_fps"}
    if set(left) != required | {"features"}:
        raise ValueError("A/B outputs lack the exact original row/clock/crop schema")
    for key in required:
        if left[key].dtype != right[key].dtype or not np.array_equal(left[key], right[key]):
            raise ValueError(f"A/B implementations changed original rows, clocks or crops: {key}")
    first, second = left["features"].astype(np.float32), right["features"].astype(np.float32)
    if not np.allclose(first, second, rtol=FEATURE_RTOL, atol=FEATURE_ATOL, equal_nan=False):
        raise ValueError("A/B embeddings disagree beyond the frozen feature tolerance")
    _deadline(deadline)
    return {"passed": True, "video_id": record["video_id"], "n_windows": record["n_windows"],
        "source_clock_and_crop_arrays_exact": True, "feature_rtol": FEATURE_RTOL, "feature_atol": FEATURE_ATOL,
        "max_feature_absolute_difference": float(np.max(np.abs(first - second))),
        "baseline_output_sha256": baseline["output_sha256"], "optimized_output_sha256": optimized["output_sha256"]}


def profile_probe_variants(job, work, encoder_info, run_variant, *, started_at, deadline, cuda_module=None):
    """Compare baseline and optimized complete probes inside the current GPU job.

    ``run_variant(variant, video_id, output_path, deadline, profile)`` must call the
    ordinary extractor, returning its complete report. The caller supplies its
    already-loaded encoder; this helper does not load weights or start a session.
    CPU instrumentation uses profile.cpu(name); the encoder uses profile.cuda(name).
    Return optimized_reports only after every A/B identity/feature check succeeds.
    No return value is a quality gate, encoder selection or full-job budget pass.
    """
    by_id = validate_job(job)
    now = time.monotonic()
    if (type(started_at) not in (float, int) or not math.isfinite(started_at) or not 0 <= started_at <= now
            or type(deadline) not in (float, int) or not math.isfinite(deadline)):
        raise ValueError("Profile clocks must be finite absolute monotonic times")
    deadline = min(deadline, started_at + job["probe_seconds"])
    arch = encoder_info.get("arch")
    if (arch not in job["encoder_preference"]
            or encoder_info.get("weights_sha256") != job["assets"]["encoders"][arch]["sha256"]
            or encoder_info.get("feature_dim") != job["assets"]["encoders"][arch]["feature_dim"]
            or encoder_info.get("preprocessing") != job["assets"]["preprocessing"]
            or encoder_info.get("preprocessing_sha256") != job["assets"]["preprocessing_sha256"]):
        raise ValueError("Profiling encoder differs from the frozen job weights")
    work = Path(work)
    directory = work / "profiling" / arch
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("A/B probe cannot resume or overwrite prior profile output")
    directory.mkdir(parents=True, exist_ok=True)
    result = {"schema": "behaviorsense.rgb_probe_profile.v1", "job_id": job["job_id"],
        "opencv_threads": job.get("opencv_threads"),
        "rgb_protocol": job.get("rgb_protocol"), "prefetch_batches": job.get("prefetch_batches", 0),
        "source_sha256": job["source_sha256"], "arch": arch, "status": "running", "comparisons": [], "runs": [],
        "scope": "all original retained windows of the same three prescribed fitting-only probe videos",
        "probe_video_ids": list(job["pilot"]["probe_video_ids"]), "automatic_continuation": False,
        "test_features_loaded": False, "quality_gate_evaluated": False, "throughput_gate_evaluated": False,
        "absolute_deadline": deadline, "deadline_elapsed_seconds": deadline - started_at,
        "feature_rtol": FEATURE_RTOL, "feature_atol": FEATURE_ATOL,
        "order": "baseline then optimized for video 1 and 3; optimized then baseline for video 2",
        "warmup": "no hidden warmup; all cold-start and profiling overhead count toward the shared probe allowance"}
    report_path = directory / "profile_report.json"

    def save():
        result["elapsed_session_seconds"] = time.monotonic() - started_at
        write_json(report_path, result)

    optimized_reports = []
    save()
    try:
        for index, vid in enumerate(job["pilot"]["probe_video_ids"]):
            pair = {}
            for variant in (("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")):
                _deadline(deadline)
                output = ((directory / "baseline") if variant == "baseline" else (work / "rgb_features" / arch)) / f"{vid}.npz"
                if output.exists():
                    raise ValueError("A/B probe requires new outputs for both implementations")
                output.parent.mkdir(parents=True, exist_ok=True)
                profile = TimingProfile(encoder_info.get("device", "cuda"), cuda_module=cuda_module)
                began = time.monotonic()
                report = run_variant(variant, vid, output, deadline, profile)
                _deadline(deadline)
                validate_report(report, by_id[vid], encoder_info)
                if (report.get("reused") is not False or Path(report.get("output_path", "")).resolve() != output.resolve()
                        or type(report.get("elapsed_seconds")) not in (float, int)
                        or not math.isfinite(report["elapsed_seconds"]) or report["elapsed_seconds"] <= 0):
                    raise ValueError("A/B runner must return fresh complete timed reports at its prescribed output path")
                timings = profile.summary()
                stages = timings["stages"]
                required_cpu, required_cuda = ("decode", "serialize"), ("transfer", "forward")
                if (any(stages.get(name, {}).get("calls", 0) < 1 for name in required_cpu + required_cuda)
                        or not any(stages.get(name, {}).get("calls", 0) for name in ("scene_crop", "person_crop"))
                        or any(stages[name]["timing_mode"] != "cpu_wall_clock" for name in required_cpu)
                        or any(stages[name]["timing_mode"] != "cuda_events_and_synchronized_wall_clock" for name in required_cuda)
                        or any(value["failed_calls"] for value in stages.values())):
                    raise ValueError("A/B profile lacks successful decode/crop/transfer/forward/serialize instrumentation")
                pair[variant] = report
                result["runs"].append({"variant": variant, "video_id": vid, "report": report,
                    "elapsed_wall_seconds": time.monotonic() - began, "timings": timings})
                save()
            comparison = compare_probe_outputs(pair["baseline"], pair["optimized"], by_id[vid], encoder_info, deadline)
            comparison["reported_elapsed_speedup"] = pair["baseline"]["elapsed_seconds"] / pair["optimized"]["elapsed_seconds"]
            result["comparisons"].append(comparison)
            optimized_reports.append(pair["optimized"])
            save()
        _deadline(deadline)
        result.update(status="complete", optimized_reports=optimized_reports, all_three_equivalence_checks_passed=True)
        save()
        return result
    except TimeoutError:
        result.update(status="incomplete_budget", all_three_equivalence_checks_passed=False)
        save()
        return result
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", all_three_equivalence_checks_passed=False)
        save()
        raise
