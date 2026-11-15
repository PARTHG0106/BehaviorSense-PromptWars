"""Forecast complete RGB work from fresh frame, batch and video measurements."""
from __future__ import annotations

import math
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_integer(value):
    return type(value) is int and value > 0


def probe_costs(report, record, batch_size):
    """Separate sparse-video decode and fixed overhead from per-batch work."""
    require(positive_integer(record.get("source_frames")), "invalid source-frame count")
    frames = report.get("decoded_frames")
    require(positive_integer(frames) and frames <= record["source_frames"], "invalid decoded-frame count")
    timing = report.get("coverage_timings", {})
    require(timing.get("schema") == "behaviorsense.rgb_stage_timings.v1", "missing fresh coverage timings")
    stages = timing.get("stages", {})
    require(isinstance(stages, dict), "invalid fresh stage timings")
    required = {"decode", "input_validation", "serialize", "forward", "transfer"}
    require(required <= set(stages) and any(k in stages for k in ("scene_crop", "person_crop")),
            "incomplete fresh stage timings")
    for name, stage in stages.items():
        require(isinstance(stage, dict) and positive_integer(stage.get("calls"))
                and stage.get("failed_calls") == 0
                and type(stage.get("wall_seconds")) in (int, float)
                and math.isfinite(stage["wall_seconds"]) and stage["wall_seconds"] >= 0,
                "invalid or failed timing stage: " + name)
        expected_mode = ("cuda_events_and_synchronized_wall_clock" if name in
                         ("forward", "transfer", "preprocess", "readback") else "cpu_wall_clock")
        require(stage.get("timing_mode") == expected_mode, "incorrect timing mode: " + name)
        if expected_mode.startswith("cuda"):
            require(type(stage.get("cuda_seconds")) in (int, float)
                    and math.isfinite(stage["cuda_seconds"]) and stage["cuda_seconds"] >= 0,
                    "missing synchronized CUDA timing: " + name)
    require(stages["decode"]["calls"] == frames, "decode timing does not cover every decoded frame")
    decode = stages["decode"]["wall_seconds"]
    work = sum(v["wall_seconds"] for k, v in stages.items()
               if k not in ("decode", "input_validation", "serialize"))
    unprofiled = max(0., report["elapsed_seconds"] - sum(v["wall_seconds"] for v in stages.values()))
    overhead = stages["input_validation"]["wall_seconds"] + stages["serialize"]["wall_seconds"] + unprofiled
    require(decode > 0 and work > 0, "fresh decode and model work must take positive time")
    batches = (record["n_windows"] + batch_size - 1) // batch_size
    return {"decode_seconds_per_frame": decode / frames,
            "work_seconds_per_batch": work / batches, "overhead_seconds_per_video": overhead,
            "batches": batches, "unprofiled_seconds": unprofiled}


def projected_coverage_budget(job, reports, elapsed, pilot_ids, by_id, reused_reports):
    coverage = job["coverage"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "invalid elapsed coverage time")
    expected_pilot = set(job["pilot"]["pilot_video_ids"]) | {
        vid for vid, row in by_id.items() if row["subject"] in ("p04", "p15")}
    require(len(pilot_ids) == len(set(pilot_ids)) and set(pilot_ids) == expected_pilot,
            "forecast must include exactly all pilot and validation videos")
    probe_ids = coverage["probe_video_ids"]
    require(len(probe_ids) == len(set(probe_ids)) == 5
            and [r.get("video_id") for r in reports] == probe_ids,
            "coverage forecast requires all five ordered fresh probes")
    require(positive_integer(job.get("batch_size")) and job["batch_size"] == 8,
            "unexpected coverage batch size")
    expected_cache = {r["video_id"]: r for r in coverage["cache_reuse"]["records"]}
    expected_reuse = set(expected_cache) - set(probe_ids)
    require(isinstance(reused_reports, dict) and set(reused_reports) == expected_reuse,
            "only the complete verified cache set can be discounted")
    for vid, report in reused_reports.items():
        expected = expected_cache[vid]
        require(vid in expected_pilot and by_id[vid]["subset"] == "training"
                and report.get("video_id") == vid and report.get("status") == "complete"
                and report.get("reused") is True and report.get("cache_reuse_verified") is True
                and report.get("output_sha256") == expected["feature_sha256"]
                and report.get("n_windows") == expected["n_windows"] == by_id[vid]["n_windows"]
                and report.get("source_shard_sha256") == expected["source_shard_sha256"] == by_id[vid]["shard_sha256"]
                and report.get("source_video_sha256") == expected["source_video_sha256"] == by_id[vid]["source_sha256"],
                "unverified, altered or forbidden cache discount: " + vid)
    measurements = []
    for report in reports:
        vid = report["video_id"]
        require(vid in expected_pilot and vid in by_id, "probe outside fitting pilot")
        row = by_id[vid]
        require(row["subject"] in coverage["folds"]["A"] + coverage["folds"]["B"] + coverage["folds"]["C"]
                and row["subset"] == "training"
                and report.get("status") == "complete" and report.get("reused") is False
                and report.get("n_windows") == row["n_windows"]
                and report.get("source_shard_sha256") == row["shard_sha256"]
                and report.get("source_video_sha256") == row["source_sha256"]
                and re.fullmatch(r"[0-9a-f]{64}", str(report.get("output_sha256", "")))
                and type(report.get("elapsed_seconds")) in (int, float)
                and math.isfinite(report["elapsed_seconds"]) and report["elapsed_seconds"] > 0,
                "coverage timings require complete, fresh, identity-bound probes")
        measurements.append({"video_id": vid, **probe_costs(report, row, job["batch_size"])})
    factor = job["throughput_safety_factor"]
    require(factor == 1.25, "coverage safety factor differs")
    rates = {name: factor * max(m[name] for m in measurements) for name in
             ("decode_seconds_per_frame", "work_seconds_per_batch", "overhead_seconds_per_video")}

    def cost(ids):
        value = 0.
        for vid in ids:
            row = by_id[vid]
            require(positive_integer(row.get("source_frames")) and positive_integer(row.get("n_windows")),
                    "forecast inventory lacks positive frame/window counts")
            value += rates["decode_seconds_per_frame"] * row["source_frames"]
            value += rates["work_seconds_per_batch"] * ((row["n_windows"] + 7) // 8)
            value += rates["overhead_seconds_per_video"]
        return value

    completed = set(probe_ids) | expected_reuse
    remaining_pilot = expected_pilot - completed
    remaining_full = set(by_id) - completed
    test = {vid for vid, row in by_id.items() if row["subset"] == "testing"}
    pilot = elapsed + cost(remaining_pilot) + job["pilot_head_reserve"]
    total = elapsed + cost(remaining_full) + job["pilot_head_reserve"] + job["full_head_and_export_reserve"]
    return {"estimator": "fresh decode/frame + summed nondecode work/batch + fixed/video; maxima times 1.25",
            "rates": rates, "probe_measurements": measurements,
            "seconds_per_window_conservative": rates["work_seconds_per_batch"] / 8,
            "projected_pilot_seconds": pilot, "projected_job_seconds": total,
            "projected_test_seconds": cost(test),
            "pilot_fits": pilot <= job["pilot_seconds"], "job_fits": total <= job["session_seconds"],
            "passed": pilot <= job["pilot_seconds"] and total <= job["session_seconds"],
            "verified_reused_videos": len(expected_reuse), "verified_reused_video_ids": sorted(expected_reuse),
            "remaining_pilot_videos": len(remaining_pilot), "remaining_full_videos": len(remaining_full),
            "remaining_pilot_windows": sum(by_id[v]["n_windows"] for v in remaining_pilot),
            "remaining_pilot_source_frames": sum(by_id[v]["source_frames"] for v in remaining_pilot),
            "timing_sums_may_overlap": True, "fresh_sparse_probes_required": True,
            "limitation": "Empirical forecast, not a guaranteed bound; unchanged phase/session deadlines remain enforced."}
