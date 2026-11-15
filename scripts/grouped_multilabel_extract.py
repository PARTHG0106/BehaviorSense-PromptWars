"""Run several frozen extraction partitions in one resumable Kaggle session.

This orchestration companion is checksummed separately from the original source
archive. It does not modify the extractor, annotations, models, or shard schema.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from multilabel_kaggle_stage import (asset, bootstrap, dataset_root, mounted_sources,
                                    read_json, sha256, write_json)


def validate_group(job):
    count = job.get("partition_count")
    indices = job.get("partition_indices")
    if job.get("stage") != "extract" or job.get("subset") not in ("training", "testing"):
        raise ValueError("Grouped execution requires an extraction subset")
    if type(count) is not int or count < 1:
        raise ValueError("partition_count must be positive")
    if (not isinstance(indices, list) or not indices
            or any(type(index) is not int or not 0 <= index < count for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError("partition_indices must be sorted, unique, and within partition_count")
    if not isinstance(job.get("stage_seconds"), (int, float)) or not 0 < job["stage_seconds"] <= 33000:
        raise ValueError("stage_seconds must be positive and within the reviewed session budget")
    return indices


def summarize_manifests(shards, job, expected_by_partition):
    indices = validate_group(job)
    summaries, fingerprints, all_seen, all_expected = [], set(), set(), set()
    windows = 0
    for index in indices:
        expected = set(expected_by_partition[index])
        if all_expected & expected:
            raise ValueError("Partition inventories overlap")
        all_expected.update(expected)
        suffix = (f"_part{index:03d}-of{job['partition_count']:03d}"
                  if job["partition_count"] > 1 else "")
        path = Path(shards) / f"manifest_{job['subset']}{suffix}.json"
        if not path.is_file():
            summaries.append({"index": index, "scope_complete": False, "completed_videos": 0,
                              "expected_videos": len(expected), "windows": 0})
            continue
        manifest = read_json(path)
        partition = manifest.get("partition", {})
        if (partition.get("index") != index or partition.get("count") != job["partition_count"]
                or manifest.get("subset") != job["subset"]):
            raise ValueError("Manifest identifies a different extraction partition")
        scope = manifest.get("scope_video_ids", [])
        complete_list = manifest.get("completed_video_ids", [])
        completed = set(complete_list)
        if (set(scope) != expected or len(scope) != len(expected)
                or len(completed) != len(complete_list) or not completed <= expected
                or manifest.get("scope_expected_videos") != len(expected)
                or manifest.get("completed_videos") != len(completed)):
            raise ValueError("Manifest video scope or completion counts disagree")
        reports = manifest.get("videos", [])
        report_ids = [report["video_id"] for report in reports]
        if set(report_ids) != completed or len(report_ids) != len(completed):
            raise ValueError("Per-video reports disagree with completed video IDs")
        if all_seen & completed:
            raise ValueError("A video occurs in multiple partition outputs")
        all_seen.update(completed)
        complete = completed == expected
        if manifest.get("scope_complete") is not complete:
            raise ValueError("Manifest claims completion without complete video coverage")
        config = manifest.get("config", {})
        if config.get("selection_mode") != "full" or config.get("windows_per_video") is not None:
            raise ValueError("Grouped extraction cannot accept capped or sampled manifests")
        fingerprint = manifest.get("fingerprint")
        if not fingerprint or (job.get("expected_config_fingerprint")
                               and fingerprint != job["expected_config_fingerprint"]):
            raise ValueError("Extraction config fingerprint differs from completed groups")
        fingerprints.add(fingerprint)
        n_windows = sum(report["retained_windows"] for report in reports)
        if manifest.get("windows") != n_windows:
            raise ValueError("Manifest window count disagrees with per-video reports")
        windows += n_windows
        summaries.append({"index": index, "scope_complete": complete,
                          "completed_videos": len(completed), "expected_videos": len(expected),
                          "windows": n_windows, "manifest": path.name})
    if len(fingerprints) > 1:
        raise ValueError("Partition extraction fingerprints differ")
    return {"scope_complete": all_seen == all_expected,
            "completed_videos": len(all_seen), "expected_videos": len(all_expected),
            "windows": windows, "config_fingerprint": next(iter(fingerprints), None),
            "partitions": summaries}


def expected_partitions(annotations, job):
    # Imported from the original frozen source, which independently verifies the
    # official inventory and hashes each video during extraction.
    from behaviorsense.data.toyota import load_tsu_annotations
    from extract_multilabel_windows import partition_videos, subset_videos
    work = subset_videos(load_tsu_annotations(annotations), job["subset"])
    groups = partition_videos(work, job["partition_count"])
    return {index: {video.vid for video in groups[index]} for index in job["partition_indices"]}


def execute_group(job, repo, work):
    indices = validate_group(job)
    began = time.monotonic()
    result = {"job_id": job["job_id"], "stage": "extract", "subset": job["subset"],
              "source_sha256": job["source_sha256"],
              "orchestration_sha256": job["group_runner_sha256"],
              "partition_count": job["partition_count"], "partition_indices": indices,
              "windows_per_video": None, "execution": "sequential_partitions_in_one_session",
              "status": "running"}

    def status(phase, **details):
        result.update(phase=phase, elapsed_seconds=round(time.monotonic() - began, 2), **details)
        write_json(work / "stage_result.json", result)
        print(json.dumps(result), flush=True)

    def save_group_manifest(summary):
        identity = {key: result[key] for key in (
            "job_id", "stage", "subset", "source_sha256", "orchestration_sha256",
            "partition_count", "partition_indices", "windows_per_video", "execution")}
        write_json(work / "group_manifest.json", {**identity, **summary})

    try:
        status("preflight")
        run = bootstrap(job, repo, work)
        sources = mounted_sources(job)
        resumes = []
        for root, previous in sources.values():
            if (previous.get("stage") != "extract" or previous.get("subset") != job["subset"]
                    or previous.get("partition_count") != job["partition_count"]
                    or previous.get("partition_indices") != indices
                    or previous.get("orchestration_sha256") != job["group_runner_sha256"]):
                raise ValueError("Resume source belongs to a different grouped extraction")
            resumes += ["--resume-from", root / "shards"]
        owner = job["owner"]
        weights = dataset_root(owner, "behavioursense-ww")
        annotations = asset(dataset_root(owner, "behaviorsense-toyota-meta"), "smarthome_CS_51.json")
        if sha256(annotations) != job["annotation_sha256"]:
            raise ValueError("Annotations differ from the frozen experiment")
        videos = dataset_root(owner, "rgb-untrimmed")
        expected = expected_partitions(annotations, job)
        if (job.get("expected_videos") is not None
                and sum(map(len, expected.values())) != job["expected_videos"]):
            raise ValueError("Grouped inventory differs from the reviewed plan")
        out = work / "shards"
        out.mkdir(parents=True, exist_ok=True)
        summary = summarize_manifests(out, job, expected)
        for index in indices:
            remaining = job["stage_seconds"] - (time.monotonic() - began)
            if remaining <= 1:
                break
            status("extract", active_partition_index=index, **summary)
            run([sys.executable, "scripts/extract_multilabel_windows.py",
                 "--videos-root", videos, "--annotations", annotations,
                 "--rtmo-model", asset(weights, "rtmo-l.onnx"),
                 "--context-model", asset(weights, "rtdetr-l.onnx"), "--out", out,
                 "--subset", job["subset"], "--selection", "full", "--candidate-stride", "15",
                 "--decode-mode", "sequential", "--partition-count", job["partition_count"],
                 "--partition-index", index, "--stop-after-seconds", remaining, *resumes])
            summary = summarize_manifests(out, job, expected)
            save_group_manifest(summary)
            current = next(part for part in summary["partitions"] if part["index"] == index)
            if not current["scope_complete"]:
                break
        summary = summarize_manifests(out, job, expected)
        save_group_manifest(summary)
        complete = summary["scope_complete"]
        status("complete" if complete else "resume_required",
               status="complete" if complete else "resumable", **summary)
    except Exception as exc:
        status("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
