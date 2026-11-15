"""Rebuild fitting-video label coverage from frozen extraction metadata, offline.

Only the nine fitting subjects are processed. No source video, feature shard,
validation array, test array, model, network connection, or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from behaviorsense.data.multilabel_skeleton import _verify_full_audit, _window_supervision
from behaviorsense.data.toyota import COARSE_V11, load_tsu_annotations, supervised_coarse, TSU_TO_COARSE

FIT_SUBJECTS = ("p03", "p06", "p07", "p09", "p12", "p13", "p17", "p19", "p25")
ORIGINAL_FOLDS = {"A": ["p12", "p17", "p25"], "B": ["p03", "p06", "p09"], "C": ["p07", "p13", "p19"]}
STATE_NAMES = ["neither", "drinking_only", "tv_only", "both"]
INPUT_PATHS = {
    "training_manifest": "artifacts/temporal-multilabel-20261011/output/candidates/asl-seed1/manifest.json",
    "rgb_plan": "artifacts/rgb-throughput-20261013/prefetch-run/plan.json",
    "annotations": "artifacts/training-tv-20261007/data/smarthome_CS_51.json",
    "pilot_manifest": "artifacts/rgb-throughput-20261013/prefetch-run/output/pilot/manifest.json",
    "pilot_protocol": "artifacts/rgb-throughput-20261013/pair-head/protocol.json",
    "pilot_provenance": "artifacts/rgb-throughput-20261013/pair-head/data/fit_provenance.json",
    "pilot_rows": "artifacts/rgb-throughput-20261013/pair-head/data/fit_rows.npz",
}
# Pin the reviewed metadata snapshot; a new source dataset requires an explicit
# inventory protocol change rather than silently mixing extraction generations.
FROZEN_INPUT_HASHES = {
    "training_manifest": "b7398ca4471feaf35dcd5f5b177dabfd16c8b2d85d071e06da534ad962babc30",
    "rgb_plan": "71e797545fd8cabea72edac10bcbf6d5d5db10aa834d63d9bcd74feef070c756",
    "annotations": "c7fc4eb24e9312d8a06354cb36030981edc6855a70c1cf6101ef3bdc93364821",
    "pilot_manifest": "4c35483007f86abd99df5c6eeec2e3a187493ca0b796540059bb787f9af2ca04",
    "pilot_protocol": "fd42b2680ed9d296fd8e0e9bc1611958c396bcbeaa90b86ccc7da273e4612003",
    "pilot_rows": "64b95148fc88a70a841638104b749c1627596bf682357cba1ff9d520f0ffcdd1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def summarize(videos):
    return {
        "n_videos": len(videos),
        "n_subjects": len({v["subject"] for v in videos}),
        "n_take_groups": len({v["take_group"] for v in videos}),
        "n_windows": sum(v["n_windows"] for v in videos),
        "positive_counts": np.sum([v["positive_counts"] for v in videos], axis=0).astype(int).tolist(),
        "known_counts": np.sum([v["known_counts"] for v in videos], axis=0).astype(int).tolist(),
        "negative_counts": np.sum([v["negative_counts"] for v in videos], axis=0).astype(int).tolist(),
        "four_state_counts": np.sum([v["four_state_counts"] for v in videos], axis=0).astype(int).tolist(),
        "actual_overlap_windows": sum(v["actual_overlap_windows"] for v in videos),
        "actual_overlap_sample_positions": sum(v["actual_overlap_sample_positions"] for v in videos),
        "duration_seconds": sum(v["duration_seconds"] for v in videos),
    }


def availability(videos, supervised_ids):
    classes = {}
    for index in supervised_ids:
        positive = [v for v in videos if v["positive_counts"][index] > 0]
        positive_subjects = sorted({v["subject"] for v in positive})
        by_fold = {}
        for fold, subjects in ORIGINAL_FOLDS.items():
            held = [v for v in positive if v["subject"] in subjects]
            fitting = [v for v in positive if v["subject"] not in subjects]
            by_fold[fold] = {
                "positive_windows": sum(v["positive_counts"][index] for v in held),
                "positive_subjects": sorted({v["subject"] for v in held}),
                "positive_take_groups": sorted({v["take_group"] for v in held}),
                "complement_positive_windows": sum(v["positive_counts"][index] for v in fitting),
            }
        classes[COARSE_V11[index]] = {
            "class_index": index,
            "positive_windows": sum(v["positive_counts"][index] for v in positive),
            "positive_subjects": positive_subjects,
            "positive_subject_count": len(positive_subjects),
            "positive_video_ids": [v["video_id"] for v in positive],
            "positive_video_count": len(positive),
            "positive_take_groups": sorted({v["take_group"] for v in positive}),
            "positive_windows_by_subject": {s: sum(v["positive_counts"][index] for v in positive if v["subject"] == s) for s in FIT_SUBJECTS},
            "original_fold_support": by_fold,
            "at_least_three_positive_subjects_possible": len(positive_subjects) >= 3,
            "positive_support_in_every_original_held_fold_possible": all(f["positive_windows"] > 0 for f in by_fold.values()),
        }
    by_subject = {s: summarize([v for v in videos if v["subject"] == s]) for s in FIT_SUBJECTS}
    gaps = [
        {"class": name, "positive_subjects": info["positive_subjects"],
         "missing_original_held_folds": [f for f, counts in info["original_fold_support"].items() if counts["positive_windows"] == 0],
         "fewer_than_three_positive_subjects": len(info["positive_subjects"]) < 3}
        for name, info in classes.items()
        if not info["at_least_three_positive_subjects_possible"] or not info["positive_support_in_every_original_held_fold_possible"]
    ]
    return {
        "per_class": classes, "per_subject": by_subject,
        "full_availability_gaps_under_original_folds": gaps,
        "all_supervised_classes_have_three_subjects": all(v["at_least_three_positive_subjects_possible"] for v in classes.values()),
        "all_supervised_classes_cover_original_folds": not gaps,
        "all_four_states_available_in_every_subject": all(all(n > 0 for n in v["four_state_counts"]) for v in by_subject.values()),
        "interpretation": "Feasibility reflects the full fitting inventory. It is not a sufficient sample-size claim. Any next-experiment folds must be frozen separately; these original folds and the failed candidate remain unchanged.",
    }


def build_inventory(root=ROOT):
    """Return a deterministic inventory and audit for the pinned fitting snapshot."""
    root = Path(root).resolve()
    require(root == ROOT.resolve(), "Alternate root cannot attest the imported label helpers")
    paths = {name: root / relative for name, relative in INPUT_PATHS.items()}
    input_files = {name: {"path": INPUT_PATHS[name], "sha256": sha256(path)} for name, path in paths.items()}
    for name, expected in FROZEN_INPUT_HASHES.items():
        require(input_files[name]["sha256"] == expected, f"Frozen {name} hash differs")
    full, plan = read_json(paths["training_manifest"]), read_json(paths["rgb_plan"])
    pilot, protocol = read_json(paths["pilot_manifest"]), read_json(paths["pilot_protocol"])
    cache_provenance = read_json(paths["pilot_provenance"])
    job = plan["job"]
    require(tuple(full["train"]["subjects"]) == FIT_SUBJECTS, "Fitting subject partition changed")
    require(protocol["folds"] == ORIGINAL_FOLDS, "Original frozen folds differ")
    require(protocol["source_plan_sha256"] == input_files["rgb_plan"]["sha256"], "Pair protocol plan hash differs")
    require(cache_provenance["frozen_manifest_sha256"] == input_files["pilot_manifest"]["sha256"], "Pilot manifest provenance differs")
    require(cache_provenance["files"]["fit_rows.npz"]["sha256"] == input_files["pilot_rows"]["sha256"], "Cached rows provenance differs")
    require(cache_provenance["status"] == "verified_complete", "Cached fitting rows are not verified")
    require(not any(cache_provenance["forbidden_inputs_loaded"].values()), "Cache provenance reports forbidden inputs")
    require(list(COARSE_V11) == full["class_names"] == pilot["class_names"], "Taxonomy differs")
    for expected in (job["annotation_sha256"], full["annotation_sha256"], pilot["annotation_sha256"], cache_provenance["annotation_sha256"]):
        require(expected == input_files["annotations"]["sha256"], "Annotation provenance differs")
    label_files = ("src/behaviorsense/data/multilabel_skeleton.py", "src/behaviorsense/data/toyota.py")
    full_code = {item["path"]: item["sha256"] for item in full["source_files"]}
    label_hashes = {}
    for relative in label_files:
        label_hashes[relative] = sha256(root / relative)
        require(label_hashes[relative] == full_code[relative] == plan["source_manifest"]["files"][relative], f"Frozen label helper differs: {relative}")

    source = full["provenance"]
    sample_fps, annotation_fps = float(source["sample_fps"]), float(source["annotation_fps"])
    window_frames, min_overlap = int(source["window_frames"]), float(source["min_overlap"])
    require((sample_fps, annotation_fps, window_frames, min_overlap) == (20., 25., 30, .5), "Label semantics changed")
    supervised_ids = list(supervised_coarse(TSU_TO_COARSE))
    require(supervised_ids == source["supervised_class_ids"], "Supervised masks differ")
    # These JSON files contain metadata for broader partitions. Only these nine
    # subjects are selected before any window reconstruction or label processing.
    shards = {item["extraction"]["video_id"]: item for item in source["source_shards"] if item["extraction"]["subject"] in FIT_SUBJECTS}
    planned = {item["video_id"]: item for item in job["video_inventory"] if item["subject"] in FIT_SUBJECTS}
    require(len(shards) == full["train"]["n_videos"] == 274, "Expected all 274 fitting videos")
    require(set(shards) == set(planned), "Planned fitting videos differ from source manifest")
    pilot_metadata = {item["extraction"]["video_id"]: item for item in pilot["source_provenance"]["source_shards"] if item["extraction"]["subject"] in FIT_SUBJECTS}
    annotations = load_tsu_annotations(paths["annotations"])
    with np.load(paths["pilot_rows"], allow_pickle=False) as saved:
        cached = {key: saved[key] for key in saved.files}
    require(set(cached["video_ids"]) == set(protocol["fitting_video_ids"]) == set(pilot_metadata), "Cached pilot IDs differ")
    require(set(cached["subjects"]) == set(FIT_SUBJECTS), "Cached rows contain a non-fitting subject")
    require(len(cached["starts"]) == 31859, "Expected 31,859 cached fitting rows")

    videos, cache_checks = [], []
    for video_id in sorted(shards):
        shard, inventory = shards[video_id], planned[video_id]
        report, annotation = shard["extraction"], annotations[video_id]
        subject = report["subject"]
        require(subject == "p" + video_id[1:3] and subject in FIT_SUBJECTS, f"Invalid subject: {video_id}")
        require(report["subset"] == annotation.subset == inventory["subset"] == "training", f"Non-fitting annotation: {video_id}")
        require(shard["sha256"] == inventory["shard_sha256"], f"Shard provenance differs: {video_id}")
        require(report["source_sha256"] == inventory["source_sha256"], f"Video provenance differs: {video_id}")
        require(report["source_fps"] == inventory["source_fps"] and report["source_frame_count"] == inventory["source_frames"], f"Source clock differs: {video_id}")
        require(report["annotation_duration_frames"] == annotation.duration, f"Annotation duration differs: {video_id}")
        starts = np.asarray(report["retained_starts"], dtype=np.int64)
        require(len(starts) == shard["n_windows"] == inventory["n_windows"], f"Retained count differs: {video_id}")
        _verify_full_audit(report, starts, annotation, sample_fps=sample_fps, annotation_fps=annotation_fps, min_overlap=min_overlap)
        source_frames = np.rint((starts[:, None] + np.arange(window_frames)) * report["source_fps"] / sample_fps).astype(np.int64)
        targets, overlap, coverage = _window_supervision(annotation, starts,
            sample_fps=sample_fps, annotation_fps=annotation_fps, min_overlap=min_overlap,
            source_frames=source_frames, source_fps=report["source_fps"])
        known = np.zeros_like(targets, dtype=bool)
        known[:, supervised_ids] = True
        known[coverage < window_frames * min_overlap] = False
        require(known[:, supervised_ids].all(), f"Retained annotation coverage differs: {video_id}")
        states = (targets[:, 10] + 2 * targets[:, 13]).astype(np.int64)
        positive = ((targets > 0) & known).sum(axis=0).astype(int)
        known_counts = known.sum(axis=0).astype(int)
        state_counts = np.bincount(states, minlength=4).tolist()
        # The legacy extractor additionally requires a simultaneous sampled
        # position for its pair stratum; the trained target requires co-labels.
        legacy_pair_count = int(((states == 3) & (overlap > 0)).sum())
        require(legacy_pair_count == inventory["pair_windows"], f"Legacy pair stratum differs: {video_id}")
        if video_id in pilot_metadata:
            require(report == pilot_metadata[video_id]["extraction"], f"Pilot extraction metadata differs: {video_id}")
            rows = cached["video_ids"] == video_id
            for name, expected in (("starts", starts), ("targets", targets), ("target_mask", known), ("tv_drinking_overlap_frames", overlap)):
                np.testing.assert_array_equal(cached[name][rows], expected, err_msg=f"{video_id} {name}")
            require(np.all(cached["subjects"][rows] == subject), f"Pilot subject differs: {video_id}")
            cache_checks.append({"video_id": video_id, "n_windows": len(starts), "starts_targets_masks_overlap_exact": True})
        videos.append({
            "video_id": video_id, "subject": subject, "take_group": video_id[:6], "camera": int(video_id[7:9]),
            "n_windows": len(starts), "positive_counts": positive.tolist(), "known_counts": known_counts.tolist(),
            "negative_counts": (known_counts - positive).tolist(), "four_state_counts": state_counts,
            "actual_overlap_windows": int((overlap > 0).sum()), "actual_overlap_sample_positions": int(overlap.sum()),
            "legacy_extraction_pair_stratum_count": legacy_pair_count,
            "colabeled_without_actual_overlap_windows": int(((states == 3) & (overlap == 0)).sum()),
            "duration_seconds": report["source_frame_count"] / report["source_fps"],
            "annotation_duration_frames": annotation.duration, "source_frame_count": report["source_frame_count"],
            "source_fps": report["source_fps"], "source_ref": inventory["source_ref"],
            "shard_filename": inventory["shard_filename"], "shard_sha256": shard["sha256"],
            "source_sha256": report["source_sha256"], "source_bytes": report["source_bytes"],
            "retained_starts_sha256": canonical_sha256(report["retained_starts"]),
            "extraction_metadata_sha256": canonical_sha256(report),
            "provenance": {"source_manifest": INPUT_PATHS["training_manifest"], "source_shard_path": shard["path"],
                "extraction_fingerprint": report["fingerprint"], "config_fingerprint": report["config_fingerprint"],
                "candidate_stride": report["candidate_stride"], "candidate_windows": report["candidate_windows"],
                "selected_windows": report["selected_windows"], "dropped_visibility": report["dropped_visibility"],
                "annotation_sha256": input_files["annotations"]["sha256"], "full_extraction_metadata_replayed": True},
        })
    summary = summarize(videos)
    expected = full["train"]
    for key, target_key in (("n_windows", "n_windows"), ("n_videos", "n_videos"), ("positive_counts", "positive"),
                            ("negative_counts", "negative"), ("actual_overlap_windows", "tv_drinking_actual_overlap_windows"),
                            ("actual_overlap_sample_positions", "tv_drinking_actual_overlap_sample_positions")):
        require(summary[key] == expected[target_key], f"Full fitting aggregate differs: {key}")
    require(summary["four_state_counts"][3] == expected["tv_drinking_colabeled_windows"], "Full fitting pair count differs")
    require(len(cache_checks) == 36 and sum(x["n_windows"] for x in cache_checks) == 31859, "Pilot cache replay incomplete")
    pilot_summary = summarize([v for v in videos if v["video_id"] in pilot_metadata])
    require(pilot_summary["positive_counts"] == pilot["train"]["positive"], "Pilot positive aggregate differs")
    require(pilot_summary["negative_counts"] == pilot["train"]["negative"], "Pilot negative aggregate differs")
    available = availability(videos, supervised_ids)
    inventory = {
        "schema": "behaviorsense.rgb_fitting_coverage_inventory.v1", "scope": "all_fitting_subject_videos_only",
        "fitting_subjects": list(FIT_SUBJECTS), "folds": ORIGINAL_FOLDS,
        "folds_note": "Original failed-candidate folds for coverage diagnosis; this file does not choose next-experiment folds.",
        "class_names": list(COARSE_V11), "supervised_class_ids": supervised_ids, "state_names": STATE_NAMES,
        "label_semantics": {"sample_fps": sample_fps, "annotation_fps": annotation_fps,
            "window_frames": window_frames, "min_overlap": min_overlap, "interval_boundary": "inclusive",
            "source_frames": "numpy.rint((retained_starts[:,None] + arange(30)) * source_fps / 20); audited decoder clock",
            "class_target": "Framewise fine-to-coarse annotation union covers at least 15 of 30 sampled positions",
            "known_mask": "Supported taxonomy class and at least 15 annotated sample positions",
            "pair_target": "drinking + 2 * watching_tv; state3 means both occupancy labels",
            "actual_overlap": "Sampled positions where both annotation unions are active simultaneously; distinct from co-label support"},
        "provenance": {"input_files": input_files, "frozen_input_hashes_verified": True,
            "label_source_files": label_hashes,
            "source_kind": "Frozen extraction metadata with retained-start audit, source-clock reconstruction and original annotation replay",
            "shard_bytes_loaded": False, "pilot_cached_fitting_rows_loaded": True,
            "pilot_cached_rows_exactly_replayed": 31859, "pilot_cached_videos_exactly_replayed": 36},
        "summary": summary, "original_pilot_summary": pilot_summary,
        "availability": available, "videos": videos,
        "limitations": [
            "Stored shard hashes are bound to the frozen plan; the 274 shard files were not loaded or byte-verified by this script.",
            "All retained-window labels match frozen training aggregates; exact cached row replay additionally covers the original 36 fitting videos.",
            "Overlapping windows and multiple camera views are correlated. Group by subject/take for diversity and split decisions.",
            "These counts describe retained annotated windows after visibility filtering, not independent events or natural activity prevalence.",
            "The seven unsupported classes have zero known counts; their zeros are not negative supervision.",
        ],
    }
    audit = {
        "schema": "behaviorsense.rgb_fitting_coverage_inventory_audit.v1", "status": "verified_complete",
        "inventory_canonical_sha256": canonical_sha256(inventory),
        "builder_sha256": sha256(Path(__file__)), "input_files": input_files,
        "all_274_extraction_metadata_audits_replayed": True, "frozen_input_hashes_verified": True,
        "frozen_label_helper_hashes_verified": True, "full_fitting_manifest_aggregates_match": True,
        "pilot_fitting_manifest_aggregates_match": True, "cached_fitting_checks": cache_checks,
        "n_videos": len(videos), "n_windows": summary["n_windows"],
        "scope": {"array_files_loaded": [INPUT_PATHS["pilot_rows"]], "validation_arrays_loaded": False,
            "test_arrays_loaded": False, "source_shard_arrays_loaded": False, "models_loaded": False,
            "model_training_performed": False, "threshold_search_performed": False,
            "network_used": False, "gpu_used": False},
        "full_availability_gaps_under_original_folds": available["full_availability_gaps_under_original_folds"],
    }
    return inventory, audit


def encoded_json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_preserving(path, value):
    """Never replace a prior snapshot; an identical repeat is a no-op."""
    path = Path(path)
    data = encoded_json(value)
    if path.exists():
        require(path.read_bytes() == data, f"Existing artifact differs; preserve it and use a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/rgb-coverage-20261013")
    args = parser.parse_args()
    inventory, audit = build_inventory(args.root)
    audit["inventory_file_sha256"] = hashlib.sha256(encoded_json(inventory)).hexdigest()
    write_preserving(args.output_dir / "inventory.json", inventory)
    write_preserving(args.output_dir / "inventory_audit.json", audit)
    print(json.dumps({"status": "verified_complete", "n_videos": inventory["summary"]["n_videos"],
        "n_windows": inventory["summary"]["n_windows"], "output_dir": str(args.output_dir),
        "availability_gaps": inventory["availability"]["full_availability_gaps_under_original_folds"]}, indent=2))


if __name__ == "__main__":
    main()
