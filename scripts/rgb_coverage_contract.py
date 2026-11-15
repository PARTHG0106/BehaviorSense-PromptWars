"""Freeze the reviewed RGB coverage plan without loading any feature arrays.

The builder hashes local evidence. The runtime validator uses only the standard
library and compact fitting metadata; it neither reads artifacts nor fits a model.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


SCHEMA = "behaviorsense.rgb_coverage_runtime.v1"
FIT_SUBJECTS = ("p03", "p06", "p07", "p09", "p12", "p13", "p17", "p19", "p25")
VALIDATION_SUBJECTS = ("p04", "p15")
BASE = "artifacts/rgb-coverage-20261013"
PLAN = BASE + "/availability-20261014/expanded_pilot.json"
PRIOR = "artifacts/rgb-throughput-20261013/prefetch-run"
KERNEL_SOURCE = "krishnagupta02468/bs-rgb-prefetch-20261013/1"
PLAN_SHA256 = "7b3afa2e07b62cd4e418fba8bb8ec931fe7c75dd45fcd73d191ebaf4e33b7fb1"
INVENTORY_SHA256 = "fad5a1c92a4e99ab7962e03dcb18638572048d4ecac532411211fa6c1108ba47"
SELECTION_SHA256 = "1866a5a65aa6dfc919fe03e40c958b9ac0d2c271850943278f73afa18fb77071"
RECIPE_SHA256 = "80dfd00c60f710c4ee57cef928ae6c8aec84b5bb34ea0caee6e0255d5d5018cf"
# This semantic digest binds every compact fact and receipt to the reviewed
# frozen inputs, including the non-selected fitting videos and cache provenance.
FROZEN_PAYLOAD_SHA256 = "beb39f359a828cde38d544736b85e2e74baa9438bbd95e801a44f571afdc2393"
FORECAST_POLICY = {
    "schema": "behaviorsense.rgb_sparse_stage_forecast.v1",
    "safety_factor": 1.25,
    "decode_rate": "max fresh decode wall seconds / decoded_frames",
    "batch_rate": "max fresh other stage wall seconds / ceil(n_windows / 8); exclude input_validation, serialize, decode",
    "video_overhead": "max fresh input_validation + serialize + max(elapsed_seconds - sum(all stage wall seconds), 0)",
    "remaining_video_cost": "1.25 * (decode_rate * source_frames + batch_rate * ceil(n_windows / 8) + video_overhead)",
    "required_fresh_probes": 5,
    "cache_reuse_credit": "only after all 113 frozen cache bytes, metadata and row alignment verify",
}
FROZEN_BUDGET = {
    "session_seconds": 40500, "pilot_seconds": 4500, "probe_seconds": 900,
    "throughput_safety_factor": 1.25, "full_head_and_export_reserve": 3600,
    "pilot_head_reserve": 600, "batch_size": 8, "automatic_continuation": False,
    "encoder_preference": ["swin3d_t"], "prefetch_batches": 2, "opencv_threads": 1,
}
GATE_CRITERIA = {
    "macro_ap_gain_over_matched_control": 0.03, "macro_ap_min": 0.6379390777,
    "pair_f1_min": 0.30, "pair_precision_min": 0.30, "pair_recall_min": 0.30,
    "subjects": {"p04": {"pair_f1_min": 0.21808510638, "macro_ap_min": 0.59126160643},
                 "p15": {"pair_f1_min": 0.03861003861, "macro_ap_min": 0.64502818523}},
}
FACT_FIELDS = (
    "video_id", "subject", "take_group", "n_windows", "positive_counts", "known_counts",
    "negative_counts", "four_state_counts", "source_ref", "source_sha256", "source_bytes",
    "source_frame_count", "source_fps", "duration_seconds", "shard_filename", "shard_sha256",
    "retained_starts_sha256", "extraction_metadata_sha256",
)
SUMMARY_FIELDS = (
    "videos", "distinct_take_groups", "fitting_windows", "pair_windows", "additional_rgb_videos",
    "additional_rgb_windows", "additional_source_frames", "per_subject", "per_class",
    "pair_positive_take_groups_by_fold", "all_available_pair_positive_take_groups",
    "all_pair_positive_takes_covered", "validation_videos", "validation_windows",
)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError("RGB coverage contract: " + message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _count(value, positive=False):
    return type(value) is int and value >= (1 if positive else 0)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ids(values, label, count=None):
    _require(isinstance(values, list) and all(isinstance(v, str) for v in values)
             and len(values) == len(set(values)), "duplicate or invalid " + label)
    _require(count is None or len(values) == count, "wrong number of " + label)
    return set(values)


def _relative(path, root):
    """Map recorded original-workspace paths to the caller's checkout."""
    spelling = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", spelling) or spelling.startswith("/"):
        current = str(Path(root).resolve()).replace("\\", "/").rstrip("/") + "/"
        if spelling.startswith(current):
            spelling = spelling[len(current):]
        else:
            marker = "/EmotionSense-Extended/"
            _require(marker in spelling, "provenance path is outside the repository")
            spelling = spelling.split(marker, 1)[1]
    parts = PurePosixPath(spelling).parts
    _require(parts and ".." not in parts and not PurePosixPath(spelling).is_absolute(),
             "unsafe provenance path")
    return "/".join(parts)


def _validate_facts(coverage):
    names, supported = coverage["class_names"], coverage["supervised_class_ids"]
    _require(len(names) == len(set(names)) == 22 and names[10] == "drinking"
             and names[13] == "watching_tv", "class order differs")
    _require(supported == [0, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21],
             "supervision class mask differs")
    folds = coverage["folds"]
    _require(set(folds) == {"A", "B", "C"} and all(len(v) == 3 for v in folds.values()),
             "fold shape differs")
    members = [s for group in folds.values() for s in group]
    _require(len(set(members)) == 9 and set(members) == set(FIT_SUBJECTS),
             "folds must partition fitting subjects only")
    facts = coverage["fitting_inventory"]
    _require(isinstance(facts, list) and len(facts) == 274, "expected all 274 fitting video facts")
    by_id = {}
    for fact in facts:
        vid, n = fact["video_id"], fact["n_windows"]
        _require(re.fullmatch(r"P\d{2}T\d{2}C\d{2}", vid) and vid not in by_id,
                 "invalid or duplicate fitting identity")
        _require(fact["subject"] in FIT_SUBJECTS and fact["subject"] == "p" + vid[1:3]
                 and fact["take_group"] == vid[:6], "held-out or mismatched fitting identity")
        _require(_count(n, True), "invalid fitting window count")
        for key in ("positive_counts", "known_counts", "negative_counts"):
            _require(isinstance(fact[key], list) and len(fact[key]) == 22
                     and all(_count(c) and c <= n for c in fact[key]), "invalid " + key)
        for c in range(22):
            _require(fact["known_counts"][c] == (n if c in supported else 0),
                     "unsupported or incomplete known mask")
            _require(fact["positive_counts"][c] + fact["negative_counts"][c] == fact["known_counts"][c],
                     "positive, negative and known counts disagree")
        states = fact["four_state_counts"]
        _require(isinstance(states, list) and len(states) == 4 and all(_count(x) for x in states)
                 and sum(states) == n and states[1] + states[3] == fact["positive_counts"][10]
                 and states[2] + states[3] == fact["positive_counts"][13], "four-state labels disagree")
        _require(all(_digest(fact[k]) for k in ("source_sha256", "shard_sha256",
                 "retained_starts_sha256", "extraction_metadata_sha256")), "invalid fitting source hash")
        _require(_count(fact["source_bytes"], True) and _count(fact["source_frame_count"], True)
                 and type(fact["source_fps"]) in (int, float) and math.isfinite(fact["source_fps"])
                 and fact["source_fps"] > 0, "invalid source timing")
        _require(type(fact["duration_seconds"]) in (int, float) and math.isfinite(fact["duration_seconds"])
                 and abs(fact["duration_seconds"] - fact["source_frame_count"] / fact["source_fps"]) < 1e-6,
                 "source duration disagrees with decoder clock")
        _require(fact["shard_filename"] == f"toyota_multilabel_20hz_training_{vid}.npz",
                 "source shard identity differs")
        by_id[vid] = fact
    _require(sum(v["n_windows"] for v in facts) == 217090, "full fitting row count differs")
    return by_id


def _sparse_probes(by_id, additional):
    first = min(additional, key=lambda vid: (-Fraction(by_id[vid]["source_frame_count"],
                                                      by_id[vid]["n_windows"]), vid))
    second = min(set(additional) - {first}, key=lambda vid: (-by_id[vid]["source_frame_count"], vid))
    return [first, second]


def _summary(coverage, by_id):
    selected = [by_id[v] for v in coverage["pilot_video_ids"]]
    original = [by_id[v] for v in coverage["original_video_ids"]]
    additional = [by_id[v] for v in coverage["additional_rgb_video_ids"]]
    folds = coverage["folds"]
    per_class = {}
    for c in coverage["supervised_class_ids"]:
        positives = [v for v in selected if v["positive_counts"][c] > 0]
        held = {f: sum(v["positive_counts"][c] for v in selected if v["subject"] in subjects)
                for f, subjects in folds.items()}
        old = sum(v["positive_counts"][c] for v in original)
        total = sum(v["positive_counts"][c] for v in positives)
        subjects = sorted({v["subject"] for v in positives})
        _require(len(subjects) >= 3 and all(0 < n < total for n in held.values()) and total >= old,
                 "class lacks fitting subject, fold or complement coverage")
        per_class[coverage["class_names"][c]] = {
            "original_positive_windows": old, "positive_windows": total, "positive_subjects": subjects,
            "positive_videos": len(positives), "positive_take_groups": len({v["take_group"] for v in positives}),
            "positive_windows_by_fold": held,
        }
    pairs = {f: sorted({v["take_group"] for v in selected if v["subject"] in members
                       and v["four_state_counts"][3] > 0}) for f, members in folds.items()}
    available = sorted({v["take_group"] for v in by_id.values() if v["four_state_counts"][3] > 0})
    _require(len(available) == 28 and all(len(takes) >= 6 for takes in pairs.values())
             and sorted({t for takes in pairs.values() for t in takes}) == available,
             "all 28 pair-positive fitting takes must be covered")
    per_subject = {s: {"videos": sum(v["subject"] == s for v in selected),
                      "four_state_counts": [sum(v["four_state_counts"][c] for v in selected if v["subject"] == s)
                                            for c in range(4)]} for s in FIT_SUBJECTS}
    _require(all(all(v["four_state_counts"]) for v in per_subject.values()), "subject lacks four-state coverage")
    return {
        "videos": len(selected), "distinct_take_groups": len({v["take_group"] for v in selected}),
        "fitting_windows": sum(v["n_windows"] for v in selected),
        "pair_windows": sum(v["four_state_counts"][3] for v in selected),
        "additional_rgb_videos": len(additional), "additional_rgb_windows": sum(v["n_windows"] for v in additional),
        "additional_source_frames": sum(v["source_frame_count"] for v in additional),
        "per_subject": per_subject, "per_class": per_class, "pair_positive_take_groups_by_fold": pairs,
        "all_available_pair_positive_take_groups": available, "all_pair_positive_takes_covered": True,
        "validation_videos": 77, "validation_windows": 63220,
    }


def _validate_selection(coverage, by_id):
    original = _ids(coverage["original_video_ids"], "original videos", 36)
    core = _ids(coverage["core_video_ids"], "core videos", 36)
    pilot = _ids(coverage["pilot_video_ids"], "pilot videos", 60)
    additional = _ids(coverage["additional_rgb_video_ids"], "additional RGB videos", 24)
    additions = _ids(coverage["pair_coverage_additions"], "pair completion videos", 4)
    _require(original | core | additions == pilot <= set(by_id) and additional == pilot - original,
             "pilot must preserve original union core plus missing pair takes")
    _require(len({by_id[v]["take_group"] for v in core}) == 36
             and all(sum(by_id[v]["subject"] == s for v in core) == 4 for s in FIT_SUBJECTS),
             "coverage core must retain four distinct takes per fitting subject")
    covered = {by_id[v]["take_group"] for v in original | core if by_id[v]["four_state_counts"][3] > 0}
    available = {v["take_group"] for v in by_id.values() if v["four_state_counts"][3] > 0}
    expected = []
    for take in sorted(available - covered):
        choices = [v for v in by_id.values() if v["take_group"] == take and v["four_state_counts"][3] > 0]
        expected.append(min(choices, key=lambda v: (-v["four_state_counts"][3], v["n_windows"], v["video_id"]))["video_id"])
    _require(coverage["pair_coverage_additions"] == expected, "pair completion policy differs")
    probes = coverage["core_probe_video_ids"]
    _require(_ids(probes, "core probes", 3) <= core
             and len({by_id[v]["subject"] for v in probes}) == 3
             and all(100 <= by_id[v]["n_windows"] <= 1000 for v in probes), "invalid complete core probes")
    sparse = _sparse_probes(by_id, additional)
    _require(coverage["sparse_probe_video_ids"] == sparse and coverage["probe_video_ids"] == probes + sparse,
             "fresh sparse probes must cover the maximum frame density and remaining frame count")
    _require(coverage["coverage_summary"] == _summary(coverage, by_id), "coverage count report differs")
    _require(coverage["coverage_summary"]["fitting_windows"] == 40193
             and coverage["coverage_summary"]["pair_windows"] == 638, "frozen pilot counts differ")


def _validate_cache(coverage, job_by_id):
    cache = coverage["cache_reuse"]
    _require(cache["kernel_source"] == KERNEL_SOURCE and cache["source_job_id"] == KERNEL_SOURCE.rsplit("/", 1)[0],
             "cache mount must use the pinned prior kernel version")
    contract = cache["feature_contract"]
    _require(canonical_hash(contract) == cache["feature_contract_sha256"], "cache feature contract digest differs")
    _require(contract["arch"] == "swin3d_t" and contract["feature_dim"] == 768
             and contract["views"] == ["scene", "person"], "cache encoder contract differs")
    records = cache["records"]
    _ids([v["video_id"] for v in records], "cache records", 113)
    expected_ids = set(coverage["original_video_ids"]) | {vid for vid, v in job_by_id.items()
                                                       if v["subject"] in VALIDATION_SUBJECTS}
    _require({v["video_id"] for v in records} == expected_ids, "cache must cover original fitting plus all validation")
    for record in records:
        vid = record["video_id"]
        source = job_by_id[vid]
        _require(record["subset"] == source["subset"] == "training"
                 and record["subject"] == source["subject"]
                 and record["role"] == ("fitting" if vid in coverage["original_video_ids"] else "validation")
                 and record["n_windows"] == source["n_windows"], "cache row identity differs")
        _require(record["source_video_sha256"] == source["source_sha256"]
                 and record["source_shard_sha256"] == source["shard_sha256"], "cache source binding differs")
        _require(record["relative_path"] == f"rgb_features/swin3d_t/{vid}.npz"
                 and _digest(record["feature_sha256"]) and _digest(record["fingerprint"]), "cache byte receipt differs")
        for key in ("arch", "feature_dim", "encoder_sha256", "preprocessing_sha256", "extraction_implementation_sha256"):
            _require(record[key] == contract[key], "cache extraction contract differs: " + key)
    inventory = [{key: v[key] for key in ("video_id", "n_windows", "feature_sha256", "source_video_sha256",
                                          "source_shard_sha256", "fingerprint")} for v in records]
    _require(canonical_hash(inventory) == cache["feature_inventory_sha256"], "cache inventory provenance differs")
    receipts = cache["provenance_files"]
    _require({v["relative_path"] for v in receipts} == {"job.json", "source_manifest.json", "extraction_swin3d_t.json",
             "encoder_selection.json", "stage_result.json", "pilot/manifest.json"}
             and len(receipts) == 6 and all(_digest(v["sha256"]) for v in receipts), "cache source receipts differ")


def validate_coverage_contract(job):
    """Reject a changed plan, coverage facts, cache lineage or job binding."""
    try:
        coverage = job["coverage"]
        _require(coverage["schema"] == SCHEMA, "unknown runtime schema")
        payload = {k: v for k, v in coverage.items() if k != "contract_sha256"}
        digest = canonical_hash(payload)
        _require(digest == coverage["contract_sha256"], "runtime contract checksum differs")
        _require(digest == FROZEN_PAYLOAD_SHA256,
                 "runtime contract differs from reviewed frozen evidence")
        _require(coverage["plan_sha256"] == PLAN_SHA256 and coverage["inventory_sha256"] == INVENTORY_SHA256
                 and coverage["core_selection_sha256"] == SELECTION_SHA256
                 and coverage["recipe_sha256"] == RECIPE_SHA256, "plan lineage differs")
        _require(coverage["forecast_policy"] == FORECAST_POLICY, "sparse timing forecast policy differs")
        for key, expected in FROZEN_BUDGET.items():
            _require(job.get(key) == expected, "frozen budget or runtime changed: " + key)
        _require(job.get("gate_criteria") == GATE_CRITERIA, "frozen quality gates changed")
        for key, expected in coverage["source_checks"].items():
            _require(job.get(key) == expected, "job source binding differs: " + key)
        by_id = _validate_facts(coverage)
        _validate_selection(coverage, by_id)
        _require(job["pilot"] == coverage["pilot"], "job pilot differs from the coverage pilot")
        pilot = coverage["pilot"]
        _require(pilot["pilot_video_ids"] == coverage["pilot_video_ids"]
                 and pilot["probe_video_ids"] == coverage["core_probe_video_ids"]
                 and pilot["n_fit_videos"] == 60 and pilot["fit_windows"] == 40193
                 and pilot["pair_windows"] == 638, "embedded pilot inventory differs")
        records = job["video_inventory"]
        job_by_id = {v["video_id"]: v for v in records}
        _require(len(job_by_id) == len(records) == 536, "job video inventory is incomplete or duplicated")
        _require(canonical_hash(sorted(records, key=lambda v: v["video_id"])) == coverage["video_inventory_sha256"],
                 "frozen complete job video inventory binding differs")
        _require(canonical_hash(job["feature_sources"]) == coverage["feature_sources_sha256"],
                 "frozen feature source receipts differ")
        fit_ids = {vid for vid, v in job_by_id.items() if v["subject"] in FIT_SUBJECTS}
        _require(fit_ids == set(by_id), "job fitting set differs from coverage inventory")
        for vid, fact in by_id.items():
            record = job_by_id[vid]
            for key in ("subject", "n_windows", "source_ref", "source_sha256", "source_bytes",
                        "source_fps", "duration_seconds", "shard_filename", "shard_sha256"):
                _require(record[key] == fact[key], "job video binding differs: " + vid + "/" + key)
            _require(record["source_frames"] == fact["source_frame_count"] and record["subset"] == "training"
                     and record["source_filename"] == vid + ".mp4", "job source frame/partition binding differs")
        validation = [v for v in records if v["subject"] in VALIDATION_SUBJECTS]
        _require(len(validation) == 77 and sum(v["n_windows"] for v in validation) == 63220
                 and all(v["subset"] == "training" for v in validation), "complete frozen validation set required")
        _validate_cache(coverage, job_by_id)
        return coverage
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("RGB coverage contract: malformed or missing field") from exc


def build_coverage_contract(plan_path, root):
    """Hash the reviewed evidence, then produce the compact runtime contract."""
    root = Path(root).resolve()
    plan_path = Path(plan_path)
    plan_path = plan_path if plan_path.is_absolute() else root / plan_path
    lineage = {}

    def verify(path, expected=None):
        relative = _relative(path, root)
        local = root / relative
        _require(local.is_file(), "missing provenance source: " + relative)
        digest = sha256(local)
        _require(expected is None or digest == expected, "provenance hash changed: " + relative)
        _require(relative not in lineage or lineage[relative] == digest, "conflicting provenance source")
        lineage[relative] = digest
        return _read(local) if local.suffix == ".json" else None

    plan = verify(plan_path, PLAN_SHA256)
    status = verify(BASE + "/continuation-status.json")
    _require(status["recommended_plan"] == "availability-20261014/expanded_pilot.json"
             and status["coverage_work_complete"] is True, "coverage continuation evidence is incomplete")
    for path, digest in status["evidence_sha256"].items():
        verify(BASE + "/" + path, digest)
    for path, digest in status["source_and_report_sha256"].items():
        verify(path, digest)
    inventory = verify(BASE + "/verified-20261014/inventory.json", INVENTORY_SHA256)
    audit = verify(BASE + "/verified-20261014/inventory_audit.json")
    selection = verify(BASE + "/availability-20261014/selection.json", SELECTION_SHA256)
    recipe = verify(BASE + "/availability-20261014/recipe.json", RECIPE_SHA256)
    forecast = verify(BASE + "/availability-20261014/planning_forecast.json", plan["planning_forecast_sha256"])
    verify(PRIOR + "/output/profiling/swin3d_t/profile_report.json", forecast["reference_profile_sha256"])
    verify(BASE + "/augment_cached_pilot.py", plan["source_sha256"])
    verify(BASE + "/availability-20261014/selector_source.py", recipe["source_sha256"])
    _require(plan["inventory_sha256"] == recipe["inventory_sha256"] == selection["inventory_sha256"] == INVENTORY_SHA256
             and plan["core_selection_sha256"] == forecast["selection_sha256"] == SELECTION_SHA256
             and selection["recipe_sha256"] == RECIPE_SHA256 and selection["source_sha256"] == recipe["source_sha256"],
             "frozen plan, selection and recipe lineage disagree")
    _require(recipe["policy_sha256"] == canonical_hash(recipe["policy"])
             and recipe["inventory_semantic_sha256"] == canonical_hash(inventory)
             and audit["inventory_canonical_sha256"] == canonical_hash(inventory)
             and audit["inventory_file_sha256"] == INVENTORY_SHA256, "inventory semantic audit disagrees")
    _require(audit["status"] == "verified_complete" and selection["status"] == "optimal"
             and selection["constraint_relaxation"] is False and selection["audit"]["passed"] is True,
             "coverage selection or inventory verification is incomplete")
    _require(audit["input_files"] == inventory["provenance"]["input_files"], "inventory input audit differs")
    for key in ("all_274_extraction_metadata_audits_replayed", "frozen_input_hashes_verified",
                "frozen_label_helper_hashes_verified", "full_fitting_manifest_aggregates_match",
                "pilot_fitting_manifest_aggregates_match"):
        _require(audit[key] is True, "missing fitting audit: " + key)
    for receipt in inventory["provenance"]["input_files"].values():
        verify(receipt["path"], receipt["sha256"])
    for path, digest in inventory["provenance"]["label_source_files"].items():
        verify(path, digest)
    for path, digest in recipe["verified_source_files"].items():
        verify(path, digest)
    original = verify(inventory["provenance"]["input_files"]["pilot_protocol"]["path"], plan["original_protocol_sha256"])
    prior = verify(PRIOR + "/plan.json", original["source_plan_sha256"])
    previous_job = prior["job"]
    output = PRIOR + "/output/"
    provenance_files = []
    loaded = {}
    for name in ("job.json", "source_manifest.json", "extraction_swin3d_t.json", "encoder_selection.json",
                 "stage_result.json", "pilot/manifest.json"):
        loaded[name] = verify(output + name)
        provenance_files.append({"relative_path": name, "sha256": lineage[output + name]})
    _require(previous_job == loaded["job.json"], "prior submitted job differs from output job")
    _require(loaded["source_manifest.json"]["sha256"] == previous_job["source_sha256"]
             == original["source_job_sha256"], "prior source bundle differs")
    for name, digest in prior["submission_hashes"].items():
        verify(PRIOR + "/submission/" + name, digest)
    manifest = loaded["pilot/manifest.json"]
    feature_contract = manifest["feature_contract"]
    reports = loaded["extraction_swin3d_t.json"]
    report_by_id = {r["video_id"]: r for r in reports}
    _require(len(reports) == len(report_by_id) == len(manifest["feature_inventory"]) == 113,
             "prior cache extraction inventory is incomplete")
    _require(canonical_hash(manifest["feature_inventory"]) == manifest["feature_inventory_sha256"],
             "prior cache inventory digest differs")
    _require(feature_contract["encoder_info"] == loaded["encoder_selection.json"]["encoder_info"],
             "prior encoder selection differs from feature contract")
    original_ids = original["fitting_video_ids"]
    _require(original_ids == previous_job["pilot"]["pilot_video_ids"], "original pilot set differs")
    job_by_id = {v["video_id"]: v for v in previous_job["video_inventory"]}
    records = []
    for item in manifest["feature_inventory"]:
        vid = item["video_id"]
        report, source = report_by_id[vid], job_by_id[vid]
        _require(report["status"] == "complete" and report["output_sha256"] == item["feature_sha256"]
                 and report["n_windows"] == item["n_windows"] == source["n_windows"]
                 and report["source_video_sha256"] == item["source_video_sha256"] == source["source_sha256"]
                 and report["source_shard_sha256"] == item["source_shard_sha256"] == source["shard_sha256"],
                 "prior feature bytes or extraction source identity differ")
        for key in ("arch", "feature_dim", "encoder_sha256", "preprocessing_sha256"):
            _require(report[key] == feature_contract[key], "prior encoder report differs")
        records.append({**item, "subject": source["subject"], "subset": source["subset"],
                        "role": "fitting" if vid in original_ids else "validation",
                        "relative_path": f"rgb_features/swin3d_t/{vid}.npz",
                        **{k: feature_contract[k] for k in ("arch", "feature_dim", "encoder_sha256",
                                                          "preprocessing_sha256", "extraction_implementation_sha256")}})
    coverage = {
        "schema": SCHEMA, "plan_sha256": PLAN_SHA256, "inventory_sha256": INVENTORY_SHA256,
        "core_selection_sha256": SELECTION_SHA256, "recipe_sha256": RECIPE_SHA256,
        "original_video_ids": original_ids, "core_video_ids": selection["selected_video_ids"],
        "pilot_video_ids": plan["pilot_video_ids"], "pair_coverage_additions": plan["pair_coverage_additions"],
        "additional_rgb_video_ids": plan["additional_rgb_video_ids"],
        "core_probe_video_ids": selection["probe_video_ids"], "folds": plan["folds"],
        "class_names": inventory["class_names"], "supervised_class_ids": inventory["supervised_class_ids"],
        "fitting_inventory": [{k: v[k] for k in FACT_FIELDS} for v in inventory["videos"]],
        "coverage_summary": {key: plan[key] for key in SUMMARY_FIELDS},
        "video_inventory_sha256": canonical_hash(sorted(previous_job["video_inventory"], key=lambda v: v["video_id"])),
        "feature_sources_sha256": canonical_hash(previous_job["feature_sources"]),
        "source_checks": {key: previous_job[key] for key in ("annotation_sha256", "extraction_source_sha256",
                                                            "assets_manifest_sha256")},
        "forecast_policy": deepcopy(FORECAST_POLICY),
        "cache_reuse": {
            "kernel_source": KERNEL_SOURCE, "source_job_id": previous_job["job_id"],
            "source_sha256": previous_job["source_sha256"],
            "extraction_source_sha256": previous_job["extraction_source_sha256"],
            "training_source_sha256": prior["submission_hashes"]["training.py"],
            "training_source_text_sha256": hashlib.sha256((root / PRIOR / "submission/training.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
            "submission_hashes": prior["submission_hashes"], "feature_contract": feature_contract,
            "feature_contract_sha256": canonical_hash(feature_contract),
            "feature_inventory_sha256": manifest["feature_inventory_sha256"],
            "provenance_files": provenance_files, "records": records,
        },
        "lineage_files": dict(sorted(lineage.items())),
    }
    _require(plan["folds"] == selection["fold_selection"]["folds"]
             and plan["probe_video_ids"] == selection["probe_video_ids"], "core fold/probe selection differs")
    facts = _validate_facts(coverage)
    coverage["sparse_probe_video_ids"] = _sparse_probes(facts, coverage["additional_rgb_video_ids"])
    coverage["probe_video_ids"] = coverage["core_probe_video_ids"] + coverage["sparse_probe_video_ids"]
    coverage["pilot"] = {
        "pilot_video_ids": coverage["pilot_video_ids"], "probe_video_ids": coverage["core_probe_video_ids"],
        "n_fit_videos": 60, "fit_windows": 40193, "pair_windows": 638,
        "covered_activities": sorted(coverage["coverage_summary"]["per_class"]),
        "selection_uses": "verified fitting subjects, fitting annotations and extraction metadata only",
        "validation": "all 77 videos belonging to p04 and p15, without video selection",
    }
    coverage["contract_sha256"] = canonical_hash(coverage)
    bound = {**previous_job, "coverage": coverage, "pilot": coverage["pilot"]}
    validate_coverage_contract(bound)
    return coverage
