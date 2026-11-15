"""Freeze and select a fitting-only, coverage-constrained whole-video RGB pilot.

This script reads inventory metadata and hashes local provenance files. It does
not load feature arrays, train models, extract videos, or contact a service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import time

FIT_SUBJECTS = ("p03", "p06", "p07", "p09", "p12", "p13", "p17", "p19", "p25")
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "behaviorsense.rgb_fitting_coverage_inventory.v1"
POLICY = {
    "fitting_subjects": list(FIT_SUBJECTS), "videos_per_subject": 4,
    "maximum_cameras_per_take": 1, "minimum_positive_subjects_per_class": 3,
    "class_positive_in_each_fold_and_complement": True,
    "minimum_pair_positive_takes_per_subject": 2,
    "subject_pair_take_rule": "require min(2, full fitting availability) distinct positive takes, at least one per subject; report every availability shortfall",
    "minimum_pair_positive_takes_per_fold": 6,
    "all_four_states_in_each_subject": True, "minimum_pair_windows": 100,
    "probe_window_range_inclusive": [100, 1000], "minimum_probe_subjects": 3,
    "probe_selection": "duration/window quantiles 0.15, 0.5, 0.85 among selected eligible videos; nearest remaining video from a distinct subject, then video ID",
    "fold_policy": "canonical three folds of three subjects; every supervised class and all four states positive in each fold; minimize full-inventory held-row spread, then pair-window spread, then lexicographic memberships",
    "objective": "minimize selected fitting windows; stable sorted video-ID rank cost totals less than one quarter window",
    "reproducibility": "Sorted input and recorded SciPy/HiGHS solver; rank sums can tie, so immutable selected IDs are authoritative across solver versions.",
    "sample_size_claim": "Coverage constraints do not establish sufficient sample size or model quality.",
    "solver_time_limit_seconds": 60, "accept_only_proven_optimal": True,
    "no_constraint_relaxation": True, "model_training": False,
    "outer_validation_arrays": False, "test_arrays": False, "gpu": False, "network": False,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_folds(folds):
    require(isinstance(folds, dict) and set(folds) == {"A", "B", "C"}, "folds must be A/B/C")
    members = []
    for label in ("A", "B", "C"):
        require(isinstance(folds[label], list) and len(folds[label]) == 3, "each fold needs three subjects")
        members.extend(folds[label])
    require(len(set(members)) == 9 and set(members) == set(FIT_SUBJECTS), "folds must partition only fitting subjects")


def validate_inventory(inventory):
    require(isinstance(inventory, dict) and inventory.get("schema") == SCHEMA, "invalid inventory schema")
    require(inventory.get("fitting_subjects") is not None
            and sorted(inventory["fitting_subjects"]) == list(FIT_SUBJECTS), "invalid fitting subjects or held-out input")
    names, supported = inventory.get("class_names"), inventory.get("supervised_class_ids")
    require(isinstance(names, list) and len(names) == len(set(names)) == 22
            and all(isinstance(name, str) and name for name in names), "expected 22 unique class names")
    require(names[10] == "drinking" and names[13] == "watching_tv", "pair class order differs")
    require(isinstance(supported, list) and len(supported) == len(set(supported)) == 15
            and all(is_count(c) and c < 22 for c in supported) and {10, 13} <= set(supported),
            "expected 15 supervised class IDs including the pair")
    validate_folds(inventory.get("folds"))
    provenance = inventory.get("provenance", {})
    require(provenance.get("frozen_input_hashes_verified") is True, "unverified inventory provenance")
    for field in ("input_files", "label_source_files"):
        require(isinstance(provenance.get(field), dict) and provenance[field], "missing provenance "+field)
    for receipt in provenance["input_files"].values():
        require(isinstance(receipt, dict) and isinstance(receipt.get("path"), str)
                and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", ""))), "invalid provenance receipt")
    for path, digest in provenance["label_source_files"].items():
        require(isinstance(path, str) and re.fullmatch(r"[0-9a-f]{64}", str(digest)), "invalid label-source hash")
    videos = inventory.get("videos")
    require(isinstance(videos, list) and videos, "missing video inventory")
    seen, subjects = set(), set()
    for video in videos:
        require(isinstance(video, dict), "invalid video record")
        vid, subject = video.get("video_id"), video.get("subject")
        require(isinstance(vid, str) and re.fullmatch(r"P\d{2}T\d{2}C\d{2}", vid), "invalid video ID")
        require(vid not in seen, "duplicate video ID")
        seen.add(vid)
        require(subject in FIT_SUBJECTS and subject == "p"+vid[1:3], "held-out or mismatched video subject")
        subjects.add(subject)
        require(video.get("take_group") == vid[:6], "take identity differs")
        require(is_count(video.get("camera")) and video["camera"] == int(vid[7:]), "camera identity differs")
        n = video.get("n_windows")
        require(is_count(n), "invalid window count")
        duration = video.get("duration_seconds")
        require(isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and math.isfinite(duration) and duration > 0, "invalid duration")
        for field in ("positive_counts", "known_counts", "negative_counts"):
            values = video.get(field)
            require(isinstance(values, list) and len(values) == 22 and all(is_count(v) and v <= n for v in values),
                    "invalid "+field)
        positive, known, negative = [video[key] for key in ("positive_counts", "known_counts", "negative_counts")]
        for c in range(22):
            require(positive[c]+negative[c] == known[c], "positive/negative/known counts disagree")
            require(known[c] == (n if c in supported else 0), "incomplete or unsupported supervision mask")
        states = video.get("four_state_counts")
        require(isinstance(states, list) and len(states) == 4 and all(is_count(v) for v in states)
                and sum(states) == n, "invalid four-state counts")
        require(states[1]+states[3] == positive[10] and states[2]+states[3] == positive[13],
                "four-state counts disagree with pair labels")
    require(subjects == set(FIT_SUBJECTS), "inventory does not cover all fitting subjects")
    return inventory


def verify_input_files(inventory, root=ROOT):
    """Hash source bytes only; do not interpret any feature or held-out arrays."""
    receipts = list(inventory["provenance"]["input_files"].values())
    receipts += [{"path": path, "sha256": digest} for path, digest in inventory["provenance"]["label_source_files"].items()]
    checked = {}
    for receipt in receipts:
        path = Path(receipt["path"])
        path = path if path.is_absolute() else Path(root)/path
        path = path.resolve()
        require(path.is_file(), "provenance source missing: "+str(path))
        digest = sha256(path)
        require(digest == receipt["sha256"], "provenance hash changed: "+str(path))
        if str(path) in checked:
            require(checked[str(path)] == digest, "conflicting source hash")
        checked[str(path)] = digest
    return checked


def verify_inventory_artifact(inventory_path, inventory, root=ROOT):
    """Bind selection facts to the completed inventory builder's audit."""
    path = Path(inventory_path)
    audit_path = path.with_name("inventory_audit.json")
    require(audit_path.is_file(), "missing verified inventory audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("schema") == "behaviorsense.rgb_fitting_coverage_inventory_audit.v1"
            and audit.get("status") == "verified_complete", "inventory audit is incomplete")
    require(audit.get("inventory_file_sha256") == sha256(path)
            and audit.get("inventory_canonical_sha256") == canonical_hash(inventory),
            "inventory differs from the audited artifact")
    builder = Path(root) / "scripts/build_rgb_coverage_inventory.py"
    require(audit.get("builder_sha256") == sha256(builder), "inventory builder source changed")
    require(audit.get("input_files") == inventory["provenance"]["input_files"],
            "inventory provenance differs from its audit")
    for key in ("all_274_extraction_metadata_audits_replayed", "frozen_input_hashes_verified",
                "frozen_label_helper_hashes_verified", "full_fitting_manifest_aggregates_match",
                "pilot_fitting_manifest_aggregates_match"):
        require(audit.get(key) is True, "missing inventory audit check: " + key)
    require(audit.get("n_videos") == len(inventory["videos"]) == 274
            and audit.get("n_windows") == sum(v["n_windows"] for v in inventory["videos"]) == 217090,
            "audited fitting inventory is incomplete")
    return {"path": str(audit_path.resolve()), "sha256": sha256(audit_path),
            "builder_sha256": audit["builder_sha256"]}


def fold_support(inventory, folds, videos=None):
    validate_folds(folds)
    videos = inventory["videos"] if videos is None else videos
    report = {}
    for label, subjects in sorted(folds.items()):
        held = [v for v in videos if v["subject"] in subjects]
        positive = [sum(v["positive_counts"][c] for v in held) for c in range(22)]
        states = [sum(v["four_state_counts"][c] for v in held) for c in range(4)]
        missing = [c for c in inventory["supervised_class_ids"] if positive[c] == 0]
        report[label] = {"subjects": sorted(subjects), "n_windows": sum(v["n_windows"] for v in held),
                         "positive_counts": positive, "four_state_counts": states,
                         "missing_class_ids": missing, "missing_class_names": [inventory["class_names"][c] for c in missing],
                         "missing_states": [c for c in range(4) if states[c] == 0]}
    return {"feasible": all(not v["missing_class_ids"] and not v["missing_states"] for v in report.values()), "folds": report}


def choose_folds(inventory):
    validate_inventory(inventory)
    original = fold_support(inventory, inventory["folds"])
    totals = {s: {"n": sum(v["n_windows"] for v in inventory["videos"] if v["subject"] == s),
                  "positive": [sum(v["positive_counts"][c] for v in inventory["videos"] if v["subject"] == s) for c in range(22)],
                  "states": [sum(v["four_state_counts"][c] for v in inventory["videos"] if v["subject"] == s) for c in range(4)]}
              for s in FIT_SUBJECTS}
    best, examined, feasible = None, 0, 0
    for others in combinations(FIT_SUBJECTS[1:], 2):
        first = (FIT_SUBJECTS[0], *others)
        remainder = tuple(s for s in FIT_SUBJECTS if s not in first)
        for others2 in combinations(remainder[1:], 2):
            second = (remainder[0], *others2)
            third = tuple(s for s in remainder if s not in second)
            groups = (first, second, third)
            examined += 1
            if any(any(sum(totals[s]["positive"][c] for s in group) == 0 for c in inventory["supervised_class_ids"])
                   or any(sum(totals[s]["states"][c] for s in group) == 0 for c in range(4)) for group in groups):
                continue
            feasible += 1
            row_counts = [sum(totals[s]["n"] for s in group) for group in groups]
            pair_counts = [sum(totals[s]["states"][3] for s in group) for group in groups]
            objective = (max(row_counts)-min(row_counts), max(pair_counts)-min(pair_counts), groups)
            if best is None or objective < best:
                best = objective
    result = {"original_folds": inventory["folds"], "original_fold_support": original,
              "partitions_examined": examined, "feasible_partitions": feasible,
              "status": "chosen" if best is not None else "proven_infeasible"}
    if best is not None:
        folds = {label: list(group) for label, group in zip(("A", "B", "C"), best[2])}
        result.update(folds=folds, row_spread=best[0], pair_window_spread=best[1], support=fold_support(inventory, folds))
    else:
        result["reason"] = "Exhaustive canonical subject partitions cannot supply every supervised class and state in each fold."
    return result


def audit_selection(inventory, selected_video_ids, folds):
    """Independent set/count checks, without referring to optimizer constraints."""
    validate_inventory(inventory)
    validate_folds(folds)
    by_id = {v["video_id"]: v for v in inventory["videos"]}
    require(isinstance(selected_video_ids, list) and len(set(selected_video_ids)) == len(selected_video_ids), "duplicate selected IDs")
    require(set(selected_video_ids) <= set(by_id), "unknown selected video")
    videos = [by_id[vid] for vid in selected_video_ids]
    errors = []
    if len(videos) != 36:
        errors.append("expected exactly 36 whole videos")
    takes = [v["take_group"] for v in videos]
    if len(takes) != len(set(takes)):
        errors.append("multiple cameras selected from the same take")
    per_subject, probe_subjects = {}, []
    for subject in FIT_SUBJECTS:
        selected = [v for v in videos if v["subject"] == subject]
        states = [sum(v["four_state_counts"][c] for v in selected) for c in range(4)]
        pair_takes = sorted({v["take_group"] for v in selected if v["four_state_counts"][3] > 0})
        probes = sorted(v["video_id"] for v in selected if 100 <= v["n_windows"] <= 1000)
        if len(selected) != 4:
            errors.append(subject+": expected exactly four videos")
        available_pair_takes = {v["take_group"] for v in inventory["videos"]
                                if v["subject"] == subject and v["four_state_counts"][3] > 0}
        required_pair_takes = max(1, min(2, len(available_pair_takes)))
        if len(pair_takes) < required_pair_takes:
            errors.append(subject+": insufficient pair-positive takes")
        if not all(states):
            errors.append(subject+": missing four-state support")
        if probes:
            probe_subjects.append(subject)
        per_subject[subject] = {"video_ids": sorted(v["video_id"] for v in selected), "four_state_counts": states,
                                "pair_positive_takes": pair_takes, "eligible_probe_video_ids": probes,
                                "available_pair_positive_takes": len(available_pair_takes),
                                "required_pair_positive_takes": required_pair_takes,
                                "pair_take_availability_shortfall": len(available_pair_takes) < 2,
                                "n_windows": sum(v["n_windows"] for v in selected)}
    per_fold_pair_takes = {}
    for fold, subjects in folds.items():
        groups = sorted({v["take_group"] for v in videos
                         if v["subject"] in subjects and v["four_state_counts"][3] > 0})
        per_fold_pair_takes[fold] = groups
        if len(groups) < POLICY["minimum_pair_positive_takes_per_fold"]:
            errors.append(fold + ": fewer than six pair-positive takes")
    per_class = {}
    for c in sorted(inventory["supervised_class_ids"]):
        subjects = sorted({v["subject"] for v in videos if v["positive_counts"][c] > 0})
        if len(subjects) < 3:
            errors.append(inventory["class_names"][c]+": fewer than three positive subjects")
        per_class[str(c)] = {"name": inventory["class_names"][c], "positive_subjects": subjects,
                             "positive_windows": sum(v["positive_counts"][c] for v in videos)}
        for label, held_subjects in folds.items():
            if not any(v["positive_counts"][c] > 0 and v["subject"] in held_subjects for v in videos):
                errors.append(label+": class absent from held fold: "+str(c))
            if not any(v["positive_counts"][c] > 0 and v["subject"] not in held_subjects for v in videos):
                errors.append(label+": class absent from fitting complement: "+str(c))
    pair_windows = sum(v["four_state_counts"][3] for v in videos)
    if pair_windows < 100:
        errors.append("fewer than 100 pair windows")
    if len(probe_subjects) < 3:
        errors.append("fewer than three subjects with an eligible selected probe video")
    if any(v["n_windows"] == 0 for v in videos):
        errors.append("empty videos cannot be selected")
    return {"passed": not errors, "violations": errors, "selected_video_count": len(videos),
            "n_windows": sum(v["n_windows"] for v in videos), "pair_windows": pair_windows,
            "per_subject": per_subject, "per_class": per_class, "probe_subjects": probe_subjects,
            "pair_positive_takes_by_fold": per_fold_pair_takes,
            "fold_support": fold_support(inventory, folds, videos)}


def select_probes(inventory, selected_video_ids):
    selected = set(selected_video_ids)
    eligible = sorted((v for v in inventory["videos"]
                       if v["video_id"] in selected and 100 <= v["n_windows"] <= 1000),
                      key=lambda v: (v["duration_seconds"] / v["n_windows"], v["video_id"]))
    probes, subjects = [], set()
    for quantile in (.15, .5, .85):
        target = round(quantile * max(0, len(eligible) - 1))
        options = [(index, v) for index, v in enumerate(eligible) if v["subject"] not in subjects]
        require(bool(options), "need three eligible probe subjects")
        _, chosen = min(options, key=lambda item: (abs(item[0] - target), item[1]["video_id"]))
        probes.append(chosen["video_id"])
        subjects.add(chosen["subject"])
    return probes


def select_videos(inventory, folds, time_limit_seconds=60):
    validate_inventory(inventory)
    validate_folds(folds)
    require(isinstance(time_limit_seconds, (int, float)) and math.isfinite(time_limit_seconds)
            and 0 < time_limit_seconds <= POLICY["solver_time_limit_seconds"], "invalid solver time limit")
    import numpy as np
    from scipy import __version__ as scipy_version
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    videos = sorted((v for v in inventory["videos"] if v["n_windows"] > 0), key=lambda v: v["video_id"])
    n = len(videos)
    classes = sorted(inventory["supervised_class_ids"])
    indicators = {(c, s): n+i for i, (c, s) in enumerate((c, s) for c in classes for s in FIT_SUBJECTS)}
    probes = {s: n+len(indicators)+i for i, s in enumerate(FIT_SUBJECTS)}
    width = n+len(indicators)+len(probes)
    row_indices, col_indices, values, lower, upper = [], [], [], [], []

    def add(coefficients, lo=-np.inf, hi=np.inf):
        row = len(lower)
        for col, value in coefficients.items():
            if value:
                row_indices.append(row); col_indices.append(col); values.append(float(value))
        lower.append(lo); upper.append(hi)

    for subject in FIT_SUBJECTS:
        indices = [i for i, v in enumerate(videos) if v["subject"] == subject]
        add({i: 1 for i in indices}, 4, 4)
        pair_indices = [i for i in indices if videos[i]["four_state_counts"][3] > 0]
        available_pair_takes = len({videos[i]["take_group"] for i in pair_indices})
        add({i: 1 for i in pair_indices}, max(1, min(2, available_pair_takes)))
        for state in range(4):
            add({i: videos[i]["four_state_counts"][state] for i in indices}, 1)
        add({**{i: 1 for i in indices if 100 <= videos[i]["n_windows"] <= 1000}, probes[subject]: -1}, 0)
    for take in sorted({v["take_group"] for v in videos}):
        add({i: 1 for i, v in enumerate(videos) if v["take_group"] == take}, hi=1)
    for subjects in folds.values():
        add({i: 1 for i, v in enumerate(videos)
             if v["subject"] in subjects and v["four_state_counts"][3] > 0},
            POLICY["minimum_pair_positive_takes_per_fold"])
    for c in classes:
        for subject in FIT_SUBJECTS:
            add({**{i: 1 for i, v in enumerate(videos) if v["subject"] == subject and v["positive_counts"][c] > 0},
                 indicators[c, subject]: -1}, 0)
        add({indicators[c, subject]: 1 for subject in FIT_SUBJECTS}, 3)
        for subjects in folds.values():
            for held in (True, False):
                add({i: 1 for i, v in enumerate(videos) if ((v["subject"] in subjects) == held) and v["positive_counts"][c] > 0}, 1)
    add({i: v["four_state_counts"][3] for i, v in enumerate(videos)}, 100)
    add({index: 1 for index in probes.values()}, 3)
    matrix = coo_matrix((values, (row_indices, col_indices)), shape=(len(lower), width)).tocsc()
    objective = np.zeros(width)
    if n:
        objective[:n] = [v["n_windows"]+(i+1)/(2*n*(n+1)) for i, v in enumerate(videos)]
    began = time.monotonic()
    solved = milp(objective, integrality=np.ones(width), bounds=Bounds(np.zeros(width), np.ones(width)),
                  constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": float(time_limit_seconds), "mip_rel_gap": 0., "presolve": True})
    status = {0: "optimal", 1: "solver_limit", 2: "proven_infeasible"}.get(solved.status, "solver_error")
    result = {"status": status, "solver_status": int(solved.status), "solver_message": str(solved.message),
              "solver": "scipy.optimize.milp/HiGHS", "scipy_version": scipy_version,
              "elapsed_seconds": time.monotonic()-began, "time_limit_seconds": time_limit_seconds,
              "selected_video_ids": [], "constraint_relaxation": False}
    for key in ("mip_gap", "mip_node_count", "mip_dual_bound", "fun"):
        value = getattr(solved, key, None)
        if value is not None and math.isfinite(float(value)):
            result[key] = float(value)
    if status == "optimal":
        vector = getattr(solved, "x", None)
        require(vector is not None and np.shape(vector) == (width,) and np.isfinite(vector).all()
                and np.max(np.abs(vector-np.rint(vector))) <= 1e-6
                and np.min(vector) >= -1e-6 and np.max(vector) <= 1+1e-6, "invalid optimal integer solution")
        selected = [v["video_id"] for i, v in enumerate(videos) if vector[i] > .5]
        audited = audit_selection(inventory, selected, folds)
        require(audited["passed"], "optimizer solution failed independent audit: "+"; ".join(audited["violations"]))
        result.update(selected_video_ids=selected, pilot_video_ids=selected,
                      probe_video_ids=select_probes(inventory, selected), audit=audited,
                      objective={"fitting_windows": audited["n_windows"],
                                 "stable_id_tie_cost": sum((i+1)/(2*n*(n+1)) for i in range(n) if vector[i] > .5)})
    elif status == "solver_limit":
        result["reason"] = "Optimizer limit reached; feasibility and optimality are unresolved. No candidate accepted and no constraints relaxed."
    elif status == "proven_infeasible":
        result["reason"] = "Solver proved the frozen whole-video constraints infeasible. No constraints relaxed."
    return result


def write_immutable(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run(inventory_path, output_dir):
    inventory_path, output_dir = Path(inventory_path).resolve(), Path(output_dir).resolve()
    inventory_bytes = inventory_path.read_bytes()
    inventory = validate_inventory(json.loads(inventory_bytes))
    audit_receipt = verify_inventory_artifact(inventory_path, inventory)
    verified_sources = verify_input_files(inventory)
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not (output_dir/"recipe.json").exists() and not (output_dir/"selection.json").exists(),
            "immutable selection output already exists")
    source_hash = sha256(__file__)
    input_hash = hashlib.sha256(inventory_bytes).hexdigest()
    recipe = {"schema": "behaviorsense.rgb_coverage_selection_recipe.v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy": POLICY, "policy_sha256": canonical_hash(POLICY), "source_sha256": source_hash,
              "inventory_path": str(inventory_path), "inventory_sha256": input_hash,
              "inventory_semantic_sha256": canonical_hash(inventory), "verified_source_files": verified_sources,
              "inventory_audit": audit_receipt}
    write_immutable(output_dir/"recipe.json", recipe)
    with (output_dir / "selector_source.py").open("xb") as frozen_source:
        frozen_source.write(Path(__file__).read_bytes())
    folds = choose_folds(inventory)
    selection = select_videos(inventory, folds["folds"]) if folds["status"] == "chosen" else {
        "status": "proven_infeasible", "reason": folds["reason"], "selected_video_ids": []}
    require(sha256(inventory_path) == input_hash and sha256(__file__) == source_hash, "frozen input or selector changed during selection")
    require(verify_input_files(inventory) == verified_sources, "provenance changed during selection")
    require(verify_inventory_artifact(inventory_path, inventory) == audit_receipt,
            "inventory audit changed during selection")
    selection.update(schema="behaviorsense.rgb_coverage_selection.v1", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     recipe_sha256=sha256(output_dir/"recipe.json"), source_sha256=source_hash,
                     inventory_sha256=input_hash, fold_selection=folds, model_candidate_created=False)
    write_immutable(output_dir/"selection.json", selection)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.inventory, args.output_dir)
    print(json.dumps({"status": result["status"], "selected_videos": len(result["selected_video_ids"]),
                      "selection": str(args.output_dir.resolve()/"selection.json")}))


if __name__ == "__main__":
    main()
