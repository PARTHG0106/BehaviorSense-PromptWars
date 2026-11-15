"""Finite RGB/control head experiment over complete, identity-bound feature caches.

Pilot fits exactly one corrected-pose/object control and one RGB+control head.
Full training fits one RGB+control head. Gates, normalization, epoch selection and
thresholds use fitting/validation subjects only; failed gates forbid test scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROGRAM_STARTED = time.monotonic()

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_adl_multilabel as baseline
import train_multilabel_temporal as temporal
from behaviorsense.data.multilabel_skeleton import load_multilabel_shards, _verify_full_audit
from behaviorsense.data.pose_contract_v2 import POSE_CONTRACT_VERSION, normalize_visible_pose, to_bone_visible
from behaviorsense.data.toyota import load_tsu_annotations
from behaviorsense.models.rgb_action import PREPROCESSING_SHA256, person_roi, canonical_sha256
from extract_rgb_action_features import SCHEMA, _load_shard, _array_sha256, _encoder_contract, source_frame_indices

FORMAT = "behaviorsense.rgb_control_multilabel.v1"
VAL_SUBJECTS = ["p04", "p15"]
FULL_COUNTS = {"training_videos": 351, "fit_windows": 217090, "validation_windows": 63220,
               "test_videos": 185, "test_windows": 153537}
CONTROL_DIM = 2 * (4 * 2 * 17 * 3 + 2 * 2 * 17 * 2) + 401
IDENTITY_KEYS = ("schema", "video_id", "subject", "n_windows", "arch", "feature_dim",
    "source_video_sha256", "source_shard_sha256", "encoder_sha256", "encoder_info",
    "preprocessing_sha256", "extraction_implementation_sha256", "source_fps", "source_frame_count",
    "source_width", "source_height", "views")
SOURCE_FILES = ("scripts/train_rgb_multilabel.py", "scripts/extract_rgb_action_features.py",
    "scripts/train_adl.py", "src/behaviorsense/models/stgcnpp.py",
    "scripts/train_adl_multilabel.py", "scripts/train_multilabel_temporal.py",
    "src/behaviorsense/data/pose_contract_v2.py", "src/behaviorsense/data/multilabel_skeleton.py",
    "src/behaviorsense/data/toyota.py", "src/behaviorsense/eval/multilabel.py",
    "src/behaviorsense/models/rgb_action.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deadline_check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("finite RGB head allocation expired")


def _hash(value, deadline=None):
    digest = hashlib.sha256()
    with Path(value).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            deadline_check(deadline)
            digest.update(block)
    deadline_check(deadline)
    return digest.hexdigest()


def validate_runtime_options(args):
    if (min(args.batch_size, args.chunk_size, args.cpu_threads) < 1
            or (args.stop_after_seconds is not None
                and (not math.isfinite(args.stop_after_seconds) or args.stop_after_seconds < 0))):
        raise ValueError("invalid finite RGB runtime configuration")


def _sha_value(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 digest")


def select_shards(args, annotations):
    """Filter a pilot from metadata before loading pose/RGB arrays or labels."""
    training = args.command == "train"
    subset = "training" if training else "testing"
    paths = baseline.expand_shards(args.shards)
    available = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            if "extraction_metadata" not in source.files:
                raise ValueError("source shard requires a complete per-video extraction audit")
            report = json.loads(str(source["extraction_metadata"]))
        vid = report.get("video_id")
        if vid not in annotations or annotations[vid].subset != subset or report.get("subset") != subset:
            raise ValueError("source shard belongs to an unknown or forbidden subject/subset")
        if vid in available:
            raise ValueError(f"duplicate source video {vid}")
        available[vid] = Path(path)
    all_expected = {vid for vid, video in annotations.items() if video.subset == subset}
    if training and args.scope == "pilot":
        if not args.pilot_video_ids:
            raise ValueError("pilot requires the prespecified fitting-video inventory")
        pilot = read(args.pilot_video_ids)
        if (not isinstance(pilot, list) or not pilot or any(not isinstance(vid, str) for vid in pilot)
                or len(set(pilot)) != len(pilot)):
            raise ValueError("pilot video IDs must be a nonempty unique string list")
        if any(vid not in all_expected or f"p{annotations[vid].subject:02d}" in VAL_SUBJECTS for vid in pilot):
            raise ValueError("pilot IDs must contain fitting subjects only")
        validation = {vid for vid in all_expected if f"p{annotations[vid].subject:02d}" in VAL_SUBJECTS}
        wanted = set(pilot) | validation
    else:
        wanted = all_expected
        if set(available) != wanted:
            raise ValueError("full coverage requires exactly every official source video")
    if not wanted <= set(available):
        raise ValueError(f"selected source coverage is missing {sorted(wanted - set(available))}")
    return [available[vid] for vid in sorted(wanted)]


def validate_rgb_cache(path, shard_path, deadline=None):
    """Bind every feature/clock/crop array to its exact source shard and pixels."""
    deadline_check(deadline)
    source, report = _load_shard(shard_path)
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError("RGB cache lacks metadata")
        metadata = json.loads(str(archive["metadata"]))
        if any(key not in metadata for key in IDENTITY_KEYS):
            raise ValueError("RGB cache has an incomplete extraction identity")
        identity = {key: metadata[key] for key in IDENTITY_KEYS}
        if metadata.get("schema") != SCHEMA or metadata.get("status") != "complete":
            raise ValueError("RGB cache must be a complete current-schema video")
        if metadata.get("fingerprint") != canonical_sha256(identity):
            raise ValueError("RGB cache identity fingerprint mismatch")
        _encoder_contract(metadata["encoder_info"])
        for name in ("source_video_sha256", "source_shard_sha256", "encoder_sha256",
                     "preprocessing_sha256", "extraction_implementation_sha256"):
            _sha_value(metadata[name], name)
        n = len(source["starts"])
        if (metadata["video_id"] != report["video_id"] or metadata["subject"] != report["subject"]
                or metadata["source_video_sha256"] != report["source_sha256"]
                or metadata["source_shard_sha256"] != _hash(shard_path, deadline)
                or metadata["n_windows"] != n or metadata["source_fps"] != report["source_fps"]
                or metadata["source_frame_count"] != report["source_frame_count"]):
            raise ValueError("RGB cache pixel/shard/row identity disagrees with source")
        info = metadata["encoder_info"]
        if (metadata["encoder_sha256"] != info["weights_sha256"] or metadata["arch"] != info["arch"]
                or metadata["feature_dim"] != info["feature_dim"]
                or metadata["preprocessing_sha256"] != PREPROCESSING_SHA256
                or metadata["views"] != ["scene", "person"]):
            raise ValueError("RGB encoder/preprocessing contract mismatch")
        frames, boundary = source_frame_indices(source["source_frames"][:, 15],
            report["source_fps"], report["source_frame_count"])
        rois = np.empty((n, 4), dtype=np.int64)
        observed = np.empty(n, dtype=bool)
        for row in range(n):
            if row % 256 == 0:
                deadline_check(deadline)
            rois[row], observed[row] = person_roi(source["skeletons"][row],
                metadata["source_width"], metadata["source_height"])
        expected = {"starts": source["starts"].astype(np.int64),
            "video_id": np.asarray(report["video_id"]), "subject": np.asarray(report["subject"]),
            "video_ids": np.full(n, report["video_id"]), "subjects": np.full(n, report["subject"]),
            "center_source_frames": source["source_frames"][:, 15].astype(np.int64),
            "source_frames_rgb": frames, "boundary_duplicate_flags": boundary,
            "person_rois": rois, "roi_observed": observed,
            "source_fps": np.asarray(report["source_fps"], dtype=np.float64)}
        if set(archive.files) != set(expected) | {"features", "metadata"}:
            raise ValueError("RGB cache arrays do not match the frozen schema")
        for key, value in expected.items():
            if archive[key].dtype != value.dtype or not np.array_equal(archive[key], value):
                raise ValueError(f"RGB cache source order/clock/crop mismatch: {key}")
        features = archive["features"]
        if features.dtype != np.float16 or features.shape != (n, 2, metadata["feature_dim"]) or not np.isfinite(features).all():
            raise ValueError("RGB features must be finite float16 [N,2,D]")
        hashes = {key: _array_sha256(value) for key, value in {**expected, "features": features}.items()}
        if hashes != metadata.get("array_sha256"):
            raise ValueError("RGB cache array checksum mismatch")
    deadline_check(deadline)
    contract = {key: metadata[key] for key in ("schema", "arch", "feature_dim", "encoder_sha256",
        "encoder_info", "preprocessing_sha256", "extraction_implementation_sha256", "views")}
    return features, contract, metadata


def pose_control_features(skeletons, object_context):
    """Mean/std/first/last pose and bone channels plus valid xy velocity moments."""
    normalized = np.stack([normalize_visible_pose(window) for window in skeletons])
    joint_tensor = torch.from_numpy(normalized.transpose(0, 4, 1, 3, 2).copy())
    bones = to_bone_visible(joint_tensor).numpy().transpose(0, 2, 4, 3, 1)
    features = []
    for values in (normalized, bones):
        flat = values.reshape(len(values), values.shape[1], -1)
        features.extend([flat.mean(axis=1), flat.std(axis=1), flat[:, 0], flat[:, -1]])
        valid = (values[:, 1:, ..., 2] >= .3) & (values[:, :-1, ..., 2] >= .3)
        velocity = (values[:, 1:, ..., :2] - values[:, :-1, ..., :2]) * np.float32(20.)
        velocity = np.where(valid[..., None], velocity, 0.)
        count = valid.sum(axis=1)[..., None].clip(1)
        mean = velocity.sum(axis=1) / count
        variance = ((velocity - mean[:, None]) ** 2 * valid[..., None]).sum(axis=1) / count
        features.extend([mean.reshape(len(values), -1), np.sqrt(variance).reshape(len(values), -1)])
    result = np.concatenate([*features, np.asarray(object_context, dtype=np.float32)], axis=1).astype(np.float32)
    if result.shape != (len(skeletons), CONTROL_DIM) or not np.isfinite(result).all():
        raise ValueError("invalid corrected-pose/object control features")
    return result


def load_data(args, deadline=None):
    """Real-source loader seam; tests may mock this without weakening production checks."""
    annotations = load_tsu_annotations(args.annotations)
    paths = select_shards(args, annotations)
    training = args.command == "train"
    full = not training or args.scope == "full"
    reports = []
    for path in paths:
        deadline_check(deadline)
        with np.load(path, allow_pickle=False) as z:
            report = json.loads(str(z["extraction_metadata"]))
            _verify_full_audit(report, z["starts"], annotations[report["video_id"]],
                sample_fps=20., annotation_fps=25., min_overlap=.5)
            if "fingerprint" not in z.files or str(z["fingerprint"]) != report["fingerprint"]:
                raise ValueError("source stored/extraction fingerprint mismatch")
            skeletons = z["skeletons"]
            visibility = (skeletons[:, :, 0, :, 2].astype(np.float32) > .3).sum(axis=2)
            if ((visibility >= 8).sum(axis=1) < 15).any():
                raise ValueError("source retained rows fail the audited visibility gate")
        reports.append(report)
    cache = Path(args.out) / "data_cache"
    data = load_multilabel_shards(paths, args.annotations, require_context=True,
        require_full_coverage=full, expected_subset="training" if training else "testing", cache_dir=cache)
    data["selected_video_ids"] = [report["video_id"] for report in reports]
    data["selected_videos_coverage_verified"] = True
    if full and len(paths) != FULL_COUNTS["training_videos" if training else "test_videos"]:
        raise ValueError("full source video count differs from the official inventory")
    contract, matrix, offset, receipts = None, None, 0, []
    for path, report in zip(paths, reports):
        deadline_check(deadline)
        rgb_path = Path(args.features_dir) / f"{report['video_id']}.npz"
        values, actual, metadata = validate_rgb_cache(rgb_path, path, deadline)
        if contract is None:
            contract = actual
            cache.mkdir(parents=True, exist_ok=True)
            matrix = np.lib.format.open_memmap(cache / "combined_features.npy", mode="w+", dtype=np.float32,
                shape=(len(data["starts"]), CONTROL_DIM + values.shape[1] * values.shape[2]))
        elif actual != contract:
            raise ValueError("RGB encoder/preprocessing/implementation contract changed between videos")
        n = len(values)
        if (not np.all(data["video_ids"][offset:offset+n] == report["video_id"])
                or not np.array_equal(data["starts"][offset:offset+n], np.asarray(report["retained_starts"]))):
            raise ValueError("source loader and RGB cache row ordering differ")
        for local in range(0, n, args.chunk_size):
            deadline_check(deadline)
            stop = min(n, local + args.chunk_size)
            rows = slice(offset + local, offset + stop)
            matrix[rows, :CONTROL_DIM] = pose_control_features(data["skeletons"][rows], data["visual_context"][rows])
            matrix[rows, CONTROL_DIM:] = values[local:stop].reshape(stop-local, -1)
        offset += n
        receipts.append({"video_id": report["video_id"], "n_windows": n, "feature_sha256": _hash(rgb_path, deadline),
            "source_video_sha256": metadata["source_video_sha256"],
            "source_shard_sha256": metadata["source_shard_sha256"], "fingerprint": metadata["fingerprint"]})
    if offset != len(data["starts"]):
        raise ValueError("RGB cache does not cover every selected source row")
    matrix.flush()
    data.update(features=matrix, feature_contract=contract, feature_inventory=receipts,
                feature_inventory_sha256=canonical_sha256(receipts))
    return data


def fit_normalization(features, indices, chunk_size=2048, deadline=None):
    """Fit each feature independently using fitting rows only, in float64 chunks."""
    indices = np.asarray(indices)
    if not len(indices):
        raise ValueError("normalization requires fitting rows")
    total, square = np.zeros(features.shape[1], np.float64), np.zeros(features.shape[1], np.float64)
    for start in range(0, len(indices), chunk_size):
        deadline_check(deadline)
        block = np.asarray(features[indices[start:start+chunk_size]], dtype=np.float64)
        total += block.sum(axis=0)
        square += np.square(block).sum(axis=0)
    mean = total / len(indices)
    std = np.sqrt(np.maximum(square / len(indices) - mean ** 2, 0)).clip(1e-4)
    return mean.astype(np.float32), std.astype(np.float32)


def standardized_tensor(features, mean, std, device, chunk_size=2048, deadline=None):
    mean, std = np.asarray(mean), np.asarray(std)
    if (features.ndim != 2 or chunk_size < 1 or mean.shape != (features.shape[1],)
            or std.shape != mean.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std < np.float32(1e-4))):
        raise ValueError("normalization requires finite per-feature means and positive floored scales")
    result = torch.empty(features.shape, dtype=torch.float32, device=device)
    for start in range(0, len(features), chunk_size):
        deadline_check(deadline)
        raw = np.asarray(features[start:start+chunk_size])
        if not np.isfinite(raw).all():
            raise ValueError("non-finite input features")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            block = np.clip((raw - mean) / std, -8., 8.).astype(np.float32)
        result[start:start+len(block)] = torch.from_numpy(block).to(device)
    deadline_check(deadline)
    return result


class RGBHead(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=.3):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 22))

    def forward(self, features):
        return self.layers(features)


@torch.no_grad()
def predict(model, features, indices, batch_size=2048, deadline=None):
    model.float().eval()
    output = []
    for start in range(0, len(indices), batch_size):
        deadline_check(deadline)
        rows = torch.as_tensor(indices[start:start+batch_size], device=features.device)
        values = model(features[rows]).float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite head predictions")
        output.append(values)
    deadline_check(deadline)
    return np.concatenate(output)


def validate_gate_criteria(criteria):
    required = {"macro_ap_gain_over_matched_control", "macro_ap_min", "pair_f1_min",
                "pair_precision_min", "pair_recall_min", "subjects"}
    if (not isinstance(criteria, dict) or set(criteria) != required
            or not isinstance(criteria["subjects"], dict)
            or set(criteria["subjects"]) != set(VAL_SUBJECTS)):
        raise ValueError("gate criteria schema must include both fixed validation subjects")
    thresholds = [criteria[key] for key in required - {"subjects"}]
    for minimum in criteria["subjects"].values():
        if not isinstance(minimum, dict) or set(minimum) != {"pair_f1_min", "macro_ap_min"}:
            raise ValueError("subject gate schema is invalid")
        thresholds.extend(minimum.values())
    if any(isinstance(value, bool) or not isinstance(value, (float, int))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("gate thresholds must be finite in [0,1]")


def gate_decision(criteria, rgb, control=None, scope="pilot"):
    """Apply the frozen aggregate and subject gates without consulting test data."""
    validate_gate_criteria(criteria)
    if scope not in ("pilot", "full"):
        raise ValueError("gate scope must be pilot or full")
    checks = {}
    def threshold(name, actual, minimum):
        checks[name] = {"actual": actual, "minimum": minimum,
            "passed": not isinstance(actual, bool) and isinstance(actual, (float, int))
            and math.isfinite(actual) and actual >= minimum}
    metrics = rgb["metrics"]
    threshold("macro_ap", metrics.get("macro_ap"), criteria["macro_ap_min"])
    if scope == "pilot":
        if control is None:
            raise ValueError("pilot requires its matched control")
        threshold("macro_ap_gain_over_matched_control", metrics["macro_ap"] - control["metrics"]["macro_ap"],
                  criteria["macro_ap_gain_over_matched_control"])
    for metric in ("f1", "precision", "recall"):
        threshold("pair_" + metric, metrics["tv_drinking_colabeled_windows"].get(metric), criteria["pair_"+metric+"_min"])
    for subject, minimum in criteria["subjects"].items():
        actual = rgb["per_subject"].get(subject, {})
        threshold(subject + ".macro_ap", actual.get("macro_ap"), minimum["macro_ap_min"])
        threshold(subject + ".pair_f1", actual.get("tv_drinking_colabeled_windows", {}).get("f1"), minimum["pair_f1_min"])
    failures = [name for name, check in checks.items() if not check["passed"]]
    return {"passed": not failures, "reason": "all predeclared validation gates passed" if not failures else "failed: " + ", ".join(failures),
        "criteria": criteria, "scope": scope, "checks": checks, "metrics": {"control": control, "rgb": rgb},
        "test_metrics_consulted": False}


def _train_head(args, data, features, train_idx, val_idx, normalization, manifest, directory, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    baseline.set_seed(0)
    model = RGBHead(features.shape[1], args.hidden, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.lr_patience,
        factor=.5, min_lr=1e-6, threshold=1e-8, threshold_mode="abs")
    targets = torch.as_tensor(data["targets"], device=args.device)
    masks = torch.as_tensor(data["target_mask"], device=args.device)
    positive = np.asarray(manifest["train"]["positive"])
    negative = np.asarray(manifest["train"]["negative"])
    weights = np.clip(np.sqrt(negative / np.maximum(positive, 1)), 1., 10.)
    weights[positive == 0] = 1.
    pos_weight = torch.as_tensor(weights, dtype=torch.float32, device=args.device)
    order_rng = torch.Generator(device=args.device).manual_seed(0)
    fitting = torch.as_tensor(train_idx, device=args.device)
    history, best, best_epoch, best_state, best_logits = [], -1., -1, None, None
    lr_reductions, last_lr_reduction_epoch = 0, -1
    config = {"input_dim": features.shape[1], "hidden": args.hidden, "dropout": args.dropout,
              "batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay, "seed": 0}
    def convergence():
        return baseline.convergence_report(len(history), best_epoch, min_epochs=args.min_epochs,
            epochs=args.epochs, patience=args.patience, lr_reductions=lr_reductions,
            last_lr_reduction_epoch=last_lr_reduction_epoch)
    def status(value, **extra):
        manifest["convergence"] = convergence()
        manifest["checkpoint_selection"] = baseline.selection_report(best, best_epoch, 0, manifest["validation_fingerprint"])
        baseline.write_json(directory / "manifest.json", manifest)
        baseline.write_json(directory / "history.json", history)
        baseline.write_json(directory / "training_status.json", {"status": value,
            "completed_epochs": len(history), "convergence": convergence(),
            "inference_checkpoint_exported": False, **extra})
    status("running")
    try:
        for epoch in range(args.epochs):
            deadline_check(deadline)
            model.train()
            shuffled = fitting[torch.randperm(len(fitting), generator=order_rng, device=args.device)]
            total, seen, began = 0., 0, time.monotonic()
            for start in range(0, len(shuffled), args.batch_size):
                deadline_check(deadline)
                rows = shuffled[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = baseline.masked_bce(model(features[rows]), targets[rows], masks[rows], pos_weight)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite RGB head loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach()) * len(rows)
                seen += len(rows)
                deadline_check(deadline)
            if seen != len(train_idx):
                raise RuntimeError("head epoch omitted fitting rows")
            logits = predict(model, features, val_idx, args.batch_size, deadline)
            metrics = temporal.assess(data, val_idx, logits)
            score = metrics["macro_ap"]
            if score is None or not math.isfinite(score):
                raise ValueError("validation has no finite macro AP")
            deadline_check(deadline)
            if score > best + 1e-8:
                best, best_epoch = score, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_logits = logits.copy()
            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(score)
            if optimizer.param_groups[0]["lr"] < previous_lr:
                lr_reductions += 1
                last_lr_reduction_epoch = epoch
            history.append({"epoch": epoch, "train_loss": total/seen, "fit_windows_seen": seen,
                "validation_macro_ap": score, "seconds": time.monotonic()-began,
                "learning_rate": optimizer.param_groups[0]["lr"]})
            status("running")
            print(f"{manifest['head']} epoch {epoch+1}: loss={total/seen:.5f} val_mAP={score:.5f}", flush=True)
            if convergence()["converged"]:
                break
        if not convergence()["converged"]:
            status("incomplete_epoch_limit")
            return None
        deadline_check(deadline)
    except TimeoutError:
        status("incomplete_budget", interrupted_epoch_counted=False)
        raise
    threshold_config = baseline.fit_thresholds(baseline.probabilities(best_logits), data["targets"][val_idx],
        data["target_mask"][val_idx], min_positive=args.threshold_min_support,
        min_negative=args.threshold_min_support, train_positive=positive)
    metrics = temporal.assess(data, val_idx, best_logits, threshold_config["thresholds"])
    if not math.isclose(metrics["macro_ap"], best, rel_tol=0., abs_tol=1e-8):
        raise RuntimeError("frozen predictions do not reproduce the validation selection score")
    subjects = temporal.assess_by_subject(data, val_idx, best_logits, threshold_config["thresholds"])
    deadline_check(deadline)
    return {"format": FORMAT, "kind": "inference", "config": config, "state_dict": best_state,
        "normalization": {"mean": normalization[0], "std": normalization[1], "source_split": "fit", "clip": 8., "std_floor": 1e-4},
        "manifest": manifest, "threshold_config": threshold_config, "validation_metrics": metrics,
        "validation_by_subject": subjects, "_logits": best_logits}


def export_head(artifact, directory, data, val_idx):
    baseline.write_json(directory / "manifest.json", artifact["manifest"])
    baseline.write_json(directory / "thresholds.json", artifact["threshold_config"])
    baseline.write_json(directory / "validation_metrics.json", artifact["validation_metrics"])
    baseline.write_json(directory / "validation_by_subject.json", artifact["validation_by_subject"])
    temporal.save_predictions(directory / "validation_logits.npz", artifact["_logits"], data, val_idx)
    saved = {key: value for key, value in artifact.items() if not key.startswith("_")}
    baseline.save_checkpoint(directory / "best_rgb_multilabel.pt", saved)
    baseline.write_json(directory / "training_status.json", {"status": "complete", "inference_checkpoint_exported": True,
        "convergence": artifact["manifest"]["convergence"], "gate_passed": artifact["manifest"].get("gate_passed", False)})


def train(args):
    validate_runtime_options(args)
    began = getattr(args, "_program_started", None) or time.monotonic()
    deadline = began + args.stop_after_seconds if args.stop_after_seconds is not None else None
    out = Path(args.out)
    if any((out/name).exists() for name in ("manifest.json", "training_status.json", "gate.json", "best_rgb_multilabel.pt")):
        raise ValueError("RGB head run cannot resume or overwrite prior outputs")
    out.mkdir(parents=True, exist_ok=True)
    criteria = read(args.gate_config)
    validate_gate_criteria(criteria)
    if (args.epochs < 1 or not 1 <= args.min_epochs <= args.epochs or args.patience < 1
            or min(args.hidden, args.lr_patience) < 1
            or not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0
            or args.threshold_min_support < 5 or not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.dropout))):
        raise ValueError("invalid finite RGB training configuration")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(True)
    def reject(reason):
        # This path belongs to this fresh run. An expired export must not leave
        # an apparently usable final inference artifact beside a failed gate.
        (out / "best_rgb_multilabel.pt").unlink(missing_ok=True)
        gate = {"passed": False, "reason": reason, "criteria": criteria, "metrics": {"control": None, "rgb": None}}
        baseline.write_json(out / "gate.json", gate)
        baseline.write_json(out / "frozen_selection.json", {"gate_passed": False, "gate": gate,
            "checkpoint_relative_path": "best_rgb_multilabel.pt", "test_metrics_consulted": False})
        prior = read(out / "training_status.json") if (out / "training_status.json").is_file() else {}
        baseline.write_json(out / "training_status.json", {**prior, "status": reason, "inference_checkpoint_exported": False})
        return gate
    try:
        deadline_check(deadline)
        data = load_data(args, deadline)
        train_idx, val_idx = baseline.subject_split(data["subjects"], val_subjects=VAL_SUBJECTS)
        if not data["target_mask"].any(axis=1).all():
            raise ValueError("verified retained source windows must all have known labels")
        train_report, validation_report = baseline.split_report(data, train_idx), baseline.split_report(data, val_idx)
        baseline.require_class_support({"train": train_report, "validation": validation_report}, args.threshold_min_support)
        if args.scope == "full" and (len(train_idx) != FULL_COUNTS["fit_windows"]
                or len(val_idx) != FULL_COUNTS["validation_windows"]):
            raise ValueError("full fitting/validation window counts differ from frozen original coverage")
        mean, std = fit_normalization(data["features"], train_idx, args.chunk_size, deadline)
        features = standardized_tensor(data["features"], mean, std, args.device, args.chunk_size, deadline)
        common = {"format": FORMAT, "scope": args.scope, "seed": 0, "model": "rgb-control-mlp",
            "class_names": list(baseline.COARSE_V11), "train": train_report, "validation": validation_report,
            "annotation_sha256": _hash(args.annotations, deadline), "source_provenance": data["provenance"],
            "selected_video_ids": data["selected_video_ids"], "selected_videos_coverage_verified": True,
            "full_eligible_window_coverage_verified": args.scope == "full",
            "feature_contract": data["feature_contract"], "feature_inventory": data["feature_inventory"],
            "feature_inventory_sha256": data["feature_inventory_sha256"],
            "validation_fingerprint": temporal.validation_fingerprint(data, val_idx),
            "fitting_fingerprint": temporal.validation_fingerprint(data, train_idx),
            "source_files": [{"path": path, "sha256": _hash(ROOT/path, deadline)} for path in SOURCE_FILES],
            "control_contract": {"pose": POSE_CONTRACT_VERSION, "dimension": CONTROL_DIM,
                "statistics": "joint/bone temporal mean,std,first,last; valid xy velocity mean/std at20Hz; original401object features"},
            "normalization_source": "fit_subjects_only", "normalization_clip": 8., "normalization_std_floor": 1e-4,
            "initialization": "scratch", "test_loaded_during_training": False,
            "previous_test_inspected": True, "automatic_continuation": False,
            "deployment_status": "experimental_not_serving_compatible", "gate_criteria": criteria,
            "evaluation_scope": "all retained eligible windows in selected videos; reused exploratory test benchmark"}
        control = None
        if args.scope == "pilot":
            control = _train_head(args, data, features[:, :CONTROL_DIM], train_idx, val_idx,
                (mean[:CONTROL_DIM], std[:CONTROL_DIM]), dict(common, head="control"), out/"control", deadline)
            if control is None:
                return reject("incomplete_epoch_limit")
            export_head(control, out/"control", data, val_idx)
        rgb = _train_head(args, data, features, train_idx, val_idx, (mean, std), dict(common, head="rgb"), out, deadline)
        if rgb is None:
            return reject("incomplete_epoch_limit")
        rgb_summary = {"metrics": rgb["validation_metrics"], "per_subject": rgb["validation_by_subject"]}
        control_summary = ({"metrics": control["validation_metrics"], "per_subject": control["validation_by_subject"]}
                           if control else None)
        gate = gate_decision(criteria, rgb_summary, control_summary, args.scope)
        deadline_check(deadline)
        rgb["manifest"]["gate_passed"] = gate["passed"]
        export_head(rgb, out, data, val_idx)
        baseline.write_json(out/"gate.json", gate)
        selection = {"format": FORMAT, "scope": args.scope, "source_split": "validation", "gate_passed": gate["passed"],
            "gate": gate, "checkpoint_relative_path": "best_rgb_multilabel.pt",
            "checkpoint_sha256": _hash(out/"best_rgb_multilabel.pt", deadline),
            "thresholds_sha256": _hash(out/"thresholds.json", deadline),
            "validation_metrics": rgb["validation_metrics"], "threshold_source_split": "validation",
            "feature_contract": data["feature_contract"], "test_metrics_consulted": False,
            "test_benchmark_previously_inspected": True}
        baseline.write_json(out/"frozen_selection.json", selection)
        deadline_check(deadline)
        return selection
    except TimeoutError:
        return reject("incomplete_budget")


def evaluate_test(args):
    validate_runtime_options(args)
    began = getattr(args, "_program_started", None) or time.monotonic()
    deadline = began + args.stop_after_seconds if args.stop_after_seconds is not None else None
    out = Path(args.out)
    if any((out/name).exists() for name in ("test_metrics.json", "test_logits.npz", "training_status.json")):
        raise ValueError("frozen test evaluation cannot overwrite prior outputs")
    out.mkdir(parents=True, exist_ok=True)
    try:
        deadline_check(deadline)
        checkpoint_sha = _hash(args.checkpoint, deadline)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        manifest = saved["manifest"]
        if (saved.get("format") != FORMAT or saved.get("kind") != "inference" or manifest.get("scope") != "full"
                or manifest.get("head") != "rgb" or manifest.get("gate_passed") is not True
                or manifest.get("convergence", {}).get("converged") is not True
                or manifest.get("test_loaded_during_training") is not False
                or saved["threshold_config"].get("source_split") != "validation"
                or manifest.get("checkpoint_selection", {}).get("source_split") != "validation"
                or manifest.get("full_eligible_window_coverage_verified") is not True
                or manifest.get("class_names") != list(baseline.COARSE_V11)):
            raise ValueError("test scoring requires a converged full-scope gate-passed frozen RGB checkpoint")
        fitting_subjects, validation_subjects = set(manifest["train"]["subjects"]), set(manifest["validation"]["subjects"])
        if (not fitting_subjects or not fitting_subjects <= baseline.TRAIN_SUBJECTS
                or validation_subjects != set(VAL_SUBJECTS) or not fitting_subjects.isdisjoint(validation_subjects)):
            raise ValueError("checkpoint lacks disjoint official fitting/fixed-validation subjects")
        gate = gate_decision(manifest["gate_criteria"],
            {"metrics": saved["validation_metrics"], "per_subject": saved["validation_by_subject"]}, scope="full")
        selection_directory = Path(args.checkpoint).parent
        selection_path = selection_directory / "frozen_selection.json"
        if not selection_path.is_file():
            raise ValueError("test scoring requires the completed frozen selection beside the checkpoint")
        selection = read(selection_path)
        if (not gate["passed"] or selection.get("format") != FORMAT or selection.get("scope") != "full"
                or selection.get("gate_passed") is not True or selection.get("gate") != gate
                or selection.get("source_split") != "validation" or selection.get("test_metrics_consulted") is not False
                or selection.get("checkpoint_sha256") != checkpoint_sha
                or selection.get("thresholds_sha256") != _hash(selection_directory / "thresholds.json", deadline)
                or saved["threshold_config"] != read(selection_directory / "thresholds.json")):
            raise ValueError("checkpoint/thresholds/validation gate disagree with the completed frozen selection")
        normalization = saved["normalization"]
        if (normalization["source_split"] != "fit" or normalization["clip"] != 8.
                or normalization.get("std_floor") != 1e-4):
            raise ValueError("checkpoint normalization is not the frozen fit-only contract")
        if _hash(args.annotations, deadline) != manifest["annotation_sha256"]:
            raise ValueError("test annotations differ from trained source")
        data = load_data(args, deadline)
        subjects = set(data["subjects"].tolist())
        train_subjects = set(manifest["train"]["subjects"]) | set(manifest["validation"]["subjects"])
        if not subjects <= baseline.TEST_SUBJECTS or not subjects.isdisjoint(train_subjects):
            raise ValueError("test subjects overlap fitting/validation or are not official testing subjects")
        if len(data["starts"]) != FULL_COUNTS["test_windows"]:
            raise ValueError("test retained row count is incomplete")
        if data["feature_contract"] != manifest["feature_contract"]:
            raise ValueError("test encoder/preprocessing/extraction contract changed")
        if data["provenance"]["visual_context"] != manifest["source_provenance"]["visual_context"]:
            raise ValueError("test object detector contract changed")
        if manifest["control_contract"]["pose"] != POSE_CONTRACT_VERSION:
            raise ValueError("test pose contract changed")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(args.cpu_threads)
        torch.use_deterministic_algorithms(True)
        features = standardized_tensor(data["features"], normalization["mean"], normalization["std"],
                                        args.device, args.chunk_size, deadline)
        model = RGBHead(saved["config"]["input_dim"], saved["config"]["hidden"], saved["config"]["dropout"])
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(args.device)
        indices = np.arange(len(data["starts"]))
        logits = predict(model, features, indices, args.batch_size, deadline)
        metrics = temporal.assess(data, indices, logits, saved["threshold_config"]["thresholds"])
        per_subject = temporal.assess_by_subject(data, indices, logits, saved["threshold_config"]["thresholds"])
        deadline_check(deadline)
        if _hash(args.checkpoint, deadline) != checkpoint_sha:
            raise RuntimeError("checkpoint changed during frozen evaluation")
        report = {"format": FORMAT, "checkpoint_sha256": checkpoint_sha, "thresholds_refit": False,
            "checkpoint_selection_on_test": False, "previous_test_inspected": True,
            "full_eligible_window_coverage_verified": True, "split": baseline.split_report(data, indices),
            "metrics": metrics, "per_subject": per_subject, "feature_contract": data["feature_contract"],
            "feature_inventory_sha256": data["feature_inventory_sha256"], "source_provenance": data["provenance"],
            "prediction_precision": "torch.float32", "serving_model_replaced": False}
        temporal.save_predictions(out/"test_logits.npz", logits, data, indices)
        deadline_check(deadline)
        baseline.write_json(out/"test_metrics.json", report)
        baseline.write_json(out/"training_status.json", {"status": "complete", "test_evaluated": True})
        return report
    except TimeoutError:
        result = {"status": "incomplete_budget", "test_evaluated": False}
        baseline.write_json(out/"training_status.json", result)
        return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--scope", choices=("pilot", "full"), required=True)
    fit.add_argument("--pilot-video-ids")
    fit.add_argument("--gate-config", required=True)
    fit.add_argument("--epochs", type=int, default=80)
    fit.add_argument("--min-epochs", type=int, default=10)
    fit.add_argument("--patience", type=int, default=12)
    fit.add_argument("--lr-patience", type=int, default=5)
    fit.add_argument("--lr", type=float, default=.001)
    fit.add_argument("--weight-decay", type=float, default=.001)
    fit.add_argument("--hidden", type=int, default=256)
    fit.add_argument("--dropout", type=float, default=.3)
    fit.add_argument("--threshold-min-support", type=int, default=5)
    test = commands.add_parser("evaluate-test")
    test.add_argument("--checkpoint", required=True)
    for command in (fit, test):
        command.add_argument("--features-dir", required=True)
        command.add_argument("--shards", nargs="+", required=True)
        command.add_argument("--annotations", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--stop-after-seconds", type=float)
        command.add_argument("--batch-size", type=int, default=2048)
        command.add_argument("--chunk-size", type=int, default=512)
        command.add_argument("--cpu-threads", type=int, default=4)
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments._program_started = PROGRAM_STARTED
    print(json.dumps(train(arguments) if arguments.command == "train" else evaluate_test(arguments),
                     indent=2, allow_nan=False, default=str))
