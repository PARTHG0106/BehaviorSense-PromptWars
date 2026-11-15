"""Train an isolated temporal multilabel model in one finite, non-resumable run.

Only official training subjects enter fitting and validation. The separate test
command uses a converged checkpoint and its frozen validation thresholds. This
experiment reuses a previously inspected test benchmark; it is exploratory.
"""

from __future__ import annotations

import argparse
import copy
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_adl_multilabel as baseline
from behaviorsense.data.multilabel_skeleton import load_multilabel_shards
from behaviorsense.data.temporal_context import build_context_neighbors
from behaviorsense.models.multilabel_losses import masked_asymmetric_loss
from behaviorsense.models.temporal_fusion import TEMPORAL_FORMAT, TemporalMultilabelFusion
from behaviorsense.eval.multilabel import average_precision

VALIDATION_SUBJECTS = ["p04", "p15"]
CONTEXT_STRIDE = 15
SOURCE_PATHS = (
    "scripts/train_multilabel_temporal.py", "scripts/train_adl_multilabel.py", "scripts/train_adl.py",
    "src/behaviorsense/models/temporal_fusion.py", "src/behaviorsense/models/multilabel_losses.py",
    "src/behaviorsense/models/stgcnpp.py", "src/behaviorsense/models/visual_context.py",
    "src/behaviorsense/data/temporal_context.py", "src/behaviorsense/data/multilabel_skeleton.py",
    "src/behaviorsense/data/skeleton_dataset.py", "src/behaviorsense/data/toyota.py",
    "src/behaviorsense/eval/multilabel.py",
)


class _BudgetExpired(RuntimeError):
    pass


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise _BudgetExpired("the finite session allocation has ended")


def save_predictions(path, logits, data, indices):
    """Replace predictions atomically, preserving the exact inference row order."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, logits=np.asarray(logits, dtype=np.float32),
            **{key: data[key][indices] for key in
               ("targets", "target_mask", "subjects", "video_ids", "starts")})
    temporary.replace(path)


class TemporalWindows(baseline.MultilabelWindows):
    """The original center-window supervision with exact-time detector neighbors."""

    def __init__(self, data, indices, *, training=False, seed=0, context_radius=6,
                 neighbor_dropout=.1, neighbors=None):
        super().__init__(data, indices, training=training, seed=seed)
        if not math.isfinite(neighbor_dropout) or not 0 <= neighbor_dropout < 1:
            raise ValueError("neighbor_dropout must be finite and in [0,1)")
        if "visual_context" not in data:
            raise ValueError("temporal windows require real detector context")
        self.context_radius, self.neighbor_dropout = context_radius, neighbor_dropout
        self.neighbors = (build_context_neighbors(data["video_ids"], data["subjects"], data["starts"],
                          radius=context_radius, stride=CONTEXT_STRIDE) if neighbors is None else neighbors)
        if (self.neighbors.shape != (len(data["starts"]), 2 * context_radius + 1)
                or self.neighbors.dtype != np.int64):
            raise ValueError("neighbors must be the int64 exact-time lookup for every input row")

    def __getitem__(self, item):
        # The base class supplies the identical normalized/augmented JOINT input
        # and center targets. Its one-frame context is replaced, never its labels.
        center = super().__getitem__(item)
        index = int(self.indices[item])
        neighbors = self.neighbors[index]
        context = np.zeros((len(neighbors), self.data["visual_context"].shape[-1]), dtype=np.float32)
        observed = neighbors >= 0
        context[observed] = self.data["visual_context"][neighbors[observed]]
        if self.training:
            context = baseline.augment_context(context, self.cfg,
                np.random.default_rng((self.seed, self.epoch, index)))
            drop = np.random.default_rng((self.seed, self.epoch, index, 17771)).random(len(neighbors))
            drop = drop < self.neighbor_dropout
            drop[self.context_radius] = False
            context[drop] = 0
        return (*center[:3], torch.from_numpy(context))


def load_data(paths, args):
    return load_multilabel_shards(paths, args.annotations, sample_fps=args.sample_fps,
        annotation_fps=args.annotation_fps, min_overlap=args.min_overlap,
        cache_dir=args.cache_dir or str(Path(args.out) / "data_cache"),
        require_context=True, require_full_coverage=args.require_full_coverage,
        expected_subset="training" if args.command == "train" else "testing")


def build_model(config):
    return TemporalMultilabelFusion(n_classes=len(baseline.COARSE_V11),
        base_channels=config.get("base_channels", 64), context_hidden=config.get("context_hidden", 128),
        context_radius=config.get("context_radius", 6), dropout=config.get("dropout", .2))


def forward_batch(model, batch, device):
    # TemporalMultilabelFusion derives its bone branch internally, exactly once.
    return model(batch[0].to(device, dtype=torch.float32, non_blocking=True),
                 batch[3].to(device, dtype=torch.float32, non_blocking=True))


@torch.no_grad()
def predict(model, loader, device, deadline=None):
    """Float32 forward passes and CPU logits, without mixed-precision scoring."""
    model.float().eval()
    predictions = []
    with torch.autocast(torch.device(device).type, enabled=False):
        for batch in loader:
            _check_deadline(deadline)
            values = forward_batch(model, batch, device).float().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("non-finite validation/test logits")
            predictions.append(values)
    _check_deadline(deadline)
    if not predictions:
        raise ValueError("prediction requires at least one window")
    return np.concatenate(predictions)


def validation_fingerprint(data, indices):
    """Model/loss-independent identity of validation rows, labels and masks."""
    indices = np.asarray(indices)
    order = np.lexsort((data["starts"][indices], data["video_ids"][indices]))
    rows = indices[order]
    identity = {key: data[key][rows].tolist() for key in ("video_ids", "subjects", "starts")}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    for key, dtype in (("targets", "<f4"), ("target_mask", "u1")):
        values = np.ascontiguousarray(data[key][rows], dtype=dtype)
        digest.update(json.dumps([key, list(values.shape), dtype]).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def assess(data, indices, logits, thresholds=None):
    """Preserve baseline metrics and add a threshold-independent pair ranking."""
    result = baseline.assess(data, indices, logits, thresholds)
    pair = list(baseline.PAIR)
    known = data["target_mask"][indices][:, pair].all(axis=1)
    truth = data["targets"][indices][:, pair].astype(bool).all(axis=1)[known]
    scores = baseline.probabilities(logits)[:, pair].min(axis=1)[known]
    result["tv_drinking_colabeled_windows"]["average_precision"] = (
        average_precision(scores, truth) if truth.any() and (~truth).any() else None)
    result["tv_drinking_colabeled_windows"]["ranking_score"] = "minimum_of_two_marginal_sigmoid_scores"
    return result


def assess_by_subject(data, indices, logits, thresholds):
    indices = np.asarray(indices)
    subjects = data["subjects"][indices]
    return {str(subject): assess(data, indices[subjects == subject], logits[subjects == subject], thresholds)
            for subject in sorted(set(subjects.tolist()))}


def _validate_training_args(args):
    if args.epochs < 1 or not 1 <= args.min_epochs <= args.epochs or args.batch_size < 1:
        raise ValueError("invalid epoch floor, epoch budget or batch size")
    if (args.patience < 1 or args.lr_patience < 1 or args.convergence_lr_reductions < 0
            or not 0 <= args.warmup_epochs < args.epochs):
        raise ValueError("invalid convergence, scheduler or warmup settings")
    if (not all(math.isfinite(value) for value in
                (args.lr, args.min_lr, args.lr_factor, args.weight_decay, args.ema_decay,
                 args.neighbor_dropout, args.pos_weight_max, args.dropout))
            or not 0 < args.min_lr <= args.lr or not 0 < args.lr_factor < 1
            or args.weight_decay < 0 or not 0 <= args.ema_decay < 1
            or not 0 <= args.neighbor_dropout < 1 or args.pos_weight_max < 1
            or not 0 <= args.dropout < 1):
        raise ValueError("invalid finite optimization/regularization settings")
    if args.workers < 0 or args.cpu_threads < 1 or args.threshold_min_support < 1 or args.seed < 0:
        raise ValueError("invalid workers, CPU threads, threshold support or seed")
    if args.stop_after_seconds is not None and (not math.isfinite(args.stop_after_seconds)
                                               or args.stop_after_seconds <= 0):
        raise ValueError("stop-after-seconds must be positive and finite")
    if args.require_full_coverage and sorted(args.val_subjects) != VALIDATION_SUBJECTS:
        raise ValueError("the full experiment fixes validation subjects to p04 and p15")


def _configure_execution(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(args.cpu_threads)
    baseline.set_seed(args.seed)
    cuda = torch.device(args.device).type == "cuda"
    use_amp = cuda and args.precision != "float32"
    if args.precision == "bfloat16" and not (cuda and torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 requires a supporting CUDA device")
    dtype = (torch.bfloat16 if use_amp and args.precision != "float16"
             and torch.cuda.is_bf16_supported() else torch.float16)
    execution = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
        "training_precision": str(dtype) if use_amp else "torch.float32",
        "prediction_precision": "torch.float32", "deterministic": args.deterministic,
        "device_type": torch.device(args.device).type,
        "gpu_capability": list(torch.cuda.get_device_capability(args.device)) if cuda else None}
    return use_amp, dtype, execution


def train(args):
    began = getattr(args, "_program_started", None) or time.monotonic()
    deadline = began + args.stop_after_seconds if args.stop_after_seconds is not None else None
    _validate_training_args(args)
    out = Path(args.out)
    if any((out / name).exists() for name in ("manifest.json", "last_multilabel.pt", "best_multilabel.pt")):
        raise ValueError("output already contains a run; this experiment never resumes")
    out.mkdir(parents=True, exist_ok=True)
    use_amp, amp_dtype, execution = _configure_execution(args)
    paths = baseline.expand_shards(args.train_shards)
    data = load_data(paths, args)
    train_idx, val_idx = baseline.subject_split(data["subjects"], val_subjects=args.val_subjects)
    known = data["target_mask"].any(axis=1)
    train_idx, val_idx = train_idx[known[train_idx]], val_idx[known[val_idx]]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("fitting and validation both need windows with known labels")
    files = [{"name": Path(path).name, "sha256": baseline.file_sha256(path)} for path in paths]
    fingerprint = validation_fingerprint(data, val_idx)
    manifest = {"format": TEMPORAL_FORMAT, "task_type": "multilabel", "activation": "sigmoid",
        "model": "temporal-fusion", "loss": args.loss, "seed": args.seed, "stream": "joint+bone",
        "class_names": list(baseline.COARSE_V11), "n_frames": 30, "sample_fps": args.sample_fps,
        "annotation_fps": args.annotation_fps, "min_overlap": args.min_overlap,
        "train": baseline.split_report(data, train_idx), "validation": baseline.split_report(data, val_idx),
        "provenance": data["provenance"], "files": files,
        "annotation_sha256": baseline.file_sha256(args.annotations),
        "source_files": [{"path": path, "sha256": baseline.file_sha256(ROOT / path)} for path in SOURCE_PATHS],
        "visual_context": data["provenance"].get("visual_context"),
        "temporal_context": {"radius": args.context_radius, "stride": CONTEXT_STRIDE,
            "sample_fps": args.sample_fps, "lookahead_seconds": args.context_radius * CONTEXT_STRIDE / args.sample_fps,
            "lookahead": args.context_radius * CONTEXT_STRIDE / args.sample_fps,
            "causal": args.context_radius == 0,
            "missing_policy": "exact_same_video_timestamp_or_masked_zero; never adjacent-row substitution",
            "neighbor_dropout": args.neighbor_dropout, "center_neighbor_dropped": False,
            "scope": "retained_annotation_eligible_pose_visible_windows"},
        "initialization": {"mode": "scratch", "pretrained_checkpoint": None},
        "test_loaded_during_training": False, "test_metrics_consulted": False,
        "previous_test_inspected": True, "excluded_unknown_windows": int((~known).sum()),
        "full_eligible_window_coverage_verified": data["provenance"].get("full_coverage_verified", False),
        "full_recording_temporal_coverage_verified": False,
        "evaluation_scope": "retained_sampled_windows; offline_with_lookahead; not full-video/population metrics",
        "selection_metric": "validation_macro_ap", "validation_comparison_fingerprint": fingerprint,
        "stopping_policy": {"minimum_epochs": args.min_epochs, "epoch_budget": args.epochs,
            "patience": args.patience, "required_lr_reductions": args.convergence_lr_reductions,
            "lr_settle_epochs": args.lr_patience, "source_split": "validation"},
        "automatic_continuation": False, "resume_supported": False, "execution": execution,
        "deployment_status": "experimental_not_serving_compatible"}
    # Retain the familiar spelling for downstream audit readers as well.
    manifest["initialisation"] = manifest["initialization"]
    if args.require_full_coverage:
        if manifest["full_eligible_window_coverage_verified"] is not True:
            raise ValueError("full training requires verified eligible-window coverage")
        baseline.require_class_support(manifest, args.threshold_min_support)
    positives = np.asarray(manifest["train"]["positive"])
    negatives = np.asarray(manifest["train"]["negative"])
    weights = np.clip(np.sqrt(negatives / np.maximum(positives, 1)), 1., args.pos_weight_max)
    weights[positives == 0] = 1.
    pos_weight = torch.as_tensor(weights, dtype=torch.float32, device=args.device) if args.loss == "bce" else None
    manifest["pos_weight"] = weights.tolist() if args.loss == "bce" else None
    manifest["loss_configuration"] = ({"name": "masked_bce", "weights_source": "fit_subjects_only",
        "positive_weight_rule": "clip(sqrt(negative/positive),1,max)", "maximum": args.pos_weight_max}
        if args.loss == "bce" else {"name": "masked_asymmetric_loss", "gamma_neg": 4.,
                                  "gamma_pos": 1., "clip": .05, "class_weights": None})
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    config.update(model="temporal-fusion", stream="joint+bone", n_classes=len(baseline.COARSE_V11), n_frames=30)
    history, best_state, best_logits = [], None, None
    best, best_epoch, step, lr_reductions, last_lr_reduction = -1., -1, 0, 0, -1

    def convergence():
        return baseline.convergence_report(len(history), best_epoch, min_epochs=args.min_epochs,
            epochs=args.epochs, patience=args.patience, lr_reductions=lr_reductions,
            required_lr_reductions=args.convergence_lr_reductions,
            last_lr_reduction_epoch=last_lr_reduction, lr_settle_epochs=args.lr_patience)

    def publish(status, *, exported=False, **details):
        manifest["convergence"] = convergence()
        manifest["checkpoint_selection"] = baseline.selection_report(best, best_epoch, args.seed, fingerprint)
        manifest["elapsed_seconds"] = time.monotonic() - began
        baseline.write_json(out / "manifest.json", manifest)
        baseline.write_json(out / "history.json", history)
        state = {"status": status, "inference_checkpoint_exported": exported,
            "convergence": manifest["convergence"], "completed_epochs": len(history),
            "automatic_continuation": False, "resume_supported": False,
            "elapsed_seconds": manifest["elapsed_seconds"], **details}
        baseline.write_json(out / "training_status.json", state)
        return state

    publish("running")
    print(f"Training {len(train_idx)} / validation {len(val_idx)} windows; loss={args.loss}; scratch", flush=True)
    print(f"Subject split: train={manifest['train']['subjects']}, validation={manifest['validation']['subjects']}", flush=True)
    try:
        _check_deadline(deadline)
        neighbors = build_context_neighbors(data["video_ids"], data["subjects"], data["starts"],
                                            radius=args.context_radius, stride=CONTEXT_STRIDE)
        train_ds = TemporalWindows(data, train_idx, training=True, seed=args.seed,
            context_radius=args.context_radius, neighbor_dropout=args.neighbor_dropout, neighbors=neighbors)
        val_ds = TemporalWindows(data, val_idx, context_radius=args.context_radius, neighbors=neighbors)
        loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=torch.device(args.device).type == "cuda")
        # Workers are recreated each epoch so each sees the current augmentation epoch.
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
        model = build_model(config).to(args.device)
        ema = baseline.EMA(model, decay=args.ema_decay)
        eval_model = copy.deepcopy(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor,
            patience=args.lr_patience, cooldown=args.lr_patience, min_lr=args.min_lr,
            threshold=1e-8, threshold_mode="abs")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        for epoch in range(args.epochs):
            _check_deadline(deadline)
            train_ds.epoch = epoch
            model.train()
            epoch_began, total_loss, seen = time.monotonic(), 0., 0
            for batch in train_loader:
                _check_deadline(deadline)
                if epoch < args.warmup_epochs:
                    warmup_lr = args.lr * (step + 1) / max(1, args.warmup_epochs * len(train_loader))
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr
                optimizer.zero_grad(set_to_none=True)
                targets = batch[1].to(args.device, non_blocking=True)
                mask = batch[2].to(args.device, non_blocking=True)
                with torch.autocast(torch.device(args.device).type, dtype=amp_dtype, enabled=use_amp):
                    logits = forward_batch(model, batch, args.device)
                    loss = (baseline.masked_bce(logits, targets, mask, pos_weight) if args.loss == "bce"
                            else masked_asymmetric_loss(logits, targets, mask))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(model.parameters(), 5.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("non-finite gradient norm")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    ema.update(model)
                total_loss += float(loss.detach()) * len(targets)
                seen += len(targets)
                step += 1
                _check_deadline(deadline)
            if seen != len(train_idx):
                raise RuntimeError("an epoch did not visit every fitting window exactly once")
            ema.copy_to(eval_model)
            logits = predict(eval_model, val_loader, args.device, deadline=deadline)
            metrics = assess(data, val_idx, logits)
            score = metrics["macro_ap"]
            if score is None or not math.isfinite(score):
                raise ValueError("validation has no finite class-supported macro AP")
            _check_deadline(deadline)
            if score > best + 1e-8:
                best, best_epoch = float(score), epoch
                best_state = {key: value.detach().cpu().clone() for key, value in eval_model.state_dict().items()}
                best_logits = logits.copy()
            if epoch >= args.warmup_epochs:
                previous_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(score)
                if optimizer.param_groups[0]["lr"] < previous_lr:
                    lr_reductions += 1
                    last_lr_reduction = epoch
            history.append({"epoch": epoch, "train_loss": total_loss / seen,
                "seconds": time.monotonic() - epoch_began, "fit_windows_seen": seen,
                "validation": metrics, "learning_rate": optimizer.param_groups[0]["lr"],
                "lr_reductions": lr_reductions})
            publish("running")
            snapshot = {"format": TEMPORAL_FORMAT, "kind": "training_snapshot", "resume_supported": False,
                "args": config, "manifest": manifest, "epoch": epoch, "step": step,
                "train_state": model.state_dict(), "ema_state": ema.shadow, "ema_step": ema.step,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "grad_scaler": scaler.state_dict(), "rng": baseline.rng_state(),
                "best": best, "best_epoch": best_epoch, "best_state": best_state,
                "lr_reductions": lr_reductions, "last_lr_reduction_epoch": last_lr_reduction,
                "history": history}
            baseline.save_checkpoint(out / "last_multilabel.pt", snapshot)
            print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / seen:.5f} "
                  f"val_mAP={score:.5f} best={best:.5f} seconds={history[-1]['seconds']:.1f}", flush=True)
            if convergence()["converged"]:
                break
        if not convergence()["converged"]:
            return publish("incomplete_epoch_limit")
        _check_deadline(deadline)
    except _BudgetExpired:
        return publish("incomplete_budget", interrupted_epoch_counted=False)
    except Exception as exc:
        publish("failed", error=f"{type(exc).__name__}: {exc}")
        raise

    # These are the float32 predictions already made with the selected EMA state,
    # not a newly tuned model. Retaining them also avoids a second GPU inference.
    threshold_config = baseline.fit_thresholds(baseline.probabilities(best_logits),
        data["targets"][val_idx], data["target_mask"][val_idx],
        min_positive=args.threshold_min_support, min_negative=args.threshold_min_support,
        train_positive=manifest["train"]["positive"])
    metrics = assess(data, val_idx, best_logits, threshold_config["thresholds"])
    if not math.isclose(metrics["macro_ap"], best, abs_tol=1e-8):
        raise RuntimeError("frozen selected predictions do not reproduce the selection score")
    artifact = {"format": TEMPORAL_FORMAT, "kind": "inference", "args": config, "config": config,
        "state_dict": best_state, "manifest": manifest, "threshold_config": threshold_config,
        "validation_metrics": metrics}
    baseline.write_json(out / "thresholds.json", threshold_config)
    baseline.write_json(out / "validation_metrics.json", metrics)
    baseline.write_json(out / "validation_by_subject.json",
        assess_by_subject(data, val_idx, best_logits, threshold_config["thresholds"]))
    save_predictions(out / "validation_logits.npz", best_logits, data, val_idx)
    baseline.save_checkpoint(out / "best_multilabel.pt", artifact)
    publish("complete", exported=True)
    return artifact


def evaluate_test(args):
    checkpoint_hash = baseline.file_sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != TEMPORAL_FORMAT or saved.get("kind") != "inference":
        raise ValueError("test evaluation requires a temporal inference checkpoint")
    manifest = saved["manifest"]
    if not manifest.get("convergence", {}).get("converged"):
        raise ValueError("test evaluation requires a converged training candidate")
    if (manifest.get("test_loaded_during_training") is not False
            or saved["threshold_config"].get("source_split") != "validation"
            or manifest.get("checkpoint_selection", {}).get("source_split") != "validation"):
        raise ValueError("checkpoint lacks validation-only selection and threshold provenance")
    if baseline.file_sha256(args.annotations) != manifest["annotation_sha256"]:
        raise ValueError("annotations changed from the trained checkpoint")
    training_subjects = set(manifest["train"]["subjects"]) | set(manifest["validation"]["subjects"])
    if not training_subjects <= baseline.TRAIN_SUBJECTS:
        raise ValueError("checkpoint fitting/validation contains non-training subjects")
    loading = copy.copy(args)
    for key in ("sample_fps", "annotation_fps", "min_overlap"):
        setattr(loading, key, saved["args"][key])
    paths = baseline.expand_shards(args.test_shards)
    data = load_data(paths, loading)
    actual_subjects = set(data["subjects"].tolist())
    if not actual_subjects or not actual_subjects <= baseline.TEST_SUBJECTS or actual_subjects & training_subjects:
        raise ValueError("test input must contain only disjoint official testing subjects")
    if args.require_full_coverage and (actual_subjects != baseline.TEST_SUBJECTS
            or data["provenance"].get("full_coverage_verified") is not True):
        raise ValueError("full testing requires all official test subjects and verified coverage")
    if data["provenance"].get("visual_context") != manifest.get("visual_context"):
        raise ValueError("test detector context contract changed from training")
    radius = saved["args"]["context_radius"]
    if manifest["temporal_context"]["radius"] != radius or manifest["temporal_context"]["stride"] != CONTEXT_STRIDE:
        raise ValueError("saved temporal sampling contract is inconsistent")
    torch.set_num_threads(args.cpu_threads)
    torch.use_deterministic_algorithms(saved["args"].get("deterministic", True))
    torch.backends.cudnn.deterministic = saved["args"].get("deterministic", True)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model = build_model(saved["args"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(args.device)
    indices = np.arange(len(data["targets"]))
    dataset = TemporalWindows(data, indices, context_radius=radius, neighbor_dropout=0.)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=torch.device(args.device).type == "cuda")
    logits = predict(model, loader, args.device)
    metrics = assess(data, indices, logits, saved["threshold_config"]["thresholds"])
    report = {"format": TEMPORAL_FORMAT, "model": "temporal-fusion", "checkpoint_sha256": checkpoint_hash,
        "thresholds_refit": False, "checkpoint_selection_on_test": False, "previous_test_inspected": True,
        "test_benchmark_previously_inspected": True, "exploratory_test_benchmark": True,
        "split": baseline.split_report(data, indices), "metrics": metrics,
        "per_subject": assess_by_subject(data, indices, logits, saved["threshold_config"]["thresholds"]),
        "training_convergence": manifest["convergence"], "checkpoint_selection": manifest["checkpoint_selection"],
        "temporal_context": manifest["temporal_context"], "evaluation_scope": manifest["evaluation_scope"],
        "full_recording_temporal_coverage_verified": False,
        "full_eligible_window_coverage_verified": data["provenance"].get("full_coverage_verified", False),
        "source_provenance": data["provenance"], "prediction_precision": "torch.float32",
        "serving_model_replaced": False}
    if baseline.file_sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during frozen evaluation")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_predictions(out / "test_logits.npz", logits, data, indices)
    baseline.write_json(out / "test_metrics.json", report)
    print(json.dumps(report["metrics"], indent=2, allow_nan=False), flush=True)
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--train-shards", "--shards", nargs="+", required=True)
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--loss", choices=("bce", "asl"), default="bce")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--val-subjects", nargs="+", default=VALIDATION_SUBJECTS)
    fit.add_argument("--epochs", type=int, default=80)
    fit.add_argument("--min-epochs", type=int, default=40)
    fit.add_argument("--patience", type=int, default=20)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--lr", type=float, default=.0003)
    fit.add_argument("--weight-decay", type=float, default=.0001)
    fit.add_argument("--warmup-epochs", type=int, default=2)
    fit.add_argument("--lr-patience", type=int, default=5)
    fit.add_argument("--lr-factor", type=float, default=.5)
    fit.add_argument("--min-lr", type=float, default=.000001)
    fit.add_argument("--convergence-lr-reductions", type=int, default=2)
    fit.add_argument("--ema-decay", type=float, default=.99)
    fit.add_argument("--pos-weight-max", type=float, default=10.)
    fit.add_argument("--threshold-min-support", type=int, default=5)
    fit.add_argument("--base-channels", type=int, default=64)
    fit.add_argument("--context-hidden", type=int, default=128)
    fit.add_argument("--context-radius", type=int, default=6)
    fit.add_argument("--neighbor-dropout", type=float, default=.1)
    fit.add_argument("--dropout", type=float, default=.2)
    fit.add_argument("--sample-fps", type=float, default=20.)
    fit.add_argument("--annotation-fps", type=float, default=25.)
    fit.add_argument("--min-overlap", type=float, default=.5)
    fit.add_argument("--precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    fit.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    fit.add_argument("--stop-after-seconds", type=float)
    test = commands.add_parser("evaluate-test")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--test-shards", nargs="+", required=True)
    test.add_argument("--annotations", required=True)
    test.add_argument("--out", required=True)
    test.add_argument("--batch-size", type=int, default=512)
    for command in (fit, test):
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--cpu-threads", type=int, default=4)
        command.add_argument("--cache-dir")
        command.add_argument("--require-full-coverage", action="store_true")
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments._program_started = PROGRAM_STARTED
    train(arguments) if arguments.command == "train" else evaluate_test(arguments)
