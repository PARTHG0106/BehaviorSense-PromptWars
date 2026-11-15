"""Train an isolated sigmoid ST-GCN++ head on real overlapping Toyota annotations.

Training accepts official CS training subjects only. Test inference is a separate
command, with frozen model weights and validation-fitted thresholds. These artifacts
intentionally use a different schema from the categorical serving checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from behaviorsense.data.multilabel_skeleton import load_multilabel_shards
from behaviorsense.data.skeleton_dataset import AugmentConfig, augment, normalise
from behaviorsense.data.toyota import (COARSE_V11, CS_TEST_SUBJECTS, CS_TRAIN_SUBJECTS,
                                     TSU_TO_COARSE, supervised_coarse)
from behaviorsense.eval.multilabel import fit_thresholds, masked_metrics
from behaviorsense.models.stgcnpp import STGCNpp, make_stream
from train_adl import EMA, cosine_lr, load_rng_state, rng_state, set_seed

FORMAT = "behaviorsense.stgcnpp.multilabel.v1"
FUSION_FORMAT = "behaviorsense.stgcnpp.context-multilabel.v1"
FORMATS = (FORMAT, FUSION_FORMAT)
TRAIN_SUBJECTS = {f"p{s:02d}" for s in CS_TRAIN_SUBJECTS}
TEST_SUBJECTS = {f"p{s:02d}" for s in CS_TEST_SUBJECTS}
PAIR = (COARSE_V11.index("watching_tv"), COARSE_V11.index("drinking"))


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, value) -> None:
    # A session ending during serialization must leave the previous epoch resumable.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expand_shards(patterns) -> list[str]:
    paths = sorted({str(Path(p).resolve()) for pattern in patterns for p in glob.glob(pattern)},
                   key=lambda path: (Path(path).name, path))
    if not paths:
        raise ValueError("no shard files matched")
    return paths


def subject_split(subjects, *, seed=0, val_subjects=None):
    subjects = np.asarray(subjects, dtype=str)
    available = set(subjects.tolist())
    if not available <= TRAIN_SUBJECTS:
        raise ValueError(f"training input contains non-training subjects: {sorted(available - TRAIN_SUBJECTS)}")
    if val_subjects is None:
        ordered = np.array(sorted(available))
        np.random.default_rng(seed).shuffle(ordered)
        val_subjects = set(ordered[:max(1, len(ordered) // 5)].tolist())
    else:
        val_subjects = set(val_subjects)
    if not val_subjects or not val_subjects < available:
        raise ValueError("validation must be a nonempty proper subset of available training subjects")
    is_val = np.isin(subjects, sorted(val_subjects))
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


class MultilabelWindows(Dataset):
    def __init__(self, data, indices, *, training=False, seed=0):
        self.data, self.indices = data, np.asarray(indices)
        self.training, self.seed, self.epoch = training, seed, 0
        self.cfg = AugmentConfig(enabled=training, temporal_scale=(1., 1.))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        # Never crop the time axis: doing so can remove the annotated overlap.
        x = normalise(self.data["skeletons"][index].astype(np.float32))
        rng_seed = (self.seed, self.epoch, index)
        x = augment(x, self.cfg, np.random.default_rng(rng_seed))
        x = np.ascontiguousarray(x.transpose(3, 0, 2, 1))
        result = (torch.from_numpy(x), torch.as_tensor(self.data["targets"][index]),
                  torch.as_tensor(self.data["target_mask"][index]))
        if "visual_context" in self.data:
            context = self.data["visual_context"][index].astype(np.float32, copy=True)
            if self.training:
                context = augment_context(context, self.cfg, np.random.default_rng(rng_seed))
            result += (torch.from_numpy(context),)
        return result

    def __getstate__(self):
        # numpy.memmap otherwise pickles its payload on Windows/spawn workers.
        state = self.__dict__.copy()
        state["data"] = {key: ({"_memmap_path": str(value.filename)} if isinstance(value, np.memmap)
                               else value) for key, value in self.data.items()}
        return state

    def __setstate__(self, state):
        state["data"] = {key: (np.load(value["_memmap_path"], mmap_mode="r")
                               if isinstance(value, dict) and "_memmap_path" in value else value)
                         for key, value in state["data"].items()}
        self.__dict__.update(state)


def augment_context(context, cfg, rng):
    """Apply the skeleton's same affine transform to relative object geometry."""
    features = context[..., 1:].reshape(*context.shape[:-1], 80, 5)
    geometry = features[..., 1:3]
    valid_geometry = (features[..., 0] > 0) & ~np.all(geometry == -1, axis=-1)
    if rng.random() < cfg.flip_prob:
        geometry[..., 0][valid_geometry] *= -1
    theta = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    c, s = np.cos(theta), np.sin(theta)
    geometry[valid_geometry] = geometry[valid_geometry] @ np.array([[c, -s], [s, c]], np.float32).T
    scale = rng.uniform(*cfg.scale_range)
    geometry[valid_geometry] *= scale
    for column in (3, 4):
        valid = (features[..., 0] > 0) & (features[..., column] >= 0)
        features[..., column][valid] *= scale
    return context


def build_model(config):
    kwargs = dict(n_classes=len(COARSE_V11), base_channels=config["base_channels"])
    if config.get("model", "skeleton") == "context-fusion":
        from behaviorsense.models.context_fusion import STGCNContextFusion
        return STGCNContextFusion(**kwargs, context_hidden=config.get("context_hidden", 128))
    return STGCNpp(**kwargs)


def forward_batch(model, batch, device):
    x = make_stream(batch[0].to(device, non_blocking=True), "bone")
    return model(x, batch[3].to(device, non_blocking=True)) if len(batch) == 4 else model(x)


def masked_bce(logits, targets, mask, pos_weight=None):
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("BCE logits, targets and known-label mask must have identical shapes")
    if not bool(mask.any()):
        raise ValueError("BCE batch contains no known labels")
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), pos_weight=pos_weight, reduction="none")
    return (loss * mask).sum() / mask.sum()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits = [forward_batch(model, batch, device).float().cpu().numpy() for batch in loader]
    return np.concatenate(logits)


def probabilities(logits):
    return torch.from_numpy(np.asarray(logits)).sigmoid().numpy()


def split_report(data, indices):
    target, mask = data["targets"][indices].astype(bool), data["target_mask"][indices]
    out = {"subjects": sorted(set(data["subjects"][indices].tolist())),
           "n_windows": len(indices),
           "n_videos": len(set(data["video_ids"][indices].tolist())),
           "positive": (target & mask).sum(0).tolist(),
           "negative": (~target & mask).sum(0).tolist(),
           "tv_drinking_colabeled_windows": int(target[:, list(PAIR)].all(1).sum())}
    if "tv_drinking_overlap_frames" in data:
        overlap = data["tv_drinking_overlap_frames"][indices]
        out["tv_drinking_actual_overlap_windows"] = int((overlap > 0).sum())
        out["tv_drinking_actual_overlap_sample_positions"] = int(overlap.sum())
    return out


def assess(data, indices, logits, thresholds=None):
    result = masked_metrics(probabilities(logits), data["targets"][indices],
                            data["target_mask"][indices], thresholds, pair=PAIR)
    if "tv_drinking_overlap_frames" in data:
        known_pair = data["target_mask"][indices][:, list(PAIR)].all(1)
        shared = (data["tv_drinking_overlap_frames"][indices] > 0) & known_pair
        scores = probabilities(logits)
        cutoffs = np.full(len(COARSE_V11), .5) if thresholds is None else np.array(thresholds)
        pair_predictions = (scores[:, list(PAIR)] > cutoffs[list(PAIR)]).all(1) & known_pair
        # This is diagnostic overlap detection at window resolution, not frame mAP.
        tp = int((shared & pair_predictions).sum())
        result["tv_drinking_actual_overlap_windows"] = {
            "known_windows": int(known_pair.sum()),
            "support": int(shared.sum()), "predicted": int(pair_predictions.sum()),
            "true_positive": tp,
            "precision": tp / max(1, int(pair_predictions.sum())),
            "recall": tp / int(shared.sum()) if shared.any() else None,
        }
    return result


def convergence_report(completed_epochs, best_epoch, *, min_epochs, epochs, patience,
                       lr_reductions=0, required_lr_reductions=0, last_lr_reduction_epoch=-1,
                       lr_settle_epochs=0):
    """Report validation plateau evidence separately from exhausting a compute budget."""
    stale = max(0, completed_epochs - 1 - best_epoch) if best_epoch >= 0 else 0
    plateau = (completed_epochs >= min_epochs and best_epoch >= 0
               and patience > 0 and stale >= patience
               and lr_reductions >= required_lr_reductions
               and completed_epochs - 1 - last_lr_reduction_epoch >= lr_settle_epochs)
    exhausted = completed_epochs >= epochs
    status = ("validation_plateau" if plateau else
              "epoch_budget_exhausted_without_validation_plateau" if exhausted else "in_progress")
    return {
        "status": status, "converged": bool(plateau), "budget_exhausted": bool(exhausted),
        "convergence_definition": "validation macro AP has not improved for patience epochs after min_epochs",
        "completed_epochs": completed_epochs, "minimum_epochs": min_epochs,
        "epoch_budget": epochs, "patience": patience, "epochs_since_improvement": stale,
        "best_epoch_zero_based": best_epoch if best_epoch >= 0 else None,
        "lr_reductions": lr_reductions, "required_lr_reductions": required_lr_reductions,
        "last_lr_reduction_epoch_zero_based": last_lr_reduction_epoch,
        "lr_settle_epochs": lr_settle_epochs,
        "stopping_metric_source": "validation", "test_metrics_consulted": False,
        "warning": ("The epoch budget ended before a validation plateau was established. "
                    "This run is not confirmed converged; review its validation curve before final selection."
                    if exhausted and not plateau else None),
    }


def selection_report(best, best_epoch, seed, fingerprint):
    return {
        "source_split": "validation", "metric": "macro_ap", "direction": "maximise",
        "best_epoch_zero_based": best_epoch, "best_score": best, "training_seed": seed,
        "threshold_source_split": "validation", "test_metrics_consulted": False,
        "candidate_scope": "within_run_epoch_selection_only",
        "validation_comparison_fingerprint": fingerprint,
    }


def initialise(model, args, val_subjects):
    """A source model is optional; unknown or exposed provenance falls back to scratch."""
    if not args.init:
        return {"mode": "scratch", "reason": "no_initial_checkpoint"}
    checkpoint_hash = file_sha256(args.init)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    provenance = None
    if args.init_provenance:
        provenance = json.loads(Path(args.init_provenance).read_text(encoding="utf-8"))
        if provenance.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("initial checkpoint provenance hash does not match checkpoint")
    elif ck.get("format") in FORMATS:
        provenance = {"fit_subjects": ck["manifest"]["train"]["subjects"],
                      "selection_subjects": ck["manifest"]["validation"]["subjects"]}
    record = {"path": str(args.init), "sha256": checkpoint_hash, "mode": "scratch"}
    if not provenance or not provenance.get("fit_subjects"):
        record["reason"] = "unknown_initial_checkpoint_subject_provenance"
        print("Initial checkpoint subject provenance is unknown; training from scratch.")
        return record
    seen = set(provenance["fit_subjects"]) | set(provenance.get("selection_subjects", []))
    if not seen <= TRAIN_SUBJECTS or seen & (set(val_subjects) | TEST_SUBJECTS):
        record["reason"] = "initial_checkpoint_exposed_heldout_or_unverified_subjects"
        record["source_subjects"] = sorted(seen)
        print("Initial checkpoint exposes held-out subjects; training from scratch.")
        return record
    saved_args = ck.get("args", {})
    if saved_args.get("stream") != "bone" or saved_args.get("n_classes") != len(COARSE_V11):
        raise ValueError("initial checkpoint must declare 22 classes and bone stream")
    if saved_args.get("sample_fps") != args.sample_fps:
        raise ValueError("initial checkpoint sampling rate does not match training data")
    state = ck.get("state_dict") if ck.get("format") in FORMATS else ck.get("ema", ck.get("model"))
    if state is None:
        raise ValueError("initial checkpoint has no complete model state")
    model.load_state_dict(state, strict=True)
    record.update(mode="pretrained", reason="verified_unexposed_subjects", provenance=provenance)
    return record


def load_data(paths, args):
    return load_multilabel_shards(paths, args.annotations, sample_fps=args.sample_fps,
        annotation_fps=args.annotation_fps, min_overlap=args.min_overlap,
        cache_dir=args.cache_dir or str(Path(args.out) / "data_cache"),
        require_context=args.model == "context-fusion" or args.require_context,
        require_full_coverage=args.require_full_coverage,
        expected_subset="training" if args.command == "train" else "testing")


def require_class_support(manifest, minimum):
    """Every supported activity must be trainable and calibratable on the held-out subjects."""
    missing = {}
    for split in ("train", "validation"):
        missing[split] = [COARSE_V11[c] for c in supervised_coarse(TSU_TO_COARSE)
                          if manifest[split]["positive"][c] < minimum
                          or manifest[split]["negative"][c] < minimum]
    if any(missing.values()):
        raise ValueError(f"all 15 supported activities need >= {minimum} positive and negative windows "
                         f"in fit/validation; insufficient support: {missing}")


def train(args):
    began = time.monotonic()
    if args.epochs < 1 or args.batch_size < 1 or args.threshold_min_support < 1:
        raise ValueError("epochs, batch size and threshold support must be positive")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be between 1 and --epochs")
    if args.patience < 0:
        raise ValueError("--patience must be nonnegative (0 disables early stopping)")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup epochs must be nonnegative and below the training budget")
    if (args.lr <= 0 or args.min_lr <= 0 or args.min_lr > args.lr or args.lr_patience < 1
            or not 0 < args.lr_factor < 1 or args.convergence_lr_reductions < 0):
        raise ValueError("invalid learning-rate scheduler settings")
    if args.workers < 0 or args.cpu_threads < 1 or args.context_hidden < 1:
        raise ValueError("invalid workers, CPU threads or context width")
    if args.stop_after_seconds is not None and args.stop_after_seconds <= 0:
        raise ValueError("--stop-after-seconds must be positive")
    if args.require_context and args.model != "context-fusion":
        raise ValueError("--require-context is for the context-fusion model")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    paths = expand_shards(args.train_shards)
    data = load_data(paths, args)
    train_idx, val_idx = subject_split(data["subjects"], seed=args.split_seed,
                                     val_subjects=args.val_subjects)
    known = data["target_mask"].any(1)
    train_idx, val_idx = train_idx[known[train_idx]], val_idx[known[val_idx]]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("training and validation both require windows with known labels")
    artifact_format = FUSION_FORMAT if args.model == "context-fusion" else FORMAT
    manifest = {"format": artifact_format, "task_type": "multilabel", "activation": "sigmoid",
                "model": args.model, "visual_context": data["provenance"].get("visual_context"),
                "stream": "bone", "class_names": list(COARSE_V11), "sample_fps": args.sample_fps,
                "annotation_fps": args.annotation_fps, "min_overlap": args.min_overlap,
                "n_frames": 30, "provenance": data["provenance"],
                "train": split_report(data, train_idx), "validation": split_report(data, val_idx),
                "test_loaded_during_training": False,
                "excluded_unknown_windows": int((~known).sum()),
                "evaluation_scope": "retained_sampled_windows; not full-video or population prevalence metrics",
                "full_recording_temporal_coverage_verified": False,
                "full_eligible_window_coverage_verified": data["provenance"].get("full_coverage_verified", False),
                "selection_metric": "validation_macro_ap", "temporal_crop": False,
                "split_seed": args.split_seed,
                "stopping_policy": {"minimum_epochs": args.min_epochs, "epoch_budget": args.epochs,
                                    "patience": args.patience, "source_split": "validation"},
                "deployment_status": "experimental_not_serving_compatible"}
    if args.require_full_coverage:
        if not manifest["full_eligible_window_coverage_verified"]:
            raise ValueError("full training requires verified coverage provenance")
        require_class_support(manifest, args.threshold_min_support)
    # Content hashes bind resume to the same labels and source windows, not just filenames.
    manifest["files"] = [{"name": Path(p).name, "sha256": file_sha256(p)} for p in paths]
    manifest["annotation_sha256"] = file_sha256(args.annotations)
    manifest["split_fingerprint"] = digest({k: manifest[k] for k in
        ("files", "annotation_sha256", "train", "validation", "sample_fps", "annotation_fps", "min_overlap")})
    train_ds = MultilabelWindows(data, train_idx, training=True, seed=args.seed)
    val_ds = MultilabelWindows(data, val_idx)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers,
                     pin_memory=str(args.device).startswith("cuda"))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    model = build_model(vars(args)).to(args.device)
    initial = initialise(model, args, manifest["validation"]["subjects"]) if not args.resume else None
    ema = EMA(model, decay=args.ema_decay)
    eval_model = copy.deepcopy(model)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, cooldown=args.lr_patience, min_lr=args.min_lr,
        threshold=1e-8, threshold_mode="abs") if args.scheduler == "plateau" else None)
    positives = np.array(manifest["train"]["positive"])
    negatives = np.array(manifest["train"]["negative"])
    weights = np.clip(np.sqrt(negatives / np.maximum(positives, 1)), 1., args.pos_weight_max)
    weights[positives == 0] = 1.
    pos_weight = torch.as_tensor(weights, dtype=torch.float32, device=args.device)
    manifest["pos_weight"] = weights.tolist()
    out = Path(args.out)
    if (out / "last_multilabel.pt").exists() and not args.resume:
        raise ValueError("output already has a run; resume it or choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(n_classes=len(COARSE_V11), stream="bone", n_frames=30)
    use_amp = str(args.device).startswith("cuda") and args.precision != "float32"
    if args.precision == "bfloat16" and not (use_amp and torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 training requires a CUDA device with bfloat16 support")
    amp_dtype = (torch.bfloat16 if use_amp and args.precision != "float16"
                 and torch.cuda.is_bf16_supported() else torch.float16)
    execution = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                 "precision": str(amp_dtype) if use_amp else "torch.float32",
                 "deterministic": args.deterministic,
                 "device_type": torch.device(args.device).type,
                 "gpu_capability": list(torch.cuda.get_device_capability(args.device)) if use_amp else None}
    manifest["execution"] = execution
    history, start, step, best, best_epoch, best_state = [], 0, 0, -1., -1, None
    invariant_keys = ("epochs", "batch_size", "lr", "warmup_epochs", "weight_decay", "seed",
                      "split_seed", "min_epochs", "base_channels", "ema_decay", "pos_weight_max",
                      "patience", "threshold_min_support", "workers", "cpu_threads", "model",
                      "context_hidden", "scheduler", "lr_patience", "lr_factor", "min_lr",
                      "convergence_lr_reductions", "precision", "deterministic", "require_full_coverage",
                      "require_convergence", "require_context")
    lr_reductions, last_lr_reduction_epoch = 0, -1
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != artifact_format or saved.get("kind") != "resume":
            raise ValueError("resume requires a multi-label last checkpoint")
        if saved["manifest"]["split_fingerprint"] != manifest["split_fingerprint"]:
            raise ValueError("resume data or subject split differs from the saved run")
        for key in invariant_keys:
            if saved["args"][key] != config[key]:
                raise ValueError(f"resume mismatch: {key}")
        if saved["manifest"].get("execution") != execution:
            raise ValueError("resume runtime/precision differs; exact continuation requires the original execution settings")
        model.load_state_dict(saved["train_state"], strict=True)
        ema.shadow = {k: v.to(args.device) for k, v in saved["ema_state"].items()}
        ema.step = saved["ema_step"]
        optimiser.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        lr_reductions = saved.get("lr_reductions", 0)
        last_lr_reduction_epoch = saved.get("last_lr_reduction_epoch", -1)
        start, step, best, best_epoch = saved["epoch"] + 1, saved["step"], saved["best"], saved["best_epoch"]
        best_state, history, initial = saved["best_state"], saved["history"], saved["initialisation"]
        load_rng_state(saved["rng"])
    manifest["initialisation"] = initial
    write_json(out / "manifest.json", manifest)
    print(f"Training {len(train_idx)} / validation {len(val_idx)} windows; source={initial['mode']}")
    print(f"Subject split: train={manifest['train']['subjects']}, validation={manifest['validation']['subjects']}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if args.resume and saved.get("grad_scaler"):
        scaler.load_state_dict(saved["grad_scaler"])
    steps_per_epoch = len(train_loader)
    def convergence(completed):
        return convergence_report(completed, best_epoch, min_epochs=args.min_epochs,
            epochs=args.epochs, patience=args.patience, lr_reductions=lr_reductions,
            required_lr_reductions=args.convergence_lr_reductions if scheduler is not None else 0,
            last_lr_reduction_epoch=last_lr_reduction_epoch,
            lr_settle_epochs=args.lr_patience if scheduler is not None else 0)
    for epoch in range(start, args.epochs):
        stopping = convergence(epoch)
        if stopping["converged"]:
            break
        train_ds.epoch = epoch
        model.train()
        t0, total_loss, seen = time.time(), 0., 0
        for batch in train_loader:
            y, mask = batch[1:3]
            if scheduler is None or epoch < args.warmup_epochs:
                lr = (cosine_lr(step, args.epochs * steps_per_epoch,
                               args.warmup_epochs * steps_per_epoch, args.lr) if scheduler is None
                      else args.lr * (step + 1) / max(1, args.warmup_epochs * steps_per_epoch))
                for group in optimiser.param_groups:
                    group["lr"] = lr
            optimiser.zero_grad(set_to_none=True)
            y, mask = y.to(args.device), mask.to(args.device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = masked_bce(forward_batch(model, batch, args.device), y, mask, pos_weight)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            previous_scale = scaler.get_scale()
            scaler.step(optimiser); scaler.update()
            if scaler.get_scale() >= previous_scale:
                ema.update(model)
            total_loss += float(loss.detach()) * len(y); seen += len(y); step += 1
        ema.copy_to(eval_model)
        logits = predict(eval_model, val_loader, args.device)
        metrics = assess(data, val_idx, logits)
        score = metrics["macro_ap"]
        if score is None:
            raise ValueError("validation has no class with both positives and negatives; cannot select a model")
        improved = score > best + 1e-8
        if improved:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        if scheduler is not None and epoch >= args.warmup_epochs:
            previous_lr = optimiser.param_groups[0]["lr"]
            scheduler.step(score)
            if optimiser.param_groups[0]["lr"] < previous_lr:
                lr_reductions += 1
                last_lr_reduction_epoch = epoch
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "seconds": time.time() - t0, "validation": metrics,
                        "learning_rate": optimiser.param_groups[0]["lr"], "lr_reductions": lr_reductions})
        manifest["convergence"] = convergence(epoch + 1)
        manifest["checkpoint_selection"] = selection_report(
            best, best_epoch, args.seed, manifest["split_fingerprint"])
        print(f"epoch {epoch + 1}/{args.epochs} loss={total_loss / seen:.4f} val_mAP={score:.4f} best={best:.4f}", flush=True)
        checkpoint = {"format": artifact_format, "kind": "resume", "args": config, "manifest": manifest,
                      "train_state": model.state_dict(), "ema_state": ema.shadow, "ema_step": ema.step,
                      "optimizer": optimiser.state_dict(), "grad_scaler": scaler.state_dict(),
                      "scheduler": scheduler.state_dict() if scheduler is not None else None,
                      "lr_reductions": lr_reductions, "last_lr_reduction_epoch": last_lr_reduction_epoch,
                      "epoch": epoch, "step": step, "best": best, "best_epoch": best_epoch,
                      "best_state": best_state, "rng": rng_state(), "history": history,
                      "initialisation": initial}
        save_checkpoint(out / "last_multilabel.pt", checkpoint)
        write_json(out / "history.json", history)
        write_json(out / "manifest.json", manifest)
        if (not manifest["convergence"]["converged"]
                and ((args.stop_after and epoch - start + 1 >= args.stop_after)
                     or (args.stop_after_seconds and time.monotonic() - began >= args.stop_after_seconds))):
            status = {"stopped_after": epoch + 1, "out": str(out), "status": "resumable_session_pause",
                      "convergence": manifest["convergence"], "inference_checkpoint_exported": False}
            write_json(out / "training_status.json", status)
            return status
    if best_state is None:
        raise RuntimeError("no completed epoch to export")
    manifest["convergence"] = convergence(len(history))
    if args.require_convergence and not manifest["convergence"]["converged"]:
        write_json(out / "manifest.json", manifest)
        status = {"status": "epoch_budget_exhausted_without_convergence", "out": str(out),
                  "convergence": manifest["convergence"], "inference_checkpoint_exported": False}
        write_json(out / "training_status.json", status)
        print(json.dumps(status), flush=True)
        return status
    eval_model.load_state_dict(best_state, strict=True)
    logits = predict(eval_model, val_loader, args.device)
    threshold_config = fit_thresholds(probabilities(logits), data["targets"][val_idx],
        data["target_mask"][val_idx], min_positive=args.threshold_min_support,
        min_negative=args.threshold_min_support, train_positive=positives)
    metrics = assess(data, val_idx, logits, threshold_config["thresholds"])
    manifest["selected_epoch"] = best_epoch
    manifest["checkpoint_selection"] = selection_report(
        best, best_epoch, args.seed, manifest["split_fingerprint"])
    manifest["convergence"] = convergence(len(history))
    final = {"format": artifact_format, "kind": "inference", "task_type": "multilabel",
             "activation": "sigmoid", "state_dict": best_state, "args": config,
             "manifest": manifest, "threshold_config": threshold_config,
             "validation_metrics": metrics, "initialisation": initial}
    save_checkpoint(out / "best_multilabel.pt", final)
    write_json(out / "thresholds.json", threshold_config)
    write_json(out / "validation_metrics.json", metrics)
    write_json(out / "manifest.json", manifest)
    write_json(out / "training_status.json", {"status": "complete", "out": str(out),
        "convergence": manifest["convergence"], "inference_checkpoint_exported": True})
    np.savez_compressed(out / "validation_logits.npz", logits=logits,
        targets=data["targets"][val_idx], target_mask=data["target_mask"][val_idx],
        subjects=data["subjects"][val_idx], video_ids=data["video_ids"][val_idx],
        starts=data["starts"][val_idx])
    print(f"Selected epoch {best_epoch + 1}; validation mAP={best:.4f}. Saved {out / 'best_multilabel.pt'}")
    if manifest["convergence"]["warning"]:
        print("WARNING: " + manifest["convergence"]["warning"], flush=True)
    return final


def evaluate_test(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") not in FORMATS or saved.get("kind") != "inference":
        raise ValueError("test evaluation requires an exported multi-label inference checkpoint")
    for key in ("sample_fps", "annotation_fps", "min_overlap"):
        setattr(args, key, saved["args"][key])
    if file_sha256(args.annotations) != saved["manifest"]["annotation_sha256"]:
        raise ValueError("test annotations differ from the version used for training")
    if saved["threshold_config"].get("source_split") != "validation":
        raise ValueError("thresholds must already be fixed on validation")
    args.model = saved["args"].get("model", "skeleton")
    expected_format = FUSION_FORMAT if args.model == "context-fusion" else FORMAT
    if saved["format"] != expected_format:
        raise ValueError("checkpoint model architecture disagrees with its declared format")
    args.require_context = args.model == "context-fusion"
    args.require_full_coverage = args.require_full_coverage or saved["args"].get("require_full_coverage", False)
    if saved["args"].get("require_convergence") and not saved["manifest"].get("convergence", {}).get("converged"):
        raise ValueError("production test evaluation requires a converged candidate")
    paths = expand_shards(args.test_shards)
    data = load_data(paths, args)
    if args.require_context and data["provenance"].get("visual_context") != saved["manifest"].get("visual_context"):
        raise ValueError("test visual context differs from the trained detector/feature contract")
    subjects = set(data["subjects"].tolist())
    if not subjects or not subjects <= TEST_SUBJECTS:
        raise ValueError("test input must contain only official CS test subjects")
    if subjects & (set(saved["manifest"]["train"]["subjects"]) | set(saved["manifest"]["validation"]["subjects"])):
        raise ValueError("test subjects overlap model-selection subjects")
    indices = np.arange(len(data["targets"]))
    model = build_model(saved["args"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(args.device)
    loader = DataLoader(MultilabelWindows(data, indices), batch_size=args.batch_size, num_workers=args.workers)
    logits = predict(model, loader, args.device)
    metrics = assess(data, indices, logits, saved["threshold_config"]["thresholds"])
    report = {"checkpoint_sha256": file_sha256(args.checkpoint), "thresholds_refit": False,
              "checkpoint_selection_on_test": False, "split": split_report(data, indices),
              "evaluation_scope": saved["manifest"]["evaluation_scope"],
              "full_recording_temporal_coverage_verified": False,
              "training_convergence": saved["manifest"].get("convergence"),
              "model": args.model,
              "full_eligible_window_coverage_verified": data["provenance"].get("full_coverage_verified", False),
              "checkpoint_selection": saved["manifest"].get("checkpoint_selection"),
              "source_provenance": data["provenance"], "metrics": metrics}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "test_metrics.json", report)
    np.savez_compressed(out / "test_logits.npz", logits=logits, targets=data["targets"],
                        target_mask=data["target_mask"], subjects=data["subjects"],
                        video_ids=data["video_ids"], starts=data["starts"])
    print(json.dumps(report["metrics"], indent=2, allow_nan=False))
    return report


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--train-shards", "--shards", nargs="+", required=True)
    fit.add_argument("--annotations", required=True)
    fit.add_argument("--out", default="runs/adl_multilabel_bone")
    fit.add_argument("--init")
    fit.add_argument("--init-provenance", help="JSON with checkpoint_sha256, fit_subjects and selection_subjects")
    fit.add_argument("--model", choices=("skeleton", "context-fusion"), default="skeleton")
    fit.add_argument("--context-hidden", type=int, default=128)
    fit.add_argument("--require-context", action="store_true")
    fit.add_argument("--require-full-coverage", action="store_true")
    fit.add_argument("--require-convergence", action="store_true")
    fit.add_argument("--cache-dir", help="disk-backed arrays; defaults to the output directory's data_cache")
    fit.add_argument("--resume")
    fit.add_argument("--val-subjects", nargs="+")
    fit.add_argument("--epochs", type=int, default=100,
                     help="maximum epoch budget; reaching it does not establish convergence")
    fit.add_argument("--min-epochs", type=int, default=30,
                     help="minimum completed epochs before validation early stopping is allowed")
    fit.add_argument("--patience", type=int, default=12,
                     help="validation epochs without improvement; 0 disables early stopping")
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--weight-decay", type=float, default=1e-4)
    fit.add_argument("--warmup-epochs", type=int, default=2)
    fit.add_argument("--scheduler", choices=("cosine", "plateau"), default="cosine")
    fit.add_argument("--lr-patience", type=int, default=5)
    fit.add_argument("--lr-factor", type=float, default=0.5)
    fit.add_argument("--min-lr", type=float, default=1e-6)
    fit.add_argument("--convergence-lr-reductions", type=int, default=2)
    fit.add_argument("--precision", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    fit.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    fit.add_argument("--ema-decay", type=float, default=.99)
    fit.add_argument("--pos-weight-max", type=float, default=10.)
    fit.add_argument("--threshold-min-support", type=int, default=5)
    fit.add_argument("--sample-fps", type=float, default=20.)
    fit.add_argument("--annotation-fps", type=float, default=25.)
    fit.add_argument("--min-overlap", type=float, default=.5)
    fit.add_argument("--base-channels", type=int, default=64)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--split-seed", type=int, default=0,
                     help="fixed subject split seed, independent of training seed comparisons")
    fit.add_argument("--workers", type=int, default=2)
    fit.add_argument("--cpu-threads", type=int, default=4)
    fit.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    fit.add_argument("--stop-after", type=int, help=argparse.SUPPRESS)
    fit.add_argument("--stop-after-seconds", type=float,
                     help="save an exact resume at an epoch boundary after this session budget")
    test = commands.add_parser("evaluate-test")
    test.add_argument("--checkpoint", required=True)
    test.add_argument("--test-shards", nargs="+", required=True)
    test.add_argument("--annotations", required=True)
    test.add_argument("--out", default="runs/adl_multilabel_test")
    test.add_argument("--batch-size", type=int, default=256)
    test.add_argument("--workers", type=int, default=2)
    test.add_argument("--cache-dir")
    test.add_argument("--require-full-coverage", action="store_true")
    test.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    arguments = parser().parse_args()
    train(arguments) if arguments.command == "train" else evaluate_test(arguments)
