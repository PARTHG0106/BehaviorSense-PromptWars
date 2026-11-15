"""Masked multi-label metrics and validation-only decision thresholds."""

from __future__ import annotations

import numpy as np


def _arrays(scores, targets, mask):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)
    if scores.ndim != 2 or scores.shape != targets.shape or scores.shape != mask.shape:
        raise ValueError("scores, targets and mask must share shape [N, C]")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite sigmoid probabilities")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")
    return scores, targets.astype(bool), mask


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Noninterpolated AP, grouping tied scores so row order cannot change AP."""
    if not targets.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y, s = targets[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, tp]) * precision) / y.sum())


def masked_metrics(scores, targets, mask, thresholds=None, *, pair=(13, 10)) -> dict:
    """Score only known labels; unsupported class metrics remain explicit nulls.

    Pair metrics describe two positive window labels. Actual frame-level overlap is
    reported separately by the data audit, since window co-labels alone cannot prove it.
    """
    scores, targets, mask = _arrays(scores, targets, mask)
    thresholds = np.full(scores.shape[1], .5) if thresholds is None else np.asarray(thresholds)
    if (thresholds.shape != (scores.shape[1],) or not np.isfinite(thresholds).all()
            or np.any((thresholds < 0) | (thresholds > 1))):
        raise ValueError("one finite threshold in [0, 1] is required per class")
    pred = scores > thresholds
    tp = (pred & targets & mask).sum(0)
    fp = (pred & ~targets & mask).sum(0)
    fn = (~pred & targets & mask).sum(0)
    positives, negatives = (targets & mask).sum(0), (~targets & mask).sum(0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float),
                   where=(2 * tp + fp + fn) > 0)
    ap = [average_precision(scores[mask[:, c], c], targets[mask[:, c], c])
          if positives[c] and negatives[c] else None for c in range(scores.shape[1])]
    eligible = positives > 0
    out = {
        "n_windows": len(scores), "known_labels": int(mask.sum()),
        "macro_ap": float(np.mean([v for v in ap if v is not None]))
        if any(v is not None for v in ap) else None,
        "macro_f1": float(f1[eligible].mean()) if eligible.any() else None,
        "micro_f1": float(2 * tp.sum() / max(1, (2 * tp + fp + fn).sum())),
        "per_class_ap": ap,
        "per_class_f1": [float(f1[c]) if positives[c] else None for c in range(len(f1))],
        "per_class_positive": positives.tolist(), "per_class_negative": negatives.tolist(),
        "per_class_tp": tp.tolist(), "per_class_fp": fp.tolist(), "per_class_fn": fn.tolist(),
    }
    if pair is not None:
        valid = mask[:, list(pair)].all(1)
        truth = targets[:, list(pair)].all(1) & valid
        guesses = pred[:, list(pair)].all(1) & valid
        hit = int((truth & guesses).sum())
        support, guessed = int(truth.sum()), int(guesses.sum())
        out["tv_drinking_colabeled_windows"] = {
            "support": support, "predicted": guessed, "true_positive": hit,
            "precision": hit / guessed if guessed else 0.,
            "recall": hit / support if support else None,
            "f1": 2 * hit / (support + guessed) if support + guessed else None,
        }
    return out


def fit_thresholds(scores, targets, mask, *, min_positive=5, min_negative=5,
                   train_positive=None) -> dict:
    """Fit per-class F1 cutoffs on the selected epoch's validation predictions only.

    A class with insufficient validation evidence is disabled (strict probability > 1
    can never fire). Equal F1 values prefer the higher, more conservative cutoff.
    """
    scores, targets, mask = _arrays(scores, targets, mask)
    if min_positive < 1 or min_negative < 1:
        raise ValueError("threshold support requirements must be positive")
    if train_positive is not None:
        train_positive = np.asarray(train_positive)
        if train_positive.shape != (scores.shape[1],) or (train_positive < 0).any():
            raise ValueError("train_positive must contain one nonnegative count per class")
    cutoffs, reasons = [], []
    for c in range(scores.shape[1]):
        y, p = targets[mask[:, c], c], scores[mask[:, c], c]
        if train_positive is not None and train_positive[c] == 0:
            cutoffs.append(1.); reasons.append("no_training_positives"); continue
        if y.sum() < min_positive or (~y).sum() < min_negative:
            cutoffs.append(1.); reasons.append("insufficient_validation_support"); continue
        # Evaluate every distinct prediction set, including probabilities below .05
        # and above .95. A fixed coarse grid can disable a well-ranked rare class.
        order = np.argsort(-p, kind="stable")
        ordered = p[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
        tp = np.cumsum(y[order])[ends]
        f1 = 2 * tp / (ends + 1 + y.sum())
        # Float32-representable cutoffs keep strict > semantics in inference.
        candidates = np.nextafter(ordered[ends].astype(np.float32), np.float32(0)).astype(float)
        valid = ordered[ends] > 0
        best = max(((float(score), float(cutoff)) for score, cutoff in
                    zip(f1[valid], candidates[valid])), default=(0., 1.))
        cutoffs.append(best[1]); reasons.append("validation_f1")
    return {"thresholds": cutoffs, "enabled": [v < 1 for v in cutoffs],
            "reasons": reasons, "comparison": ">", "source_split": "validation",
            "objective": "per_class_f1", "min_positive": min_positive,
            "min_negative": min_negative, "search": "all_distinct_validation_prediction_sets",
            "validation_positive": (targets & mask).sum(0).tolist(),
            "validation_negative": (~targets & mask).sum(0).tolist()}
