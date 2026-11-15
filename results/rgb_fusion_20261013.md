# Fixed RGB fusion — 13 October 2026

**The fixed 75% existing ensemble / 25% RGB probability blend increases validation
mAP from 61.79% to 65.71%, a gain of 3.91 percentage points.** Both validation
subjects improve. It clears the 63.79% mAP requirement, but TV/drinking precision,
recall and F1 remain below their 30% requirements. This is a validation result,
not a new test result or a deployed model.

Exactly one new weight combination was evaluated. The existing ensemble remains
the fixed 50/50 context-seed-2 / temporal-ASL-seed-1 probability average. Thus the
new weights are 37.5% context, 37.5% temporal and 25% RGB. The combination uses
float32 probabilities directly, without an extra logit round trip.

## Complete validation comparison

All results cover the same **63,220 retained windows from 77 videos**, with p04
and p15 held out from model fitting. Fifteen of the 22 taxonomy classes have known
supervision. Rows, video/start keys, subjects, targets and masks match exactly
between all three components; labels and temporal overlap were replayed from the
original annotations.

| Validation metric | Existing ensemble | Fixed RGB blend | Requirement |
|---|---:|---:|---:|
| mAP | 61.79% | **65.71%** | ≥63.79% |
| Macro F1 | 60.28% | **62.74%** | — |
| Micro F1 | 71.37% | **73.46%** | — |
| TV/drinking precision | 15.14% | **23.90%** | ≥30% |
| TV/drinking recall | 16.10% | **18.54%** | ≥30% |
| TV/drinking F1 | 15.60% | **20.88%** | ≥30% |
| TV/drinking pair AP | 6.90% | **11.82%** | — |

The blend makes **159 pair predictions**, of which **38** are true positives,
against **205 positive pair windows**. The existing ensemble made 218 predictions
with 33 true positives. These are co-labeled window metrics; frame-level temporal
overlap is a separate diagnostic.

| Subject | Existing mAP | Blend mAP | Change | Blend pair precision / recall / F1 |
|---|---:|---:|---:|---:|
| p04 | 61.70% | **64.27%** | +2.57 points | 38.10% / 16.24% / 22.78% |
| p15 | 66.52% | **71.11%** | +4.59 points | 8.00% / 75.00% / 14.46% |

All four original subject-level requirements pass. Overall pair precision, recall
and F1 are the three failed checks. p15 has only eight positive pair windows, so
its recall estimate is particularly sensitive to individual examples.

## Calibration and audit

The blend weight, input hashes, comparison method, threshold policy and quality
requirements were saved before scoring. The existing ensemble uses its original
frozen thresholds. The candidate uses the same per-class validation-F1 method,
requiring at least five positive and negative validation examples and fitting
support. No pair-specific threshold or alternate blend weight was searched.

The first evaluator attempt stopped while replaying the old ensemble because the
direct metric helper omitted pair AP fields. The repair added the established
minimum-marginal-probability AP report. The original evaluator is preserved, and a
repair receipt binds both source versions. The scientific recipe and already
saved thresholds were retained; this did not introduce another candidate.

The baseline ensemble and RGB pilot metrics reproduce their saved reports.
The successful evaluation took about **22 seconds on CPU**. No GPU inference,
GPU training, test-array loading or serving-model change occurred.

The components and earlier validation results had already been inspected when this
recipe was proposed. Thresholds are fitted and scored on the same two validation
subjects. This is therefore exploratory complementarity evidence. It does not
establish a 65.71% test score or reliable concurrent detection.

## Completed pair-aware follow-up

The gain supports a small follow-up using cached, frozen RGB and raw pose/object
features. A four-state linear classifier represents neither activity, drinking
only, TV only, and both. Its recipe is fixed before fitting: L2 logistic regression
with C=0.1, balanced class weights and at most 400 solver iterations. No supervised
baseline logits or hidden features enter its fitting folds.

The 36 pilot fitting videos contain 31,859 windows and 562 co-labeled pairs across
all nine fitting subjects. Folds hold out entire subjects; normalization is fitted
inside each fitting fold. A pair cutoff is selected using pooled predictions for
held-out fitting subjects, requiring at least 30% precision, recall and F1. Failure
stops before a full fitting head or outer-validation inference.

If that check passes, one head is fitted on all 36 videos, its checkpoint and
cutoff are frozen, and the complete p04/p15 split is evaluated once. TV/drinking
scores and decisions then come from the four-state model; the other activities
retain the fixed blend. The full original absolute and subject requirements
still apply. The pair score is the probability of the joint state, so its decoder
is explicitly different from independently thresholding two marginal scores.

All 36 fitting-video caches were recovered and verified from completed jobs.
The three subject-separated fits converged in 164.74 seconds of worker time,
but no pooled cutoff met the required pair precision, recall and F1. Its best
diagnostic result was **20.00% precision, 13.88% recall and 16.39% F1**: 78 true
positives from 390 pair predictions against 562 positives. These are fitting
out-of-fold metrics, not p04/p15 validation metrics. The frozen protocol stopped
before fitting a full head or evaluating it on the outer validation split.

The [completed pair-head report](rgb_pair_head_20261013.md) records the recovery
audit, errors and coverage diagnosis. No new GPU job, test evaluation or
serving-model change occurred. The accuracy goal remains unresolved; the fixed
blend clears mAP, but neither candidate meets every pair requirement.

Private evidence: `artifacts/rgb-throughput-20261013/fusion-7525/` contains the
frozen recipe, source/repair receipts, calibrated thresholds, probability arrays
and full result. The conditional fitting protocol is under `pair-head/`.
