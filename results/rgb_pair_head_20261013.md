# Four-state pair head — 13 October 2026

**The CPU pair-aware experiment completed, but failed its subject-separated
fitting gate.** At its best diagnostic cutoff, TV/drinking precision was
**20.00%**, recall **13.88%** and F1 **16.39%**, below the required 30% each.
No cutoff satisfied all three requirements. The frozen protocol therefore
stopped before fitting a full head or evaluating this candidate on p04/p15.

This completes the experiment interrupted by the earlier chat error. Its saved
fitting worker took **164.74 seconds**, within the 1,200-second CPU allowance.
All three solvers converged. The failure is measured quality, not a timeout or
an unfinished training run. Recovery did not repeat fitting or launch a GPU job.

## What was fitted

One fixed four-state logistic classifier distinguishes neither activity,
drinking only, TV only, and both. It uses 3,025 features: 1,489 raw corrected
pose/object summaries and 1,536 frozen scene/person RGB features. No supervised
baseline predictions or hidden representations enter these folds.

The **31,859 windows from 36 videos across nine fitting subjects** were divided
into three groups. Each model fitted six subjects and predicted only the other
three. Means and standard deviations were fitted inside each fitting fold.
The model used C=0.1, balanced class weights, at most 400 iterations, and two CPU
threads. No hyperparameter sweep or second candidate was run.

The joint score is the probability of the "both" state. The protocol selected
one cutoff from pooled predictions for held-out fitting subjects, maximizing F1
subject to the precision/recall/F1 floors and preferring higher cutoffs on ties.
Since none passed, the unconstrained best F1 is diagnostic only; it is not an
approved threshold for deployment or further evaluation.

## Held-subject fitting result

| Metric | Result | Requirement |
|---|---:|---:|
| Pair precision | 20.00% | ≥30% |
| Pair recall | 13.88% | ≥30% |
| Pair F1 | 16.39% | ≥30% |
| Pair average precision | 7.89% | — |
| Positive pair windows | 562 | — |
| Pair predictions | 390 | — |
| True positives / false positives / false negatives | 78 / 312 / 484 | — |
| Diagnostic cutoff, strict `>` | 0.8260419368743895 | No passing cutoff |

All fold metrics below use that same pooled diagnostic cutoff.

| Fold | Held subjects | Held windows | Pair positives | Precision | Recall | F1 | Solver iterations |
|---|---|---:|---:|---:|---:|---:|---:|
| A | p12, p17, p25 | 10,642 | 192 | 22.86% | 25.00% | 23.88% | 272 |
| B | p03, p06, p09 | 10,790 | 188 | 16.31% | 12.23% | 13.98% | 259 |
| C | p07, p13, p19 | 10,427 | 182 | 17.95% | 3.85% | 6.33% | 221 |

Pooled cutoff selection and scoring reuse the same out-of-fold predictions.
These numbers are selection evidence, not an independent final evaluation.
The existing 75/25 blend's **65.71% mAP and 20.88% pair F1** were measured on the
different p04/p15 validation split; they are not directly comparable with this
head's fitting-fold F1. The [fixed blend report](rgb_fusion_20261013.md) remains
the source for those results. Neither experiment passes every quality gate.

## Failure and coverage diagnosis

Of the 312 false pair predictions, 144 are neither activity, 64 are drinking
only, and 104 are TV only. The problem includes both missed positives and false
joint detections; lowering the cutoff cannot meet the frozen requirements on
these scores.

All nine fitting subjects contribute pair positives, spanning 29 videos but
only 24 subject/take groups. Twenty of those 29 videos have zero true positives;
53 of the 78 detected windows come from just two videos. Overlapping windows and
multiple camera views of a recording are correlated observations; 562 positive
windows do not mean 562 independent concurrent episodes.
The broader multilabel pilot also has weak class coverage:

| Activity | Positive windows | Videos with positives | Subjects with positives |
|---|---:|---:|---:|
| Cleaning | 159 | 1 | 1 |
| Hygiene | 15 | 5 | 2 |
| Eating | 84 | 5 | 3 |

Cleaning has no fitting positives when fold B is held out, and hygiene has none
when fold C is held out. These are limitations for a future multilabel pilot.
The current four-state pair target has all four states represented in every
training and held-subject fold, so those other class gaps do not establish the
cause of its failure.

The labels count both activities only when each meets its per-window overlap
rule. There are 562 co-labeled pair windows, 561 of which contain actual sampled
temporal overlap. A separate diagnostic finds 795 windows with any actual
overlap, including 234 that do not meet both class-label cutoffs. These are
different targets and should not be conflated with frame-level concurrent
detection.

The next pilot should select complete videos using fitting annotations only,
with positive coverage across subjects and distinct recordings, and enough
examples of neither activity, drinking only, TV only, and both. Selection should
check positive support in every training fold and its held-subject group,
count camera views as the same recording for diversity, and report infeasible
coverage before extraction. A concrete starting requirement is positives from
at least three subjects and three distinct recordings per supervised activity,
with all four pair states represented in every fold. These are proposed sampling
requirements, not a launched experiment or evidence that a larger pilot will
pass. Keep the original absolute, pair, subject and compute requirements.

## Recovery evidence

The recovery audit binds the original protocol, trainer, input receipts, fold
checkpoints and prediction files by SHA-256. Both the validation-cache guard and
the validation evaluator reject this terminal state before array access,
authentication or inference. No full-head checkpoint or post-gate validation
artifacts are present. The earlier fusion audit remains bound to its unchanged
recipe and result.

The independent audit byte-verified all 72 fitting cache files and reconstructed
all 31,859 raw feature rows, labels and masks exactly. Each row was held out
once; all three fold normalizations and exported predictions replayed exactly.
An independent metric implementation reproduced the counts, AP, best diagnostic
cutoff and zero feasible cutoffs.

The inherited audit initially assumed float64 probability precision because
that was the storage dtype. Inference actually used float32 before copying into
the float64 array. The audit now checks the correct numerical tolerance and
also requires exact float32 prediction replay. The largest probability-sum
error was 1.81e-7; an independent float64 inference check differed by at most
2.43e-6. This repaired an audit assumption without changing the protocol,
trainer, checkpoints, predictions or quality decision.

Private reproducibility artifacts are under
`artifacts/rgb-throughput-20261013/pair-head/`: the frozen `protocol.json`,
`fit_result.json`, `oof_result.json`, fold checkpoints and predictions, fitting
cache provenance, `independent_oof_audit.json`, `coverage_diagnosis.json`,
`recovery_audit.json`, and their reproducible scripts. This audit
checks artifacts and relevant guard calls, not operating-system execution
tracing. No new test result or serving-model replacement is established.
