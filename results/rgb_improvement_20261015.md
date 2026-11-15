# RGB hybrid improvement — 15 October 2026

**Independently verified test improvement: mAP 65.29% → 69.21%, and TV-plus-drinking F1 19.07% → 28.12%.** The fixed 30% pair-precision and pair-F1 requirements still fail; this is a measured improvement, not a passing quality gate.

The single frozen benchmark covers all **153,537 retained windows from 185 videos and seven test subjects**. mAP is the average over the **15 classes with benchmark labels**. The other seven output classes have unknown labels and remain unscored.

These results remain exploratory: validation guided candidate development and the test benchmark had been inspected earlier in the project. No weights, thresholds, blend weights, or decoder settings changed after the test freeze. No serving model was replaced.

## Frozen test result

| Metric | Previous equal ensemble | Frozen RGB hybrid | Change |
|---|---:|---:|---:|
| mAP | 65.29% | 69.21% | +3.92 pp |
| Macro F1 | 61.09% | 62.29% | +1.20 pp |
| Micro F1 | 66.90% | 68.06% | +1.16 pp |
| TV + drinking precision | 12.59% | 17.84% | +5.25 pp |
| TV + drinking recall | 39.37% | 66.38% | +27.01 pp |
| TV + drinking F1 | 19.07% | 28.12% | +9.05 pp |
| TV + drinking ranking AP | 11.83% | 23.01% | +11.18 pp |

The fixed test checks were: mAP ≥63.79%, mAP strictly above the 65.29% reference, and pair precision, recall and F1 each ≥30%. Both mAP checks and pair recall pass. Pair precision misses its floor by 12.16 percentage points, and pair F1 by 1.88 points.

The hybrid predicts 2,590 pair-positive windows against 696 labeled positives. True positives rise **274 → 462**, false negatives fall **422 → 234**, and false positives rise **1,903 → 2,128**. Of its false positives, 946 are drinking-only windows, 969 are TV-only windows, and 213 have neither central-window label. These are descriptive counts from the frozen result, not a basis for adjusting this candidate.

Pair labels denote both activities in the central labeled window. The separately reported sampled-overlap endpoint has 928 positives, 589 true positives, 22.74% precision and 63.47% recall; it is not the prespecified pair endpoint.

## Test subjects

All seven subjects improve mAP. Pair F1 improves on six; p11 declines from 27.59% to 26.77%. The 153 extra true positives on p10 account for 81.4% of the aggregate net pair-TP gain. These subject summaries are descriptive, not extra acceptance gates.

| Subject | Previous mAP | Hybrid mAP | Hybrid pair precision | Hybrid pair recall | Previous pair F1 | Hybrid pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| p02 | 66.65% | 69.90% | 15.00% | 21.43% | 13.97% | 17.65% |
| p10 | 67.89% | 70.36% | 27.19% | 80.20% | 38.22% | 40.62% |
| p11 | 73.60% | 76.35% | 15.74% | 89.47% | 27.59% | 26.77% |
| p14 | 71.65% | 75.51% | 6.17% | 61.84% | 3.86% | 11.22% |
| p16 | 61.49% | 63.04% | 6.32% | 40.00% | 3.43% | 10.91% |
| p18 | 60.65% | 64.71% | 7.14% | 50.00% | 5.65% | 12.50% |
| p20 | 64.69% | 68.74% | 27.04% | 44.54% | 21.59% | 33.65% |

Pair recall decreases on p02, p18 and p20; pair precision decreases on p10 and p11. For overall class decisions, p16 macro F1 falls 56.78% → 54.91% and micro F1 falls 69.59% → 67.64%; p11 and p20 also have small micro-F1 declines. The aggregate improvement is not uniform across metrics.

## Test classes

AP improves on 14 of the 15 labeled classes. Cooking is the exception. Device and phone F1 decline despite improved ranking AP. Taking-medication AP and F1 improve while true positives fall 344 → 258, so this does not establish better recall for that activity.

| Class | Previous AP | Hybrid AP | Previous F1 | Hybrid F1 |
|---|---:|---:|---:|---:|
| walking | 87.94% | 89.00% | 77.37% | 77.18% |
| lying down | 76.53% | 81.00% | 70.18% | 75.71% |
| standing up | 87.43% | 87.89% | 81.55% | 83.29% |
| sitting down | 82.48% | 83.52% | 77.67% | 79.72% |
| eating | 70.85% | 76.73% | 67.50% | 69.80% |
| drinking | 68.67% | 77.76% | 62.43% | 69.28% |
| cooking food prep | 76.04% | 73.42% | 67.33% | 67.10% |
| taking medication | 17.05% | 31.09% | 26.40% | 29.42% |
| watching tv | 82.33% | 85.32% | 73.66% | 77.05% |
| reading | 74.07% | 83.00% | 75.72% | 77.42% |
| using phone | 47.73% | 50.38% | 35.66% | 32.86% |
| cleaning housework | 76.63% | 76.79% | 70.61% | 70.65% |
| personal hygiene | 30.07% | 30.13% | 37.76% | 38.18% |
| using device | 41.44% | 50.93% | 35.47% | 29.47% |
| object interaction | 60.11% | 61.15% | 57.04% | 57.29% |

Unscored classes: standing, sitting, bending/reaching, falling, fallen-on-ground, interacting-with-person and other/idle. This benchmark does not establish fall-detection or healthcare deployment performance.

## Frozen candidate and validation evidence

The class ranking scores retain the fixed probability blend: 37.5% original
context seed 2, 37.5% temporal ASL seed 1 and 25% expanded RGB. Its **65.87% validation mAP
is inherited from that blend**, not produced by the temporal decision rule.

The joint decoder begins with the existing RGB head's frozen TV and drinking
thresholds. It takes the weaker normalized confidence for both activities in
each window, then the maximum of those same-window joint confidences within
±30 start frames (±1.5 seconds) in the same video. It additionally requires the
fixed blend to identify TV activity in the central window. Neither thresholds
nor the time window were searched during this candidate's evaluation.

Both final TV/drinking flags are true exactly when this joint rule passes. If
the rule fails but both independent blend flags are true, the weaker normalized
margin is suppressed; ties suppress drinking. Other class decisions remain
unchanged. All class, pair, macro and micro F1 values use this final boolean
matrix. Class ranking scores and pair confidence are separate outputs and do
not constitute a calibrated joint probability distribution.

The temporal rule adds 1.5 seconds of look-ahead beyond the underlying models'
context. It supports offline inference; it is not a zero-latency detector. The
ground truth remains the original central-window labels and masks.

| Validation check | Result | Required |
|---|---:|---:|
| Overall mAP | **65.87%** | 63.79% |
| Pair precision | **31.98%** | 30.00% |
| Pair recall | **30.73%** | 30.00% |
| Pair F1 | **31.34%** | 30.00% |
| p04 mAP | **65.01%** | 59.13% |
| p04 pair F1 | **37.13%** | 21.81% |
| p15 mAP | **71.13%** | 64.50% |
| p15 pair F1 | **12.63%** | 3.86% |

The candidate makes **197 pair predictions with 63 true positives**, against 205
positive windows. Pair ranking AP is 23.51%. Macro F1 is 62.53% and micro F1 is
73.72%. These pair labels mean watching TV and drinking within the same labeled
window; actual sampled temporal overlap is reported separately in the artifacts.

The preceding temporal-only candidate achieved 29.59% precision, 38.54% recall
and 33.47% F1, failing precision. The central-TV condition was an adaptive next
hypothesis prompted by that failure. It removes 70 temporal pair predictions:
54 false pairs and 16 true pairs. This tradeoff is part of the reported result.
The selected candidate changes 265 class decisions relative to the fixed blend.

## Fitting-only research retained

The expanded raw fitting cache contains **40,193 rows × 3,025 features**, from
60 videos, 51 recordings and nine subjects. All original 36-video features and
labels reproduced exactly. No validation or test arrays entered these searches.

| Study | Fixed comparisons | Best diagnostic precision / recall / F1 | Outcome |
|---|---:|---|---|
| Five joint model families × three temporal decoders | 15 | 18.84% / 31.03% / 23.45% | Rejected |
| Complementary joint-score ensembles | 9 | 20.38% / 42.16% / 27.48% | Rejected |
| Separate TV/drinking models and modality comparisons | 24 | 21.15% / 47.18% / 29.21% | Rejected |
| Joint/factorized score ensembles | 9 | 23.95% / 38.56% / 29.55% | Rejected |

These are pooled predictions for held-out fitting subjects used to select
models, decoders and cutoffs. They are selection evidence, not independent final
accuracy estimates. None met precision, recall and F1 floors together, and none
exported a full fitted model. Those failed protocols remain closed. The selected
candidate above is a separate untrained postprocessor of already fitted models.

Independent audits replayed all 200,965 joint-model held predictions and all
321,544 factorized marginal predictions, including training-fold normalization,
source/input/checkpoint hashes, threshold selection and temporal grouping.

## Execution and verification

The private run `krishnagupta02468/bs-rgb-hybrid-test-20261015`, version 1, completed all required extraction and frozen-head inference in **3183.27 seconds (53m 03s)**. Its three metadata-selected whole-video probes passed in 40.45 seconds. The conservative initial projection was 7,217 seconds, including a 1,800-second export reserve, within the fixed 40,500-second session budget. Exactly one version was submitted and accepted.

The transport preserves the original 88-file scientific archive and all original pilot checkpoint/normalizer/threshold bytes. The new source wrapper is 787,661 bytes, below Kaggle's 1 MB source limit. The job ran on the pinned RTX PRO 6000 runtime. It did no training, recalibration, model selection or metric evaluation; the single metric pass ran locally after collection.

Twenty targeted tests passed. The benchmark and independently implemented auditor both rehearsed all 63,220 validation rows exactly before the test freeze. The complete export then passed an identity, runtime, byte-hash, timing and coverage audit before scoring. The final independent replay verified all 153,537 unique rows, component targets and masks, score blending, temporal witnesses, final flags, all aggregate/class/subject metrics and success checks, with absolute numeric tolerance **1e-12**. Its grouped AP implementation also matched scikit-learn; it did not import the benchmark, decoder or project metric functions.

The mAP gain belongs to the fixed score blend; the temporal decoder affects final decisions, not class ranking. It requires future context and is an offline window-level system. The original RGB checkpoint remains a 60-video pilot whose standalone gate failed; the composite's separate test gate also fails on precision/F1. This candidate and its benchmark are closed to retuning. Any further improvement must use a new fitting-only protocol and a separately justified evaluation, not a search over these test scores.

Private evidence is under `artifacts/rgb-improve-20261015/`:

- [Frozen test protocol](../artifacts/rgb-improve-20261015/frozen-test/test_protocol.json)
- [Benchmark result](../artifacts/rgb-improve-20261015/frozen-test/benchmark_result.json)
- [Output audit](../artifacts/rgb-improve-20261015/frozen-test/run/output_audit.json)
- [Independent benchmark replay](../artifacts/rgb-improve-20261015/frozen-test/independent_benchmark_audit.json)
- [Independent terminal interpretation](../artifacts/rgb-improve-20261015/frozen-test/independent_terminal_interpretation_20261015.json)
- [Selected validation result](../artifacts/rgb-improve-20261015/temporal-tv-anchor/result.json)
- [Selected validation audit](../artifacts/rgb-improve-20261015/temporal-tv-anchor/independent_audit.json)
