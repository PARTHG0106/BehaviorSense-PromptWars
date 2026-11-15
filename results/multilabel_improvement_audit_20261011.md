# Multilabel improvement audit — 11 October 2026

There is measurable room to improve validation performance. Combining the existing models produces modest gains without retraining, but reliable simultaneous TV watching and drinking remains unresolved. The comparisons below use the saved validation predictions only; they are not new official test results.

## Measured validation comparison

All six sets of predictions were aligned by video ID and window start. Targets, masks and subject IDs match across all 63,220 rows. Each individual candidate reproduces its saved validation results. Ensemble thresholds use the same validation-only per-class F1 method as the original trainer.

Four combinations were specified before their scores were computed: the three context models or all six models, each with either mean probabilities or mean pre-sigmoid logits. All members receive equal weights.

| Candidate | Validation mAP | Macro F1 | Micro F1 | TV/drinking co-label F1 | Model passes per window |
|---|---:|---:|---:|---:|---:|
| Selected context model, seed 2 | 59.25% | 0.58010 | 0.70535 | 0.14488 | 1 |
| Three context models, mean probabilities | 60.53% | 0.59550 | 0.70540 | 0.15278 | 3 |
| Three context models, mean logits | 60.61% | 0.59576 | 0.70290 | 0.14912 | 3 |
| All six models, mean probabilities | 60.60% | 0.59696 | 0.71208 | 0.09665 | 6 |
| All six models, mean logits | 60.82% | 0.59566 | 0.70956 | 0.12609 | 6 |

The three-context probability ensemble improves validation mAP by **1.27 percentage points** and co-label F1 from **0.14488 to 0.15278**. This is a small concurrency improvement, not evidence of reliable detection. Classification requires three model passes; detector extraction can still be shared.

The six-model logit ensemble achieves the highest validation mAP, **60.82%**, an increase of **1.57 percentage points**, but co-label F1 falls to **0.12609**. It is not the preferred next step for the TV/drinking objective.

These are exploratory validation comparisons. Thresholds are fitted and scored on the same validation set, as in the baseline. The extra ensemble search adds selection flexibility. No test predictions were loaded for this audit, no GPU jobs were launched, and no checkpoint was promoted. The prior official test results have already been inspected; further development must not be described as an untouched confirmatory experiment on that same test set.

## What limits the current experiment

1. **Limited visual evidence.** The active model uses 30 skeleton frames at 20 Hz, about 1.5 seconds, plus RT-DETR class scores and geometry from one center RGB frame. It does not consume learned RGB video features. Seeing a cup or TV in one frame does not by itself distinguish drinking or watching from object presence.
2. **Rare simultaneous examples.** Only 711 of 217,090 fitting windows carry both TV and drinking labels, about 0.33%. They are overlapping windows, not 711 independent events. Validation has 205 such windows: 197 from p04 and only 8 from p15. This makes the pair score and its thresholds sensitive to the subject distribution.
3. **Optimization favors individual classes.** Training uses masked, class-weighted binary cross-entropy, and model selection uses overall validation mAP. Pair success is reported but is not the selection objective. The six-model ensemble demonstrates why better overall mAP need not improve concurrent activity detection.
4. **More epochs alone has weak support.** The selected context model peaked at epoch 15 with validation mAP 0.59254. By epoch 40, training loss had fallen from 0.09607 to 0.05584 while validation mAP fell to 0.58222. This is consistent with overfitting; it does not show that every longer or different training schedule would fail.

A calibration transfer check fitted thresholds on one validation subject and scored the other, then pooled predictions. Baseline macro F1 dropped from 0.58010 to 0.47220 and pair F1 from 0.14488 to 0.02362. The three-context probability ensemble scored 0.48463 macro F1 and 0.03156 pair F1 in that check. This flags threshold instability with only two validation subjects. Because epoch selection already used both subjects, this is a calibration diagnostic, not independent model cross-validation.

## Recommended next experiments

1. **Prespecify concurrent-activity success alongside overall quality.** Track pair AP/precision/recall and per-class scores on several subject-held-out validation folds. Keep window definitions fixed so improvements are comparable. Retain the existing official result as the baseline.
2. **Add discriminative video features.** Compare a pretrained RGB video encoder, including person/hand/object crops, with the existing skeleton and context features. Use several RGB observations and longer surrounding context, such as 6–12 seconds, while retaining the current center-window target. This may help separate hand-to-mouth actions from nearby objects, but requires new feature extraction and validation.
3. **Improve learning from rare examples.** Test additional weights for real concurrent windows and difficult negatives such as TV-only, holding a cup without drinking, eating and medication. Retain every eligible training window; weights and explicit losses can emphasize difficult cases without changing the coverage claim. Audit labels and add genuinely new subjects/scenes with concurrent activity if available.
4. **Validate a small pilot before the full run.** Compare against the frozen baseline with a fixed budget and subject split, then repeat promising changes across seeds and additional held-out subjects. Treat the three-context ensemble as an inexpensive reference, not a solution to concurrency.

No particular final accuracy or test improvement is established by this audit. The strongest demonstrated gain so far is in validation mAP; the larger proposed changes remain hypotheses to test.

[Machine-readable comparisons, input hashes and fitted threshold configurations](../artifacts/multilabel-improvement-audit-20261011/validation_ensembles.json)

[Completed official baseline evaluation](multilabel_test_20261011.md)

The follow-up [research and single-session experiment](../docs/10_temporal_multilabel_research.md)
completed on Kaggle in one session. It tested fresh joint/bone models with ordered
multi-window detector context and a BCE/ASL comparison. The
[verified follow-up results](temporal_multilabel_20261012.md) show a modest F1/concurrency
gain but no overall test mAP improvement.
