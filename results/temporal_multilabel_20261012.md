# Temporal multilabel experiment — verified 12 October 2026

**All six candidates converged in one Kaggle session.** The pipeline completed in **6 hours, 3 minutes, 40 seconds** (21,820.41 seconds), including final evaluation, within the requested 12-hour limit. There were no continuation jobs. The actual accelerator was the NVIDIA RTX PRO 6000 Blackwell Server Edition, with native sm_120 support and the pinned runtime.

The result is mixed: **F1 and TV/drinking detection improved modestly, while overall test mAP did not improve**. The validation-selected model is **temporal fusion, asymmetric loss (ASL), seed 1**. It uses fresh joint/bone backbones and ordered detector observations spanning nine seconds, with 4.5 seconds of future context around the unchanged 1.5-second target window.

## Comparison with the frozen baseline

| Metric | Previous selected model | New selected model | Change |
|---|---:|---:|---:|
| Validation mAP | 59.25% | 60.64% | +1.39 pp |
| Test mAP | 62.61% | 62.55% | -0.054 pp |
| Test macro F1 | 57.96% | 58.90% | +0.93 pp |
| Test micro F1 | 65.21% | 65.69% | +0.48 pp |
| Test TV/drinking co-label F1 | 0.15666 | 0.17827 | +0.02160 |
| Test pair precision | 9.44% | 10.74% | +1.30 pp |
| Test pair recall | 46.12% | 52.44% | +6.32 pp |

Both models were evaluated on the exact same **153,537 retained windows from all 185 test videos**. Targets, masks, window identities and original row order match. AP and F1 are not classification accuracy. The tiny mAP decrease does not establish statistical significance; neither does the validation gain establish general improvement across people.

For TV/drinking, true positives increased from **321 to 365**, while predicted pairs were almost unchanged (**3,402 to 3,399**). Nevertheless, **89.26% of predicted pairs remain incorrect**. The new result does not establish reliable simultaneous activity detection.

## Prespecified selection and all six candidates

Every candidate used all **217,090 fitting windows** and **63,220 validation windows** from the same 351 training videos. Only the loss and seed differ between the new candidates; their architecture and schedule match. Candidate checkpoints were selected by validation mAP, followed by validation-only threshold fitting.

The frozen gate required mAP above the previous selected checkpoint's 0.5925371084, pair F1 at least 0.1448818898 and pair precision at least 0.1069767442. ASL seed 1 was the **only** candidate that passed all three requirements. Its weights and thresholds were frozen before loading test features.

| Candidate | Validation mAP | Pair F1 | Pair precision | Completed epochs | Best epoch | Gate passed |
|---|---:|---:|---:|---:|---:|---|
| BCE seed 0 | 0.58179 | 0.22030 | 0.19767 | 51 | 26 | No |
| BCE seed 1 | 0.58735 | 0.24213 | 0.24038 | 40 | 18 | No |
| BCE seed 2 | 0.59029 | 0.23828 | 0.19870 | 40 | 18 | No |
| ASL seed 0 | 0.60339 | 0.13568 | 0.09137 | 45 | 12 | No |
| ASL seed 1 | 0.60644 | 0.14700 | 0.12179 | 40 | 18 | Yes |
| ASL seed 2 | 0.59217 | 0.24638 | 0.24402 | 40 | 13 | No |

BCE mean validation mAP across three seeds: **58.648%**; mean pair F1: **0.23357**. ASL mean validation mAP across three seeds: **60.067%**; mean pair F1: **0.17635**.

ASL improves average ranking in this comparison but has lower average pair F1 than BCE. **ASL seed 2** is an interesting validation trade-off: pair F1 0.24638 and precision 24.40%, with mAP 59.2169%. It misses the baseline mAP gate by only 0.0368 percentage points. It was not selected or test-evaluated; changing the criterion after seeing test results would be a new exploratory decision, not this run's prespecified result.

The selected model's pooled validation pair F1 rises only from 0.14488 to 0.14700, while pair F1 decreases in each validation subject separately: p04 from 0.21809 to 0.18582, and p15 from 0.03861 to 0. The aggregate score reflects a change in the distribution of predictions across subjects. It does not show uniform improvement. Validation contains only 197 positive pair windows for p04 and eight for p15.

## Per-class test results

Changes below are for the full package of architecture, schedule and loss changes; this experiment does not isolate the effect of temporal context alone.

| Activity | Previous AP | New AP | AP change (pp) | Previous F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| Walking | 0.83754 | 0.88947 | +5.19 | 0.75731 | 0.80210 |
| Lying down | 0.73875 | 0.75313 | +1.44 | 0.58621 | 0.68142 |
| Standing up | 0.83974 | 0.87340 | +3.37 | 0.79893 | 0.82415 |
| Sitting down | 0.79098 | 0.81933 | +2.84 | 0.73729 | 0.77868 |
| Eating | 0.73457 | 0.60419 | -13.04 | 0.65352 | 0.59428 |
| Drinking | 0.60565 | 0.68416 | +7.85 | 0.54311 | 0.58360 |
| Cooking / food preparation | 0.70754 | 0.73755 | +3.00 | 0.65189 | 0.64596 |
| Taking medication | 0.16593 | 0.13207 | -3.39 | 0.25154 | 0.21503 |
| Watching TV | 0.82465 | 0.78205 | -4.26 | 0.72789 | 0.71128 |
| Reading | 0.72221 | 0.73740 | +1.52 | 0.73839 | 0.74921 |
| Using a phone | 0.44511 | 0.42782 | -1.73 | 0.39028 | 0.41012 |
| Cleaning / housework | 0.76191 | 0.68538 | -7.65 | 0.70585 | 0.63317 |
| Personal hygiene | 0.28568 | 0.25746 | -2.82 | 0.26735 | 0.30474 |
| Using a device | 0.36097 | 0.41163 | +5.07 | 0.33117 | 0.33971 |
| Object interaction | 0.56980 | 0.58795 | +1.81 | 0.55391 | 0.56136 |

Drinking and walking improve, while eating, cleaning and watching TV lose AP. Those regressions offset the gains in the overall average.

## Variation across test subjects

| Subject | Windows | Positive TV/drinking windows | Pair true positives | Pair precision | Pair recall | Pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| p02 | 16,931 | 42 | 10 | 0.03236 | 0.23810 | 0.05698 |
| p10 | 16,354 | 394 | 229 | 0.34540 | 0.58122 | 0.43330 |
| p11 | 14,845 | 19 | 16 | 0.21918 | 0.84211 | 0.34783 |
| p14 | 29,179 | 76 | 14 | 0.01224 | 0.18421 | 0.02295 |
| p16 | 25,916 | 30 | 11 | 0.03691 | 0.36667 | 0.06707 |
| p18 | 28,353 | 16 | 13 | 0.02500 | 0.81250 | 0.04851 |
| p20 | 21,959 | 119 | 72 | 0.18367 | 0.60504 | 0.28180 |

Pair F1 improves in five of seven test subjects, but worsens for p02 and p14. The net 44 additional true positives comes from p10 (+26) and p20 (+18); changes in the other subjects cancel. The subject variation and false positives matter more for practical use than the modest aggregate gain alone.

## Verification and limits

- The receipt records exactly one accepted version: **1**, kernel ID **138168743**. The source archive SHA-256 is `9082c6ea5ada02e66a6f5905362eca1b68e5f821ccd56d3c31c24582937c5f65`.
- All six downloaded checkpoints, threshold files, validation metrics and prediction arrays match their completion-receipt hashes. Histories, selected epochs, convergence evidence and full-data feature inventories agree.
- Selected checkpoint SHA-256: `8a4bc8f7052d552cf6ce9d539d7b68a0cfb18c6b39ac6ccc3dfdbbb323a4c49c`.
- Local CPU replay reproduces aggregate and per-subject validation/test metrics within **1e-12**, using the saved thresholds. Annotation replay reconstructs the center-window targets and sampled annotation overlap from the original annotations and source-frame clocks. No threshold refitting, model inference or training was performed in this audit.
- Coverage remains retained, annotation-eligible, pose-visible windows. Overlapping windows are correlated; results are not full-recording temporal detection or independent-event statistics.
- The test benchmark was already inspected before this experiment, so the comparison is **exploratory**. The model uses **4.5 seconds of future context** and is not an instantaneous live predictor.
- The run does not replace the serving model. No additional training session was launched while completing this audit.

The evidence supports a modest F1/concurrency gain, not a broad upgrade. The next substantive development question is whether richer RGB action evidence and more diverse concurrent examples can improve precision consistently across people. A follow-up should define the mAP/pair trade-off and subject-level checks before inspecting further benchmark results.

## Saved artifacts

- [Completed Kaggle run](https://www.kaggle.com/code/krishnagupta02468/bs-temporal-multilabel-20261011)
- [Full test metrics](../artifacts/temporal-multilabel-20261011/output/test/test_metrics.json)
- [Selected checkpoint](../artifacts/temporal-multilabel-20261011/output/candidates/asl-seed1/best_multilabel.pt)
- [Metric replay audit](../artifacts/temporal-multilabel-20261011/metric-replay-20261012.json)
- [Reproducible audit script](../artifacts/temporal-multilabel-20261011/metric_replay_20261012.py)
- [Frozen selection](../artifacts/temporal-multilabel-20261011/output/frozen_selection.json)
- [All-candidate session result](../artifacts/temporal-multilabel-20261011/output/stage_result.json)
- [Research and committed protocol](../docs/10_temporal_multilabel_research.md)
