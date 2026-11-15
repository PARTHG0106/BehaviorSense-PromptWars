# RGB fitting coverage comparison — 16 October 2026

**Audited fitting-only comparison: mAP 41.44% → 58.25%; TV-plus-drinking F1 4.53% → 13.88%.**
The full-coverage arm does not pass every frozen research advancement check.

Both arms are evaluated on the same **217,090 retained windows from 274 videos and nine fitting subjects**. Each model trains on three subjects, calibrates on three different subjects and evaluates on the remaining three. Across three rotations, each subject is evaluated once per arm. This is a coverage comparison, not a final nine-subject model or a new test benchmark.

## Aggregate result

| Metric | Cached fitting coverage | Full fitting coverage | Change |
|---|---:|---:|---:|
| mAP | 41.44% | 58.25% | +16.81 pp |
| Macro F1 | 41.55% | 56.68% | +15.13 pp |
| Micro F1 | 44.48% | 66.57% | +22.09 pp |
| TV + drinking precision | 5.94% | 16.83% | +10.90 pp |
| TV + drinking recall | 3.66% | 11.81% | +8.16 pp |
| TV + drinking f1 | 4.53% | 13.88% | +9.36 pp |

Pair support is 711 windows. Pair predictions change **438 → 499**, true positives **26 → 84**, false positives **412 → 415**, and false negatives **685 → 627**.

| Pair false-positive truth state | Cached coverage | Full coverage |
|---|---:|---:|
| Neither activity | 123 | 87 |
| Drinking only | 262 | 211 |
| TV only | 27 | 117 |

## Frozen advancement checks

| Check | Passed |
|---|---|
| mAP noninferior to cached coverage | Yes |
| Pair F1 strictly improves | Yes |
| Pair precision noninferior | Yes |
| Pair precision ≥30% | No |
| Pair recall ≥30% | No |
| Pair F1 ≥30% | No |

Advancement authorizes further research. No serving model was replaced.

## Rotations

| Evaluation group | Cached mAP | Full mAP | Cached pair P / R / F1 | Full pair P / R / F1 | Cached / full fitting windows |
|---|---:|---:|---|---|---:|
| A | 44.12% | 61.54% | 2.01% / 4.14% / 2.71% | 7.81% / 13.79% / 9.98% | 10,670 / 73,882 |
| B | 42.66% | 58.62% | 12.07% / 1.59% / 2.81% | 40.00% / 11.82% / 18.25% | 9,107 / 70,827 |
| C | 45.64% | 58.62% | 15.85% / 10.32% / 12.50% | 10.62% / 9.52% / 10.04% | 20,416 / 72,381 |

mAP improves for **9 of nine subjects**. Pooled AP ranks scores from three separately fitted heads; rotation and subject results should be read alongside the aggregate.

## Subjects

| Subject | Cached mAP | Full mAP | Cached pair F1 | Full pair F1 | Full pair precision | Full pair recall |
|---|---:|---:|---:|---:|---:|---:|
| p03 | 41.03% | 54.14% | 0.80% | 5.88% | 3.21% | 35.29% |
| p06 | 42.73% | 64.21% | 5.98% | 27.72% | 53.85% | 18.67% |
| p07 | 52.32% | 72.95% | 9.90% | 10.31% | 55.56% | 5.68% |
| p09 | 55.42% | 65.19% | 21.43% | 15.58% | 40.00% | 9.68% |
| p12 | 45.79% | 60.19% | 0.00% | 11.17% | 19.61% | 7.81% |
| p13 | 39.52% | 56.55% | 15.69% | 13.04% | 13.04% | 13.04% |
| p17 | 51.06% | 65.60% | 0.00% | 5.17% | 4.00% | 7.32% |
| p19 | 47.51% | 61.20% | 0.00% | 0.00% | 0.00% | 0.00% |
| p25 | 50.81% | 70.44% | 0.00% | 18.00% | 15.00% | 22.50% |

## Scientific settings and limits

The comparison fits the original **60 videos / 40,193 windows** or all **274 videos / 217,090 windows** from the same fitting subjects. Both arms have complete, identical calibration and evaluation cohorts. The extra 176,897 windows add mostly negative and other-activity diversity. The 73 extra pair positives are alternate views of already represented positive recordings.

The training code is byte-identical to the initial protocol: fixed Swin3D-T scene/person encoder, 3,025-feature head, 256 hidden units, dropout 0.3, seed 0, AdamW learning rate and weight decay 0.001, batch size 2,048, minimum 10 epochs, maximum 80 and patience 12. Each arm fits its normalization and class weights on its own fitting rows. Calibration subjects select the epoch and class thresholds. No historical supervised model is blended into this study.

These internal groups have guided earlier research; the results remain exploratory. Ranking scores, window decisions and pair co-labels are distinct endpoints. This study does not establish unseen-population or deployment performance.

## Execution and verification

The complete run finished in **67.88 minutes** within its six-hour worker limit. All six fits converged. The independent replay checked the private version/source identity, row and subject coverage, identical initial parameters, learning-rate schedules, calibration thresholds, per-rotation and pooled metrics, and every advancement decision. It checked stored normalization statistics and fitting-row provenance; it did not replay training or normalization arithmetic from the complete feature matrix.

The original local auditor encountered Windows/Linux rounding differences. An additive portable replay reconstructed all six initial parameter hashes and every exported probability exactly, preserving the original auditor and all metric/acceptance rules. Its five regression checks passed. [Audit portability correction](../artifacts/rgb-next-20261016/coverage-study-r2/AUDIT_PORTABILITY.md).

The earlier four-hour attempt stopped after its three probes forecast 5.51 hours, before any head was trained or scored. A copied notebook title then caused a rejected six-hour submission; absence checks verified that it created no new job. Both rejected packets and their receipts remain preserved. The accepted run is version 1 of `krishnagupta02468/bs-rgb-fit-coverage-20261016`.

RGB feature caches are retained in the remote output. The combined feature matrix was temporary and was not exported.

- [Frozen protocol](../artifacts/rgb-next-20261016/coverage-study-r2/protocol.json)
- [Audited comparison](../artifacts/rgb-next-20261016/coverage-study-r2/run/output/study_result.json)
- [Independent replay](../artifacts/rgb-next-20261016/coverage-study-r2/independent_audit.json)
- [Full cohort provenance](../artifacts/rgb-next-20261016/coverage-study-r2/run/output/cohort_manifest.json)
- [Continuation state](../artifacts/rgb-next-20261016/coverage-study-r2/continuation_status.json)

The [RGB hybrid benchmark](rgb_improvement_20261015.md) remains the latest verified test result. Official validation and test arrays did not enter this study.
