# Fixed equal-probability ensemble — 12 October 2026

A single, prespecified **50/50 probability average** of the original selected context model (seed 2) and the selected temporal model (ASL, seed 1) improves test mAP to **65.29%**. It uses existing predictions and required no GPU inference or training. Concurrent TV/drinking precision and F1 improve, but recall falls and detection remains unreliable.

| Test metric | Original context | Temporal ASL | Equal ensemble |
|---|---:|---:|---:|
| mAP | 62.61% | 62.55% | **65.29%** |
| Macro F1 | 0.57964 | 0.58899 | **0.61092** |
| Micro F1 | 0.65210 | 0.65695 | **0.66902** |
| TV/drinking F1 | 0.15666 | 0.17827 | **0.19074** |
| TV/drinking precision | 9.44% | 10.74% | **12.59%** |
| TV/drinking recall | 46.12% | **52.44%** | 39.37% |

The ensemble gains **2.68 percentage points of test mAP** over the original model and **2.74 points** over the temporal model. Its mAP exceeds both members on each of the seven test subjects, although the improvement over the better member is very small for p11. Per-subject results are diagnostic; they are not seven independent experiments.

There are **274 true positives among 2,177 concurrent predictions**, against 696 ground-truth concurrent windows. The higher precision comes with a **13.07 percentage-point recall reduction** from the temporal model. These remain window co-label metrics, not proof of frame-level simultaneous action detection.

The recipe and input hashes were recorded before scoring. Exactly one candidate was evaluated, with weights fixed at 0.5 and no weight search. Per-class thresholds were fitted using the same validation-only F1 procedure as the original models, requiring at least five positive and negative validation windows and fitting-class support. Validation mAP reached **61.79%**, versus 59.25% and 60.64% for the members. The gain over the stronger member was **1.15 points**, passing the prespecified minimum of one point. Pair F1 reached 0.15603 and precision 15.14%, also passing the requirement that neither decline from either member. The fitted thresholds and passing gate were saved before any test arrays were deserialized; the frozen ensemble was then evaluated once.

Both members' rows, subjects, targets and known-label masks match on all **63,220 validation windows** and **153,537 test windows from 185 videos**. Individual metrics reproduce their saved reports. The audit includes all per-class and per-subject comparisons.

This is an exploratory comparison on an already inspected test benchmark. Validation thresholds are fitted and scored on the same two subjects. The ensemble adds no RGB information, retains the temporal model's **4.5-second lookahead**, and requires both checkpoints. It is a useful complementarity result, not a substitute for fixing pose preprocessing and retaining interaction-relevant visual information. No serving model was changed.

Evidence: [full audit](../artifacts/pipeline-rebuild-20261012/equal_ensemble_audit.json), [predeclared recipe and input hashes](../artifacts/pipeline-rebuild-20261012/equal_ensemble_recipe.json), [thresholds and validation gate frozen before testing](../artifacts/pipeline-rebuild-20261012/equal_ensemble_validation_freeze.json).
