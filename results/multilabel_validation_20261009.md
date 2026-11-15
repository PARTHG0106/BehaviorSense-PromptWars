# Full-data multilabel validation — 9 October 2026

All six prespecified candidates completed training and met the experiment's validation-plateau stopping rule. **Context fusion, seed 2** was selected by the highest validation macro average precision (mAP): **0.59254**. Across three seeds, context fusion improved mean validation mAP by **3.00 percentage points** over the skeleton baseline. These are validation results; the completed official test evaluation is reported [separately](multilabel_test_20261011.md).

Selection is recorded in [frozen_selection.json](../artifacts/training-tv-20261007/full-run/frozen_selection.json). On 9 October, the subsequent `xte00` submission was rejected by Kaggle: **“Maximum weekly GPU quota of 30.00 hours reached.”** No job was created by that rejected submission.

**Completion verified, 11 October 2026:** test extraction resumed after quota refreshed on 10 October. All 185 test videos have now been processed, and the frozen checkpoint was evaluated on 153,537 retained windows: **test mAP 0.62607**, macro F1 0.57964 and micro F1 0.65210. The downloaded predictions reproduced these metrics using the saved validation thresholds. See the [official test report](multilabel_test_20261011.md). The validation results below are unchanged, and the serving model has not been replaced.

## Candidates and convergence

Both architectures were trained from scratch on the same extracted windows, using seeds 0, 1 and 2. Context fusion adds frozen RT-DETR object scores and person-relative geometry to the skeleton backbone. All scores below range from 0 to 1; AP and F1 are not classification accuracy. Best epochs are one-based.

| Model | Seed | Validation mAP | Macro F1 | Micro F1 | Best epoch | Completed epochs | LR reductions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Skeleton | 0 | 0.55185 | 0.56013 | 0.68305 | 31 | 56 | 3 |
| Skeleton | 1 | 0.56213 | 0.55983 | 0.68014 | 24 | 49 | 2 |
| Skeleton | 2 | 0.56378 | 0.55929 | 0.66728 | 22 | 47 | 2 |
| Context fusion | 0 | 0.58383 | 0.58202 | 0.68959 | 19 | 44 | 2 |
| Context fusion | 1 | 0.59139 | 0.58277 | 0.70363 | 18 | 43 | 2 |
| **Context fusion** | **2** | **0.59254** | **0.58010** | **0.70535** | **15** | **40** | **2** |

Every candidate completed at least 40 epochs, had 25 epochs without improvement in validation mAP, completed at least two learning-rate reductions, and had at least five epochs to settle after the last reduction. None exhausted its 150-epoch budget. This establishes the prespecified stopping condition; it does not establish a globally optimal model.

The downloaded histories, completion statuses and validation metrics agree on each candidate's epoch count, best epoch and best score. Training loss, epoch duration, learning rate and validation mAP are finite throughout. All six inference checkpoints were downloaded and their SHA-256 hashes verified against the corresponding stage reports. The frozen trainer, data loader and evaluation implementation match the local source files.

| Architecture | Mean validation mAP | Population SD across three seeds | Best validation mAP |
|---|---:|---:|---:|
| Skeleton | 0.55925 | 0.00528 | 0.56378 |
| Context fusion | 0.58925 | 0.00386 | 0.59254 |

Every context candidate outperformed every skeleton candidate on overall validation mAP. The selected context candidate exceeds the best skeleton candidate by 2.88 percentage points. The three-seed standard deviations describe training-seed variation on one validation split; they are not confidence intervals or evidence of statistical significance across people.

## Selected artifact and per-class results

Selected checkpoint: [context fusion, seed 2 — best_multilabel.pt](../artifacts/training-tv-20261007/completion-audit/bs-mlfull-20261007-ctx-all-a00/candidates/seed2/best_multilabel.pt).

Verified SHA-256: `32f7c33e0234b0931fb28b26144fef9ce3b9e3a096c50d1112009b901a7ef384`.

The selection criterion was highest overall validation mAP across the six prespecified candidates. It did not optimize TV/drinking concurrency or choose a different model for each activity. The 15 supervised coarse classes are listed below; support counts positive retained validation windows, not independent events.

| Supported activity | Positive windows | AP | F1 |
|---|---:|---:|---:|
| Walking | 4,330 | 0.82334 | 0.77380 |
| Lying down | 43 | 0.79410 | 0.72222 |
| Standing up | 355 | 0.81076 | 0.78830 |
| Sitting down | 396 | 0.77614 | 0.76283 |
| Eating | 3,588 | 0.60369 | 0.59175 |
| Drinking | 1,582 | 0.50857 | 0.53819 |
| Cooking / food preparation | 5,520 | 0.65312 | 0.58352 |
| Taking medication | 361 | 0.13013 | 0.20803 |
| Watching TV | 2,990 | 0.60205 | 0.54821 |
| Reading | 32,285 | 0.90416 | 0.85301 |
| Using a phone | 5,128 | 0.36475 | 0.38390 |
| Cleaning / housework | 4,272 | 0.81570 | 0.73733 |
| Personal hygiene | 74 | 0.16438 | 0.22222 |
| Using a device | 3,003 | 0.34406 | 0.42549 |
| Object interaction | 4,110 | 0.59311 | 0.56272 |

Standing, sitting, bending/reaching, falling, fallen on ground, interacting with a person and other/idle lack direct supervision in this experiment. They are masked and excluded from the reported 15-class averages. Low support, especially 43 lying-down and 74 personal-hygiene windows, limits how confidently those individual scores can be generalized.

## TV and drinking together

Context improves the three-seed mean AP for TV from 0.54651 to 0.63284 and for drinking from 0.47287 to 0.51745. However, simultaneous detection remains weak. For the selected checkpoint:

| Window-level diagnostic | Ground-truth positives | Predicted positives | True positives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Both activity labels present in the window | 205 | 430 | 46 | 0.10698 | 0.22439 | 0.14488 |
| Annotations overlap at one or more sampled positions | 272 | 430 | 50 | 0.11628 | 0.18382 | Not reported |

The first row measures co-label detection after applying the per-class validation thresholds. The second uses actual temporal annotation overlap as a diagnostic reference, but the predictions still describe whole windows. Neither is frame-level temporal detection. These results do not establish reliable simultaneous TV watching and drinking.

## Evaluation scope and limitations

All 351 official training videos were extracted, producing 280,310 retained windows: 217,090 fitting windows from 274 videos and 63,220 validation windows from 77 videos. Fitting subjects were `p03, p06, p07, p09, p12, p13, p17, p19, p25`; validation subjects were `p04, p15`. All candidates report full eligible-window coverage and the same validation comparison fingerprint, `89fbef7f6b41ee4a36268426e243cd85dbdc319608e6f267f8264b003e77e510`. No test data was loaded during training or consulted for model selection.

Coverage applies to annotation-eligible windows retained after pose-visibility filtering. The experiment samples 30-frame windows at 20 Hz with a 15-frame stride and requires at least 50% annotation coverage. It does not measure performance on every frame, every person or all unannotated periods. Overlapping windows are correlated, and large window counts do not create an equally large number of independent observations.

Validation subjects select the best epoch, per-class F1 thresholds and final architecture/seed. AP is threshold-independent, but it still guides selection on this split. F1 and concurrency diagnostics reuse thresholds fitted on these same validation data, so they are development-set results and may be optimistic. The two validation subjects and three seeds provide limited evidence about broader subject and scene variation. The completed official test evaluation used the frozen checkpoint and thresholds without refitting or selecting on test results.

Source evidence: [skeleton stage report](../artifacts/training-tv-20261007/completion-audit/bs-mlfull-20261007-sk-all-a00/stage_result.json), [context stage report](../artifacts/training-tv-20261007/completion-audit/bs-mlfull-20261007-ctx-all-a00/stage_result.json), each candidate's adjacent `history.json` and `training_status.json`, [selected-model validation metrics](../artifacts/training-tv-20261007/completion-audit/bs-mlfull-20261007-ctx-all-a00/candidates/seed2/validation_metrics.json), and the [execution plan](../artifacts/training-tv-20261007/full-run/plan.json).
