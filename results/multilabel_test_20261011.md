# Official multilabel test results — verified 11 October 2026

**Official testing is complete.** Kaggle reports all four extraction sessions and the final evaluation as `COMPLETE`. The validation-selected **context fusion, seed 2** checkpoint was evaluated on all **185 official test videos**, producing **153,537 retained windows** from seven held-out subjects.

| Metric | Official test result |
|---|---:|
| Macro average precision (mAP) | 0.62607 (62.61%) |
| Macro F1 | 0.57964 (57.96%) |
| Micro F1 | 0.65210 (65.21%) |
| TV + drinking co-label F1 | 0.15666 |

AP and F1 are not classification accuracy. These metrics cover the 15 supervised coarse activities on retained, annotation-eligible windows. The validation mAP of 0.59254 and test mAP of 0.62607 describe different subjects and activity distributions; their difference is not an estimate of model improvement. Only the validation-selected checkpoint was evaluated on test, so this run does not provide a test-set comparison against the skeleton baseline.

## Frozen selection and verification

The selected checkpoint SHA-256 is `32f7c33e0234b0931fb28b26144fef9ce3b9e3a096c50d1112009b901a7ef384`. It matches the frozen local selection, the downloaded test selection and the evaluation report. Epoch, model, seed and per-class thresholds were selected on validation; thresholds were not refitted and the checkpoint was not selected on test.

The test subjects are `p02, p10, p11, p14, p16, p18, p20` and are disjoint from fitting and validation subjects. The downloaded source and annotation identities agree with the frozen experiment, and the test visual-context contract agrees with the selected checkpoint. All prediction rows match the retained window starts in the 185 extraction records.

The downloaded logits were scored locally with the checkpoint's frozen validation thresholds. Overall AP/F1, all per-class scores and counts, and the TV/drinking co-label metrics agreed with the saved results (absolute comparison tolerance: 0.000001). The actual temporal-overlap row below is the remote annotation diagnostic; it was not independently recomputed in this local replay.

The local coordinator completed and exited. The serving model has not been replaced.

## Per-class results

Positive support counts retained windows, not independent activity events. Unsupported classes remain masked and are excluded from the averages.

| Supported activity | Positive windows | AP | F1 |
|---|---:|---:|---:|
| Walking | 6,618 | 0.83754 | 0.75731 |
| Lying down | 276 | 0.73875 | 0.58621 |
| Standing up | 1,373 | 0.83974 | 0.79893 |
| Sitting down | 1,433 | 0.79098 | 0.73729 |
| Eating | 8,356 | 0.73457 | 0.65352 |
| Drinking | 5,331 | 0.60565 | 0.54311 |
| Cooking / food preparation | 11,829 | 0.70754 | 0.65189 |
| Taking medication | 1,305 | 0.16593 | 0.25154 |
| Watching TV | 20,150 | 0.82465 | 0.72789 |
| Reading | 62,254 | 0.72221 | 0.73839 |
| Using a phone | 4,302 | 0.44511 | 0.39028 |
| Cleaning / housework | 5,924 | 0.76191 | 0.70585 |
| Personal hygiene | 283 | 0.28568 | 0.26735 |
| Using a device | 23,089 | 0.36097 | 0.33117 |
| Object interaction | 7,741 | 0.56980 | 0.55391 |

## TV and drinking together

| Window diagnostic | Ground-truth positives | Predicted positives | True positives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Both activity labels present | 696 | 3,402 | 321 | 0.09436 | 0.46121 | 0.15666 |
| Annotations overlap at sampled positions | 928 | 3,402 | 361 | 0.10611 | 0.38901 | Not reported |

Co-label detection remains weak: only **9.44%** of predicted TV/drinking pairs were correct, with **46.12%** recall. Watching TV and drinking individually have AP of 0.82465 and 0.60565, but those individual results do not establish reliable simultaneous detection. Both diagnostics use whole-window predictions; neither measures frame-level temporal detection.

## Extraction coverage

| Stage | Videos | Retained windows | Runtime seconds |
|---|---:|---:|---:|
| `xte00` | 46 | 40,285 | 4,865.68 |
| `xte01` | 47 | 33,177 | 4,619.66 |
| `xte02` | 46 | 38,112 | 4,715.12 |
| `xte03` | 46 | 41,963 | 5,080.88 |
| **Total** | **185** | **153,537** | **19,281.34** |

The source sampling grid contained 251,977 candidate windows. Annotation coverage excluded 69,346, leaving 182,631 eligible windows. Pose visibility excluded another 29,094; **153,537 (84.07% of eligible windows)** were retained and evaluated. Every expected test video was processed, and no per-video window cap was applied.

The experiment uses 30-frame windows sampled at 20 Hz with a 15-frame stride and at least 50% annotation coverage. Overlapping windows are correlated. These are retained-window results, not full-recording temporal coverage or population activity prevalence. Seven test subjects in the Toyota Smarthome setting do not establish reliability across other homes or deployments.

## Saved evidence

- [Final Kaggle evaluation](https://www.kaggle.com/code/krishnagupta02468/bs-mlfull-20261007-test-a00)
- [Full test metrics](../artifacts/training-tv-20261007/full-run/bs-mlfull-20261007-test-a00/output/test/test_metrics.json)
- [Saved test logits and targets](../artifacts/training-tv-20261007/full-run/bs-mlfull-20261007-test-a00/output/test/test_logits.npz)
- [Verification and artifact checksums](../artifacts/training-tv-20261007/completion-audit/test-verification-20261011.json)
- [Five remote completion checks](../artifacts/training-tv-20261007/completion-audit/test-remote-status-20261011.json)
- [Completed execution plan](../artifacts/training-tv-20261007/full-run/plan.json)
- [Validation comparison across six candidates](multilabel_validation_20261009.md)
