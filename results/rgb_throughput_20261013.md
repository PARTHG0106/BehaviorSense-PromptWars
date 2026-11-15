# RGB throughput work — 13 October 2026

**The overlap follow-up has completed: throughput passed, but the RGB pilot failed
its quality gate after 33 minutes 37 seconds.** Saved predictions reproduce
48.11% validation mAP and 6.34% TV/drinking recall. See the
[verified pilot outcome](rgb_pilot_20261013.md) for the complete result and audit.

**Native OpenCV padding made extraction 2.30× faster on the RTX PRO 6000, with
bit-identical embeddings for all 1,915 comparison windows.** The CPU change kept
the same pretrained weights, pixels, source clocks, labels and fitting/validation
split. Downloaded feature files independently passed their hashes and equality
checks; no model inference was repeated during that audit.

## Verified padding comparison

| Fitting video | Original padding | Native padding | Speedup |
|---|---:|---:|---:|
| P03T03C04 | 60.41s | 24.74s | 2.44× |
| P25T06C05 | 59.49s | 27.21s | 2.19× |
| P07T06C05 | 21.90s | 9.68s | 2.26× |

This bounded job ended after 228.564 seconds. The conservative full-job forecast
improved to **6.75 hours**, within the 12-hour allocation. The pilot still forecast
**85.27 minutes**, above its 75-minute limit, so no head training or test evaluation
was allowed.

The profile identified what remained: GPU forward inference consumed roughly
49–58% of optimized extraction wall time. CPU preparation and GPU inference were
still sequential. These timings support overlapping those operations without
changing the representation.

## Bounded overlap experiment

A single inference worker now runs the existing model while the main thread
continues decoding/cropping. At most two inference futures are outstanding;
the current CPU batch adds bounded preparation memory. The model is never used
concurrently by multiple workers. Failed or unfinished batches cannot publish a
complete feature video, and the Linux parent can stop all workers at the deadline.

The new explicit `native-prefetch-profile.v3` job compares native sequential
extraction against native extraction with the two-batch queue on the same three
complete fitting videos. It requires matching row/clock/crop arrays and embeddings
before applying the original throughput forecast. Its optimized caches are reused
only after verification. All comparison time counts toward the 900-second probe
and 4500-second pilot. Quality gates and the 40,500-second internal session cap
remain unchanged. Any complete training/evaluation remains within this single job.

Completed job: [version 1](https://www.kaggle.com/code/krishnagupta02468/bs-rgb-prefetch-20261013).
The overlap comparison produced a further 1.64× aggregate speedup with bit-identical
embeddings across all 1,915 paired windows. Its conservative forecasts were
55.71 minutes for the pilot and 4.57 hours for the full job. The measured pilot
completed all 113 videos and both heads, then stopped at `quality_rejected`.

165 focused checks passed before this submission; one Linux-specific process-tree
cleanup test was skipped on the Windows host. Tests cover producer/inference
overlap, exact pixel batches and output arrays, single-worker access, bounded
queues, failure/timeout cleanup, cached-feature identity, gates and frozen export.
The 83-file archive also passed isolated import and command checks.

The previous 65.29% test-mAP ensemble remains the stronger saved accuracy reference.
Its low pair precision remains unresolved. The RGB pilot improved over its matched
control but failed overall mAP, pair recall/F1 and several subject requirements.
Full-data training and test evaluation did not proceed.

Private receipts, logs and comparisons are under
`artifacts/rgb-throughput-20261013/{run,prefetch-run}/`.
