# RGB throughput correction — 13 October 2026

The 12 October run stopped because both encoders missed the time forecast. The
13 October experiments changed CPU crop construction and overlapped preparation
with inference. Throughput now passes; the completed pilot failed its quality
gate. See the [verified pilot outcome](../results/rgb_pilot_20261013.md).
The first experiment changed CPU crop construction and instrumented its actual cost.
It preserves the pretrained weights, 32-frame sampling, two views, input windows,
validation subjects and quality requirements.

The original spatial implementation filled a square image with an RGB vector
through NumPy, then copied the image content into that square. The replacement
uses native OpenCV constant padding; fully contained person crops can be resized
directly. The same OpenCV uint8 bilinear resize and center crop are used afterwards.
A retained reference implementation permits a complete comparison inside the job.

Local tests require exact pixel equality across boundary, odd-size, empty,
non-contiguous and reversed input cases. A complete 797-window fitting video also
produced identical hashes for every encoder input batch. Its measured crop time
fell from 165.35 to 44.97 seconds on the local CPU, though full processing improved
less and local timings varied. These numbers are not a GPU throughput forecast.

## Measured execution

[The padding-only job](https://www.kaggle.com/code/krishnagupta02468/bs-rgb-video-20261013), version 1,
completed on the RTX PRO 6000 Blackwell with the pinned CUDA runtime. It stopped
at the throughput forecast after 228.564 seconds. Native padding achieved 2.30×
measured speedup, but its 85.27-minute pilot forecast exceeded the 75-minute cap.

The explicit `native-border-profile.v2` protocol uses Tiny only, since Base failed
the prior forecast. OpenCV is fixed to one CPU thread. Each of the same three
prescribed fitting videos is fully processed by the reference and native crop
implementations, alternating their order. CPU and synchronized CUDA timers record
decoding, cropping, packing, transfer, preprocessing, forward pass and readback.

Before optimized reports can feed the throughput gate, all row/clock/crop arrays
must match exactly and float16 embeddings must match within rtol=atol=0.001.
Only the optimized caches can be reused for subsequent extraction. All comparison
time is charged to the original 900-second probe and 4500-second pilot budgets.
The 40,500-second internal session limit and all validation gates are unchanged.
No continuation job or split training is enabled.

## Preflight evidence

157 focused checks passed, with one Linux-only process cleanup check skipped on
Windows. The 83-file frozen archive passed isolated imports and trainer command
checks. Both private datasets were rechecked before the one accepted submission.

Frozen source SHA-256: `c8ccd3853a8084f3a7808c7e4c2a2490bbc0581ac1d5f72bed4f9f424ac945e8`.
Private evidence: `artifacts/rgb-throughput-20261013/` and its `run/` directory.

The subsequent `native-prefetch-profile.v3` job added a bounded two-batch queue
and one inference worker. Its same-video comparison achieved another 1.64×
aggregate speedup with bit-identical embeddings. It passed throughput and
completed all 113 pilot videos plus both heads in 33 minutes 37 seconds.
RGB validation mAP was 48.11% versus 40.69% for the matched control; pair recall
was 6.34%. The original quality gate rejected it before full training or testing.

At that stage, the previous 65.29% test-mAP ensemble was the stronger saved reference, with
poor pair precision. This subset pilot did not establish a replacement.

The subsequent [fixed 75/25 probability blend](../results/rgb_fusion_20261013.md)
raised validation mAP from 61.79% to 65.71%, but pair precision, recall and F1
still missed their 30% floors. A [CPU four-state pair head](../results/rgb_pair_head_20261013.md)
then failed its subject-separated fitting gate at 20.00% precision, 13.88%
recall and 16.39% F1. It stopped before full fitting or outer-validation
inference. These completed follow-ups used cached features and predictions;
they do not provide a new test score or a serving-model replacement.


The padding-only run has since completed and stopped at its throughput gate. The measured results and bounded overlap follow-up are recorded in [the throughput report](../results/rgb_throughput_20261013.md).

On 15 October, the separately frozen expanded-RGB hybrid completed its test
inference and independent audit: mAP rose **65.29% → 69.21%** and pair F1 rose
**19.07% → 28.12%** across all 153,537 retained test windows. Pair precision
(17.84%) and F1 still fail their 30% requirements. The benchmark was previously
inspected, and no serving model was replaced. See the
[complete result and tradeoffs](../results/rgb_improvement_20261015.md).
