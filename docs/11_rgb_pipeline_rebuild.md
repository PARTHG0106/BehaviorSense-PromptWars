# RGB pipeline rebuild — 12 October 2026

The six-hour temporal run did not deliver a meaningful overall test mAP gain.
The next approach starts with input correctness, a stronger inexpensive reference,
and pretrained RGB action features. Additional GPU work must pass an early quality
gate before expanding to the full dataset.

## Work completed before further GPU use

A single predeclared 50/50 probability ensemble of the existing context model and
temporal model passed validation gates. Its recipe and validation thresholds were
frozen before its one test replay. It achieved **65.29% test mAP**, macro F1
**61.09%**, and micro F1 **66.90%**, using saved predictions and no new GPU work.
Every test subject's mAP exceeded both individual models. Pair F1 is **0.19074**,
precision **12.59%**, and recall **39.37%**. The recall trade-off and low precision
mean this is not yet reliable concurrent detection. This is an exploratory result
on the already inspected benchmark. See the [ensemble audit](../results/equal_ensemble_20261012.md).

The input audit reproduced four deterministic problems:

| Finding | Correction |
|---|---|
| Directed graph normalization erased 10 of 34 anatomical edges, including wrist-to-elbow edges. | New directed-column normalization preserves all 34 edges. |
| Bone conversion subtracted confidence and created geometry through missing joints. | New bone conversion requires both endpoints and uses their minimum confidence. |
| Invisible hips/shoulders changed visible-joint normalization. | New normalization uses reliable landmarks and explicit fallbacks only. |
| Rejected distant track matches could immediately refill occupied slots. | Future extraction leaves those slots blank for the intended transition. |

The corrected pose functions use the explicit `behaviorsense.pose-visible.v2`
contract. Legacy normalization, bone conversion and adjacency defaults remain
available for existing checkpoints; silently changing their inputs would invalidate
their recorded results. The new RGB head's pose control uses v2. Stored tracking
switches cannot be repaired by changing the future extractor.

The detector feature vector also discards all but the highest-scoring object of each
class. Adding a held cup beside the actor's wrist/head changed none of its 401
features when a table cup had a higher score. Temporal modeling cannot recover an
object that was removed from its input. The estimated accuracy effect of each code
defect remains unmeasured; their reproduction does not establish that they caused
all errors.

## Real validation-clip review

Nine fixed validation windows were selected by median confidence within failure
category and subject. Seven source videos were downloaded and SHA-256 checked
against the original extraction metadata. Three source frames per window were
reviewed alongside the original annotations; no labels were changed.

- `P04T04C05`, 949.5 seconds: annotated telephone use, with a small object by the
  side of the head. The earlier model predicted both TV and drinking.
- `P15T01C05`, 128.25 seconds: annotated drinking, with a dark TV screen. Object
  presence alone does not distinguish this from TV watching.
- `P04T05C05`, 589.5 seconds: annotated drinking while the person is oriented away
  from the playing television.
- `P04T03C04`, 14.25 seconds: annotated TV watching, hands near the lap and drinks
  on the table. The model incorrectly added drinking.
- The missed-pair examples contain small or partly occluded hand-to-mouth evidence.
  A larger person view can retain details lost in a full-room representation.

These are diagnostic examples, not a representative accuracy estimate. Window
labels aggregate 30 samples; a class positive for the window need not be active
at its exact center frame. The earlier label/clock replay and zero-lag diagnostic
did not indicate a global timestamp shift.

Private evidence lives under `artifacts/pipeline-rebuild-20261012/`, including the
input audit, clip selection plan, verified videos, contact sheets and a successful
CPU forward pass through the actual official Swin-T weights. That forward verifies
the weight/preprocessing interface; it is not an RGB accuracy result.

## Materially different representation

The [Video Swin paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)
and [official torchvision implementation](https://github.com/pytorch/vision/blob/v0.26.0/torchvision/models/video/swin_transformer.py)
provide pretrained spatiotemporal RGB features. The primary candidate is Swin3D-B
pretrained on ImageNet-22K and Kinetics-400. Swin3D-T/Kinetics-400 is the throughput
fallback. Neither is pretrained on Toyota subjects. Official Toyota-trained I3D
features were rejected because their pretraining includes this study's validation
subjects.

Both encoders and the matching Python 3.13 torchvision 0.26.0 CUDA 12.8 wheel are
staged privately, with full upstream hashes and licenses. The run performs no
online model download. Raw RGB videos remain in the existing private Kaggle
dataset.

Each retained target window receives 32 real RGB samples at 15 Hz, centered on its
recorded source frame at pose offset 15. Source-frame rounding and boundary padding
are explicit. A whole-scene view preserves the room/TV; a stable, confidence-based
person crop supplies hand/object detail. Crop geometry uses poses, never labels.
The encoder is frozen and exports pooled embeddings before its classifier.

OpenCV uint8 bilinear resizing, padding, cropping, channel order and normalization
are recorded exactly. This implementation is not claimed to be byte-identical to
torchvision's tensor resize. Each video is decoded sequentially with a bounded
frame cache, and every complete feature file binds source video, source shard,
encoder, preprocessing, implementation and row hashes. Failed or partial videos
cannot count as complete coverage.

## One session with early stopping gates

The maximum Kaggle allocation remains **one 12-hour RTX PRO 6000 session**, with a
40,500-second internal deadline and no automatic continuation.

1. **Throughput probe, at most 15 minutes.** Three complete fitting videos measure
   actual decode, crop and encoder throughput. Prefer Base only when a conservative
   forecast fits both the full job and pilot budgets; otherwise try Tiny within
   the same probe budget. Stop if neither fits.
2. **Whole pilot, at most 75 minutes including setup/probe.** Use 36 fitting videos,
   four per fitting subject, with 31,859 retained fitting windows. They cover all
   15 annotated activities and 561 concurrent windows. Include every one of the
   77 validation videos and 63,220 validation windows. Train matched compact control
   and RGB-fusion heads; retain all selected-video windows.
3. **Quality gate.** RGB validation mAP must reach **0.6379390777**, two percentage
   points above the stronger saved ensemble's validation mAP, and exceed the matched
   control by at least three points. Pair F1, precision and recall must each be at
   least **0.30**. Neither validation subject may fall below the frozen subject-level
   baseline floors. Failure stops the job before full extraction/training.
4. **Full training only after a pass.** Reuse pilot embeddings, finish all 351
   training videos, and train one compact full-data head. Reapply the absolute,
   pair and subject quality requirements. Freeze its checkpoint and thresholds.
5. **Test only after a full-data pass.** Extract the 185 test videos and evaluate
   the frozen model. Incomplete test extraction cannot produce a partial test score.

The pilot is an intentional subset experiment, not a full-data result. If promoted,
the final fitting stage still visits every retained fitting window. There is no
six-seed backbone sweep. The pretrained encoder is reused and only small heads are
trained. Gates express requirements for spending more compute; they are not promised
future scores. The test comparison remains exploratory because the benchmark has
already been inspected, and the calibration split contains only two subjects.

## Completion of the interrupted implementation

The complete extractor was also exercised locally with the official Tiny weights
on a synthetic one-window video. It decoded sequentially, produced distinct scene
and person embeddings with shape `[1, 2, 768]`, and stayed within its bounded frame
cache. This CPU check used no GPU allocation and measures interface correctness,
not accuracy or GPU throughput. Its receipt is stored at
`artifacts/pipeline-rebuild-20261012/local-extractor-cpu-smoke/report.json`.

Training checks reload the exported checkpoint and fitting-only normalization to
reproduce validation logits exactly. Before test data is loaded, evaluation now
verifies the checkpoint, thresholds and frozen-selection hashes and recomputes
the recorded full-data validation gates. Invalid normalization, nonfinite time
limits and incomplete exports are rejected.

The Linux parent supervisor runs one GPU worker and enforces the 15-minute probe,
75-minute pilot and 40,500-second session limits outside the decoder and encoder.
It can terminate the worker and its trainer subprocesses even when a native call
stalls. Only a timely, verified pilot-pass receipt extends execution beyond the
pilot. Base encoder GPU memory failures may fall back to Tiny inside the same
probe allowance; budget expiry is reported separately from measured quality
failure. These checks do not establish that the throughput or quality gates will
pass on the actual dataset.


## Measured run outcome

Kaggle version 1 completed its bounded experiment in **347.065 seconds** and
stopped at `throughput_rejected`. Both encoders processed all three prescribed
probe videos, then failed the time forecast. Base projected **217.97 minutes**
for the pilot and **17.03 hours** for the full job; Tiny projected **171.88 minutes**
and **13.32 hours**. No head training or test evaluation occurred.

The runtime/source audit and all six downloaded probe feature files passed their
identity and hash checks. This is a verified compute-budget rejection, not a model
quality result. The accuracy goal remains unresolved. See the
[verified run report](../results/rgb_pipeline_20261012.md) for measurements and limits.
