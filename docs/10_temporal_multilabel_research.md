# Temporal multilabel improvement experiment — 11 October 2026

**Completion verified, 12 October 2026:** [Kaggle version 1](https://www.kaggle.com/code/krishnagupta02468/bs-temporal-multilabel-20261011)
completed all six candidates and test evaluation in **6 hours, 3 minutes, 40 seconds**
within one RTX PRO 6000 session. Kernel ID is `138168743`. The frozen source SHA-256 is
`9082c6ea5ada02e66a6f5905362eca1b68e5f821ccd56d3c31c24582937c5f65`.
The setup passed 131 focused checks, including real-format tiny-shard fitting,
checkpoint reload, validation-only thresholds and frozen test evaluation. These
checks establish execution contracts, not the new model's recognition score.

The selected candidate is **ASL, seed 1**, the only candidate passing the prespecified
validation gate. Its validation mAP is **60.64%**, versus 59.25% for the baseline.
Test results are mixed: mAP **62.61% → 62.55%**, macro F1 **57.96% → 58.90%**, and
TV/drinking pair F1 **0.15666 → 0.17827**. Pair precision is still only **10.74%**.
All six artifact hashes and the selected model's full aggregate/per-subject metrics
were verified locally against the saved predictions and annotations. See the
[complete comparison](../results/temporal_multilabel_20261012.md).

This experiment learned from longer, ordered visual context while retaining the
existing annotation-aligned skeleton windows. The one-session experiment produced
a modest F1/concurrency gain, without an overall test mAP improvement.

## Evidence and decision

[PDAN (WACV 2021)](https://openaccess.thecvf.com/content/WACV2021/html/Dai_PDAN_Pyramid_Dilated_Attention_Network_for_Action_Detection_WACV_2021_paper.html)
models short and long temporal relationships for densely annotated, multilabel
action detection. Its [official implementation](https://github.com/dairui01/PDAN)
uses ordered features, residual dilated attention, and independent sigmoid
outputs; it also links a Toyota Smarthome Untrimmed evaluation pipeline.
[Long-Term Feature Banks (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)
provides complementary evidence that cached surrounding features can improve
short-clip recognition. These motivate temporal context here; the planned model
is an adaptation, not a reproduction of either architecture.

[MS-TCT (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Dai_MS-TCT_Multi-Scale_Temporal_ConvTransformer_for_Action_Detection_CVPR_2022_paper.html)
also evaluates multi-scale local and global temporal modeling on TSU and explicitly
addresses composite and co-occurring actions. This supports testing temporal order
and multiple context scales in this setting.

[Asymmetric Loss (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html)
addresses the dominance of easy negative labels in multilabel classification.
Its [official loss implementation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py)
uses separate positive/negative focusing and shifted negative probabilities. We
compare masked ASL (`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`) with the existing
masked, class-weighted binary cross-entropy. ASL receives no additional class
weights. Focal weights are detached as in the standard reference implementation;
the local implementation preserves the caller's gradient mode and computes in
float32 under mixed precision. Stable unclipped log probabilities preserve
gradients for confidently wrong positives. Unknown labels never enter the loss.

The [Toyota Smarthome Untrimmed paper](https://arxiv.org/abs/2010.14982) and
[official repository](https://github.com/dairui01/Toyota_Smarthome) establish that
long recordings, temporal variation and object interactions are substantial
challenges in this dataset. Learned RGB video features could supply evidence
missing from detector outputs, particularly hand/cup/face interactions. They are
deferred for this bounded session because new decoding and feature extraction
would consume an unmeasured portion of its time budget.

## Prespecified candidates and coverage

`TemporalMultilabelFusion` starts from scratch and learns from joint and bone
streams for the same 30-frame, 20 Hz center skeleton window. It also receives 13
ordered cached context observations at offsets `-6, ..., 6`, separated by the
existing 15-sample stride. The first and last observation centers span **9
seconds**. This supplies **4.5 seconds of look-ahead** relative to the center
observation; it is an offline model, not zero-latency live inference. The target
still describes the original **1.5-second center window**.

Neighbor lookup uses exact video identity and sampled starts, never adjacent row
numbers. Visibility rejects and missing neighbors remain explicitly masked;
they cannot compress time or cross video boundaries. Neighbor dropout during
fitting helps the model tolerate missing surrounding observations. No new
concurrency labels are fabricated and no eligible center windows are discarded
to favor rare activities.

The single notebook trains BCE and ASL candidates with seeds 0, 1 and 2, using
the same fitting/validation subjects and full retained-window coverage. Planned
settings are batch size 512, a maximum of 80 epochs, a minimum of 40 epochs and
20 validation epochs of patience. A runtime guard must reserve time to export
artifacts before the 12-hour session limit; it must never silently launch a
second session. A candidate stopped by the compute budget must be identified as
such, rather than reported as converged.

## Resource basis

The previous context group completed 127 epochs across three seeds in **5,471.7
seconds (1.52 hours)**. Its saved histories average **42.21 seconds per epoch**.
The skeleton group took 6,407.9 seconds (1.78 hours). These are useful baselines,
not predictions for the larger joint/bone temporal architecture or batch size.
Actual throughput must be measured in the new session and checked against its
remaining budget.

The original full training feature extraction took **32,781.2 seconds (9.11
hours)**. Repeating that extraction plus the previous six-candidate training
would already exceed 12 hours. Reusing frozen, verified shards leaves much more
time for the change being tested. This does not establish the runtime of a new
RGB encoder, whose throughput has not been benchmarked here.

## Evaluation limits

Choose checkpoints and configurations using validation only, then freeze the
selected artifact and validation-fitted thresholds before evaluating it on the
official test split. Report overall mAP, per-class results and TV/drinking pair
AP, precision, recall and F1. Better overall mAP alone does not establish improved
simultaneous detection: the prior ensemble audit already demonstrated this.
Negative suppression by ASL could improve ranking while worsening pair
precision, so both outcomes need to be visible.

The frozen selection gate requires validation mAP above **0.5925371084**, pair F1
at least **0.1448818898**, and pair precision at least **0.1069767442**, matching the
previous selected model's validation reference. Choose the highest mAP among
passing candidates. If none passes, label that outcome explicitly and evaluate
the highest-mAP candidate as a research comparison without recommending promotion.
All six candidates must converge before selection. Pair AP uses the minimum of
the two marginal sigmoid scores as a ranking score; it is not a calibrated joint
probability. Per-subject validation and test reports are also saved.

Only 711 fitting windows contain both activities. Validation has 205 such
windows, with 197 from p04 and eight from p15. Overlapping windows are correlated,
and per-subject results are essential context for an unstable pair score. Do not
interpret 205 windows as 205 independent concurrent events or assert statistical
significance from a window-level bootstrap.

The original test results have already been inspected. Any new test comparison
uses a **reused benchmark and is exploratory**, even when the new candidate is
selected without further test feedback. Published Toyota/PDAN frame mAP uses a
different taxonomy and evaluation protocol and cannot be numerically compared
with this project's 15-class retained-window mAP. No particular gain is
guaranteed, and this experiment does not automatically replace the serving model.

The primary paper pages and linked official implementations above were fetched
and inspected on 11 October 2026. See the
[existing improvement audit](../results/multilabel_improvement_audit_20261011.md)
for the measured baseline and ensemble comparisons.

## Execution and artifacts

The job mounts five completed training extraction outputs and four completed test
outputs. Only training feature files are read before fitting and validation-only
selection finish. It retains every fitting window each epoch. The test features
are loaded after the selected checkpoint and thresholds have been frozen.

Kaggle's session timeout is 43,200 seconds. The internal job budget is 40,500 seconds,
including a 900-second evaluation/export reserve. Remaining training time is shared
among the remaining candidates. A timeout or unconverged candidate is reported as
incomplete; the code never schedules another job or resumes another session.

The accepted plan, submission receipt, precommit verification and saved source bundle
are under `artifacts/temporal-multilabel-20261011/`. All orchestration occurs within the
remote job; the local computer does not need to coordinate the subsequent candidates.

```powershell
python scripts/launch_temporal_kaggle.py status --directory artifacts/temporal-multilabel-20261011
python scripts/launch_temporal_kaggle.py output --directory artifacts/temporal-multilabel-20261011
```

The output command verifies job/source identity, the actual RTX PRO 6000 runtime,
and every completed checkpoint hash. Do not rebuild or push the accepted experiment
again; its immutable commit has completed and its outputs are verified locally.
