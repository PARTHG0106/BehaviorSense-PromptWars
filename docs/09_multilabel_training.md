# Full-data multilabel experiment — 7 October 2026

**Current status (11 October 2026):** training and official test evaluation are complete.
All six candidates converged using 217,090 fitting windows and 63,220 validation windows
from the 351 extracted training videos. The completed Save & Run All sessions are:

- [Skeleton seeds 0–2](https://www.kaggle.com/code/krishnagupta02468/bs-mlfull-20261007-sk-all-a00)
- [Context-fusion seeds 0–2](https://www.kaggle.com/code/krishnagupta02468/bs-mlfull-20261007-ctx-all-a00)

| Model | Seed 0 validation mAP | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Skeleton | 0.55185 | 0.56213 | 0.56378 | 0.55925 |
| Context fusion | 0.58383 | 0.59139 | **0.59254** | **0.58925** |

Validation-only selection is frozen to **context fusion, seed 2**. All six exported
checkpoints have been downloaded and their SHA-256 hashes verified against the completed
stage reports. The selected hash is
`32f7c33e0234b0931fb28b26144fef9ce3b9e3a096c50d1112009b901a7ef384`.
Histories and convergence reports agree; early stopping occurred after 40–56 epochs.
The mean improvement is 3.00 percentage points of validation mAP. Simultaneous TV/drinking
remains weak: the selected candidate's co-labeled-window validation F1 is 0.1449.
These are retained-window validation results, with thresholds fitted on validation.
See the [full validation report](../results/multilabel_validation_20261009.md).

**Official testing is complete.** All four extraction jobs processed the 185 test videos,
retaining 153,537 of 182,631 annotation-eligible windows. The
[final evaluation](https://www.kaggle.com/code/krishnagupta02468/bs-mlfull-20261007-test-a00)
reports **0.62607 test mAP, 0.57964 macro F1 and 0.65210 micro F1** for the frozen context-fusion
seed 2 checkpoint. TV/drinking co-label F1 is **0.15666**, with **9.44% precision**; reliable
simultaneous detection remains unresolved. These are retained-window test metrics.

On 11 October, all five test jobs were confirmed `COMPLETE`. Downloaded predictions
reproduced the main, per-class and co-label metrics using the saved validation
thresholds. Checkpoint identity, subject separation and all retained window starts were
verified. The coordinator completed and exited. See the
[official test report](../results/multilabel_test_20261011.md) for coverage, per-class results
and saved evidence.

The 9 October launch was rejected with `Maximum weekly GPU quota of 30.00 hours reached.`
That rejection remains in the plan's history. Quota refreshed and test extraction resumed
on 10 October; the previous coordinator logs and plan were preserved before resumption.

The two models train independently from the same extracted features; context fusion does
not need a completed skeleton checkpoint. The original single-seed `sk0-a01` session was
cancelled by the user and replaced by these grouped runs. Its predecessor failed while
mounting a detector-weight dataset before any user code ran. Training and evaluation now
mount only annotations and extracted features, omitting detector weights and inference
wheels. The frozen scientific source archive and trainer settings remain unchanged.

All 79 focused launcher, grouped-seed, evaluation and parallel-coordinator checks pass.
The coordinator journals both notebook identities independently; one failed job cannot
prevent its sibling from progressing. The six-candidate selection requirement has been met.
The serving model has not been replaced.

This private candidate learns independent activity outputs from Toyota Smarthome
Untrimmed. TV watching and drinking motivated the work, but **all 15 annotated coarse
activities** receive multilabel supervision. The active serving model is not replaced.

The original 32-window pilot was cancelled. The replacement uses the complete official
inventory: 351 training videos and 185 test videos, about 149 hours of recordings.
An annotation-only audit estimates 312,593 eligible training windows and 182,631 test
windows. These are planning counts assuming 25 fps source videos, before actual decoder
checks and pose visibility filtering; they are not model results.

## Coverage and provenance

- Sample real source frames at 20 Hz; retain every annotation-eligible 30-frame window
  on a 15-frame grid. There is no per-video window cap or activity-based subsampling.
- Require at least 50% annotation coverage. Annotation gaps do not become negative labels.
  Keep exact source-frame indices, rates, starts, overlap counts and exclusion reasons.
- Reject a changed official inventory, missing videos, wrong subjects, incompatible clocks,
  truncated decoding, nonfinite inference, or mismatched extraction settings.
- Record visibility rejects explicitly: the primary pose slot needs at least eight joints
  above 0.3 confidence in at least half the frames. Performance describes retained eligible
  windows, not every video frame or population activity prevalence.
- Bind resume files to source video SHA-256, annotations, both detector weights, and the
  extraction implementation. Pilot shards cannot be reused as full-data shards.
- Use eight duration-balanced training partitions and four test partitions. All training
  partitions are complete; partitions 4–7 shared one RTX PRO 6000 Blackwell session.
  Save each video atomically and stop between videos before the
  session budget. Later sessions mount completed output without discarding completed work.

## Models and selection

Two architectures are compared using prespecified seeds 0, 1 and 2:

1. ST-GCN++ bone-stream skeleton baseline.
2. The same backbone with a learned residual fusion head consuming actual frozen RT-DETR
   object scores and person-relative geometry from each window's center RGB frame.

The RT-DETR adapter validates the staged export's exact SHA and tensor contract. The
features describe detected objects and two-dimensional proximity. They do not establish
attention, TV use, or an activity by themselves. No hand-written TV/drinking boost or
invented concurrent label is added. The visual branch must earn its place on validation.

All six candidates start from scratch. The older served checkpoint has unverified subject
exposure and is unsuitable for an untouched official test claim. Fit subjects are p03,
p06, p07, p09, p12, p13, p17, p19 and p25. Validation subjects p04 and p15 select epochs,
per-class F1 thresholds and the final model/seed. All candidates must use identical
validation targets. All 15 supported classes need sufficient positive and negative
fit/validation support. Unsupported outputs are masked and disabled.

The initial schedule uses batch size 256, AdamW at 0.0003, at most 150 epochs,
at least 40 epochs, and 25 validation epochs of patience. Validation plateaus reduce the
learning rate; convergence additionally requires two reductions and time to settle after
the last one. Hitting the epoch limit without convergence requires review and does not
silently launch empty resumes or export a final model. Sessions save optimizer, scheduler,
EMA, scaler and random-number states for exact continuation. Disk-backed arrays bound RAM.

After all six candidates converge, highest validation macro AP chooses one checkpoint.
Only then are test windows extracted and the frozen checkpoint/thresholds evaluated on
p02, p10, p11, p14, p16, p18 and p20. Test outputs report per-class AP/F1 and concurrent
window-label metrics, plus coverage and provenance. Test data never enters model selection.

Supported activities: walking, lying down, standing up, sitting down, eating, drinking,
cooking/food preparation, taking medication, watching TV, reading, using a phone, cleaning,
personal hygiene, using a device, and object interaction. Standing, sitting, bending/reaching,
falling, fallen on ground, interacting with a person, and other/idle lack direct supervision
under this mapping and are not presented as trained by this experiment.

## Runtime and operation

The verified image uses Python 3.13, Torch 2.11.0+cu128, native sm_120 support and an RTX PRO
6000 Blackwell with about 102 GB of VRAM. A separate private dataset stages checksummed
Python 3.13 ONNX Runtime GPU and rtmlib wheels. Every stage refuses CPU fallback and a GPU
other than the configured Blackwell. A real-GPU detector preflight precedes extraction.

`scripts/launch_multilabel_kaggle.py build` freezes a source-only archive and a finite plan.
`run` coordinates the two training groups concurrently and the other stages sequentially, and can resume the saved plan.
GPU jobs run on Kaggle; the local coordinator must remain running to submit subsequent
jobs. Progress is saved in `artifacts/training-tv-20261007/full-run/plan.json`. An OS lock
prevents two coordinators submitting the same stage. Credentials stay local.

```powershell
python scripts/launch_multilabel_kaggle.py build --credentials kaggle-krishna.json
python scripts/launch_multilabel_kaggle.py run --credentials kaggle-krishna.json
python scripts/launch_multilabel_kaggle.py status --credentials kaggle-krishna.json
python scripts/launch_multilabel_kaggle.py logs --follow --credentials kaggle-krishna.json
```

Build once per frozen experiment. `push` submits or advances one stage; it does not keep a
coordinator running. Inspect `needs_attention` or `needs_review` before resuming failed work.
Never rebuild over a running experiment. Kernel IDs are unique per stage/resume attempt;
datasets, outputs and source bundles remain private. No serving deployment is automatic.

```powershell
python run_tests.py --suite test_multilabel_data --suite test_multilabel_extraction --suite test_train_adl_multilabel --suite test_visual_context --suite test_multilabel_launcher --suite test_grouped_multilabel_extract --suite test_seed_training_stage --suite test_feature_training_stage --suite test_parallel_training
```

The user's example intervals (385–405, 478–497, 522–552 and 591–596 seconds) remain
qualitative follow-up checks. They are not injected into training labels; the source video
must be identified before treating any such clip as unseen.
