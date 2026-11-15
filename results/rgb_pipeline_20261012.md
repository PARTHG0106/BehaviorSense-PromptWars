# RGB pipeline experiment — verified 12 October 2026

**The run stopped after 5 minutes 47 seconds because neither encoder met the
throughput budget. No head was trained, no test data was extracted or evaluated,
and no new accuracy result was produced.** Kaggle reports version 1 complete;
that means the bounded experiment ended, not that the model improvement succeeded.

[Single Kaggle job](https://www.kaggle.com/code/krishnagupta02468/bs-rgb-video-20261012) · terminal state: `throughput_rejected`.

## Measured throughput and stop decision

Each encoder processed the same three complete fitting videos, containing
**1,915 retained windows**, with separate scene and person views. Both completed
without a GPU memory failure or extraction timeout. Both were rejected by the
prespecified time forecast.

| Encoder | Observed extraction windows/s | Conservative pilot forecast | Conservative full-job forecast |
|---|---:|---:|---:|
| swin3d_b | 10.49 | 217.97 minutes | 17.03 hours |
| swin3d_t | 13.74 | 171.88 minutes | 13.32 hours |

The limits were **75 minutes for the entire pilot** and **11.25 hours internally
for the complete job**, within a single 12-hour Kaggle allocation. Forecasts used
1.25 times the slowest whole-video seconds per retained window, including decoding,
cropping and encoding, plus the reserved head-training/export time. They are
projections, not elapsed training times or guarantees across the full dataset.

Tiny would need approximately **2.64× faster extraction** to fit the same
pilot policy at its decision point. The next engineering step is to measure
decoding/cropping, transfer and encoder time separately, then improve the measured
bottleneck or use a cheaper representation. The current reports do not identify
which component dominates. Increasing epochs would not address this failure.
No second GPU job was launched, and no gate or coverage requirement was relaxed.

The task's accuracy goal remains unresolved. The stronger saved reference is the
previous fixed 50/50 context/temporal ensemble at **65.29% test mAP** and **12.59%
TV/drinking pair precision**; those are existing results, not this experiment's.

## Implementation completed

- Frozen pretrained Video Swin scene/person embeddings supply visual evidence
  that the prior skeleton/object summaries discard.
- Corrected visible-pose and bone statistics provide the matched control input.
- A Linux parent supervisor enforces the **15-minute probe**, **75-minute pilot**
  and **40,500-second session limit**, including termination of stalled worker and
  trainer processes. This run stopped voluntarily from its failed forecast.
- Base GPU memory failures can try Tiny inside the same probe allowance. Budget
  exhaustion, quality rejection and successful training are distinct outcomes.
- Test evaluation requires matching checkpoint, threshold and frozen-selection
  hashes and replayed validation gates. Incomplete runs cannot export a final model.

The unexecuted pilot would use 36 fitting videos (31,859 windows) and all 77 validation
videos (63,220 windows). Its unchanged quality requirements are **63.79% validation
mAP**, a **3 percentage point** gain over its matched control, at least **30% pair
F1, precision and recall**, and both subject-level floors. Full extraction and
training remain forbidden without a pilot pass.

## Verification

**173 checks passed** across eight related isolated CPU suites. One real Linux
process-group cleanup test was skipped on the local Windows host; mocked supervisor
and timeout-path checks passed. The 82-file frozen source archive was independently
unpacked, hash-verified and imported, including both trainer command interfaces.

Live output verified the actual **RTX PRO 6000 Blackwell**, native sm_120 support,
Python 3.13, Torch 2.11.0+cu128 and torchvision 0.26.0+cu128. Downloaded job/source/assets
identities and all 351 mounted training-shard hashes matched the frozen plan.
The probe budget calculations were independently reproduced from their receipts.

All **six GPU-extracted feature files (12,895,043 bytes)** were saved locally.
File hashes, array hashes, metadata, feature shapes and finite float16 contents
passed verification. No new model inference or raw-video replay was performed
while auditing those files. The earlier complete-extractor CPU check with official
Tiny weights also passed on a synthetic video without using GPU time.

- Execution time: **347.065 seconds** inside the job, excluding Kaggle provisioning
  and output packaging; billed GPU time was not independently measured.
- Frozen source SHA-256: `da591cce928d951d0ed8c19b72e841b1d68c88c421f179d1f57c14b69724bfe4`.
- Kaggle kernel ID: `138320335`; version: `1`.
- Training performed: **no**. Test evaluated: **no**. Serving model replaced: **no**.
- Automatic continuation: **disabled**. Local observer: **stopped**.

The benchmark has already been inspected, so any future test comparison remains
exploratory.

[Pipeline design and input audit](../docs/11_rgb_pipeline_rebuild.md)

Private local evidence: `artifacts/pipeline-rebuild-20261012/run/` contains
`preflight_audit.json`, `plan.json`, `submission_receipt.json`, `output_audit.json`,
`probe_feature_audit.json`, the six feature files, and the downloaded probe/runtime
receipts. Re-run the metadata audit with `python -B
artifacts/pipeline-rebuild-20261012/run/audit_outputs.py`.
