# Expanded RGB execution — 14 October 2026

**The expanded 60-video experiment completed in 354.239 seconds and was rejected
by its validation quality gate.** RGB validation mAP improved from the previous
pilot's 48.11% to **52.57%**, but remains below the 63.79% requirement. Pair
precision and F1 passed their 30% floors; pair recall did not. No full-data
training, test evaluation or serving-model replacement occurred.

Run: [bs-rgb-coverage-20261014](https://www.kaggle.com/code/krishnagupta02468/bs-rgb-coverage-20261014).
Accepted kernel ID: `138670264`. Active evidence directory:
`artifacts/rgb-runtime-20261014/submit-compact/`.

## Verified result

| Metric | Previous RGB pilot | Expanded RGB pilot | Required |
|---|---:|---:|---:|
| Validation mAP | 48.11% | **52.57%** | 63.79% |
| Pair precision | 36.11% | **41.18%** | 30.00% |
| Pair recall | 6.34% | **23.90%** | 30.00% |
| Pair F1 | 10.79% | **30.25%** | 30.00% |
| Pair ranking AP | 19.85% | **13.78%** | No fixed floor |

The previous and expanded pilots have the same validation fingerprint. These
are descriptive results on the reused validation set, with independently
selected validation thresholds for each candidate. The expanded pilot improves
threshold-dependent pair decisions while its pair ranking AP declines.

The matched control achieved **42.42% mAP**; adding RGB produced a **10.15
percentage-point gain**, passing the required three-point improvement. Both heads
met the frozen validation-plateau stopping rule: control completed 26 epochs
(best epoch 14), RGB completed 19 (best epoch 7). Neither exhausted its budget.

Exactly four checks failed: aggregate mAP, pair recall, p04 mAP and p15 mAP.
Subject p04 reached **49.48% mAP** against a 59.13% floor, and p15 reached
**57.95%** against 64.50%. Both subject-specific pair-F1 floors passed.

The RGB head correctly predicted **49 of 205** pair-positive windows. It missed
156 and made 70 false pair predictions. The pair labels are watching TV and
drinking within a window; actual frame overlap is a separate metric. Personal
hygiene remains particularly weak at **1.18% AP** despite its expanded fitting
coverage. These results do not establish the requested accuracy target.

## Execution and output verification

The expected RTX PRO 6000 Blackwell, native `sm_120`, Python 3.13, Torch
2.11.0+cu128 and torchvision 0.26.0+cu128 were verified. All 113 original caches
were checked; 110 were installed in 15.149 seconds. The three paired core probes
had exactly matching embeddings. The two sparse probes also completed, and the
throughput phase passed at 133.425 seconds.

The conservative forecast was 1,162.34 seconds for the pilot and 20,487.67 seconds
for a full session. Actual extraction covered all **60 fitting / 77 validation
videos**, with **40,193 fitting / 63,220 validation windows**, by 306.750 seconds.
The final quality rejection at 354.239 seconds occurred within all applicable
phase limits. Kaggle subsequently reported `COMPLETE`.

The independent terminal audit verified the source/runtime identity, cache and
probe receipts, convergence, checkpoint and threshold hashes, and all nine gate
decisions. A separate validation audit recomputed aggregate and subject metrics
from both heads' saved logits, including sklearn average precision, to an
absolute tolerance of 1e-12. All 63,220 rows, labels and masks match between heads
and reproduce the validation fingerprints. Thresholds were not refitted and
weights were not loaded. Frame-overlap metrics were not recomputed because those
arrays are absent from the prediction exports.

The failed gate prevented full training and test access. The output contains no
full-training or test directory, pilot-pass marker, root frozen selection, or
testing feature inventory. The experiment is finished; no further GPU run is
scheduled. Any subsequent candidate needs a new protocol that addresses the
remaining class and pair-ranking weaknesses while preserving the quality gates.

Evidence: [terminal audit](../artifacts/rgb-runtime-20261014/submit-compact/execution_output_audit.json),
[validation replay](../artifacts/rgb-runtime-20261014/submit-compact/independent_validation_audit.json),
and [transport equivalence audit](../artifacts/rgb-runtime-20261014/submit-compact/independent_equivalence_audit.json).

## Submission repair

The original frozen `training.py` was 1,390,568 bytes. Kaggle rejected the initial
request with HTTP 400. Its original launcher recorded the attempt but omitted
the server's error details. Repeated authenticated status checks reported
`No runs found for this kernel.` A separately journaled, unchanged retry captured
the exact server message:

> The kernel source must be less than 1 megabytes in size.

Only the embedded JSON job metadata was compressed with zlib and base64. The
compact submission is **916,477 bytes**, below a conservative 1,000,000-byte
limit. An independent AST comparison verified that the runtime is unchanged
apart from decoding the same job and importing zlib. The decoded job, Kaggle
metadata, source archive and manifest match the frozen originals, and all 88
archive member hashes passed. The independent check did not execute training.

The repaired request was accepted as version 1 with no invalid input sources.
The two failed API attempts and their immutable bundles remain in `run-ready/`
and `submit-reconciled/`; they are not additional accepted training versions.

The launcher now rejects oversized source locally during build and before push,
and preserves bounded HTTP error codes/messages for diagnosis. **76 focused
tests passed:** 27 temporal launcher, 32 RGB launcher and 17 coverage launcher.
The existing one-attempt journal remains in place.

An initial quota interpretation was incorrect because the SDK's string/JSON
serializer dropped the whole-day portion of a `timedelta`. Reading the actual
duration with `total_seconds()` gave a 30-hour allowance, 12.16 hours used and
17.84 hours available before submission. Quota exhaustion was not the cause.

## Preserved experiment

The [coverage plan](rgb_coverage_20261014.md) and
[runtime integration](rgb_runtime_20261014.md) remain the scientific protocol:

- 60 fitting videos, 40,193 windows and 638 pair-positive windows, covering all
  28 available pair-positive recordings.
- 110 verified cached outputs plus five fresh probes, including two sparse cases.
- Shared 900-second probe, 4,500-second pilot and 40,500-second session limits.
- The original accuracy gates and test-access guards; no serving-model replacement.

Frozen source archive SHA-256:
`2f4088fad5fe1a8c28af346be107d4e4d23e6880ec00408889cf66c298c66464`.

Compact submitted source SHA-256:
`d20e48452427d43958e3ab33cb9e9e9283e95868e949e1ab3d12dfab97f4c689`.

The submitted source was read back from Kaggle and matched exactly. The saved
terminal artifacts establish a completed, converged pilot quality rejection.
