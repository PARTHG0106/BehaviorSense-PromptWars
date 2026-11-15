# RGB fitting coverage comparison — 15 October 2026

**The four-hour attempt stopped at its throughput gate before training or
scoring.** The three fresh complete probes finished in 136.24 seconds. Their
conservative full-session projection was **19,838 seconds (5.51 hours)**,
including a 7,200-second training/export reserve, exceeding the 14,400-second
worker limit. Version 1 of `krishnagupta02468/bs-rgb-fit-coverage-20261015`
remains preserved. There are no cached-versus-full model scores from this attempt.

The downloaded bytes, source identity, environment, 60 cache receipts, three
fresh probes and projection arithmetic were verified. A separately frozen
[six-hour comparison](rgb_fitting_coverage_20261016.md) continues the same
scientific design. No weights, training settings, subject groups or quality
criteria were changed in response to a model score.

The experiment tests whether additional fitting data improves generalization
and reduces TV-plus-drinking false positives. It compares the original
**60 videos / 40,193 windows** with **274 videos / 217,090 windows**, holding the
encoder, head architecture, optimizer, seed, stopping rule and threshold procedure
fixed. The extra 214 videos contain 176,897 fitting windows.

Each arm trains on three subjects, calibrates on three other subjects and
evaluates on the remaining three. Both arms have the same complete calibration
and evaluation cohorts. Across the three rotations, all nine fitting subjects
are evaluated once per arm. This measures coverage; it is not a final model
trained on all nine subjects.

| Rotation | Fitting | Calibration | Evaluation |
|---|---|---|---|
| A | p09, p13, p17 | p06, p12, p19 | p03, p07, p25 |
| B | p03, p07, p25 | p09, p13, p17 | p06, p12, p19 |
| C | p06, p12, p19 | p03, p07, p25 | p09, p13, p17 |

The full cohort adds mostly negative examples and other-activity diversity.
Its 73 additional pair-positive windows are alternate views of already
represented positive recordings. These internal subject groups have guided
earlier research, so the results remain exploratory.

## Frozen comparison

The original 3,025-feature RGB/pose/object head retains 256 hidden units,
dropout 0.3, seed 0, AdamW learning rate 0.001, weight decay 0.001 and batch size
2,048. It must converge within the original 80-epoch budget, with minimum 10
epochs and patience 12. Each arm fits its normalization and class weights only
on its own fitting rows. Epoch selection and threshold calibration use only
the calibration subjects.

All six fits must converge and freeze before evaluation begins. Advancement
requires noninferior pooled mAP and pair precision, strictly higher pair F1,
and pair precision, recall and F1 each at least 30%. Rotation and subject results
will accompany the pooled scores, whose rankings combine three separately
fitted heads. Advancement authorizes further research.

The worker has a four-hour hard limit, a 900-second fresh-probe gate and a
two-hour training/export reserve. The external timeout is 4.5 hours. The
accepted source is 967,981 bytes. There is one accepted submission and no
automatic resubmission.

## Verification and recovery

Fourteen scientific/runtime checks and four submission-adapter checks passed.
An actual fitting-data rehearsal reproduced 128 combined feature rows exactly
and verified 1,829 source rows, including byte identities, labels, masks,
starts and source clocks. The frozen archive and all 88 scientific files compile.

The inherited temporal client verifier assumed a 12-hour timeout and rejected
the shorter plan locally before a submission occurred. The separate client
adapter verifies the actual frozen limits; the submitted scientific bundle
was preserved byte for byte.

The data builder enforces a fitting-only allowlist. The final replay checks
coverage, subject separation, identical initial parameters, learning-rate
schedules, calibration thresholds, class and pair metrics, and advancement.
The accepted job's version and source identity are checked before and after
output collection. RGB feature caches remain in the remote output for reuse.

- [Reviewed protocol](../artifacts/rgb-next-20261015/coverage-study/PROTOCOL_REVIEW.md)
- [Frozen protocol](../artifacts/rgb-next-20261015/coverage-study/protocol.json)
- [Continuation state](../artifacts/rgb-next-20261015/coverage-study/continuation_status.json)
- [Run plan](../artifacts/rgb-next-20261015/coverage-study/run/plan.json)
- [Actual-data rehearsal](../artifacts/rgb-next-20261015/coverage-study/local_real_data_rehearsal.json)
- [Verified operational rejection](../artifacts/rgb-next-20261015/coverage-study/terminal_output_audit.json)

The previous [frozen RGB hybrid benchmark](rgb_improvement_20261015.md) remains
the latest verified test result. Its scored test rows do not enter this study.
