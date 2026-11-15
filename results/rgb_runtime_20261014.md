# Expanded RGB runtime integration — 14 October 2026

**The 60-video pilot is integrated, tested and frozen as a local submission
bundle.** It reuses verified features from the completed job and freshly times
both ordinary and sparse videos before allowing training. No GPU submission,
new model training, validation scoring or test-array access occurred during
this integration. The accuracy goal remains unresolved.

This completes the runtime work identified in the
[coverage plan](rgb_coverage_20261014.md). The prepared job ID is
`krishnagupta02468/bs-rgb-coverage-20261014`; this is a planned identifier, not
an executed Kaggle run. Its active local directory is
`artifacts/rgb-runtime-20261014/run-ready/`.

## Executable sequence

| Stage | Required behavior |
|---|---|
| Cache verification | Check all 113 original RGB files against their byte hashes, source provenance, encoder/preprocessing contract and matching pose rows. Atomically install 110; keep the three cached core probes fresh. |
| Core profiling | Run the existing paired sequential/prefetch comparison on three complete fitting videos, requiring matching clocks, crops and embeddings. |
| Sparse profiling | Freshly process the densest source-frame/retained-window case and the remaining new video with the most source frames. Complete both within the shared probe deadline. |
| Pilot | Process all 60 fitting videos plus all 77 validation videos, then fit the existing matched control/RGB heads. |
| Expansion | Full fitting and test evaluation remain conditional on the existing pilot and full-data quality gates. |

The five fresh probes are P25T01C05, P09T05C05, P07T04C05,
**P09T14C07** and **P06T12C07**. The first sparse case has only one retained
window, so an ordinary per-window rate could badly misrepresent its decode cost.

The new forecast separates three measured costs: decoding per source frame,
other work per eight-window batch (including a partial batch), and per-video
validation/serialization overhead. Each rate uses the worst of the five fresh
measurements with the existing 1.25 safety factor. Profile sums may overlap;
they are used conservatively rather than presented as an exact decomposition
of wall time. Only the complete verified cache set receives a reuse discount.

The parent supervisor requires complete cache and sparse-probe receipts, their
hashes and timely timestamps before extending execution past the probe phase.
Cache verification, copying, profiling and setup are charged to the existing
**900-second probe, 4,500-second pilot and 40,500-second session limits**.
The 600-second pilot-head and 3,600-second full-head/export reserves remain.
Full training also reserves the new frame/batch/video estimate for test extraction.

The accuracy requirements are unchanged: overall validation mAP at least
**63.79%**, at least a three-point gain over the matched control, pair precision,
recall and F1 each at least **30%**, and the frozen subject-specific floors.
Cache verification or a throughput pass is not a quality pass.

## Verification performed

**163 focused tests passed; two platform-specific tests were skipped.** The
seven isolated suites cover coverage lineage, launcher behavior, cache integrity
and rollback, sparse timing, full execution/early stops, and the existing profile
and runtime paths. Skips cover Linux process cleanup on Windows and unavailable
Windows directory-symlink creation. The cache suite now has a runnable pytest
entry point and explicit assertion totals for the repository's isolated runner.

The real cache rehearsal retrieved missing files from completed jobs and checked
**113 RGB files against 113 matching pose shards**. It completed verification
and publication in **22.86 seconds on CPU**, installing 33 fitting and 77
validation caches. Three cached probes were deliberately excluded; both new
sparse probes were absent as required. All source bytes remained unchanged.
The audit recorded 226 array loads, zero test-array loads, zero model-weight
loads and zero inference/training calls. This timing covers local verification,
not network recovery or GPU extraction.

All **88 source files** in the frozen bundle passed byte/hash checks, isolated
imports and job-contract validation. The private encoder/raw-video datasets
and completed cache source passed the live read-only readiness check.

Kaggle's version-qualified read requests failed during preflight. The corrected
verifier reads the supported current resource and requires exact private
version-1 identity and source text hashes both before and after its status check.
It rejects intervening version, source, identity or privacy changes. The actual
cache mount remains pinned to version 1, and every mounted file is independently
hash-verified. The earlier unsubmitted build is marked superseded.

## Prepared artifacts and next execution

Frozen source SHA-256:
`2f4088fad5fe1a8c28af346be107d4e4d23e6880ec00408889cf66c298c66464`.

The private `run-ready/` directory contains the plan, submission source and
metadata, source manifest, bundle preflight and remote-readiness receipts.
`cache-rehearsal/verification.json`, `array_load_audit.json`, `recovery.json`
and `work/cache_reuse.json` record the real data checks.

The launcher accepts `--coverage-plan` and requires an explicit new `--run-slug`.
Legacy frozen jobs retain their original validation path. It will recheck inputs
on submission and record the single submission attempt. No automatic continuation
or model deployment is enabled.

The next execution can submit the **existing `run-ready` bundle**. It must still
measure the five actual GPU probes and pass the fixed gates; the CPU rehearsal
does not establish GPU throughput or improved model accuracy.
