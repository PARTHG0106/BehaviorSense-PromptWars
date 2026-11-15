# Saved RGB progress — 16 October 2026

Work is saved at the user's request: “just save the work with whatever improvement we have achieved.” Further experiments are on hold. This save point supersedes the continuation intent in older experiment records, which remain intact as audit evidence.

| Completed comparison | mAP | TV + drinking F1 | Evidence |
|---|---:|---:|---|
| Frozen test benchmark, 15 October | 65.29% → **69.21%** | 19.07% → **28.12%** | [Verified test report](rgb_improvement_20261015.md) |
| Fitting coverage study, 16 October | 41.44% → **58.25%** | 4.53% → **13.88%** | [Verified coverage report](rgb_fitting_coverage_20261016.md) |

These comparisons use different cohorts. The test benchmark covers 153,537 windows from 185 videos and seven test subjects; the coverage study evaluates 217,090 windows from 274 videos and nine fitting subjects across three subject-separated rotations. The test result remains the latest verified benchmark. Both results are exploratory, and their pair quality gates remain unmet. The serving model has not been replaced.

The local checkpoint contains the current project source, documentation and tests, all six coverage-study heads with their calibration thresholds and predictions, the three frozen hybrid component checkpoints, their bound supporting files, and the completed studies' reports, protocols and audits. The unfinished four-state study's source and tests are included as drafts. Current working files and Git staging are preserved.

- [Local backup archive](../artifacts/rgb-next-20261016/saved-progress-20261016/behaviorsense-progress-20261016.zip)
- [File hashes and backup verification](../artifacts/rgb-next-20261016/saved-progress-20261016/manifest.json)
- [Current saved state](../artifacts/rgb-next-20261016/saved_progress.json)
- [Coverage model checkpoints and thresholds](../artifacts/rgb-next-20261016/coverage-study-r2/run/output/study/)
- [Unfinished four-state study](../artifacts/rgb-next-20261016/joint-model/)

The save verifies the existing audit bindings and every downloaded coverage output, then reads every archive entry back and checks its SHA-256 hash. It does not repeat training or benchmark scoring. The four-state additions receive a syntax check only; the complete new study has not been validated, frozen or submitted, and no trained pair-model result is claimed.

The archive preserves repository-relative paths and can be extracted into a separate directory for recovery. It is a local private backup: credentials, machine settings, raw datasets and installed runtimes are excluded. The complete RGB feature caches remain in the accepted private Kaggle output, version 1 of `krishnagupta02468/bs-rgb-fit-coverage-20261016`; the temporary combined feature matrix was not exported. Reproduction still requires those data and runtime dependencies.

Resume only after a new user request, starting from this note and the saved-state record. Keep the existing scored benchmark closed to retuning and review the unfinished study before any new run.
