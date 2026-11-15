# Submission validation — 15 November 2026

Local validation covers **49 suites**, with **910/910 reported tests passed** and **5 skipped**. These totals combine the initial full run and corrective suite/check reruns on Windows with Python 3.14.3 and two CPU threads per test process.

The initial `python -X utf8 -u run_tests.py --all` run passed 45/49 suites. It exposed a startup-sensitive timeout fixture, notebook-reader/header assumptions, an unreported hybrid suite, and a training-suite timeout. The following corrective validations resolve those outcomes:

| Area | Final validation | Correction and scope |
|---|---|---|
| Runner isolation | 11/11 passed; 1 skipped | Allow Python startup in the heartbeat fixture while checking its original fixed deadline and process cleanup |
| Notebook contracts | 39/40 passed in the suite rerun, then the corrected header check passed 1/1 | Accept both string and line-list cell sources; require explicit online headers; inspect all discovered notebooks |
| Frozen RGB hybrid serving | 12/12 passed; 2 skipped | Execute the suite under the isolated runner and report its completed pytest counts |
| Training stack | 21/21 passed | Use bounded 96-window synthetic fixtures and no data-loader workers for the CPU imbalance test; retain the full model and six-epoch uninterrupted-versus-resumed comparison |
| Notebook 04 end to end | 1/1 passed | Execute its actual notebook cells against synthetic inputs |

Commands used for corrective suite runs:

```powershell
python -X utf8 -u run_tests.py --suite test_runner --suite test_notebooks
python -X utf8 -u run_tests.py --suite test_rgb_hybrid_serving
python -X utf8 -u run_tests.py --suite test_training
```

The final header check was selected with pytest inside a fresh source-only snapshot. Its result was `1 passed, 39 deselected`. Other passed suites from the full run cover perception, activity and routine processing, report verification and translation, service contracts, chunked uploads, concurrent workers and cancellation, model/data alignment, extraction/training orchestration, and JavaScript interface contracts.

## Skipped checks

- Two directory-symlink cases could not run under this Windows account.
- One real POSIX process-group cleanup case requires Linux.
- Two checkpoint-parity cases require the local model bundle, which is excluded from the isolated source snapshot.

These skipped checks are excluded from the passed-test count. Source snapshots exclude operator datasets, model weights, credentials and resident galleries.

## GPU evidence and public checks

The supplied Kaggle export records successful FP32 model-forward checks on both Tesla T4 GPUs. See [the recorded output](kaggle_backend_smoke_20261115.md). This confirms model execution and backend startup; it does not establish a new accuracy or clinical benchmark.

The staged source passed whitespace and Python syntax checks, contained no known credential/token-pattern matches or binary runtime assets, and the primary documentation's local links resolved. The included GitHub Actions workflow runs the full CPU suite on Ubuntu and Windows with Python 3.11 and Node.js 22; its results are separate from this local validation record.

## Competition snapshot checks

The [competition repository](https://github.com/PARTHG0106/BehaviorSense-PromptWars) preserves the application, model, training and test source from the original validated snapshot. Clean-environment CI exposed two undeclared dependencies and Windows temporary-directory aliases. The dependency lists now include `python-multipart` for upload routes and `requests` for coordinator retry handling. The test runner resolves its working directory and inherited temporary paths before launching suites.

All **9 affected/package suites passed: 142/142 reported tests, 3 skipped**, using an actual Windows 8.3 temporary-path alias. These checks cover runner isolation, notebook clip jobs, extraction, orchestration, feature stages, parallel training coordination, coverage provenance, RGB hybrid upload routes and the web contract. The skipped cases are the Windows directory-symlink fixture and two checkpoint-parity cases requiring local model weights. Full cross-platform CI is also enabled for the competition snapshot.

## Care-management and accessibility extension

The care API passed **17/17 tests** covering shared-profile authorization, protected access codes, session expiry, caregiver attribution, version conflicts, preserved intake history, appointments, reminder deduplication, preview-to-SMTP transition, stale-event suppression, TLS transport and private storage. A regression run passed **75/75 checks** across the care service, existing service API, notebook clip jobs, web contract and report-language UI before the final preview-transition case was added; the final care suite was then rerun successfully.

The real-browser accessibility test passed all six audited views with **zero axe violations**, no JavaScript errors, and successful keyboard, confirmation, language and mobile-reflow checks. Its source and pinned dependencies are included, and CI runs it separately. See [the accessibility record](accessibility_20261115.md).
