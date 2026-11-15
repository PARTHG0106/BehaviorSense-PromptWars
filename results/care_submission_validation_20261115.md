# Final care-platform checks — 15 November 2026

The care homepage, separate activity page, walkthrough, daily confirmation counts, optional caregiver notes and print action were checked before publication. The original model and notebook validation remains in [submission_validation_20261115.md](submission_validation_20261115.md).

## Targeted source validation

```powershell
python -X utf8 -u run_tests.py --suite test_care_service --suite test_web_localization --suite test_web_contract
```

All three suites passed in an isolated source copy: **61/61 tests**, with no skips, failures, timeouts or incomplete suites.

| Suite | Result | Scope |
|---|---|---|
| Shared care service | 19/19 | Profile isolation, salted codes, expiring/revoked sessions, concurrent updates, dose history and original notes, schedule boundaries, reminder deduplication, transport safety, private storage and correct page routes |
| Web video contract | 17/17 | Notebook/backend/browser payload compatibility and report contracts |
| JavaScript UI | 25/25 | Caregiver localization, report data, clip batches, uploads and daily counts that do not infer ingestion from unconfirmed records |

## Real browser validation

`npm run test:accessibility` passed using local Chrome and a temporary preview-only care server. Six views had **zero axe violations** and **zero JavaScript errors**. The exercised workflow created fictional medication records, saved taken and skipped confirmations with notes, verified the counts and history, marked an appointment attended, requested printing, navigated between pages, and checked three languages, keyboard access, enlarged text, high contrast and narrow-screen reflow. See [the audit](accessibility_20261115.md).

The print-button test intercepts the browser print request and checks print CSS; it does not send output to a physical printer. Email tests use preview or mocked SMTP, and browser fixtures contain no real patient data. These checks do not establish clinical validity or complete WCAG conformance.
