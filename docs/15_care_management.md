# Shared elderly care management

Run `python web/care_backend.py` and open `http://127.0.0.1:8787`. Care profiles, medication schedules, caregiver confirmations, appointments and reminder previews require no GPU or hosted model. Activity observation remains available through the optional Kaggle/local-bridge workflow.

The care homepage is `web/index.html`; `/`, `/index.html` and `/care.html` open that same page. `/activity` and `/activity.html` open the separate observation interface. A short walkthrough and fictional demo make the care workflow available immediately.

## Code structure

- `care_store.py` owns profile persistence, schedules, confirmations and reminder event state.
- `care_api.py` validates typed requests and enforces authenticated profile access.
- `care_email.py` owns preview/SMTP transport configuration and delivery.
- `web/scripts/care.js` coordinates forms and renders the care workflow.
- `care-messages.js` contains the English, Hindi and Marathi interface dictionary.
- `care-summary.js` derives daily confirmation counts from the returned records.

`npm run format:check` verifies the care controller, dictionary, counts, markup, styles and browser fixtures. The GitHub code-quality job runs this check on every push.

## Profile access and storage

Each profile receives a random 128-bit ID and generated access code. Codes are stored as PBKDF2-HMAC-SHA256 hashes with a random salt and 600,000 rounds. Bearer sessions use random tokens, stored as SHA256 hashes, and expire after 12 hours. Login failures are throttled. Every private API operation derives its profile from the validated token; medication and appointment IDs cannot access another profile.

The private SQLite database defaults to `data/care.sqlite3`, which is Git-ignored and is not mounted as static content. SQL is parameterized. Profiles, prescriptions and appointments have versions; stale updates return a conflict instead of overwriting another caregiver's edits. The default server uses localhost. Any remote deployment needs HTTPS and operator access controls appropriate to personal care data.

## Medication and appointment logic

Caregivers enter prescribed medication names, doses, instructions, daily times and start dates. The profile time zone is India (`Asia/Kolkata`) or UTC. It is locked after schedules are added, preventing silent reinterpretation of their times.

Taken/skipped confirmations are explicit caregiver reports with a name and timestamp. Repeated confirmations are idempotent; conflicting changes are refused. The original medication name and dose remain in the confirmation history when the prescription is edited or stopped. Predictions from uploaded video never write intake confirmations.

Each confirmation can include an optional caregiver note. The original note is retained on an idempotent retry. The daily summary counts today's current schedule by confirmation status; it includes doses scheduled for later today and does not reconstruct historical adherence from an edited schedule. A print action exposes the current care plan through the browser's print dialog.

An overdue confirmation means no caregiver confirmation is recorded after the schedule's grace period. It does not prove medication was missed. Appointment records support edits, attended status and cancellation within the plan; external clinic booking is outside this prototype.

## Email reminders

The background worker checks every 30 seconds. Medication reminders start 15 minutes before a dose; overdue-confirmation reminders start 60 minutes after. Appointment reminders start 60 minutes before the visit, with follow-up reminders 30 minutes after an unconfirmed visit.

Unique event keys deduplicate each occurrence. Before sending, the worker checks the current medication/appointment version, confirmations, cancellations and the profile's reminder preference. Changed or completed events are suppressed. Interrupted and failed deliveries are identified without claiming success.

Default mode is `BS_MAIL_MODE=preview`; messages stay in the private outbox and no email is sent. SMTP mode requires local `BS_SMTP_HOST`, `BS_SMTP_FROM`, optional credentials, and port 587 (STARTTLS) or 465 (TLS). The recipient and reminder preference belong to the care profile. The server must remain running. Tests use a mocked SMTP transport and fictional addresses.

## API surface

`/api/care/profiles` creates a profile; `/api/care/login` opens shared access. Authenticated `/profile`, `/medications`, `/doses`, `/history`, `/appointments` and `/reminders` routes manage the plan. API responses have `Cache-Control: no-store`; the care page uses a same-origin CSP. The database and Python server sources are not exposed through the static routes.

Run the targeted checks with `python run_tests.py --suite test_care_service`.
