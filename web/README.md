# web/

Run the shared care profile, medication diary, appointments and reminders with:

```bash
python web/care_backend.py
# http://127.0.0.1:8787
```

The care platform uses a private, Git-ignored SQLite database. Its interface supports English, Hindi and Marathi, keyboard navigation, large text, high contrast and reduced motion. Email is preview-only unless SMTP is explicitly configured. See [care management](../docs/15_care_management.md) and [accessibility](../docs/16_accessibility.md).

The activity/report interface remains available at `http://127.0.0.1:8787/activity` and connects to the reporting bridge at `http://127.0.0.1:8899`.

The care homepage is `index.html`, served together with its private API by the care server.
The optional activity interface is `activity.html`. It uses static HTML, CSS and JavaScript
with no build step and can also be opened with a static server.

```bash
python -m http.server 5173 --directory web
# then http://localhost:5173/activity.html
```

## What it is

A single page with a report ledger, evidence checks and a video upload workflow. The
ledger's structured claims are verified in the browser: `scripts/verifier.js` is a port of
`src/behaviorsense/agents/reasoning/verifier.py` — same tolerances (2% relative on values,
2 percentage points on changes), same direction word lists, same treatment of a percentage
quoted against a zero baseline. The five cells beside each claim are real check results, and
the arithmetic under a withheld claim is the actual comparison that failed.

C1–C5 apply to the separate structured claims, including the text inside each claim. They
do not verify the free-text summary or recommendation. Those remain visible as model prose
and carry no arithmetic guarantee. Passing the checks establishes consistency with the
recorded evidence, not that the perception model or the clinical interpretation is correct.

The fault controls mirror `HallucinatingStubLLM`: the same six corruptions the Python test
suite injects, applied as fixed scripts rather than at random so the page is reproducible and
each slip can name the fault it was given.

One deliberate difference from the Python. When a citation does not resolve, the verifier
returns C2–C5 as `false` because there is nothing left to compare against; here they are drawn
as *not reached* instead, because showing five failures for one fault overstates what went
wrong. The verdict — unfaithful, withheld — is identical.

## Run the current inference path

### Caregiver report languages

Use **Report language** in the page header to choose **English, हिन्दी or मराठी**.
The choice is remembered on this browser. The caregiver report, its controls, upload
instructions and progress messages use the selected language; detailed research and
engineering sections remain in English.

The caregiver view contains only accepted claims. Hindi and Marathi numeric findings are
rendered from the original structured values with localized feature names and units; missing
comparison fields remain absent. The original English report and evidence audit remain
available in collapsible sections. The verifier's checks do not
certify the prose translation, summary, recommendation, or medical interpretation.

Detected activities remain reportable without baseline history. For example, 29.25 seconds
of cooking in a 30-second clip is an observed value; its baseline, percentage change, and
direction are unavailable. The verifier rejects comparisons without supporting evidence.
Restart the local bridge after updating the Python code, refresh the page, and run a new
analysis to apply this behavior to a previously generated report.

The stored replay has complete Hindi/Marathi prose translations and works offline. For a new
report, the local bridge's `POST /translate` translates only its summary and recommendation
using the existing OpenRouter model and credentials. Restart the bridge after updating the
code. The endpoint accepts `{language, summary, recommendation}`, supports `hi` and `mr`,
and rejects incomplete output or changed numeric tokens. It needs no GPU or additional key.
Requests are serialized across report widgets, and obsolete language responses cannot replace
the selected language. If prose translation fails, translated findings remain visible with a
retry button and access to the original English text.

`web/dev_backend.py` translates its known fixture reports without external requests; it
returns an explicit unavailable response for arbitrary prose. A direct connection to an older
Kaggle notebook or the separate structured-ingest API does not supply this translation route.

Run the relevant isolated checks with:

```bash
python run_tests.py --suite test_web_contract --suite test_translation --suite test_web_localization
```

### Connect the inference bridge

For the current RGB hybrid demo, run the supplied
[`notebookef04f4f9bb (1).ipynb`](../notebooks/notebookef04f4f9bb%20%281%29.ipynb).
Its [setup guide](../docs/13_rgb_hybrid_serving.md) lists the two public inputs.
It uses both T4s for independent clips and requires no Qwen dataset. Refresh the web page
after updating the local code to display the hybrid's model details.

The browser connects to `web/local_backend.py` on the operator's machine. That process sends
the video to the supplied Kaggle notebook for Agents 1–2, then runs Agent 3 and calls a hosted OpenRouter model
for Agent 4. The API keys stay on the operator's machine; the reporting provider receives
structured evidence, not video.

```
browser :5173  ──fetch──▶  local backend :8899  ──tunnel──▶  Kaggle GPU
                         Agent 3                          Agent 1: RTMO + tracking
                         Agent 4: OpenRouter call         Agent 2: RGB/pose/context hybrid
                         C1–C5 verification
```

1. Import `notebooks/notebookef04f4f9bb (1).ipynb` into Kaggle with **Internet On** and **GPU T4 x2**.
   Attach [penguin069/behaviorsense-rgb-hybrid-serving](https://www.kaggle.com/datasets/penguin069/behaviorsense-rgb-hybrid-serving)
   and [krishnagupta02468/behavioursense-ww](https://www.kaggle.com/datasets/krishnagupta02468/behavioursense-ww).
   These public datasets contain the GPU runtime, activity heads, thresholds, encoders and detectors.
   No separate code dataset, old Toyota training runs or Qwen model is required.
2. The last cell prints an address. **Leave that cell running** — the tunnel dies with it.
3. On the operator's machine, configure `OPENROUTER_API_KEY_LIST` (comma-separated) or
   `OPENROUTER_API_KEY_1..N` in the environment, then run:

   ```bash
   python web/local_backend.py --kaggle https://YOUR-SESSION.trycloudflare.com
   ```

4. Serve the page with the static-server command above. In the page header, connect to
   `http://127.0.0.1:8899`. If the notebook requires `BS_TOKEN`, pass that value to the local
   backend with `--token`; its upstream requests carry `X-BS-Token`.

The bridge calls `/video?stages=12`, rebuilds `ActivitySegment` objects, and uses the shared
`behaviorsense.service.staging` implementation for Agents 3–4. The supplied notebook loads the
saved RGB hybrid with frozen thresholds; `/health` and stage cards report the actual model.
The served tracker is `SimpleTracker`. The historical Qwen reporting evaluation and BF16 RGB
benchmark remain separate from this hosted-reporting, T4 FP32 demo.

The tunnel address changes each session, so restart the bridge with the new address. A direct
browser-to-Kaggle connection is also possible with a legacy notebook that has its own hosted reporter
configured, but the local bridge is the path that keeps reporting keys on this machine. The
Docker service in `src/behaviorsense/service/api.py` has a different structured-ingest API and
is not a substitute for these `/demo` and `/video` routes.

### Why both slow endpoints stream

The quick tunnel previously cut slow synchronous requests before inference completed. Both
`/demo` and `/video` now use newline-delimited JSON. Stages describe completed agent output;
heartbeats keep the connection active during longer computations and hosted-model calls.

```
{"event":"accepted"}                request accepted
{"event":"stage","stage":{...}}     completed agent output
{"event":"heartbeat",...}           computation is still running
{"event":"progress",...}            named waypoint, where emitted
{"event":"result","payload":{...}}  final payload
{"event":"error","detail":"..."}    error after streaming began
```

Errors after the response begins travel in-band. Pre-stream rejections use HTTP statuses;
when the bridge has already opened its own stream, it forwards upstream failures as error
events. The browser times out after **90 seconds without bytes**, not after a fixed total
duration. The bridge forwards GPU heartbeats and emits its own during local work. In the
current bridge, Agent 3 and Agent 4 cards arrive together after local reporting completes.

### What connecting buys

`Generate live` in the ledger runs `/demo` on the backend: the same simulator the evaluation
uses picks a day that alerts, the configured hosted model writes the report, and the response
carries the claims, the server's C1–C5 verdicts, **and the evidence table used to check them**.
This is live model output over simulated behavior, not a resident's observed history.

The page then re-derives the verdicts from that evidence with its own verifier and compares.
Agreement between Python and JavaScript is stated explicitly; disagreement produces a warning.
The video result instead displays the Python per-claim checks supplied in its staged payload.

### States, and why each is named

| indicator | what it means |
|---|---|
| `Replay` | no backend. The ledger runs a stored day, verified locally. Fully useful. |
| `Checking` | probing `/health`. |
| `Loading` | the backend answers, but its GPU-side models are not ready. |
| `Live` | ready. `/health` reported loaded streams, GPU information and the reporting model. |
| `No answer` | nothing responded. Session ended, or the wrong address. |
| `Wrong service` | something answered but did not identify as BehaviorSense. |
| `Backend error` | `/health` returned a load or upstream failure; inspect the local process and notebook. |

The notebook loads models in the background. The bridge's health response includes upstream
readiness, so a running local process cannot by itself make the page claim GPU availability.

## Upload a clip

The **On your own footage** band takes a video and runs **all four agents** over it, showing
each stage's own output where it was produced:

- **Agent 1** — **RTMO** extracts COCO-17 poses, `SimpleTracker` associates detections across
  frames, and **OSNet** can match them to an enrolled resident when weights and a gallery
  are available. Without a match, the subject is selected by a declared presence heuristic.
- **Agent 2** — **ST-GCN++** labels each person's activity independently, on their own
  windows, at the τ and temperature notebook 04 selected; one timeline per person, falls
  struck in pencil
- **Agent 3** — durations, transition counts, fall count and observed span measured from the
  clip, then robust-z and the alert rules
- **Agent 4** — the configured hosted model writes from those numbers alone, and **C1–C5 are shown
  per claim**, in the same `.slip`/`.gutter` markup the replay ledger uses. C5 asks whether
  the sentence actually quotes the figure the field records — the check the latest measured
  run justified, where C2 was zero and prose paraphrase was the largest failure category. A
  backend built before C5 renders a warning banner rather than letting the missing key read
  as a pass.

Each card names the schema type it hands on (`PersonObservation` → `ActivitySegment` →
`BehaviourState` → `CaregiverReport`), so the static pipeline diagram above is describing
*this run* rather than an idealised one.

Identity is shown by **weight and fill, not colour**. The palette has three hues and each
already means something — ink is verified, pencil is a correction, ochre is "watch this" — so
using pencil for "second person" would say that person is a mistake. The resident is drawn
heavy with filled joints, everyone else light with hollow ones, and the role is stated in mono
type beside the head, which is where this system puts facts.

A person who is tracked but too occluded to classify is drawn as a **dashed box labelled
"pose unusable"** rather than omitted. "Present, unreadable" and "not there" are different
facts, and Agent 2 abstains on the first rather than guessing.

### Enrollment and the gallery

An optional name in the upload form enrolls the selected subject from usable OSNet embeddings.
The bridge saves a normalized centroid and name in `gallery.json` outside the static `web/`
directory. On later uploads, it sends the saved gallery's names, roles and centroids to Kaggle
for matching. The persistent file is local; its contents are transmitted for inference.
To clear enrollment, stop the bridge, delete the local file and restart it; the running
process also holds the gallery in memory. The store contains identity embeddings rather
than raw crops, and is gitignored.

### The baseline is declared, because one clip cannot supply fourteen days

Agent 3 does two separable things and only one of them needs history:

| | needs history? | on one clip |
|---|---|---|
| feature extraction — durations, transition counts, falls, observed span | no | **real, measured from your video** |
| deviation detection — robust-z and CUSUM against a rolling baseline | yes | against a **declared simulated reference**; deviations are withheld when observation coverage is insufficient |

Staging primes the behavior analyzer with 21 simulated reference days and dates the clip
after that window. `baseline_provenance` records this in the payload, and `stages.js` draws
an **ochre banner above the figures**. Clips below the daily coverage requirement are marked
unreliable and do not display daily deviations. Measured clip features remain available.

C1–C5 are unaffected, and that is the point. Every claim is still checked against the state
computed from *this* video, so an invented figure, a wrong percentage, an inverted
direction or a sentence that never quotes its number is caught arithmetically exactly as it
is on the stored day. What a reference baseline cannot support is the clinical reading —
"mobility declined" — because the comparison point is not this person. Both facts ship in
the response and both are on screen.

### Limits, and why they are refusals rather than truncations

**100 MB per file and 36,000 retained frames**, about 30 minutes at the current 20 Hz target rate.
These are configurable application bounds: `BS_MAX_UPLOAD_MB` and `BS_MAX_FRAMES`.
MB denotes 1,048,576 bytes throughout the API; a file exactly at the bound is accepted.
The rate is matched to the selected checkpoint; sources below that rate are not upsampled,
so duration at the frame cap depends on the actual sampling rate. A longer upload is **rejected**, not silently
cut short: a report over the first twenty seconds of a ten-minute video, presented as a report
over the video, is the kind of quiet misrepresentation this project keeps finding and removing.

Select up to **two clips** together. Each has separate progress, video and report tabs.
On T4 x2, each active clip owns one GPU for pose, optional re-identification and its own
classifier instance. `BS_MAX_PARALLEL_CLIPS` defaults to 2 and may be set to 1; one visible
GPU always produces one worker. `/health` reports capacity, active and queued counts.
Two additional jobs can wait with streamed queue heartbeats. A full queue returns 429.
A disconnected client cannot release a GPU while its decoder is still running.

Files use `POST /uploads`, sequential `PUT /uploads/{id}?offset=N` chunks of at most 8 MB,
and `POST /uploads/{id}/complete` for the analysis stream. Upload ownership tokens,
exact byte offsets and bounded disk reservations prevent cross-request mixing.
This keeps a 100 MB file's multipart overhead from exceeding the tunnel body limit.
The original `/video` multipart route remains supported for compatible clients.

Update the code dataset and notebook 05 together, choose T4 x2, restart the notebook and
local bridge, then refresh the page. A running old notebook continues enforcing its old
limits. The local fixture server verifies UI behavior only; live T4 throughput needs a GPU run.

Decode runs in a **child process** (`behaviorsense.video.extract_isolated`). cv2 delegates to
ffmpeg, ffmpeg raises SIGSEGV/SIGABRT on malformed streams, and a signal is not an exception —
`try/except` cannot see it and the interpreter simply stops. Notebook 02 lost finished corpora
to exactly this. Inline, one bad upload would kill the kernel, the tunnel and the demo
together; behind the boundary it is a reported extraction error rather than a kernel crash.

## Developing the live path without a GPU

Booking a GPU session to check that a banner renders is absurd, so there is a stand-in:

```bash
python web/dev_backend.py          # http://127.0.0.1:8899
```

This fixture server and the local bridge use the same default port. Run only one there, or
set `BS_PORT` to another port for the fixture server and connect the page to that address.
It reads and discards the upload; the returned poses and activities do not describe the file.

Same routes, same shapes, no model. Three things about its fixtures are deliberate: the report
fixture contains one claim with a right number told backwards, so the page can be seen
*rejecting* something; its withheld note carries the **real failing arithmetic** in
`verifier.py`'s own wording, because a stub that returned no note would render the generic
fallback and quietly demonstrate the weaker version of the feature; and `/video` returns the
**full four-stage payload** over two synthetic figures — one of whom falls — with a
twelve-frame stretch where one pose is unusable, so the overlay's scaling, the two-person
distinction, the dashed-box state, every stage card and the five-cell C1–C5 check table are
all exercised rather than assumed.

It also **streams**, chunked and flushed per line, exactly as the notebook does. A stub that
replied in one shot would let the page be built against a shape the GPU never sends — the
specific failure this file exists to prevent. `BS_STREAM_DELAY` (default `0.6`, seconds per
line) fakes the per-agent wait so the progressive reveal is visible locally; the test suite
sets it to `0`.

`tests/test_web_contract.py` asserts the stub and notebook 05 emit the same `/video` keys
(W1–W4), the same NDJSON event names (W7), and — since the payload is built by real code
rather than described by it — **executes** the notebook's own staging function over real
schema objects (W6). A stub that drifts is worse than no stub.

## Security, stated plainly

The Kaggle API is **unauthenticated by default**. Anyone with the URL can post to it while the
session lives. That is acceptable for a demo you start and stop deliberately; it is not a
deployment. Set `BS_TOKEN` in the notebook to require a shared secret. In the split setup,
give it to the bridge with `--token`; for a direct Kaggle connection, use the page's token
field. The local bridge binds to `127.0.0.1`, accepts cross-origin requests and does not
enforce an incoming browser token.

The browser saves its connection address and optional token in localStorage. The local
bridge keeps OpenRouter credentials in its environment, uploads the video and saved gallery
contents to Kaggle, and sends structured evidence to the reporting provider. Keep those
distinct data flows in mind when deciding what footage to use in the demo.

Model-generated text is untrusted. Claim bodies and evidence references are escaped through
`escapeHtml` in `scripts/verifier.js` when rendered. The arithmetic verifier is a claim
consistency check; it does not provide authentication or validate arbitrary HTML.

## Files

```
index.html            care homepage and labelled care forms
activity.html         activity/report structure and copy
styles/tokens.css     colour, type and spacing tokens, with the reasoning behind them
styles/base.css       reset, type roles, the sticky rail, connection states
styles/components.css ledger, checks, stages, readouts, figures, dock, footage, close
scripts/verifier.js   C1-C5, ported from the Python, plus escapeHtml
scripts/fixtures.js   one analysed day + the six fault transforms
scripts/ledger.js     renders the report, resolves the checks, swaps in live payloads
scripts/charts.js     the two figures, hand-drawn SVG
scripts/api.js        the backend connection: validation, health polling, /demo, /video
scripts/dock.js       the paste-an-address panel and every state message
scripts/overlay.js    COCO-17 skeletons drawn over the playing video
scripts/stages.js     the four agent cards and the per-claim C1-C5 table
scripts/upload.js     the clip panel: upload, overlay, wiring
scripts/main.js       wiring
local_backend.py      Kaggle bridge, local behavior/reporting, and gallery persistence
dev_backend.py        a stand-in backend for developing the live path
```

When changing activity scripts or styles, update the version strings in `activity.html` so the import map
and stylesheet links request the new files. A hard reload (`Ctrl`/`Cmd`+`Shift`+`R`) also clears
stale module responses during development.

## Design notes

The direction is a diazo specification sheet on a desk: cool blue-grey print stock, content on
lifted sheets, hairline rules. The load-bearing decision is that **verification is the
difference between ink and pencil** — a claim that passes is set in ink and is
indistinguishable from the page's own voice, while a claim that fails is struck in red pencil
and stamped. So there is no green "verified" badge anywhere: the verified state is simply the
document. Pencil is the only saturated hue in the system, which is why the eye lands on the one
thing that matters.

Type carries three roles and each earns its place. **Instrument Sans** is the instrument's own
voice. **Newsreader** appears only inside the generated report, because the report is a
document and the chrome is not. **IBM Plex Mono** is reserved for measurement.

The connection states borrow the same logic rather than inventing a palette: ink for live,
ochre — the hue reserved for "watch this" — for loading, pencil for every failure. Nothing in
the system is green, so nothing about being connected is congratulatory.

There is no dark variant, declared with `color-scheme: light` rather than left to default. The
scheme rests on ink being permanent and pencil being a correction, and neither gesture survives
inversion — pencil on black is not the same mark. The ground is a mid grey-green rather than
white, which is what makes a light-only page tolerable at night.
