"""FastAPI service: schema objects in, verified reports out.

Design constraints carried over from the agent architecture:

  - Agents exchange versioned Pydantic schemas. SQLite stores daily features and full
    analysed states, so reports and chronological behaviour history survive a restart.
  - The perception boundary is enforced at the API level: `/ingest/frame` accepts
    `FrameObservation` (already-structured), never image bytes. There is deliberately no
    endpoint that accepts video - if one existed, the "LLM never sees pixels" claim would
    depend on client discipline rather than on the interface.
  - `/report` returns only verified claims. The separately named summary and
    recommendation fields are model prose and are outside the claim verifier's scope.

Run:
    PYTHONPATH=src uvicorn behaviorsense.service.api:app --reload
"""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from behaviorsense.agents.behaviour import BehaviourAnalyzer, aggregate_daily_features
from behaviorsense.agents.reasoning.reporter import (
    CaregiverReporter,
    FaithfulStubLLM,
    ReporterConfig,
)
from behaviorsense.schemas import (
    ActivitySegment,
    Alert,
    BehaviourState,
    DailyFeatures,
    FrameObservation,
    Role,
)

DB_PATH = Path("data/behaviorsense.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frames (
    frame_idx INTEGER, ts TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY, track_id INTEGER, role TEXT,
    activity_id INTEGER, start_time TEXT, end_time TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS daily_features (
    day TEXT, role TEXT, payload TEXT, PRIMARY KEY (day, role)
);
CREATE TABLE IF NOT EXISTS behaviour_states (
    day TEXT, role TEXT, payload TEXT, PRIMARY KEY (day, role)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, day TEXT, kind TEXT, severity TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY, day TEXT, hallucination_rate REAL, payload TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_day ON alerts(day);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(start_time);
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the service DB, resolving DB_PATH at CALL time.

    The default used to be `path: Path = DB_PATH`, which binds when the function is
    defined - so `api.DB_PATH = tmp / "test.db"` had no effect and every test run wrote
    to the real `data/behaviorsense.db` instead of its temp directory. That is why the
    committed database had accumulated 66 alert rows, and why `test_x5` finally failed:
    the count grew by 2 per run until it crossed the `?limit=50` page size. The tests
    were passing for 40 runs while proving nothing about isolation.
    """
    path = Path(path if path is not None else DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    with closing(conn.cursor()) as cur:
        cur.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


app = FastAPI(
    title="BehaviorSense AI",
    version="1.0.0",
    description="Identity-aware behaviour monitoring with verified LLM reporting.",
)

_reporter = CaregiverReporter(FaithfulStubLLM(), config=ReporterConfig())
_report_lock = Lock()  # Stub bind() and local model backends carry per-call state.


def _stored_state(conn: sqlite3.Connection, day: date, role: Role) -> BehaviourState | None:
    row = conn.execute(
        "SELECT payload FROM behaviour_states WHERE day = ? AND role = ?",
        (day.isoformat(), role.value),
    ).fetchone()
    return None if row is None else BehaviourState.model_validate_json(row["payload"])


def _persist_state(conn: sqlite3.Connection, state: BehaviourState) -> None:
    """Replace one day's derived records within the caller's transaction.

    Keep the original alerts/reports table layouts readable. Older databases carry
    role in the JSON payload rather than a dedicated column.
    """
    day, role = state.report_day.isoformat(), state.subject_role.value
    payload = state.model_dump_json()
    old = conn.execute(
        "SELECT payload FROM behaviour_states WHERE day = ? AND role = ?", (day, role)
    ).fetchone()
    if old is not None and old["payload"] == payload:
        return

    # Corrections must remove obsolete alerts and reports, including when a rule no
    # longer fires. Other roles on the same day retain their own records.
    for table, key in (("alerts", "alert_id"), ("reports", "report_id")):
        rows = conn.execute(f"SELECT {key}, payload FROM {table} WHERE day = ?", (day,))
        ids = [(row[key],) for row in rows
               if json.loads(row["payload"]).get("subject_role") == role]
        conn.executemany(f"DELETE FROM {table} WHERE {key} = ?", ids)
    conn.execute(
        "INSERT OR REPLACE INTO behaviour_states (day, role, payload) VALUES (?, ?, ?)",
        (day, role, payload),
    )
    conn.executemany(
        "INSERT OR REPLACE INTO alerts (alert_id, day, kind, severity, payload) "
        "VALUES (?, ?, ?, ?, ?)",
        [(a.alert_id, day, a.kind.value, a.severity.value, a.model_dump_json())
         for a in state.alerts],
    )


def _replay_role(conn: sqlite3.Connection, role: Role) -> dict[date, BehaviourState]:
    """Rebuild through Agent 3 itself, in date order, without duplicating its maths.

    One row per day makes retries idempotent. Replaying also corrects later baselines,
    sustained-deviation windows and CUSUM after an earlier day is added or corrected.
    SQLite serializes the caller's write transaction across threads and workers.
    """
    rows = conn.execute(
        "SELECT day, payload FROM daily_features WHERE role = ? ORDER BY day", (role.value,)
    ).fetchall()
    analyzer = BehaviourAnalyzer()
    states = {}
    for row in rows:
        features = DailyFeatures.model_validate_json(row["payload"])
        if features.day.isoformat() != row["day"] or features.subject_role != role:
            raise ValueError("stored daily features disagree with their day or role")
        state = analyzer.analyze_day(features, subject_role=role)
        # Agent 3 gives each alert a UUID. Use a stable service ID so replaying the
        # same day/rule does not create a new incident or invalidate an identical report.
        for alert in state.alerts:
            alert.alert_id = str(uuid5(
                NAMESPACE_URL,
                f"behaviorsense/alerts/{role.value}/{row['day']}/{alert.rule_name}",
            ))
        _persist_state(conn, state)
        states[features.day] = state
    return states


def _state_for_report(day: date, role: Role) -> BehaviourState | None:
    with closing(connect()) as conn:
        state = _stored_state(conn, day, role)
        if state is not None:
            return state
        # Additive migration for databases written before full states were persisted.
        # Only an already analysed daily_features row can authorize reconstruction.
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT 1 FROM daily_features WHERE day = ? AND role = ?",
                (day.isoformat(), role.value),
            ).fetchone()
            return _replay_role(conn, role).get(day) if exists else None


class IngestAck(BaseModel):
    stored: int
    kind: str


class DayResponse(BaseModel):
    day: date
    features: DailyFeatures | None = None
    alerts: list[Alert] = []
    deviations: dict[str, float] = {}
    drift_signals: dict[str, float] = {}


class ReportResponse(BaseModel):
    report_day: date
    summary: str
    verified_claims: list[dict[str, Any]]
    withheld_claims: int
    hallucination_rate: float | None
    recommendation: str
    escalate: bool
    model_name: str
    constrained_decoding: bool
    parse_failed: bool = False
    notes: list[str] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "behaviorsense", "version": app.version}


@app.post("/ingest/frames", response_model=IngestAck)
def ingest_frames(frames: list[FrameObservation]) -> IngestAck:
    """Accept Agent 1 output. Structured observations only - never image data."""
    with closing(connect()) as conn:
        conn.executemany(
            "INSERT INTO frames (frame_idx, ts, payload) VALUES (?, ?, ?)",
            [(f.frame_idx, f.timestamp.isoformat(), f.model_dump_json()) for f in frames],
        )
        conn.commit()
    return IngestAck(stored=len(frames), kind="frame_observation")


@app.post("/ingest/segments", response_model=IngestAck)
def ingest_segments(segments: list[ActivitySegment]) -> IngestAck:
    """Accept Agent 2 output."""
    with closing(connect()) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO segments "
            "(segment_id, track_id, role, activity_id, start_time, end_time, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(s.segment_id, s.track_id, s.role.value, s.activity_id,
              s.start_time.isoformat(), s.end_time.isoformat(), s.model_dump_json())
             for s in segments],
        )
        conn.commit()
    return IngestAck(stored=len(segments), kind="activity_segment")


@app.post("/analyze/day", response_model=DayResponse)
def analyze_day(
    day: date, role: Role = Role.RESIDENT, features: DailyFeatures | None = None,
    observed_hours: float = Query(default=0.0, ge=0.0, le=24.0),
    tracking_coverage: float = Query(default=1.0, ge=0.0, le=1.0),
) -> DayResponse:
    """Run Agent 3 for one day.

    If `features` is omitted, they are aggregated from stored segments for that day -
    which is the normal path; passing features directly exists for replay and testing.
    Segment ingestion does not establish camera uptime: specify observed_hours and
    tracking_coverage for that path, or absence-based comparisons stay suppressed.
    Features supplied in the body carry their own coverage and must match day/role.
    """
    if features is not None and (features.day != day or features.subject_role != role):
        raise HTTPException(422, "features.day and features.subject_role must match day and role")
    if features is None:
        with closing(connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM segments WHERE date(start_time) = ?",
                (day.isoformat(),),
            ).fetchall()
        segments = [ActivitySegment.model_validate_json(r["payload"]) for r in rows]
        if not any(s.role == role for s in segments):
            raise HTTPException(404, f"no segments stored for {day} / {role.value}")
        features = aggregate_daily_features(
            segments, day=day, subject_role=role, observed_hours=observed_hours,
            tracking_coverage=tracking_coverage,
        )

    with closing(connect()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = _stored_state(conn, day, role)
        if previous is not None and previous.today == features:
            state = previous
        else:
            conn.execute(
                "INSERT OR REPLACE INTO daily_features (day, role, payload) VALUES (?, ?, ?)",
                (day.isoformat(), role.value, features.model_dump_json()),
            )
            state = _replay_role(conn, role)[day]
    return DayResponse(
        day=state.report_day,
        features=state.today,
        alerts=state.alerts,
        deviations=state.deviations,
        drift_signals=state.drift_signals,
    )


@app.get("/alerts", response_model=list[Alert])
def list_alerts(since: date | None = None, limit: int = 100) -> list[Alert]:
    q = "SELECT payload FROM alerts"
    params: tuple = ()
    if since:
        q += " WHERE day >= ?"
        params = (since.isoformat(),)
    q += " ORDER BY day DESC LIMIT ?"
    params = (*params, limit)
    with closing(connect()) as conn:
        rows = conn.execute(q, params).fetchall()
    return [Alert.model_validate_json(r["payload"]) for r in rows]


@app.post("/report", response_model=ReportResponse)
def generate_report(day: date, role: Role = Role.RESIDENT) -> ReportResponse:
    """Generate and VERIFY a caregiver report. Unverified claims never leave this call."""
    state = _state_for_report(day, role)
    if state is None:
        raise HTTPException(
            404,
            f"no analysed state for {day}. POST /analyze/day first - reports are "
            "generated from stored state, never from raw input.",
        )

    with _report_lock:
        outcome = _reporter.report(state)
    faithful = {v.claim_id for v in outcome.report.verifications if v.is_faithful}
    rate = outcome.hallucination_rate
    rate = rate if outcome.report.claims and math.isfinite(rate) else None
    with closing(connect()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        current = _stored_state(conn, day, role)
        if current != state:
            raise HTTPException(409, "analysis changed during report generation; request the report again")
        conn.execute(
            "INSERT OR REPLACE INTO reports (report_id, day, hallucination_rate, payload) "
            "VALUES (?, ?, ?, ?)",
            (outcome.report.report_id, day.isoformat(), rate,
             outcome.report.model_dump_json()),
        )

    return ReportResponse(
        report_day=day,
        summary=outcome.report.summary,
        verified_claims=[
            json.loads(c.model_dump_json()) for c in outcome.report.claims
            if c.claim_id in faithful
        ],
        withheld_claims=outcome.n_emitted_claims - len(faithful),
        hallucination_rate=rate,
        recommendation=outcome.report.recommendation,
        escalate=outcome.report.escalate,
        model_name=outcome.report.model_name,
        constrained_decoding=outcome.report.constrained_decoding,
        parse_failed=outcome.parse_failed,
        notes=outcome.notes,
    )


@app.get("/stats")
def stats() -> dict[str, Any]:
    with closing(connect()) as conn:
        def count(t: str) -> int:
            return conn.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"]
        return {
            "frames": count("frames"),
            "segments": count("segments"),
            "days_analysed": count("daily_features"),
            "alerts": count("alerts"),
            "reports": count("reports"),
            "generated_at": datetime.now().isoformat(),
        }
