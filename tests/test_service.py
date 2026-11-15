"""Tests for the FastAPI service.

The invariants worth testing here are architectural, not CRUD:

  - the perception boundary is enforced by the INTERFACE, not by client discipline
    (there must be no endpoint that accepts pixels)
  - `/report` cannot emit an unverified claim, even when the generator lies
  - `/report` refuses to invent a report for a day that was never analysed

Everything else (row counts, status codes) is incidental. Run:
    python tests/test_service.py
"""

from __future__ import annotations

import sys
import importlib
import json
import tempfile
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    print("SKIP: fastapi not installed (pip install -r requirements-serve.txt)")
    sys.exit(0)

from behaviorsense.data.simulator import BehaviourSimulator, PERSONAS  # noqa: E402
from behaviorsense.schemas import DailyFeatures, Role  # noqa: E402
from behaviorsense.service import api  # noqa: E402

D0 = date(2026, 2, 1)


def fresh_client(tmp: Path) -> TestClient:
    """Point the service at a throwaway DB.

    The redirect is ASSERTED, not assumed. `connect()` used to bind DB_PATH as a default
    argument, so this reassignment did nothing and every test wrote to the real
    data/behaviorsense.db - undetected for as long as row counts stayed under the page
    size. A fixture that silently fails to isolate makes every test above it vacuous.
    """
    api.DB_PATH = tmp / "test.db"
    with closing(api.connect()) as conn:
        landed = Path(conn.execute("PRAGMA database_list").fetchone()["file"]).resolve()
    assert landed == api.DB_PATH.resolve(), (
        f"service opened {landed}, not the temp DB {api.DB_PATH.resolve()} - tests are "
        "writing to the real database and are not isolated")
    return TestClient(api.app)


def seed_days(client: TestClient, n: int = 24) -> list[date]:
    """Analyse n simulated days through the HTTP layer."""
    sim = BehaviourSimulator(PERSONAS["regular_margaret"], seed=0)
    result = sim.generate(start=D0, n_days=n, anomalies=())
    days = []
    for feats in result.days:
        r = client.post(
            f"/analyze/day?day={feats.day.isoformat()}&role=resident",
            json=feats.model_dump(mode="json"),
        )
        assert r.status_code == 200, r.text
        days.append(feats.day)
    return days


def test_x1_health_and_no_pixel_endpoint():
    """No endpoint may accept image or video data.

    The "LLM never sees pixels" claim must hold because the interface makes it
    impossible, not because callers are careful. If a /ingest/video route ever appears,
    the claim silently becomes a convention.
    """
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        assert client.get("/health").json()["status"] == "ok"

        paths = client.get("/openapi.json").json()["paths"]
        banned = ("video", "image", "frame_bytes", "upload", "pixels")
        for p in paths:
            for word in banned:
                assert word not in p.lower(), f"endpoint {p} accepts raw media"

        # /ingest/frames must take FrameObservation (structured), not bytes.
        body = paths["/ingest/frames"]["post"]["requestBody"]["content"]
        assert "application/json" in body, "frames endpoint accepts non-JSON payloads"
        assert "multipart/form-data" not in body
        print(f"  X1 {len(paths)} endpoints, none accept raw media; "
              f"/ingest/frames is JSON-only")


def test_x2_analyze_then_report_is_verified():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = seed_days(client, 24)

        r = client.post(f"/report?day={days[-1].isoformat()}&role=resident")
        assert r.status_code == 200, r.text
        rep = r.json()
        assert rep["hallucination_rate"] == 0.0, rep
        assert rep["withheld_claims"] == 0
        assert rep["verified_claims"], "no claims returned"
        for c in rep["verified_claims"]:
            assert c["evidence_ref"].startswith("feat:")
        print(f"  X2 24 days analysed -> report with {len(rep['verified_claims'])} "
              f"verified claims, {rep['withheld_claims']} withheld, "
              f"rate {rep['hallucination_rate']:.1%}")


def test_x3_report_never_returns_unfaithful_claims():
    """Swap in a lying generator: the endpoint must withhold, not warn.

    This is the test that matters. A response that included false claims with a
    `verified: false` flag would be worse than useless - a caveat beside confident prose
    is not protection, because the prose is what gets remembered.
    """
    from behaviorsense.agents.reasoning.reporter import (
        CaregiverReporter,
        HallucinatingStubLLM,
        ReporterConfig,
    )

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = seed_days(client, 24)

        original = api._reporter
        try:
            api._reporter = CaregiverReporter(
                HallucinatingStubLLM(rate=1.0, seed=2), config=ReporterConfig()
            )
            r = client.post(f"/report?day={days[-1].isoformat()}&role=resident")
            assert r.status_code == 200, r.text
            rep = r.json()
            assert rep["verified_claims"] == [], (
                f"unverified claims leaked to the caller: {rep['verified_claims']}"
            )
            assert rep["withheld_claims"] > 0
            assert rep["hallucination_rate"] == 1.0
            print(f"  X3 fully-corrupted generator -> 0 claims returned, "
                  f"{rep['withheld_claims']} withheld, rate 100%")
        finally:
            api._reporter = original


def test_x4_report_refuses_unanalysed_day():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        seed_days(client, 10)
        future = (D0 + timedelta(days=99)).isoformat()
        r = client.post(f"/report?day={future}&role=resident")
        assert r.status_code == 404, f"invented a report for an unanalysed day: {r.text}"
        assert "analyze/day" in r.json()["detail"]
        print("  X4 unanalysed day -> 404 with a pointer to /analyze/day")


def test_x5_alerts_and_stats_round_trip():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        sim = BehaviourSimulator(PERSONAS["regular_margaret"], seed=1)
        from behaviorsense.data.simulator import AnomalyKind, InjectedAnomaly

        result = sim.generate(
            start=D0, n_days=40,
            anomalies=(InjectedAnomaly(kind=AnomalyKind.FALL_EVENT,
                                       onset_day=D0 + timedelta(days=30)),),
        )
        for feats in result.days:
            client.post(
                f"/analyze/day?day={feats.day.isoformat()}&role=resident",
                json=feats.model_dump(mode="json"),
            )

        alerts = client.get("/alerts?limit=50").json()
        assert alerts, "injected fall produced no alerts"
        assert all("kind" in a and "evidence" in a for a in alerts)

        s = client.get("/stats").json()
        assert s["days_analysed"] == 40, s
        # Compare against an unpaginated read: /alerts?limit=50 is a PAGE, and asserting
        # a total equals a page silently passes only while the total stays under it.
        every = client.get("/alerts?limit=10000").json()
        assert s["alerts"] == len(every), (
            f"stats says {s['alerts']} alerts, full listing has {len(every)}")
        assert len(alerts) == min(50, len(every)), (
            f"limit=50 returned {len(alerts)}, expected {min(50, len(every))}")
        # A fresh DB must hold only this test's rows.
        assert len(every) < 200, (
            f"{len(every)} alerts from 40 simulated days - implausible, and the signature "
            "of a leaked database carrying prior runs")
        print(f"  X5 40 days -> {s['alerts']} alerts persisted "
              f"(kinds: {sorted({a['kind'] for a in alerts})})")


def test_x6_segments_ingest_and_aggregate():
    """Segments in -> features derived server-side, no client-supplied features."""
    from datetime import datetime

    from behaviorsense.schemas import ActivitySegment

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        day = date(2026, 3, 2)
        base = datetime(2026, 3, 2, 8, 0, 0)
        segs = []
        for i, (name, aid, mins) in enumerate([
            ("walking", 0, 10), ("eating", 9, 30), ("sitting", 2, 60),
            ("walking", 0, 12), ("eating", 9, 25),
        ]):
            start = base + timedelta(minutes=i * 90)
            segs.append(ActivitySegment(
                segment_id=f"s{i}", track_id=0, role=Role.RESIDENT,
                activity_id=aid, activity_name=name,
                start_time=start, end_time=start + timedelta(minutes=mins),
                confidence=0.9, room="kitchen", n_windows=mins,
            ))
        r = client.post("/ingest/segments",
                        json=[s.model_dump(mode="json") for s in segs])
        assert r.status_code == 200 and r.json()["stored"] == 5, r.text

        r = client.post(f"/analyze/day?day={day.isoformat()}&role=resident")
        assert r.status_code == 200, r.text
        feats = r.json()["features"]
        assert feats["meal_events"] == 2, feats["meal_events"]
        assert feats["walking_duration_s"] == 22 * 60, feats["walking_duration_s"]
        print(f"  X6 5 segments -> meals={feats['meal_events']}, "
              f"walking={feats['walking_duration_s']:.0f}s aggregated server-side")


def test_x7_restart_preserves_reports_and_drift_history():
    from behaviorsense.agents.behaviour import BehaviourAnalyzer

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = BehaviourSimulator(PERSONAS["regular_margaret"], seed=3).generate(
            start=D0, n_days=29, anomalies=()).days
        oracle = BehaviourAnalyzer()
        for features in days[:-1]:
            oracle.analyze_day(features)
            response = client.post(f"/analyze/day?day={features.day}",
                                   json=features.model_dump(mode="json"))
            assert response.status_code == 200, response.text
        before = client.post(f"/report?day={days[-2].day}")
        assert before.status_code == 200, before.text
        db_path = api.DB_PATH
        # Reload the module as a fresh worker would; no analyser cache is copied.
        importlib.reload(api)
        api.DB_PATH = db_path
        client = TestClient(api.app)
        after = client.post(f"/report?day={days[-2].day}")
        assert after.status_code == 200 and after.json() == before.json(), after.text
        expected = oracle.analyze_day(days[-1])
        actual = client.post(f"/analyze/day?day={days[-1].day}",
                             json=days[-1].model_dump(mode="json"))
        assert actual.status_code == 200, actual.text
        assert actual.json()["deviations"] == expected.deviations
        assert actual.json()["drift_signals"] == expected.drift_signals
        print("  X7 restart retained report evidence and next-day baseline/CUSUM")


def test_x8_retries_and_corrections_do_not_duplicate_or_mix_roles():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        resident = DailyFeatures(subject_role=Role.RESIDENT, day=D0, fall_events=1)
        visitor = resident.model_copy(update={"subject_role": Role.VISITOR})
        for features in (resident, visitor):
            response = client.post(
                f"/analyze/day?day={D0}&role={features.subject_role.value}",
                json=features.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            assert response.json()["alerts"][0]["subject_role"] == features.subject_role.value
        initial_alerts = client.get("/alerts").json()
        for _ in range(3):
            retry = client.post(f"/analyze/day?day={D0}", json=resident.model_dump(mode="json"))
            assert retry.status_code == 200, retry.text
        assert client.get("/alerts").json() == initial_alerts
        assert client.get("/stats").json()["days_analysed"] == 2
        for role in ("resident", "visitor"):
            response = client.post(f"/report?day={D0}&role={role}")
            assert response.status_code == 200, response.text
        corrected = resident.model_copy(update={"fall_events": 0})
        response = client.post(f"/analyze/day?day={D0}", json=corrected.model_dump(mode="json"))
        assert response.status_code == 200 and not response.json()["alerts"], response.text
        remaining = client.get("/alerts").json()
        assert len(remaining) == 1 and remaining[0]["subject_role"] == "visitor"
        assert client.get("/stats").json()["reports"] == 1, "stale resident report survived correction"
        print("  X8 retries are idempotent; corrections remove stale records only for their role")


def test_x9_backfill_matches_chronological_analysis_and_refreshes_future_reports():
    from behaviorsense.agents.behaviour import BehaviourAnalyzer

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = BehaviourSimulator(PERSONAS["regular_margaret"], seed=2).generate(
            start=D0, n_days=26, anomalies=()).days
        # Insert the last day first, then fill history in reverse order.
        for features in reversed(days):
            response = client.post(f"/analyze/day?day={features.day}",
                                   json=features.model_dump(mode="json"))
            assert response.status_code == 200, response.text
        expected = None
        oracle = BehaviourAnalyzer()
        for features in days:
            expected = oracle.analyze_day(features)
        actual = client.post(f"/analyze/day?day={days[-1].day}",
                             json=days[-1].model_dump(mode="json"))
        assert actual.json()["deviations"] == expected.deviations
        assert actual.json()["drift_signals"] == expected.drift_signals
        assert client.post(f"/report?day={days[-1].day}").status_code == 200
        # A historical correction changes a later baseline, so that report must expire.
        earlier = days[-3].model_copy(update={"walking_duration_s": 0.0})
        response = client.post(f"/analyze/day?day={earlier.day}",
                               json=earlier.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        assert client.get("/stats").json()["reports"] == 0
        assert client.post(f"/report?day={days[-1].day}").status_code == 200
        print("  X9 backfill matches chronological analysis and invalidates changed future reports")


def test_x10_mismatched_request_identity_is_rejected_without_writes():
    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        for features in (
            DailyFeatures(subject_role=Role.RESIDENT, day=D0 + timedelta(days=1)),
            DailyFeatures(subject_role=Role.VISITOR, day=D0),
        ):
            response = client.post(f"/analyze/day?day={D0}", json=features.model_dump(mode="json"))
            assert response.status_code == 422, response.text
        assert client.get("/stats").json()["days_analysed"] == 0
        print("  X10 mismatched day/role rejected before persistence")


def test_x11_segment_coverage_is_explicit_and_visitors_are_counted():
    from datetime import datetime
    from behaviorsense.schemas import ActivitySegment

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        start = datetime.combine(D0, datetime.min.time())
        resident = ActivitySegment(
            segment_id="resident-walk", track_id=1, role=Role.RESIDENT,
            activity_id=0, activity_name="walking", start_time=start,
            end_time=start + timedelta(minutes=2), confidence=0.9, n_windows=1)
        visitor = resident.model_copy(update={"segment_id": "visitor-walk", "track_id": 2,
                                              "role": Role.VISITOR})
        client.post("/ingest/segments", json=[s.model_dump(mode="json") for s in (resident, visitor)])
        unknown = client.post(f"/analyze/day?day={D0}").json()["features"]
        assert unknown["observed_hours"] == 0.0, "a two-minute segment invented 24h coverage"
        assert unknown["visitor_count"] == 1
        assert unknown["walking_duration_s"] == 120, "visitor activity mixed with resident"
        known = client.post(f"/analyze/day?day={D0}&observed_hours=9&tracking_coverage=0.8")
        assert known.status_code == 200, known.text
        assert known.json()["features"]["observed_hours"] == 9
        assert known.json()["features"]["tracking_coverage"] == 0.8
        assert client.post(f"/analyze/day?day={D0}&observed_hours=25").status_code == 422
        print("  X11 unknown coverage stays unknown; supplied coverage and visitor context retained")


def test_x12_legacy_database_reconstructs_state_and_failed_replay_rolls_back():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        features = DailyFeatures(subject_role=Role.RESIDENT, day=D0, fall_events=1)
        # Legacy schema had daily features and alerts but no full analysed states.
        with closing(api.connect()) as conn, conn:
            conn.execute("INSERT INTO daily_features VALUES (?, ?, ?)",
                         (str(D0), "resident", features.model_dump_json()))
            conn.execute("DROP TABLE behaviour_states")
        response = client.post(f"/report?day={D0}")
        assert response.status_code == 200, response.text
        before = client.get("/alerts").json()
        original = api._persist_state
        def fail_after_write(conn, state):
            original(conn, state)
            raise RuntimeError("injected persistence failure")
        broken = TestClient(api.app, raise_server_exceptions=False)
        with patch.object(api, "_persist_state", side_effect=fail_after_write):
            response = broken.post(f"/analyze/day?day={D0}",
                                   json=features.model_copy(update={"fall_events": 0}).model_dump(mode="json"))
        assert response.status_code == 500
        assert client.get("/alerts").json() == before
        assert client.get("/stats").json()["reports"] == 1
        assert client.post(f"/report?day={D0}").status_code == 200
        print("  X12 legacy evidence restored; failed replay leaves prior records intact")


def test_x13_report_rejects_evidence_changed_during_generation():
    from unittest.mock import patch
    from behaviorsense.agents.reasoning.reporter import CaregiverReporter, FaithfulStubLLM

    class UpdatingLLM(FaithfulStubLLM):
        def generate(self, prompt, **kwargs):
            api.analyze_day(day=D0, role=Role.RESIDENT,
                            features=DailyFeatures(subject_role=Role.RESIDENT, day=D0, fall_events=2))
            return super().generate(prompt, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        features = DailyFeatures(subject_role=Role.RESIDENT, day=D0, fall_events=1)
        client.post(f"/analyze/day?day={D0}", json=features.model_dump(mode="json"))
        with patch.object(api, "_reporter", CaregiverReporter(UpdatingLLM())):
            response = client.post(f"/report?day={D0}")
        assert response.status_code == 409, response.text
        assert client.get("/stats").json()["reports"] == 0
        assert client.post(f"/report?day={D0}").status_code == 200
        print("  X13 stale generation refused; current evidence remains reportable")


def test_x14_rejected_claims_are_counted_and_unscorable_rates_are_json_null():
    from unittest.mock import patch
    from behaviorsense.agents.reasoning.reporter import CaregiverReporter, FaithfulStubLLM

    class DuplicatingLLM(FaithfulStubLLM):
        def generate(self, prompt, **kwargs):
            payload = json.loads(super().generate(prompt, **kwargs))
            claim = payload["claims"][0]
            payload["claims"] = [claim, {**claim, "text": "Invented value 9999.",
                                          "claimed_value": 9999.0}]
            return json.dumps(payload)

    class MalformedLLM(FaithfulStubLLM):
        def generate(self, prompt, **kwargs):
            return "no JSON was generated"

    with tempfile.TemporaryDirectory() as td:
        client = fresh_client(Path(td))
        days = seed_days(client, 10)
        with patch.object(api, "_reporter", CaregiverReporter(DuplicatingLLM())):
            response = client.post(f"/report?day={days[-1]}")
        assert response.status_code == 200, response.text
        assert response.json()["verified_claims"] == []
        assert response.json()["withheld_claims"] == 2
        assert response.json()["hallucination_rate"] is None
        assert response.json()["notes"]
        with patch.object(api, "_reporter", CaregiverReporter(MalformedLLM())):
            response = client.post(f"/report?day={days[-1]}")
        assert response.status_code == 200, response.text
        assert response.json()["parse_failed"] is True
        assert response.json()["hallucination_rate"] is None
        print("  X14 duplicate-ID claims withheld and counted; unscorable output is valid JSON")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            print(f"{fn.__name__}:")
            fn()
            print("  PASS")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
