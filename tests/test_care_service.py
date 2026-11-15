"""Care-team isolation, immutable intake history, reminders and clean CPU setup."""
from datetime import datetime
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastapi.testclient import TestClient
from behaviorsense.service.care_api import create_care_app
from behaviorsense.service.care_email import MailSettings
from behaviorsense.service.care_store import CareStore, CODE_ROUNDS


class CareTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.now = datetime.fromisoformat("2026-11-15T03:15:00+00:00").timestamp()
        self.store = CareStore(Path(self.temporary.name) / "care.sqlite3", clock=lambda: self.now)
        self.client = TestClient(create_care_app(self.store, ROOT / "web", MailSettings()))
        self.code = "family-test-code-12345"

    def profile(self, name="Demo Resident", reminders=False):
        result = self.client.post("/api/care/profiles", json={"name": name, "caregiver": "Caregiver A",
            "email": "care-test@example.invalid" if reminders else "", "reminders_enabled": reminders,
            "access_code": self.code})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    @staticmethod
    def headers(session):
        return {"Authorization": "Bearer " + session["token"]}

    def medication(self, session, **updates):
        values = {"name": "Prescribed medicine (demo)", "dose": "Prescribed dose (demo)", "instructions": "Fictional fixture",
                  "times": ["09:00"], "start_day": "2026-11-15", "active": True, **updates}
        result = self.client.post("/api/care/medications", json=values, headers=self.headers(session))
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def confirm(self, session, medication, status="taken"):
        return self.client.post(f"/api/care/medications/{medication['id']}/confirm", headers=self.headers(session),
                                json={"day": "2026-11-15", "time": "09:00", "status": status})

    def test_profiles_use_salted_codes_and_hashed_expiring_sessions(self):
        a, b = self.profile(), self.profile("Second fictional resident")
        self.assertEqual(a["access_code"], self.code)
        with self.store.db() as db:
            rows = db.execute("SELECT * FROM profiles").fetchall()
            tokens = db.execute("SELECT token_hash FROM sessions").fetchall()
        self.assertNotEqual(rows[0]["code_salt"], rows[1]["code_salt"])
        self.assertNotEqual(rows[0]["code_hash"], rows[1]["code_hash"])
        self.assertEqual(hashlib.pbkdf2_hmac("sha256", self.code.encode(), rows[0]["code_salt"], CODE_ROUNDS), rows[0]["code_hash"])
        self.assertTrue(all(token[0] != a["token"] for token in tokens))
        self.assertNotIn("code_hash", a["profile"])
        self.now += 12 * 3600 + 1
        self.assertEqual(self.client.get("/api/care/profile", headers=self.headers(a)).status_code, 401)

    def test_shared_access_reads_the_same_profile_and_records_caregiver_alias(self):
        a = self.profile()
        joined = self.client.post("/api/care/login", json={"profile_id": a["profile"]["id"], "access_code": self.code,
                                                          "caregiver": "Caregiver B"})
        self.assertEqual(joined.status_code, 200)
        b = joined.json()
        self.assertEqual(b["profile"]["id"], a["profile"]["id"])
        med = self.medication(a)
        self.assertEqual(len(self.client.get("/api/care/medications", headers=self.headers(b)).json()), 1)
        result = self.confirm(b, med)
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["caregiver"], "Caregiver B")
        self.assertEqual(result.json()["confirmation_source"], "caregiver_report")

    def test_authentication_and_profile_isolation_cover_every_private_route(self):
        a, b = self.profile(), self.profile("Other resident")
        med = self.medication(a)
        for path in ("profile", "medications", "doses", "history", "appointments", "reminders"):
            self.assertEqual(self.client.get("/api/care/" + path).status_code, 401, path)
        self.assertEqual(self.client.get("/api/care/medications", headers=self.headers(b)).json(), [])
        self.assertEqual(self.confirm(b, med).status_code, 404)
        self.assertEqual(self.client.delete(f"/api/care/medications/{med['id']}", headers=self.headers(b)).status_code, 404)

    def test_repeated_wrong_codes_are_rate_limited_and_logout_revokes_access(self):
        a = self.profile()
        for _ in range(5):
            self.assertEqual(self.client.post("/api/care/login", json={"profile_id": a["profile"]["id"],
                "access_code": "wrong-code", "caregiver": "Caregiver B"}).status_code, 401)
        blocked = self.client.post("/api/care/login", json={"profile_id": a["profile"]["id"],
                "access_code": self.code, "caregiver": "Caregiver B"})
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.headers["retry-after"], "900")
        self.assertEqual(self.client.post("/api/care/logout", headers=self.headers(a)).status_code, 204)
        self.assertEqual(self.client.get("/api/care/profile", headers=self.headers(a)).status_code, 401)

    def test_profile_updates_are_versioned_and_prevent_timezone_reinterpretation(self):
        a = self.profile()
        p = a["profile"]
        values = {key: p[key] for key in ("name", "allergies", "care_note", "email", "timezone", "reminders_enabled", "version")}
        values["care_note"] = "Updated fictional care note"
        result = self.client.put("/api/care/profile", json=values, headers=self.headers(a))
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["version"], 2)
        self.assertEqual(self.client.put("/api/care/profile", json=values, headers=self.headers(a)).status_code, 409)
        self.medication(a)
        values.update(version=2, timezone="UTC")
        self.assertEqual(self.client.put("/api/care/profile", json=values, headers=self.headers(a)).status_code, 409)

    def test_intake_is_explicit_idempotent_and_preserves_original_dose_history(self):
        a = self.profile()
        med = self.medication(a)
        self.assertIsNone(self.client.get("/api/care/doses", headers=self.headers(a)).json()[0]["confirmation_source"])
        self.assertEqual(self.confirm(a, med).status_code, 200)
        self.assertEqual(self.confirm(a, med).status_code, 200)
        self.assertEqual(self.confirm(a, med, "skipped").status_code, 409)
        update = {key: med[key] for key in ("name", "dose", "instructions", "times", "start_day", "active", "version")}
        update.update(name="New prescribed name", dose="New prescribed dose")
        self.assertEqual(self.client.put(f"/api/care/medications/{med['id']}", json=update, headers=self.headers(a)).status_code, 200)
        history = self.client.get("/api/care/history", headers=self.headers(a)).json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["dose"], med["dose"])
        self.assertEqual(self.client.get("/api/care/doses", headers=self.headers(a)).json()[0]["dose"], med["dose"])

    def test_future_or_unscheduled_intake_cannot_be_confirmed(self):
        a = self.profile()
        med = self.medication(a, times=["20:00"])
        result = self.client.post(f"/api/care/medications/{med['id']}/confirm", headers=self.headers(a),
                                 json={"day": "2026-11-15", "time": "20:00", "status": "taken"})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.confirm(a, med).status_code, 404)

    def test_medication_stop_preserves_confirmation_audit(self):
        a = self.profile()
        med = self.medication(a)
        self.confirm(a, med)
        self.assertEqual(self.client.delete(f"/api/care/medications/{med['id']}", headers=self.headers(a)).status_code, 204)
        self.assertEqual(self.client.get("/api/care/doses", headers=self.headers(a)).json(), [])
        self.assertEqual(len(self.client.get("/api/care/history", headers=self.headers(a)).json()), 1)

    def test_appointments_use_profile_time_and_support_edit_and_cancel(self):
        a = self.profile()
        values = {"title": "Fictional care visit", "doctor": "Demo clinician", "local_at": "2026-11-15T10:00",
                  "location": "Example", "notes": "Fixture", "status": "scheduled"}
        result = self.client.post("/api/care/appointments", json=values, headers=self.headers(a))
        self.assertEqual(result.status_code, 201, result.text)
        appointment = result.json()
        self.assertEqual(appointment["due_at"], "2026-11-15T04:30:00+00:00")
        values.update(version=appointment["version"], status="attended")
        self.assertEqual(self.client.put(f"/api/care/appointments/{appointment['id']}", json=values, headers=self.headers(a)).status_code, 200)
        self.assertEqual(self.client.delete(f"/api/care/appointments/{appointment['id']}", headers=self.headers(a)).status_code, 204)

    def test_due_and_overdue_reminders_are_deduplicated_without_inventing_intake(self):
        a = self.profile(reminders=True)
        self.medication(a)
        self.assertEqual(self.store.queue_reminders("preview"), 1)
        self.assertEqual(self.store.queue_reminders("preview"), 0)
        self.now += 2 * 3600
        self.assertEqual(self.store.queue_reminders("preview"), 1)
        rows = self.store.outbox(a["profile"]["id"])
        self.assertEqual({row["status"] for row in rows}, {"preview"})
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.store.history(a["profile"]["id"]), [])
        self.assertEqual(self.store.doses(a["profile"]["id"])[0]["status"], "overdue")

    def test_confirmed_changed_or_cancelled_events_are_suppressed_before_delivery(self):
        a = self.profile(reminders=True)
        med = self.medication(a)
        self.store.queue_reminders("smtp")
        self.confirm(a, med)
        sender = Mock()
        self.assertEqual(self.store.dispatch(sender), 0)
        sender.assert_not_called()
        self.assertEqual(self.store.outbox(a["profile"]["id"])[0]["status"], "cancelled")

    def test_switching_from_preview_to_smtp_promotes_only_current_due_messages(self):
        a = self.profile(reminders=True)
        self.medication(a)
        self.assertEqual(self.store.queue_reminders("preview"), 1)
        self.assertEqual(self.store.queue_reminders("smtp"), 1)
        self.assertEqual(self.store.queue_reminders("smtp"), 0)
        self.assertEqual(self.store.outbox(a["profile"]["id"])[0]["status"], "queued")

    def test_email_delivery_records_success_once_and_failure_without_secret_details(self):
        a = self.profile(reminders=True)
        self.medication(a)
        self.store.queue_reminders("smtp")
        sender = Mock()
        self.assertEqual(self.store.dispatch(sender), 1)
        self.assertEqual(self.store.dispatch(sender), 0)
        sender.assert_called_once()
        self.assertEqual(self.store.outbox(a["profile"]["id"])[0]["status"], "sent")
        self.medication(a, name="Second fictional medicine")
        self.store.queue_reminders("smtp")
        failed_sender = Mock(side_effect=RuntimeError("sensitive transport diagnostic"))
        self.assertEqual(self.store.dispatch(failed_sender), 0)
        failed = next(row for row in self.store.outbox(a["profile"]["id"]) if row["status"] == "failed")
        self.assertNotIn("sensitive", failed["error"])

    def test_input_validation_origin_boundaries_and_private_cache_headers(self):
        self.assertEqual(self.client.post("/api/care/profiles", json={"name": "X\r\nInjected"}).status_code, 422)
        self.assertEqual(self.client.post("/api/care/profiles", json={"name": "Demo", "reminders_enabled": True}).status_code, 422)
        self.assertEqual(self.client.post("/api/care/profiles", json={"name": "Demo"}, headers={"Origin": "https://untrusted.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/care/profiles", json={"name": "Demo", "care_note": "x" * 17000}).status_code, 413)
        a = self.profile()
        result = self.client.get("/api/care/profile", headers=self.headers(a))
        self.assertEqual(result.headers["cache-control"], "no-store")
        self.assertEqual(result.headers["x-content-type-options"], "nosniff")
        self.assertIn("script-src 'self'", self.client.get("/").headers["content-security-policy"])
        bad = {"name": "Demo", "dose": "As prescribed", "times": ["09:00", "09:00"], "start_day": "2026-11-15"}
        self.assertEqual(self.client.post("/api/care/medications", json=bad, headers=self.headers(a)).status_code, 422)

    def test_sql_metacharacters_remain_data(self):
        a = self.profile("Demo'); DROP TABLE profiles; --")
        self.assertEqual(self.store.profile(a["profile"]["id"])["name"], "Demo'); DROP TABLE profiles; --")
        self.assertEqual(self.client.get("/api/care/profile", headers=self.headers(a)).status_code, 200)

    def test_private_database_is_not_served_by_static_routes(self):
        self.profile()
        self.assertEqual(self.client.get("/data/care.sqlite3").status_code, 404)
        self.assertEqual(self.client.get("/care_backend.py").status_code, 404)
        self.assertEqual(self.client.get("/care.html").status_code, 200)
        self.assertEqual(self.client.get("/activity").status_code, 200)

    def test_preview_transport_never_connects_and_live_transport_requires_tls(self):
        with patch.dict(os.environ, {"BS_MAIL_MODE": "preview", "BS_SMTP_PORT": "invalid"}), patch("smtplib.SMTP") as smtp:
            settings = MailSettings.from_environment()
            with self.assertRaises(RuntimeError):
                settings.send("example@example.invalid", "Fixture", "Body")
            smtp.assert_not_called()
        with patch("smtplib.SMTP") as smtp:
            instance = smtp.return_value.__enter__.return_value
            settings = MailSettings("smtp", "smtp.example.invalid", 587, "care@example.invalid", "fixture", "fixture-password")
            settings.send("test@example.invalid", "Fixture", "Body")
            instance.starttls.assert_called_once()
            instance.login.assert_called_once()
            instance.send_message.assert_called_once()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CareTests))
    total = result.testsRun - len(result.skipped)
    passed = total - len(result.failures) - len(result.errors)
    print(f"{passed}/{total} passed; {len(result.skipped)} skipped")
    raise SystemExit(0 if result.wasSuccessful() else 1)
