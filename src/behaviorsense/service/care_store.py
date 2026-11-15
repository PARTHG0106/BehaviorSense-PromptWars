"""Private SQLite care plans; video predictions never create intake confirmations."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time as wall_time, timedelta, timezone
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Callable

ZONES = {"Asia/Kolkata": timezone(timedelta(hours=5, minutes=30)), "UTC": timezone.utc}
SESSION_SECONDS = 12 * 60 * 60
CODE_ROUNDS = 600_000


class CareError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def identifier() -> str:
    return secrets.token_hex(16)


SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, allergies TEXT NOT NULL, care_note TEXT NOT NULL,
 email TEXT NOT NULL, timezone TEXT NOT NULL, reminders_enabled INTEGER NOT NULL,
 code_salt BLOB NOT NULL, code_hash BLOB NOT NULL, created_at REAL NOT NULL, version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
 token_hash TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 caregiver TEXT NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS login_attempts (
 key TEXT PRIMARY KEY, failures INTEGER NOT NULL, window_start REAL NOT NULL, blocked_until REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS medications (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 name TEXT NOT NULL, dose TEXT NOT NULL, instructions TEXT NOT NULL, times TEXT NOT NULL,
 start_day TEXT NOT NULL, active INTEGER NOT NULL, created_at REAL NOT NULL, version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS confirmations (
 profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE, medication_id TEXT NOT NULL,
 day TEXT NOT NULL, time TEXT NOT NULL, status TEXT NOT NULL, caregiver TEXT NOT NULL,
 note TEXT NOT NULL, confirmed_at REAL NOT NULL, medication_name TEXT NOT NULL, dose TEXT NOT NULL,
 PRIMARY KEY (profile_id, medication_id, day, time)
);
CREATE TABLE IF NOT EXISTS appointments (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 title TEXT NOT NULL, doctor TEXT NOT NULL, local_at TEXT NOT NULL, utc_at REAL NOT NULL,
 location TEXT NOT NULL, notes TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
 id INTEGER PRIMARY KEY AUTOINCREMENT, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
 event_key TEXT NOT NULL UNIQUE, recipient TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL,
 status TEXT NOT NULL, created_at REAL NOT NULL, sent_at REAL, error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS medications_owner ON medications(profile_id, active);
CREATE INDEX IF NOT EXISTS appointments_owner ON appointments(profile_id, utc_at);
CREATE INDEX IF NOT EXISTS confirmations_owner ON confirmations(profile_id, day);
"""


class CareStore:
    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.lock = threading.RLock()
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("UPDATE outbox SET status='failed',error='Delivery was interrupted. Check before retrying.' WHERE status='sending'")

    @contextmanager
    def db(self, *, write: bool = False):
        with self.lock:
            connection = sqlite3.connect(self.path, timeout=5)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            try:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    @staticmethod
    def profile_public(row) -> dict:
        return {key: row[key] for key in ("id", "name", "allergies", "care_note", "email", "timezone", "version")} | {
            "reminders_enabled": bool(row["reminders_enabled"]), "created_at": iso(row["created_at"])}

    def create_profile(self, values: dict, caregiver: str, code: str | None = None) -> dict:
        code = code or secrets.token_urlsafe(12)
        if not 10 <= len(code) <= 128:
            raise CareError("Use an access code with 10–128 characters.")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", code.encode(), salt, CODE_ROUNDS)
        now, profile_id = self.clock(), identifier()
        with self.db(write=True) as db:
            if db.execute("SELECT count(*) FROM profiles").fetchone()[0] >= 1000:
                raise CareError("This care server has reached its profile limit.", 429)
            db.execute("INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                profile_id, values["name"], values.get("allergies", ""), values.get("care_note", ""),
                values.get("email", ""), values.get("timezone", "Asia/Kolkata"),
                int(values.get("reminders_enabled", False)), salt, digest, now, 1))
        return self.new_session(profile_id, caregiver) | {"access_code": code}

    def new_session(self, profile_id: str, caregiver: str) -> dict:
        token = secrets.token_urlsafe(32)
        now = self.clock()
        with self.db(write=True) as db:
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (
                hashlib.sha256(token.encode()).hexdigest(), profile_id, caregiver, now + SESSION_SECONDS))
        return {"token": token, "expires_at": iso(now + SESSION_SECONDS), "profile": self.profile(profile_id),
                "caregiver": caregiver}

    def login(self, profile_id: str, code: str, caregiver: str, client: str) -> dict:
        now = self.clock()
        key = hashlib.sha256((client + ":" + profile_id).encode()).hexdigest()
        with self.db() as db:
            attempt = db.execute("SELECT * FROM login_attempts WHERE key=?", (key,)).fetchone()
            if attempt and attempt["blocked_until"] > now:
                raise CareError("Too many unsuccessful attempts. Try again in 15 minutes.", 429)
            row = db.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        salt = row["code_salt"] if row else b"unknown-profile!"
        expected = row["code_hash"] if row else bytes(32)
        digest = hashlib.pbkdf2_hmac("sha256", code.encode(), salt, CODE_ROUNDS)
        if not row or not hmac.compare_digest(digest, expected):
            with self.db(write=True) as db:
                attempt = db.execute("SELECT * FROM login_attempts WHERE key=?", (key,)).fetchone()
                count = attempt["failures"] + 1 if attempt and now - attempt["window_start"] < 900 else 1
                start = attempt["window_start"] if attempt and now - attempt["window_start"] < 900 else now
                db.execute("INSERT OR REPLACE INTO login_attempts VALUES (?,?,?,?)", (
                    key, count, start, now + 900 if count >= 5 else 0))
                db.execute("DELETE FROM login_attempts WHERE window_start < ?", (now - 3600,))
            raise CareError("The profile ID or access code is incorrect.", 401)
        with self.db(write=True) as db:
            db.execute("DELETE FROM login_attempts WHERE key=?", (key,))
        return self.new_session(profile_id, caregiver)

    def authenticate(self, token: str) -> dict:
        if len(token) > 128:
            raise CareError("Sign in to access this care profile.", 401)
        with self.db() as db:
            row = db.execute("SELECT profile_id, caregiver FROM sessions WHERE token_hash=? AND expires_at>?", (
                hashlib.sha256(token.encode()).hexdigest(), self.clock())).fetchone()
        if not row:
            raise CareError("Your session has expired. Sign in again.", 401)
        return dict(row)

    def logout(self, token: str) -> None:
        with self.db(write=True) as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def profile(self, profile_id: str) -> dict:
        with self.db() as db:
            row = db.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
        if not row:
            raise CareError("Care profile not found.", 404)
        return self.profile_public(row)

    def update_profile(self, profile_id: str, values: dict) -> dict:
        with self.db(write=True) as db:
            prior = db.execute("SELECT timezone FROM profiles WHERE id=?", (profile_id,)).fetchone()
            if prior and prior["timezone"] != values["timezone"]:
                scheduled = db.execute("SELECT count(*) FROM medications WHERE profile_id=?", (profile_id,)).fetchone()[0]
                scheduled += db.execute("SELECT count(*) FROM appointments WHERE profile_id=?", (profile_id,)).fetchone()[0]
                if scheduled:
                    raise CareError("Keep the existing time zone once schedules have been added. Review medication times with the care team.", 409)
            changed = db.execute("UPDATE profiles SET name=?,allergies=?,care_note=?,email=?,timezone=?,"
                                 "reminders_enabled=?,version=version+1 WHERE id=? AND version=?", (
                values["name"], values["allergies"], values["care_note"], values["email"], values["timezone"],
                int(values["reminders_enabled"]), profile_id, values["version"])).rowcount
            if not changed:
                raise CareError("The profile was updated elsewhere. Reload before editing.", 409)
        return self.profile(profile_id)

    def medications(self, profile_id: str) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM medications WHERE profile_id=? ORDER BY created_at", (profile_id,)).fetchall()
        return [dict(row) | {"times": json.loads(row["times"]), "active": bool(row["active"])} for row in rows]

    def save_medication(self, profile_id: str, values: dict, medication_id: str | None = None) -> dict:
        medication_id = medication_id or identifier()
        with self.db(write=True) as db:
            if "version" in values:
                changed = db.execute("UPDATE medications SET name=?,dose=?,instructions=?,times=?,start_day=?,"
                                     "active=?,version=version+1 WHERE id=? AND profile_id=? AND version=?", (
                    values["name"], values["dose"], values["instructions"], json.dumps(values["times"]),
                    values["start_day"], int(values["active"]), medication_id, profile_id, values["version"])).rowcount
                if not changed:
                    raise CareError("The medication was updated elsewhere or is unavailable. Reload the plan.", 409)
            else:
                if db.execute("SELECT count(*) FROM medications WHERE profile_id=?", (profile_id,)).fetchone()[0] >= 100:
                    raise CareError("A profile can contain at most 100 medications.", 429)
                db.execute("INSERT INTO medications VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    medication_id, profile_id, values["name"], values["dose"], values["instructions"],
                    json.dumps(values["times"]), values["start_day"], int(values["active"]), self.clock(), 1))
        return next(item for item in self.medications(profile_id) if item["id"] == medication_id)

    def deactivate_medication(self, profile_id: str, medication_id: str) -> None:
        with self.db(write=True) as db:
            changed = db.execute("UPDATE medications SET active=0,version=version+1 WHERE id=? AND profile_id=?", (
                medication_id, profile_id)).rowcount
            if not changed:
                raise CareError("Medication not found.", 404)

    def doses(self, profile_id: str, day: str | None = None) -> list[dict]:
        profile = self.profile(profile_id)
        zone = ZONES[profile["timezone"]]
        now = self.clock()
        chosen = date.fromisoformat(day) if day else datetime.fromtimestamp(now, zone).date()
        with self.db() as db:
            rows = db.execute("SELECT * FROM confirmations WHERE profile_id=? AND day=?", (profile_id, chosen.isoformat())).fetchall()
        confirmed = {(row["medication_id"], row["time"]): dict(row) for row in rows}
        result = []
        for medication in self.medications(profile_id):
            if not medication["active"] or chosen.isoformat() < medication["start_day"]:
                continue
            for at in medication["times"]:
                due = datetime.combine(chosen, wall_time.fromisoformat(at), tzinfo=zone).timestamp()
                record = confirmed.get((medication["id"], at))
                result.append({"medication_id": medication["id"], "name": record["medication_name"] if record else medication["name"],
                    "dose": record["dose"] if record else medication["dose"], "version": medication["version"],
                    "instructions": medication["instructions"], "day": chosen.isoformat(), "time": at,
                    "due_at": iso(due), "status": record["status"] if record else ("overdue" if now > due + 3600 else "pending"),
                    "caregiver": record["caregiver"] if record else "", "note": record["note"] if record else "",
                    "confirmed_at": iso(record["confirmed_at"]) if record else None,
                    "confirmation_source": "caregiver_report" if record else None})
        return sorted(result, key=lambda item: item["due_at"])

    def confirm_dose(self, profile_id: str, medication_id: str, values: dict, caregiver: str) -> dict:
        candidates = self.doses(profile_id, values["day"])
        target = next((item for item in candidates if item["medication_id"] == medication_id and item["time"] == values["time"]), None)
        if not target:
            raise CareError("This dose is not part of the active medication schedule.", 404)
        if datetime.fromisoformat(target["due_at"]).timestamp() > self.clock() + 900:
            raise CareError("Confirm a dose when it is due, or within 15 minutes before its scheduled time.")
        with self.db(write=True) as db:
            prior = db.execute("SELECT status FROM confirmations WHERE profile_id=? AND medication_id=? AND day=? AND time=?", (
                profile_id, medication_id, values["day"], values["time"])).fetchone()
            if prior and prior["status"] != values["status"]:
                raise CareError("This dose already has a different confirmation. Keep the original record and add a care note.", 409)
            db.execute("INSERT OR IGNORE INTO confirmations VALUES (?,?,?,?,?,?,?,?,?,?)", (
                profile_id, medication_id, values["day"], values["time"], values["status"], caregiver,
                values.get("note", ""), self.clock(), target["name"], target["dose"]))
        return next(item for item in self.doses(profile_id, values["day"])
                    if item["medication_id"] == medication_id and item["time"] == values["time"])

    def history(self, profile_id: str) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM confirmations WHERE profile_id=? ORDER BY confirmed_at DESC LIMIT 100", (profile_id,)).fetchall()
        return [dict(row) | {"confirmed_at": iso(row["confirmed_at"])} for row in rows]

    def appointments(self, profile_id: str) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM appointments WHERE profile_id=? ORDER BY utc_at", (profile_id,)).fetchall()
        return [dict(row) | {"due_at": iso(row["utc_at"])} for row in rows]

    def save_appointment(self, profile_id: str, values: dict, appointment_id: str | None = None) -> dict:
        profile = self.profile(profile_id)
        local = datetime.fromisoformat(values["local_at"])
        if local.tzinfo:
            raise CareError("Enter the appointment in the care profile's local time.")
        seconds = local.replace(tzinfo=ZONES[profile["timezone"]]).timestamp()
        appointment_id = appointment_id or identifier()
        with self.db(write=True) as db:
            if "version" in values:
                changed = db.execute("UPDATE appointments SET title=?,doctor=?,local_at=?,utc_at=?,location=?,notes=?,"
                                     "status=?,version=version+1 WHERE id=? AND profile_id=? AND version=?", (
                    values["title"], values["doctor"], values["local_at"], seconds, values["location"], values["notes"],
                    values["status"], appointment_id, profile_id, values["version"])).rowcount
                if not changed:
                    raise CareError("The appointment was updated elsewhere or is unavailable. Reload the plan.", 409)
            else:
                if db.execute("SELECT count(*) FROM appointments WHERE profile_id=?", (profile_id,)).fetchone()[0] >= 500:
                    raise CareError("A profile can contain at most 500 appointments.", 429)
                db.execute("INSERT INTO appointments VALUES (?,?,?,?,?,?,?,?,?,?)", (
                    appointment_id, profile_id, values["title"], values["doctor"], values["local_at"], seconds,
                    values["location"], values["notes"], values["status"], 1))
        return next(item for item in self.appointments(profile_id) if item["id"] == appointment_id)

    def cancel_appointment(self, profile_id: str, appointment_id: str) -> None:
        with self.db(write=True) as db:
            changed = db.execute("UPDATE appointments SET status='cancelled',version=version+1 WHERE id=? AND profile_id=?", (
                appointment_id, profile_id)).rowcount
            if not changed:
                raise CareError("Appointment not found.", 404)

    def queue_reminders(self, mode: str, *, profile_id: str | None = None) -> int:
        now, inserted = self.clock(), 0
        with self.db() as db:
            profiles = db.execute("SELECT id FROM profiles WHERE reminders_enabled=1 AND email!=''"
                                  + (" AND id=?" if profile_id else ""), (profile_id,) if profile_id else ()).fetchall()
        for row in profiles:
            owner = row["id"]
            profile = self.profile(owner)
            events = []
            for dose in self.doses(owner):
                if dose["status"] in {"taken", "skipped"}:
                    continue
                due = datetime.fromisoformat(dose["due_at"]).timestamp()
                kind = "upcoming" if due - 900 <= now < due else ("overdue" if now >= due + 3600 else None)
                if kind:
                    key = f"dose:{owner}:{dose['medication_id']}:{dose['version']}:{dose['day']}:{dose['time'].replace(':', '')}:{kind}"
                    subject = f"Medication {kind}: {profile['name']}"
                    body = (f"{dose['name']} — {dose['dose']}\nScheduled: {dose['day']} {dose['time']} ({profile['timezone']}).\n"
                            f"Instructions: {dose['instructions']}\nNo caregiver confirmation has been recorded. Please check the care plan.")
                    events.append((key, subject, body))
            for appointment in self.appointments(owner):
                if appointment["status"] != "scheduled":
                    continue
                due = appointment["utc_at"]
                kind = "upcoming" if due - 3600 <= now < due else ("follow-up" if due + 1800 <= now < due + 86400 else None)
                if kind:
                    key = f"appointment:{owner}:{appointment['id']}:{appointment['version']}:{appointment['local_at'].replace(':', '')}:{kind}"
                    events.append((key, f"Appointment {kind}: {profile['name']}",
                        f"{appointment['title']} with {appointment['doctor']}\n{appointment['local_at']} ({profile['timezone']})\n"
                        f"Location: {appointment['location']}\nPlease review or confirm the appointment in the care plan."))
            with self.db(write=True) as db:
                for key, subject, body in events:
                    inserted += db.execute("INSERT INTO outbox (profile_id,event_key,recipient,subject,body,status,created_at)"
                                          " VALUES (?,?,?,?,?,?,?) ON CONFLICT(event_key) DO UPDATE SET status=excluded.status"
                                          " WHERE outbox.status='preview' AND excluded.status='queued'", (
                        owner, key, profile["email"], subject, body, "preview" if mode == "preview" else "queued", now)).rowcount
        return inserted

    def outbox(self, profile_id: str) -> list[dict]:
        with self.db() as db:
            rows = db.execute("SELECT * FROM outbox WHERE profile_id=? ORDER BY created_at DESC LIMIT 100", (profile_id,)).fetchall()
        return [dict(row) | {"created_at": iso(row["created_at"]), "sent_at": iso(row["sent_at"]) if row["sent_at"] else None} for row in rows]

    def dispatch(self, send: Callable[[str, str, str], None]) -> int:
        with self.db(write=True) as db:
            rows = db.execute("SELECT * FROM outbox WHERE status='queued' ORDER BY id LIMIT 20").fetchall()
            for row in rows:
                db.execute("UPDATE outbox SET status='sending' WHERE id=? AND status='queued'", (row["id"],))
        sent = 0
        for row in rows:
            parts = row["event_key"].split(":")
            with self.db() as db:
                if parts[0] == "dose":
                    current = db.execute("SELECT active,version FROM medications WHERE id=? AND profile_id=?", (parts[2], row["profile_id"])).fetchone()
                    confirmed = db.execute("SELECT 1 FROM confirmations WHERE profile_id=? AND medication_id=? AND day=? AND time=?", (
                        row["profile_id"], parts[2], parts[4], parts[5][:2] + ":" + parts[5][2:])).fetchone()
                    valid = current and current["active"] and current["version"] == int(parts[3]) and not confirmed
                else:
                    current = db.execute("SELECT status,version FROM appointments WHERE id=? AND profile_id=?", (parts[2], row["profile_id"])).fetchone()
                    valid = current and current["status"] == "scheduled" and current["version"] == int(parts[3])
                profile = db.execute("SELECT email,reminders_enabled FROM profiles WHERE id=?", (row["profile_id"],)).fetchone()
                valid = valid and profile and profile["reminders_enabled"] and profile["email"] == row["recipient"]
            if not valid:
                with self.db(write=True) as db:
                    db.execute("UPDATE outbox SET status='cancelled' WHERE id=?", (row["id"],))
                continue
            try:
                send(row["recipient"], row["subject"], row["body"])
            except Exception:
                with self.db(write=True) as db:
                    db.execute("UPDATE outbox SET status='failed',error='Email delivery failed; check SMTP settings.' WHERE id=?", (row["id"],))
            else:
                with self.db(write=True) as db:
                    db.execute("UPDATE outbox SET status='sent',sent_at=? WHERE id=?", (self.clock(), row["id"]))
                sent += 1
        return sent
