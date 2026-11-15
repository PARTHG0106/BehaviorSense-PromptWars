"""Authenticated care plans and accessible UI, independent of GPU inference."""
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
import logging
import re
import threading
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .care_email import MailSettings
from .care_store import CareError, CareStore, ZONES


class Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProfileBody(Body):
    name: str = Field(min_length=1, max_length=80)
    allergies: str = Field(default="", max_length=1000)
    care_note: str = Field(default="", max_length=2000)
    email: str = Field(default="", max_length=254)
    timezone: Literal["Asia/Kolkata", "UTC"] = "Asia/Kolkata"
    reminders_enabled: bool = False

    @field_validator("name", "email")
    @classmethod
    def single_line(cls, value):
        if any(ord(c) < 32 for c in value):
            raise ValueError("Use a single line without control characters.")
        return value.strip()

    @model_validator(mode="after")
    def contact(self):
        if not self.name or (self.email and not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", self.email)):
            raise ValueError("Enter a name and a valid optional email address.")
        if self.reminders_enabled and not self.email:
            raise ValueError("An email address is required to enable reminders.")
        return self


class CreateProfile(ProfileBody):
    caregiver: str = Field(default="Family member", min_length=1, max_length=80)
    access_code: str | None = Field(default=None, min_length=10, max_length=128)

    @field_validator("caregiver")
    @classmethod
    def caregiver_name(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Enter a single-line caregiver name.")
        return value.strip()


class UpdateProfile(ProfileBody):
    version: int = Field(ge=1)


class LoginBody(Body):
    profile_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    access_code: str = Field(min_length=1, max_length=128)
    caregiver: str = Field(min_length=1, max_length=80)

    @field_validator("caregiver")
    @classmethod
    def caregiver_name(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Enter a single-line caregiver name.")
        return value.strip()


class MedicationBody(Body):
    name: str = Field(min_length=1, max_length=120)
    dose: str = Field(min_length=1, max_length=120)
    instructions: str = Field(default="", max_length=600)
    times: list[str] = Field(min_length=1, max_length=8)
    start_day: date
    active: bool = True

    @field_validator("times")
    @classmethod
    def valid_times(cls, values):
        if any(not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) for value in values) or len(set(values)) != len(values):
            raise ValueError("Use unique 24-hour times such as 08:00 and 20:00.")
        return sorted(values)

    @field_validator("name", "dose")
    @classmethod
    def nonempty(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("Enter a single-line medication name and prescribed dose.")
        return value.strip()


class UpdateMedication(MedicationBody):
    version: int = Field(ge=1)


class ConfirmBody(Body):
    day: date
    time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    status: Literal["taken", "skipped"]
    note: str = Field(default="", max_length=600)


class AppointmentBody(Body):
    title: str = Field(min_length=1, max_length=120)
    doctor: str = Field(default="", max_length=120)
    local_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")
    location: str = Field(default="", max_length=250)
    notes: str = Field(default="", max_length=1000)
    status: Literal["scheduled", "attended", "cancelled"] = "scheduled"

    @field_validator("local_at")
    @classmethod
    def calendar_time(cls, value):
        datetime.fromisoformat(value)
        return value

    @field_validator("title", "doctor")
    @classmethod
    def single_line(cls, value, info):
        if any(ord(c) < 32 for c in value):
            raise ValueError("Use a single line without control characters.")
        if info.field_name == "title" and not value.strip():
            raise ValueError("Enter an appointment title.")
        return value.strip()


class UpdateAppointment(AppointmentBody):
    version: int = Field(ge=1)


def create_care_app(store: CareStore, web: Path, mail: MailSettings | None = None) -> FastAPI:
    mail = mail or MailSettings()
    stop = threading.Event()
    login_rates = {}
    rate_lock = threading.Lock()

    def reminders():
        while not stop.wait(30):
            try:
                store.queue_reminders(mail.mode)
                if mail.mode == "smtp":
                    store.dispatch(mail.send)
            except Exception:
                logging.getLogger(__name__).warning("Reminder processing failed; the next background check will retry.")

    @asynccontextmanager
    async def lifespan(app):
        worker = threading.Thread(target=reminders, name="care-reminders", daemon=True)
        worker.start()
        yield
        stop.set()
        worker.join(timeout=12)

    app = FastAPI(title="BehaviorSense Care", lifespan=lifespan)
    app.state.store, app.state.mail = store, mail
    bearer = HTTPBearer(auto_error=False)

    @app.exception_handler(CareError)
    async def care_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status,
                            headers={"Cache-Control": "no-store", **({"Retry-After": "900"} if error.status == 429 else {})})

    @app.middleware("http")
    async def private_responses(request: Request, call_next):
        if request.url.path.startswith("/api/care/"):
            try:
                if int(request.headers.get("content-length", "0")) > 16_384:
                    return JSONResponse({"detail": "Request is too large."}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid request length."}, status_code=400)
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse({"detail": "Use the care app from its own server address."}, status_code=403)
            if request.url.path in {"/api/care/login", "/api/care/profiles"}:
                client = request.client.host if request.client else "local"
                now = store.clock()
                with rate_lock:
                    count, start = login_rates.get(client, (0, now))
                    if now - start >= 60:
                        count, start = 0, now
                    if count >= 20:
                        return JSONResponse({"detail": "Too many access requests. Try again in one minute."}, status_code=429, headers={"Retry-After": "60"})
                    if len(login_rates) > 2048:
                        login_rates.clear()
                    login_rates[client] = (count + 1, start)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path in {"/", "/care.html"}:
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
        if request.url.path.startswith("/api/care/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def identity(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not credentials:
            raise HTTPException(401, "Sign in to access a care profile.")
        return store.authenticate(credentials.credentials)

    @app.get("/api/care/health")
    def health():
        return {"ready": True, "mail_mode": mail.mode, "confirmation_source": "caregiver_report"}

    @app.post("/api/care/profiles", status_code=201)
    def create(body: CreateProfile):
        return store.create_profile(body.model_dump(exclude={"access_code", "caregiver"}), body.caregiver, body.access_code)

    @app.post("/api/care/login")
    def login(body: LoginBody, request: Request):
        return store.login(body.profile_id, body.access_code, body.caregiver, request.client.host if request.client else "local")

    @app.post("/api/care/logout", status_code=204)
    def logout(credentials: HTTPAuthorizationCredentials = Depends(bearer), who=Depends(identity)):
        store.logout(credentials.credentials)

    @app.get("/api/care/profile")
    def profile(who=Depends(identity)):
        return store.profile(who["profile_id"]) | {"caregiver": who["caregiver"]}

    @app.put("/api/care/profile")
    def update_profile(body: UpdateProfile, who=Depends(identity)):
        return store.update_profile(who["profile_id"], body.model_dump())

    @app.get("/api/care/medications")
    def medications(who=Depends(identity)):
        return store.medications(who["profile_id"])

    @app.post("/api/care/medications", status_code=201)
    def add_medication(body: MedicationBody, who=Depends(identity)):
        return store.save_medication(who["profile_id"], body.model_dump(mode="json"))

    @app.put("/api/care/medications/{medication_id}")
    def edit_medication(medication_id: str, body: UpdateMedication, who=Depends(identity)):
        return store.save_medication(who["profile_id"], body.model_dump(mode="json"), medication_id)

    @app.delete("/api/care/medications/{medication_id}", status_code=204)
    def stop_medication(medication_id: str, who=Depends(identity)):
        store.deactivate_medication(who["profile_id"], medication_id)

    @app.get("/api/care/doses")
    def doses(day: date | None = None, who=Depends(identity)):
        return store.doses(who["profile_id"], day.isoformat() if day else None)

    @app.post("/api/care/medications/{medication_id}/confirm")
    def confirm(medication_id: str, body: ConfirmBody, who=Depends(identity)):
        return store.confirm_dose(who["profile_id"], medication_id, body.model_dump(mode="json"), who["caregiver"])

    @app.get("/api/care/history")
    def history(who=Depends(identity)):
        return store.history(who["profile_id"])

    @app.get("/api/care/appointments")
    def appointments(who=Depends(identity)):
        return store.appointments(who["profile_id"])

    @app.post("/api/care/appointments", status_code=201)
    def add_appointment(body: AppointmentBody, who=Depends(identity)):
        return store.save_appointment(who["profile_id"], body.model_dump())

    @app.put("/api/care/appointments/{appointment_id}")
    def edit_appointment(appointment_id: str, body: UpdateAppointment, who=Depends(identity)):
        return store.save_appointment(who["profile_id"], body.model_dump(), appointment_id)

    @app.delete("/api/care/appointments/{appointment_id}", status_code=204)
    def cancel_appointment(appointment_id: str, who=Depends(identity)):
        store.cancel_appointment(who["profile_id"], appointment_id)

    @app.get("/api/care/reminders")
    def outbox(who=Depends(identity)):
        return {"mode": mail.mode, "messages": store.outbox(who["profile_id"])}

    @app.post("/api/care/reminders/check")
    def check_reminders(who=Depends(identity)):
        inserted = store.queue_reminders(mail.mode, profile_id=who["profile_id"])
        return {"queued": inserted, "mode": mail.mode, "messages": store.outbox(who["profile_id"])}

    @app.get("/")
    @app.get("/care.html")
    def home():
        return FileResponse(web / "care.html")

    @app.get("/activity")
    @app.get("/index.html")
    def activity():
        return FileResponse(web / "index.html")

    app.mount("/styles", StaticFiles(directory=web / "styles"), name="styles")
    app.mount("/scripts", StaticFiles(directory=web / "scripts"), name="scripts")
    return app
