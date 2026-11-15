"""Opt-in SMTP transport. Preview mode performs no network calls."""
from dataclasses import dataclass
from email.message import EmailMessage
import os
import smtplib
import ssl


@dataclass(frozen=True)
class MailSettings:
    mode: str = "preview"
    host: str = ""
    port: int = 587
    sender: str = ""
    username: str = ""
    password: str = ""

    @classmethod
    def from_environment(cls):
        mode = os.environ.get("BS_MAIL_MODE", "preview").lower()
        if mode not in {"preview", "smtp"}:
            raise ValueError("BS_MAIL_MODE must be preview or smtp.")
        if mode == "preview":
            return cls()
        settings = cls(mode, os.environ.get("BS_SMTP_HOST", ""), int(os.environ.get("BS_SMTP_PORT", "587")),
                       os.environ.get("BS_SMTP_FROM", ""), os.environ.get("BS_SMTP_USER", ""),
                       os.environ.get("BS_SMTP_PASSWORD", ""))
        if mode == "smtp" and (not settings.host or not settings.sender or settings.port not in {465, 587}
                               or any(c in settings.sender for c in "\r\n")):
            raise ValueError("SMTP mode requires a host, sender address and TLS port 465 or 587.")
        return settings

    def send(self, recipient: str, subject: str, body: str) -> None:
        if self.mode != "smtp":
            raise RuntimeError("Preview mode does not send email.")
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self.sender, recipient, subject
        message.set_content(body)
        context = ssl.create_default_context()
        if self.port == 465:
            transport = smtplib.SMTP_SSL(self.host, self.port, timeout=10, context=context)
        else:
            transport = smtplib.SMTP(self.host, self.port, timeout=10)
        with transport as smtp:
            if self.port != 465:
                smtp.starttls(context=context)
            if self.username:
                smtp.login(self.username, self.password)
            smtp.send_message(message)
