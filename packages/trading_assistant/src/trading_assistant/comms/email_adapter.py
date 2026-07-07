# comms/email_adapter.py
"""Async email adapter — sends HTML emails via SMTP with lifecycle and retry."""
from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

from trading_assistant.comms.base_channel import BaseChannel


@dataclass
class EmailConfig:
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_address: str
    use_tls: bool = True


class EmailAdapter(BaseChannel):
    """Async email sender with retry support."""

    def __init__(self, config: EmailConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = config

    async def _start(self) -> None:
        import aiosmtplib  # noqa: F401 - fail fast if the optional dependency is missing

    async def _stop(self) -> None:
        pass

    async def _send(self, to: str, subject: str, html_body: str, attachment_paths: list[str] | None = None) -> None:
        return await self.send(to=to, subject=subject, html_body=html_body, attachment_paths=attachment_paths)

    def _create_smtp(self):
        import aiosmtplib
        return aiosmtplib.SMTP(
            hostname=self._config.smtp_host,
            port=self._config.smtp_port,
        )

    async def send(
        self,
        to: str,
        subject: str,
        html_body: str,
        attachment_paths: list[str] | None = None,
    ) -> None:
        msg = EmailMessage()
        msg["From"] = self._config.from_address
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(html_body, subtype="html")

        if attachment_paths:
            for path_str in attachment_paths:
                path = Path(path_str)
                if path.exists():
                    data = path.read_bytes()
                    msg.add_attachment(
                        data,
                        maintype="application",
                        subtype="octet-stream",
                        filename=path.name,
                    )

        smtp = self._create_smtp()
        await smtp.connect()

        # P3-1: STARTTLS and login are independent. Some providers accept
        # plaintext-auth on submission ports; others require STARTTLS first.
        if self._config.use_tls:
            await smtp.starttls()
        if self._config.username and self._config.password:
            await smtp.login(self._config.username, self._config.password)

        await smtp.send_message(msg)
        await smtp.quit()
