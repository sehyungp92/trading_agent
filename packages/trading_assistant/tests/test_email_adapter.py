# tests/test_email_adapter.py
"""Tests for async email adapter."""
import pytest
from unittest.mock import AsyncMock

from trading_assistant.comms.email_adapter import EmailAdapter, EmailConfig


class TestEmailConfig:
    def test_defaults(self):
        cfg = EmailConfig(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="test@example.com",
            password="secret",
            from_address="bot@example.com",
        )
        assert cfg.smtp_host == "smtp.gmail.com"
        assert cfg.use_tls is True

    def test_custom_config(self):
        cfg = EmailConfig(
            smtp_host="localhost",
            smtp_port=25,
            username="",
            password="",
            from_address="bot@local",
            use_tls=False,
        )
        assert cfg.use_tls is False


class TestEmailAdapter:
    @pytest.fixture
    def mock_smtp(self):
        smtp = AsyncMock()
        smtp.connect = AsyncMock()
        smtp.starttls = AsyncMock()
        smtp.login = AsyncMock()
        smtp.send_message = AsyncMock()
        smtp.quit = AsyncMock()
        return smtp

    @pytest.fixture
    def adapter(self):
        config = EmailConfig(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="test@example.com",
            password="secret",
            from_address="bot@example.com",
        )
        return EmailAdapter(config)

    @pytest.mark.asyncio
    async def test_send_email(self, adapter, mock_smtp):
        adapter._create_smtp = lambda: mock_smtp
        await adapter.send(
            to="user@example.com",
            subject="Test Subject",
            html_body="<html><body>Hello</body></html>",
        )
        mock_smtp.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_email_with_attachments(self, adapter, mock_smtp):
        adapter._create_smtp = lambda: mock_smtp
        await adapter.send(
            to="user@example.com",
            subject="Weekly Digest",
            html_body="<html><body>Report</body></html>",
            attachment_paths=[],
        )
        mock_smtp.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_calls_connect_and_quit(self, adapter, mock_smtp):
        adapter._create_smtp = lambda: mock_smtp
        await adapter.send(
            to="user@example.com",
            subject="Test",
            html_body="<html><body>Hi</body></html>",
        )
        mock_smtp.connect.assert_called_once()
        mock_smtp.quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_tls(self, adapter, mock_smtp):
        adapter._create_smtp = lambda: mock_smtp
        await adapter.send(
            to="user@example.com",
            subject="Test",
            html_body="<html><body>Hi</body></html>",
        )
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_without_tls(self, mock_smtp):
        config = EmailConfig(
            smtp_host="localhost",
            smtp_port=25,
            username="",
            password="",
            from_address="bot@local",
            use_tls=False,
        )
        adapter = EmailAdapter(config)
        adapter._create_smtp = lambda: mock_smtp
        await adapter.send(to="user@local", subject="Test", html_body="<html>Hi</html>")
        mock_smtp.starttls.assert_not_called()
        mock_smtp.login.assert_not_called()
