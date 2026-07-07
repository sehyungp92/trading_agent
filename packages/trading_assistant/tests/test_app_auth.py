from __future__ import annotations

from httpx import ASGITransport, AsyncClient

import pytest

from trading_assistant.orchestrator.app import create_app
from trading_assistant.orchestrator.config import AppConfig


@pytest.fixture
def protected_app(tmp_path):
    config = AppConfig(orchestrator_api_key="super-secret")
    return create_app(db_dir=str(tmp_path), config=config)


class TestPublicBindRequiresAuth:
    """P0-2: refuse startup when bound non-loopback without an API key."""

    def test_default_without_key_is_refused(self, tmp_path):
        config = AppConfig(orchestrator_api_key="", bind_host="127.0.0.1")
        with pytest.raises(RuntimeError, match="ORCHESTRATOR_API_KEY"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_explicit_loopback_dev_without_key_is_allowed(self, tmp_path):
        config = AppConfig(orchestrator_api_key="", bind_host="127.0.0.1")
        config.allow_unauthenticated_local = True
        # Should not raise.
        create_app(db_dir=str(tmp_path), config=config)

    def test_env_loaded_implicit_loopback_without_key_is_refused(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="127.0.0.1",
            bind_host_explicit=False,
            allow_unauthenticated_local=True,
        )
        with pytest.raises(RuntimeError, match="BIND_HOST_EXPLICIT=False"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_production_local_escape_hatch_without_key_is_refused(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="127.0.0.1",
            allow_unauthenticated_local=True,
            environment="production",
        )
        with pytest.raises(RuntimeError, match="ENVIRONMENT='production'"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_localhost_alias_treated_as_loopback(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="localhost",
            allow_unauthenticated_local=True,
        )
        create_app(db_dir=str(tmp_path), config=config)

    def test_public_bind_without_key_refused(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="0.0.0.0",
            allow_unauthenticated_local=True,
        )
        with pytest.raises(RuntimeError, match="ORCHESTRATOR_API_KEY"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_bind_host_mismatch_with_public_uvicorn_host_is_refused(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="127.0.0.1",
            uvicorn_host="0.0.0.0",
            allow_unauthenticated_local=True,
        )
        with pytest.raises(RuntimeError, match="BIND_HOST and UVICORN_HOST disagree"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_loopback_bind_host_aliases_can_match(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="localhost",
            uvicorn_host="127.0.0.1",
            allow_unauthenticated_local=True,
        )
        create_app(db_dir=str(tmp_path), config=config)

    def test_empty_bind_without_key_refused(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="",
            bind_host="",
            allow_unauthenticated_local=True,
        )
        with pytest.raises(RuntimeError, match="ORCHESTRATOR_API_KEY"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_public_bind_with_key_allowed(self, tmp_path):
        config = AppConfig(orchestrator_api_key="secret", bind_host="0.0.0.0")
        create_app(db_dir=str(tmp_path), config=config)

    def test_production_requires_bots_and_ingest_mode(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="secret",
            environment="production",
            bot_ids=[],
            relay_url="",
            direct_ingest_only=False,
        )
        with pytest.raises(RuntimeError, match="BOT_IDS"):
            create_app(db_dir=str(tmp_path), config=config)

    def test_production_relay_requires_api_key(self, tmp_path):
        config = AppConfig(
            orchestrator_api_key="secret",
            environment="production",
            bot_ids=["bot1"],
            relay_url="https://relay.example",
            relay_api_key="",
        )
        with pytest.raises(RuntimeError, match="RELAY_API_KEY"):
            create_app(db_dir=str(tmp_path), config=config)


class TestOrchestratorApiKeyAuth:
    @pytest.mark.asyncio
    async def test_health_remains_open(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json()["auth_enabled"] is True

    @pytest.mark.asyncio
    async def test_metrics_requires_api_key(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.get("/metrics")

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API key"

    @pytest.mark.asyncio
    async def test_metrics_allows_valid_api_key(self, protected_app):
        async with protected_app.router.lifespan_context(protected_app):
            async with AsyncClient(
                transport=ASGITransport(app=protected_app),
                base_url="http://test",
            ) as client:
                response = await client.get(
                    "/metrics",
                    headers={"X-Api-Key": "super-secret"},
                )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mutating_endpoint_requires_api_key(self, protected_app):
        async with AsyncClient(
            transport=ASGITransport(app=protected_app),
            base_url="http://test",
        ) as client:
            response = await client.put(
                "/notifications/preferences",
                json={"channels": []},
            )

        assert response.status_code == 401
