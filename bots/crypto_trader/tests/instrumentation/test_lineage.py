from __future__ import annotations

from pathlib import Path

from crypto_trader.instrumentation.lineage import (
    from_live_engine_inputs,
    redact_secrets,
    stable_hash,
    strip_secret_fields,
)
from crypto_trader.live.config import LiveConfig
from crypto_trader.portfolio.config import PortfolioConfig, StrategyAllocation


def test_stable_hash_ignores_mapping_order() -> None:
    assert stable_hash({"a": 1, "b": {"c": 2}}) == stable_hash({"b": {"c": 2}, "a": 1})


def test_redact_secrets_recursively() -> None:
    payload = {
        "relay_secret": "secret",
        "nested": {"postgres_dsn": "postgres://user:pass@host/db"},
        "safe": "value",
    }

    redacted = redact_secrets(payload)

    assert redacted["relay_secret"] == "***REDACTED***"
    assert redacted["nested"]["postgres_dsn"] == "***REDACTED***"
    assert redacted["safe"] == "value"


def test_strip_secret_fields_removes_secret_keys_recursively() -> None:
    payload = {
        "wallet_address": "0x" + "1" * 40,
        "relay_secret": "secret",
        "nested": {
            "postgres_dsn": "postgres://user:pass@host/db",
            "safe": "value",
        },
        "items": [{"api_key": "k"}, {"name": "kept"}],
    }

    stripped = strip_secret_fields(payload)

    assert "wallet_address" not in stripped
    assert "relay_secret" not in stripped
    assert "postgres_dsn" not in stripped["nested"]
    assert "api_key" not in stripped["items"][0]
    assert stripped["nested"]["safe"] == "value"
    assert stripped["items"][1]["name"] == "kept"


def test_live_lineage_versions_are_secret_safe_and_specific(tmp_path: Path) -> None:
    cfg = LiveConfig(
        wallet_address="0x" + "1" * 40,
        private_key="0x" + "2" * 64,
        bot_id="paper_bot",
        relay_secret="relay-secret",
        postgres_dsn="postgres://secret",
        family_id="crypto_perps",
        portfolio_id="portfolio_a",
        account_alias="paper",
        strategy_configs={"momentum": tmp_path / "momentum.json"},
    )
    portfolio = PortfolioConfig(
        strategies=(StrategyAllocation(strategy_id="momentum", base_risk_pct=0.01),),
        heat_cap_R=3.0,
    )

    lineage = from_live_engine_inputs(
        config=cfg,
        portfolio_config=portfolio,
        strategy_configs={"momentum": {"strategy": {"risk": 1}}},
        deployment_manifest={"candidate_id": "round_3"},
        cwd=Path.cwd(),
    )

    assert lineage.family_id == "crypto_perps"
    assert lineage.portfolio_id == "portfolio_a"
    assert lineage.account_alias == "paper"
    assert lineage.strategy_config_versions["momentum"]
    assert lineage.risk_config_version != lineage.allocation_version
    assert lineage.config_version
    assert "secret" not in str(lineage.__dict__).lower()
