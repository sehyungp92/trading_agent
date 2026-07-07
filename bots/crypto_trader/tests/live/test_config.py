"""Tests for live trading configuration."""

import json
from pathlib import Path

from click.testing import CliRunner

from crypto_trader.cli import (
    _deployment_preflight_errors,
    _live_config_path_errors,
    _live_config_public_hash,
    cli,
)
from crypto_trader.live.config import LiveConfig

VALID_WALLET = "0x" + "1" * 40
VALID_PRIVATE_KEY = "0x" + "2" * 64
STRATEGY_IDS = ("momentum", "trend", "breakout")
REPO_ROOT = Path(__file__).resolve().parents[4]


def _valid_live_config_payload(tmp_path: Path) -> dict:
    manifest_ref = _write_deployment_bundle(tmp_path)
    return {
        "wallet_address": VALID_WALLET,
        "private_key": VALID_PRIVATE_KEY,
        "is_testnet": True,
        "poll_interval_sec": 15.0,
        "fill_poll_interval_sec": 30.0,
        "fill_query_overlap_sec": 300.0,
        "equity_snapshot_interval_sec": 300.0,
        "health_check_interval_sec": 60.0,
        "health_report_interval_sec": 300.0,
        "funnel_report_interval_sec": 3600.0,
        "rate_limit_per_sec": 5.0,
        "max_slippage_pct": 0.005,
        "reconciliation_policy": "block",
        "allow_manual_flatten": False,
        "strict_live_parity": False,
        "symbols": ["BTC", "ETH", "SOL"],
        "data_dir": "data",
        "state_dir": "data/live_state",
        "asset_meta_path": None,
        "bot_id": "paper_bot_01",
        "family_id": "crypto_perps",
        "portfolio_id": "round_3_crypto_perps",
        "account_alias": "paper_testnet",
        "relay_url": "http://10.0.0.10:8001/events",
        "relay_secret": "secret",
        "strategy_configs": {
            strategy_id: f"output/portfolio/round_3/recommended_strategy_configs/{strategy_id}.json"
            for strategy_id in STRATEGY_IDS
        },
        "portfolio_config_path": "output/portfolio/round_3/recommended_portfolio_config.json",
        "deployment_manifest_path": str(manifest_ref).replace("\\", "/"),
        "postgres_dsn": "",
    }


def _effective_payload_for(config_payload: dict) -> dict:
    return {
        "materialized_config": {
            "runtime_config_contract": {
                "schema_version": "crypto_runtime_config_contract.v1",
                "mounted_config_path": "config/live_config.json",
                "public_live_config_sha256": _live_config_public_hash(config_payload),
                "public_hash_excludes": [
                    "postgres_dsn",
                    "private_key",
                    "relay_secret",
                    "relay_url",
                    "wallet_address",
                ],
                "required_non_empty_fields": [
                    "wallet_address",
                    "private_key",
                    "relay_url",
                    "relay_secret",
                    "bot_id",
                ],
                "sidecar_forwarding_required": True,
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_deployment_bundle(
    tmp_path: Path,
    *,
    root: Path = Path("output/portfolio/round_3"),
    strategy_ids: tuple[str, ...] = STRATEGY_IDS,
) -> Path:
    portfolio_ref = root / "recommended_portfolio_config.json"
    strategy_refs = {
        strategy_id: root / "recommended_strategy_configs" / f"{strategy_id}.json"
        for strategy_id in strategy_ids
    }
    _write_json(tmp_path / portfolio_ref, {"portfolio": "promoted", "strategies": list(strategy_ids)})
    for strategy_id, path in strategy_refs.items():
        _write_json(tmp_path / path, {"strategy": {"name": strategy_id, "version": "promoted"}})
    _write_json(
        tmp_path / "output" / "portfolio" / "rounds_manifest.json",
        {"rounds": [{"round": 1}, {"round": 2}, {"round": 3}]},
    )
    parity_ref = root / "parity_alignment.json"
    _write_json(
        tmp_path / parity_ref,
        {"portfolio_metric_replay": {"status": "matched", "max_abs_delta": 0.0, "tolerance": 1e-9}},
    )
    manifest_ref = root / "deployment_manifest.json"
    _write_json(
        tmp_path / manifest_ref,
        {
            "schema_version": 1,
            "required_strategy_ids": list(strategy_ids),
            "portfolio_config_path": str(portfolio_ref).replace("\\", "/"),
            "strategy_configs": {
                strategy_id: str(path).replace("\\", "/")
                for strategy_id, path in strategy_refs.items()
            },
            "portfolio_rounds_manifest_path": "output/portfolio/rounds_manifest.json",
            "required_portfolio_rounds": [1, 2, 3],
            "parity_alignment_path": str(parity_ref).replace("\\", "/"),
        },
    )
    return manifest_ref


class TestLiveConfig:
    def test_defaults(self):
        cfg = LiveConfig()
        assert cfg.is_testnet is True
        assert cfg.poll_interval_sec == 15.0
        assert cfg.symbols == ["BTC", "ETH", "SOL"]
        assert cfg.max_slippage_pct == 0.005
        assert cfg.health_report_interval_sec == 300.0
        assert cfg.funnel_report_interval_sec == 3600.0

    def test_validate_empty(self):
        cfg = LiveConfig()
        errors = cfg.validate()
        assert len(errors) >= 1
        assert any("wallet_address" in e for e in errors)

    def test_validate_valid(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=VALID_PRIVATE_KEY)
        errors = cfg.validate()
        assert len(errors) == 0

    def test_validate_read_only(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=None)
        errors = cfg.validate()
        assert any("read-only" in e for e in errors)

    def test_validate_rejects_placeholder_credentials(self):
        cfg = LiveConfig(
            wallet_address="0xYOUR_WALLET_ADDRESS_HERE",
            private_key="0xYOUR_PRIVATE_KEY_HERE",
        )
        errors = cfg.validate()
        assert any("wallet_address must be replaced" in e for e in errors)
        assert any("private_key must be replaced" in e for e in errors)

    def test_validate_rejects_malformed_hex_credentials(self):
        cfg = LiveConfig(wallet_address="0x123", private_key="0xabc")
        errors = cfg.validate()
        assert any("wallet_address must be 0x followed by 40 hex characters" in e for e in errors)
        assert any("private_key must be 0x followed by 64 hex characters" in e for e in errors)

    def test_live_config_example_credentials_do_not_validate(self):
        payload = json.loads(Path("config/live_config.example.json").read_text(encoding="utf-8"))
        errors = LiveConfig.from_dict(payload).validate()
        assert any("wallet_address" in e for e in errors)
        assert any("private_key" in e for e in errors)

    def test_live_config_example_leaves_local_postgres_disabled(self):
        payload = json.loads(Path("config/live_config.example.json").read_text(encoding="utf-8"))
        assert payload["postgres_dsn"] == ""

    def test_deployment_preflight_accepts_secret_changes_with_same_public_config(self, tmp_path):
        config_payload = _valid_live_config_payload(tmp_path)
        effective_payload = _effective_payload_for(config_payload)
        mounted_payload = {
            **config_payload,
            "wallet_address": "0x" + "3" * 40,
            "private_key": "0x" + "4" * 64,
            "relay_url": "http://10.1.2.3:8001/events",
            "relay_secret": "different-secret",
        }

        errors = _deployment_preflight_errors(
            mounted_payload,
            effective_payload,
            runtime_root=tmp_path,
        )

        assert errors == []

    def test_deployment_preflight_requires_crypto_relay_sidecar_fields(self, tmp_path):
        config_payload = _valid_live_config_payload(tmp_path)
        effective_payload = _effective_payload_for(config_payload)
        mounted_payload = {**config_payload, "relay_url": "", "relay_secret": ""}

        errors = _deployment_preflight_errors(
            mounted_payload,
            effective_payload,
            runtime_root=tmp_path,
        )

        assert "relay_url is required by runtime_config_contract" in errors
        assert "relay_secret is required by runtime_config_contract" in errors

    def test_deployment_preflight_rejects_public_config_hash_drift(self, tmp_path):
        config_payload = _valid_live_config_payload(tmp_path)
        effective_payload = _effective_payload_for(config_payload)
        mounted_payload = {**config_payload, "symbols": ["BTC", "ETH"]}

        errors = _deployment_preflight_errors(
            mounted_payload,
            effective_payload,
            runtime_root=tmp_path,
        )

        assert "mounted live config public hash does not match generated effective config" in errors

    def test_emit_deployment_metadata_writes_all_contract_artifacts(self, tmp_path):
        effective_path = tmp_path / "live_config.effective.json"
        _write_json(
            effective_path,
            {
                "effective_config_hash": "test-effective-config",
                "materialized_config": {},
            },
        )
        contract_work_root = tmp_path / "contract_work"
        state_dir = tmp_path / "state"

        result = CliRunner().invoke(
            cli,
            [
                "emit-deployment-metadata",
                "--effective-config",
                str(effective_path),
                "--contract-source-root",
                str(REPO_ROOT / "contracts" / "strategy_plugins"),
                "--contract-work-root",
                str(contract_work_root),
                "--state-dir",
                str(state_dir),
                "--repo-root",
                str(REPO_ROOT),
                "--runtime-started-at-utc",
                "2026-07-07T00:00:00Z",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload["deployment_metadata_paths"]) == {
            "crypto_breakout_v1",
            "crypto_momentum_v1",
            "crypto_trend_v1",
        }
        for bridge_id, raw_path in payload["deployment_metadata_paths"].items():
            metadata_path = Path(raw_path)
            assert metadata_path.is_file(), bridge_id
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            assert metadata["strategy_id"] == bridge_id
            assert metadata["telemetry_schema_versions"]

    def test_base_url_testnet(self):
        cfg = LiveConfig(is_testnet=True)
        assert "testnet" in cfg.base_url

    def test_base_url_mainnet(self):
        cfg = LiveConfig(is_testnet=False)
        assert "testnet" not in cfg.base_url

    def test_from_dict(self):
        d = {
            "wallet_address": "0x123",
            "is_testnet": False,
            "symbols": ["BTC"],
            "poll_interval_sec": 30.0,
            "health_report_interval_sec": 120.0,
            "funnel_report_interval_sec": 1800.0,
            "strategy_configs": {"momentum": "configs/momentum.json"},
            "deployment_manifest_path": "deployments/manifest.json",
        }
        cfg = LiveConfig.from_dict(d)
        assert cfg.wallet_address == "0x123"
        assert cfg.is_testnet is False
        assert cfg.symbols == ["BTC"]
        assert cfg.poll_interval_sec == 30.0
        assert cfg.health_report_interval_sec == 120.0
        assert cfg.funnel_report_interval_sec == 1800.0
        assert cfg.strategy_configs["momentum"] == Path("configs/momentum.json")
        assert cfg.deployment_manifest_path == Path("deployments/manifest.json")

    def test_to_dict_excludes_private_key(self):
        cfg = LiveConfig(wallet_address=VALID_WALLET, private_key=VALID_PRIVATE_KEY)
        d = cfg.to_dict()
        assert "private_key" not in d
        assert d["wallet_address"] == VALID_WALLET

    def test_to_dict_redacted_omits_secret_fields(self):
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            bot_id="paper_bot",
            relay_url="https://relay.example.com",
            relay_secret="secret",
            postgres_dsn="postgres://user:pass@host/db",
        )

        d = cfg.to_dict(redacted=True)

        assert "private_key" not in d
        assert "wallet_address" not in d
        assert "relay_secret" not in d
        assert "postgres_dsn" not in d
        assert d["bot_id"] == "paper_bot"
        assert d["relay_url"] == "https://relay.example.com"

    def test_roundtrip(self):
        cfg = LiveConfig(
            wallet_address="0x123",
            is_testnet=True,
            symbols=["BTC", "ETH"],
            poll_interval_sec=20.0,
        )
        d = cfg.to_dict()
        cfg2 = LiveConfig.from_dict(d)
        assert cfg2.wallet_address == cfg.wallet_address
        assert cfg2.symbols == cfg.symbols
        assert cfg2.poll_interval_sec == cfg.poll_interval_sec

    def test_live_config_path_errors_require_explicit_runtime_files(self, tmp_path):
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=tmp_path / "missing_portfolio.json",
            strategy_configs={"trend": tmp_path / "missing_trend.json"},
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio_config_path" in error for error in errors)
        assert any("strategy_configs.trend" in error for error in errors)

    def test_live_config_path_errors_accept_asset_meta_covering_symbols(self, tmp_path):
        _write_json(tmp_path / "portfolio.json", {})
        _write_json(tmp_path / "trend.json", {})
        _write_json(
            tmp_path / "asset_meta.json",
            {
                "asset_index": {"BTC": 0, "ETH": 1},
                "tick_sizes": {"BTC": 0.5, "ETH": 0.05},
                "lot_sizes": {"BTC": 0.001, "ETH": 0.01},
            },
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            symbols=["BTC", "ETH"],
            portfolio_config_path=Path("portfolio.json"),
            strategy_configs={"trend": Path("trend.json")},
            asset_meta_path=Path("asset_meta.json"),
        )

        assert _live_config_path_errors(
            cfg,
            runtime_root=tmp_path,
            require_deployment_manifest=False,
        ) == []

    def test_live_config_path_errors_reject_incomplete_asset_meta(self, tmp_path):
        _write_json(tmp_path / "portfolio.json", {})
        _write_json(tmp_path / "trend.json", {})
        _write_json(
            tmp_path / "asset_meta.json",
            {
                "asset_index": {"BTC": 0},
                "tick_sizes": {"BTC": 0.5},
                "lot_sizes": {"BTC": 0.001},
            },
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            symbols=["BTC", "ETH"],
            portfolio_config_path=Path("portfolio.json"),
            strategy_configs={"trend": Path("trend.json")},
            asset_meta_path=Path("asset_meta.json"),
        )

        errors = _live_config_path_errors(
            cfg,
            runtime_root=tmp_path,
            require_deployment_manifest=False,
        )

        assert any("asset_meta_path.asset_index missing symbols: ETH" in error for error in errors)
        assert any("asset_meta_path.tick_sizes missing symbols: ETH" in error for error in errors)
        assert any("asset_meta_path.lot_sizes missing symbols: ETH" in error for error in errors)

    def test_live_config_path_errors_accept_exact_materialized_bundle_copies(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        expected_bundle = tmp_path / "output" / "portfolio" / "round_3"
        portfolio = tmp_path / "config" / "portfolio_config.json"
        portfolio.parent.mkdir(parents=True)
        portfolio.write_text(
            (expected_bundle / "recommended_portfolio_config.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        strategy_configs = {}
        for strategy_id in STRATEGY_IDS:
            strategy = tmp_path / "config" / "strategies" / f"{strategy_id}.json"
            strategy.parent.mkdir(parents=True, exist_ok=True)
            strategy.write_text(
                (expected_bundle / "recommended_strategy_configs" / f"{strategy_id}.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            strategy_configs[strategy_id] = Path("config") / "strategies" / f"{strategy_id}.json"
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("config/portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs=strategy_configs,
        )

        assert _live_config_path_errors(cfg, runtime_root=tmp_path) == []

    def test_live_config_path_errors_reject_stale_existing_runtime_files(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        expected_bundle = tmp_path / "output" / "portfolio" / "round_3"
        stale_portfolio = tmp_path / "config" / "portfolio_config.json"
        stale_portfolio.parent.mkdir(parents=True)
        stale_portfolio.write_text('{"portfolio": "stale"}', encoding="utf-8")
        strategy_configs = {}
        for strategy_id in STRATEGY_IDS:
            strategy = tmp_path / "config" / "strategies" / f"{strategy_id}.json"
            strategy.parent.mkdir(parents=True, exist_ok=True)
            source = expected_bundle / "recommended_strategy_configs" / f"{strategy_id}.json"
            strategy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            strategy_configs[strategy_id] = Path("config") / "strategies" / f"{strategy_id}.json"
        (tmp_path / "config" / "strategies" / "trend.json").write_text(
            '{"strategy": {"name": "trend", "version": "stale"}}',
            encoding="utf-8",
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("config/portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs=strategy_configs,
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio_config_path does not match deployment manifest reference" in error for error in errors)
        assert any("strategy_configs.trend does not match deployment manifest reference" in error for error in errors)

    def test_live_config_path_errors_reject_missing_required_deployment_strategy(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                "momentum": Path("output/portfolio/round_3/recommended_strategy_configs/momentum.json"),
                "breakout": Path("output/portfolio/round_3/recommended_strategy_configs/breakout.json"),
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("missing required deployment strategies: trend" in error for error in errors)

    def test_live_config_path_errors_validate_portfolio_manifest_rounds(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(tmp_path / "output" / "portfolio" / "rounds_manifest.json", {"rounds": [{"round": 1}, {"round": 2}]})
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio rounds manifest rounds [1, 2] do not match required [1, 2, 3]" in error for error in errors)

    def test_live_config_path_errors_require_complete_parity_numeric_fields(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(
            tmp_path / "output" / "portfolio" / "round_3" / "parity_alignment.json",
            {"portfolio_metric_replay": {"status": "matched", "tolerance": 1e-9}},
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio parity evidence missing numeric fields: max_abs_delta" in error for error in errors)

    def test_live_config_path_errors_reject_negative_parity_abs_delta(self, tmp_path):
        manifest_path = _write_deployment_bundle(tmp_path)
        _write_json(
            tmp_path / "output" / "portfolio" / "round_3" / "parity_alignment.json",
            {"portfolio_metric_replay": {"status": "matched", "max_abs_delta": -1.0, "tolerance": 1e-9}},
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("output/portfolio/round_3/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                strategy_id: Path("output/portfolio/round_3/recommended_strategy_configs") / f"{strategy_id}.json"
                for strategy_id in STRATEGY_IDS
            },
        )

        errors = _live_config_path_errors(cfg, runtime_root=tmp_path)

        assert any("portfolio parity evidence has non-finite or negative numeric fields" in error for error in errors)

    def test_live_config_path_errors_are_manifest_driven_not_round3_hardcoded(self, tmp_path):
        manifest_path = _write_deployment_bundle(
            tmp_path,
            root=Path("deployments/reduced_risk"),
            strategy_ids=("trend",),
        )
        cfg = LiveConfig(
            wallet_address=VALID_WALLET,
            private_key=VALID_PRIVATE_KEY,
            portfolio_config_path=Path("deployments/reduced_risk/recommended_portfolio_config.json"),
            deployment_manifest_path=manifest_path,
            strategy_configs={
                "trend": Path("deployments/reduced_risk/recommended_strategy_configs/trend.json"),
            },
        )

        assert _live_config_path_errors(cfg, runtime_root=tmp_path) == []
