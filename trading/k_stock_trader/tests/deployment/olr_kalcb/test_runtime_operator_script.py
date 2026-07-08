from __future__ import annotations

import asyncio
import json
from pathlib import Path
from datetime import date, datetime, time
from types import SimpleNamespace

import pytest
from trading_contracts import relay_acceptance

from strategy_common.clock import KST
from strategy_common.market import MarketBar
from deployment.olr_kalcb.market_data_coordinator import KISWebSocketCompletedBarSource
from deployment.olr_kalcb import runtime_launch
from kis_core.ws_client import TickMessage
from scripts import run_olr_kalcb_runtime_session as operator


def test_operator_loads_sector_map_from_wrapped_json(tmp_path):
    path = tmp_path / "sector_map.json"
    path.write_text(json.dumps({"sector_map": {"5930": "semiconductors", "035420": "it"}}), encoding="utf-8")

    sector_map = operator._load_sector_map(path)

    assert sector_map == {"005930": "SEMICONDUCTORS", "035420": "IT"}


def test_operator_rejects_unknown_portfolio_policy_fields():
    with pytest.raises(ValueError, match="unknown portfolio policy fields"):
        operator._portfolio_config_from_payload({"max_gross_notional": 1_000_000.0, "paper_trading_approved": True})


def test_operator_fixture_health_ok_is_dry_run_only():
    checks = operator._load_health_checks(None, mode="dry_run", fixture_health_ok=True)

    assert checks["artifact_only_gate_passed"]["passed"] is True
    assert "fixture-only" in checks["market_data_ok"]["detail"]
    with pytest.raises(ValueError, match="only valid"):
        operator._load_health_checks(None, mode="paper", fixture_health_ok=True)


def test_operator_augments_paper_health_checks_with_raw_oms_health(monkeypatch):
    payload = {"status": "ok", "stop_protection_status": "ok", "idempotency_status": "ok"}
    monkeypatch.setattr(operator, "_fetch_oms_health_payload", lambda url: payload)

    checks = operator._with_oms_health_payload(
        {"paper_trading_approved": True},
        SimpleNamespace(oms_url="http://oms.example"),
        mode="paper",
    )

    assert checks["oms_health_payload"] == payload


def test_operator_relay_heartbeat_failure_becomes_health_check(monkeypatch):
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("SIDECAR_RELAY_URL", raising=False)
    monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)

    checks = operator._with_relay_heartbeat({}, mode="paper")

    assert checks["assistant_relay_accepted"]["passed"] is False
    assert "assistant relay heartbeat failed" in checks["assistant_relay_accepted"]["detail"]


def test_operator_rejects_static_supplied_relay_health_check(monkeypatch) -> None:
    monkeypatch.delenv("RELAY_URL", raising=False)
    monkeypatch.delenv("SIDECAR_RELAY_URL", raising=False)
    monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)

    checks = operator._with_relay_heartbeat(
        {"assistant_relay_accepted": {"passed": True, "detail": "operator supplied"}},
        mode="paper",
    )

    assert checks["assistant_relay_accepted"]["passed"] is False
    assert "operator supplied" not in checks["assistant_relay_accepted"]["detail"]


def test_operator_accepts_fresh_signed_relay_probe(monkeypatch) -> None:
    calls = []
    monkeypatch.setenv("SIDECAR_RELAY_URL", "http://127.0.0.1:8001")
    monkeypatch.setenv("ALLOW_LOOPBACK_RELAY", "1")
    monkeypatch.setenv("INSTRUMENTATION_HMAC_SECRET", "0123456789abcdef0123456789ABCDEF")
    monkeypatch.setenv("K_STOCK_SIDECAR_BOT_ID", "k_stock_trader")
    monkeypatch.setenv("RELAY_API_KEY", "assistant-read-key")
    monkeypatch.delenv("ALLOW_BOT_RELAY_EXACT_ACK", raising=False)
    monkeypatch.delenv("ASSISTANT_RELAY_EXACT_ACK_API_KEY", raising=False)
    monkeypatch.setattr(operator, "_effective_config_hash", lambda: "b" * 64)
    monkeypatch.setattr(
        relay_acceptance,
        "probe_relay_acceptance",
        lambda **kwargs: calls.append(kwargs)
        or SimpleNamespace(ok=True, event_id="relay-heartbeat-k-stock", error=""),
    )

    checks = operator._with_relay_heartbeat(
        {"assistant_relay_accepted": {"passed": True, "detail": "operator supplied"}},
        mode="paper",
    )

    assert checks["assistant_relay_accepted"]["passed"] is True
    assert "relay-heartbeat-k-stock" in checks["assistant_relay_accepted"]["detail"]
    assert "operator supplied" not in checks["assistant_relay_accepted"]["detail"]
    assert calls[0]["require_exact_ack"] is False


def test_operator_has_watch_bars_runtime_entrypoint():
    args = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )

    assert args.command == "watch-bars"
    assert args.mode == "paper"
    assert args.poll_seconds > 0


def test_operator_parser_uses_deployment_env_defaults(monkeypatch):
    monkeypatch.setenv("OLR_KALCB_BASELINE_MANIFEST", "approved/baseline_manifest.json")
    monkeypatch.setenv("OLR_KALCB_PORTFOLIO_POLICY", "approved/portfolio_policy.json")
    monkeypatch.setenv("OLR_KALCB_SECTOR_MAP", "approved/sector_map.yaml")

    args = operator._parser().parse_args(["preflight", "--trade-date", "2026-02-02"])

    assert args.baseline_manifest == "approved/baseline_manifest.json"
    assert args.portfolio_policy == "approved/portfolio_policy.json"
    assert args.sector_map == "approved/sector_map.yaml"


def test_operator_kis_websocket_watch_does_not_require_bars_parquet():
    args = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--market-data-source",
            "kis_websocket",
        ]
    )

    assert args.command == "watch-bars"
    assert args.bars_parquet is None


def test_runtime_launch_owns_env_path_assembly():
    args = runtime_launch.build_watch_args(
        {
            "OLR_KALCB_TRADE_DATE": "2026-02-02",
            "OLR_KALCB_RUNTIME_MODE": "paper",
            "OLR_KALCB_MARKET_DATA_SOURCE": "auto",
            "OLR_KALCB_DEPLOYMENT_METADATA_PATH": "data/paper_live/olr_kalcb/2026-02-02/deployment_metadata.json",
            "OLR_KALCB_STRATEGY_PLUGIN_CONTRACT": "contracts/strategy_plugins/k_stock_olr_kalcb/strategy_plugin_contract.json",
            "OLR_KALCB_ONCE": "1",
        }
    )

    assert args[:7] == ["watch-bars", "--trade-date", "2026-02-02", "--mode", "paper", "--market-data-source", "auto"]
    assert args[args.index("--session-root") + 1] == "data/paper_live/olr_kalcb/2026-02-02"
    assert args[args.index("--health-checks-json") + 1] == "data/paper_live/olr_kalcb/2026-02-02/health_checks.json"
    assert args[args.index("--account-state-json") + 1] == "data/paper_live/olr_kalcb/2026-02-02/account_state.json"
    assert args[args.index("--positions-json") + 1] == "data/paper_live/olr_kalcb/2026-02-02/positions.json"
    assert args[args.index("--deployment-metadata-environment") + 1] == "paper_vps"
    assert "--once" in args
    assert "--bars-parquet" not in args


def test_runtime_launch_blocks_invalid_runtime_modes_and_missing_bars():
    with pytest.raises(runtime_launch.LaunchConfigError, match="preflight-only"):
        runtime_launch.build_watch_args(
            {"OLR_KALCB_TRADE_DATE": "2026-02-02", "OLR_KALCB_RUNTIME_MODE": "artifact_only"}
        )

    with pytest.raises(runtime_launch.LaunchConfigError, match="OLR_KALCB_BARS_PARQUET"):
        runtime_launch.build_watch_args(
            {"OLR_KALCB_TRADE_DATE": "2026-02-02", "OLR_KALCB_RUNTIME_MODE": "dry_run"}
        )


def test_operator_auto_market_data_source_is_promotional_ws_owned():
    dry_run = operator._parser().parse_args(
        [
            "dry-run-bars",
            "--trade-date",
            "2026-02-02",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )
    paper = operator._parser().parse_args(
        [
            "watch-bars",
            "--trade-date",
            "2026-02-02",
            "--mode",
            "paper",
            "--bars-parquet",
            "data/paper_live/olr_kalcb/2026-02-02/market_bars_5m.parquet",
        ]
    )

    assert operator._resolve_market_data_source(dry_run, "dry_run") == "external_completed_bars"
    assert operator._resolve_market_data_source(paper, "paper") == "kis_websocket"


def test_operator_watch_processing_routes_unseen_bars_through_coordinator():
    trade_date = date(2026, 2, 2)
    bar = MarketBar(
        symbol="005930",
        timestamp=datetime.combine(trade_date, time(9, 35), tzinfo=KST),
        timeframe="5m",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000.0,
    )

    class Plan:
        async def handle_bar(self, _bar):
            raise AssertionError("watch-bars must route through the market-data coordinator when one exists")

    class Coordinator:
        def __init__(self):
            self.routed = []

        async def route_completed_bar(self, plan, routed_bar):
            self.routed.append(routed_bar)
            return ()

    coordinator = Coordinator()
    seen: set[str] = set()

    processed = asyncio.run(operator._process_unseen_bars(Plan(), [bar], seen, coordinator=coordinator))
    duplicate = asyncio.run(operator._process_unseen_bars(Plan(), [bar], seen, coordinator=coordinator))

    assert processed == 1
    assert duplicate == 0
    assert coordinator.routed == [bar]


def test_kis_websocket_completed_bar_source_runs_client_and_emits_completed_5m_bar():
    timestamp = datetime.combine(date(2026, 2, 2), time(9, 0), tzinfo=KST)
    ticks = [
        TickMessage("005930", 100.0, 10.0, 10.0, 1000.0, 0.0, timestamp),
        TickMessage("005930", 101.0, 20.0, 30.0, 3030.0, 0.0, timestamp.replace(minute=1)),
        TickMessage("005930", 102.0, 30.0, 60.0, 6120.0, 0.0, timestamp.replace(minute=5)),
    ]

    class FakeWebSocketClient:
        def __init__(self):
            self.callbacks = []
            self.run_started = False

        def on_tick(self, callback):
            self.callbacks.append(callback)

        async def run(self, auto_reconnect=True):
            self.run_started = auto_reconnect
            for tick in ticks:
                for callback in self.callbacks:
                    callback(tick)
            await asyncio.sleep(60)

    client = FakeWebSocketClient()
    source = KISWebSocketCompletedBarSource(client)

    async def scenario():
        await source.start()
        bar = await source.next_bar(timeout_s=0.5)
        await source.stop()
        return bar

    bar = asyncio.run(scenario())

    assert client.run_started is True
    assert bar is not None
    assert bar.symbol == "005930"
    assert bar.timestamp == timestamp
    assert bar.timeframe == "5m"
    assert bar.open == 100.0
    assert bar.high == 101.0
    assert bar.close == 101.0
    assert bar.volume == 30.0
    assert bar.source == "kis_websocket"


def test_single_vps_wrappers_enforce_artifact_gate_before_runtime_restart():
    bot_root = Path(__file__).resolve().parents[3]
    repo_root = Path(__file__).resolve().parents[5]
    premarket = (bot_root / "infra" / "cron" / "olr_kalcb_premarket_restart.sh").read_text(encoding="utf-8")
    afternoon = (bot_root / "infra" / "cron" / "olr_kalcb_afternoon_restart.sh").read_text(encoding="utf-8")
    entrypoint = (bot_root / "deployment" / "olr_kalcb" / "runtime_service_entrypoint.sh").read_text(
        encoding="utf-8"
    )
    launch = (bot_root / "deployment" / "olr_kalcb" / "runtime_launch.py").read_text(encoding="utf-8")
    deployed_entrypoint = (repo_root / "deployments" / "k_stock" / "runtime-entrypoint.sh").read_text(
        encoding="utf-8"
    )

    assert "flock -n" in premarket
    assert "timeout \"$ARTIFACT_TIMEOUT\" docker compose run --rm runtime" in premarket
    assert "DAILY_UNIVERSE_FILE" in premarket
    assert "approved OLR/KALCB deployment universe file is missing" in premarket
    assert 'daily_args=(--daily-universe-file "$DAILY_UNIVERSE_FILE")' in premarket
    assert "generate_olr_kalcb_artifacts.py daily" in premarket
    assert "--mode artifact_only_stage1" in premarket
    assert premarket.index("generate_olr_kalcb_artifacts.py daily") < premarket.index("docker compose up -d runtime")

    assert "generate_olr_kalcb_artifacts.py afternoon" in afternoon
    assert "--mode artifact_only" in afternoon
    assert afternoon.index("generate_olr_kalcb_artifacts.py afternoon") < afternoon.index("--force-recreate runtime")

    assert "python -m deployment.olr_kalcb.runtime_launch" in entrypoint
    assert "python -m deployment.olr_kalcb.runtime_launch" in deployed_entrypoint
    assert "OLR_KALCB_TRADE_DATE must be set" in launch
    assert "watch-bars" in launch
    assert "--market-data-source" in launch
    assert "artifact_only" in launch
    assert "artifact_only_stage1" in launch
    for env_name, cli_flag in (
        ("OLR_KALCB_SESSION_ROOT", "--session-root"),
        ("OLR_KALCB_HEALTH_CHECKS_JSON", "--health-checks-json"),
        ("OLR_KALCB_ACCOUNT_STATE_JSON", "--account-state-json"),
        ("OLR_KALCB_POSITIONS_JSON", "--positions-json"),
        ("OLR_KALCB_KIS_WS_URL", "--kis-ws-url"),
        ("OLR_KALCB_DEPLOYMENT_METADATA_PATH", "--deployment-metadata-json"),
        ("OLR_KALCB_STRATEGY_PLUGIN_CONTRACT", "--strategy-plugin-contract"),
        ("OLR_KALCB_DEPLOYMENT_METADATA_ENV", "--deployment-metadata-environment"),
    ):
        assert env_name in launch
        assert cli_flag in launch


def test_monorepo_deployment_preserves_reference_paper_runtime_surfaces():
    repo_root = Path(__file__).resolve().parents[5]
    dockerfile = (repo_root / "trading" / "k_stock_trader" / "Dockerfile").read_text(encoding="utf-8")
    compose = (repo_root / "deployments" / "k_stock" / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (repo_root / "deployments" / "k_stock" / ".env.example").read_text(encoding="utf-8")

    assert "COPY trading/k_stock_trader/scripts /app/trading/k_stock_trader/scripts" in dockerfile
    assert "K_STOCK_TRADER_ROOT: /app/trading/k_stock_trader" in compose
    assert "K_STOCK_HOST_DATA_ROOT=/opt/trading_agent/runtime_data/k_stock" in env_example
    assert "OLR_KALCB_SESSION_ROOT:" not in compose
    assert "OLR_KALCB_SESSION_ROOT=" not in env_example

    for subdir in (
        "strategy",
        "backtests",
        "live_readiness",
        "krx_daily_parquet",
        "kis_intraday_parquet",
        "oms",
        "paper_live",
    ):
        mount = f"${{K_STOCK_HOST_DATA_ROOT:-../../trading/k_stock_trader/data}}/{subdir}:/app/trading/k_stock_trader/data/{subdir}"
        assert mount in compose
