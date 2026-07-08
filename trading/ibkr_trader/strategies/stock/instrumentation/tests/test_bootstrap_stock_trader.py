import asyncio
import json
from unittest.mock import MagicMock

import pytest

from libs.instrumentation.lineage import lineage_from_config
from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
from strategies.stock.instrumentation.src.bootstrap import InstrumentationManager, _load_config


class _DummyEventBus:
    def unsubscribe_all(self, queue) -> None:
        return None


class _DummyOMS:
    def __init__(self) -> None:
        self.event_bus = _DummyEventBus()

    def stream_all_events(self):
        return asyncio.Queue()


def test_load_config_applies_stock_trader_defaults():
    config = _load_config("IARIC_v1", "strategy_iaric")

    assert config["bot_id"] == "stock_trader"
    assert config["strategy_id"] == "IARIC_v1"
    assert config["strategy_type"] == "strategy_iaric"
    assert config["data_source_id"] == "ibkr_us_equities"
    assert config["heartbeat_interval_seconds"] == 30
    assert config["daily_snapshot_checkpoint_interval_seconds"] == 300
    assert config["market_snapshots"]["symbols"] == ["SPY", "QQQ", "IWM"]
    assert config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_load_config_applies_alcb_strategy_identity():
    config = _load_config("ALCB_v1", "strategy_alcb")

    assert config["bot_id"] == "stock_trader"
    assert config["strategy_id"] == "ALCB_v1"
    assert config["strategy_type"] == "strategy_alcb"
    assert config["data_source_id"] == "ibkr_us_equities"
    assert config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_manager_maps_config_modules_and_attachs_provider():
    manager = InstrumentationManager(
        oms=_DummyOMS(),
        strategy_id="IARIC_v1",
        strategy_type="strategy_iaric",
    )
    provider = object()

    manager.attach_data_provider(provider)

    assert manager.bot_id == "stock_trader"
    assert manager.config["strategy_id"] == "IARIC_v1"
    assert manager.config_watcher is not None
    assert manager.config_watcher._config_modules == ["strategy_iaric.config"]
    assert manager.snapshot_service._data_provider is provider
    assert manager.regime_classifier.data_provider is provider
    assert manager.config["sidecar"]["hmac_secret_env"] == "INSTRUMENTATION_HMAC_SECRET"


def test_manager_lineage_includes_applied_portfolio_rules_config():
    rules = PortfolioRulesConfig(same_sector_heat_cap_R=2.75)
    config = _load_config("IARIC_v1", "strategy_iaric")
    config["family_id"] = "stock"
    expected = lineage_from_config(
        config,
        family_id="stock",
        strategy_id="IARIC_v1",
        portfolio_rules_config=rules,
    ).risk_config_version

    manager = InstrumentationManager(
        oms=_DummyOMS(),
        strategy_id="IARIC_v1",
        strategy_type="strategy_iaric",
        get_applied_config=lambda: rules,
    )

    assert manager.lineage.risk_config_version == expected
    assert manager.trade_logger._lineage.risk_config_version == expected


def test_manager_maps_alcb_config_module():
    manager = InstrumentationManager(
        oms=_DummyOMS(),
        strategy_id="ALCB_v1",
        strategy_type="strategy_alcb",
    )

    assert manager.bot_id == "stock_trader"
    assert manager.config["strategy_id"] == "ALCB_v1"
    assert manager.config_watcher is not None
    assert manager.config_watcher._config_modules == ["strategy_alcb.config"]


def test_periodic_loop_runs_trade_and_missed_backfills():
    async def _run():
        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="IARIC_v1",
            strategy_type="strategy_iaric",
        )
        provider = object()
        manager.attach_data_provider(provider)
        manager.snapshot_service.run_periodic = MagicMock()
        manager.trade_logger.run_post_exit_backfill = MagicMock()
        manager.missed_logger.run_backfill = MagicMock()
        manager.config_watcher = MagicMock()
        manager.daily_builder.build = MagicMock(return_value=MagicMock())
        manager.daily_builder.save = MagicMock()
        manager.config["daily_snapshot_checkpoint_interval_seconds"] = 0
        manager._running = True

        task = asyncio.create_task(manager._periodic_snapshot_loop(0.01))
        await asyncio.sleep(0.03)
        manager._running = False
        await task

        manager.trade_logger.run_post_exit_backfill.assert_called_with(provider)
        manager.missed_logger.run_backfill.assert_called_with(provider)

    asyncio.run(_run())


def test_periodic_loop_checkpoints_daily_snapshot():
    async def _run():
        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="ALCB_v1",
            strategy_type="strategy_alcb",
        )
        manager.snapshot_service.run_periodic = MagicMock()
        manager.trade_logger.run_post_exit_backfill = MagicMock()
        manager.missed_logger.run_backfill = MagicMock()
        manager.config_watcher = MagicMock()
        manager.daily_builder.build = MagicMock(return_value=MagicMock())
        manager.daily_builder.save = MagicMock()
        manager.config["daily_snapshot_checkpoint_interval_seconds"] = 1
        manager._last_snapshot_checkpoint_at = 0.0
        manager._running = True

        task = asyncio.create_task(manager._periodic_snapshot_loop(0.01))
        await asyncio.sleep(0.03)
        manager._running = False
        await task

        manager.daily_builder.build.assert_called()
        manager.daily_builder.save.assert_called()

    asyncio.run(_run())


def test_start_missing_sidecar_auth_in_paper_disables_forwarding_but_keeps_local_startup(monkeypatch, tmp_path):
    async def _run():
        monkeypatch.setenv("TRADING_MODE", "paper")
        monkeypatch.delenv("INSTRUMENTATION_HMAC_SECRET", raising=False)

        manager = InstrumentationManager(
            oms=_DummyOMS(),
            strategy_id="ALCB_v1",
            strategy_type="strategy_alcb",
            write_daily_closeout_on_stop=False,
            stop_sidecar_on_stop=False,
        )
        manager._config["data_dir"] = str(tmp_path)
        manager.sidecar.start = MagicMock()

        await manager.start()

        assert manager._running is True
        manager.sidecar.start.assert_not_called()
        assert list((tmp_path / "deployments").glob("*.jsonl"))
        await manager.stop()

    asyncio.run(_run())


def test_startup_events_use_hydrated_oms_state(tmp_path):
    class FakeRepo:
        async def get_positions_for_strategies(self, strategy_ids):
            return [{
                "strategy_id": "IARIC_v1",
                "instrument_symbol": "AAPL",
                "net_qty": 2,
                "avg_price": 100.0,
                "open_risk_R": 0.5,
            }]

    class HydratedOMS(_DummyOMS):
        def __init__(self) -> None:
            super().__init__()
            self._family_strategy_ids = ["IARIC_v1"]
            self._oms_repo = FakeRepo()
            self._portfolio_risk_state = {"open_risk_R": 0.5}
            self._allocation_targets = {"strategies": {"IARIC_v1": 1.0}}
            self._account_state_provider = lambda: {"equity": 100_000.0, "raw_nav": 100_000.0}

    async def _run():
        manager = InstrumentationManager(
            oms=HydratedOMS(),
            strategy_id="IARIC_v1",
            strategy_type="strategy_iaric",
            family_strategy_ids=["IARIC_v1"],
            write_daily_closeout_on_stop=False,
            stop_sidecar_on_stop=False,
        )
        manager._config["data_dir"] = str(tmp_path)
        manager.sidecar.validate_configuration = MagicMock()
        manager.sidecar.start = MagicMock()

        await manager.start()

        [positions_path] = (tmp_path / "positions").glob("*.jsonl")
        [portfolio_path] = (tmp_path / "portfolio").glob("*.jsonl")
        [allocation_path] = (tmp_path / "allocations").glob("*.jsonl")
        position_event = json.loads(positions_path.read_text(encoding="utf-8").splitlines()[0])
        portfolio_event = json.loads(portfolio_path.read_text(encoding="utf-8").splitlines()[0])
        allocation_event = json.loads(allocation_path.read_text(encoding="utf-8").splitlines()[0])

        assert position_event["symbol"] == "AAPL"
        assert position_event["qty"] == 2.0
        assert position_event["open_risk_R"] == 0.5
        assert "positions" not in position_event
        assert portfolio_event["portfolio_heat_R"] == 0.5
        assert portfolio_event["reconciliation_status"] == "startup_snapshot"
        assert "portfolio_state" not in portfolio_event
        assert allocation_event["strategy_target_weights"]["IARIC_v1"] == 1.0
        assert "allocation_state" not in allocation_event
        await manager.stop()

    asyncio.run(_run())
