"""Unit tests for PostgresSink with mocked psycopg pool."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import ModuleType

import pytest

from crypto_trader.instrumentation.types import (
    DailySnapshot,
    ErrorEvent,
    EventMetadata,
    HealthReportSnapshot,
    InstrumentedTradeEvent,
    MarketContext,
    MissedOpportunityEvent,
    PipelineFunnelSnapshot,
)


# ---------------------------------------------------------------------------
# Mock psycopg_pool at module level (psycopg not installed in dev)
# ---------------------------------------------------------------------------

_mock_psycopg_pool = ModuleType("psycopg_pool")
_MockConnectionPool = MagicMock()
_mock_psycopg_pool.ConnectionPool = _MockConnectionPool
sys.modules.setdefault("psycopg_pool", _mock_psycopg_pool)

from crypto_trader.instrumentation.postgres_sink import PostgresSink  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**overrides) -> EventMetadata:
    defaults = dict(
        event_id="evt_001",
        bot_id="test_bot",
        strategy_id="momentum",
        exchange_timestamp=datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return EventMetadata(**defaults)


def _make_trade_event(**overrides) -> InstrumentedTradeEvent:
    defaults = dict(
        metadata=_make_metadata(),
        trade_id="t_001",
        pair="BTC",
        side="long",
        entry_time=datetime(2026, 4, 20, 10, 0, 0, tzinfo=timezone.utc),
        exit_time=datetime(2026, 4, 20, 14, 0, 0, tzinfo=timezone.utc),
        entry_price=90000.0,
        exit_price=91000.0,
        position_size=0.1,
        pnl=100.0,
        price_pnl_gross=102.5,
        total_fees=2.0,
        realized_pnl_net=100.0,
        commission=2.0,
        funding_paid=0.5,
        setup_grade="A",
        exit_reason="trailing_stop",
        entry_method="market",
        confluences=["ema_stack", "adx_trend"],
        r_multiple=1.5,
        mae_r=-0.3,
        mfe_r=2.0,
        exit_efficiency=0.75,
    )
    defaults.update(overrides)
    return InstrumentedTradeEvent(**defaults)


def _make_daily_event(**overrides) -> DailySnapshot:
    defaults = dict(
        metadata=_make_metadata(),
        date="2026-04-20",
        total_trades=5,
        win_count=3,
        loss_count=2,
        gross_pnl=500.0,
        net_pnl=480.0,
        max_drawdown_pct=2.5,
        sharpe_rolling_30d=1.8,
        sortino_rolling_30d=2.1,
        per_strategy_summary={"momentum": {"trades": 3}},
    )
    defaults.update(overrides)
    return DailySnapshot(**defaults)


def _make_sink(error_callback=None):
    """Create PostgresSink with a fresh mock pool and connection."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    # connection() returns a context manager yielding mock_conn
    mock_pool.connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_pool.connection.return_value.__exit__ = MagicMock(return_value=False)

    _MockConnectionPool.reset_mock()
    _MockConnectionPool.side_effect = None
    _MockConnectionPool.return_value = mock_pool

    sink = PostgresSink(
        "postgresql://test:test@localhost/test",
        error_callback=error_callback,
    )
    return sink, mock_conn, mock_pool


def _execute_call(mock_conn, sql_fragment: str):
    for call in mock_conn.execute.call_args_list:
        if sql_fragment in call.args[0]:
            return call
    raise AssertionError(f"SQL fragment not found: {sql_fragment}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWriteTrade:
    def test_maps_fields_correctly(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event()

        sink.write_trade(event)

        sql, params = _execute_call(mock_conn, "INSERT INTO trades").args
        assert "INSERT INTO trades" in sql
        assert "ON CONFLICT (trade_id) DO NOTHING" in sql
        # Verify key field positions
        assert params[0] == "t_001"  # trade_id
        assert params[1] == "momentum"  # strategy_id
        assert params[2] == "BTC"  # symbol
        assert params[3] == "long"  # direction
        assert params[9] == 100.0  # pnl
        assert params[10] == 100.0  # net_pnl uses canonical realized net, no funding double-count
        assert params[11] == 1.5  # r_multiple

    def test_prefers_explicit_realized_net_pnl(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event(pnl=100.0, realized_pnl_net=97.5)

        sink.write_trade(event)

        _, params = _execute_call(mock_conn, "INSERT INTO trades").args
        assert params[10] == 97.5

    def test_keeps_zero_realized_net_pnl(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event(pnl=12.0, realized_pnl_net=0.0)

        sink.write_trade(event)

        _, params = _execute_call(mock_conn, "INSERT INTO trades").args
        assert params[10] == 0.0

    def test_legacy_event_without_explicit_economics_falls_back_to_pnl(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event(
            pnl=12.0,
            price_pnl_gross=0.0,
            total_fees=0.0,
            funding_paid=0.0,
            realized_pnl_net=0.0,
        )

        sink.write_trade(event)

        _, params = _execute_call(mock_conn, "INSERT INTO trades").args
        assert params[10] == 12.0

    def test_idempotent_no_exception(self):
        """Duplicate trade_id should not raise (ON CONFLICT DO NOTHING)."""
        sink, mock_conn, _ = _make_sink()
        event = _make_trade_event()

        # Call twice — should not raise
        sink.write_trade(event)
        sink.write_trade(event)
        assert len([
            call for call in mock_conn.execute.call_args_list
            if "INSERT INTO trades" in call.args[0]
        ]) == 2
        assert len([
            call for call in mock_conn.execute.call_args_list
            if "INSERT INTO instrumentation_events" in call.args[0]
        ]) == 2

    def test_with_market_context(self):
        sink, mock_conn, _ = _make_sink()
        ctx = MarketContext(atr=100.0, adx=25.0, rsi=55.0)
        event = _make_trade_event(market_context=ctx)

        sink.write_trade(event)

        _, params = _execute_call(mock_conn, "INSERT INTO trades").args
        # market_context should be JSON string
        mc_json = params[-1]  # last param
        parsed = json.loads(mc_json)
        assert parsed["atr"] == 100.0
        assert parsed["adx"] == 25.0


class TestWriteDaily:
    def test_upserts_daily_snapshot(self):
        sink, mock_conn, _ = _make_sink()
        event = _make_daily_event()

        sink.write_daily(event)

        sql, params = _execute_call(mock_conn, "INSERT INTO daily_snapshots").args
        assert "INSERT INTO daily_snapshots" in sql
        assert "ON CONFLICT (trade_date) DO UPDATE" in sql
        assert params[0] == "2026-04-20"
        assert params[1] == 5  # total_trades
        assert params[2] == 3  # win_count


class TestWriteHealthReport:
    def test_extracts_assessment(self):
        sink, mock_conn, _ = _make_sink()
        event = HealthReportSnapshot(
            timestamp="2026-04-20T12:00:00+00:00",
            report={
                "assessment": "healthy",
                "uptime_sec": 3600.0,
                "alerts": ["stale_feed_BTC_15m"],
                "positions": [],
            },
        )

        sink.write_health_report(event)

        sql, params = _execute_call(mock_conn, "INSERT INTO health_snapshots").args
        assert "INSERT INTO health_snapshots" in sql
        assert params[1] == "healthy"  # assessment extracted from report
        assert params[2] == 3600.0  # uptime_sec


class TestWriteEquity:
    def test_inserts_equity_snapshot(self):
        sink, mock_conn, _ = _make_sink()
        ts = datetime(2026, 4, 20, 12, 0, 0, tzinfo=timezone.utc)

        sink.write_equity(10500.0, ts)

        mock_conn.execute.assert_called_once()
        sql, params = mock_conn.execute.call_args.args
        assert "INSERT INTO equity_snapshots" in sql
        assert params[0] == ts
        assert params[1] == 10500.0


class TestUpsertPositions:
    def test_full_sync_delete_then_insert(self):
        sink, mock_conn, _ = _make_sink()
        positions = [
            {
                "strategy_id": "momentum",
                "symbol": "BTC",
                "direction": "long",
                "qty": 0.1,
                "avg_entry": 90000.0,
                "unrealized_pnl": 100.0,
                "risk_r": 0.5,
                "stop_price": 89000.0,
                "entry_time": datetime(2026, 4, 20, 10, 0, tzinfo=timezone.utc),
            },
            {
                "strategy_id": "trend",
                "symbol": "ETH",
                "direction": "short",
                "qty": 1.0,
                "avg_entry": 3000.0,
                "unrealized_pnl": -20.0,
                "risk_r": 0.3,
            },
        ]

        sink.upsert_positions(positions)

        calls = mock_conn.execute.call_args_list
        # First call: DELETE
        assert "DELETE FROM positions" in calls[0].args[0]
        # Then 2 INSERTs
        assert "INSERT INTO positions" in calls[1].args[0]
        assert "INSERT INTO positions" in calls[2].args[0]
        # Verify first position params
        assert calls[1].args[1][1] == "BTC"
        # Verify second position params
        assert calls[2].args[1][1] == "ETH"


class TestAllocationTables:
    def test_upserts_strategy_position_allocations_without_touching_legacy_positions(self):
        sink, mock_conn, _ = _make_sink()

        sink.upsert_strategy_position_allocations([{
            "position_instance_id": "pos_1",
            "strategy_id": "momentum",
            "symbol": "BTC",
            "direction": "LONG",
            "allocated_qty": 0.1,
            "avg_entry": 90_000.0,
            "risk_r": 0.5,
            "entry_time": "2026-06-04T00:00:00+00:00",
            "status": "OPEN",
            "confidence": "exact",
            "source": "lifecycle",
            "entry_order_ids": ["entry_1"],
            "entry_fill_ids": ["fill_1"],
            "exit_order_ids": [],
            "exit_fill_ids": [],
            "metadata": {},
        }])

        calls = mock_conn.execute.call_args_list
        assert "DELETE FROM strategy_position_allocations" in calls[0].args[0]
        assert "INSERT INTO strategy_position_allocations" in calls[1].args[0]
        assert all("DELETE FROM positions" not in call.args[0] for call in calls)

    def test_upserts_exchange_positions_separately_from_strategy_rows(self):
        sink, mock_conn, _ = _make_sink()

        sink.upsert_exchange_positions([{
            "symbol": "BTC",
            "direction": "LONG",
            "qty": 0.2,
            "avg_entry": 90_000.0,
            "unrealized_pnl": 50.0,
            "liquidation_price": None,
            "observed_at": "2026-06-04T00:00:00+00:00",
            "metadata": {},
        }])

        calls = mock_conn.execute.call_args_list
        assert "DELETE FROM exchange_positions" in calls[0].args[0]
        assert "INSERT INTO exchange_positions" in calls[1].args[0]
        assert all("INSERT INTO positions" not in call.args[0] for call in calls)


class TestGenericOnlyMethods:
    def test_events_without_typed_tables_write_generic_events(self):
        sink, mock_conn, _ = _make_sink()

        sink.write_missed(MagicMock(spec=MissedOpportunityEvent))
        sink.write_error(MagicMock(spec=ErrorEvent))
        sink.write_funnel(MagicMock(spec=PipelineFunnelSnapshot))

        assert len([
            call for call in mock_conn.execute.call_args_list
            if "INSERT INTO instrumentation_events" in call.args[0]
        ]) == 3


def test_instrumentation_events_indexes_target_canonical_payload_join_keys():
    migration = Path("infra/postgres/migrations/003_instrumentation_events.sql").read_text(encoding="utf-8")

    assert "payload->'payload'->>'decision_id'" in migration
    assert "payload->'payload'->>'bar_id'" in migration


def test_position_allocation_migration_is_additive():
    migration = Path("infra/postgres/migrations/004_position_allocations.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS strategy_position_allocations" in migration
    assert "CREATE TABLE IF NOT EXISTS exchange_positions" in migration
    assert "DROP TABLE positions" not in migration
    assert "CREATE OR REPLACE VIEW positions" not in migration


class TestConnectionErrorHandling:
    def test_write_trade_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.write_trade(_make_trade_event())

    def test_write_event_emits_structured_error_callback(self):
        errors: list[dict] = []
        sink, mock_conn, _ = _make_sink(error_callback=errors.append)
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        sink.write_event("trade", _make_trade_event())

        assert len(errors) == 1
        assert errors[0]["component"] == "postgres_sink"
        assert errors[0]["error_type"] == "RuntimeError"
        assert errors[0]["event_type"] == "trade"
        assert errors[0]["recovery_action"] == "continue_without_postgres"
        assert "connection refused" in errors[0]["message"]

    def test_write_equity_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.write_equity(10000.0, datetime.now(timezone.utc))

    def test_upsert_positions_swallows_exception(self):
        sink, mock_conn, _ = _make_sink()
        mock_conn.execute.side_effect = RuntimeError("connection refused")

        # Should not raise
        sink.upsert_positions([{"symbol": "BTC", "direction": "long", "qty": 0.1, "avg_entry": 90000.0}])

    def test_close_swallows_exception(self):
        sink, _, mock_pool = _make_sink()
        mock_pool.close.side_effect = RuntimeError("already closed")

        # Should not raise
        sink.close()

    def test_engine_postgres_init_failure_emits_error_event(self, tmp_path):
        from crypto_trader.live.config import LiveConfig
        from crypto_trader.live.engine import LiveEngine

        _MockConnectionPool.side_effect = RuntimeError("pool unavailable")
        engine = None
        try:
            engine = LiveEngine(LiveConfig(
                state_dir=tmp_path,
                data_dir=tmp_path / "data",
                bot_id="bot1",
                postgres_dsn="postgresql://test:test@localhost/test",
                postgres_async_enabled=False,
            ))

            row = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
            assert row["component"] == "postgres_sink"
            assert row["error_type"] == "RuntimeError"
            assert row["recovery_action"] == "disable_postgres_sink"
            assert "pool unavailable" in row["message"]
        finally:
            _MockConnectionPool.side_effect = None
            if engine is not None:
                engine._oms.close()

    def test_engine_error_event_has_pre_lineage_startup_fallback(self, tmp_path):
        from crypto_trader.instrumentation.emitter import EventEmitter
        from crypto_trader.instrumentation.sinks import JsonlSink
        from crypto_trader.live.config import LiveConfig
        from crypto_trader.live.engine import LiveEngine

        engine = object.__new__(LiveEngine)
        engine._config = LiveConfig(
            state_dir=tmp_path,
            data_dir=tmp_path / "data",
            bot_id="bot1",
            family_id="crypto_perps",
            portfolio_id="paper",
            account_alias="acct",
            symbols=["BTC"],
        )
        engine._emitter = EventEmitter()
        engine._emitter.add_sink(JsonlSink(tmp_path))

        engine._emit_error_event(
            "postgres_sink",
            RuntimeError("pool unavailable"),
            severity="medium",
            recovery_action="disable_postgres_sink",
            error_type="RuntimeError",
        )

        row = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["component"] == "postgres_sink"
        assert row["error_type"] == "RuntimeError"
        assert row["recovery_action"] == "disable_postgres_sink"
        assert row["metadata"]["portfolio_id"] == "paper"
        assert row["lineage"]["symbol_universe"] == ["BTC"]

    def test_engine_postgres_error_callback_skips_postgres_sink_recursion(self, tmp_path):
        from crypto_trader.instrumentation.emitter import EventEmitter
        from crypto_trader.instrumentation.sinks import JsonlSink
        from crypto_trader.live.config import LiveConfig
        from crypto_trader.live.engine import LiveEngine

        class FakePostgresSink:
            def __init__(self) -> None:
                self.errors: list[ErrorEvent] = []

            def write_error(self, event: ErrorEvent) -> None:
                self.errors.append(event)

        engine = object.__new__(LiveEngine)
        engine._config = LiveConfig(
            state_dir=tmp_path,
            data_dir=tmp_path / "data",
            bot_id="bot1",
            portfolio_id="paper",
            symbols=["BTC"],
        )
        engine._emitter = EventEmitter()
        engine._emitter.add_sink(JsonlSink(tmp_path))
        pg_sink = FakePostgresSink()
        engine._pg_sink = pg_sink
        engine._emitter.add_sink(pg_sink)

        engine._emit_postgres_error_event({
            "component": "postgres_sink",
            "message": "queue full",
            "error_type": "QueueFull",
            "recovery_action": "jsonl_backfill_required",
            "severity": "critical",
        })

        row = json.loads((tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["component"] == "postgres_sink"
        assert row["error_type"] == "QueueFull"
        assert row["recovery_action"] == "jsonl_backfill_required"
        assert pg_sink.errors == []
