"""Per-symbol freshness tracking in ETFCoreLiveEngine.liveness_payload.

The watchdog (apps/watchdog/checks.py:check_liveness) reads
``last_decision_details.liveness.symbol_freshness`` to detect per-symbol
data stalls. Before this fix, ``ETFCoreLiveEngine.liveness_payload`` (used
by TPC) did not emit the field, so the watchdog silently skipped TPC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from strategies.swing._shared.etf_core import ETFBarInput
from strategies.swing._shared.etf_live_engine import ETFCoreLiveEngine


def _make_engine(symbols: list[str]) -> ETFCoreLiveEngine:
    state = SimpleNamespace(positions=[], pending_orders=[], last_bar_ts=None)

    class _CoreLogic:
        @staticmethod
        def on_bar(_state, bar_input: ETFBarInput, _cfg):
            new = SimpleNamespace(
                positions=[],
                pending_orders=[],
                last_bar_ts=bar_input.timestamp,
            )
            return new, [], []

    class _Serializers:
        @staticmethod
        def restore_state(_snapshot):
            return state

        @staticmethod
        def snapshot_state(_state):
            return {}

    return ETFCoreLiveEngine(
        strategy_id="TPC",
        ib_session=object(),
        oms_service=object(),
        instruments={s: object() for s in symbols},
        config={s: SimpleNamespace() for s in symbols},
        core_logic=_CoreLogic,
        serializers=_Serializers,
    )


def _bar(symbol: str, ts: datetime | None) -> ETFBarInput:
    return ETFBarInput(symbol=symbol, timestamp=ts, equity=10_000.0)


def test_symbol_freshness_tracks_latest_bar_per_symbol() -> None:
    engine = _make_engine(["QQQ", "SPY"])
    t1 = datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(minutes=15)
    t3 = t1 + timedelta(minutes=30)

    engine.process_bar_input(_bar("QQQ", t1))
    engine.process_bar_input(_bar("SPY", t2))
    engine.process_bar_input(_bar("QQQ", t3))

    payload = engine.liveness_payload()
    assert payload["bars_processed"] == 3
    assert payload["symbol_freshness"] == {
        "QQQ": t3.isoformat(),
        "SPY": t2.isoformat(),
    }


def test_symbol_freshness_falls_back_to_state_last_bar_ts() -> None:
    """When ETFBarInput.timestamp is None, fall back to state.last_bar_ts."""
    engine = _make_engine(["QQQ"])
    state_ts = datetime(2026, 5, 10, 16, 30, tzinfo=timezone.utc)

    class _CoreLogic:
        @staticmethod
        def on_bar(_state, bar_input, _cfg):
            return (
                SimpleNamespace(positions=[], pending_orders=[], last_bar_ts=state_ts),
                [],
                [],
            )

    engine._core_logic = _CoreLogic
    engine.process_bar_input(_bar("QQQ", None))

    assert engine.liveness_payload()["symbol_freshness"] == {"QQQ": state_ts.isoformat()}


def test_symbol_freshness_excludes_inactive_symbols() -> None:
    """If a symbol is dropped from self._config, it disappears from the payload."""
    engine = _make_engine(["QQQ", "SPY"])
    t1 = datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(minutes=15)

    engine.process_bar_input(_bar("QQQ", t1))
    engine.process_bar_input(_bar("SPY", t2))
    assert set(engine.liveness_payload()["symbol_freshness"]) == {"QQQ", "SPY"}

    engine._config.pop("SPY")
    payload = engine.liveness_payload()
    assert set(payload["symbol_freshness"]) == {"QQQ"}
    assert payload["symbol_freshness"]["QQQ"] == t1.isoformat()


def test_unconfigured_symbol_is_dropped() -> None:
    """A bar for a symbol not in self._config short-circuits before _bars_processed updates."""
    engine = _make_engine(["QQQ"])
    t1 = datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc)

    engine.process_bar_input(_bar("FOO", t1))
    payload = engine.liveness_payload()
    assert payload["bars_processed"] == 0
    assert payload["symbol_freshness"] == {}
