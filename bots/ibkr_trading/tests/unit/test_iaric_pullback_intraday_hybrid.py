from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.iaric_pullback_engine import IARICPullbackEngine, IARICPullbackResult, _daily_signal_bundle
from backtests.stock.engine.iaric_pullback_intraday_hybrid_engine import IARICPullbackIntradayHybridEngine, _PBHybridState
from strategies.stock.iaric.config import StrategySettings
from strategies.stock.iaric.models import Bar, MarketSnapshot, RegimeSnapshot, WatchlistArtifact


def _business_dates(count: int, start: date) -> list[date]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


class _FakeReplay:
    def __init__(self) -> None:
        self._dates = _business_dates(70, date(2025, 9, 1))
        self.trade_date = self._dates[-1]
        self.prev_date = self._dates[-2]
        self._universe = [("AAA", "Tech", None)]

        closes = np.linspace(100.0, 127.0, num=len(self._dates))
        closes[-3] = 126.0
        closes[-2] = 124.0
        closes[-1] = 124.6
        opens = closes - 0.4
        highs = closes + 1.0
        lows = closes - 1.2

        bars = self._build_intraday_bars(self.trade_date)
        opens[-1] = bars[0].open
        highs[-1] = max(bar.high for bar in bars)
        lows[-1] = min(bar.low for bar in bars)
        closes[-1] = bars[-1].close

        self._daily_arrs = {
            "AAA": {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.full(len(self._dates), 1_000_000.0),
            }
        }
        self._daily_didx = {"AAA": (self._dates, list(range(len(self._dates))))}
        self._daily_flow = {"AAA": np.ones(len(self._dates), dtype=float)}
        self._bars_by_date = {("AAA", self.trade_date): bars}

    def _build_intraday_bars(self, trade_date: date) -> list[Bar]:
        specs = [
            (123.0, 123.1, 122.0, 122.1, 1_000),
            (122.1, 122.2, 121.8, 121.9, 1_200),
            (121.9, 123.1, 121.8, 122.95, 2_500),
            (122.95, 123.4, 122.7, 123.3, 1_800),
            (123.3, 124.8, 123.2, 124.6, 2_200),
            (124.6, 124.9, 124.3, 124.8, 1_800),
            (124.8, 124.85, 124.4, 124.5, 1_500),
            (124.5, 124.7, 124.4, 124.6, 1_400),
            (124.6, 124.75, 124.5, 124.65, 1_300),
            (124.65, 124.8, 124.55, 124.7, 1_250),
            (124.7, 124.85, 124.6, 124.75, 1_200),
            (124.75, 124.9, 124.65, 124.8, 1_150),
        ]
        bars: list[Bar] = []
        current = datetime.combine(trade_date, time(14, 30), tzinfo=timezone.utc)
        for open_, high, low, close, volume in specs:
            end = current + timedelta(minutes=5)
            bars.append(
                Bar(
                    symbol="AAA",
                    start_time=current,
                    end_time=end,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            current = end
        return bars

    def tradable_dates(self, start: date, end: date) -> list[date]:
        return [day for day in self._dates if start <= day <= end]

    def get_prev_trading_date(self, trade_date: date) -> date | None:
        idx = self._dates.index(trade_date)
        return self._dates[idx - 1] if idx > 0 else None

    def get_next_trading_date(self, trade_date: date) -> date | None:
        idx = self._dates.index(trade_date)
        return self._dates[idx + 1] if idx + 1 < len(self._dates) else None

    def get_daily_ohlc(self, symbol: str, trade_date: date):
        idx = self._dates.index(trade_date)
        arrs = self._daily_arrs[symbol]
        return (
            float(arrs["open"][idx]),
            float(arrs["high"][idx]),
            float(arrs["low"][idx]),
            float(arrs["close"][idx]),
        )

    def get_daily_close(self, symbol: str, trade_date: date) -> float | None:
        return self.get_daily_ohlc(symbol, trade_date)[3]

    def get_flow_proxy_last_n(self, symbol: str, trade_date: date, n: int):
        idx = self._dates.index(trade_date)
        arr = self._daily_flow[symbol]
        start = max(0, idx - n + 1)
        return list(arr[start : idx + 1])

    def iaric_selection_for_date(self, trade_date: date, settings=None) -> WatchlistArtifact:
        return WatchlistArtifact(
            trade_date=trade_date,
            generated_at=datetime(trade_date.year, trade_date.month, trade_date.day, tzinfo=timezone.utc),
            regime=RegimeSnapshot(
                score=0.75,
                tier="B",
                risk_multiplier=1.0,
                price_ok=True,
                breadth_ok=True,
                vol_ok=True,
                credit_ok=True,
            ),
            items=[],
            tradable=[],
            overflow=[],
            market_wide_institutional_selling=False,
            held_positions=[],
        )

    def get_5m_bar_objects_for_date(self, symbol: str, trade_date: date) -> list[Bar]:
        return list(self._bars_by_date.get((symbol, trade_date), []))


class _FakeReplayCarryWindow(_FakeReplay):
    def __init__(self) -> None:
        super().__init__()
        next_date = self.trade_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        self._dates.append(next_date)
        for key, value in {
            "open": 124.5,
            "high": 125.2,
            "low": 123.9,
            "close": 124.8,
            "volume": 1_000_000.0,
        }.items():
            self._daily_arrs["AAA"][key] = np.append(self._daily_arrs["AAA"][key], value)
        self._daily_flow["AAA"] = np.append(self._daily_flow["AAA"], 1.0)
        self._daily_didx["AAA"] = (self._dates, list(range(len(self._dates))))


class _FakeReplayNoIntraday(_FakeReplay):
    def __init__(self) -> None:
        super().__init__()
        self._bars_by_date = {}


class _FakeReplayOnlyMarketOpenBar(_FakeReplay):
    def __init__(self) -> None:
        super().__init__()
        self._bars_by_date[("AAA", self.trade_date)] = self._bars_by_date[("AAA", self.trade_date)][:1]


class _FakeReplayDelayedConfirm(_FakeReplay):
    def _build_intraday_bars(self, trade_date: date) -> list[Bar]:
        specs = [
            (123.0, 123.05, 122.90, 122.98, 1_000),
            (122.98, 123.00, 122.75, 122.88, 1_050),
            (122.88, 123.05, 122.82, 122.99, 1_100),
            (122.99, 123.15, 122.95, 123.10, 1_200),
            (123.10, 123.25, 123.00, 123.18, 1_350),
            (123.18, 123.45, 123.15, 123.42, 2_600),
            (123.42, 124.20, 123.35, 124.00, 2_200),
            (124.00, 124.50, 123.90, 124.30, 1_900),
            (124.30, 124.65, 124.10, 124.50, 1_700),
        ]
        bars: list[Bar] = []
        current = datetime.combine(trade_date, time(14, 30), tzinfo=timezone.utc)
        for open_, high, low, close, volume in specs:
            end = current + timedelta(minutes=5)
            bars.append(
                Bar(
                    symbol="AAA",
                    start_time=current,
                    end_time=end,
                    open=open_,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                )
            )
            current = end
        return bars


class _FakeReplayRefinePriority(_FakeReplay):
    def __init__(self) -> None:
        super().__init__()
        self._universe = [("AAA", "Tech", None), ("BBB", "Tech", None)]
        self._daily_arrs["BBB"] = {key: value.copy() for key, value in self._daily_arrs["AAA"].items()}
        self._daily_didx["BBB"] = self._daily_didx["AAA"]
        self._daily_flow["BBB"] = self._daily_flow["AAA"].copy()
        self._bars_by_date = {("AAA", self.trade_date): list(self._bars_by_date[("AAA", self.trade_date)])}


class _FakeReplayCoverageUniverse(_FakeReplayRefinePriority):
    def __init__(self) -> None:
        super().__init__()
        self._5m_didx = {"AAA": ([self.trade_date], [0])}
        self._intraday_5m_cache = {"AAA": object()}


def test_dispatcher_uses_daily_engine_stub(monkeypatch):
    sentinel = IARICPullbackResult(
        trades=[],
        equity_curve=np.array([10_000.0]),
        timestamps=np.array([]),
        daily_selections={},
    )

    class _StubDaily:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            return sentinel

    from backtests.stock.engine import iaric_pullback_engine as engine_module

    monkeypatch.setattr(engine_module, "IARICPullbackDailyEngine", _StubDaily)
    config = IARICBacktestConfig(
        start_date="2026-01-01",
        end_date="2026-01-02",
        param_overrides={"pb_execution_mode": "daily"},
    )

    result = IARICPullbackEngine(config, replay=SimpleNamespace(
        _universe=[], tradable_dates=lambda s, e: [], _daily_arrs={}, _daily_flow={},
    )).run()

    assert result is sentinel


def test_intraday_hybrid_engine_enters_and_logs_fsm():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_score_family": "route_momentum_v1",
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_partial_r": 1.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "EOD_FLATTEN"
    assert trade.metadata["intraday_setup_type"] in {"OPENING_FLUSH", "SESSION_FLUSH", "DELAYED_CONFIRM"}
    assert trade.metadata["entry_trigger"] in {"OPENING_RECLAIM", "DELAYED_CONFIRM"}
    assert trade.metadata["entry_route_family"] in {"OPENING_RECLAIM", "DELAYED_CONFIRM"}
    assert "bars_to_exit" in trade.metadata
    assert "entry_score_component_daily_signal" in trade.metadata
    assert "carry_decision_path" in trade.metadata
    assert result.fsm_log
    assert any(row["to_state"] == "READY" for row in result.fsm_log)
    assert any(row["to_state"] == "IN_POSITION" for row in result.fsm_log)
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] == "entered"


def test_intraday_hybrid_queues_ready_entries_for_next_bar_open_fill():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_score_family": "route_momentum_v1",
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_partial_r": 1.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    meta = trade.metadata
    bars = replay.get_5m_bar_objects_for_date("AAA", replay.trade_date)

    assert meta["accepted_bar_index"] >= 0
    assert meta["entry_bar_index"] == meta["accepted_bar_index"] + 1
    assert trade.entry_time == bars[meta["entry_bar_index"]].start_time
    assert meta["accepted_timestamp"] == bars[meta["accepted_bar_index"]].end_time.isoformat()
    assert meta["ready_timestamp"]

    assert result.candidate_ledger is not None
    ledger = result.candidate_ledger[replay.trade_date][0]
    assert ledger["accepted_bar_index"] + 1 == ledger["entry_bar_index"]


def test_intraday_hybrid_does_not_carry_when_carry_is_disabled():
    replay = _FakeReplayCarryWindow()
    replay._bars_by_date = {}
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.get_next_trading_date(replay.trade_date).isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_carry_enabled": False,
            "pb_partial_r": 1.0,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "EOD_FLATTEN"
    assert trade.metadata["entry_trigger"] == "OPEN_SCORED_ENTRY"
    assert trade.metadata["carry_profile"] == "OPEN_SCORED"
    assert trade.metadata["carry_binary_ok"] is False


def test_intraday_hybrid_route_specific_open_scored_carry_can_hold_overnight():
    replay = _FakeReplayCarryWindow()
    replay._bars_by_date = {}
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.get_next_trading_date(replay.trade_date).isoformat(),
        param_overrides={
            "pb_max_hold_days": 10,
            "pb_open_scored_max_hold_days": 10,
            "pb_execution_mode": "intraday_hybrid",
            "pb_carry_enabled": True,
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_open_scored_carry_min_r": -0.10,
            "pb_open_scored_carry_close_pct_min": 0.0,
            "pb_open_scored_carry_mfe_gate_r": 0.0,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
            "pb_carry_score_threshold": 0.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason in {"END_OF_BACKTEST", "EOD_FLATTEN"}
    assert trade.hold_bars == 2
    assert trade.metadata["entry_trigger"] == "OPEN_SCORED_ENTRY"
    assert trade.metadata["carry_profile"] == "OPEN_SCORED"
    assert trade.metadata["carry_binary_ok"] is True
    assert trade.metadata["carry_decision_path"] == "binary"


def test_intraday_hybrid_route_specific_carry_daily_signal_floor_blocks_daily_fallback_carry():
    replay = _FakeReplayCarryWindow()
    replay._bars_by_date = {}
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.get_next_trading_date(replay.trade_date).isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_carry_enabled": True,
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
            "pb_open_scored_carry_min_r": -0.10,
            "pb_open_scored_carry_close_pct_min": 0.0,
            "pb_open_scored_carry_mfe_gate_r": 0.0,
            "pb_open_scored_carry_min_daily_signal_score": 1000.0,
            "pb_open_scored_carry_score_fallback_enabled": False,
            "pb_open_scored_carry_score_threshold": 0.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "EOD_FLATTEN"
    assert trade.hold_bars == 1
    assert trade.metadata["carry_profile"] == "OPEN_SCORED"
    assert trade.metadata["carry_binary_ok"] is False
    assert trade.metadata["carry_score_ok"] is False
    assert trade.metadata["carry_decision_path"] == "flatten"


def test_intraday_hybrid_route_specific_score_fallback_can_carry_daily_fallback_positions():
    replay = _FakeReplayCarryWindow()
    replay._bars_by_date = {}
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.get_next_trading_date(replay.trade_date).isoformat(),
        param_overrides={
            "pb_max_hold_days": 10,
            "pb_open_scored_max_hold_days": 10,
            "pb_execution_mode": "intraday_hybrid",
            "pb_carry_enabled": True,
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
            "pb_open_scored_carry_min_r": 10.0,
            "pb_open_scored_carry_close_pct_min": 0.0,
            "pb_open_scored_carry_mfe_gate_r": 0.0,
            "pb_open_scored_carry_score_fallback_enabled": True,
            "pb_open_scored_carry_score_threshold": 0.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason in {"END_OF_BACKTEST", "EOD_FLATTEN"}
    assert trade.hold_bars == 2
    assert trade.metadata["carry_binary_ok"] is False
    assert trade.metadata["carry_score_ok"] is True
    assert trade.metadata["carry_decision_path"] == "score_fallback"


def test_intraday_hybrid_requires_5m_bars():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_intraday_bar_minutes": 30,
        },
    )

    with pytest.raises(ValueError, match="requires 5-minute bars"):
        IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()


def test_intraday_hybrid_open_scored_defaults_to_first_eligible_5m_open():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_carry_enabled": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    bars = replay.get_5m_bar_objects_for_date("AAA", replay.trade_date)
    expected_fill = round(bars[1].open + bars[1].open * config.slippage.slip_bps_normal / 10_000, 2)
    assert trade.metadata["entry_trigger"] == "OPEN_SCORED_ENTRY"
    assert trade.entry_time == bars[1].start_time
    assert trade.entry_price == expected_fill
    assert trade.metadata["entry_bar_index"] == 1
    assert result.candidate_ledger is not None
    ledger = result.candidate_ledger[replay.trade_date][0]
    assert ledger["entry_open"] == bars[1].open
    assert ledger["entry_price"] == expected_fill
    assert ledger["entry_bar_index"] == 1
    assert ledger["open_scored_fill_timing"] == "next_5m_open"
    assert trade.risk_per_share == pytest.approx(ledger["risk_per_share"])


def test_intraday_hybrid_default_open_scored_rejects_when_5m_missing():
    replay = _FakeReplayNoIntraday()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_rescue_flow_enabled": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 0
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] == "open_scored_no_post_open_bar"


def test_intraday_hybrid_default_open_scored_rejects_when_no_post_open_bar():
    replay = _FakeReplayOnlyMarketOpenBar()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_opening_reclaim_enabled": False,
            "pb_delayed_confirm_enabled": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 0
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] == "open_scored_no_post_open_bar"


def test_intraday_hybrid_core_candidate_legacy_same_open_fallback_when_intraday_missing():
    replay = _FakeReplayNoIntraday()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_rescue_flow_enabled": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.metadata["entry_trigger"] == "OPEN_SCORED_ENTRY"
    assert trade.metadata["entry_route_family"] == "OPEN_SCORED_ENTRY"
    assert trade.metadata["intraday_setup_type"] == "OPEN_SCORED_ENTRY"
    assert trade.entry_time.hour == 14
    assert trade.entry_time.minute == 30


def test_intraday_hybrid_missing_5m_open_entry_still_requires_open_route_quality_gate():
    replay = _FakeReplayNoIntraday()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_open_scored_min_score": 1000.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 0
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] == "open_scored_gate_reject"


def test_intraday_hybrid_daily_signal_floor_blocks_weak_candidates_before_routing():
    replay = _FakeReplayNoIntraday()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_daily_signal_min_score": 1000.0,
            "pb_v2_signal_floor": 1000.0,
            "pb_v2_enabled": False,
            "pb_open_scored_min_score": 0.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 0
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] == "daily_signal_floor_reject"


def test_intraday_hybrid_fallback_does_not_depend_on_diagnostics_mode():
    replay = _FakeReplayNoIntraday()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_rescue_flow_enabled": False,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=False).run()

    assert len(result.trades) == 1
    assert result.trades[0].metadata["entry_trigger"] == "OPEN_SCORED_ENTRY"


def test_intraday_hybrid_can_use_delayed_confirm_for_core_names():
    replay = _FakeReplayDelayedConfirm()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 55.0,
            "pb_delayed_confirm_after_bar": 5,
            "pb_delayed_confirm_score_min": 40.0,
            "pb_entry_strength_sizing": False,
            "pb_ready_min_volume_ratio": 0.5,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.metadata["intraday_setup_type"] == "DELAYED_CONFIRM"
    assert trade.metadata["entry_trigger"] == "DELAYED_CONFIRM"
    assert trade.entry_time.hour >= 14


def test_intraday_hybrid_intraday_scoring_does_not_depend_on_diagnostics_mode():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_score_family": "route_momentum_v1",
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=False).run()

    assert len(result.trades) == 1
    assert result.trades[0].metadata["entry_trigger"] in {"OPENING_RECLAIM", "DELAYED_CONFIRM"}


def test_intraday_hybrid_emits_live_aligned_decision_stream_and_trade_outcomes() -> None:
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_score_family": "route_momentum_v1",
            "pb_entry_strength_sizing": False,
            "pb_ready_min_volume_ratio": 0.5,
            "pb_partial_r": 1.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()
    codes = [event["code"] for event in result.decision_stream]

    assert len(result.trades) == 1
    assert "ENTRY_REQUESTED" in codes
    assert "ENTRY_FILLED" in codes
    assert "EXIT_FILLED" in codes
    assert len(result.trade_outcomes) == len(result.trades) == 1
    assert result.trade_outcomes[0]["symbol"] == result.trades[0].symbol
    assert result.trade_outcomes[0]["exit_reason"] == result.trades[0].exit_reason


def test_intraday_hybrid_entry_bundle_supports_round3_route_score_families():
    replay = _FakeReplay()
    bars = replay.get_5m_bar_objects_for_date("AAA", replay.trade_date)
    bar_idx = 4
    bar = bars[bar_idx]
    market = MarketSnapshot(
        symbol="AAA",
        session_vwap=123.4,
        last_30m_bar=bars[3],
    )
    state = _PBHybridState(
        symbol="AAA",
        item=SimpleNamespace(
            expected_5m_volume=400.0,
            average_30m_volume=2400.0,
            sponsorship_state="STRONG",
        ),
        record=None,
        trigger_type="RSI",
        entry_rsi=4.0,
        entry_gap_pct=-0.6,
        entry_sma_dist_pct=4.0,
        entry_cdd=2,
        entry_rank=4,
        entry_rank_pct=20.0,
        n_candidates=8,
        prev_iloc=0,
        sector="Tech",
        daily_atr=2.0,
        daily_signal_score=62.0,
        intraday_setup_type="DELAYED_CONFIRM",
        route_family="DELAYED_CONFIRM",
        reclaim_level=122.9,
        stop_level=122.0,
        flush_bar_idx=1,
    )

    quality_engine = IARICPullbackIntradayHybridEngine(
        IARICBacktestConfig(
            start_date=replay.trade_date.isoformat(),
            end_date=replay.trade_date.isoformat(),
            param_overrides={
                "pb_execution_mode": "intraday_hybrid",
                "pb_entry_score_family": "route_quality_v1",
                "pb_ready_min_volume_ratio": 0.5,
            },
        ),
        replay,
    )
    early_engine = IARICPullbackIntradayHybridEngine(
        IARICBacktestConfig(
            start_date=replay.trade_date.isoformat(),
            end_date=replay.trade_date.isoformat(),
            param_overrides={
                "pb_execution_mode": "intraday_hybrid",
                "pb_entry_score_family": "route_early_reversal_v1",
                "pb_ready_min_volume_ratio": 0.5,
            },
        ),
        replay,
    )

    quality_bundle = quality_engine._entry_score_bundle(state, bar, market, bars, bar_idx)
    early_bundle = early_engine._entry_score_bundle(state, bar, market, bars, bar_idx)

    assert quality_bundle["score"] > 0
    assert early_bundle["score"] > 0
    assert quality_bundle["daily_signal"] > early_bundle["daily_signal"]
    assert quality_bundle["reclaim"] < early_bundle["reclaim"]
    assert quality_bundle["score"] != early_bundle["score"]


def test_intraday_hybrid_reserves_capacity_for_5m_refinement():
    replay = _FakeReplayRefinePriority()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_score_family": "route_momentum_v1",
            "pb_ready_min_volume_ratio": 0.5,
            "pb_entry_strength_sizing": False,
            "pb_max_positions": 1,
            "pb_intraday_priority_reserve_slots": 1,
            "pb_open_scored_enabled": True,
            "pb_open_scored_min_score": 0.0,
            "pb_open_scored_rank_pct_max": 100.0,
            "pb_open_scored_fill_timing": "same_open",
            "pb_entry_rank_pct_max": 100.0,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 1
    assert result.trades[0].symbol == "AAA"
    assert result.candidate_ledger is not None
    day_records = {row["symbol"]: row for row in result.candidate_ledger[replay.trade_date]}
    assert day_records["BBB"]["disposition"] == "intraday_priority_reserve"
    assert day_records["BBB"]["blocked_by_capacity_reason"] == "intraday_priority_reserve"


def test_daily_pullback_engine_can_restrict_backtest_to_5m_covered_universe():
    replay = _FakeReplayCoverageUniverse()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "daily",
            "pb_rsi_entry": 20.0,
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_entry_rank_pct_max": 100.0,
            "pb_backtest_intraday_universe_only": True,
        },
    )

    result = IARICPullbackEngine(config, replay=replay).run()

    assert len(result.trades) == 1
    assert result.trades[0].symbol == "AAA"


def test_intraday_hybrid_can_disable_opening_reclaim_route():
    replay = _FakeReplay()
    config = IARICBacktestConfig(
        start_date=replay.trade_date.isoformat(),
        end_date=replay.trade_date.isoformat(),
        param_overrides={
            "pb_execution_mode": "intraday_hybrid",
            "pb_daily_signal_min_score": 0.0,
            "pb_v2_signal_floor": 0.0,
            "pb_v2_enabled": False,
            "pb_rsi_entry": 20.0,
            "pb_entry_score_min": 45.0,
            "pb_entry_strength_sizing": False,
            "pb_opening_reclaim_enabled": False,
            "pb_delayed_confirm_enabled": False,
            "pb_open_scored_enabled": False,
        },
    )

    result = IARICPullbackIntradayHybridEngine(config, replay, collect_diagnostics=True).run()

    assert len(result.trades) == 0
    assert result.candidate_ledger is not None
    assert result.candidate_ledger[replay.trade_date][0]["disposition"] in {"watching_expired", "entry_window_expired", "no_route", "no_intraday_setup"}


def test_daily_signal_score_prefers_multifactor_quality_over_rsi_only():
    settings = StrategySettings(pb_daily_signal_family="balanced_v1", pb_rsi_entry=20.0)

    deep_rsi = _daily_signal_bundle(
        settings=settings,
        regime_tier="B",
        item=SimpleNamespace(daily_rank=0.25),
        entry_rsi=2.0,
        gap_pct=1.1,
        sma_dist_pct=14.0,
        cdd=7,
        flow_negative=True,
        sector_count=5,
        total_candidates=6,
        effective_min_candidates_day=6,
    )
    balanced = _daily_signal_bundle(
        settings=settings,
        regime_tier="B",
        item=SimpleNamespace(daily_rank=0.85),
        entry_rsi=8.0,
        gap_pct=-0.8,
        sma_dist_pct=5.0,
        cdd=2,
        flow_negative=False,
        sector_count=1,
        total_candidates=6,
        effective_min_candidates_day=6,
    )

    assert deep_rsi["rsi"] > balanced["rsi"]
    assert balanced["score"] > deep_rsi["score"]
