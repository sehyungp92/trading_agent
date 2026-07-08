from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import pytest

from backtests.stock.analysis.alcb_diagnostics import _entry_bar_number
from backtests.stock.analysis.alcb_qe_replacement import _time_bucket
from backtests.stock.analysis.alcb_shadow_tracker import ALCBShadowTracker
from backtests.stock.auto.greedy_optimize import run_greedy
from backtests.stock.auto.scoring import CompositeScore
from backtests.stock.config_alcb import ALCBAblationFlags, ALCBBacktestConfig
from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
from backtests.stock.engine.research_replay import ResearchReplayEngine
from backtests.stock.models import Direction as BTDirection, TradeRecord
from strategies.stock.alcb.config import StrategySettings
from strategies.stock.alcb.models import (
    Bar,
    CandidateArtifact,
    CandidateItem,
    Campaign,
    RegimeSnapshot,
    ResearchDailyBar,
)


def _make_daily_bars(trade_date: date, n: int = 25) -> list[ResearchDailyBar]:
    bars: list[ResearchDailyBar] = []
    current = trade_date - timedelta(days=n)
    while len(bars) < n:
        if current.weekday() < 5:
            close = 10.0 + 0.02 * len(bars)
            bars.append(
                ResearchDailyBar(
                    trade_date=current,
                    open=close - 0.1,
                    high=11.0 if len(bars) == n - 1 else close + 0.25,
                    low=close - 0.25,
                    close=close,
                    volume=1_000_000,
                )
            )
        current += timedelta(days=1)
    return bars


def _make_candidate(trade_date: date) -> CandidateItem:
    daily_bars = _make_daily_bars(trade_date)
    return CandidateItem(
        symbol="AAA",
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        point_value=1.0,
        sector="Technology",
        adv20_usd=50_000_000.0,
        median_spread_pct=0.001,
        selection_score=8,
        selection_detail={"rs": 3},
        stock_regime="BULL",
        market_regime="BULL",
        sector_regime="BULL",
        daily_trend_sign=1,
        relative_strength_percentile=0.95,
        accumulation_score=0.8,
        ttm_squeeze_bonus=0,
        average_30m_volume=60_000.0,
        median_30m_volume=60_000.0,
        tradable_flag=True,
        direction_bias="LONG",
        price=daily_bars[-1].close,
        earnings_risk_flag=False,
        campaign=Campaign(symbol="AAA"),
        daily_bars=daily_bars,
        bars_30m=[],
    )


def _make_artifact(trade_date: date) -> CandidateArtifact:
    item = _make_candidate(trade_date)
    regime = RegimeSnapshot(
        score=0.9,
        tier="A",
        risk_multiplier=1.0,
        price_ok=True,
        breadth_ok=True,
        vol_ok=True,
        credit_ok=True,
        market_regime="BULL",
    )
    return CandidateArtifact(
        trade_date=trade_date,
        generated_at=datetime.combine(trade_date, time(0, 0), tzinfo=timezone.utc),
        regime=regime,
        items=[item],
        tradable=[item],
        overflow=[],
        long_candidates=[item],
        short_candidates=[],
        market_wide_institutional_selling=False,
    )


def _make_bar_series(trade_date: date, specs: list[tuple[float, float, float, float, float]]) -> list[Bar]:
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


class _FakeReplay:
    def __init__(self, trade_date: date, bars: list[Bar]) -> None:
        self.trade_date = trade_date
        self._artifact = _make_artifact(trade_date)
        self._bars = {"AAA": list(bars)}

    def alcb_selection_for_date(self, trade_date: date, settings=None, *, as_of_date=None):
        return self._artifact

    def get_5m_bar_objects_for_date(self, symbol: str, trade_date: date):
        if trade_date != self.trade_date:
            return []
        return list(self._bars.get(symbol, []))


def _make_config(trade_date: date, *, eod_flatten_time: time) -> ALCBBacktestConfig:
    ablation = ALCBAblationFlags(
        use_regime_gate=False,
        use_sector_limit=False,
        use_heat_cap=False,
        use_long_only=False,
        use_rvol_filter=False,
        use_cpr_filter=False,
        use_avwap_filter=False,
        use_momentum_score_gate=False,
        use_flow_reversal_exit=False,
        use_carry_logic=False,
        use_partial_takes=False,
    )
    return ALCBBacktestConfig(
        start_date=trade_date.isoformat(),
        end_date=trade_date.isoformat(),
        ablation=ablation,
        param_overrides={
            "opening_range_bars": 2,
            "entry_window_start": time(9, 35),
            "entry_window_end": time(10, 5),
            "rvol_threshold": 0.0,
            "cpr_threshold": 0.0,
            "momentum_score_min": 0,
            "combined_breakout_score_min": 0,
            "or_breakout_score_min": 0,
            "late_entry_score_min": 0,
            "eod_flatten_time": eod_flatten_time,
            "max_positions": 5,
            "max_positions_per_sector": 5,
            "intraday_leverage": 4.0,
        },
    )


def test_alcb_selection_for_date_uses_previous_close_and_settings_aware_cache(monkeypatch, tmp_path):
    replay = ResearchReplayEngine(data_dir=tmp_path)
    trade_date = date(2026, 2, 3)
    prev_date = date(2026, 2, 2)
    calls: list[tuple[date, date, float, float]] = []

    monkeypatch.setattr(replay, "get_prev_trading_date", lambda d: prev_date)

    def fake_build(trade_date_arg, *, min_price=None, min_adv_usd=None, as_of_date=None):
        calls.append((trade_date_arg, as_of_date, min_price, min_adv_usd))
        return SimpleNamespace(trade_date=trade_date_arg)

    monkeypatch.setattr(replay, "build_alcb_snapshot", fake_build)
    monkeypatch.setattr(replay, "run_alcb_selection", lambda snapshot, settings=None: _make_artifact(snapshot.trade_date))

    settings_a = StrategySettings(min_price=15.0, min_adv_usd=20_000_000.0)
    settings_b = StrategySettings(min_price=20.0, min_adv_usd=20_000_000.0)

    replay.alcb_selection_for_date(trade_date, settings_a)
    replay.alcb_selection_for_date(trade_date, settings_a)
    replay.alcb_selection_for_date(trade_date, settings_b)

    assert calls == [
        (trade_date, prev_date, 15.0, 20_000_000.0),
        (trade_date, prev_date, 20.0, 20_000_000.0),
    ]


def test_intraday_engine_fills_on_next_bar_open_not_signal_bar_close():
    trade_date = date(2026, 2, 3)
    bars = _make_bar_series(
        trade_date,
        [
            (10.00, 10.10, 9.90, 10.00, 1_000),
            (10.00, 10.10, 9.95, 10.00, 1_000),
            (10.00, 10.60, 9.95, 10.50, 2_000),
            (10.55, 10.70, 10.45, 10.60, 1_500),
        ],
    )
    replay = _FakeReplay(trade_date, bars)
    engine = ALCBIntradayEngine(_make_config(trade_date, eod_flatten_time=time(9, 45)), replay)

    result = engine.run()

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.signal_time == bars[2].end_time
    assert trade.entry_time == bars[3].start_time
    assert trade.fill_time == bars[3].start_time
    assert trade.signal_bar_index == 2
    assert trade.fill_bar_index == 3
    assert trade.entry_price != pytest.approx(bars[2].close)


def test_intraday_engine_skips_last_bar_signal_without_next_fill_bar():
    trade_date = date(2026, 2, 4)
    bars = _make_bar_series(
        trade_date,
        [
            (10.00, 10.10, 9.90, 10.00, 1_000),
            (10.00, 10.10, 9.95, 10.00, 1_000),
            (10.00, 10.60, 9.95, 10.50, 2_000),
        ],
    )
    replay = _FakeReplay(trade_date, bars)
    engine = ALCBIntradayEngine(_make_config(trade_date, eod_flatten_time=time(10, 30)), replay)

    result = engine.run()

    assert result.trades == []


def test_intraday_engine_requires_rearm_before_same_day_reentry_and_keeps_net_accounting_consistent():
    trade_date = date(2026, 2, 5)
    bars = _make_bar_series(
        trade_date,
        [
            (10.00, 10.10, 9.80, 9.90, 1_000),
            (9.90, 10.10, 9.70, 10.00, 1_000),
            (10.00, 10.70, 9.90, 10.60, 2_000),
            (10.65, 10.70, 10.15, 10.50, 1_800),
            (10.50, 10.80, 10.40, 10.40, 1_600),
            (10.40, 10.45, 9.70, 9.85, 1_700),
            (9.90, 10.50, 9.85, 10.30, 2_100),
            (10.35, 10.40, 10.20, 10.25, 1_500),
        ],
    )
    replay = _FakeReplay(trade_date, bars)
    engine = ALCBIntradayEngine(_make_config(trade_date, eod_flatten_time=time(10, 5)), replay)

    result = engine.run()

    assert len(result.trades) == 2
    first, second = result.trades
    assert first.exit_reason == "CLOSE_STOP"
    assert second.exit_reason == "EOD_FLATTEN"
    assert first.signal_bar_index == 2
    assert second.signal_bar_index == 6
    assert first.reentry_sequence == 0
    assert second.reentry_sequence == 1
    assert second.signal_time == bars[6].end_time
    net_pnl = sum(t.pnl_net for t in result.trades)
    assert result.equity_curve[-1] - result.equity_curve[0] == pytest.approx(net_pnl)


def test_entry_bar_and_qe_time_bucket_use_eastern_time():
    trade = TradeRecord(
        strategy="alcb",
        symbol="AAA",
        direction=BTDirection.LONG,
        entry_time=datetime(2026, 2, 6, 14, 35, tzinfo=timezone.utc),
        exit_time=datetime(2026, 2, 6, 15, 5, tzinfo=timezone.utc),
        entry_price=10.0,
        exit_price=10.1,
        quantity=100,
        pnl=10.0,
        r_multiple=0.2,
        risk_per_share=0.5,
        commission=1.0,
        slippage=0.5,
    )

    assert _entry_bar_number(trade) == 2
    assert _time_bucket(trade.entry_time) == "09:30"


def test_shadow_funnel_report_separates_pass_and_rejection_counts():
    tracker = ALCBShadowTracker()
    tracker.record_funnel("evaluated")
    tracker.record_funnel("entry_signal")
    tracker.record_funnel("entered")
    tracker.record_funnel("avwap_filter")

    report = tracker.funnel_report()

    assert "Cumulative Pass Counts" in report
    assert "Rejection Counts By Gate" in report
    assert "entered" in report
    assert "avwap_filter" in report


def test_alcb_greedy_optimizer_prefers_frequency_within_two_percent_of_best_expectancy(monkeypatch):
    def fake_score_config(replay, mutations, initial_equity, cfg_kwargs):
        case = mutations.get("case", "base")
        mapping = {
            "base": (50.0, 80.0, 5.0, 0.30),
            "best_exp": (120.0, 100.0, 8.0, 0.20),
            "near_best_high_freq": (110.0, 99.0, 20.0, 0.10),
            "bad": (-10.0, -5.0, 30.0, 0.90),
        }
        net_profit, expectancy_dollar, trades_per_month, total = mapping[case]
        score = CompositeScore(0, 0, 0, 0, total=total, rejected=net_profit <= 0 or expectancy_dollar <= 0)
        metrics = SimpleNamespace(
            total_trades=max(1, int(trades_per_month * 3)),
            profit_factor=1.5,
            max_drawdown_pct=0.05,
            net_profit=net_profit,
            expectancy_dollar=expectancy_dollar,
            trades_per_month=trades_per_month,
        )
        return score, metrics

    monkeypatch.setattr("backtests.stock.auto.greedy_optimize._score_config", fake_score_config)

    result = run_greedy(
        replay=object(),
        strategy="alcb",
        tier=2,
        base_mutations={},
        candidates=[
            ("best_exp", {"case": "best_exp"}),
            ("near_best_high_freq", {"case": "near_best_high_freq"}),
            ("bad", {"case": "bad"}),
        ],
        verbose=False,
        max_workers=1,
    )

    assert result.kept_features[0] == "near_best_high_freq"


def test_alcb_greedy_optimizer_ignores_stale_checkpoint_without_matching_metadata(monkeypatch, tmp_path):
    def fake_score_config(replay, mutations, initial_equity, cfg_kwargs):
        case = mutations.get("case", "base")
        mapping = {
            "base": (50.0, 80.0, 5.0, 0.30),
            "best_exp": (120.0, 100.0, 8.0, 0.20),
            "near_best_high_freq": (110.0, 99.0, 20.0, 0.10),
            "bad": (-10.0, -5.0, 30.0, 0.90),
        }
        net_profit, expectancy_dollar, trades_per_month, total = mapping[case]
        score = CompositeScore(0, 0, 0, 0, total=total, rejected=net_profit <= 0 or expectancy_dollar <= 0)
        metrics = SimpleNamespace(
            total_trades=max(1, int(trades_per_month * 3)),
            profit_factor=1.5,
            max_drawdown_pct=0.05,
            net_profit=net_profit,
            expectancy_dollar=expectancy_dollar,
            trades_per_month=trades_per_month,
        )
        return score, metrics

    monkeypatch.setattr("backtests.stock.auto.greedy_optimize._score_config", fake_score_config)

    checkpoint_path = tmp_path / "stale_checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "strategy": "alcb",
                "tier": 2,
                "base_mutations": {},
                "baseline_score": 0.30,
                "current_mutations": {"case": "best_exp"},
                "current_score": 0.20,
                "kept_features": ["best_exp"],
                "rounds": [],
                "remaining": [["bad", {"case": "bad"}]],
            }
        ),
        encoding="utf-8",
    )

    result = run_greedy(
        replay=object(),
        strategy="alcb",
        tier=2,
        base_mutations={},
        candidates=[
            ("best_exp", {"case": "best_exp"}),
            ("near_best_high_freq", {"case": "near_best_high_freq"}),
            ("bad", {"case": "bad"}),
        ],
        verbose=False,
        max_workers=1,
        checkpoint_path=checkpoint_path,
    )

    assert result.kept_features[0] == "near_best_high_freq"
