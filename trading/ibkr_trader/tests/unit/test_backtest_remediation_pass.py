from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from backtests.shared.diagnostics.snapshot import build_group_snapshot
from backtests.swing.analysis.optimized_baseline import (
    load_phase_mutation_source,
    summarize_optimizer_reference,
)
from backtests.swing.config import BacktestConfig, SlippageConfig
from backtests.swing.engine.backtest_engine import BacktestEngine
from backtests.swing.engine.sim_broker import FillResult, FillStatus, OrderSide, OrderType, SimOrder
from backtests.momentum.analysis.downturn_diagnostics import DownturnMetrics
from backtests.momentum.auto.downturn.plugin import _metrics_from_dict
from backtests.momentum.auto.nqdtc.plugin import PHASE_HARD_REJECTS, PHASE_WEIGHTS
from backtests.momentum.auto.nqdtc.phase_diagnostics import generate_phase_diagnostics as generate_nqdtc_phase_diagnostics
from backtests.momentum.auto.nqdtc.scoring import NQDTCMetrics, composite_score as nqdtc_composite_score, extract_nqdtc_metrics
from backtests.momentum.auto.vdubus.phase_diagnostics import generate_phase_diagnostics as generate_vdubus_phase_diagnostics
from backtests.momentum.auto.vdubus.scoring import extract_vdubus_metrics
from backtests.swing.analysis.atrss_diagnostics import atrss_entry_type_drilldown
from backtests.swing.analysis.helix_diagnostics import helix_class_drilldown
from backtests.swing.auto.atrss.scoring import extract_atrss_metrics
from backtests.swing.auto.helix.scoring import extract_helix_metrics
from strategies.swing.atrss.config import SYMBOL_CONFIGS
from strategies.swing.atrss.models import Direction, LegType, PositionBook, PositionLeg


def _make_atrss_engine() -> BacktestEngine:
    symbol, sym_cfg = next(iter(SYMBOL_CONFIGS.items()))
    config = BacktestConfig(
        symbols=[symbol],
        initial_equity=10_000.0,
        fixed_qty=1,
        slippage=SlippageConfig(commission_per_contract=1.0),
    )
    engine = BacktestEngine(symbol, sym_cfg, config, point_value=getattr(sym_cfg, "multiplier", 1.0))
    engine._last_entry_context = {}
    engine._last_entry_type = "PULLBACK"
    return engine


def _make_momentum_trade(
    *,
    pnl_dollars: float,
    commission: float,
    entry_time: datetime,
    exit_time: datetime,
    r_multiple: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        pnl_dollars=pnl_dollars,
        commission=commission,
        r_multiple=r_multiple,
        mfe_r=2.0,
        session="RTH",
        direction=1,
        composite_regime="Trend",
        tp1_hit=False,
        tp2_hit=False,
        entry_time=entry_time,
        exit_time=exit_time,
        exit_reason="TARGET",
        overnight_sessions=1,
        entry_session="RTH",
        sub_window="RTH",
        bars_held_15m=5,
    )


def _make_swing_trade(
    *,
    pnl_dollars: float,
    commission: float,
    r_multiple: float = 1.0,
    entry_type: str = "PULLBACK",
    exit_reason: str = "FLATTEN",
    direction: int = 1,
    setup_class: str = "A",
    regime_at_entry: str = "BULL",
    exit_tier: str = "ALIGNED",
) -> SimpleNamespace:
    return SimpleNamespace(
        pnl_dollars=pnl_dollars,
        commission=commission,
        r_multiple=r_multiple,
        mfe_r=2.0,
        mae_r=-0.5,
        bars_held=5,
        direction=direction,
        entry_type=entry_type,
        exit_reason=exit_reason,
        setup_class=setup_class,
        regime_at_entry=regime_at_entry,
        exit_tier=exit_tier,
        tp1_done=False,
        tp2_done=False,
        runner_active=False,
    )


def test_atrss_full_close_trade_commissions_include_entry_and_exit_sides():
    engine = _make_atrss_engine()
    entry_time = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
    exit_time = entry_time + timedelta(hours=2)
    initial_equity = 10_000.0

    engine.position = PositionBook(
        symbol=engine.symbol,
        direction=Direction.LONG,
        legs=[
            PositionLeg(
                leg_type=LegType.BASE,
                qty=2,
                entry_price=100.0,
                initial_stop=95.0,
                fill_time=entry_time,
                entry_commission=4.0,
            ),
            PositionLeg(
                leg_type=LegType.ADDON_A,
                qty=1,
                entry_price=102.0,
                initial_stop=95.0,
                fill_time=entry_time,
                entry_commission=2.0,
            ),
        ],
        current_stop=95.0,
        entry_time=entry_time,
        bars_held=4,
        mfe=2.0,
    )
    engine.equity = initial_equity - 6.0

    engine._close_position(110.0, exit_time, "FLATTEN", exit_commission=3.0)

    assert [trade.commission for trade in engine.trades] == pytest.approx([6.0, 3.0])
    assert engine.equity - initial_equity == pytest.approx(
        sum(trade.pnl_dollars - trade.commission for trade in engine.trades)
    )


def test_atrss_stop_fill_trade_commission_includes_entry_side():
    engine = _make_atrss_engine()
    entry_time = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
    stop_time = entry_time + timedelta(hours=1)
    initial_equity = 10_000.0

    engine.position = PositionBook(
        symbol=engine.symbol,
        direction=Direction.LONG,
        legs=[
            PositionLeg(
                leg_type=LegType.BASE,
                qty=3,
                entry_price=100.0,
                initial_stop=95.0,
                fill_time=entry_time,
                entry_commission=5.0,
            )
        ],
        current_stop=95.0,
        entry_time=entry_time,
        bars_held=2,
        mfe=0.5,
    )
    engine.equity = initial_equity - 5.0

    fill = FillResult(
        order=SimOrder(
            order_id="SIM-STOP",
            symbol=engine.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            qty=3,
            stop_price=95.0,
        ),
        status=FillStatus.FILLED,
        fill_price=95.0,
        fill_time=stop_time,
        commission=3.0,
    )

    engine._handle_stop_fill(fill, stop_time)

    assert len(engine.trades) == 1
    assert engine.trades[0].commission == pytest.approx(8.0)
    assert engine.equity - initial_equity == pytest.approx(
        engine.trades[0].pnl_dollars - engine.trades[0].commission
    )


def test_atrss_partial_close_allocates_entry_commission_pro_rata():
    engine = _make_atrss_engine()
    entry_time = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
    partial_time = entry_time + timedelta(hours=1)
    initial_equity = 10_000.0

    base_leg = PositionLeg(
        leg_type=LegType.BASE,
        qty=4,
        entry_price=100.0,
        initial_stop=95.0,
        fill_time=entry_time,
        entry_commission=8.0,
    )
    engine.position = PositionBook(
        symbol=engine.symbol,
        direction=Direction.LONG,
        legs=[base_leg],
        current_stop=95.0,
        entry_time=entry_time,
        bars_held=2,
        mfe=1.0,
    )
    engine.equity = initial_equity - 8.0

    engine._partial_close_base(
        110.0,
        partial_time,
        frac=0.5,
        reason="EARLY_STALL_PARTIAL",
        exit_commission=2.0,
        override_qty=2,
    )

    assert len(engine.trades) == 1
    assert engine.trades[0].commission == pytest.approx(6.0)
    assert engine.position.base_leg is not None
    assert engine.position.base_leg.qty == 2
    assert engine.position.base_leg.entry_commission == pytest.approx(4.0)
    assert engine.equity - initial_equity == pytest.approx(
        engine.trades[0].pnl_dollars - 8.0 - 2.0
    )


def test_downturn_metric_rename_keeps_legacy_alias_compatibility():
    metrics = DownturnMetrics(correction_pnl_pct=12.5)
    assert metrics.correction_alpha_pct == pytest.approx(12.5)

    metrics.correction_alpha_pct = 7.5
    assert metrics.correction_pnl_pct == pytest.approx(7.5)

    restored = _metrics_from_dict({"total_trades": 3, "correction_alpha_pct": 2.25})
    assert restored.correction_pnl_pct == pytest.approx(2.25)


def test_fee_net_profit_factor_is_used_in_nqdtc_and_vdubus_scorers():
    t0 = datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc)
    trades = [
        _make_momentum_trade(
            pnl_dollars=100.0,
            commission=90.0,
            entry_time=t0,
            exit_time=t0 + timedelta(hours=1),
            r_multiple=1.0,
        ),
        _make_momentum_trade(
            pnl_dollars=-20.0,
            commission=0.0,
            entry_time=t0 + timedelta(days=1),
            exit_time=t0 + timedelta(days=1, hours=1),
            r_multiple=-0.5,
        ),
    ]
    timestamps = [trades[0].entry_time, trades[-1].exit_time]

    nqdtc = extract_nqdtc_metrics(trades, [1_000.0, 990.0], timestamps, 1_000.0)
    vdubus = extract_vdubus_metrics(trades, [1_000.0, 990.0], timestamps, 1_000.0)

    assert nqdtc.profit_factor == pytest.approx(0.5)
    assert vdubus.profit_factor == pytest.approx(0.5)


def test_nqdtc_and_vdubus_scorers_accept_numpy_timestamps():
    t0 = np.datetime64("2026-04-01T14:30:00")
    trades = [
        _make_momentum_trade(
            pnl_dollars=50.0,
            commission=5.0,
            entry_time=datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 1, 15, 30, tzinfo=timezone.utc),
            r_multiple=1.0,
        ),
        _make_momentum_trade(
            pnl_dollars=-20.0,
            commission=1.0,
            entry_time=datetime(2026, 4, 2, 14, 30, tzinfo=timezone.utc),
            exit_time=datetime(2026, 4, 2, 15, 30, tzinfo=timezone.utc),
            r_multiple=-0.5,
        ),
    ]
    timestamps = np.array([t0, t0 + np.timedelta64(2, "D")])

    nqdtc = extract_nqdtc_metrics(trades, [1_000.0, 1_040.0], timestamps, 1_000.0)
    vdubus = extract_vdubus_metrics(trades, [1_000.0, 1_040.0], timestamps, 1_000.0)

    assert nqdtc.sharpe != 0.0
    assert vdubus.trades_per_month > 0.0


def test_nqdtc_round2_baseline_stays_in_phase_selection_pool():
    metrics = NQDTCMetrics(
        total_trades=97,
        win_rate=0.588,
        profit_factor=1.89,
        max_dd_pct=0.1519,
        net_return_pct=241.2,
        calmar=15.88,
        sharpe=2.0,
        sortino=6.0,
        avg_r=0.429,
        capture_ratio=0.406,
        tp1_hit_rate=0.557,
        tp2_hit_rate=0.0,
        avg_mfe_r=2.356,
        robust_net_return_pct=195.4,
        largest_win_pnl_share=0.19,
        largest_winner_r=6.0,
    )

    for phase, hard_rejects in PHASE_HARD_REJECTS.items():
        score = nqdtc_composite_score(metrics, PHASE_WEIGHTS[phase], hard_rejects=hard_rejects)
        assert not score.rejected, f"phase {phase} rejected baseline: {score.reject_reason}"
        assert score.total > 0.40


def test_helix_scoring_uses_fee_net_profit_factor():
    trades = [
        SimpleNamespace(
            pnl_dollars=100.0,
            commission=90.0,
            r_multiple=2.0,
            mfe_r=3.0,
            exit_reason="TARGET",
            bars_held=12,
            regime_at_entry="BULL",
        ),
        SimpleNamespace(
            pnl_dollars=-20.0,
            commission=0.0,
            r_multiple=-1.0,
            mfe_r=1.0,
            exit_reason="STALE",
            bars_held=8,
            regime_at_entry="BULL",
        ),
    ]
    result = SimpleNamespace(
        symbol_results={"QQQ": SimpleNamespace(trades=trades)},
        combined_equity=np.array([1_000.0, 995.0, 990.0]),
    )

    metrics = extract_helix_metrics(result, 1_000.0)

    assert metrics.profit_factor == pytest.approx(0.5)
    assert metrics.bull_pf == pytest.approx(0.5)
    assert metrics.min_regime_pf == pytest.approx(0.5)


def test_nqdtc_and_vdubus_phase_diagnostics_report_fee_net_pnl():
    trade = SimpleNamespace(
        pnl_dollars=100.0,
        commission=25.0,
        r_multiple=1.0,
        mfe_r=2.0,
        session="RTH",
        direction=1,
        composite_regime="Trend",
        entry_time=datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc),
        exit_time=datetime(2026, 4, 1, 15, 30, tzinfo=timezone.utc),
        exit_reason="TARGET",
        entry_session="RTH",
        bars_held_15m=3,
    )

    nqdtc_metrics = SimpleNamespace(
        total_trades=1,
        win_rate=1.0,
        profit_factor=1.5,
        net_return_pct=5.0,
        robust_net_return_pct=5.0,
        max_dd_pct=0.05,
        calmar=1.0,
        sharpe=1.0,
        sortino=1.0,
        avg_r=1.0,
        capture_ratio=0.5,
        tp1_hit_rate=0.0,
        tp2_hit_rate=0.0,
        avg_winner_r=1.0,
        avg_loser_r=0.0,
        avg_mfe_r=2.0,
        largest_winner_r=1.0,
        largest_win_pnl_share=0.0,
        avg_hold_hours=1.0,
        eth_short_wr=0.0,
        eth_short_trades=0,
        range_regime_pct=0.0,
        burst_trade_pct=0.0,
    )
    vdubus_metrics = SimpleNamespace(
        total_trades=1,
        win_rate=1.0,
        profit_factor=2.0,
        net_return_pct=5.0,
        max_dd_pct=0.05,
        calmar=1.0,
        sharpe=1.0,
        sortino=1.0,
        avg_r=1.0,
        capture_ratio=0.5,
        stale_exit_pct=0.0,
        multi_session_pct=0.0,
        fast_death_pct=0.0,
        avg_winner_r=1.0,
        avg_loser_r=0.0,
        avg_mfe_r=2.0,
        avg_hold_hours=1.0,
        evening_trade_pct=0.0,
        evening_avg_r=0.0,
    )

    nqdtc_text = generate_nqdtc_phase_diagnostics(2, nqdtc_metrics, None, None, [trade], False)
    vdubus_text = generate_vdubus_phase_diagnostics(2, vdubus_metrics, None, None, [trade], False)

    assert "PnL=$+75" in nqdtc_text
    assert "PnL=$+75" in vdubus_text


def test_swing_diagnostic_tables_report_fee_net_pnl():
    trade = _make_swing_trade(pnl_dollars=100.0, commission=25.0)

    atrss_text = atrss_entry_type_drilldown([trade])
    helix_text = helix_class_drilldown([trade])

    assert "+75" in atrss_text
    assert "+75" in helix_text


def test_atrss_scoring_accepts_numeric_combined_timestamps():
    trades = [
        SimpleNamespace(
            entry_time=datetime(2026, 4, 1, 14, 30, tzinfo=timezone.utc),
            pnl_dollars=80.0,
            commission=8.0,
            r_multiple=1.5,
            mfe_r=2.0,
        ),
        SimpleNamespace(
            entry_time=datetime(2026, 4, 3, 14, 30, tzinfo=timezone.utc),
            pnl_dollars=-20.0,
            commission=2.0,
            r_multiple=-0.5,
            mfe_r=0.5,
        ),
    ]
    result = SimpleNamespace(
        symbol_results={"QQQ": SimpleNamespace(trades=trades)},
        combined_equity=np.array([1_000.0, 1_040.0, 1_020.0]),
        combined_timestamps=np.array([0, 86400, 2 * 86400], dtype=np.int64),
    )

    metrics = extract_atrss_metrics(result, 1_000.0)

    assert metrics.calmar > 0.0
    assert metrics.trades_per_month > 0.0


def test_group_snapshot_prefers_clean_positive_buckets_for_strengths():
    trades = [
        SimpleNamespace(session="ETH", pnl_dollars=150.0, commission=0.0, r_multiple=-0.20),
        SimpleNamespace(session="ETH", pnl_dollars=120.0, commission=0.0, r_multiple=-0.10),
        SimpleNamespace(session="RTH", pnl_dollars=80.0, commission=0.0, r_multiple=0.60),
        SimpleNamespace(session="RTH", pnl_dollars=-10.0, commission=0.0, r_multiple=-0.10),
    ]

    snapshot = build_group_snapshot(
        "Snapshot",
        trades,
        [("session", lambda trade: trade.session)],
        min_count=1,
        top_n=2,
    )

    assert "Best session: RTH" in snapshot
    assert "Best session: ETH" not in snapshot


def test_group_snapshot_surfaces_loss_concentration_when_losses_are_clustered():
    trades = [
        SimpleNamespace(bucket="A", pnl_dollars=120.0, commission=0.0, r_multiple=1.0),
        SimpleNamespace(bucket="A", pnl_dollars=90.0, commission=0.0, r_multiple=0.8),
        SimpleNamespace(bucket="A", pnl_dollars=-100.0, commission=0.0, r_multiple=-1.0),
        SimpleNamespace(bucket="B", pnl_dollars=-80.0, commission=0.0, r_multiple=-0.8),
        SimpleNamespace(bucket="C", pnl_dollars=-10.0, commission=0.0, r_multiple=-0.1),
    ]

    snapshot = build_group_snapshot(
        "Snapshot",
        trades,
        [("bucket", lambda trade: trade.bucket)],
        min_count=1,
        top_n=2,
    )

    assert "Losses are concentrated: top 2 losers drive" in snapshot


def test_phase_mutation_source_loads_current_and_phase_specific_views(tmp_path: Path):
    state_path = tmp_path / "phase_state.json"
    state_path.write_text(
        """
{
  "current_phase": 4,
  "cumulative_mutations": {"flags.alpha": true},
  "phase_results": {
    "1": {"final_mutations": {"flags.alpha": false}, "final_score": 0.61},
    "4": {
      "final_mutations": {"flags.alpha": true},
      "final_score": 0.82,
      "final_metrics": {"profit_factor": 2.4, "net_return_pct": 18.5, "max_dd_pct": 0.04, "total_trades": 123}
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    current = load_phase_mutation_source(state_path, "current")
    phase1 = load_phase_mutation_source(state_path, "1")

    assert current.mutations == {"flags.alpha": True}
    assert current.phase_label == "CURRENT OPTIMIZED BASELINE"
    assert current.optimizer_reference["final_score"] == pytest.approx(0.82)
    assert phase1.mutations == {"flags.alpha": False}
    assert phase1.phase_label == "PHASE 1 FINAL BASELINE"


def test_optimizer_reference_summary_labels_independent_metrics():
    lines = summarize_optimizer_reference(
        {
            "final_score": 0.8123,
            "final_metrics": {
                "total_trades": 256,
                "profit_factor": 6.98,
                "net_return_pct": 29.6,
                "max_dd_pct": 0.034,
            },
        }
    )

    text = "\n".join(lines)
    assert "independent fast-path" in text
    assert "Score: 0.8123" in text
    assert "Trades: 256" in text
    assert "Profit factor: 6.98" in text
