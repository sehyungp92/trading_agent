"""Verification script for stock backtest framework components.

Tests models, configs, SimBroker, portfolio engine, metrics, and reports
using synthetic data — no IBKR download required.

Plan verification steps covered:
- Component integration (all modules instantiate and wire together)
- Portfolio merge (8R directional cap, symbol collision, heat cap)
- Metrics computation (Sharpe, drawdown, profit factor)
- Report generation (format_summary, breakdowns)

Run: python -m backtests.stock.tests.verify_components
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta

import numpy as np

# ── Helpers ──────────────────────────────────────────────────────────
_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {label}")
    else:
        _failed += 1
        print(f"  [FAIL] {label}  {detail}")


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ── 1. Models ────────────────────────────────────────────────────────
def test_models():
    section("1. Models — TradeRecord + Direction")
    from backtests.stock.models import Direction, TradeRecord

    t = TradeRecord(
        strategy="ALCB",
        symbol="AAPL",
        direction=Direction.LONG,
        entry_time=datetime(2024, 6, 1, 10, 0),
        exit_time=datetime(2024, 6, 5, 15, 30),
        entry_price=190.0,   # already slipped
        exit_price=195.0,    # already slipped
        quantity=100,
        pnl=500.0,           # (195 - 190) * 100 = 500
        r_multiple=1.25,
        risk_per_share=4.0,
        commission=1.0,
        slippage=0.50,       # informational only
        entry_type="A",
        exit_reason="TP1",
        sector="Technology",
        regime_tier="A",
        hold_bars=40,
        max_favorable=600.0,
        max_adverse=-100.0,
    )
    # pnl_net should be pnl - commission (NOT minus slippage — already baked in)
    check("pnl_net = pnl - commission", abs(t.pnl_net - 499.0) < 0.01,
          f"got {t.pnl_net}")
    check("is_winner True for positive pnl", t.is_winner is True)
    check("hold_hours > 0", t.hold_hours > 0, f"got {t.hold_hours}")
    check("Direction.LONG.value == 1", Direction.LONG.value == 1)
    check("Direction.SHORT.value == -1", Direction.SHORT.value == -1)

    # Losing trade
    t2 = TradeRecord(
        strategy="IARIC", symbol="TSLA", direction=Direction.SHORT,
        entry_time=datetime(2024, 7, 1, 10, 0),
        exit_time=datetime(2024, 7, 1, 15, 0),
        entry_price=250.0, exit_price=255.0,
        quantity=50, pnl=-250.0, r_multiple=-0.8,
        risk_per_share=6.25, commission=0.50, slippage=0.30,
        entry_type="FSM", exit_reason="TIME_STOP",
        sector="Consumer Discretionary", regime_tier="B",
        hold_bars=6, max_favorable=50.0, max_adverse=-280.0,
    )
    check("is_winner False for negative pnl", t2.is_winner is False)
    check("SHORT pnl_net correct", abs(t2.pnl_net - (-250.50)) < 0.01,
          f"got {t2.pnl_net}")


# ── 2. Configs ───────────────────────────────────────────────────────
def test_configs():
    section("2. Configs — ALCB, IARIC, Portfolio")
    from backtests.stock.config import SlippageConfig, UniverseConfig
    from backtests.stock.config_alcb import ALCBBacktestConfig
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.config_portfolio import PortfolioBacktestConfig

    slip = SlippageConfig()
    check("SlippageConfig defaults", slip.commission_per_share == 0.005)

    alcb = ALCBBacktestConfig(start_date="2024-01-01", end_date="2025-01-01")
    check("ALCB config instantiates", alcb.initial_equity == 10_000.0)
    check("ALCB max_positions=5", alcb.max_positions == 5)
    check("ALCB heat_cap_r=6.0", alcb.heat_cap_r == 6.0)

    iaric = IARICBacktestConfig(start_date="2024-01-01", end_date="2025-01-01")
    check("IARIC config instantiates", iaric.initial_equity == 10_000.0)

    port = PortfolioBacktestConfig()
    check("Portfolio family_directional_cap_r=8.0", port.family_directional_cap_r == 8.0)
    check("Portfolio symbol_collision_half_size=True", port.symbol_collision_half_size is True)
    check("Portfolio combined_heat_cap_r=10.0", port.combined_heat_cap_r == 10.0)
    check("Portfolio base_risk_fraction=0.005", port.base_risk_fraction == 0.005)

    alcb_cfg, iaric_cfg = port.build_strategy_configs()
    check("build_strategy_configs returns ALCB", alcb_cfg is not None)
    check("build_strategy_configs returns IARIC", iaric_cfg is not None)
    check("Shared start_date propagates", alcb_cfg.start_date == port.start_date)


# ── 3. SimBroker ─────────────────────────────────────────────────────
def test_sim_broker():
    section("3. SimBroker — Order execution + slippage")
    from backtests.stock.config import SlippageConfig
    from backtests.stock.engine.sim_broker import (
        FillResult,
        OrderSide,
        OrderType,
        SimBroker,
        SimOrder,
    )

    slip = SlippageConfig(commission_per_share=0.005, slip_bps_normal=5.0)
    broker = SimBroker(slippage_config=slip)

    # Market buy order
    order = SimOrder(
        order_id="T-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=100,
    )
    broker.submit_order(order)
    check("Order submitted", len(broker.pending_orders) == 1)

    # Process bar — market order should fill at open + slippage
    bar_time = datetime(2024, 6, 1, 10, 0)
    fills = broker.process_bar("AAPL", bar_time, 190.0, 192.0, 189.0, 191.0)
    check("Market order filled", len(fills) == 1)
    if fills:
        f = fills[0]
        check("Fill price >= bar_open (buy slippage)", f.fill_price >= 190.0,
              f"got {f.fill_price}")
        check("Commission computed", f.commission > 0, f"got {f.commission}")
        check("Fill qty=100", f.order.qty == 100)

    # Stop order that doesn't trigger
    stop_order = SimOrder(
        order_id="T-2",
        symbol="MSFT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        qty=50,
        stop_price=380.0,  # stop below current price
    )
    broker.submit_order(stop_order)
    bar_time2 = datetime(2024, 6, 1, 10, 30)
    fills2 = broker.process_bar("MSFT", bar_time2, 390.0, 395.0, 385.0, 392.0)
    check("Stop not triggered (low > stop)", len(fills2) == 0)

    # Bar that triggers the stop
    bar_time3 = datetime(2024, 6, 1, 11, 0)
    fills3 = broker.process_bar("MSFT", bar_time3, 382.0, 383.0, 378.0, 379.0)
    check("Stop triggered (low < stop)", len(fills3) == 1)


# ── 4. Portfolio Engine ──────────────────────────────────────────────
def test_portfolio_engine():
    section("4. Portfolio Engine — Merge + family rules")
    from backtests.stock.config_portfolio import PortfolioBacktestConfig
    from backtests.stock.engine.portfolio_engine import StockPortfolioEngine
    from backtests.stock.models import Direction, TradeRecord

    cfg = PortfolioBacktestConfig(
        initial_equity=10_000.0,
        family_directional_cap_r=8.0,
        symbol_collision_half_size=True,
        combined_heat_cap_r=10.0,
        base_risk_fraction=0.005,  # 1R = $500
    )
    engine = StockPortfolioEngine(cfg)

    base_time = datetime(2024, 6, 1, 10, 0)

    def make_trade(strategy, symbol, direction, entry_time, exit_time,
                   pnl, risk_per_share=5.0, quantity=100):
        return TradeRecord(
            strategy=strategy, symbol=symbol, direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=100.0, exit_price=100.0 + pnl / quantity,
            quantity=quantity, pnl=pnl,
            r_multiple=pnl / (risk_per_share * quantity),
            risk_per_share=risk_per_share, commission=0.50, slippage=0.25,
            entry_type="A", exit_reason="TP1",
            sector="Technology", regime_tier="A",
            hold_bars=10, max_favorable=abs(pnl), max_adverse=-50.0,
        )

    # Test 1: Basic merge — 2 non-overlapping trades accepted
    alcb_trades = [
        make_trade("ALCB", "AAPL", Direction.LONG,
                    base_time, base_time + timedelta(days=3), pnl=200.0),
    ]
    iaric_trades = [
        make_trade("IARIC", "MSFT", Direction.LONG,
                    base_time + timedelta(days=5),
                    base_time + timedelta(days=6), pnl=150.0),
    ]
    result = engine.run(alcb_trades, iaric_trades)
    check("Both trades accepted", len(result.trades) == 2,
          f"got {len(result.trades)}")
    check("No blocked trades", len(result.blocked_trades) == 0,
          f"got {len(result.blocked_trades)}")

    # Test 2: Directional cap — 8R max same direction
    # Each trade: risk = 5.0 * 100 = $500 = 1R. So 9 trades = 9R > 8R cap.
    many_alcb = []
    many_iaric = []
    for i in range(5):
        many_alcb.append(make_trade(
            "ALCB", f"SYM{i}", Direction.LONG,
            base_time + timedelta(hours=i),
            base_time + timedelta(days=30),  # all stay open
            pnl=100.0,
        ))
    for i in range(5):
        many_iaric.append(make_trade(
            "IARIC", f"SYM{i+5}", Direction.LONG,
            base_time + timedelta(hours=i+5),
            base_time + timedelta(days=30),
            pnl=100.0,
        ))
    result2 = engine.run(many_alcb, many_iaric)
    check("Directional cap blocks some trades",
          len(result2.blocked_trades) > 0,
          f"accepted={len(result2.trades)}, blocked={len(result2.blocked_trades)}")
    # 1R each, cap at 8R → max 8 accepted
    check("Max 8 trades accepted (8R cap)",
          len(result2.trades) <= 8,
          f"got {len(result2.trades)}")

    # Test 3: Symbol collision — same ticker in both strategies → half size
    overlap_alcb = [
        make_trade("ALCB", "AAPL", Direction.LONG,
                    base_time, base_time + timedelta(days=10), pnl=500.0,
                    quantity=200),
    ]
    overlap_iaric = [
        make_trade("IARIC", "AAPL", Direction.LONG,
                    base_time + timedelta(hours=1),  # after ALCB entry, ALCB still open
                    base_time + timedelta(days=2), pnl=300.0,
                    quantity=200),
    ]
    result3 = engine.run(overlap_alcb, overlap_iaric)
    if len(result3.trades) == 2:
        iaric_trade = [t for t in result3.trades if t.strategy == "IARIC"][0]
        check("Symbol collision halved IARIC qty",
              iaric_trade.quantity == 100,
              f"got {iaric_trade.quantity}")
        check("collision_halved in metadata",
              iaric_trade.metadata.get("collision_halved") is True)
    else:
        check("Both trades accepted for collision test",
              False, f"got {len(result3.trades)} trades")

    # Test 4: Combined heat cap
    # Heat cap = 10R = $5000. Each trade risk = $500 = 1R.
    # With 10 simultaneous trades, the 11th should be blocked.
    heat_alcb = []
    for i in range(6):
        heat_alcb.append(make_trade(
            "ALCB", f"HA{i}", Direction.LONG,
            base_time + timedelta(hours=i),
            base_time + timedelta(days=30),
            pnl=50.0,
        ))
    heat_iaric = []
    for i in range(6):
        heat_iaric.append(make_trade(
            "IARIC", f"HI{i}", Direction.LONG,
            base_time + timedelta(hours=i+6),
            base_time + timedelta(days=30),
            pnl=50.0,
        ))
    result4 = engine.run(heat_alcb, heat_iaric)
    # Directional cap (8R) will block before heat cap (10R) kicks in
    check("Heat or directional cap blocks excess",
          len(result4.blocked_trades) > 0,
          f"accepted={len(result4.trades)}, blocked={len(result4.blocked_trades)}")


# ── 5. Metrics ───────────────────────────────────────────────────────
def test_metrics():
    section("5. Metrics — PerformanceMetrics computation")
    from backtests.stock.analysis.metrics import PerformanceMetrics, compute_metrics

    # 10 trades: 6 winners, 4 losers
    pnls = np.array([200, 300, -150, 100, -200, 250, -100, 400, 150, -180], dtype=float)
    risks = np.array([400] * 10, dtype=float)
    hold_hours = np.array([24, 48, 12, 36, 8, 72, 16, 96, 24, 6], dtype=float)
    commissions = np.array([1.0] * 10, dtype=float)

    # Simple equity curve
    initial = 100_000.0
    equity = [initial]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity_curve = np.array(equity)

    # Timestamps (daily)
    base = datetime(2024, 1, 1)
    timestamps = np.array([
        np.datetime64(base + timedelta(days=i)) for i in range(len(equity))
    ])

    m = compute_metrics(
        trade_pnls=pnls,
        trade_risks=risks,
        trade_hold_hours=hold_hours,
        trade_commissions=commissions,
        equity_curve=equity_curve,
        timestamps=timestamps,
        initial_equity=initial,
    )

    check("total_trades = 10", m.total_trades == 10)
    check("winning_trades = 6", m.winning_trades == 6)
    check("losing_trades = 4", m.losing_trades == 4)
    check("win_rate = 0.6", abs(m.win_rate - 0.6) < 0.01, f"got {m.win_rate}")
    check("net_profit = sum(pnls)", abs(m.net_profit - 770.0) < 0.01,
          f"got {m.net_profit}")
    check("gross_profit > 0", m.gross_profit > 0)
    check("gross_loss < 0", m.gross_loss < 0)
    check("profit_factor > 1", m.profit_factor > 1.0, f"got {m.profit_factor}")
    check("max_drawdown_pct >= 0 (positive convention)", m.max_drawdown_pct >= 0)
    check("sharpe computed", not np.isnan(m.sharpe), f"got {m.sharpe}")
    check("expectancy > 0", m.expectancy > 0, f"got {m.expectancy}")
    check("total_commissions = 10", abs(m.total_commissions - 10.0) < 0.01)

    # Empty metrics
    m0 = PerformanceMetrics()
    check("Empty metrics total_trades=0", m0.total_trades == 0)
    check("Empty metrics sharpe=0", m0.sharpe == 0.0)


# ── 6. Reports ───────────────────────────────────────────────────────
def test_reports():
    section("6. Reports — format_summary + breakdowns")
    from backtests.stock.analysis.reports import (
        compute_and_format,
        entry_type_breakdown,
        exit_reason_breakdown,
        format_summary,
        full_report,
        regime_breakdown,
        sector_breakdown,
    )
    from backtests.stock.models import Direction, TradeRecord

    trades = []
    base = datetime(2024, 3, 1, 10, 0)
    for i in range(20):
        d = Direction.LONG if i % 3 != 0 else Direction.SHORT
        pnl = 100.0 * (1 if i % 2 == 0 else -0.5)
        trades.append(TradeRecord(
            strategy="ALCB" if i < 10 else "IARIC",
            symbol=f"SYM{i % 5}",
            direction=d,
            entry_time=base + timedelta(days=i),
            exit_time=base + timedelta(days=i, hours=6),
            entry_price=100.0, exit_price=100.0 + pnl / 50,
            quantity=50, pnl=pnl, r_multiple=pnl / 250.0,
            risk_per_share=5.0, commission=0.25, slippage=0.10,
            entry_type="A" if i < 7 else "B" if i < 14 else "C",
            exit_reason="TP1" if pnl > 0 else "STOP",
            sector=["Technology", "Healthcare", "Financials"][i % 3],
            regime_tier=["A", "B"][i % 2],
            hold_bars=6, max_favorable=150.0, max_adverse=-80.0,
        ))

    initial = 100_000.0
    equity = [initial]
    for t in trades:
        equity.append(equity[-1] + t.pnl_net)
    eq_curve = np.array(equity)
    ts = np.array([
        np.datetime64(t.entry_time.replace(tzinfo=None)) for t in trades
    ])

    # compute_and_format
    metrics, summary = compute_and_format(trades, eq_curve, ts, initial, "Test Report")
    check("Summary contains title", "Test Report" in summary)
    check("Summary contains trades count", "20" in summary)
    check("Metrics object valid", metrics.total_trades == 20)

    # Breakdowns
    regime = regime_breakdown(trades)
    check("Regime breakdown has Tier A", "Tier A" in regime)
    check("Regime breakdown has Tier B", "Tier B" in regime)

    sector = sector_breakdown(trades)
    check("Sector breakdown has Technology", "Technology" in sector)

    entry = entry_type_breakdown(trades)
    check("Entry type has A", "A" in entry)
    check("Entry type has B", "B" in entry)

    exit_r = exit_reason_breakdown(trades)
    check("Exit reason has TP1", "TP1" in exit_r)
    check("Exit reason has STOP", "STOP" in exit_r)

    # Full report
    report = full_report(trades, eq_curve, ts, initial, "ALCB")
    check("Full report non-empty", len(report) > 100)

    # Empty trades edge case
    _, empty_summary = compute_and_format([], np.array([initial]), np.array([]), initial)
    check("Empty report doesn't crash", "0" in empty_summary)


# ── 7. Research Replay Instantiation ─────────────────────────────────
def test_research_replay_init():
    section("7. Research Replay — Instantiation (no data)")
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    engine = ResearchReplayEngine.__new__(ResearchReplayEngine)
    check("ResearchReplayEngine class exists", engine is not None)
    check("Has build_alcb_snapshot", hasattr(ResearchReplayEngine, "build_alcb_snapshot"))
    check("Has build_iaric_snapshot", hasattr(ResearchReplayEngine, "build_iaric_snapshot"))
    check("Has load_all_data", hasattr(ResearchReplayEngine, "load_all_data"))


# ── 8. Engine Instantiation ──────────────────────────────────────────
def test_engine_instantiation():
    section("8. Engine Instantiation — ALCB engines")
    from backtests.stock.config_alcb import ALCBBacktestConfig
    from backtests.stock.config_iaric import IARICBacktestConfig
    from backtests.stock.engine.research_replay import ResearchReplayEngine

    cfg_alcb = ALCBBacktestConfig(start_date="2024-01-01", end_date="2024-06-01")
    cfg_iaric = IARICBacktestConfig(start_date="2024-01-01", end_date="2024-06-01")

    # Create replay engine (no data loaded, just for instantiation check)
    replay = ResearchReplayEngine()

    # ALCB Intraday
    from backtests.stock.engine.alcb_engine import ALCBIntradayEngine
    e3 = ALCBIntradayEngine(cfg_alcb, replay)
    check("ALCBIntradayEngine instantiates", e3 is not None)
    check("ALCBIntradayEngine has run()", hasattr(e3, "run"))



# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 60)
    print("  Stock Backtest Framework — Component Verification")
    print("#" * 60)

    tests = [
        test_models,
        test_configs,
        test_sim_broker,
        test_portfolio_engine,
        test_metrics,
        test_reports,
        test_research_replay_init,
        test_engine_instantiation,
    ]

    errors = []
    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            global _failed
            _failed += 1
            errors.append((test_fn.__name__, e))
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\n  Errors:")
        for name, err in errors:
            print(f"    {name}: {err}")

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
