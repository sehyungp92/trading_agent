"""Final comparison: baseline vs optimized NQDTC config."""
from __future__ import annotations
import sys, logging
from pathlib import Path

_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_root))
logging.basicConfig(level=logging.WARNING)

from backtests.momentum.config_nqdtc import NQDTCBacktestConfig
from backtests.momentum.engine.nqdtc_engine import NQDTCEngine
from backtests.momentum.auto.nqdtc.scoring import extract_nqdtc_metrics
from backtests.momentum.auto.nqdtc.worker import load_worker_data

data_dir = Path("backtests/momentum/data/raw")
data = load_worker_data("NQ", data_dir)
print("Data loaded.\n")

EQUITY = 10_000.0

# --- Baseline (greedy-only: max_loss_cap + max_stop_width) ---
cfg_base = NQDTCBacktestConfig(initial_equity=EQUITY, data_dir=data_dir, fixed_qty=10)
cfg_base = cfg_base.__class__(
    **{**cfg_base.__dict__, 'flags': cfg_base.flags.__class__(
        **{**cfg_base.flags.__dict__, 'max_loss_cap': True, 'max_stop_width': True}
    )}
)
e1 = NQDTCEngine("MNQ", cfg_base)
r1 = e1.run(**data)
m1 = extract_nqdtc_metrics(r1.trades, r1.equity_curve, r1.timestamps, EQUITY)

# --- Optimized (all 8 mutations) ---
from backtests.momentum.auto.config_mutator import mutate_nqdtc_config

optimized_muts = {
    "flags.max_loss_cap": True,
    "flags.max_stop_width": True,
    "param_overrides.TP1_R": 0.88,
    "param_overrides.TP1_PARTIAL_PCT": 0.75,
    "param_overrides.TP2_R": 3.5,
    "param_overrides.TP2_PARTIAL_PCT": 0.25,
    "param_overrides.SCORE_NORMAL": 0.5,
    "flags.block_eth_shorts": True,
}

cfg_opt = NQDTCBacktestConfig(initial_equity=EQUITY, data_dir=data_dir, fixed_qty=10)
cfg_opt = mutate_nqdtc_config(cfg_opt, optimized_muts)
e2 = NQDTCEngine("MNQ", cfg_opt)
r2 = e2.run(**data)
m2 = extract_nqdtc_metrics(r2.trades, r2.equity_curve, r2.timestamps, EQUITY)

# --- Print comparison ---
def pct_change(old, new):
    if old == 0:
        return "N/A"
    return f"{(new - old) / abs(old) * 100:+.1f}%"

print(f"{'Metric':<22} {'Baseline':>12} {'Optimized':>12} {'Change':>10}")
print("-" * 60)
rows = [
    ("Total trades", f"{m1.total_trades}", f"{m2.total_trades}", pct_change(m1.total_trades, m2.total_trades)),
    ("Win rate", f"{m1.win_rate:.1%}", f"{m2.win_rate:.1%}", pct_change(m1.win_rate, m2.win_rate)),
    ("Profit factor", f"{m1.profit_factor:.2f}", f"{m2.profit_factor:.2f}", pct_change(m1.profit_factor, m2.profit_factor)),
    ("Net return %", f"{m1.net_return_pct:.1f}%", f"{m2.net_return_pct:.1f}%", pct_change(m1.net_return_pct, m2.net_return_pct)),
    ("Max DD %", f"{m1.max_dd_pct:.2%}", f"{m2.max_dd_pct:.2%}", pct_change(m1.max_dd_pct, m2.max_dd_pct)),
    ("Calmar", f"{m1.calmar:.1f}", f"{m2.calmar:.1f}", pct_change(m1.calmar, m2.calmar)),
    ("Sharpe", f"{m1.sharpe:.2f}", f"{m2.sharpe:.2f}", pct_change(m1.sharpe, m2.sharpe)),
    ("Sortino", f"{m1.sortino:.2f}", f"{m2.sortino:.2f}", pct_change(m1.sortino, m2.sortino)),
    ("Avg R", f"{m1.avg_r:.3f}", f"{m2.avg_r:.3f}", pct_change(m1.avg_r, m2.avg_r)),
    ("Capture ratio", f"{m1.capture_ratio:.3f}", f"{m2.capture_ratio:.3f}", pct_change(m1.capture_ratio, m2.capture_ratio)),
    ("TP1 hit rate", f"{m1.tp1_hit_rate:.1%}", f"{m2.tp1_hit_rate:.1%}", pct_change(m1.tp1_hit_rate, m2.tp1_hit_rate)),
    ("TP2 hit rate", f"{m1.tp2_hit_rate:.1%}", f"{m2.tp2_hit_rate:.1%}", "N/A" if m1.tp2_hit_rate == 0 else pct_change(m1.tp2_hit_rate, m2.tp2_hit_rate)),
    ("Burst trade %", f"{m1.burst_trade_pct:.1%}", f"{m2.burst_trade_pct:.1%}", pct_change(m1.burst_trade_pct, m2.burst_trade_pct)),
    ("ETH short trades", f"{m1.eth_short_trades}", f"{m2.eth_short_trades}", pct_change(m1.eth_short_trades, m2.eth_short_trades)),
    ("ETH short WR", f"{m1.eth_short_wr:.1%}", f"{m2.eth_short_wr:.1%}", pct_change(m1.eth_short_wr, m2.eth_short_wr)),
    ("Avg hold hours", f"{m1.avg_hold_hours:.1f}", f"{m2.avg_hold_hours:.1f}", pct_change(m1.avg_hold_hours, m2.avg_hold_hours)),
]
for label, base, opt, chg in rows:
    print(f"{label:<22} {base:>12} {opt:>12} {chg:>10}")

final_eq_1 = r1.equity_curve[-1] if r1.equity_curve else EQUITY
final_eq_2 = r2.equity_curve[-1] if r2.equity_curve else EQUITY
print(f"\nFinal equity: ${final_eq_1:,.2f} -> ${final_eq_2:,.2f} ({pct_change(final_eq_1, final_eq_2)})")
