"""Signal funnel analysis — instruments every gate in the trend H1 pipeline."""

import sys
sys.path.insert(0, "src")

from collections import defaultdict
from datetime import date
from pathlib import Path

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.runner import run
from crypto_trader.strategy.trend.config import TrendConfig
from crypto_trader.strategy.trend.strategy import TrendStrategy, WARMUP_BARS
from crypto_trader.core.models import Side, TimeFrame, Order, OrderType
from crypto_trader.optimize.config_mutator import apply_mutations

import uuid
import structlog

log = structlog.get_logger()

# Apply optimized mutations
cfg = TrendConfig()
cfg = apply_mutations(cfg, {
    "setup.min_room_r": 2.0,
    "regime.b_min_adx": 8.0,
    "stops.min_stop_atr": 1.5,
    "exits.time_stop_bars": 16,
    "exits.time_stop_action": "exit",
    "risk.risk_pct_b": 0.018,
})

bt_config = BacktestConfig(
    start_date=date(2026, 2, 25),
    end_date=date(2026, 4, 18),
    symbols=["BTC", "ETH", "SOL"],
    warmup_days=60,
)

# Funnel counters
funnel = defaultdict(lambda: defaultdict(int))
setup_sub = defaultdict(lambda: defaultdict(int))
regime_stats = defaultdict(lambda: defaultdict(int))
confirmation_stats = defaultdict(lambda: defaultdict(int))
setup_detail = defaultdict(list)  # near-misses

original_handle_h1 = TrendStrategy._handle_h1


def instrumented_handle_h1(self, bar, sym, ctx):
    from crypto_trader.strategy.trend.strategy import _PositionMeta

    f = funnel[sym]
    f["total_h1_bars"] += 1

    # Update indicators (same as original)
    self._h1_bar_count[sym] += 1
    snap = self._h1_inc[sym].update(bar)
    if snap is not None:
        self._h1_indicators[sym] = snap

    if self._h1_bar_count[sym] < WARMUP_BARS:
        f["blocked_warmup"] += 1
        return

    h1_ind = self._h1_indicators[sym]
    if h1_ind is None:
        f["blocked_no_indicators"] += 1
        return

    f["past_warmup"] += 1

    pos = ctx.broker.get_position(sym)
    if pos is not None:
        f["blocked_in_position"] += 1
        return

    f["past_position_check"] += 1

    h1_bars_list = ctx.bars.get(sym, TimeFrame.H1, count=50)
    if not h1_bars_list:
        f["blocked_no_h1_bars"] += 1
        return

    # Re-entry
    is_reentry = False
    min_conf_override = None
    recent = self._recent_exits.get(sym, {})
    if recent and self._cfg.reentry.enabled:
        bars_since = self._h1_bar_count[sym] - recent.get("bar_idx", 0)
        loss_r = abs(recent.get("loss_r", 0))
        count = self._reentry_count.get(sym, 0)
        if (
            bars_since >= self._cfg.reentry.cooldown_bars
            and loss_r <= self._cfg.reentry.max_loss_r
            and count < self._cfg.reentry.max_reentries
        ):
            is_reentry = True
            min_conf_override = self._cfg.reentry.min_confluences_override
        else:
            f["blocked_reentry_cooldown"] += 1
            return
    elif recent:
        f["blocked_reentry_disabled"] += 1
        return

    f["past_reentry"] += 1

    equity = ctx.broker.get_equity()
    if self._risk_manager.is_session_stopped(equity, bar.timestamp):
        f["blocked_risk_manager"] += 1
        return

    f["past_risk"] += 1

    regime = self._current_regime.get(sym)
    tier = regime.tier if regime else "no_regime"
    regime_stats[sym][tier] += 1
    if regime is None or regime.tier == "none" or regime.direction is None:
        f["blocked_regime"] += 1
        return

    f["past_regime"] += 1
    direction = regime.direction

    sf = self._cfg.symbol_filter
    rule = getattr(sf, f"{sym.lower()}_direction", "both")
    if rule == "disabled":
        f["blocked_symbol_filter"] += 1
        return
    if rule == "long_only" and direction == Side.SHORT:
        f["blocked_symbol_filter"] += 1
        return
    if rule == "short_only" and direction == Side.LONG:
        f["blocked_symbol_filter"] += 1
        return

    f["past_symbol_filter"] += 1

    # --- Detailed setup sub-gate analysis ---
    sd = self._setup_detector
    sd_cfg = sd._cfg
    atr = h1_ind.atr

    ss = setup_sub[sym]
    ss["total_setup_calls"] += 1

    if atr <= 0:
        ss["blocked_atr_zero"] += 1
        f["blocked_setup"] += 1
        return

    current_bar = h1_bars_list[-1]
    current_price = current_bar.close

    impulse = sd._find_impulse(h1_bars_list, direction, atr)
    if impulse is None:
        ss["blocked_no_impulse"] += 1
        f["blocked_setup"] += 1
        return

    ss["past_impulse"] += 1
    imp_start, imp_end, imp_atr_move = impulse
    imp_range = abs(imp_end - imp_start)
    if imp_range == 0:
        ss["blocked_imp_range_zero"] += 1
        f["blocked_setup"] += 1
        return

    if direction == Side.LONG:
        pullback_depth = (imp_end - current_price) / imp_range
    else:
        pullback_depth = (current_price - imp_end) / imp_range

    if pullback_depth < 0:
        ss["blocked_pullback_negative"] += 1
        f["blocked_setup"] += 1
        return
    if pullback_depth > sd_cfg.pullback_max_retrace:
        ss["blocked_pullback_too_deep"] += 1
        f["blocked_setup"] += 1
        return

    ss["past_pullback"] += 1

    ema_f = h1_ind.ema_fast
    ema_m = h1_ind.ema_mid
    in_zone = sd._in_ema_zone(current_price, ema_f, ema_m, direction)
    if not in_zone:
        ss["blocked_not_in_ema_zone"] += 1
        f["blocked_setup"] += 1
        return

    ss["past_ema_zone"] += 1

    # Full setup detection for remaining gates (confluences/stop/room/grading)
    d1_ind = self._d1_indicators.get(sym)
    weekly_tracker = self._weekly_trackers.get(sym)
    weekly_high = weekly_tracker.prior_week_high if weekly_tracker else None
    weekly_low = weekly_tracker.prior_week_low if weekly_tracker else None

    setup = sd.detect(
        h1_bars_list, h1_ind, d1_ind, regime, weekly_high, weekly_low, min_conf_override
    )
    if setup is None:
        ss["blocked_post_zone"] += 1
        f["blocked_setup"] += 1
        # Record near-miss details
        stop_level = sd._estimate_stop(h1_bars_list, direction, atr)
        stop_distance = abs(current_price - stop_level)
        if stop_distance > 0:
            if direction == Side.LONG:
                room_r = (imp_end - current_price) / stop_distance
            else:
                room_r = (current_price - imp_end) / stop_distance
            setup_detail[sym].append({
                "ts": str(bar.timestamp)[:16],
                "stage": "post_zone_reject",
                "room_r": round(room_r, 2),
                "pullback": round(pullback_depth, 2),
                "imp_atr": round(imp_atr_move, 2),
                "dir": direction.value,
            })
        return

    f["past_setup"] += 1
    ss["past_all"] += 1

    # Confirmation
    trigger = self._trigger_detector.check(h1_bars_list, setup.direction, h1_ind)
    if trigger is None:
        f["blocked_confirmation"] += 1
        confirmation_stats[sym]["no_trigger"] += 1
        setup_detail[sym].append({
            "ts": str(bar.timestamp)[:16],
            "stage": "no_confirmation",
            "grade": setup.grade.value,
            "confluences": list(setup.confluences),
            "room_r": round(setup.room_r, 2),
            "dir": direction.value,
        })
        return

    f["past_confirmation"] += 1
    confirmation_stats[sym][trigger.pattern] += 1

    weak = {"micro_structure_shift", "shooting_star"}
    if trigger.pattern in weak:
        min_c = self._cfg.setup.min_confluences_for_weak
        if len(setup.confluences) < min_c:
            f["blocked_weak_confluence"] += 1
            return

    f["past_confluence_gate"] += 1
    f["reached_entry"] += 1

    # Record this successful signal for analysis
    setup_detail[sym].append({
        "ts": str(bar.timestamp)[:16],
        "stage": "ENTRY_SIGNAL",
        "grade": setup.grade.value,
        "confluences": list(setup.confluences),
        "room_r": round(setup.room_r, 2),
        "trigger": trigger.pattern,
        "dir": direction.value,
    })

    # Undo bar count / indicator update so original can do it fresh
    self._h1_bar_count[sym] -= 1
    # Call the ORIGINAL _handle_h1 for actual order submission
    original_handle_h1(self, bar, sym, ctx)
    # Count if an order was actually submitted (position meta exists now)
    if sym in self._position_meta and self._position_meta[sym].entry_bar_index == self._h1_bar_count[sym]:
        f["reached_order_submit"] += 1


# Patch
TrendStrategy._handle_h1 = instrumented_handle_h1

# Run
data_dir = Path("data")
result = run(cfg, bt_config, data_dir, strategy_type="trend")

# ─── Report ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SIGNAL FUNNEL ANALYSIS — Trend Strategy M15 (Optimized Config)")
print("=" * 70)

for sym in ["BTC", "ETH", "SOL"]:
    f = funnel[sym]
    pw = max(f["past_warmup"], 1)
    pr = max(f["past_risk"], 1)
    psf = max(f["past_symbol_filter"], 1)
    ps = max(f["past_setup"], 1)

    print()
    print("=" * 50)
    print(f"  {sym} — H1 Signal Funnel")
    print("=" * 50)
    print(f"  Total H1 bars:          {f['total_h1_bars']}")
    print(f"  Blocked warmup:         {f['blocked_warmup']}")
    print(f"  Past warmup:            {f['past_warmup']}")
    blocked_pos = f["blocked_in_position"]
    print(f"  Blocked in position:    {blocked_pos} ({blocked_pos/pw*100:.1f}%)")
    print(f"  Past position check:    {f['past_position_check']}")
    blocked_re = f["blocked_reentry_cooldown"] + f["blocked_reentry_disabled"]
    print(f"  Blocked reentry:        {blocked_re}")
    print(f"  Past reentry:           {f['past_reentry']}")
    print(f"  Blocked risk manager:   {f['blocked_risk_manager']}")
    print(f"  Past risk:              {f['past_risk']}")
    blocked_reg = f["blocked_regime"]
    print(f"  Blocked regime:         {blocked_reg} ({blocked_reg/pr*100:.1f}%)")
    print(f"  Past regime:            {f['past_regime']}")
    print(f"  Blocked symbol filter:  {f['blocked_symbol_filter']}")
    print(f"  Past symbol filter:     {f['past_symbol_filter']}")
    blocked_setup = f["blocked_setup"]
    print(f"  Blocked setup:          {blocked_setup} ({blocked_setup/psf*100:.1f}%)")
    print(f"  Past setup:             {f['past_setup']}")
    print(f"  Blocked confirmation:   {f['blocked_confirmation']}")
    print(f"  Past confirmation:      {f['past_confirmation']}")
    print(f"  Blocked weak conflu:    {f['blocked_weak_confluence']}")
    print(f"  Past confluence gate:   {f['past_confluence_gate']}")
    blocked_ss = f["blocked_stop"] + f["blocked_sizing"]
    print(f"  Blocked stop/sizing:    {blocked_ss}")
    print(f"  Orders submitted:       {f['reached_order_submit']}")

    ss = setup_sub[sym]
    if ss["total_setup_calls"] > 0:
        tc = ss["total_setup_calls"]
        pi = max(ss["past_impulse"], 1)
        pp = max(ss["past_pullback"], 1)
        pe = max(ss["past_ema_zone"], 1)
        print()
        print(f"  --- Setup Sub-Gates ({sym}) ---")
        print(f"  Setup calls:            {tc}")
        print(f"  No impulse:             {ss['blocked_no_impulse']} ({ss['blocked_no_impulse']/tc*100:.1f}%)")
        print(f"  Past impulse:           {ss['past_impulse']} ({ss['past_impulse']/tc*100:.1f}%)")
        print(f"  PB negative (past imp): {ss['blocked_pullback_negative']} ({ss['blocked_pullback_negative']/pi*100:.1f}%)")
        print(f"  PB too deep:            {ss['blocked_pullback_too_deep']} ({ss['blocked_pullback_too_deep']/pi*100:.1f}%)")
        print(f"  Past pullback:          {ss['past_pullback']} ({ss['past_pullback']/pi*100:.1f}%)")
        print(f"  Not in EMA zone:        {ss['blocked_not_in_ema_zone']} ({ss['blocked_not_in_ema_zone']/pp*100:.1f}%)")
        print(f"  Past EMA zone:          {ss['past_ema_zone']} ({ss['past_ema_zone']/pp*100:.1f}%)")
        print(f"  Post-zone rejection:    {ss['blocked_post_zone']} (conflu/stop/room/grading)")
        print(f"  PASSED all setup:       {ss['past_all']}")

    print()
    print(f"  --- Regime Distribution ({sym}) ---")
    for tier, ct in sorted(regime_stats[sym].items()):
        pct = ct / pr * 100
        print(f"    {tier:>10}: {ct:>4} ({pct:.1f}%)")

    if confirmation_stats[sym]:
        print()
        print(f"  --- Confirmation ({sym}) ---")
        for p, ct in sorted(confirmation_stats[sym].items()):
            print(f"    {p:>20}: {ct}")

    if setup_detail[sym]:
        print()
        print(f"  --- Near-Misses ({sym}, first 15) ---")
        for d in setup_detail[sym][:15]:
            print(f"    {d}")

# Aggregate
print()
print("=" * 70)
print("AGGREGATE FUNNEL")
print("=" * 70)
agg = defaultdict(int)
for sym in ["BTC", "ETH", "SOL"]:
    for k, v in funnel[sym].items():
        agg[k] += v

agg_ss = defaultdict(int)
for sym in ["BTC", "ETH", "SOL"]:
    for k, v in setup_sub[sym].items():
        agg_ss[k] += v

t = max(agg["total_h1_bars"], 1)
pw = max(agg["past_warmup"], 1)
pr = max(agg["past_risk"], 1)
psf = max(agg["past_symbol_filter"], 1)
ps = max(agg["past_setup"], 1)

print(f"H1 bars total:       {agg['total_h1_bars']}")
print(f"Past warmup:         {agg['past_warmup']} ({agg['past_warmup']/t*100:.0f}%)")
print(f"Available (no pos):  {agg['past_position_check']} ({agg['past_position_check']/pw*100:.0f}%)")
print(f"Past reentry:        {agg['past_reentry']}")
print(f"Past risk:           {agg['past_risk']}")
print(f"Past regime:         {agg['past_regime']} ({agg['past_regime']/pr*100:.0f}%)")
print(f"Past sym filter:     {agg['past_symbol_filter']}")
print(f"Past setup:          {agg['past_setup']} ({agg['past_setup']/psf*100:.1f}%)")
print(f"Past confirmation:   {agg['past_confirmation']} ({agg['past_confirmation']/ps*100:.1f}%)")
print(f"Orders submitted:    {agg['reached_order_submit']}")

tc = max(agg_ss["total_setup_calls"], 1)
pi = max(agg_ss["past_impulse"], 1)
pp = max(agg_ss["past_pullback"], 1)
print()
print(f"Setup sub-gates (aggregate of {agg_ss['total_setup_calls']} calls):")
print(f"  No impulse:        {agg_ss['blocked_no_impulse']} ({agg_ss['blocked_no_impulse']/tc*100:.0f}%)")
print(f"  Past impulse:      {agg_ss['past_impulse']} ({agg_ss['past_impulse']/tc*100:.0f}%)")
print(f"  PB negative:       {agg_ss['blocked_pullback_negative']} ({agg_ss['blocked_pullback_negative']/pi*100:.0f}%)")
print(f"  PB too deep:       {agg_ss['blocked_pullback_too_deep']} ({agg_ss['blocked_pullback_too_deep']/pi*100:.0f}%)")
print(f"  Past pullback:     {agg_ss['past_pullback']} ({agg_ss['past_pullback']/pi*100:.0f}%)")
print(f"  Not in EMA zone:   {agg_ss['blocked_not_in_ema_zone']} ({agg_ss['blocked_not_in_ema_zone']/pp*100:.0f}%)")
print(f"  Past EMA zone:     {agg_ss['past_ema_zone']} ({agg_ss['past_ema_zone']/pp*100:.0f}%)")
print(f"  Post-zone reject:  {agg_ss['blocked_post_zone']}")
print(f"  PASSED all:        {agg_ss['past_all']}")

pm = result.metrics
print()
print(f"Final: {pm.total_trades} trades, WR {pm.win_rate:.1f}%, PF {pm.profit_factor:.2f}")
print(f"DD {pm.max_drawdown_pct:.2f}%, net {pm.net_return_pct:.2f}%, Sharpe {pm.sharpe_ratio:.2f}")
