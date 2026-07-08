# ALCB Backtest Engine - Complete Signal Funnel Analysis

## Overview

The ALCB Tier 2 intraday engine (`research/backtests/stock/engine/alcb_engine.py`, 885 lines) replays 30m bars across ~500 S&P 500 symbols. A potential trade must pass through **15 sequential gates** spanning 4 phases: nightly selection, daily box detection, intraday entry triggers, and portfolio constraints.

---

## Phase 0: Universe Construction (ResearchReplayEngine)

**File**: `research/backtests/stock/engine/research_replay.py`

| # | Gate | Location | Default Threshold |
|---|------|----------|-------------------|
| 0a | **SP500 base universe** | `research_replay.py:294` | ~500 constituents from `SP500_CONSTITUENTS` |
| 0b | **Pre-filter: price** | `research_replay.py:910-911` | `min_price=5.0` |
| 0c | **Pre-filter: ADV** | `research_replay.py:910-911` | `min_adv20_usd=5,000,000` |
| 0d | **Data availability** | `research_replay.py:917+` | Must have daily bars in cache |

---

## Phase 1: Nightly Selection Pipeline (runs once per trading day)

**File**: `strategies/stock/alcb/research.py`
**Entry**: `daily_selection_from_snapshot()` at line 269

### Gate 1 — Market Regime Gate
- **Line**: `research.py:277-287` + `alcb_engine.py:241`
- `compute_market_regime()` returns tier A/B/C
- If `ablation.use_regime_gate=True` AND tier == "C" => **skip entire day** (engine line 241-245)
- Also: selection returns empty `tradable=[]` for tier C (research.py:278)

### Gate 2 — Liquid Common Stock Filter
- **Line**: `research.py:290` => `filter_liquid_common_stocks()` at `research.py:73-92`
- Checks (ALL must pass):
  - `price >= min_price` (default 15.0 in StrategySettings, line 57)
  - `adv20_usd >= min_adv_usd` (default 5,000,000)
  - `median_spread_pct <= max_median_spread_pct`
  - NOT etf, preferred, otc, blacklisted, halted, severe_news
  - NOT within `earnings_block_days` of earnings
  - ADR blocked unless `allow_adrs=True`
  - Biotech blocked unless `allow_biotech=True`

### Gate 3 — Compression Box Detection (Selection Phase)
- **Line**: `research.py:125` (inside `_selection_score_long`) and `research.py:226` (inside `build_candidate_item`)
- Calls `detect_compression_box()` from `signals.py:257-294`
- Only symbols with `box is not None` make it into `all_items` (research.py:300)
- Symbols without a box get score=0 for compression component

### Gate 4 — Breakout Qualification (Selection Phase)
- **Line**: `research.py:233-237` inside `build_candidate_item()`
- Only runs if box exists; calls `qualify_breakout()` at `signals.py:480`
- Sub-gates within `qualify_breakout`:
  - **Structural breakout**: `qualifies_structural_breakout()` (signals.py:423) — today's close must be above `box.high - tolerance` (LONG) or below `box.low + tolerance` (SHORT)
  - **Displacement pass**: `displacement_pass()` (signals.py:435) — `|close - AVWAP| / ATR14` must exceed adaptive quantile threshold (default base_q_disp=0.70)
  - **Breakout reject**: `breakout_reject()` (signals.py:459) — rejects overextended bars (range > 2x ATR14 with bad body/wick ratio and high volume)
  - **Volume confirmation**: (signals.py:496) — if `daily_rvol < breakout_min_rvol_d`, displacement threshold is raised by `breakout_low_vol_disp_premium`
  - **Opposite reentry block**: (signals.py:500) — blocks direction reversal if displacement < 1.10x threshold

### Gate 5 — Selection Ranking & Cutoff
- **Line**: `research.py:302-310`
- Long candidates sorted by (long_score, selection_score, RS percentile) descending
- Short candidates sorted similarly
- Only top `selection_long_count` longs and `selection_short_count` shorts make `tradable` list

---

## Phase 2: Daily Box Detection in Engine (re-done per day)

**File**: `alcb_engine.py:255-290`

### Gate 6 — Engine-Level Box Detection
- **Line**: `alcb_engine.py:257-276`
- For each tradable symbol NOT already in a position:
  - Must have >= 21 daily bars (line 262)
  - `detect_compression_box(prior_bars, settings)` called on bars EXCLUDING today (line 264-265)
  - If box is None => **skip** (line 266-267)

### Gate 7 — Engine-Level Breakout Direction
- **Line**: `alcb_engine.py:268-276`
- Today's close must be > `box.high - tolerance` (LONG) or < `box.low + tolerance` (SHORT)
- `tolerance = box.height * breakout_tolerance_pct` (default 0.0 = **zero tolerance**)
- If close is inside the box => **skip** (line 276: `continue`)

---

## Phase 3: Intraday 30m Bar Entry Triggers

**File**: `alcb_engine.py:494-744` (within bar loop)

### Gate 8 — Campaign State Check
- **Line**: `alcb_engine.py:499-504`
- Campaign must be in `BREAKOUT` or `CONTINUATION` state
- Campaign must have a non-None `breakout` object

### Gate 9 — Entry Window (Time of Day)
- **Line**: `alcb_engine.py:507-511`
- Bar time must be 10:00 ET <= bar_time < 15:30 ET
- First 30 minutes (09:30-10:00) excluded; last 30 minutes (15:30-16:00) excluded

### Gate 10 — Long-Only Ablation Gate
- **Line**: `alcb_engine.py:515-518`
- If `ablation.use_long_only=True` (default **False**), SHORT direction => **skip**

### Gate 11 — Short Conviction Gate
- **Line**: `alcb_engine.py:521-534`
- If `ablation.use_short_conviction_gate=True` (default **False**) AND direction is SHORT:
  - Box tier must be `GOOD` or entry is rejected
  - Rejection logged to shadow tracker as "short_conviction_reject"

### Gate 12 — Entry A: AVWAP Retest / Box Edge Pullback
- **Line**: `alcb_engine.py:542-549`
- **LONG**: `bar.low <= box.high * 1.01` AND `bar.close > box.high` (price dips to box edge then closes above)
- **SHORT**: `bar.high >= box.low * 0.99` AND `bar.close < box.low` (price pokes above box low then closes below)
- This is an *approximation* of the live AVWAP retest — requires price to touch the box boundary zone and close on the right side

### Gate 12b — Entry B: Sweep & Reclaim (optional)
- **Line**: `alcb_engine.py:553-571`
- Only attempted if Entry A did NOT trigger AND `ablation.use_entry_b=True` (default **False**)
- Requires AVWAP anchor; computes AVWAP from 30m bars
- **LONG**: bar.low sweeps below `AVWAP - 0.25*ATR14`, then bar.close > AVWAP
- **SHORT**: bar.high sweeps above `AVWAP + 0.25*ATR14`, then bar.close < AVWAP

### Gate 13 — Weekday Filter
- **Line**: `alcb_engine.py:581-591`
- If `entry_blocked_weekdays` is set (default empty tuple), specific weekdays blocked
- Shadow tracker logs "weekday_blocked"

### Gate 14 — Sector Blocklist
- **Line**: `alcb_engine.py:594-604`
- If `sector_blocked_list` is set (default empty tuple), specific sectors blocked
- Shadow tracker logs "sector_blocked"

### Gate 15 — Stop Distance Sanity
- **Line**: `alcb_engine.py:607-618`
- `|entry_price - stop_price| >= 0.01` required
- `risk_per_share > 0` required (includes cost buffer)

### Gate 16 — Position Size Viability
- **Line**: `alcb_engine.py:646-649`
- `qty = floor(risk_dollars / risk_per_share)` must be >= 1
- With conviction sizing (if enabled), risk_dollars is adjusted by score_mult * tier_mult

### Gate 17 — Intraday Score Gate (T2 only)
- **Line**: `alcb_engine.py:652-661`
- If intraday score exists AND `min_intraday_score > 0` (default 0), score must meet threshold
- Short-specific gate if `use_short_conviction_gate`: `short_min_intraday_score` checked

---

## Phase 4: Portfolio Constraints

### Gate 18 — Max Positions
- **Line**: `alcb_engine.py:689-693`
- `portfolio.occupied_slots() < settings.max_positions` (default 5)
- Shadow tracker logs "max_positions"

### Gate 19 — Sector Limit
- **Line**: `alcb_engine.py:694-697`
- Same-sector positions < `max_positions_per_sector` (default 2)
- Shadow tracker logs "sector_limit"

### Gate 20 — Heat Cap
- **Line**: `alcb_engine.py:699-705`
- `current_heat + (qty * risk_per_share) <= equity * heat_cap_r * base_risk_fraction`
- Default: `heat_cap_r=6.0`, `base_risk_fraction=0.0050` => max heat = 3% of equity
- Shadow tracker logs "heat_cap"

---

## Squeeze Indicator Details

**NOT BBands/Keltner.** The ALCB squeeze is a simpler ratio:

```
squeeze_metric = box_height / ATR50
```

- **File**: `signals.py:267`
- **Timeframe**: Daily bars only
- `box_height = max(highs over L bars) - min(lows over L bars)`
- `ATR50` = 50-period ATR of daily bars
- **Gate**: `squeeze_metric <= max_squeeze_metric` (default 1.10) — box range must be at most 110% of ATR50
- **Tier classification** (signals.py:272-279):
  - Computes rolling squeeze history over last 60 days (default `squeeze_lookback=60`)
  - `GOOD`: squeeze_metric <= p30 of history (tightest 30%)
  - `LOOSE`: squeeze_metric >= p65 of history
  - `NEUTRAL`: between p30 and p65

---

## Box Detection Details

**File**: `signals.py:224-294`

### Box Length Selection (`choose_box_length`, signals.py:224)
- Uses ATR14/ATR50 ratio over last `hysteresis_days=3` days for stability
- `ratio < 0.75` => `box_length_low=8` days (low volatility regime)
- `0.75 <= ratio <= 1.20` => `box_length_mid=12` days (normal)
- `ratio > 1.20` => `box_length_high=18` days (high volatility)
- Majority vote across 3 days for hysteresis

### Box Identification (`detect_compression_box`, signals.py:257)
1. Choose L via `choose_box_length`
2. Take last L daily bars as window
3. `range_high = max(highs)`, `range_low = min(lows)`
4. `box_height = range_high - range_low`
5. `squeeze_metric = box_height / ATR50`
6. `containment = fraction of bars with close inside [range_low, range_high]`
7. **Reject if**: `containment < 0.80` OR `squeeze_metric > 1.10`
8. Classify tier via historical quantiles

---

## Containment Check

- **Line**: `signals.py:268`
- `containment = count(bars where range_low <= close <= range_high) / L`
- Default threshold: `min_containment = 0.80` (80% of bars must close within box)
- This is checked at TWO points:
  1. During nightly selection (`build_candidate_item` => `detect_compression_box`)
  2. During engine daily loop (`alcb_engine.py:265`)

---

## Ablation Flags Impact

| Flag | Default | Effect on Funnel |
|------|---------|-----------------|
| `use_regime_gate` | (from config) | Skips entire Tier-C days (engine:241) |
| `use_long_only` | False | Blocks all SHORT entries (engine:516) |
| `use_short_conviction_gate` | False | Requires GOOD box tier for shorts (engine:521) |
| `use_conviction_sizing` | False | Adjusts risk_dollars by score*tier multiplier (engine:626-644) |
| `use_box_height_targets` | False | Replaces regime-based TP1/TP2 with box-height multiples (engine:672-676) |
| `use_entry_b` | False | Enables sweep-and-reclaim entries after Entry A fails (engine:553) |
| `use_no_runner_mode` | False | Closes 100% at TP1 instead of partial (engine:386) |
| `use_breakeven_acceleration` | False | Accelerates BE shift after TP1 (engine:408) |
| `use_graduated_stale` | False | Replaces binary stale exit with progressive trim (engine:456) |
| `use_stale_exit` | True | Binary stale exit after `stale_exit_days=10` (engine:478) |
| `use_trailing_stop` | True | Runner trailing on 4H ATR (engine:442) |
| `use_partial_tp` | True | TP1/TP2 partial exits enabled |

---

## Shadow Tracker Rejection Logging

**File**: `analysis/alcb_shadow_tracker.py`

The shadow tracker records two things:
1. **Funnel counters** (`record_funnel`): "entry_signal" (engine:578), "entered" (engine:714)
2. **Rejected setups** (`record_rejection`): full ShadowSetup with `rejection_gate` tag

Rejection gates logged:
- `"short_conviction_reject"` — engine:527-532
- `"weekday_blocked"` — engine:584-589
- `"sector_blocked"` — engine:597-602
- `"max_positions"` — engine:691
- `"sector_limit"` — engine:695
- `"heat_cap"` — engine:703

Each rejected setup is then **simulated bar-by-bar** (shadow_tracker.py:68+) to compute hypothetical R-multiple, MFE, MAE over max 13 bars (~1 day).

---

## Why Only 16 Trades in 26 Months — Likely Bottlenecks

The funnel has multiplicative attrition. Key chokepoints:

1. **Box detection (Gate 3/6)**: `min_containment=0.80` + `max_squeeze_metric=1.10` together reject MOST box candidates. Only ~5-15% of S&P 500 stocks show tight enough compression at any given time.

2. **Breakout direction (Gate 7)**: `breakout_tolerance_pct=0.0` means zero tolerance — close must be STRICTLY above box.high (long) or below box.low (short). Even a few cents inside the box = rejected.

3. **Selection ranking cutoff (Gate 5)**: Only top N longs + M shorts make the `tradable` list. Many valid boxes never become tradable.

4. **Entry A trigger (Gate 12)**: Requires price to dip TO box edge AND close above/below it within the SAME 30m bar. This is very restrictive — most breakout days show continuation moves away from the box, not pullbacks to it.

5. **Entry B disabled by default**: `use_entry_b=False` means sweep-and-reclaim entries never fire.

6. **Portfolio constraints (Gates 18-20)**: With `max_positions=5` and `heat_cap_r=6.0`, these rarely bind with only 16 trades total, but they would constrain during clustered setups.

The primary bottleneck is likely **Gate 12 (Entry A)** combined with **Gate 7 (zero breakout tolerance)**. The backtest requires a box breakout on the daily timeframe AND an intraday pullback to the box edge within the same session — a very specific price action pattern.
