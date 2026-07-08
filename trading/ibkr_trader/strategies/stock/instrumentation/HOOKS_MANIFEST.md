# Instrumentation Hooks Manifest -- Stock Family

Maps strategy classes to their instrumented events via `InstrumentationKit`.

All hooks use the Kit facade (`instrumentation/src/facade.py`), which handles:
- Exception safety (never crashes trading)
- Automatic regime classification on entry
- Automatic process scoring on exit
- Strategy ID injection

## Hook Points

| Hook | Kit Method | When Fired |
|------|-----------|------------|
| Trade Entry | `log_entry()` via InstrumentedTradeRecorder | After entry fill confirmed |
| Trade Exit | `log_exit()` via InstrumentedTradeRecorder | After exit fill confirmed |
| Missed Opportunity | `kit.log_missed()` | Signal blocked by filter/gate/constraint |
| Indicator Snapshot | `kit.on_indicator_snapshot()` | Periodic (every 6th 5m bar = 30 min) |
| Order Event | `kit.on_order_event()` | After entry order submitted to OMS |
| Process Scoring | Auto (inside `log_exit`) | Automatically on every exit |

## Per-Strategy Coverage

### IARIC V2 (`iaric/engine.py`) -- bot_id: `strategy_iaric`

**Entry/Exit:** Via `InstrumentedTradeRecorder` bridge (coordinator wiring).

Entry meta enrichment:
- `signal_factors` -- 9 score components from `compute_entry_score_bundle()`
- `filter_decisions` -- 4-6 gate results (max_positions, sector, spread, regime, score, stopped_out)
- `portfolio_state` -- 5 keys (open positions, pending, equity, risk fraction, sectors)
- `sizing_inputs` -- 7 keys (price, stop, qty, risk/share, sizing_mult, base_risk, equity)
- `strategy_params` -- 11 keys (route, daily score, triggers, tiers, sizing_mult, mfe_stage, stop0, cdd_value, entry_atr, regime_tier, regime_score)

Exit meta enrichment:
- `hold_days`, `carry_decision_path`, `v2_partial_taken`, `trail_active`, `breakeven_activated`
- `daily_signal_score`, `trigger_tier`, `trend_tier`

**Missed Opportunities (~11 call sites):**

| Location | blocked_by | block_reason |
|----------|-----------|--------------|
| `_check_entry_routes` | `not_tradable` | `no_signal_no_tradable_flag` |
| `_check_entry_routes` | `timing_gate` | `outside_entry_window` |
| `_check_entry_routes` | `regime_gate` | `regime_no_new_entries` |
| `_check_entry_routes` | `max_positions` | `at_max_positions` |
| `_check_entry_routes` | `sector_limit` | `sector_cap_reached` |
| `_check_entry_routes` | `spread_gate` | `spread_too_wide` |
| `_submit_entry` | `portfolio_constraints` | (dynamic from adjust_qty) |
| `_try_open_scored_entry` | `entry_score` | `score_N_below_M` |
| `_try_delayed_confirm` | `stopped_out_today` | `same_day_stop_gate` |
| `_try_vwap_bounce` | `stopped_out_today` | `same_day_stop_gate` |
| `_try_afternoon_retest` | `stopped_out_today` | `same_day_stop_gate` |

**Indicator Snapshots:**
- Every 6th 5m bar (30-min intervals) for symbols not in WATCHING/INVALIDATED stage
- Indicators: bars_seen_today, daily_signal_score, intraday_score, mfe_stage, stop_level, daily_atr, hold_bars
- Context: route_family, trigger_tier, trend_tier, stage

**Diagnostics (JSONL):**
- `MFE_STAGE` -- stage transitions with mfe_r and new_stop
- `V2_PARTIAL` -- partial profit trigger with mfe_r and qty
- `CARRY_OVERNIGHT` -- carry decision with path, unrealized_r, hold_days
- `FLATTEN_EOD` -- flatten decision with reason

**Signal Factors (9 components from `compute_entry_score_bundle`):**

| Factor | Max Points | Weight |
|--------|-----------|--------|
| daily_signal_score | 100 | /100 |
| intraday_score | 100 | /100 |
| reclaim | 8 | /8 |
| volume | 12 | /12 |
| vwap_hold | 5 | /5 |
| cpr | 6 | /6 |
| speed | 8 | /8 |
| context | 100 | /100 |
| extension | 100 | /100 |

**Filter Decisions:**

| Filter | Threshold Source | Type |
|--------|----------------|------|
| max_positions | `pb_max_positions` | capacity |
| sector_limit | `max_positions_per_sector` | capacity |
| spread_gate | `max_median_spread_pct * 2.0` | quality |
| regime_gate | OMS risk state | risk |
| entry_score | `pb_entry_score_min` | quality |
| stopped_out_today | boolean | risk |

### US_ORB (`us_orb/engine.py`) -- bot_id: `strategy_us_orb`

Full instrumentation via Kit facade. Entry/exit/missed/snapshot/order events all wired.

### ALCB P14 (`alcb/engine.py`) -- bot_id: `strategy_alcb`

**Entry/Exit:** Via `InstrumentedTradeRecorder` bridge (coordinator wiring).

Entry meta enrichment:
- `signal_factors` -- 5 score components from `_entry_signal_factors()`
- `filter_decisions` -- 7 gate results (avwap, rvol_cap, momentum_score, max_positions, sector, heat, regime)
- `portfolio_state` -- from `_portfolio_state_snapshot()`
- `sizing_inputs` -- 8 keys (qty, prices, risk, base_risk_fraction, equity, regime_mult)
- `strategy_params` -- 17 keys (momentum_score, entry_type, regime_tier, adx, atr, research scores, regimes)

Exit meta enrichment:
- `hold_bars`, `hold_days`, `mfe_r`, `mae_r`, `exit_efficiency`
- `partial_taken`, `fr_trailing_active`, `trade_class`, `carry_days`
- `sector`, `or_high`, `or_low`, `regime_tier`

**Missed Opportunities (~23 call sites):**

| Location | blocked_by | block_reason |
|----------|-----------|--------------|
| `_check_entry` | `entry_window` | `outside_entry_window` |
| `_check_entry` | `or_width` | `too_narrow` |
| `_check_entry` | `rvol_cap` | `rvol_above_max` |
| ... | (17+ additional filter points) | ... |

**Indicator Snapshots:**
- Every 6th 5m bar (30-min intervals) for open positions
- Indicators: hold_bars, unrealized_r, mfe_r, mae_r, current_stop, avwap, bar_close, partial_taken, fr_trailing_active, trade_class
