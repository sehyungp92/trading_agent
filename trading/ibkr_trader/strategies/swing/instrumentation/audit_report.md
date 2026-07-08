# Instrumentation Audit Report

## Bot Identity
- Bot ID: `swing_multi_01`
- Strategy type: Multi-strategy (ATRSS trend-follow, Helix divergence-swing, EMA crossover overlay)
- Exchange(s): Interactive Brokers (IBKR) — futures + equities
- Default pairs: QQQ, GLD (ETFs — default production config)
- Available pairs: MNQ, MCL, MGC, MBT, NQ, CL, GC, BRR, BT, QQQ, GLD, USO (configurable via `ATRSS_SYMBOL_SET`, `AKCHELIX_SYMBOL_SET` env vars)
- Architecture: Hybrid (async polling hourly cycle + event-driven OMS fills + daily overlay rebalance)

## Entry Logic
- Signal generation:
  - [strategy/signals.py:112] `pullback_signal()` — detects pullback-to-EMA-value entries in TREND/STRONG_TREND regime
  - [strategy/signals.py:148] `check_breakout_arm()` — detects breakout arm setup in STRONG_TREND
  - [strategy/signals.py:168] `breakout_pullback_signal()` — 30-50% retracement after arm
  - [strategy/signals.py:209] `reverse_entry_ok()` — stop-and-reverse eligibility
  - [strategy_2/signals.py] Helix: divergence, momentum, catchup signals (CLASS_A/B/C)
  - [shared/overlay/engine.py] Overlay: EMA fast > slow crossover on daily closes (QQQ 10/21, GLD 13/21)
- Signal strength available: YES — [strategy/signals.py:70] `compute_entry_quality()` scores 0-7 (ADX, DI alignment, EMA separation, touch distance, momentum)
- Filters:
  - Filter 1: [strategy/engine.py:89] `_is_entry_restricted(now)` — blocks 09:30-09:35 and 15:55-16:00 ET
  - Filter 2: [strategy/engine.py:385] Halt detection — `halt_states[sym].is_halted`
  - Filter 3: [strategy/engine.py:390] Short direction — `shorts_enabled` + `short_symbol_gate()`
  - Filter 4: [strategy/engine.py:399] Per-symbol time/day — `blocked_hours_et`, `blocked_weekdays`
  - Filter 5: [strategy/engine.py:414] Re-entry cooldown — `same_direction_reentry_allowed()`
  - Filter 6: [strategy/engine.py:424] Quality gate — `score < QUALITY_GATE_THRESHOLD (4.0)`
  - Filter 7: [strategy/engine.py:445] Momentum filter — `momentum_ok()` for breakout signals
  - Filter 8: [strategy/engine.py:408] Post-halt delay — 1-hour delay after halt reopen
  - Filter 9: [strategy/allocator.py:43] Portfolio heat cap — `allocate()` limits by MAX_PORTFOLIO_HEAT
- Order placement: [strategy/engine.py:866] `_submit_entry()` — STOP_LIMIT order via OMS Intent
- Overlay order placement: [shared/overlay/engine.py] `MarketOrder` placed directly via IB API (bypasses OMS)

## Exit Logic
- Exit triggers:
  - TAKE_PROFIT: [strategy/engine.py:651] TP1 at 1.0R (33%), TP2 at 2.0R (33%) — ATR-based R multiples
  - STOP_LOSS: [strategy/stops.py:14] `compute_initial_stop()` — max(daily_atr*2.2, hourly_atr*3.0)
  - TRAILING: [strategy/stops.py:64] `compute_chandelier_stop()` — HH(20d) - chand_mult*ATR20, activated at 1.25R MFE
  - SIGNAL (bias flip): [strategy/engine.py:741] Trend direction changes → FLATTEN_BIAS_FLIP
  - TIMEOUT: [strategy/engine.py:681] MAX_HOLD_HOURS=480 bars AND cur_r < 1.0 → FLATTEN_TIME_DECAY
  - STALL: [strategy/engine.py:671] 36 bars + MFE < 0.4R → FLATTEN_STALL
  - EARLY_STALL: [strategy/engine.py:661] 12 bars + MFE < 0.3R → partial 50% exit
  - CATASTROPHIC: [strategy/engine.py:642] R < -2.0 → FLATTEN_CATASTROPHIC_CAP
  - MANUAL: via risk halt DB update → RISK_HALT event
  - BREAKEVEN: [strategy/stops.py:46] `compute_be_stop()` — entry ± 0.1*daily_ATR20, triggered at 1.25R MFE
  - EMA_BEARISH: [shared/overlay/engine.py] Overlay exits when EMA fast crosses below slow (daily rebalance sells to 0)

## Position Sizing
- Active strategies: [strategy/allocator.py:109] `compute_position_size()` — qty = floor((equity * risk_pct) / (|entry - stop| * point_value))
- Overlay: [shared/overlay/engine.py] target_shares = floor(equity * 0.85 * weight / price), equal-weight across bullish symbols
- Inputs: account equity (IBKR NetLiquidation), base_risk_pct (per-symbol 0.5-1.2%), stop distance, point_value
- Risk limits:
  - [main_multi.py:199] Portfolio heat cap: 2.0R across all strategies
  - [main_multi.py:200] Portfolio daily stop: 3.0R
  - [main_multi.py:161] Per-strategy daily stop: 2.0-2.5R
  - [main_multi.py:163] Per-strategy max heat: 0.65-1.50R
  - [shared/oms/risk/calculator.py:8] Unit risk computation

## Data Sources
- Price: IBKR via `ib_async` — `reqHistoricalDataAsync()` for OHLC bars (daily + hourly)
- Bid/Ask: NOT CURRENTLY CAPTURED (available via IBKR `reqMktData()` but not subscribed)
- Funding: N/A — equities/futures, not perpetuals
- OI: NOT TRACKED (available from IBKR but not requested)

## Existing Logging
- Current format: Structured text (Python logging module)
- Current location: stdout (stream to console/Docker logs)
- Trade logging detail level: Moderate — entry/exit logged with symbol, direction, qty, prices, OMS order ID
- Error logging: Yes — IBKR errors classified via error_map.py, state machine transitions logged
- Signal logging (including blocked): PARTIAL — aggregate counts only ("N generated, M accepted"), no per-signal block reasons

## Configuration
- Config location: `strategy/config.py` (Python dataclasses), `config/contracts.yaml`, `config/ibkr_profiles.yaml`, `.env`
- Configurable params: EMA periods, ATR multipliers, ADX thresholds, risk percentages, TP/SL R-multiples, symbol sets, time filters
- Strategy profiles: YES — 4 active strategies + 1 overlay with per-strategy risk params defined in the swing coordinator

## State Management
- Position tracking: In-memory dicts + PostgreSQL (graceful degradation to in-memory)
- Overlay position tracking: `overlay_state.json` (shares, entry trade IDs, last rebalance date)
- Restart recovery: YES — OMS reconciliation loads DB state, compares to broker, resolves discrepancies; overlay loads from JSON
- Health check: IB connection heartbeat, OMS timeout monitor, strategy state persistence

## Dependencies
- Python: 3.12.6
- Key packages: ib_async>=2.1, asyncpg>=0.31, numpy>=1.26, pandas>=2.3, pydantic>=2.12, pyyaml>=6.0

## Integration Plan

### Hook Points (where to attach instrumentation)
1. **Pre-entry hook:** [strategy/engine.py:371] `_collect_candidates()` — capture all signals before allocation filter
2. **Post-entry hook:** [strategy/engine.py:1275] `_on_fill()` — capture fill confirmation with price, qty, time
3. **Pre-exit hook:** [strategy/engine.py:594] `_manage_position()` — capture exit decision reason
4. **Post-exit hook:** [strategy/engine.py:1421] `_on_stop_fill()` / `_on_flatten_fill()` — capture exit fill
5. **Signal generation hook:** [strategy/signals.py:70] `compute_entry_quality()` — capture all quality scores
6. **Filter hooks:**
   - [strategy/engine.py:89] Time restriction filter
   - [strategy/engine.py:385] Halt filter
   - [strategy/engine.py:390] Short gate filter
   - [strategy/engine.py:399] Time/day block filter
   - [strategy/engine.py:414] Re-entry cooldown filter
   - [strategy/engine.py:424] Quality gate filter
   - [strategy/engine.py:445] Momentum filter
   - [strategy/allocator.py:43] Portfolio heat allocation filter
7. **Error hook:** [strategy/engine.py] try/except blocks around bar fetching, order submission
8. **Main loop hook:** [strategy/engine.py:232] `_hourly_scheduler()` — attach snapshot service to hourly cycle

### Missing Data (must be added)
- [ ] Bid/ask spread at entry/exit (available from IBKR, not subscribed)
- [ ] Per-signal block reason (only aggregate counts logged currently)
- [ ] Slippage vs. expected (fill price vs. trigger price, partially available)
- [ ] Market regime classification (indicators exist, not tagged as regime labels)
- [ ] Signal strength as 0-1 normalized (currently 0-7 integer score)

### Risks
- Async context: all instrumentation must be async-safe or use sync file writes in try/except
- OMS event bus: hooking into event processing must not block or slow fill handling
- Hourly cycle timing: snapshot capture must not delay bar evaluation or order placement
- PostgreSQL dependency: instrumentation must work even when DB is unavailable (write to JSONL first)
- Multiple strategies: instrumentation must tag which strategy generated each event
