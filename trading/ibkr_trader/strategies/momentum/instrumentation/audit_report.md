# Instrumentation Audit Report

## Bot Identity
- Bot ID: `momentum_nq_01`
- Strategy type: Multi-strategy NQ futures (3 concurrent strategies)
  - **NQDTC v2.1** — 30m box-breakout directional (strategy_2/)
  - **VdubusNQ v4.2** — 15m VWAP pullback swing (strategy_3/)
- Exchange(s): Interactive Brokers (IBKR) — CME/GLOBEX
- Pairs traded: NQ (Nasdaq-100 E-mini), MNQ (Micro Nasdaq-100)
- Architecture: Hybrid event-driven + async polling

## Entry Logic

- Signal generation: `strategy/signals.py:106-379` — Class M (1H pullback), Class T (4H trend continuation)
- Signal strength: `strategy/signals.py:37-56` — alignment_score (0-2) + trend_strength (float)
- Filters:
  - Session gate: `strategy/gates.py:65-76` — blocks DAILY_HALT, REOPEN_DEAD
  - ETH_EUROPE regime: `strategy/gates.py:78-116` — shorts blocked; longs need score>=2
  - Short block RTH_PRIME1: `strategy/gates.py:118-124` — M shorts blocked (25% WR)
  - News block: `strategy/gates.py:126-131` — CPI/NFP/FOMC windows
  - Extension block: `strategy/gates.py:133-145` — daily close > ema +-2.5*ATR
  - High vol block: `strategy/gates.py:147-152` — vol percentile > 97
  - Min stop distance: `strategy/gates.py:154-165` — stop < 5pts
  - Spike filter: `strategy/gates.py:167-174` — bar range > 2.5 ATR
  - Spread gate: `strategy/gates.py:176-189` — bid-ask > session max
  - Heat caps: `strategy/gates.py:191-202` — total risk > 3.0R or directional > 2.5R
  - DOW block: `strategy/engine.py:362-396` — Monday(0) + Wednesday(2) blocked
- Order placement: `strategy/execution.py:49-103` — STOP order (GTC, 12h TTL)
- Fill confirmation: `strategy/engine.py:547-605` — OMS event FILL callback

### NQDTC v2.1
- Signal generation: `strategy_2/engine.py:596-707` — structural breakout of 30m box
- Signal strength: `strategy_2/engine.py:663-685` — evidence scorecard (0-4.5)
- Filters:
  - Regime hard block: `strategy_2/signals.py:84-87` — 4H counter + daily opposes
  - Chop halt: `strategy_2/engine.py:621` — mode=HALT
  - Displacement filter: `strategy_2/signals.py:157-178` — adaptive quantile threshold
  - Quality reject: `strategy_2/signals.py:181-202` — bar structure + RVOL
  - Score threshold: `strategy_2/engine.py:663-685` — score < 1.5 (normal) / 2.5 (degraded)
  - Hour blocks: `strategy_2/engine.py:810-838` — 04/05/06/09/12 ET hours
  - News blackout: `strategy_2/engine.py:456-476` — economic events
  - Daily halt: `strategy_2/engine.py:791-804` — realized_R <= -2.5
  - Friction gate: `strategy_2/engine.py:891-896` — cost > 10% of 1R
- Order placement: `strategy_2/engine.py:993-1203` — LIMIT/STOP-LIMIT OCO groups
- Fill confirmation: `strategy_2/engine.py:1738-1879` — OMS fill event handler

### VdubusNQ v4.2
- Signal generation: `strategy_3/signals.py:82-107` — VWAP touch + reclaim (Type A)
- Signal strength: `strategy_3/signals.py:16-37` — MACD histogram slope + Predator overlay
- Filters:
  - Daily trend: `strategy_3/regime.py:118-141` — ES SMA200 must match direction
  - Vol state: `strategy_3/regime.py:118-141` — SHOCK blocks all entries
  - 1H alignment: `strategy_3/regime.py:118-141` — NQ EMA50 trend must match
  - Chop gate: `strategy_3/regime.py:118-141` — CI > 62 limits to 1 trade/direction/day
  - Midday dead: `strategy_3/config.py` — 10:45-15:00 blocked
  - 20h block: `strategy_3/config.py` — 20:00 ET hour blocked (v4.2)
  - Heat cap: `strategy_3/risk.py:110-132` — 3.5x unit_risk
  - Daily breaker: `strategy_3/risk.py:110-132` — -2.0x unit_risk daily loss
- Order placement: `strategy_3/engine.py:914-981` — STOP-LIMIT (GTC, 4-bar TTL)
- Fill confirmation: `strategy_3/engine.py:1141-1197` — OMS fill event

## Exit Logic

- TAKE_PROFIT: `strategy/partials.py:18-43` — P1 at +1.0R (30%), P2 at +1.5R (30%), 40% runner
- STOP_LOSS: `strategy/execution.py:154-183` — initial stop from setup, STOP order
- TRAILING: `strategy/trail.py:25-46` — chandelier `mult = max(1.5, 3.0 - R/5)`, 24-bar lookback
- MFE_RATCHET: `strategy/positions.py:197-204` — locks 65% of peak MFE at 1.5R+
- EARLY_ADVERSE: `strategy/positions.py:154-158` — bars<=4 AND R<=-0.80
- STALE: `strategy/positions.py:219-223` — bars>=20 AND R<0.30
- CATASTROPHIC: `strategy/positions.py:148-152` — R < -1.5
- TIMEOUT: `strategy/positions.py:213-217` — early_stale bars>=16 with adverse
- MANUAL: N/A (no manual exit facility)

### NQDTC
- TAKE_PROFIT: `strategy_2/config.py:198-202` — TP1 at 1.5R (25%), 75% runner
- TRAILING: `strategy_2/config.py:220-227` — 5-tier chandelier (0-1.5R→4R+)
- RATCHET: `strategy_2/config.py:212-217` — lock 25% of peak_r at 1.5R+
- STALE: `strategy_2/stops.py:174-193` — 20 bars normal, 14 degraded
- NEWS_FLATTEN: `strategy_2/engine.py:456-476` — flatten pre-event

### Vdubus
- PARTIAL: `strategy_3/exits.py:55-68` — +1R partial (33%)
- TRAILING: `strategy_3/exits.py:129-166` — 12-bar 15m, staged mult (2.5→1.5 ATR)
- MFE_RATCHET: `strategy_3/exits.py:192-231` — 4-tier progressive floor
- EARLY_KILL: `strategy_3/exits.py:14-48` — bars<=4, R<-0.25, MFE<0.25
- VWAP_FAIL: `strategy_3/exits.py:173-185` — 2 consecutive bars wrong side
- STALE: `strategy_3/exits.py:238-250` — window-adaptive (5-12 bars)
- DECISION_GATE: `strategy_3/exits.py:257-288` — 15:50 ET hold/flatten
- MAX_DURATION: `strategy_3/exits.py:118-122` — 128 bars (32h)

## Position Sizing
- NQDTC: `strategy_2/sizing.py:41-109` — 0.80% equity * quality_mult (regime * chop * disp) with risk floor 0.12%
- Vdubus: `strategy_3/risk.py:63-103` — 0.80% equity * vol_factor * class_mult (predator/flip) * session_mult
- Risk limits (shared): `shared/oms/risk/gateway.py:50-139` — heat cap, daily/weekly stops, directional cap, portfolio rules

## Data Sources
- Price: IB Historical Data API (`reqHistoricalDataAsync`) — 1D, 4H, 1H, 30m, 15m, 5m bars, `ib_async` library
- Bid/Ask: IB market data subscription (`reqMktData`) — real-time tick updates, used for spread validation
- Funding: N/A (equity futures, no funding rate)
- OI: N/A (not tracked)
- ES daily: Used by NQDTC and Vdubus for macro trend (SMA200)

## Existing Logging
- Current format: Plain text (Python logging module) + JSONL (NQDTC telemetry)
- Current location: stdout (all strategies), `nqdtc_telemetry.jsonl` (NQDTC), `nqdtc_state.json` (NQDTC state)
- Trade logging detail level: Basic (entry/exit at INFO level with prices, PnL)
- Error logging: Python logger at ERROR/EXCEPTION level

## Configuration
- Shared config: `shared/oms/config/risk_config.py`, `shared/oms/config/portfolio_config.py`
- External config: `config/contracts.yaml`, `config/routing.yaml`, `config/ibkr_profiles.yaml`
- Configurable params: All indicator periods, TP/SL levels, session windows, sizing multipliers, risk limits

## State Management
- Position tracking: In-memory (all strategies) + PostgreSQL (OMS positions table)
- NQDTC: Additional JSON state file (`nqdtc_state.json`) for full engine state persistence
- Health check: OMS reconciliation loop (60-180s interval), timeout monitor for stuck orders

## Dependencies
- Python: 3.12
- Key packages: `numpy>=1.26`, `ib_async>=1.0`, `asyncpg>=0.29`, `pytz>=2024.1`, `PyYAML>=6.0`, `pydantic>=2.0`

## Integration Plan

### Hook Points (where to attach instrumentation)

**All strategies share the same OMS, so the cleanest hooks are at the OMS level:**

1. **Pre-entry hook (signal generation)**:
   - NQDTC: `strategy_2/engine.py:596-707` — capture breakout qualification
   - Vdubus: `strategy_3/engine.py:451-632` — capture entry evaluation
2. **Post-entry hook (fill confirmed)**:
   - NQDTC: `strategy_2/engine.py:1738-1879` (`_on_fill()`)
   - Vdubus: `strategy_3/engine.py:1141-1197` (fill handler)
3. **Pre-exit hook (exit decision)**:
   - NQDTC: `strategy_2/engine.py:1334-1488` (`_manage_position()`)
   - Vdubus: `strategy_3/engine.py:695-853` (`_manage_positions()`)
4. **Post-exit hook (exit fill confirmed)**:
   - OMS level: `shared/oms/services/factory.py:322-470` (on_fill callback, EXIT role)
5. **Signal generation hook (all signals, including blocked)**:
   - NQDTC: `strategy_2/engine.py:791-969` — hard gates + soft blocks
   - Vdubus: `strategy_3/regime.py:118-141` + `strategy_3/risk.py:110-132`
6. **Filter hooks**: See filter chains above (one per filter per strategy)
7. **Error hook**: Each strategy's try/except blocks + OMS factory callbacks
8. **Main loop hook**:
   - NQDTC: `strategy_2/engine.py:335-434` (`_on_5m_close()` — 5m cycle)
   - Vdubus: `strategy_3/engine.py:240-256` (15m scheduler)

### Missing Data (must be added)
- [ ] Bid/ask spread at entry/exit: Available from IB market data (`_bid`, `_ask` cached in engines)
- [ ] Funding rate: N/A for equity futures
- [ ] Open interest: Not currently tracked
- [ ] Structured trade event with full context (signal + filters + market state)
- [ ] Missed opportunity tracking (blocked signals)
- [ ] Process quality scoring
- [ ] Daily aggregate rollups
- [ ] Regime classification at trade time

### Risks
- **Async contention**: Instrumentation must not acquire `_eval_lock` or compete with bar processing
- **IB rate limits**: Additional market data requests for snapshots could hit IB pacing limits
- **Disk I/O**: JSONL writes on hot path (entry/exit) must use buffered I/O
- **State persistence timing**: NQDTC writes state every 5m — instrumentation writes must not conflict
- **Multi-strategy coordination**: Bot ID must be per-strategy or use strategy_id prefix for event dedup
