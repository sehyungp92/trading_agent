# Instrumentation Hooks Manifest

Maps strategy classes to their instrumented events via `InstrumentationKit`.

All hooks use the Kit facade (`instrumentation/src/kit.py`), which handles:
- Exception safety (never crashes trading)
- Automatic regime classification on entry
- Automatic process scoring on exit
- Strategy ID injection

## Hook Points

| Hook | Kit Method | When Fired |
|------|-----------|------------|
| Trade Entry | `kit.log_entry()` | After entry fill confirmed |
| Trade Exit | `kit.log_exit()` | After exit fill confirmed (stop, signal, flatten) |
| Missed Opportunity | `kit.log_missed()` | Signal blocked by filter/gate/allocator |
| Market Snapshot | `kit.capture_snapshot()` | Each decision cycle (periodic) |
| Regime Classification | `kit.classify_regime()` | Each decision cycle + auto on every entry |
| Process Scoring | Auto (inside `log_exit`) | Automatically on every exit |

## Enriched Data Captured

All strategies pass these enriched fields on entry (where available):
- `signal_factors` — factors contributing to the entry signal
- `sizing_inputs` — risk %, equity, volatility basis, sizing model
- `filter_decisions` — threshold vs actual for each filter evaluated
- `portfolio_state_at_entry` — exposure, direction, correlated positions

## Per-Strategy Coverage

### ATRSS (`strategy/engine.py`) — bot_id: `ATRSS`

| Event | Location | Details |
|-------|----------|---------|
| Entry | `engine.py:1443` | Signals: PULLBACK, BREAKOUT, REVERSE. Signal strength from quality_score. |
| Exit (stop) | `engine.py:1577` | Per-leg exit on stop loss |
| Exit (flatten) | `engine.py:1647` | Per-leg exit on flatten/bias_flip/timeout/stall |
| Missed (allocator) | `engine.py:311` | Candidate rejected by portfolio allocator |
| Missed (short disabled) | `engine.py:417` | Short trading disabled for symbol |
| Missed (short gate) | `engine.py:428` | Short symbol gate check failed |
| Missed (quality gate PB) | `engine.py:465` | Pullback signal below quality threshold |
| Missed (quality gate BO) | `engine.py:495` | Breakout signal below quality threshold |
| Regime | `engine.py:327` | All symbols each cycle |
| Snapshot | `engine.py:328` | All symbols each cycle |

### AKC_HELIX (`strategy_2/engine.py`) — bot_id: `AKC_HELIX`

| Event | Location | Details |
|-------|----------|---------|
| Entry | `engine.py:1992` | Signals: CLASS_A, CLASS_B, CLASS_C, CLASS_D. Signal strength 0.5. |
| Exit (flatten) | `engine.py:1740` | Setup flatten exit |
| Exit (stop) | `engine.py:2095` | Stop loss exit |
| Missed (allocator) | `engine.py:459` | Setup rejected by portfolio allocator |
| Regime | `engine.py:488` | All symbols each cycle |
| Snapshot | `engine.py:489` | All symbols each cycle |

## Coverage Gaps

| Strategy | Missing Hooks | Impact |
|----------|---------------|--------|
| AKC_HELIX | No filter_decisions populated | Cannot analyze gate threshold sensitivity |
| All | portfolio_state_at_entry not yet wired | Requires position book access at entry time |
