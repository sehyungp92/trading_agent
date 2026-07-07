# Instrumentation Audit Report

## Bot identity

- Bot family: `stock_trader`
- Relay bot IDs: `IARIC_v1`, `US_ORB_v1`, `ALCB_v1`
- Venue: IBKR U.S. equities
- Shared infrastructure: PostgreSQL, dashboard, and relay provided by `bots/ibkr_trading`
- Architecture: hybrid event-driven plus async polling

## Strategy coverage

### `IARIC_v1`

- Runtime entrypoint: `python -m strategy_iaric`
- Core engine: `strategy_iaric/engine.py`
- Entry decision flow:
  - setup detection and acceptance state machine in `strategy_iaric/engine.py`
  - final order submission in `IARICEngine._submit_entry()`
- Exit flow:
  - position management in `IARICEngine._manage_position()`
  - fill handling in `IARICEngine._handle_fill()`
- Instrumentation hooks now active:
  - bridged entry and exit logging through `instrumentation/src/pg_bridge.py`
  - missed-opportunity logging on final blocked branches and entry terminal events
  - order lifecycle logging on submit, replace, cancel, and OMS fill/status events
  - heartbeat emission from `strategy_iaric/main.py`
  - periodic indicator snapshots from the 5 minute signal cycle

### `US_ORB_v1`

- Runtime entrypoint: `python -m strategy_orb`
- Core engine: `strategy_orb/engine.py`
- Entry decision flow:
  - candidate ranking and OR evaluation in `USORBEngine._finalize_opening_scan()`
  - state machine progression in `USORBEngine._advance_symbol()`
  - final order submission in `USORBEngine._submit_entry()`
- Exit flow:
  - signal-based exits in `USORBEngine._advance_symbol()`
  - fill handling in `USORBEngine._handle_fill()`
- Instrumentation hooks now active:
  - bridged entry and exit logging through `instrumentation/src/pg_bridge.py`
  - missed-opportunity logging for sector caps, acceptance timeout, quality gates, TTL expiry, and terminal entry failures
  - order lifecycle logging on submit, replace, cancel, and OMS fill/status events
  - heartbeat emission from `strategy_orb/main.py`
  - indicator snapshots at candidate selection and ready-to-submit decision points

### `ALCB_v1`

- Runtime entrypoint: `python -m strategy_alcb`
- Core engine: `strategy_alcb/engine.py`
- Entry decision flow:
  - campaign-state and regime gating in `ALCBEngine._advance_symbol()`
  - final order submission in `ALCBEngine._submit_entry()`
- Exit flow:
  - stop/target/add management in `ALCBEngine._manage_position()`
  - fill handling in `ALCBEngine._handle_fill()`
- Instrumentation hooks now active:
  - bridged entry and exit logging through `instrumentation/src/pg_bridge.py`
  - missed-opportunity logging on entry gates, portfolio caps, friction blocks, and entry terminal events
  - order lifecycle logging on submit, replace, cancel, and OMS fill/status events
  - structured engine error logging for OMS submission failures
  - indicator snapshots at the intraday-evidence decision point
  - assistant-facing 5 minute and 1 hour OHLCV provider support for trade/missed backfills

## Shared instrumentation runtime

- Bootstrap and sidecar lifecycle: `instrumentation/src/bootstrap.py`
- PG-first recorder bridge: `instrumentation/src/pg_bridge.py`
- Strategy data providers:
  - `IARICInstrumentationDataProvider`
  - `ALCBInstrumentationDataProvider`
  - `USORBInstrumentationDataProvider`
- Standardized relay-facing artifacts:
  - trades
  - missed opportunities
  - orders
  - heartbeats
  - market snapshots
  - daily snapshots

## Deployment topology

- This repo does not deploy a relay.
- Containers send signed batches to `http://host.docker.internal:8001/events`.
- The relay implementation lives in `bots/ibkr_trading/relay/`.
- The relay forwards relevant events to `packages/trading_assistant/`.
