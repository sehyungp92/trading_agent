# ADR 0001: Monthly Evidence And Replay Foundation

Date: 2026-05-12

## Decision

Monthly validation is the authoritative learning loop for material strategy and
configuration decisions. Daily, weekly, WFO, and early outcome measurements stay
as sensors, screening, and context.

`trading_assistant` owns orchestration, manifests, coverage checks, ledgers, and
approval routing. Full-fidelity replay engines stay outside this repo behind a
manifest-driven `BACKTEST_REPO_PATH` boundary.

## Boundaries

- Production market data lives under `MARKET_DATA_ROOT`; it is not committed.
- Backtest artifacts live under `BACKTEST_ARTIFACT_ROOT`.
- Model output cannot bypass deterministic coverage, lineage, parity, leakage,
  objective, and risk gates.
- Material trading behavior changes remain manually approved.
- Shadow mode writes evidence and ledgers but creates no approval-ready action.

## Phase 1 Exit

One strategy can complete a shadow monthly incumbent validation with telemetry
coverage, market-data coverage, replay parity, gap attribution, a monthly report,
and a `StrategyChangeLedger` monthly-review record.
