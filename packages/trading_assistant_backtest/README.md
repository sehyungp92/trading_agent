# trading_assistant_backtest

`trading_assistant_backtest` is the manifest-driven replay and optimizer lab consumed by
`trading_assistant`. It reads a frozen `MonthlyRunManifest`, validates data/plugin inputs,
emits replay diagnostics and monthly optimizer artifacts under `artifact_root`, and never
places live orders or mutates deployment state.

## Boundaries

This repo owns local replay, diagnostics, phased-auto orchestration seams, OOS-repair
artifacts, decision-parity evidence, candidate workspace tracking, and artifact emission.
It does not own canonical market data, approval routing, Telegram/operator cards, OMS
state, live order routing, or deployment.

## Commands

From the enclosing monorepo, set `BACKTEST_REPO_PATH` to:

```text
packages/trading_assistant_backtest
```

The control plane invokes the compatibility runner with its working directory set to that
repo path. From `packages/trading_assistant`, the equivalent relative value is
`../trading_assistant_backtest`.

Native runner:

```bash
python -m trading_assistant_backtest.monthly --manifest path/to/run_manifest.json
```

Formal crypto decision parity validation:

```bash
python -m trading_assistant_backtest.validation.decision_parity_run --artifact-root artifacts/validation/crypto_trend_v1/decision_parity
```

Live repo bridge readiness inventory:

```bash
python -m trading_assistant_backtest.validation.bridge_readiness --artifact-root artifacts/validation/bridge_readiness
```

Compatibility runner used by the current control plane:

```bash
python -m backtests.shared.monthly_repair --manifest path/to/run_manifest.json
```

The default implementation is conservative. Incumbent validation emits the required core
artifact contract. Optimizer modes emit a deterministic no-adoption sequence unless a
future strategy plugin produces a replay-backed candidate that passes the shared gates.

## Live Strategy Bridge

Approval-ready plugins must be wired to the production strategy source of truth, not to a
backtest-only implementation. The intended bridge is:

```text
read-only deployment metadata
  -> local clean checkout at deployed_commit_sha
  -> deterministic live decision API
  -> backtest adapter on identical fixtures
  -> decision_parity_report.json
```

VPSes only publish deployment facts such as `repo_url`, `deployed_commit_sha`,
`config_hash`, strategy version, and telemetry schema version. This repo may clone and
validate that pinned commit locally; it must not import mutable VPS state or deploy code.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m trading_assistant_backtest.monthly --help
python -m backtests.shared.monthly_repair --help
```
