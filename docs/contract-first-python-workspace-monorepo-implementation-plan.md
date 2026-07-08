# Contract-First Python Workspace Monorepo Implementation Plan

## Purpose

This plan describes how to migrate the existing trading systems into one contract-first Python workspace monorepo with many packages and three independently deployable bot images:

- IBKR bot, currently in `trading/ibkr_trader`
- KIS/KRX bot, currently in `trading/k_stock_trader`
- Hyperliquid crypto bot, currently in `trading/crypto_trader`

The trading assistant, currently in `packages/trading_assistant`, should also live in the same monorepo as first-class packages. It should not become a direct in-process dependency of the live trading runtimes. The assistant remains a control/data/backtest plane connected through versioned manifests, artifact indexes, deployment metadata, and parity reports.

The final refactor test is not cosmetic. The latest phased auto-optimization outputs for all strategies must remain unchanged, and every live configuration must be provably derived from or aligned with the latest approved backtest configuration.

## Current Assessment

### Shared Patterns Worth Keeping

The four reference systems already converge on the same architectural ideas:

- Manifest-driven workflows for monthly optimization, backtest execution, data bundles, and deployment evidence.
- Phased auto-optimization with round manifests, provenance, OOS checks, smoke tests, and promotion criteria.
- Strategy plugin contracts that bridge assistant-side backtests to production strategy implementations.
- Runtime deployments that emit deployment metadata and can fail closed when contract evidence is missing or stale.
- Separate broker implementations, live runtime loops, and portfolio policies for each venue.

The refactor should preserve these properties and make them explicit across the whole workspace.

### Bot-Specific Reality

IBKR trading:

- Runtime entrypoint is currently `apps/runtime/runtime.py`.
- Existing layout already separates `apps`, `libs`, `strategies`, `backtests`, `regime`, and `config`.
- Deployment is a single VPS stack with Postgres, IB Gateway, unified runtime, relay, dashboard, and watchdog.
- Backtests use `backtests/shared/auto` and strategy-family output directories.
- Live registry is centered on `config/strategies.yaml`.

KIS/KRX trading:

- Runtime entrypoint is currently `deployment/olr_kalcb/runtime.py`.
- Runtime modes include `artifact_only_stage1`, `artifact_only`, `dry_run`, `paper`, and `live`.
- Deployment stack includes OMS, dashboard, optional PCIM research, KALCB/OLR runtime, and Postgres.
- Runtime readiness depends on artifacts, resource plan, bridge contract, paper/live readiness evidence, and KIS limits.
- Optimization config is in `config/optimization`.
- K-stock latest round artifacts have been restored into `data/backtests/output` and frozen under `backtests/baselines/k_stock`; preserve that restored evidence and do not infer latest K-stock rounds from live config alone.

Crypto trading:

- Main package is already under `src/crypto_trader`.
- CLI owns data download, backtest, optimize, paper trading, deployment manifest preflight, and portfolio workflows.
- Deployment stack includes Postgres, trader, data-refresh, dashboard, Watchtower, and tool profiles for backtest/optimize.
- Live strategy configs are in `config/strategies`.
- Backtest and optimization code already carries contract hashes, phase state contracts, live parity profiles, and optimized config metadata.

Trading assistant:

- Existing package split is `trading_assistant`, `trading_assistant_data`, and `trading_assistant_backtest`.
- It is already contract-oriented. It couples packages and external bots through JSON manifests, artifact indexes, deployment metadata, and parity reports.
- Important contracts include `MarketDataManifest`, `DataBundleManifest`, `MonthlyRunManifest`, `StrategyPluginContract`, `DecisionParityReport`, `artifact_index.json`, `confirmatory_rerank.json`, `repair_ablation_matrix.jsonl`, and `rounds_manifest.json`.
- The assistant should move into the monorepo, but its control/data/backtest package boundaries should stay intact.

### Corrections From Second-Pass Review

The migration has created the root workspace, `packages/`, `trading/`, `contracts/`, `deployments/`, baseline artifacts, and verification tools. The implementation must still preserve the original distinction between these actions:

- Adapt existing package metadata from `trading/ibkr_trader`, `trading/crypto_trader`, `packages/trading_assistant`, `packages/trading_assistant_data`, and `packages/trading_assistant_backtest`.
- Create new package metadata for `trading/k_stock_trader`, which currently has `requirements.txt` and `pytest.ini` but no root `pyproject.toml`.
- Move code out of `legacy source snapshots` only after baseline and import-compatibility gates are in place.

Final `legacy source snapshots` policy:

- `legacy source snapshots` is temporary migration source material only, not a supported dependency or provenance pointer for the finished monorepo.
- Every relevant implementation, fixture, contract, config, test, baseline, and validation artifact from the reference repos must be ported into monorepo-owned paths with hashes/provenance captured there.
- Provenance fields such as `source_repo`, `source_reference`, and `archived_source_path` must resolve to monorepo-owned artifacts plus source hashes, not legacy snapshot paths.
- Final acceptance must fail on any committed runtime, test, verification, deployment metadata, contract, manifest, artifact index, acceptance report, or repo documentation artifact that still points at `legacy source snapshots...`; any archival copy must live outside the repository and must not be referenced by repo artifacts.

Existing assets that should be reused rather than recreated:

- IBKR already has `pyproject.toml`, parity fixtures, `scripts/verify_backtest_baseline_regeneration.py`, `backtests/shared/parity/baseline_regeneration.py`, and a frozen baseline manifest at `tests/fixtures/backtest_baselines/manifest.json`.
- IBKR also already has targeted live/backtest alignment tests, including ALCB round 2 live-default checks and swing portfolio-synergy live parity checks.
- Crypto already has `pyproject.toml`, `crypto_trader.optimize.contracts`, phase-state contract hashes, deployment manifest preflight checks, `parity_alignment.json`, assistant bridge contracts, and a portfolio round 3 live deployment bundle.
- K-stock already has strong runtime readiness, bridge-contract generation, deployment metadata emission, artifact hygiene tests, paper/live replay tooling, and OLR/KALCB runtime gates. It needs workspace packaging, not a rewrite.
- The assistant already has package-local `pyproject.toml` files and a root `tools/run_workspace_checks.py` tier system. Those checks should become part of the root monorepo gate instead of being replaced.

Resolved findings and remaining guardrails from code extraction:

- K-stock latest optimizer output was restored under `data/backtests/output` and frozen into `backtests/baselines/k_stock`; any future baseline refresh must keep exact source hashes and provenance.
- Crypto portfolio round 3 `deployment_manifest.json` referenced a missing `output/portfolio/rounds_manifest.json`; the migration records this through explicit supersession evidence under `backtests/baselines/crypto/portfolio`.
- KALCB frontier-size alignment has been corrected in the active K-stock source to 103 across `config/kalcb.yaml`, `strategy_kalcb/config.py`, and `backtests/strategies/kalcb/phase_candidates.py`. The migration must preserve that live/default/optimizer-base alignment and fail if a stale 104 value reappears.
- The assistant already defines `RoundsManifest`, `RoundManifestRecord`, `ConfirmatoryRerank`, `BacktestArtifactIndex`, and `DecisionParityReport`. The shared contract package should lift or wrap these names, not invent incompatible replacements.
- Existing packages are split between Python 3.11 and 3.12 declarations. The root runtime should standardize on Python 3.12 for live images while preserving 3.11-compatible assistant-backtest/crypto code where that compatibility still has value.

## Target Architecture

Use one Python workspace repository with separately versioned internal packages, shared contracts, shared test gates, and independent deployable Docker images.

```text
trading_agent/
  pyproject.toml
  uv.lock
  README.md
  docs/
    contract-first-python-workspace-monorepo-implementation-plan.md
    strategy-implementation-lessons.md

  packages/
    trading_contracts/
    trading_config/
    trading_backtest/
    trading_optimizer/
    trading_instrumentation/
    trading_deployment/
    trading_strategy_shared/        # create only after two real extractions need it
    trading_assistant/
    trading_assistant_data/
    trading_assistant_backtest/

  trading/
    ibkr_trader/
      src/ibkr_trading/
      config/
      tests/
      Dockerfile
    k_stock_trader/
      src/k_stock_trader/
      config/
      tests/
      Dockerfile
    crypto_trader/
      src/crypto_trader/
      config/
      tests/
      Dockerfile

  backtests/
    baselines/
    data_portability_manifest.json
    fixtures/
    smoke/

  deployments/
    ibkr/
      docker-compose.yml
    k_stock/
      docker-compose.yml
    crypto/
      docker-compose.yml

  contracts/
    strategy_plugins/
    deployment/
    schemas/

  artifacts/
    README.md

  tools/
    build_bot_image.py
    freeze_optimization_baselines.py
    run_backtest_integrity_matrix.py
    run_decision_parity_matrix.py
    verify_backtest_data_portability.py
    verify_latest_round_no_drift.py
    verify_live_config_promotions.py
    verify_strategy_plugin_contracts.py
    verify_deployment_metadata.py
    verify_optimizer_compatibility.py
    verify_refactor_acceptance.py
```

## Workspace Tooling Decisions

Use `uv` as the workspace and lockfile authority. Do not leave this as `uv` vs constraints-file optionality; the monorepo needs one reproducible resolver and one source of truth for image builds.

Required mechanics:

- Root `pyproject.toml` declares `[tool.uv.workspace]` members for `packages/*` and `trading/*`.
- Every internal package has package-local metadata and explicit dependencies.
- Root live-image target is Python 3.12.
- Packages that currently support Python 3.11, especially crypto and assistant-backtest, may keep `requires-python >=3.11` while the root lock is resolved for the live-image Python 3.12 target.
- The bootstrap phase runs and records a workspace dependency-resolution audit before any code move.
- Docker builds install per-bot dependency subsets, using `uv sync --frozen --no-dev --package <bot-package>` or a wrapper-verified equivalent. Live bot images must not install assistant control-plane packages unless a bot explicitly needs a shared contract/runtime helper package.
- CI uses the same lockfile as local development and image builds. No separate requirements files should be generated by hand.

The exact shell commands should live behind repo scripts, for example `tools/workspace_lock_check.py`, `tools/build_bot_image.py`, and `tools/detect_affected_images.py`, so the implementation can adapt to `uv` CLI changes without rewriting the architecture.

## Core Design Principles

### 1. Contracts Before Code Reuse

The first shared package should be `trading_contracts`, not a generic utility package. Shared implementation can move later, but the monorepo needs stable contracts immediately:

- Optimizer round manifests
- Optimized config metadata
- Strategy plugin contracts
- Monthly run manifests
- Data bundle manifests
- Decision parity reports
- Deployment metadata
- Runtime readiness reports
- Artifact indexes
- Promotion manifests
- Telemetry event envelopes

The shared contracts package should expose Pydantic models, JSON Schema generation, canonical JSON hashing, compatibility helpers, and contract validation CLIs.

Do this by lifting the existing assistant contract models and adding compatibility adapters for bot-specific legacy manifests. Do not fork the assistant contract language into a second incompatible schema set.

### 2. Production Strategy Code Remains the Authority

Backtests and assistant workflows can call adapters, but the live strategy implementation remains the source of truth for decisions. The bridge must stay:

```text
production strategy source
  -> deterministic live decision API
  -> backtest adapter
  -> parity fixture set
  -> decision_parity_report.json
```

The migration must avoid creating a second strategy implementation just to make backtests cleaner.

The current implementations are not assumed to be uniformly complete on this point. K-stock is closest to the target shared-core shape, crypto mostly shares strategy objects but still has registry/factory placement to neutralize, and IBKR has parity/replay hooks while some legacy engines still mirror live orchestration. Phase 5 acceptance must therefore prove, strategy by strategy, that promoted backtests are thin replay/simulation adapters over the production decision core, or explicitly mark the strategy as transitional/non-promoted with parity and no-drift evidence plus a deletion/conversion task.

### 3. Shared Optimizer Kernel, Bot-Specific Plugins

The three bots have very similar phased optimization machinery, but the venue semantics differ. Extract shared optimizer orchestration only after baselines are frozen.

Shared:

- Phase specs
- Round management
- Candidate workspaces
- Provenance and mutation tracking
- OOS smoke execution
- Artifact index writing
- Promotion manifest writing
- Contract hash validation

Bot-specific:

- Strategy parameter spaces
- Data loading
- Fill timing and session policy
- Fees, slippage, funding, borrow, and exchange rules
- Broker/live parity constraints
- Portfolio-level allocation logic

### 4. Live Configs Are Promotion Outputs

Live configs should not be manually copied from optimized configs. The target flow is:

```text
optimized_config.json
  + rounds_manifest.json
  + decision_parity_report.json
  + deployment policy
  + venue/runtime overlay
  -> strategy_promotion_manifest.json
  -> generated effective live config
  -> runtime deployment metadata
```

Each live runtime should fail closed when the configured strategy version, config hash, promotion manifest, or deployment metadata does not match the approved optimization artifacts.

### 5. Three Images, Shared Workspace

The repo should build three independent bot images:

- `trading-agent/ibkr-trading`
- `trading-agent/k-stock-trader`
- `trading-agent/crypto-trader`

Each image installs only the packages needed for that bot, plus shared internal packages. Assistant packages can have their own optional images for batch/control workflows, but the live bot images should not require the assistant service at runtime.

## Package Responsibilities

### `packages/trading_contracts`

Owns stable schemas and canonical hashing.

Responsibilities:

- Pydantic models for all cross-package JSON artifacts.
- JSON Schema generation and schema versioning.
- Canonical JSON serialization and SHA-256 hashing.
- Backward-compatible readers for legacy artifacts from the three reference systems.
- Validation CLI for contracts, manifests, and deployment metadata.

Do not put broker clients, strategy logic, data downloads, or optimization algorithms here.

### `packages/trading_config`

Owns config loading, merging, canonicalization, and promotion resolution.

Responsibilities:

- Typed config models.
- Environment overlay loading.
- Secret redaction.
- Config fingerprinting.
- Promotion manifest resolution.
- Generated effective live config artifacts.

This package should replace bot-specific ad hoc config hashing over time.

### Optional `packages/trading_data`

This package should not be created as a second canonical data product. The canonical data product already exists as `trading_assistant_data`.

If a shared runtime-neutral data helper is still needed after two bots need the same interface, create a small package for data contracts and replay adapters only.

Responsibilities:

- Data bundle manifest models.
- Calendar/session policy interfaces.
- Canonical OHLCV/bar schemas.
- Replay bundle integrity checks.
- Shared fixture generation for parity tests.

Bot-specific market data downloaders should stay inside bot packages or `trading_assistant_data` until the interfaces stabilize. When `trading_assistant_data` becomes the go-forward download surface, its adapters must wrap or faithfully port the tried bot-owned download behavior for decision rows; canonical manifests and lineage may be added around those rows, but fetch, pagination, timestamp, merge, dedupe, and aggregation semantics must be proven identical by cross-layer equivalence tests before acceptance.

### `packages/trading_backtest`

Owns shared replay and backtest interfaces.

Responsibilities:

- Backtest run descriptors.
- Replay clock interfaces.
- Bar stream adapters.
- Shared accounting invariants.
- OOS smoke test harness.
- Baseline comparison helpers.

Avoid centralizing venue-specific execution too early. Session policy and fill timing must be explicit inputs.

### `packages/trading_optimizer`

Owns the shared phased optimizer kernel.

Responsibilities:

- Phase runner core.
- Phase state contracts.
- Round manager.
- Candidate mutation ledger.
- Confirmatory rerank interface.
- OOS smoke orchestration.
- Artifact index integration.
- Optimizer provenance and no-drift checks.

The first implementation should wrap existing bot optimizer code with compatibility shims. Only then should duplicated internals be removed.

### `packages/trading_instrumentation`

Owns telemetry behavior, emitters, sinks, lineage propagation, and helpers. Canonical telemetry event models and schema versions live in `trading_contracts`.

Responsibilities:

- Assistant-compatible event stream helpers.
- Lineage and run ID propagation.
- Structured log format helpers.
- Redaction policy.

Runtime logging implementation can stay bot-local until event contracts are unified.

### `packages/trading_deployment`

Owns deployment behavior, readiness checks, image/build metadata capture, and fail-closed runtime gates. Canonical deployment/readiness models and schema versions live in `trading_contracts`.

Responsibilities:

- Deployment metadata emitters and validators.
- Runtime readiness checks.
- Image label helpers.
- Git commit and dirty-state capture.
- Config hash and contract hash validation.
- Fail-closed deployment gate helpers.

### `packages/trading_assistant`

Move from `packages/trading_assistant`.

Responsibilities remain:

- Control-plane planning.
- Approval workflow.
- Evidence verification.
- Learning ledger checks.
- Human-in-the-loop orchestration.

It should depend on `trading_contracts`, not on bot runtime internals.

### `packages/trading_assistant_data`

Move from `packages/trading_assistant_data`.

Responsibilities remain:

- Canonical data product.
- Raw imports.
- Data bundle manifests.
- Coverage manifests.
- Calendar and checksum policy.

It can share model definitions with `trading_contracts`.

### `packages/trading_assistant_backtest`

Move from `packages/trading_assistant_backtest`.

Responsibilities remain:

- Monthly optimizer runner.
- Manifest-driven replay.
- Diagnostics.
- Phased auto optimization orchestration.
- Repair and confirmatory workflows.
- Decision parity reporting.

It should call bot strategy adapters through declared plugin contracts, not by importing arbitrary runtime modules.

### `trading/ibkr_trader`

Owns IBKR-specific runtime behavior.

Responsibilities:

- IBKR broker adapter and session management.
- IB Gateway connection handling.
- IBKR runtime shell.
- IBKR OMS integration.
- Swing, momentum, stock runtime family coordinators.
- IBKR-specific deployment config and Dockerfile.

### `trading/k_stock_trader`

Owns KIS/KRX-specific runtime behavior.

Responsibilities:

- KIS broker adapter.
- KRX calendars and session policy.
- OMS integration.
- OLR/KALCB runtime modes.
- Resource plan and API quota enforcement.
- KIS-specific deployment config and Dockerfile.

### `trading/crypto_trader`

Owns Hyperliquid-specific runtime behavior.

Responsibilities:

- Hyperliquid exchange adapter.
- Crypto live/paper loop.
- Crypto portfolio policy.
- Funding/fees/exchange constraints.
- Crypto-specific deployment config and Dockerfile.

## Contract Model

### Model Ownership Rule

`trading_contracts` owns canonical Pydantic models, JSON Schemas, schema versions, compatibility readers, and canonical hashing. Packages such as `trading_deployment`, `trading_instrumentation`, `trading_config`, `trading_optimizer`, and `trading_backtest` own behavior that imports those models.

Do not duplicate canonical model definitions in behavior packages. If a behavior package needs a deployment metadata payload, telemetry event envelope, readiness report, promotion manifest, or round manifest, it imports the contract model and adds behavior around it.

### Required Cross-System Contracts

Create or consolidate these models in `trading_contracts`, preserving existing assistant names where they already exist:

- `DataBundleManifest`
- `MonthlyRunManifest`
- `StrategyPluginContract`
- `DecisionParityReport`
- `RoundsManifest`
- `RoundManifestRecord`
- `OptimizationContract`
- `OptimizerPhaseState`
- `OptimizedConfigMetadata`
- `ArtifactIndex`
- `BacktestArtifactIndex`
- `ConfirmatoryRerank`
- `RepairAblationMatrixEntry`
- `StrategyPromotionManifest`
- `DeploymentMetadata`
- `RuntimeReadinessReport`
- `TelemetryEventEnvelope`

### Promotion Manifest

Add a promotion manifest as the single bridge from latest backtest config to live config.

```json
{
  "schema_version": "strategy_promotion_manifest.v1",
  "strategy_id": "example_strategy",
  "bot_id": "ibkr_trading",
  "venue": "ibkr",
  "optimizer_round": {
    "round_id": 3,
    "rounds_manifest_path": "backtests/output/example/rounds_manifest.json",
    "optimized_config_path": "backtests/output/example/round_3/optimized_config.json",
    "optimized_config_hash": "sha256:..."
  },
  "evidence": {
    "artifact_index_path": "backtests/output/example/round_3/artifact_index.json",
    "decision_parity_report_path": "backtests/output/example/round_3/decision_parity_report.json",
    "oos_smoke_report_path": "backtests/output/example/round_3/oos_smoke_report.json"
  },
  "live_config": {
    "template_path": "trading/ibkr_trader/config/strategies.yaml",
    "generated_path": "deployments/ibkr/generated/strategies.effective.yaml",
    "generated_config_hash": "sha256:..."
  },
  "approval": {
    "status": "approved",
    "approved_at": "2026-05-23T00:00:00Z",
    "approved_by": "assistant-workflow"
  }
}
```

Live runtimes should read generated effective configs plus their promotion manifests. They should emit the promotion hash and config hash in deployment metadata.

## Baseline Freeze

Before moving implementation code, freeze all latest optimization outputs.

### Known Latest Rounds From Local References

IBKR latest non-archived/non-invalidated rounds found locally:

| Family | Strategy | Round | Timestamp | Key observed metrics |
| --- | --- | ---: | --- | --- |
| momentum | downturn | 4 | 2026-05-04 | 31 mutations, 127 trades, profit factor about 3.14 |
| momentum | nqdtc | 5 | 2026-05-25 | 23 mutations, 161 trades, profit factor about 1.87 |
| momentum | nq_regime | 6 | 2026-05-05 | 92 mutations, 681 trades, profit factor about 7.46 |
| momentum | portfolio_synergy | 2 | 2026-05-25 | 14 mutations, 1238 trades, profit factor about 3.40 |
| momentum | vdubus | 3 | 2026-05-23 | 29 mutations, 212 trades, profit factor about 2.65 |
| regime | crisis | 9 | 2026-05-06 | 29 mutations |
| stock | alcb | 2 | 2026-05-23 | 35 mutations, 589 trades, profit factor about 2.21 |
| stock | iaric | 1 | 2026-05-24 | 52 mutations, 1446 trades, profit factor about 2.31 |
| stock | portfolio_synergy | 3 | 2026-05-24 | 8 mutations, 1684 trades, profit factor about 2.24 |
| swing | atrss | 3 | 2026-05-23 | 14 mutations, 264 trades, profit factor about 6.58 |
| swing | helix | 5 | 2026-05-23 | 39 mutations, 325 trades, profit factor about 2.16 |
| swing | portfolio_synergy | 3 | 2026-05-23 | 165 mutations, 716 trades, profit factor about 3.36 |
| swing | tpc | 8 | 2026-05-23 | 96 mutations, 126 trades, profit factor about 2.44 |

Crypto latest rounds found locally:

| Strategy | Round | Timestamp | Key observed metrics |
| --- | ---: | --- | --- |
| breakout | 3 | 2026-05-27 | 5 mutations, 17 trades, profit factor about 23.33 |
| momentum | 3 | 2026-05-26 | 6 mutations, 46 trades, profit factor about 2.72 |
| trend | 3 | 2026-05-25 | 2 mutations, 29 trades, profit factor about 5.93 |

Latest-round disambiguation rule:

- Do not select by round number alone. Some IBKR manifests contain multiple entries for the same round number after reruns.
- Prefer the manifest's explicit `latest_round` designation when present, then select the latest non-archived, non-invalidated entry for that round by timestamp.
- If no `latest_round` exists, select the latest non-archived, non-invalidated entry by timestamp.
- Freeze the exact manifest entry, timestamp, selection fingerprint, diagnostics fingerprint, config hash, and metrics used for the decision.
- If a manifest entry points to `archive_path`, `cleanup_note`, or other relocated artifacts, the freezer must resolve those references or fail closed. It must not silently skip phase state, optimized config, diagnostics, or candidate ledger files for cleaned rounds.

K-stock latest rounds:

- The local checkout now contains restored `data/backtests/output` artifacts for KALCB, OLR, and portfolio synergy.
- The frozen baseline index must retain source hashes, output-tree fingerprints, and provenance for those restored artifacts.
- If those restored artifacts are ever replaced, rerun the latest accepted optimizer workflow in a frozen environment and mark regenerated artifacts as the baseline only with explicit approval.

Crypto portfolio adoption bundle:

- Strategy-level crypto rounds are present for momentum, trend, and breakout.
- The live deployment example points to portfolio round 3 recommended strategy configs and `output/portfolio/round_3/deployment_manifest.json`.
- `output/portfolio/round_3/parity_alignment.json` says the portfolio bundle is matched.
- `output/portfolio/round_3/deployment_manifest.json` references `output/portfolio/rounds_manifest.json`, which remains absent from the bot/reference tree; the migration records explicit supersession evidence in `backtests/baselines/crypto/portfolio/rounds_manifest.superseded.json`.

Existing IBKR baseline fixture:

- `trading/ibkr_trader/tests/fixtures/backtest_baselines/manifest.json` already defines the artifact/regeneration shape for frozen diagnostic baselines.
- Extend that format into the root baseline index instead of inventing a second baseline manifest shape.
- Existing helper code in `trading/ibkr_trader/backtests/shared/parity/baseline_regeneration.py` already supports sandbox regeneration, normalized hashes, metric checks, and explicit regeneration metadata.

### Baseline Artifacts To Preserve

For every strategy, copy or register the following into `backtests/baselines`:

- `rounds_manifest.json`
- latest `optimized_config.json`
- phase state files
- candidate ledger and mutation ledger
- confirmatory rerank report, where present
- OOS smoke report
- decision parity report, where present
- artifact index
- deployment metadata snapshot, where present
- source commit and dirty-state snapshot
- deployment manifest or live deployment bundle, where present
- parity alignment report, where present
- source strategy config hashes and effective live config hashes
- selected manifest entry metadata, including duplicate-round disambiguation fields
- archive or cleanup provenance, where present

The baseline freeze should store hashes, not rely only on file names.

### Backtest Data Portability Prerequisite

Latest-round no-drift is not complete unless the historical data needed to rerun those rounds is present in monorepo-owned paths or explicitly frozen as accepted baseline evidence.

Required data evidence:

- IBKR raw stock, momentum, swing, and regime backtest market data under `trading/ibkr_trader/backtests/*/data/raw`, tracked through Git LFS where binary.
- Crypto candle, funding, and asset metadata under `trading/crypto_trader/data`, tracked through Git LFS where binary.
- K-stock accepted latest-round evidence under `backtests/baselines/k_stock`; large generated `data/backtests/output` trees are local/object-storage audit evidence only, not the source-control authority.
- `backtests/data_portability_manifest.json`, generated and verified by `python tools/verify_backtest_data_portability.py --bot all`, with file counts, sizes, tree hashes, and per-file hashes.

Strict `python tools/verify_latest_round_no_drift.py ... --strict` must fail if `tools/verify_backtest_data_portability.py` fails for the requested bot. Live Docker contexts must exclude these historical data trees through `.dockerignore` so reproducibility data does not leak into deployment images.

## Implementation Phases

### Phase Transition Gates

Each phase has local tasks, but phase movement is governed by the acceptance matrix. The rule is cumulative: when moving from phase N to phase N+1, all gates that were already green must still be green. Any unexplained optimizer drift, live-config mismatch, missing contract evidence, or `legacy source snapshots` runtime dependency stops the migration at the current phase.

| Transition | Required acceptance gates | Must be true before moving on |
| --- | --- | --- |
| Start Phase -1 | none | References have been read and no code has been moved yet. |
| Phase -1 -> Phase 0 | A00 | Root repository, VCS, remote, CI, artifact storage, uv workspace policy, and dependency-resolution audit are established enough for later commit/config/deployment hashes to be meaningful. |
| Phase 0 -> Phase 1 | A0, A2, A3, A4, A5 decision | Source inventory exists; IBKR and crypto latest rounds are frozen; required historical backtest data is ported and hashed in `backtests/data_portability_manifest.json`; crypto portfolio bundle integrity is restored or explicitly superseded; K-stock has restored/regenerated baseline artifacts or a documented scoped blocker; KALCB `frontier.size` 103 alignment is verified and recorded. |
| Phase 1 -> Phase 2 | A1 plus prior gates | Root workspace imports work; IBKR, crypto, assistant, and K-stock entrypoints still behave through compatibility paths; no runtime behavior has intentionally changed. |
| Phase 2 -> Phase 3 | A9, A12 plus prior gates | Shared contract schemas/readers validate existing assistant, IBKR, crypto, and K-stock artifacts without changing their meaning; strategy plugin contracts are canonicalized under the monorepo path; runtime/deployment metadata defaults no longer point at legacy assistant-backtest contract paths. |
| Phase 3 -> Phase 4 | A4, A6, A7, A8 plus prior gates | Every enabled live strategy maps to latest frozen backtest evidence or an explicit overlay/non-promotion decision; generated effective live configs contain materialized runtime values, not only hash snapshots; crypto portfolio round 3 is resolved or explicitly superseded; KALCB `frontier.size=103` preservation is parsed from live config, strategy default, optimizer base mutation, and deployment universe. Runtime metadata emission and fail-closed startup validation remain Phase 7/A16 gates. |
| Phase 4 -> Phase 5 | A13 plus A2, A3, A5 re-run | Shared optimizer adapters prove compatibility with existing bot runners; latest phased optimizer outputs still match frozen baselines after optimizer wrapping. |
| Phase 5 -> Phase 6 | A10, A14 plus prior gates | Shared backtest/replay invariants pass; promoted strategies have decision parity evidence or are explicitly not promoted; a thin-adapter audit proves no promoted backtest owns a separate strategy brain; strategy implementation lessons are enforced by tests. |
| Phase 6 -> Phase 7 | A11 plus A9, A10 re-run | Assistant packages are in the workspace, existing assistant workspace tiers pass, `trading_contracts` has no reverse dependency on assistant-backtest runtime code, and live-bot coupling to assistant packages remains contract-only. |
| Phase 7 -> Phase 8 | A15, A16 plus A6, A7, A8 re-run | Three independent bot images build; each image can start in artifact-only, dry-run, or paper-compatible mode; deployment metadata includes commit, image, config, promotion, contract, strategy, telemetry, and runtime provenance; image dependency reports prove assistant control-plane/backtest packages are absent unless explicitly approved. |
| Phase 8 -> Phase 9 | A10, A13, A14 plus A2, A3, A5 re-run | Any shared strategy-core extraction has proven decision parity and optimizer no-drift; venue-specific execution/session semantics remain explicit. |
| Phase 9 -> Complete | A17 plus all prior gates | Final acceptance report proves latest phased auto-optimization artifacts are unchanged, live configs align with latest approved backtest configs, every relevant reference asset has been ported into monorepo-owned paths with hashes/provenance, no committed artifact points at `legacy source snapshots`, no nested `legacy source snapshots` remains under production package paths, and local runtime outputs are ignored or moved out of deployable trees. |

### Phase -1 - Repository and Tooling Bootstrap

Goal: make repository identity, CI, lockfile resolution, and artifact storage real before any baseline or deployment evidence depends on them.

Tasks:

- Initialize the root as a git repository if it is still not one.
- Choose and configure the remote host and CI platform.
- Add branch protection rules for the future main integration branch.
- Decide artifact storage policy: Git LFS, object storage, or both, with exact file classes assigned to each.
- Create root workspace metadata using `uv` and `[tool.uv.workspace]`.
- Define the Python compatibility policy: Python 3.12 for live images, package-local 3.11 compatibility only where still required.
- Run a dependency-resolution audit across the planned workspace packages.
- Add repo scripts that wrap workspace lock checks, package import smoke, and affected-image detection.
- Document the bootstrap decisions in `docs/repo-bootstrap-decisions.md`.

Deliverables:

- `.git/` initialized at the root with remote configured.
- CI provider selected and stub workflow added.
- `pyproject.toml` workspace skeleton.
- `uv.lock` generated or a dependency-conflict report explaining what must be fixed first.
- Artifact storage decision record.
- `docs/repo-bootstrap-decisions.md`.

Required checks:

- `python tools/check_repo_bootstrap.py`
- `python tools/workspace_lock_check.py`
- `python tools/detect_affected_images.py --explain-only`

Do not proceed to baseline freeze until repository identity and lockfile strategy are settled. Otherwise deployment metadata cannot truthfully prove commit SHA, dirty state, or reproducible dependency resolution.

### Phase 0 - Freeze and Inventory

Goal: create a no-drift safety net before moving code.

Tasks:

- Create `tools/freeze_optimization_baselines.py`.
- Reuse the existing IBKR baseline fixture schema and regeneration helpers where possible.
- Scan all bot output directories for latest valid `rounds_manifest.json`.
- Resolve the latest optimized config for each strategy using the duplicate-round disambiguation rule.
- Resolve `archive_path`, `cleanup_note`, and relocated artifact references for cleaned rounds; fail closed when a latest-round artifact cannot be found.
- Generate `backtests/baselines/baseline_index.json`.
- Generate one `StrategyPromotionManifest` draft per strategy.
- Capture current live config hashes.
- Capture current strategy plugin contracts from assistant and K-stock references.
- Restore or regenerate missing K-stock backtest outputs.
- Restore, regenerate, or explicitly supersede crypto `output/portfolio/rounds_manifest.json`.
- Verify KALCB `frontier.size` is 103 in `config/kalcb.yaml`, `strategy_kalcb/config.py`, and `backtests/strategies/kalcb/phase_candidates.py`; fail closed if any stale 104 live/default/optimizer-base value remains.

Deliverables:

- `backtests/baselines/baseline_index.json`
- `backtests/baselines/<bot>/<strategy>/...`
- `contracts/strategy_plugins/...`
- `contracts/promotions/draft/...`

Required checks:

- `python tools/freeze_optimization_baselines.py --check`
- `python tools/verify_latest_round_no_drift.py --baseline backtests/baselines/baseline_index.json`
- `python tools/verify_live_config_promotions.py --bot all --require-latest-round --require-effective-configs --strict`

Do not proceed until this phase is green for IBKR, crypto, and K-stock, the crypto portfolio rounds manifest is restored or explicitly superseded, and the K-stock baseline decision is backed by restored or approved regenerated artifacts.

### Phase 1 - Create Workspace Without Behavior Changes

Goal: import existing code into the monorepo with minimal edits.

Tasks:

- Finalize root `pyproject.toml` workspace members from the Phase -1 skeleton.
- Keep Python 3.12 as the live-image runtime target.
- Adapt existing package metadata from IBKR, crypto, and assistant packages.
- Create new package metadata for K-stock from `requirements.txt`, `pytest.ini`, and its runtime/test entrypoints.
- Add package directories under `packages` and bot directories under `trading`.
- Leave the physical assistant package move for Phase 6; in Phase 1, only adapt metadata and compatibility paths needed for import checks.
- Move or copy bot code into `trading` with compatibility entrypoints.
- Keep old import paths working through temporary compatibility modules.
- Keep existing configs and compose files functionally unchanged.

Deliverables:

- Root workspace config.
- Editable installs for all packages.
- Bot CLI entrypoints that match current behavior.
- Temporary import compatibility layer.

Required checks:

- Existing unit tests for each bot.
- Assistant contract checks.
- Import smoke test for every runtime entrypoint.
- `python -m apps.runtime.cli --help` or equivalent compatibility command for IBKR.
- `python -m crypto_trader.cli --help` compatibility command for crypto.
- K-stock runtime/session CLI compatibility command for `scripts/run_olr_kalcb_runtime_session.py`.
- No new bot image build is required in this phase; Docker build gates start in Phase 7 after Dockerfiles exist.

Rollback:

- Because behavior is unchanged, rollback is path-level: redeploy current reference images or old compose stacks.

### Phase 2 - Establish Shared Contracts

Goal: make contracts canonical before extracting implementation code.

Tasks:

- Implement `packages/trading_contracts`.
- Lift assistant contract models into shared models where appropriate, preserving existing names and versions.
- Add compatibility readers for crypto `optimizer_contract_v1`, crypto deployment manifest schema version 1, K-stock `k_stock_olr_kalcb_strategy_plugin_contract_v1`, and IBKR legacy round manifests.
- Define contract models for `DeploymentMetadata`, `RuntimeReadinessReport`, and `TelemetryEventEnvelope` in `trading_contracts`.
- Add schema version constants and JSON Schema output.
- Implement canonical JSON serialization and hash helpers.
- Add legacy readers for existing IBKR, K-stock, and crypto artifacts.
- Add validation CLI:
  - `trading-contracts validate-rounds-manifest`
  - `trading-contracts validate-plugin-contract`
  - `trading-contracts validate-promotion`
  - `trading-contracts validate-deployment-metadata`

Deliverables:

- Shared models.
- Generated schemas in `contracts/schemas`.
- Contract validation CLI.

Required checks:

- All existing assistant contracts validate unchanged.
- All known latest IBKR and crypto round manifests validate unchanged.
- Restored K-stock manifests validate or have a documented compatibility adapter.
- Existing assistant `tools/run_workspace_checks.py` contract tiers still pass after import updates.

### Phase 3 - Canonical Live Config Promotion

Goal: make live configs provably aligned with latest frozen backtest evidence before runtime cutover.

Tasks:

- Implement `packages/trading_config`.
- Define config merge order:
  1. Strategy defaults
  2. Latest optimized config
  3. Venue/runtime overlay
  4. Environment-specific deployment overlay
  5. Secrets by environment only, never committed
- Create promotion manifests for every live strategy with Phase 3 evidence status.
- Generate materialized effective live config artifacts for all three bots; hash-only snapshots may exist as interim evidence but do not satisfy A6-A8.
- Record promotion manifest hashes and effective config hashes in generated config artifacts for Phase 7 deployment metadata.
- Leave runtime deployment metadata emission and fail-closed startup checks to Phase 7/A16 checklist items 30 and 31.

Deliverables:

- `contracts/promotions/<bot>/<strategy>.json`
- `deployments/<bot>/generated/*.effective.*`
- `tools/verify_live_config_promotions.py`

Required checks:

- Generated effective live config artifacts include materialized runtime values and match current live behavior.
- Every enabled strategy in `config/strategies.yaml`, K-stock runtime config, and crypto `config/strategies/*.json` maps to a promotion manifest with non-draft Phase 3 evidence status.
- Disabled/research strategies are explicitly marked as `disabled`, `research`, or `not_promoted`.
- Crypto live config points to the portfolio round 3 bundle with explicit supersession evidence or an approved successor bundle.
- KALCB `frontier.size` equals the deployment universe/optimizer baseline at 103 in live config, strategy default, and phase optimizer base mutation; any future divergence requires an approved promotion overlay.
- IBKR hard-coded strategy defaults covered by existing live parity tests are represented in promotion evidence before those tests are removed or generalized.
- Promotion manifests and generated effective config artifacts use monorepo paths, not `legacy source snapshots` paths.

### Phase 4 - Extract Shared Optimizer Kernel

Goal: remove optimizer duplication while proving no drift.

Tasks:

- Implement `packages/trading_optimizer`.
- Start by wrapping existing per-bot `PhaseRunner` implementations behind a common interface.
- Add a compatibility test suite that runs each old and new runner on a fixed fixture.
- Move shared concepts:
  - `PhaseSpec`
  - `PhaseAnalysisPolicy`
  - round management adapters, including IBKR/K-stock `RoundManager` compatibility and crypto's procedural `rounds_manifest.json` writer
  - candidate workspace handling
  - mutation/provenance ledger
  - contract hash attachment
  - artifact index writing
- Keep bot-specific plugins in bot or strategy packages.
- Do not force crypto to grow a `RoundManager` class during the wrap phase. Its optimizer contracts are currently dict/procedural; wrap that shape first and extract only after compatibility evidence is green.
- Migrate one strategy family at a time.

Deliverables:

- Common optimizer package.
- Bot-specific optimizer adapters.
- No-drift fixture tests.

Required checks:

- Latest round manifests remain byte-for-byte unchanged where rerunning is deterministic.
- Where byte-for-byte stability is impossible because of timestamps or path normalization, canonical hashes and metrics must match with explicit ignored fields.
- OOS smoke reports remain unchanged in pass/fail decision.

### Phase 5 - Shared Backtest and Replay Interfaces

Goal: standardize replay contracts without erasing venue semantics.

Tasks:

- Implement `packages/trading_backtest`.
- Introduce shared `DecisionEvent`, `TradeOutcome`, replay clock, and execution causality invariants from `docs/strategy-implementation-lessons.md`.
- Convert backtest drivers into thin adapters around production strategy decision APIs.
- Produce a per-strategy thin-adapter audit that classifies each promoted backtest as shared-core replay, shared strategy object replay, or transitional mirror with non-promotion/deletion evidence.
- Centralize artifact hygiene checks.
- Centralize OOS smoke harness.
- Keep venue fill timing explicit:
  - IBKR strategy family policy
  - KRX `next_5m_open` and auction constraints
  - crypto exchange/funding policy

Deliverables:

- Shared backtest interfaces.
- Adapter tests for every promoted strategy.
- Parity fixture set per strategy family.

Required checks:

- Existing backtest outputs still resolve through compatibility readers.
- Decision parity report is generated for every promoted strategy.
- Execution causality and accounting invariant tests pass.

### Phase 6 - Move Assistant Into Workspace

Goal: make assistant packages first-class without coupling them to live runtimes.

Tasks:

- Move:
  - `packages/trading_assistant`
  - `packages/trading_assistant_data`
  - `packages/trading_assistant_backtest`
- Update imports to use shared contracts.
- Keep assistant workflows manifest-driven.
- Update strategy plugin contracts to point at new monorepo paths.
- Add clean checkout and pinned commit validation against the monorepo.
- Keep assistant runtime and live bot runtime images separate.

Deliverables:

- Assistant packages under `packages`.
- Updated assistant workflow documentation.
- Updated `StrategyPluginContract` paths.

Required checks:

- `python tools/run_workspace_checks.py monthly-focused`
- `python tools/run_workspace_checks.py data-contracts`
- `python tools/run_workspace_checks.py backtest-monthly`
- `python tools/run_workspace_checks.py backtest-approval`
- `python tools/run_workspace_checks.py loop-contracts`
- `python tools/run_workspace_checks.py performance-learning-ledger`
- `python tools/run_workspace_checks.py monthly-verifier`
- monthly optimizer runner dry run
- monthly evidence verifier dry run
- strategy plugin contract validation

### Phase 7 - Bot Images and Deployments

Goal: deploy each bot independently while sharing package source.

Tasks:

- Create Dockerfiles:
  - `trading/ibkr_trader/Dockerfile`
  - `trading/k_stock_trader/Dockerfile`
  - `trading/crypto_trader/Dockerfile`
- Keep separate compose stacks under `deployments`.
- Implement or migrate `packages/trading_deployment` behavior for metadata emission, image labels, dirty-state checks, readiness gates, and fail-closed startup validation.
- Implement or migrate `packages/trading_instrumentation` behavior for event emission, lineage propagation, structured logging helpers, and redaction using contract-owned event models.
- Include image labels:
  - git commit
  - package versions
  - contract schema version
  - promotion manifest hash
  - effective config hash
- Keep bot-specific secrets out of images.
- Ensure each runtime emits deployment metadata on startup.
- Ensure each runtime validates its own promotion and strategy plugin contracts before trading.

Deliverables:

- Three independent deployable images.
- Three compose stacks.
- Deployment metadata snapshots.

Required checks:

- `docker build` for all three images.
- Runtime import smoke in each image.
- Contract validation in each image.
- Per-image `.dockerignore` or equivalent build-context rules exclude `legacy source snapshots`, bot-local `output/`, bot-local `backtests/output/`, and bot-local `data/backtests/output/`.
- Artifact-only or dry-run mode starts without broker secrets.

### Phase 8 - Strategy Core Consolidation

Goal: extract real shared strategy code only when it removes duplication safely.

Tasks:

- Identify duplicated strategy primitives:
  - indicators
  - signal filters
  - completed-bar policy
  - sizing helpers
  - stop and exit helpers
  - portfolio allocation helpers
- Move stable primitives into `packages/trading_strategy_shared` or package-specific shared modules.
- For every extraction, run decision parity against frozen fixtures.
- Do not merge venue-specific strategy behavior unless the contract proves it is identical.

Deliverables:

- Shared strategy primitives.
- Parity tests for extracted logic.
- Removed duplicated code only after parity is green.

Required checks:

- Strategy fixture decisions unchanged.
- Latest optimizer baseline checks unchanged.
- Live config promotion checks unchanged.

### Phase 9 - Remove Compatibility Layer

Goal: finish the migration once production evidence is stable.

Tasks:

- Remove old import shims.
- Remove every committed dependency or path reference to `legacy source snapshots` after porting, including tests, runtime, active verification paths, package-local deployment metadata, manifests, artifact indexes, acceptance reports, and docs.
- Move any archival reference snapshots outside the repository after relevant assets have been ported and provenance is captured in monorepo-owned artifacts.
- Update all documentation to use new monorepo paths and provenance hashes instead of reference-repo paths.
- Lock the workspace.

Deliverables:

- Clean package imports.
- No committed runtime, test, active verification, deployment metadata, manifest, artifact index, acceptance report, or documentation dependency or path reference to `legacy source snapshots`.
- Final migration report.

Required checks:

- Full CI green.
- All baseline no-drift checks green.
- All runtime/live promotion manifests have final deployment approval status.
- All three dry-run/paper deployments validated.

## Per-Bot Migration Notes

### IBKR Bot

Keep the unified runtime shell and family coordinator model. Move broker and runtime code into `trading/ibkr_trader`, while extracting reusable runtime metadata and contract validation to shared packages.

Important preservation points:

- `apps/runtime/runtime.py` behavior
- IB Gateway preflight checks
- strategy family coordinator registry
- `config/strategies.yaml` enabled/disabled semantics
- relay, dashboard, watchdog deployment topology
- current swing, momentum, stock strategy IDs

The live strategy registry should eventually be generated from promotion manifests, but initially it should be compared against the generated output to prove no behavior drift.

### K-Stock Bot

Keep K-stock runtime modes and readiness gates intact. This bot has the strongest runtime artifact gate and should influence the shared deployment package.

Important preservation points:

- `artifact_only_stage1`, `artifact_only`, `dry_run`, `paper`, and `live` modes
- OMS and runtime split
- KIS resource plan
- action router
- bridge contract
- paper/live readiness evidence
- `live_parity_fill_timing`
- auction and KRX session constraints

The restored `data/backtests/output` baseline evidence must stay frozen before optimizer extraction. Do not infer latest K-stock rounds from live config alone.

### Crypto Bot

Keep the package shape of `src/crypto_trader` initially because it is already close to the target. Move it under `trading/crypto_trader/src/crypto_trader` with minimal import changes.

Important preservation points:

- Click CLI commands
- deployment manifest preflight
- optimizer contract hashes
- phase state contracts
- `LIVE_PARITY_PROFILE`
- portfolio round output mounts
- Hyperliquid exchange adapter
- strategy JSON configs

Crypto can be the first bot to adopt shared contracts because it already has strong contract metadata in the optimizer path.

### Trading Assistant

Move assistant packages into `packages`, but keep the assistant logically separate from live trading runtimes.

The assistant should own orchestration and evidence verification, not trading. Live runtimes should be able to run without importing assistant control-plane code. They only need shared contracts and deployment validation helpers.

## CI and Verification Gates

Create a CI matrix with these jobs:

| Job | Purpose |
| --- | --- |
| repo-bootstrap | verify git remote, CI metadata, artifact policy, uv workspace skeleton, and lockfile resolution |
| workspace-lock | run the root `uv` lock/sync audit used by local development and Docker builds |
| workspace-imports | import every package and runtime entrypoint |
| lint-type | run root Ruff checks and package-appropriate type checks with a checked baseline only for approved legacy findings |
| contracts | validate schemas, strategy plugin contracts, deployment metadata, and promotions |
| baselines | verify latest round no-drift against frozen baseline |
| live-configs | verify enabled live strategies map to approved promotions |
| optimizer-smoke | run deterministic optimizer fixture tests |
| backtest-smoke | run OOS smoke tests for promoted strategies |
| parity | run decision parity fixture checks |
| assistant | run monthly workflow contract and evidence checks |
| affected-images | compute which bot images must rebuild from changed package/config/contract paths |
| docker | build the three bot images |
| deploy-smoke | start artifact-only or dry-run mode for each bot image |

CI mechanics:

- Root CI uses the committed `uv.lock`.
- Affected-image detection maps changes in `packages/trading_contracts`, `packages/trading_config`, `packages/trading_deployment`, and `packages/trading_instrumentation` to all three bot images.
- Changes in a bot package, bot config, bot deployment files, or bot strategy contracts rebuild only that bot plus any shared images that depend on it.
- Changes in assistant-only packages run assistant gates but do not rebuild live bot images unless shared contracts or generated promotion/deployment artifacts changed.
- Ruff is the default root linter. Mypy or pyright can be package-scoped where strict typing is already realistic; legacy packages may start with a checked baseline and no-new-errors policy.
- Root pytest should be split by package and marker so parity smoke, optimizer smoke, assistant monthly checks, and image smoke can be run independently.

Merge should be blocked by:

- baseline drift without an approved migration note
- live config without promotion evidence
- deployment metadata missing commit/config/contract hashes
- strategy plugin contract mismatch
- decision parity failure for promoted strategies
- enabled strategy missing latest round mapping
- workspace lock drift or unresolved dependency conflicts
- affected bot image not rebuilt after a relevant shared-package or deployment change

## Acceptance Matrix

This matrix is the implementation contract. A phase is not accepted because files moved or images build; it is accepted only when the evidence below is present and the pass criteria are met.

### Acceptance Evidence Quality Rule

The main false-completion risk is PASS-looking evidence that was synthesized by verification helpers without exercising the runtime, image, CI, or cutover path it claims to prove. A helper may orchestrate a gate for repeatability, but it must not be the behavioral proof.

For any gate that claims runtime or deployment behavior, PASS evidence must include the command that exercised the real surface, its exit code, bounded stdout/stderr, input hashes, output artifact hashes, and the normalized report path. The command recorded as behavioral evidence must be one of the actual bot entrypoints, in-image startup/import/help commands, CI workflow commands, or deployment/cutover commands. Direct calls into private metadata builders, direct JSON fabrication, helper-only generators, static path scans, or return-code-only wrappers are preflight evidence at most.

When a helper normalizes evidence, it must keep the raw runtime artifact and prove that the raw artifact came from the real command. The verifier must also include fail-closed probes for the specific loopholes it is closing, such as helper-emitted deployment metadata, broker-login output during image smoke, unsynced CI jobs, placeholder rollback records, grouped acceptance rows, and omitted early bootstrap gates.

This rule is separate from the `legacy source snapshots` policy. Porting useful logic, fixtures, and contracts from the reference repos into monorepo-owned paths is necessary, but it does not close a gate unless the resulting verifier proves the promoted runtime behavior through the real execution path.

| ID | Gate | Scope | Existing evidence to reuse | New or migrated artifact | Required check | Pass criteria | Blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A00 | Repository and tooling bootstrap | root repo | root is not yet a git repository; assistant already has workspace-check scripts | `.git`, remote, CI stub, `pyproject.toml` workspace skeleton, `uv.lock`, artifact storage decision, bootstrap decision record | `python tools/check_repo_bootstrap.py`; `python tools/workspace_lock_check.py`; `python tools/detect_affected_images.py --explain-only` | root git identity is usable; CI and branch protection are selected; declared required branch checks are present in both CI and the bootstrap verifier; artifact storage classes are assigned; uv lock resolves planned packages for Python 3.12 live images; affected-image mapping is explainable | baseline freeze, deployment metadata, image builds |
| A0 | Workspace starting-state inventory | root repo | pre-bootstrap source material is under `docs/` and `legacy source snapshots` | `docs/migration_inventory.md` or baseline inventory section in `baseline_index.json` | `python tools/freeze_optimization_baselines.py --inventory-only` | all source roots, pyprojects, requirements, configs, compose files, contracts, artifact roots, duplicate-round policy, and archive-resolution policy are listed with hashes | any code move |
| A1 | Python/package metadata | all packages | IBKR pyproject, crypto pyproject, assistant package pyprojects, K-stock requirements/pytest.ini | root workspace `pyproject.toml`; K-stock package pyproject | `python tools/workspace_import_smoke.py --all-packages` after `uv sync --frozen` or wrapper-verified equivalent | all package entrypoints import; K-stock no longer relies on bare repo root path hacks; live bot package metadata does not declare assistant control-plane/backtest packages unless an explicit runtime need is documented | package extraction |
| A2 | IBKR latest round freeze | all active IBKR strategies and portfolio-synergy outputs | IBKR `backtests/output/**/rounds_manifest.json`; existing baseline regeneration helpers; stock/momentum/swing/regime raw backtest data | `backtests/baselines/ibkr/...`; `baseline_index.json`; `backtests/data_portability_manifest.json` | `python tools/verify_backtest_data_portability.py --bot ibkr`; `python tools/verify_latest_round_no_drift.py --bot ibkr --baseline backtests/baselines/baseline_index.json --strict` | active non-archived latest rounds match by canonical hash and key metrics; duplicate round entries are disambiguated by `latest_round` plus timestamp; archived/cleaned artifacts are resolved or fail closed; ignored fields are explicit; required raw backtest data is present under monorepo-owned paths, hashed by the data portability manifest, and excluded from live Docker contexts | optimizer extraction, strategy moves |
| A3 | Crypto latest round freeze | momentum, trend, breakout | crypto strategy `rounds_manifest.json`, optimized configs, `parity_alignment.json`; candle/funding/asset metadata used by optimizer rounds | `backtests/baselines/crypto/...`; `backtests/data_portability_manifest.json` | `python tools/verify_backtest_data_portability.py --bot crypto`; `python tools/verify_latest_round_no_drift.py --bot crypto --baseline backtests/baselines/baseline_index.json --strict` | strategy-level round 3 artifacts match canonical hashes, contract hashes, profile hashes, and key metrics; required crypto historical data is present under monorepo-owned paths, hashed by the data portability manifest, and excluded from live Docker contexts | crypto package move, optimizer extraction |
| A4 | Crypto portfolio bundle integrity | portfolio round 3 | `output/portfolio/round_3/deployment_manifest.json`, `parity_alignment.json`, recommended configs | restored or superseded `output/portfolio/rounds_manifest.json`; portfolio promotion manifest | `python tools/verify_live_config_promotions.py --bot crypto --require-portfolio-bundle` | deployment manifest references exist; parity alignment is matched; live config paths equal approved bundle paths | crypto deployment cutover |
| A5 | K-stock latest baseline decision | KALCB, OLR, portfolio synergy | K-stock optimization configs and scripts; restored `data/backtests/output` artifacts; frozen accepted baseline evidence | restored/regenerated K-stock baseline artifacts and provenance note; `backtests/data_portability_manifest.json` | `python tools/verify_backtest_data_portability.py --bot k_stock`; `python tools/verify_latest_round_no_drift.py --bot k_stock --baseline backtests/baselines/baseline_index.json --strict` | restored latest accepted artifacts validate, or regenerated replacements have frozen command, source SHAs, and explicit approval; selected accepted evidence under `backtests/baselines/k_stock` is hashed as the source-control authority; any large restored output tree is optional local/object-storage audit evidence and cannot be required by CI or deployment images | K-stock optimizer extraction, K-stock live promotion |
| A6 | Live config alignment: IBKR | enabled IBKR strategies | `config/strategies.yaml`; existing ALCB and swing parity tests | IBKR promotion manifests and generated effective runtime config | `python tools/verify_live_config_promotions.py --bot ibkr --require-latest-round --require-effective-configs --strict` | every enabled strategy maps to latest frozen optimized config with a non-draft promotion state appropriate to the gate or an approved overlay; disabled/non-promoted strategies are tagged; strict mode rejects draft or draft-derived canonical promotion manifests; generated effective config includes materialized runtime values and a reproducible hash | IBKR live image cutover |
| A7 | Live config alignment: crypto | live strategy configs and portfolio config | `config/live_config.example.json`, `config/strategies/*.json`, portfolio round 3 bundle | crypto promotion manifests and generated effective runtime config | `python tools/verify_live_config_promotions.py --bot crypto --require-latest-round --require-effective-configs --strict` | live config points to portfolio round 3 supersession evidence or an approved successor bundle; strategy config hashes match promotion evidence in a non-draft promotion state; strict mode rejects draft or draft-derived canonical promotion manifests; generated effective config includes materialized runtime values and a reproducible hash | crypto live image cutover |
| A8 | Live config alignment: K-stock | KALCB, OLR, OLR/KALCB portfolio | `config/kalcb.yaml`, `config/optimization/*.yaml`, `strategy_kalcb/config.py`, `backtests/strategies/kalcb/phase_candidates.py`, bridge contract, resource plan | K-stock promotion manifests and generated effective runtime config | `python tools/verify_live_config_promotions.py --bot k_stock --require-latest-round --require-effective-configs --strict` | KALCB/OLR config values match latest frozen artifacts with a non-draft promotion state appropriate to the gate or approved overlays; `frontier.size` is parsed as 103 from live config, strategy default, optimizer base mutation, and deployment universe evidence; strict mode rejects draft or draft-derived canonical promotion manifests; generated effective config includes materialized runtime values and a reproducible hash | K-stock paper/live cutover |
| A9 | Strategy plugin contracts | assistant bridges for IBKR, K-stock, crypto | assistant contracts, K-stock generated bridge contract, crypto bridge contracts | canonical contracts under `contracts/strategy_plugins` | `python tools/verify_strategy_plugin_contracts.py --all` plus runtime metadata path scan | mature contracts include live repo path, full commit SHA, adapter path, config schema, decision API, telemetry schemas, fixtures, and hashes; bot runtime/deployment metadata defaults and assistant validation defaults resolve to canonical `contracts/strategy_plugins/...` paths, not package-local `trading_assistant_backtest/contracts/...` or `legacy source snapshots` | assistant adoption gates |
| A10 | Decision parity | promoted strategies and families | IBKR parity fixtures/tests, crypto parity traces, K-stock replay fixtures | normalized parity reports in baseline/promotion evidence | `python tools/run_decision_parity_matrix.py --promoted-only` | every promoted strategy/family has a parsed PASS report for signals, filters, entries, exits, stops, sizing, risk caps, and order intent, or an explicit non-promotion status; fixture existence alone does not satisfy this gate | live promotion |
| A11 | Assistant workspace checks | assistant control/data/backtest | existing `tools/run_workspace_checks.py` tiers | migrated checks under root workspace | `python tools/run_workspace_checks.py deployment-gate` | assistant import, CLI, structure, monthly, data, backtest, loop, verifier, validation, and deployment package tiers pass or skip only for documented local-only data; package-local test runs must resolve workspace dependencies such as `trading_contracts` without ad hoc manual path injection | assistant package migration |
| A12 | Shared contracts compatibility | all legacy artifacts | assistant contract models, crypto optimizer contracts, K-stock bridge/deployment metadata, IBKR round manifests | `packages/trading_contracts` schemas and readers | `trading-contracts validate --all-known-reference-artifacts`; `python tools/verify_dependency_boundaries.py` | existing artifacts validate unchanged or through named compatibility adapters; `trading_contracts` owns canonical models without importing `trading_assistant_backtest`; assistant/backtest packages depend on `trading_contracts`; duplicate assistant-local canonical model classes are removed or reduced to compatibility aliases with parity tests | shared package adoption |
| A13 | Optimizer compatibility | three phased optimizer implementations | IBKR/K-stock/crypto `PhaseRunner` and plugin protocols | `packages/trading_optimizer` adapter layer plus smoke fixtures | `python tools/verify_optimizer_compatibility.py --bot all --fixture-set smoke` | legacy runner and shared adapter runner execute against the same smoke fixture inputs and produce identical cumulative mutations, gate decisions, selected candidates, and canonical round outputs; PASS evidence includes input hashes, command records, runner source hashes, output hashes, and compared payloads; latest-round no-drift and archived/frozen-output canonicalization are preflight evidence only and cannot close this gate | deleting duplicated optimizer internals |
| A14 | Backtest integrity invariants | all promoted backtests | `docs/strategy-implementation-lessons.md`; existing artifact hygiene and parity tests; data portability manifest | shared invariant test suite plus thin-adapter audit plus `artifacts/validation/backtest_data_portability_report.json` | `python tools/run_backtest_integrity_matrix.py --promoted-only` | completed-bar, next-bar fill, broker path, MTM risk, net/gross accounting, shared-capital portfolio, diagnostics, timestamp, artifact hygiene, and stress gates pass as named hard-fail checks with per-invariant command/evidence records; validation-matrix errors are gate failures, not advisory fields; every promoted backtest is proven to be a replay/simulation adapter over the production decision core, with no separate backtest-owned decision implementation except documented transitional/non-promoted cases; data reproduction and replay evidence reports include the data portability manifest/report as hard evidence and fail closed with explicit `missing_artifact_paths` when any required artifact is absent; adapter path existence or repeated broad evidence notes do not satisfy this gate | accepting regenerated performance |
| A15 | Deployment image build | three bots | existing compose/Dockerfiles where present | `trading/*/Dockerfile`, `deployments/*/docker-compose.yml`, `.dockerignore` or equivalent build-context rules, image dependency reports | `python tools/build_bot_image.py --bot all --emit-dependency-reports`; docker build matrix for all bot images; in-image runtime import/help/startup smoke | each image builds independently on the approved Python live-runtime base or a documented exception; image smokes run without broker secrets and do not attempt broker/exchange/KRX/IBKR authentication or emit credential/login failure output; images include only required bot/shared packages; assistant control-plane/backtest packages and package-local contract trees are absent from live bot images unless explicitly approved; nested `legacy source snapshots`, bot-local `output/`, bot-local `backtests/output/`, bot-local raw historical data, crypto historical data, and bot-local `data/backtests/output/` are excluded from image contexts; static Dockerfile/dependency-report verification or return-code-only smoke evidence is only preflight evidence and does not satisfy this gate | VPS deployment |
| A16 | Deployment metadata | three bot runtimes | K-stock metadata emitter, crypto live metadata, IBKR registry artifact | normalized deployment metadata emitted by bot runtime startup paths | `python tools/verify_deployment_metadata.py --bot all` | full commit, actual clean worktree state, image version, materialized config hash, promotion hash, canonical strategy plugin contract hash, strategy version, telemetry schema, and runtime entrypoint are present in runtime-emitted metadata; PASS evidence includes each actual bot entrypoint artifact-only/dry-run/paper startup command, its return code, bounded output, normalized metadata path, and raw runtime artifact hash; the verifier rejects locally fabricated metadata, hard-coded clean-worktree assertions, artifacts produced only by `tools/generate_runtime_deployment_metadata.py`, and matrix/helper commands such as `tools/run_runtime_deployment_metadata_matrix.py` when they are the behavioral proof instead of orchestration around real bot startup commands; auto-update services such as Watchtower cannot bypass fail-closed validation | live trading enablement |
| A17 | Final refactor acceptance | entire monorepo | all previous gates | `migration_acceptance_report.json`; `backtests/data_portability_manifest.json`; `artifacts/validation/backtest_data_portability_report.json` | `python tools/verify_refactor_acceptance.py --bot all --strict`; `python tools/verify_backtest_data_portability.py --bot all`; `python tools/run_workspace_checks.py deployment-gate`; `python -m pytest packages/trading_assistant_data/tests/test_source_layer_equivalence.py`; `python tools/check_workspace_structure.py --layout final` | latest phased auto-optimization artifacts are unchanged and their required historical data/evidence is present, hashed, and monorepo-owned; materialized live configs align with latest approved backtest configs; go-forward `trading_assistant_data` source adapters prove decision-row equivalence to bot-owned download layers for fetch, pagination, timestamp, merge, dedupe, and aggregation semantics; every acceptance matrix row A00-A17 is represented as an individual row-level PASS record with evidence paths, commands, and skip reasons, not grouped labels or omitted early bootstrap rows; final root structure includes required shared packages or the missing-package rows keep A17 red; the strict verifier reconciles required finite-checklist items with the matrix and keeps A17 red when gate-closing work remains open without an explicit non-applicability/blocker record; strict fixture gates are present in CI, run under a locked workspace environment using `uv sync --frozen --all-packages` or `uv run --frozen` instead of raw unsynced Python on clean runners, and are required before merge; A17 rejects A13 archived-output-only compatibility, A14 advisory/static invariant proof or green data/replay reports with missing required artifact paths, A15 preflight-only image evidence or broker-secret startup side effects, A16 locally generated/helper-emitted metadata or verifier output that lacks real runtime-emission command/hash evidence, no-drift evidence without a passing data portability report, and cutover records with placeholder rollback tags or missing previous compose/config/hash evidence; all relevant source assets have been ported into monorepo-owned paths with hashes/provenance; no committed runtime, test, verification, tooling, deployment metadata, contract, manifest, artifact index, acceptance report, or documentation artifact points at `legacy source snapshots`; no nested `legacy source snapshots` or unignored local runtime artifacts remain under production package paths | refactor completion |

## Finite Implementation Checklist

Track this checklist directly. Do not add broad rolling tasks; split any new discovery into a finite item with an owner and an acceptance gate.

- [x] B1. Initialize the root git repository and configure the real remote.
- [x] B2. Choose CI provider, add the first workflow stub, and define branch protection expectations.
- [x] B3. Decide artifact storage classes: source control, Git LFS, object storage, and generated local-only outputs.
- [x] B4. Create the root `uv` workspace skeleton and generate or audit `uv.lock`.
- [x] B5. Add bootstrap scripts for repo checks, workspace lock checks, import smoke, and affected-image detection.
- [x] B6. Record bootstrap decisions in `docs/repo-bootstrap-decisions.md`.
- [x] B7. Reconcile `[tool.trading_agent.required-branch-checks]`, CI workflow jobs, and `tools/check_repo_bootstrap.py` so bootstrap cannot pass with missing required checks.
- [x] 1. Generate source inventory for `trading/ibkr_trader`, `trading/k_stock_trader`, `trading/crypto_trader`, and `packages/trading_assistant`.
- [x] 2. Record all existing pyprojects, requirements, pytest configs, scripts, Dockerfiles, compose files, contracts, live configs, and artifact roots in the inventory.
- [x] 3. Freeze IBKR latest non-archived round manifests and latest optimized configs into `backtests/baselines/ibkr`, using `latest_round` plus timestamp disambiguation for duplicate round numbers.
- [x] 4. Extend the existing IBKR baseline fixture shape into the root baseline index, including `archive_path`/`cleanup_note` artifact resolution.
- [x] 5. Freeze crypto strategy-level latest round 3 artifacts for momentum, trend, and breakout.
- [x] 6. Freeze crypto portfolio round 3 recommended strategy configs, recommended portfolio config, deployment manifest, parity alignment, and lineage correction.
- [x] 7. Restore or regenerate crypto `output/portfolio/rounds_manifest.json`, or replace the deployment manifest with an approved successor reference.
- [x] 8. Restore K-stock `data/backtests/output` from the authoritative artifact source.
- [x] 9. Not required: K-stock artifacts were restored, so no regenerated-artifact fallback was used.
- [x] 10. Verify the current KALCB `frontier.size` source values are 103 in `config/kalcb.yaml`, `strategy_kalcb/config.py`, and `backtests/strategies/kalcb/phase_candidates.py`.
- [x] 11. Create final root workspace metadata and committed `uv.lock` policy.
- [x] 12. Adapt IBKR package metadata from `trading/ibkr_trader/pyproject.toml`.
- [x] 13. Adapt crypto package metadata from `trading/crypto_trader/pyproject.toml`.
- [x] 14. Move assistant package metadata from the three assistant package pyprojects without changing behavior.
- [x] 15. Create K-stock package metadata from `requirements.txt`, `pytest.ini`, runtime entrypoints, and test layout.
- [x] 15a. Remove assistant control-plane/backtest dependencies from live bot package metadata, or document an approved runtime-only exception with image dependency evidence.
- [x] 16. Add import compatibility shims for old IBKR, K-stock, crypto, and assistant paths.
- [x] 17. Move assistant packages into `packages/` and run existing assistant workspace checks.
- [x] 18. Seed `packages/trading_contracts` by wrapping assistant contract models and adding legacy adapters.
- [x] 18a. Lift or locally wrap canonical contract models so `trading_contracts` no longer imports `trading_assistant_backtest`.
- [x] 18b. Make assistant/backtest packages import canonical models from `trading_contracts`; replace duplicate assistant-local canonical model classes with compatibility aliases or remove them after parity tests pass.
- [x] 19. Generate canonical JSON schemas into `contracts/schemas`.
- [x] 20. Move strategy plugin contracts into `contracts/strategy_plugins` and update contract payload paths to monorepo locations.
- [x] 20a. Update bot runtime/deployment metadata defaults and verifier coverage so strategy plugin contract paths resolve to canonical `contracts/strategy_plugins/...` locations.
- [x] 20b. Retire active package-local strategy plugin contract defaults under `packages/trading_assistant_backtest/contracts`; assistant validation commands must default to root `contracts/strategy_plugins/...`.
- [x] 20c. Remove or archive package-local assistant deployment metadata that still points at `legacy source snapshots`; verifiers must scan package-local deployment metadata as well as strategy plugin contracts.
- [x] 21. Add `StrategyPromotionManifest` schema and validators.
- [x] 22. Generate draft promotion manifests for every enabled IBKR strategy.
- [x] 23. Generate draft promotion manifests for crypto momentum, trend, breakout, and portfolio round 3 bundle.
- [x] 24. Generate draft promotion manifests for KALCB, OLR, and K-stock portfolio synergy after baseline restoration.
- [x] 25. Implement generated effective live config artifacts for each bot.
- [x] 26. Upgrade generated effective live config artifacts to include materialized runtime config values and prove they match current live behavior or approved overlays.
- [x] 26a. Extend live-config verifiers so `--strict` requires materialized effective configs and parses KALCB `frontier.size=103` from live config, strategy default, optimizer base mutation, and deployment universe.
- [x] 26b. Separate draft promotion generation from canonical approved promotion manifests; `--strict` must reject draft promotion states for runtime/live acceptance and config generation must not delete approved canonical manifests by copying drafts over them.
- [x] 27. Move crypto into `trading/crypto_trader` with CLI compatibility.
- [x] 28. Move IBKR into `trading/ibkr_trader` with runtime CLI compatibility.
- [x] 29. Move K-stock into `trading/k_stock_trader` with runtime/session CLI compatibility.
- [x] 29a. Remove nested bot-local `legacy source snapshots` copies from production package paths.
- [x] 29b. Ignore and package-exclude all bot-local generated output trees, including `output/`, `backtests/output/`, and `data/backtests/output/`; keep canonical frozen evidence under `backtests/baselines`.
- [x] 30. Add deployment metadata validation to all three runtimes.
- [x] 30a. Replace the current matrix/helper-emitted metadata path in `tools/verify_deployment_metadata.py --bot all` with runtime metadata validation that requires metadata written by actual bot artifact-only/dry-run/paper startup commands, actual clean-worktree evidence, image version, materialized config hash, promotion hash, canonical contract hash, strategy version, telemetry schema, runtime entrypoint, raw runtime artifact hashes, and auto-update fail-closed evidence; metadata written only by `tools/generate_runtime_deployment_metadata.py`, `tools/run_runtime_deployment_metadata_matrix.py`, or any helper that calls metadata emitters directly cannot satisfy A16 even when the recorded source names a runtime file.
- [x] 31. Prove fail-closed startup checks for missing/stale promotion, contract, config, and metadata evidence in live/paper modes by exercising the actual bot startup paths before any trading loop can run.
- [x] 32. Wrap existing optimizer runners behind the shared optimizer adapter interface.
- [x] 32a. Replace archived/frozen-output optimizer smoke evidence in `tools/verify_optimizer_compatibility.py --bot all --fixture-set smoke` with same-input fixture executions of legacy bot runners and the shared optimizer adapter, including cumulative mutations, gate decisions, selected candidates, canonical round outputs, input hashes, command records, runner source hashes, output hashes, and compared payloads.
- [x] 33. Run optimizer compatibility fixtures before moving shared optimizer internals.
- [x] 34. Defer shared optimizer internals extraction until compatibility fixtures and baseline no-drift gates are green; the current accepted shared layer is the wrapper/adapter boundary.
- [x] 35. Add shared backtest integrity invariant tests from `docs/strategy-implementation-lessons.md`.
- [x] 35a. Produce the per-strategy thin-adapter audit for IBKR, crypto, and K-stock; convert promoted backtests to production decision-core replay or mark transitional mirrors as non-promoted with parity, no-drift, and deletion evidence.
- [x] 35b. Replace advisory/static backtest-integrity evidence in `tools/run_backtest_integrity_matrix.py --promoted-only` with hard-fail named invariant checks for completed-bar policy, next-bar fill, broker path, MTM risk, net/gross accounting, shared capital, diagnostics, timestamp hygiene, artifact hygiene, and stress gates; validation-matrix errors must fail the gate, each invariant needs specific command/evidence records, data reproduction/replay reports must fail closed on missing required artifact paths, and the tool must consume the thin-adapter audit.
- [x] 35c. Replace the current fixture-exists smoke wrapper in `tools/run_decision_parity_matrix.py --promoted-only` with parsed PASS/FAIL parity reports for signals, filters, entries, exits, stops, sizing, risk caps, and order intent across IBKR, crypto, and K-stock fixtures.
- [x] 35d. Run the decision parity matrix for every promoted strategy/family before accepting any promoted backtest as production-core replay.
- [x] 36. Preserve venue-specific fill timing, session, fee, slippage, funding, and broker semantics as explicit inputs.
- [x] 37. Add per-bot Dockerfiles and per-VPS compose files under `deployments/`.
- [x] 37a. Add per-image `.dockerignore` or equivalent build-context checks so `legacy source snapshots`, bot-local `output/`, bot-local `backtests/output/`, and bot-local `data/backtests/output/` cannot enter deployable images.
- [x] 37b. Implement `tools/build_bot_image.py --bot all --emit-dependency-reports` or a wrapper-verified equivalent that emits per-image dependency reports and enforces the A15 build-context exclusions.
- [x] 37c. Replace or quarantine legacy Dockerfiles that use non-approved Python bases, pip-only dependency installs, or package-local contract copies; the image verifier must scan every deployable Dockerfile, not only the new root targets.
- [x] 38. Build all three images and run clean in-image import/help smoke records for each image; `tools/build_bot_image.py --bot all --emit-dependency-reports` must include real `docker build` evidence, no broker/exchange/KRX/IBKR authentication attempts, no credential/login failure output, and more than return-code-only preflight evidence.
- [x] 39. Run artifact-only, dry-run, or installed-entrypoint image startup smoke for each bot image without broker secrets and without import-time broker login side effects.
- [x] 40. Re-run decision parity matrix for every promoted strategy/family after image packaging and before live cutover.
- [x] 41. Final re-run of the live config promotion verifier for all bots with `--require-latest-round --require-effective-configs --strict` and final deployment approval states.
- [x] 42. Final re-run of the latest round no-drift verifier for all bots with the frozen baseline index.
- [x] 43. Final run of the assistant deployment gate and validation matrix after path updates.
- [x] 44. Retain only documented compatibility shims still exercised by runtime/test paths; remove them after dependency-free evidence exists, rather than removing active import surfaces prematurely.
- [x] 45. Remove every committed dependency or path reference to `legacy source snapshots` after porting, including runtime, tests, migration/validation tools, manifests, contracts, deployment metadata, artifact indexes, acceptance reports, docs, and nested bot-local `legacy source snapshots` directories.
- [x] 45a. Update baseline/no-drift, validation tools, data requirement manifests, tests, package-local metadata, and docs so final acceptance consumes monorepo-owned baselines/evidence with captured hashes/provenance instead of requiring or pointing at `legacy source snapshots`.
- [x] 45b. Port and hash the historical backtest data needed to reproduce latest phased auto-optimization rounds: IBKR stock/momentum/swing/regime raw data under `trading/ibkr_trader/backtests/*/data/raw`, crypto candle/funding/asset metadata under `trading/crypto_trader/data`, K-stock accepted evidence under `backtests/baselines/k_stock`, and a verified `backtests/data_portability_manifest.json`; keep large generated output trees out of source-control authority and live Docker contexts.
- [x] 45c. Prove `trading_assistant_data` go-forward source adapters are faithful to bot-owned download layers for decision rows, including Hyperliquid duplicate/overlap precedence, IBKR bar normalization, KIS intraday parsing/aggregation, and LRS daily table export.
- [x] 46. Produce `migration_acceptance_report.json` with explicit A00-A17 row statuses, evidence paths, commands, and skip reasons; the current report exists, but must be regenerated after A15, A16, and A17 strictness is corrected.
- [x] 46a. Update `tools/verify_refactor_acceptance.py --bot all --strict` so it validates every acceptance matrix row as an individual A00-A17 record, runs `python tools/run_workspace_checks.py deployment-gate`, scans Python, tests, data, contract metadata, manifests, artifact indexes, deployment metadata, acceptance reports, migration/verification tooling, and docs for `legacy source snapshots` path references, rejects legacy snapshot provenance fields such as `source_repo`, `source_reference`, and `archived_source_path` when they point outside monorepo-owned artifacts, enforces zero remaining matches after porting, checks required final shared package structure, reconciles required finite-checklist items with matrix status, rejects A13 archived-output-only compatibility, A14 advisory/static invariant proof or green data/replay reports with missing required artifact paths, A15 preflight-only image evidence or broker-secret startup side effects, A16 local-generator/helper-emitted metadata or missing runtime-emission command/hash evidence, unsynced CI fixture gates, and placeholder cutover rollback records, and fails when a required row is omitted, grouped, or represented only by a smoke/static wrapper.
- [x] 46b. Add final root `README.md` and `artifacts/` policy directory or placeholder, then make `python tools/check_workspace_structure.py --layout final` pass.
- [x] 46c. Add CI and branch-protection coverage for final acceptance gates once they are no longer local-only: deployment gate, decision parity, optimizer compatibility, backtest integrity, deployment metadata, image builds/import smoke, and strict refactor acceptance; package-dependent jobs must run under `uv sync --frozen --all-packages` or `uv run --frozen` instead of raw Python on unsynced clean runners.
- [x] 47. Replace placeholder one-bot-at-a-time VPS cutover records with real per-bot candidate image tags and artifact-only/dry-run or paper startup evidence before any live enablement.
- [x] 48. For each bot cutover, record the real rollback image tag, previous compose file and hash, previous live config hashes, and tested restore command before enabling paper/live mode; placeholder tags such as `previous-production-*` do not satisfy A17.
- [x] 49. Remove `legacy source snapshots` snapshots from the committed repo after all relevant assets have been ported and provenance has been captured in monorepo-owned artifacts; any archival copy must live outside the repository and must not be referenced by repo artifacts.

## Deployment Model

Each VPS should receive only the image and compose stack for its bot.

IBKR VPS:

- `deployments/ibkr/docker-compose.yml`
- image: `trading-agent/ibkr-trading`
- services: Postgres, IB Gateway, runtime, relay, dashboard, watchdog

K-stock VPS:

- `deployments/k_stock/docker-compose.yml`
- image: `trading-agent/k-stock-trader`
- services: OMS, runtime, dashboard, Postgres, optional PCIM/research profiles

Crypto VPS:

- `deployments/crypto/docker-compose.yml`
- image: `trading-agent/crypto-trader`
- services: Postgres, trader, data-refresh, dashboard, Watchtower, optional tool profiles

Crypto Watchtower assumption:

- Watchtower, if retained, must not make `latest`-tag auto-updates the deployment authority.
- Auto-updated images must still pass runtime fail-closed validation for promotion manifest hash, effective config hash, strategy plugin contract hash, deployment manifest evidence, and image label metadata before paper/live trading starts.
- If that cannot be guaranteed, disable Watchtower for paper/live VPS profiles and use explicit image-tag promotion instead.

Assistant/control plane:

- Can run as a separate batch/control image or local operator workflow.
- Should write manifests and read artifacts through declared paths or artifact storage.
- Should not be required for live runtime startup except when validating previously generated evidence.

Per-bot rollback rule:

- Before enabling paper or live mode on a VPS, record the previous image tag, compose file hash, live config hash, promotion manifest hash, deployment metadata path, and restore command.
- Keep the previous image and compose file deployable until the new bot has passed at least one full artifact-only/dry-run or paper session gate.
- Rollback never rewrites baseline artifacts. It only restores runtime image/config/deployment state to the last approved promotion.

## Artifact and Data Policy

Use source control for:

- schemas
- promotion manifests
- small baseline indexes
- strategy plugin contracts
- deployment templates
- deterministic fixtures

Use artifact storage or Git LFS for:

- large backtest outputs
- raw/canonical market data
- replay bundles
- large parquet/csv files
- historical telemetry

Generated files should have clear ownership:

- Assistant generates monthly run manifests and evidence checks.
- Data package generates data bundle manifests.
- Optimizer generates round artifacts.
- Config package generates effective live configs.
- Runtime generates deployment metadata.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| Optimizer drift during extraction | Freeze baselines first, wrap old runners before deleting duplication, compare canonical hashes and metrics |
| Live config silently diverges | Promotion manifests become mandatory, runtime emits and validates config hash |
| Assistant becomes tightly coupled to live bots | Assistant imports shared contracts only; strategy interaction goes through plugin contracts |
| K-stock artifacts are missing | Restore from production artifact store or rerun frozen workflow before migration proceeds |
| Root repo identity is missing | Phase -1 initializes git, remote, CI, branch protection, and bootstrap checks before deployment metadata is trusted |
| Workspace dependency resolution drifts | Use one committed `uv.lock`, CI lock checks, and per-image dependency subsetting from the workspace |
| Docker images become too large | Per-bot dependency groups and multi-stage builds |
| Auto-update bypasses promotion evidence | Watchtower or similar services must still pass runtime fail-closed validation, or they are disabled for paper/live profiles |
| Shared packages become dumping grounds | Enforce ownership rules and keep venue behavior bot-specific |
| Strategy refactor changes decisions | Require parity fixtures and latest round no-drift checks for every extraction |

## Definition of Done

The migration is complete when:

- The root repo has real git identity, remote, CI, branch protection expectations, artifact storage policy, and committed `uv.lock`.
- The repo has one Python workspace with all shared packages, assistant packages, bot packages, bot-owned strategies, optional shared strategy primitives, contracts, deployments, and tests.
- The three bot images build independently and pass in-image runtime import smoke; static Dockerfile/dependency-report checks are not build evidence.
- Each bot can run artifact-only, dry-run, or paper mode from its own VPS compose stack.
- All promoted strategies have a `StrategyPromotionManifest`.
- Every enabled live strategy maps to the latest approved optimized config.
- Latest phased auto-optimization baselines are unchanged or have explicitly approved migration notes.
- Decision parity reports exist and pass for all promoted strategies.
- Deployment metadata includes git commit, image version, strategy version, config hash, promotion hash, and contract hash.
- Assistant monthly workflow checks run against the monorepo paths.
- No committed runtime, test, active verification, final-acceptance tooling, contract, manifest, artifact index, deployment metadata, acceptance report, or repo documentation path references `legacy source snapshots`; all provenance is represented by monorepo-owned evidence and hashes.
- `python tools/run_workspace_checks.py deployment-gate` passes, including package-local assistant/backtest test tiers after the `trading_contracts` dependency move.
- `python -m pytest packages/trading_assistant_data/tests/test_source_layer_equivalence.py` passes, proving the go-forward data adapters remain faithful to bot-owned download behavior for decision rows.
- `python tools/verify_latest_round_no_drift.py --bot all --baseline backtests/baselines/baseline_index.json --strict` passes.
- `python tools/verify_live_config_promotions.py --bot all --require-latest-round --require-effective-configs --strict` passes.
- `python tools/verify_refactor_acceptance.py --bot all --strict` emits an acceptance report with explicit A00-A17 row statuses, evidence paths, commands, skip reasons, and all matrix rows passing.

## Recommended First Pull Requests

1. Bootstrap repository identity and workspace tooling: git remote, CI stub, artifact policy, uv workspace skeleton, lock audit, and affected-image detection.
2. Add inventory and baseline freeze tooling, explicitly reusing the existing IBKR baseline-regeneration shape.
3. Preserve restored or superseded artifact evidence: K-stock `data/backtests/output` and crypto `output/portfolio/rounds_manifest.json`.
4. Add `trading_contracts` by lifting assistant contract models and adding legacy artifact readers.
5. Add promotion manifest schema and live config promotion verifier, including the KALCB `frontier.size=103` preservation check.
6. Move crypto into `trading/crypto_trader` with minimal behavior changes and preserve portfolio round 3 deployment-manifest preflight.
7. Move IBKR into `trading/ibkr_trader` with runtime compatibility entrypoints and preserve existing parity/baseline tests.
8. Create K-stock package metadata, freeze restored baselines, then move K-stock into `trading/k_stock_trader`.
9. Move assistant packages into `packages` with contract-only bot coupling and preserve `tools/run_workspace_checks.py` tiers.
10. Extract shared optimizer kernel behind compatibility adapters only after no-drift gates are green.
11. Add Docker build matrix, deployment metadata gates, final acceptance verifier, per-VPS compose outputs, and rollback scripts.
