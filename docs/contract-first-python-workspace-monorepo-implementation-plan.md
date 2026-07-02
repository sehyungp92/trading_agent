# Contract-First Python Workspace Monorepo Implementation Plan

## Purpose

This plan describes how to migrate the existing trading systems into one contract-first Python workspace monorepo with many packages and three independently deployable bot images:

- IBKR bot, currently in `_references/trading`
- KIS/KRX bot, currently in `_references/k_stock_trader`
- Hyperliquid crypto bot, currently in `_references/crypto_trader`

The trading assistant, currently in `_references/trading_assistant_agent`, should also live in the same monorepo as first-class packages. It should not become a direct in-process dependency of the live trading runtimes. The assistant remains a control/data/backtest plane connected through versioned manifests, artifact indexes, deployment metadata, and parity reports.

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
- The reference checkout does not currently include `data/backtests/output`, so K-stock latest round artifacts must be restored or rebuilt before the baseline freeze can be complete.

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

The target repo currently contains only `docs/` and `_references/`. There is no root workspace, no root `pyproject.toml`, no `packages/`, and no `bots/` directory yet. The implementation must therefore distinguish three actions:

- Adapt existing package metadata from `_references/trading`, `_references/crypto_trader`, and `_references/trading_assistant_agent/packages/*`.
- Create new package metadata for `_references/k_stock_trader`, which currently has `requirements.txt` and `pytest.ini` but no root `pyproject.toml`.
- Move code out of `_references` only after baseline and import-compatibility gates are in place.

Existing assets that should be reused rather than recreated:

- IBKR already has `pyproject.toml`, parity fixtures, `scripts/verify_backtest_baseline_regeneration.py`, `backtests/shared/parity/baseline_regeneration.py`, and a frozen baseline manifest at `tests/fixtures/backtest_baselines/manifest.json`.
- IBKR also already has targeted live/backtest alignment tests, including ALCB round 2 live-default checks and swing portfolio-synergy live parity checks.
- Crypto already has `pyproject.toml`, `crypto_trader.optimize.contracts`, phase-state contract hashes, deployment manifest preflight checks, `parity_alignment.json`, assistant bridge contracts, and a portfolio round 3 live deployment bundle.
- K-stock already has strong runtime readiness, bridge-contract generation, deployment metadata emission, artifact hygiene tests, paper/live replay tooling, and OLR/KALCB runtime gates. It needs workspace packaging, not a rewrite.
- The assistant already has package-local `pyproject.toml` files and a root `tools/run_workspace_checks.py` tier system. Those checks should become part of the root monorepo gate instead of being replaced.

Known gaps to close before code extraction:

- K-stock latest optimizer output is not present in this checkout under `data/backtests/output`; restore it from production/backups or regenerate it in a frozen environment before freezing K-stock baselines.
- Crypto portfolio round 3 `deployment_manifest.json` references `output/portfolio/rounds_manifest.json`, but that file is not present in this reference checkout. Restore it, regenerate it, or update the deployment manifest with explicit approval evidence before cutover.
- KALCB live config has `kalcb.frontier.size: 104`, while the optimizer config has `kalcb.frontier.size: 103` and the OLR deployment universe is named 103. Treat this as a live-config alignment finding: either prove it is an approved runtime overlay or correct it before promotion.
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

  bots/
    ibkr_trading/
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
    freeze_optimization_baselines.py
    verify_latest_round_no_drift.py
    verify_live_config_promotions.py
    verify_strategy_plugin_contracts.py
    verify_deployment_metadata.py
```

## Workspace Tooling Decisions

Use `uv` as the workspace and lockfile authority. Do not leave this as `uv` vs constraints-file optionality; the monorepo needs one reproducible resolver and one source of truth for image builds.

Required mechanics:

- Root `pyproject.toml` declares `[tool.uv.workspace]` members for `packages/*` and `bots/*`.
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
- Generated effective live config snapshots.

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

Bot-specific market data downloaders should stay inside bot packages or `trading_assistant_data` until the interfaces stabilize.

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

Move from `_references/trading_assistant_agent/packages/trading_assistant`.

Responsibilities remain:

- Control-plane planning.
- Approval workflow.
- Evidence verification.
- Learning ledger checks.
- Human-in-the-loop orchestration.

It should depend on `trading_contracts`, not on bot runtime internals.

### `packages/trading_assistant_data`

Move from `_references/trading_assistant_agent/packages/trading_assistant_data`.

Responsibilities remain:

- Canonical data product.
- Raw imports.
- Data bundle manifests.
- Coverage manifests.
- Calendar and checksum policy.

It can share model definitions with `trading_contracts`.

### `packages/trading_assistant_backtest`

Move from `_references/trading_assistant_agent/packages/trading_assistant_backtest`.

Responsibilities remain:

- Monthly optimizer runner.
- Manifest-driven replay.
- Diagnostics.
- Phased auto optimization orchestration.
- Repair and confirmatory workflows.
- Decision parity reporting.

It should call bot strategy adapters through declared plugin contracts, not by importing arbitrary runtime modules.

### `bots/ibkr_trading`

Owns IBKR-specific runtime behavior.

Responsibilities:

- IBKR broker adapter and session management.
- IB Gateway connection handling.
- IBKR runtime shell.
- IBKR OMS integration.
- Swing, momentum, stock runtime family coordinators.
- IBKR-specific deployment config and Dockerfile.

### `bots/k_stock_trader`

Owns KIS/KRX-specific runtime behavior.

Responsibilities:

- KIS broker adapter.
- KRX calendars and session policy.
- OMS integration.
- OLR/KALCB runtime modes.
- Resource plan and API quota enforcement.
- KIS-specific deployment config and Dockerfile.

### `bots/crypto_trader`

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
    "template_path": "bots/ibkr_trading/config/strategies.yaml",
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

- The local reference checkout did not contain `data/backtests/output`.
- Before migration, restore the latest K-stock optimizer artifacts from the current production artifact store, VPS, backup, or original checkout.
- If restoration is impossible, rerun the latest accepted optimizer workflow in a frozen environment and mark the regenerated artifacts as the baseline with explicit provenance.

Crypto portfolio adoption bundle:

- Strategy-level crypto rounds are present for momentum, trend, and breakout.
- The live deployment example points to portfolio round 3 recommended strategy configs and `output/portfolio/round_3/deployment_manifest.json`.
- `output/portfolio/round_3/parity_alignment.json` says the portfolio bundle is matched.
- `output/portfolio/round_3/deployment_manifest.json` references `output/portfolio/rounds_manifest.json`, which is missing in this checkout. Treat that as an artifact-integrity blocker until restored, regenerated, or explicitly superseded.

Existing IBKR baseline fixture:

- `_references/trading/tests/fixtures/backtest_baselines/manifest.json` already defines the artifact/regeneration shape for frozen diagnostic baselines.
- Extend that format into the root baseline index instead of inventing a second baseline manifest shape.
- Existing helper code in `_references/trading/backtests/shared/parity/baseline_regeneration.py` already supports sandbox regeneration, normalized hashes, metric checks, and explicit regeneration metadata.

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

## Implementation Phases

### Phase Transition Gates

Each phase has local tasks, but phase movement is governed by the acceptance matrix. The rule is cumulative: when moving from phase N to phase N+1, all gates that were already green must still be green. Any unexplained optimizer drift, live-config mismatch, missing contract evidence, or `_references` runtime dependency stops the migration at the current phase.

| Transition | Required acceptance gates | Must be true before moving on |
| --- | --- | --- |
| Start Phase -1 | none | References have been read and no code has been moved yet. |
| Phase -1 -> Phase 0 | A00 | Root repository, VCS, remote, CI, artifact storage, uv workspace policy, and dependency-resolution audit are established enough for later commit/config/deployment hashes to be meaningful. |
| Phase 0 -> Phase 1 | A0, A2, A3, A4, A5 decision | Source inventory exists; IBKR and crypto latest rounds are frozen; crypto portfolio bundle integrity is restored or explicitly superseded; K-stock has restored/regenerated baseline artifacts or a documented scoped blocker; known config findings, including KALCB `frontier.size`, are recorded. |
| Phase 1 -> Phase 2 | A1 plus prior gates | Root workspace imports work; IBKR, crypto, assistant, and K-stock entrypoints still behave through compatibility paths; no runtime behavior has intentionally changed. |
| Phase 2 -> Phase 3 | A9, A12 plus prior gates | Shared contract schemas/readers validate existing assistant, IBKR, crypto, and K-stock artifacts without changing their meaning; strategy plugin contracts are canonicalized under the monorepo path. |
| Phase 3 -> Phase 4 | A4, A6, A7, A8 plus prior gates | Every enabled live strategy maps to latest approved backtest evidence or an approved overlay; generated effective live configs match current behavior; crypto portfolio round 3 and K-stock overlay findings are resolved. |
| Phase 4 -> Phase 5 | A13 plus A2, A3, A5 re-run | Shared optimizer adapters prove compatibility with existing bot runners; latest phased optimizer outputs still match frozen baselines after optimizer wrapping. |
| Phase 5 -> Phase 6 | A10, A14 plus prior gates | Shared backtest/replay invariants pass; promoted strategies have decision parity evidence or are explicitly not promoted; strategy implementation lessons are enforced by tests. |
| Phase 6 -> Phase 7 | A11 plus A9, A10 re-run | Assistant packages are in the workspace, existing assistant workspace tiers pass, and assistant coupling to bots remains contract-only. |
| Phase 7 -> Phase 8 | A15, A16 plus A6, A7, A8 re-run | Three independent bot images build; each image can start in artifact-only, dry-run, or paper-compatible mode; deployment metadata includes commit, image, config, promotion, contract, strategy, telemetry, and runtime provenance. |
| Phase 8 -> Phase 9 | A10, A13, A14 plus A2, A3, A5 re-run | Any shared strategy-core extraction has proven decision parity and optimizer no-drift; venue-specific execution/session semantics remain explicit. |
| Phase 9 -> Complete | A17 plus all prior gates | Final acceptance report proves latest phased auto-optimization artifacts are unchanged, live configs align with latest approved backtest configs, and no runtime/test path depends on `_references`. |

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
- Record the KALCB `frontier.size` 104 vs 103 alignment finding as either approved overlay evidence or a promotion blocker.

Deliverables:

- `backtests/baselines/baseline_index.json`
- `backtests/baselines/<bot>/<strategy>/...`
- `contracts/strategy_plugins/...`
- `contracts/promotions/draft/...`

Required checks:

- `python tools/freeze_optimization_baselines.py --check`
- `python tools/verify_latest_round_no_drift.py --baseline backtests/baselines/baseline_index.json`
- `python tools/verify_live_config_promotions.py --bot all --require-latest-round`

Do not proceed until this phase is green for IBKR and crypto, has restored or superseded the crypto portfolio rounds manifest, and has an explicit K-stock baseline decision.

### Phase 1 - Create Workspace Without Behavior Changes

Goal: import existing code into the monorepo with minimal edits.

Tasks:

- Finalize root `pyproject.toml` workspace members from the Phase -1 skeleton.
- Keep Python 3.12 as the live-image runtime target.
- Adapt existing package metadata from IBKR, crypto, and assistant packages.
- Create new package metadata for K-stock from `requirements.txt`, `pytest.ini`, and its runtime/test entrypoints.
- Add package directories under `packages` and bot directories under `bots`.
- Leave the physical assistant package move for Phase 6; in Phase 1, only adapt metadata and compatibility paths needed for import checks.
- Move or copy bot code into `bots` with compatibility entrypoints.
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

Goal: make live configs provably aligned with latest approved backtests.

Tasks:

- Implement `packages/trading_config`.
- Define config merge order:
  1. Strategy defaults
  2. Latest optimized config
  3. Venue/runtime overlay
  4. Environment-specific deployment overlay
  5. Secrets by environment only, never committed
- Create promotion manifests for every live strategy.
- Generate effective live configs for all three bots.
- Update runtimes to emit promotion manifest hash and effective config hash in deployment metadata.
- Add fail-closed checks for stale or missing promotion evidence.

Deliverables:

- `contracts/promotions/<bot>/<strategy>.json`
- `deployments/<bot>/generated/*.effective.*`
- `tools/verify_live_config_promotions.py`

Required checks:

- Generated live configs match current live behavior.
- Every enabled strategy in `config/strategies.yaml`, K-stock runtime config, and crypto `config/strategies/*.json` maps to a promotion manifest.
- Disabled/research strategies are explicitly marked as `disabled`, `research`, or `not_promoted`.
- Crypto live config points to the approved portfolio round 3 bundle or an approved successor bundle.
- KALCB `frontier.size` and deployment universe size are either equal or explicitly justified by an approved runtime overlay.
- IBKR hard-coded strategy defaults covered by existing live parity tests are represented in promotion evidence before those tests are removed or generalized.

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
  - `_references/trading_assistant_agent/packages/trading_assistant`
  - `_references/trading_assistant_agent/packages/trading_assistant_data`
  - `_references/trading_assistant_agent/packages/trading_assistant_backtest`
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
  - `bots/ibkr_trading/Dockerfile`
  - `bots/k_stock_trader/Dockerfile`
  - `bots/crypto_trader/Dockerfile`
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
- Remove `_references` dependency from tests and runtime.
- Archive reference snapshots outside production package paths.
- Update all documentation to use new paths.
- Lock the workspace.

Deliverables:

- Clean package imports.
- No runtime dependency on `_references`.
- Final migration report.

Required checks:

- Full CI green.
- All baseline no-drift checks green.
- All promotion manifests approved.
- All three dry-run/paper deployments validated.

## Per-Bot Migration Notes

### IBKR Bot

Keep the unified runtime shell and family coordinator model. Move broker and runtime code into `bots/ibkr_trading`, while extracting reusable runtime metadata and contract validation to shared packages.

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

The missing `data/backtests/output` baseline must be resolved before optimizer extraction. Do not infer latest K-stock rounds from live config alone.

### Crypto Bot

Keep the package shape of `src/crypto_trader` initially because it is already close to the target. Move it under `bots/crypto_trader/src/crypto_trader` with minimal import changes.

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

| ID | Gate | Scope | Existing evidence to reuse | New or migrated artifact | Required check | Pass criteria | Blocks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A00 | Repository and tooling bootstrap | root repo | root is not yet a git repository; assistant already has workspace-check scripts | `.git`, remote, CI stub, `pyproject.toml` workspace skeleton, `uv.lock`, artifact storage decision, bootstrap decision record | `python tools/check_repo_bootstrap.py`; `python tools/workspace_lock_check.py`; `python tools/detect_affected_images.py --explain-only` | root git identity is usable; CI and branch protection are selected; artifact storage classes are assigned; uv lock resolves planned packages for Python 3.12 live images; affected-image mapping is explainable | baseline freeze, deployment metadata, image builds |
| A0 | Workspace starting-state inventory | root repo | pre-bootstrap source material is under `docs/` and `_references/` | `docs/migration_inventory.md` or baseline inventory section in `baseline_index.json` | `python tools/freeze_optimization_baselines.py --inventory-only` | all source roots, pyprojects, requirements, configs, compose files, contracts, artifact roots, duplicate-round policy, and archive-resolution policy are listed with hashes | any code move |
| A1 | Python/package metadata | all packages | IBKR pyproject, crypto pyproject, assistant package pyprojects, K-stock requirements/pytest.ini | root workspace `pyproject.toml`; K-stock package pyproject | `python tools/workspace_import_smoke.py --all-packages` after `uv sync --frozen` or wrapper-verified equivalent | all package entrypoints import; K-stock no longer relies on bare repo root path hacks | package extraction |
| A2 | IBKR latest round freeze | all active IBKR strategies and portfolio-synergy outputs | IBKR `backtests/output/**/rounds_manifest.json`; existing baseline regeneration helpers | `backtests/baselines/ibkr/...`; `baseline_index.json` | `python tools/verify_latest_round_no_drift.py --bot ibkr --baseline backtests/baselines/baseline_index.json` | active non-archived latest rounds match by canonical hash and key metrics; duplicate round entries are disambiguated by `latest_round` plus timestamp; archived/cleaned artifacts are resolved or fail closed; ignored fields are explicit | optimizer extraction, strategy moves |
| A3 | Crypto latest round freeze | momentum, trend, breakout | crypto strategy `rounds_manifest.json`, optimized configs, `parity_alignment.json` | `backtests/baselines/crypto/...` | `python tools/verify_latest_round_no_drift.py --bot crypto --baseline backtests/baselines/baseline_index.json` | strategy-level round 3 artifacts match canonical hashes, contract hashes, profile hashes, and key metrics | crypto package move, optimizer extraction |
| A4 | Crypto portfolio bundle integrity | portfolio round 3 | `output/portfolio/round_3/deployment_manifest.json`, `parity_alignment.json`, recommended configs | restored or superseded `output/portfolio/rounds_manifest.json`; portfolio promotion manifest | `python tools/verify_live_config_promotions.py --bot crypto --require-portfolio-bundle` | deployment manifest references exist; parity alignment is matched; live config paths equal approved bundle paths | crypto deployment cutover |
| A5 | K-stock latest baseline decision | KALCB, OLR, portfolio synergy | K-stock optimization configs and scripts; missing `data/backtests/output` in this checkout | restored/regenerated K-stock baseline artifacts and provenance note | `python tools/verify_latest_round_no_drift.py --bot k_stock --baseline backtests/baselines/baseline_index.json` | either restored latest accepted artifacts validate, or regenerated artifacts have frozen command, source SHAs, and explicit approval | K-stock optimizer extraction, K-stock live promotion |
| A6 | Live config alignment: IBKR | enabled IBKR strategies | `config/strategies.yaml`; existing ALCB and swing parity tests | IBKR promotion manifests and generated effective registry | `python tools/verify_live_config_promotions.py --bot ibkr --require-latest-round` | every enabled strategy maps to latest approved optimized config or approved overlay; disabled strategies are tagged | IBKR live image cutover |
| A7 | Live config alignment: crypto | live strategy configs and portfolio config | `config/live_config.example.json`, `config/strategies/*.json`, portfolio round 3 bundle | crypto promotion manifests and generated effective live config | `python tools/verify_live_config_promotions.py --bot crypto --require-latest-round` | live config points to approved portfolio bundle; strategy config hashes match promotion evidence | crypto live image cutover |
| A8 | Live config alignment: K-stock | KALCB, OLR, OLR/KALCB portfolio | `config/kalcb.yaml`, `config/optimization/*.yaml`, bridge contract, resource plan | K-stock promotion manifests and generated effective runtime config | `python tools/verify_live_config_promotions.py --bot k_stock --require-latest-round` | KALCB/OLR config values match latest artifacts or approved overlays; `frontier.size` 104 vs 103 is resolved | K-stock paper/live cutover |
| A9 | Strategy plugin contracts | assistant bridges for IBKR, K-stock, crypto | assistant contracts, K-stock generated bridge contract, crypto bridge contracts | canonical contracts under `contracts/strategy_plugins` | `python tools/verify_strategy_plugin_contracts.py --all` | mature contracts include live repo path, full commit SHA, adapter path, config schema, decision API, telemetry schemas, fixtures, and hashes | assistant adoption gates |
| A10 | Decision parity | promoted strategies and families | IBKR parity fixtures/tests, crypto parity traces, K-stock replay fixtures | normalized parity reports in baseline/promotion evidence | `python tools/run_decision_parity_matrix.py --promoted-only` | every promoted strategy/family has PASS for signals, filters, entries, exits, stops, sizing, risk caps, and order intent, or an explicit non-promotion status | live promotion |
| A11 | Assistant workspace checks | assistant control/data/backtest | existing `tools/run_workspace_checks.py` tiers | migrated checks under root workspace | `python tools/run_workspace_checks.py deployment-gate` | assistant import, CLI, structure, monthly, data, backtest, loop, verifier, validation, and deployment package tiers pass or skip only for documented local-only data | assistant package migration |
| A12 | Shared contracts compatibility | all legacy artifacts | assistant contract models, crypto optimizer contracts, K-stock bridge/deployment metadata, IBKR round manifests | `packages/trading_contracts` schemas and readers | `trading-contracts validate --all-known-reference-artifacts` | existing artifacts validate unchanged or through named compatibility adapters | shared package adoption |
| A13 | Optimizer compatibility | three phased optimizer implementations | IBKR/K-stock/crypto `PhaseRunner` and plugin protocols | `packages/trading_optimizer` adapter layer | `python tools/verify_optimizer_compatibility.py --bot all --fixture-set smoke` | old and new runners produce the same cumulative mutations, gate decisions, selected candidates, and canonical round outputs on fixtures | deleting duplicated optimizer internals |
| A14 | Backtest integrity invariants | all promoted backtests | `docs/strategy-implementation-lessons.md`; existing artifact hygiene and parity tests | shared invariant test suite | `python tools/run_backtest_integrity_matrix.py --promoted-only` | completed-bar, next-bar fill, broker path, MTM risk, net/gross accounting, shared-capital portfolio, diagnostics, timestamp, artifact hygiene, and stress gates pass | accepting regenerated performance |
| A15 | Deployment image build | three bots | existing compose/Dockerfiles where present | `bots/*/Dockerfile`, `deployments/*/docker-compose.yml` | `docker build` matrix plus runtime import smoke | each image builds independently and includes only required bot/shared packages | VPS deployment |
| A16 | Deployment metadata | three bot runtimes | K-stock metadata emitter, crypto live metadata, IBKR registry artifact | normalized deployment metadata in runtime state/artifacts | `python tools/verify_deployment_metadata.py --bot all` | full commit, clean worktree policy, image version, config hash, promotion hash, contract hash, strategy version, telemetry schema, runtime entrypoint are present; auto-update services such as Watchtower cannot bypass fail-closed validation | live trading enablement |
| A17 | Final refactor acceptance | entire monorepo | all previous gates | `migration_acceptance_report.json` | `python tools/verify_refactor_acceptance.py --bot all --strict` | latest phased auto-optimization artifacts are unchanged; live configs align with latest approved backtest configs; no runtime depends on `_references` | refactor completion |

## Finite Implementation Checklist

Track this checklist directly. Do not add broad rolling tasks; split any new discovery into a finite item with an owner and an acceptance gate.

- [ ] B1. Initialize the root git repository and configure the real remote.
- [ ] B2. Choose CI provider, add the first workflow stub, and define branch protection expectations.
- [ ] B3. Decide artifact storage classes: source control, Git LFS, object storage, and generated local-only outputs.
- [ ] B4. Create the root `uv` workspace skeleton and generate or audit `uv.lock`.
- [ ] B5. Add bootstrap scripts for repo checks, workspace lock checks, import smoke, and affected-image detection.
- [ ] B6. Record bootstrap decisions in `docs/repo-bootstrap-decisions.md`.
- [ ] 1. Generate source inventory for `_references/trading`, `_references/k_stock_trader`, `_references/crypto_trader`, and `_references/trading_assistant_agent`.
- [ ] 2. Record all existing pyprojects, requirements, pytest configs, scripts, Dockerfiles, compose files, contracts, live configs, and artifact roots in the inventory.
- [ ] 3. Freeze IBKR latest non-archived round manifests and latest optimized configs into `backtests/baselines/ibkr`, using `latest_round` plus timestamp disambiguation for duplicate round numbers.
- [ ] 4. Extend the existing IBKR baseline fixture shape into the root baseline index, including `archive_path`/`cleanup_note` artifact resolution.
- [ ] 5. Freeze crypto strategy-level latest round 3 artifacts for momentum, trend, and breakout.
- [ ] 6. Freeze crypto portfolio round 3 recommended strategy configs, recommended portfolio config, deployment manifest, parity alignment, and lineage correction.
- [ ] 7. Restore or regenerate crypto `output/portfolio/rounds_manifest.json`, or replace the deployment manifest with an approved successor reference.
- [ ] 8. Restore K-stock `data/backtests/output` from the authoritative artifact source.
- [ ] 9. If K-stock artifacts cannot be restored, regenerate them in a frozen environment and record exact commands, data fingerprints, source SHAs, and approval note.
- [ ] 10. Decide and document the KALCB `frontier.size` 104 vs 103 alignment finding.
- [ ] 11. Create final root workspace metadata and committed `uv.lock` policy.
- [ ] 12. Adapt IBKR package metadata from `_references/trading/pyproject.toml`.
- [ ] 13. Adapt crypto package metadata from `_references/crypto_trader/pyproject.toml`.
- [ ] 14. Move assistant package metadata from the three assistant package pyprojects without changing behavior.
- [ ] 15. Create K-stock package metadata from `requirements.txt`, `pytest.ini`, runtime entrypoints, and test layout.
- [ ] 16. Add import compatibility shims for old IBKR, K-stock, crypto, and assistant paths.
- [ ] 17. Move assistant packages into `packages/` and run existing assistant workspace checks.
- [ ] 18. Implement `packages/trading_contracts` by lifting assistant contract models and adding legacy adapters.
- [ ] 19. Generate canonical JSON schemas into `contracts/schemas`.
- [ ] 20. Move strategy plugin contracts into `contracts/strategy_plugins` and update paths to monorepo locations.
- [ ] 21. Add `StrategyPromotionManifest` schema and validators.
- [ ] 22. Generate draft promotion manifests for every enabled IBKR strategy.
- [ ] 23. Generate draft promotion manifests for crypto momentum, trend, breakout, and portfolio round 3 bundle.
- [ ] 24. Generate draft promotion manifests for KALCB, OLR, and K-stock portfolio synergy after baseline restoration.
- [ ] 25. Implement generated effective live config snapshots for each bot.
- [ ] 26. Prove generated effective live configs match current live behavior or record approved overlays.
- [ ] 27. Move crypto into `bots/crypto_trader` with CLI compatibility.
- [ ] 28. Move IBKR into `bots/ibkr_trading` with runtime CLI compatibility.
- [ ] 29. Move K-stock into `bots/k_stock_trader` with runtime/session CLI compatibility.
- [ ] 30. Add deployment metadata validation to all three runtimes.
- [ ] 31. Add fail-closed startup checks for missing/stale promotion evidence in live/paper modes.
- [ ] 32. Wrap existing optimizer runners behind the shared optimizer adapter interface.
- [ ] 33. Run optimizer compatibility fixtures before moving shared optimizer internals.
- [ ] 34. Extract shared optimizer code only after compatibility fixtures and baseline no-drift gates pass.
- [ ] 35. Add shared backtest integrity invariant tests from `docs/strategy-implementation-lessons.md`.
- [ ] 36. Preserve venue-specific fill timing, session, fee, slippage, funding, and broker semantics as explicit inputs.
- [ ] 37. Add per-bot Dockerfiles and per-VPS compose files under `deployments/`.
- [ ] 38. Build all three images and run import smoke inside each image.
- [ ] 39. Run artifact-only or dry-run mode for each bot image without broker secrets.
- [ ] 40. Run decision parity matrix for every promoted strategy/family.
- [ ] 41. Run live config promotion verifier for all bots with `--require-latest-round`.
- [ ] 42. Run latest round no-drift verifier for all bots with the frozen baseline index.
- [ ] 43. Run assistant deployment gate and validation matrix after path updates.
- [ ] 44. Remove compatibility shims only after no runtime/test path depends on them.
- [ ] 45. Remove runtime and test dependencies on `_references`.
- [ ] 46. Produce `migration_acceptance_report.json` with all acceptance matrix rows and evidence paths.
- [ ] 47. Cut over VPS deployments one bot at a time, starting in artifact-only/dry-run or paper mode.
- [ ] 48. For each bot cutover, record rollback image tag, previous compose file, previous live config hashes, and restore command before enabling paper/live mode.
- [ ] 49. Archive `_references` snapshots outside production import paths after all gates pass.

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
- The three bot images build independently.
- Each bot can run artifact-only, dry-run, or paper mode from its own VPS compose stack.
- All promoted strategies have a `StrategyPromotionManifest`.
- Every enabled live strategy maps to the latest approved optimized config.
- Latest phased auto-optimization baselines are unchanged or have explicitly approved migration notes.
- Decision parity reports exist and pass for all promoted strategies.
- Deployment metadata includes git commit, image version, strategy version, config hash, promotion hash, and contract hash.
- Assistant monthly workflow checks run against the monorepo paths.
- No runtime code depends on `_references`.
- `python tools/verify_latest_round_no_drift.py --bot all --baseline backtests/baselines/baseline_index.json --strict` passes.
- `python tools/verify_live_config_promotions.py --bot all --require-latest-round --strict` passes.
- `python tools/verify_refactor_acceptance.py --bot all --strict` emits an acceptance report with all matrix rows passing.

## Recommended First Pull Requests

1. Bootstrap repository identity and workspace tooling: git remote, CI stub, artifact policy, uv workspace skeleton, lock audit, and affected-image detection.
2. Add inventory and baseline freeze tooling, explicitly reusing the existing IBKR baseline-regeneration shape.
3. Restore or explicitly supersede missing artifact evidence: K-stock `data/backtests/output` and crypto `output/portfolio/rounds_manifest.json`.
4. Add `trading_contracts` by lifting assistant contract models and adding legacy artifact readers.
5. Add promotion manifest schema and live config promotion verifier, including the KALCB `frontier.size` alignment finding.
6. Move crypto into `bots/crypto_trader` with minimal behavior changes and preserve portfolio round 3 deployment-manifest preflight.
7. Move IBKR into `bots/ibkr_trading` with runtime compatibility entrypoints and preserve existing parity/baseline tests.
8. Create K-stock package metadata, freeze restored baselines, then move K-stock into `bots/k_stock_trader`.
9. Move assistant packages into `packages` with contract-only bot coupling and preserve `tools/run_workspace_checks.py` tiers.
10. Extract shared optimizer kernel behind compatibility adapters only after no-drift gates are green.
11. Add Docker build matrix, deployment metadata gates, final acceptance verifier, per-VPS compose outputs, and rollback scripts.
