# Repository Bootstrap Decisions

Date: 2026-07-02

## Scope

This record covers the bootstrap decisions that still govern
`docs/contract-first-python-workspace-monorepo-implementation-plan.md`.
The monorepo now owns the bot, package, contract, deployment, baseline, artifact, and verification paths
needed by the acceptance matrix; historical source snapshots are not runtime, test, or verification inputs.

## Repository Identity

- Root VCS: Git repository rooted at `C:\Users\sehyu\Documents\Other\Projects\trading_agent`.
- Default integration branch: `main`.
- Git author identity: use the configured local Git identity.
- Remote host: GitHub is selected to match the reference repositories.
- Remote status: configured. `origin` is set to
  `https://github.com/sehyungp92/trading_agent.git`.
- Remote reachability: verified on 2026-07-02 with `python tools/check_repo_bootstrap.py`; `origin`
  HEAD is reachable from this environment.

## CI and Branch Protection

- CI provider: GitHub Actions.
- Initial CI stub: `.github/workflows/ci.yml`.
- Protected branch: `main`.
- Branch protection expectation: require pull requests and require the migration gate jobs that are
  available for the current phase. The full planned required-check set is:
  `repo-bootstrap`, `workspace-lock`, `workspace-imports`, `contracts`, `baselines`,
  `live-configs`, `decision-parity`, `optimizer-compatibility`, `backtest-integrity`,
  `deployment-gate`, `deployment-metadata`, `affected-images`, `docker`, and
  `strict-refactor-acceptance`.
- Branch protection status: selected but not enforceable locally until the real remote repository
  exists.

## Python and Workspace Policy

- Resolver and lockfile authority: `uv`.
- Live-image Python target: Python 3.12.
- Package-local Python 3.11 compatibility may remain only where existing package metadata already
  declares it, especially crypto and assistant-backtest.
- Root workspace members are declared as `packages/*` and `trading/*`.
- Historical source snapshots are excluded from the `uv` workspace and from committed runtime,
  test, verification, deployment, and documentation paths.

## Artifact Storage Classes

- Source control: schemas, promotion manifests, small baseline indexes, deterministic fixtures,
  strategy plugin contracts, deployment templates, docs, and bootstrap gate scripts.
- Git LFS: binary or data-like files that may be intentionally versioned, including `.parquet`,
  `.arrow`, `.feather`, `.db`, `.sqlite`, `.sqlite3`, `.pkl`, and `.zip`. Bounded historical
  backtest data slices required for latest-round optimizer reproduction may use Git LFS when
  they are listed in `backtests/data_portability_manifest.json`.
- Object storage: large backtest outputs, raw/canonical market data, replay bundles, large
  telemetry exports, and optimizer workspaces that are not small deterministic fixtures.
- Generated local-only outputs: scratch optimizer runs, local deployment metadata, temporary
  baselines, local effective configs, secrets, and environment overlays.

## Bootstrap Gate Status

The bootstrap scaffold is present and A00 remains part of final acceptance. The local scripts must
continue to fail closed if the root remote, lockfile policy, required CI jobs, artifact storage policy,
or affected-image mapping stops validating.
