# Repository Bootstrap Decisions

Date: 2026-07-02

## Scope

This record covers Phase -1 of `docs/contract-first-python-workspace-monorepo-implementation-plan.md`.
No reference code has been moved, no baselines have been frozen, and `_references/` remains the
source material for later phases.

## Repository Identity

- Root VCS: Git repository rooted at `C:\Users\sehyu\Documents\Other\Projects\trading_agent`.
- Default integration branch: `main`.
- Git author identity: use the configured local Git identity.
- Remote host: GitHub is selected to match the reference repositories.
- Remote status: blocked. The inferred candidate `https://github.com/sehyungp92/trading_agent.git`
  was tested on 2026-07-02 and was not reachable from this environment. A real root remote must be
  created or supplied before A00 can pass.

## CI and Branch Protection

- CI provider: GitHub Actions.
- Initial CI stub: `.github/workflows/ci.yml`.
- Protected branch: `main`.
- Branch protection expectation: require pull requests and require the migration gate jobs that are
  available for the current phase. The full planned required-check set is:
  `repo-bootstrap`, `workspace-lock`, `workspace-imports`, `contracts`, `baselines`,
  `live-configs`, `affected-images`, and `docker`.
- Branch protection status: selected but not enforceable locally until the real remote repository
  exists.

## Python and Workspace Policy

- Resolver and lockfile authority: `uv`.
- Live-image Python target: Python 3.12.
- Package-local Python 3.11 compatibility may remain only where existing package metadata already
  declares it, especially crypto and assistant-backtest.
- Root workspace members are declared as `packages/*` and `bots/*`.
- `_references/*` is excluded from the `uv` workspace; it remains migration input, not runtime
  workspace code.

## Artifact Storage Classes

- Source control: schemas, promotion manifests, small baseline indexes, deterministic fixtures,
  strategy plugin contracts, deployment templates, docs, and bootstrap gate scripts.
- Git LFS: binary or data-like files that may be intentionally versioned, including `.parquet`,
  `.arrow`, `.feather`, `.db`, `.sqlite`, `.sqlite3`, `.pkl`, and `.zip`.
- Object storage: large backtest outputs, raw/canonical market data, replay bundles, large
  telemetry exports, and optimizer workspaces that are not small deterministic fixtures.
- Generated local-only outputs: scratch optimizer runs, local deployment metadata, temporary
  baselines, local effective configs, secrets, and environment overlays.

## Bootstrap Gate Status

The Phase -1 local scaffold is present, but A00 is intentionally not considered green until a real
root remote is configured and reachable. The local scripts must continue to report this blocker so
baseline, deployment metadata, and image evidence are not trusted prematurely.
