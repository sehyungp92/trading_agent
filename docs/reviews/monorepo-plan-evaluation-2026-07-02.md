# Evaluation: contract-first-python-workspace-monorepo-implementation-plan.md

Date: 2026-07-02. Method: full read of the 1,157-line plan plus filesystem verification of its factual claims against `_references/`.

## Verdict

**Near-optimal on migration safety and architecture; not yet optimal as an executable plan.** The gating skeleton (baseline freeze → contracts → promotion → wrapped extraction → parity-gated consolidation) is the correct shape for live trading systems where behavior drift is the dominant risk. The gaps are concentrated in build/tooling mechanics and a handful of internal inconsistencies.

## Verified factual grounding (all claims checked are TRUE)

- Root contains only `docs/`, `_references/`, `CLAUDE.md`. **Not a git repository.**
- IBKR: `pyproject.toml`, `apps/runtime/runtime.py`, baseline fixture manifest, `baseline_regeneration.py`, `verify_backtest_baseline_regeneration.py`, `config/strategies.yaml` — all exist.
- K-stock: no `pyproject.toml`; has `requirements.txt`/`pytest.ini`; `data/backtests/output` **missing** as claimed; runtime + session script exist.
- Crypto: `src/crypto_trader` layout confirmed; `output/portfolio/rounds_manifest.json` **missing** while `round_3/deployment_manifest.json` and `parity_alignment.json` exist — exactly the integrity blocker the plan flags.
- Assistant: three package pyprojects + `tools/run_workspace_checks.py` exist.
- Python split confirmed: mix of `>=3.11` (crypto, one assistant pkg) and `>=3.12`.
- KALCB optimizer config `kalcb.frontier.size: 103` confirmed.

## Strengths (why the skeleton is right)

1. Freeze-before-move, wrap-before-delete, parity-on-every-extraction — correct ordering for no-drift guarantees.
2. Contracts-first with **lifting** of existing assistant models instead of forking a second schema language.
3. Promotion manifests make live configs provably derived from approved backtests — closes the real alignment failure mode (e.g. frontier.size 104 vs 103).
4. Cumulative acceptance matrix (A0–A17) with named commands and re-run requirements; finite checklist resists scope creep.
5. Correct restraint: `trading_data` and `trading_strategy_shared` deferred until proven need; assistant coupled contract-only, excluded from live images.

## Defects and gaps (why it is not yet optimal)

1. **No VCS/CI bootstrap.** The plan presumes PRs, merge-blocking CI, commit-SHA and dirty-state capture — but the repo has no `.git`. Needs a pre-Phase-0 item: `git init`, remote/host, CI platform choice, branch protection, LFS-vs-artifact-store decision (currently left as "or").
2. **Workspace tooling unspecified.** "uv.lock or constraints/lock file" is the only mention. Must commit to uv workspace mechanics: `[tool.uv.workspace]` members, single-lock resolution across the 3.11/3.12 `requires-python` split, and per-image dependency subsetting (`uv sync --package` / `uv export --package` → multi-stage Dockerfiles). A15's "image includes only required packages" is unverifiable without this.
3. **Phase 1 contradictions.** (a) Requires "Docker build smoke for all three bot images", but Dockerfiles are created in Phase 7 — and references contain no IBKR bot Dockerfile (only compose) and only OMS/PCIM Dockerfiles for K-stock. (b) Phase 1 tasks already "move assistant packages into `packages`", duplicating Phase 6's entire purpose.
4. **Model ownership overlaps.** DeploymentMetadata/readiness models assigned to both `trading_contracts` and `trading_deployment`; TelemetryEventEnvelope to both `trading_contracts` and `trading_instrumentation`. Needs the explicit rule: models in contracts, behavior elsewhere.
5. **Unscheduled packages.** `trading_instrumentation` and `trading_deployment` are specified but never assigned to any phase.
6. **Naming inconsistency.** Phase 8 says `strategies/shared`; the target tree says `packages/trading_strategy_shared`.
7. **CI mechanics silent.** No affected-package change detection for the three images; no root lint/type/format standardization (ruff/mypy); no root pytest strategy.
8. **Dependency-conflict audit unscheduled.** One lockfile forces one resolution across four codebases. Spot-check (pandas>=2.0–2.3, numpy>=1.26, pydantic>=2.x) shows compatible lower bounds, so risk is low — but A0 should include a workspace resolution dry-run gate.
9. **Rollback notes only for Phase 1.** Later phases lean on gates; Phase 7 cutover deserves an explicit per-bot rollback procedure (checklist item 47 partially covers this).

## Recommended amendments (in priority order)

1. Add Phase −1: repo bootstrap (git init, host, CI platform, LFS/artifact-store decision).
2. Add a "Workspace Tooling" section pinning uv workspace + lock + per-image sync/export mechanics and the Python 3.11/3.12 resolution strategy.
3. Fix Phase 1: drop or reword the Docker smoke check (reference-image regression only), and remove the assistant move (leave it in Phase 6).
4. Add the models-vs-behavior ownership rule; assign `trading_instrumentation`/`trading_deployment` to phases (natural homes: Phase 2 models, Phase 7 behavior).
5. Unify Phase 8 target path with the tree; add dependency-resolution dry-run to A0; add CI change-detection and lint/type standardization to the CI section.
