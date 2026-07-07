# Consistency Check: implementation plan vs `legacy source snapshots` codebases

Date: 2026-07-02. Method: automated verification of the plan's per-system claims against the four reference checkouts (file existence, compose service parsing, class-definition scans, rounds-manifest JSON extraction).

## Verdict

**The plan is factually consistent with all four reference systems.** Every load-bearing claim checked — entrypoints, runtime modes, contract names, deployment stacks, optimizer class names, config findings, missing-artifact gaps, and all 16 baseline-round table rows — matches the actual code and artifacts. Findings below are nuances and trivial omissions, not contradictions.

## Verified claims by system

### IBKR (`bots/ibkr_trading`)
- Layout `apps/`, `libs/`, `strategies/`, `backtests/`, `regime/`, `config/`, `backtests/shared/auto` — all exist.
- Entrypoints: `apps/runtime/runtime.py` and `apps/runtime/cli.py` exist (Phase 1 check `python -m apps.runtime.cli --help` is valid).
- Compose services exactly match the plan: postgres, ib-gateway, runtime, relay, dashboard, watchdog.
- Optimizer classes exist as named: `PhaseRunner` (backtests/shared/auto/phase_runner.py), `PhaseSpec`/`PhaseAnalysisPolicy` (plugin.py), `RoundManager` (round_manager.py).
- Live parity tests exist: `test_swing_portfolio_synergy_live_parity.py`, `test_momentum_portfolio_synergy_live_parity.py`, `test_alcb_round2_alpha_controls.py` (plausibly the "ALCB round 2 live-default checks").
- **All 13 rounds-table rows match**, including three rows (vdubus r3, tpc r8, stock/portfolio_synergy r3) where the manifest holds *duplicate entries per round number* (April run + May re-run). The plan consistently used the latest-timestamped entry: vdubus r3 2026-05-23 / 29 mut / 212 trades / pf 2.647; tpc r8 2026-05-23 / 96 mut / 126 trades / pf 2.441; stock/ps r3 / 8 mut / 1684 trades / pf 2.241, with `latest_round: 3` designated in the manifest.

### Crypto (`bots/crypto_trader`)
- `src/crypto_trader` layout, `cli.py`, `optimize/contracts.py` with `optimizer_contract_v1`, `phase_state`, `optimized_config` markers; `LIVE_PARITY_PROFILE` (10 files), preflight (5 files), `PhaseRunner`/`PhaseSpec` present.
- Compose matches: postgres, trader, data-refresh, dashboard + backtest/optimize behind `tools` profile.
- Rounds table exact: momentum r3 2026-05-26 pf 2.716/46 trades/6 mut; trend r3 2026-05-25 pf 5.928/29/2; breakout r3 2026-05-27 pf 23.330/17/5.
- Portfolio round_3 `deployment_manifest.json` references `portfolio_rounds_manifest_path` while `output/portfolio/rounds_manifest.json` is missing — exactly the integrity blocker the plan flags. `parity_alignment.json` reports matched.

### K-stock (`bots/k_stock_trader`)
- All five runtime modes present in `deployment/olr_kalcb/runtime.py`: artifact_only_stage1, artifact_only, dry_run, paper, live.
- Compose: oms, pcim (profile-gated = plan's "optional PCIM research"), runtime, dashboard, postgres.
- `k_stock_olr_kalcb_strategy_plugin_contract_v1` in `deployment/olr_kalcb/bridge_contract.py`; `live_parity_fill_timing` (36 files), `next_5m_open` (19), `resource_plan` (31), `deployment_metadata` (10); `PhaseRunner`/`PhaseSpec`/`RoundManager` in `backtests/auto/shared/`.
- `config/kalcb.yaml` frontier `size: 104` vs `config/optimization/kalcb.yaml` `kalcb.frontier.size: 103` — the plan's alignment finding is exact.
- No `pyproject.toml`; `requirements.txt`/`pytest.ini` present; `data/backtests/output` missing — as stated.

### Assistant (`packages/trading_assistant`)
- Exactly the three claimed packages.
- All nine claimed existing contract classes at exact names/locations: MarketDataManifest, DataBundleManifest, MonthlyRunManifest, StrategyPluginContract, DecisionParityReport, RoundsManifest, RoundManifestRecord, ConfirmatoryRerank, BacktestArtifactIndex (all in `trading_assistant/schemas/`), plus DeploymentMetadata in `trading_assistant_backtest`. `repair_ablation` referenced in 10 files including `schemas/backtest_artifacts.py`.
- All eight `run_workspace_checks.py` tiers the plan invokes exist: monthly-focused, data-contracts, backtest-monthly, backtest-approval, loop-contracts, performance-learning-ledger, monthly-verifier, deployment-gate.
- Models the plan proposes as NEW (StrategyPromotionManifest, TelemetryEventEnvelope, RuntimeReadinessReport, OptimizerPhaseState, OptimizationContract, OptimizedConfigMetadata) are correctly absent from the assistant.

## Nuances the plan should absorb (none are contradictions)

1. **Duplicate round entries.** IBKR `rounds_manifest.json` files contain multiple entries per round number (re-runs). The plan's tables silently used the latest-timestamped entry, but the plan text never states the disambiguation rule. `freeze_optimization_baselines.py` / `verify_latest_round_no_drift.py` must select by timestamp and/or the manifest's `latest_round` designation, not round number alone — otherwise the no-drift check can bind to the wrong entry.
2. **Cleaned round artifacts.** `stock/portfolio_synergy` has only `round_1/`, `round_2/` subdirectories plus `archive_path`/`cleanup_note` manifest keys, while `latest_round` is 3. Phase 0's "copy phase state files, candidate ledger, …" must resolve archived/cleaned rounds via `archive_path`, or the freeze will silently miss round-3 artifacts.
3. **Compose omissions (trivial).** Plan's K-stock stack list omits postgres; crypto stack list omits watchtower. Watchtower is worth a mention: an auto-update service interacts with the promotion-evidence model (mitigated by the plan's fail-closed startup validation, but it should be a stated assumption).
4. **Naming nit.** The plan's required-contracts list says `ConfirmatoryRerankReport`; the assistant class is `ConfirmatoryRerank`. This violates the plan's own "preserve existing assistant names" rule — rename in the list.
5. **Crypto has no `RoundManager`.** Phase 4's shared-concepts list includes RoundManager, which exists in IBKR and K-stock only; crypto handles rounds differently (no class, `rounds_manifest` written procedurally) and its contracts are dict-based, not Pydantic. The wrap-first approach accommodates this, but the plan could note crypto's variant explicitly.

## Relationship to prior review

The separate evaluation (`monorepo-plan-evaluation-2026-07-02.md`) covers plan-internal defects (VCS bootstrap, uv workspace mechanics, Phase 1 contradictions, package-ownership overlaps). This document confirms the plan's *external* consistency with the reference systems; items 1–2 above should be added to that review's amendment list as Phase 0 tooling requirements.
