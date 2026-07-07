# Monthly Learning Loop Cleanup Plan

Date: 2026-05-12
Status: Draft for approval

## Purpose

This plan converts the monthly learning loop implementation from an additive
integration into a streamlined target-state architecture. The cleanup should
remove stale WFO, autonomous suggestion, simplified replay, and duplicate schema
paths only after the monthly pipeline owns the relevant behavior end to end.

Primary references:

- `docs/2026-05-11-workflow-learning-loop-target-state.md`
- `docs/plans/2026-05-11-monthly-learning-loop-target-state-implementation-plan.md`
- `docs/adr/0001-monthly-evidence-replay-foundation.md`
- Current implementation in `analysis/`, `orchestrator/`, `schemas/`, `skills/`, and `tests/`

## Codebase Review Corrections

This plan was rechecked against the current codebase. The following corrections
must guide the cleanup:

- There is no `schemas.replay_artifacts` module. The canonical homes are
  `schemas.monthly_run_manifest`, `schemas.replay_parity`, and
  `schemas.market_data_manifest`.
- `schemas/coverage_manifest.py`, `schemas/parity_report.py`, and
  `schemas/run_manifest.py` are true backward-compatible aliases, but their
  canonical targets differ by artifact type.
- Many monthly target-state modules already exist and should be promoted as the
  canonical path, not removed as cleanup noise:
  `orchestrator/backtest_invocation.py`, `orchestrator/lineage_audit.py`,
  `orchestrator/market_data_jobs.py`, `schemas/backtest_artifacts.py`,
  `schemas/market_data_manifest.py`, `schemas/monthly_run_manifest.py`,
  `schemas/monthly_validation.py`, `schemas/monthly_candidates.py`,
  `schemas/monthly_model_review.py`, `schemas/monthly_outcome.py`,
  `schemas/outcome_priors.py`, `schemas/replay_parity.py`,
  `schemas/strategy_change_ledger.py`, `schemas/telemetry_manifest.py`,
  `skills/backtest_runner_client.py`, `skills/coverage_manifest_writer.py`,
  `skills/market_data_catalog.py`, `skills/market_data_sync.py`,
  `skills/monthly_candidate_pipeline.py`,
  `skills/monthly_gap_attribution.py`,
  `skills/monthly_outcome_measurer.py`,
  `skills/monthly_outcome_scorer.py`,
  `skills/monthly_validation_orchestrator.py`,
  `skills/outcome_prior_store.py`, `skills/replay_parity_checker.py`,
  `skills/rollback_advisor.py`, `skills/search_allocation_policy.py`, and
  `skills/strategy_change_ledger.py`.
- Monthly approval is not fully independent yet. `ApprovalTracker`,
  `ApprovalHandler`, `ConfigRegistry`, `PRBuilder`, repo workspace support, and
  Telegram approval callbacks are still created under `AUTONOMOUS_ENABLED` in
  `orchestrator/app.py`. Approval-gated monthly validation therefore suppresses
  approval requests unless the old autonomous feature flag is also enabled.
- `DeploymentMonitor` is also coupled to `AUTONOMOUS_ENABLED`; it warns and
  runs without PR/config/file-change helpers when autonomous is disabled.
- Legacy WFO is still active in runtime wiring:
  `orchestrator/app.py`, `orchestrator/orchestrator_brain.py`,
  `orchestrator/worker.py`, `orchestrator/scheduler.py`,
  `orchestrator/catchup.py`, and `orchestrator/handlers.py`.
- `analysis/monthly_repair_prompt_assembler.py` exists and is tested, but it is
  not invoked by the monthly handler. The current monthly candidate pipeline
  only validates a `model_review.json` if the backtest artifact directory already
  contains one.
- `analysis/context_builder.py` still loads `search_reports.jsonl`,
  `search_signals.jsonl`, and `backtest_calibration.jsonl`, and still has a WFO
  workflow priority block. That context must be deliberately migrated to monthly
  outcomes and priors, or clearly retained as historical read-only context.
- User-facing docs still advertise WFO/autonomous/parameter-search paths as
  active, especially `README.md`, `CLAUDE.md`, `.env.example`, and agent
  preference UI text.

## Target State

The monthly learning loop should be the single authoritative path for material
strategy learning:

- Monthly evidence foundation produces the trusted package:
  replay parity, live-vs-backtest deltas, counterfactuals, leakage checks,
  regime slices, cost realism, and objective thresholds.
- Monthly candidate generation produces structured proposals with calibrated
  predictions, evidence references, scope, risk, and follow-up requirements.
- Human approval moves proposals through proposal, deployment, strategy-change,
  and learning ledgers.
- Approval and PR/deployment plumbing is shared infrastructure, independent of
  the old autonomous suggestion pipeline.
- Deployment monitoring and monthly outcome measurement close the loop using
  live-first evidence.
- Forecast, hypothesis, suggestion, and scorecard context feed the next cycle.
- Daily and weekly systems remain early-warning and context providers only.
- Legacy WFO, autonomous approval, and simplified local backtest paths no longer
  create deployable strategy changes.

## Cleanup Principles

1. Remove aliases and dead wrappers first; remove runtime paths later.
2. Extract shared types before deleting legacy modules that still export them.
3. Preserve historical data readers where needed, but stop writing new records
   from stale pipelines.
4. Keep daily/weekly alerting useful, but prevent it from finalizing material
   strategy outcomes.
5. Prefer canonical monthly artifact schemas and ledgers over compatibility
   imports.
6. Promote already-existing monthly modules before deleting their legacy
   counterparts.
7. Decouple approval, PR, deployment, and Telegram approval callbacks from
   `AUTONOMOUS_ENABLED` before deleting autonomous code.
8. Delete tests for removed behavior only after replacement tests prove the
   monthly behavior is covered.
9. Run the full suite at the end of every phase that removes executable code.

## Approval Gates

Each phase below should be approved independently if code deletion is involved.
Do not collapse all removals into one change; this makes regressions hard to
localize.

Required gates:

- Gate A: Remove schema aliases and import compatibility files.
- Gate B: Promote shared approval/PR infrastructure out of autonomous ownership.
- Gate C: Disable legacy WFO runtime wiring.
- Gate D: Delete legacy WFO modules and tests.
- Gate E: Retire autonomous suggestion-to-approval path.
- Gate F: Delete simplified replay/search stack.
- Gate G: Simplify early-warning outcome measurement.
- Gate H: Remove stale context/docs/UI references.

## Phase 0 - Baseline And Inventory

Goal: prove the current system is green before removing anything.

Actions:

- Run `pytest tests/ -q` and record the result in the cleanup PR summary.
- Capture reference maps with `rg` for:
  - `coverage_manifest`
  - `parity_report`
  - `run_manifest`
  - `handle_wfo`
  - `run_wfo`
  - `autonomous_pipeline`
  - `suggestion_backtester`
  - `backtest_simulator`
  - `parameter_searcher`
  - `param_optimizer`
  - `backtest_calibration`
  - `ApprovalTracker`
  - `ApprovalHandler`
  - `DeploymentMonitor`
  - `monthly_repair_prompt_assembler`
  - `search_reports`
  - `search_signals`
  - `wfo_status`
- Identify every public import that still points at stale modules.
- Separate runtime dependencies from test-only dependencies.
- Identify monthly target-state modules that are untracked or dirty before
  deletion work starts, so cleanup does not accidentally discard implementation
  work that should be promoted.

Acceptance criteria:

- Full test suite passes before cleanup begins.
- Every removal candidate has a reference map.
- Shared survivors are identified before their current modules are deleted.

Rollback:

- No code changes in this phase.

## Phase 1 - Remove Dead Schema Compatibility Aliases

Goal: eliminate stale import surfaces that only re-export monthly artifact
schemas.

Remove after import updates:

- `schemas/coverage_manifest.py`
- `schemas/parity_report.py`
- `schemas/run_manifest.py`

Canonical replacements:

- `schemas.coverage_manifest.CoverageManifest` ->
  `schemas.market_data_manifest.MarketDataManifest`
- `schemas.parity_report.ParityReport` ->
  `schemas.replay_parity.ReplayParityReport`
- `schemas.run_manifest.RunManifest` ->
  `schemas.monthly_run_manifest.MonthlyRunManifest`

Actions:

- Replace imports from the compatibility modules with direct imports from the
  artifact-specific canonical schema module.
- Update tests to import canonical schemas.
- Delete the compatibility files.

Checks:

- `rg "schemas\.(coverage_manifest|parity_report|run_manifest)"`
- `rg "from schemas\.(coverage_manifest|parity_report|run_manifest)"`
- Run targeted monthly artifact tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- No references to the three alias modules remain.
- Monthly evidence, replay parity, and validation tests pass.

Rollback:

- Restore the alias files only if an external integration imports them. Prefer
  updating the integration instead if it lives in this repo.

## Phase 2 - Extract Shared Survivors From Legacy Modules

Goal: prevent accidental deletion of types and helpers that the monthly loop
still legitimately needs.

Extract or preserve:

- `SimulationMetrics` from `schemas/wfo_results.py`
- Approval and repository-change schemas currently imported from
  `schemas/autonomous_pipeline.py`
- Parameter/config profile schemas currently imported from
  `schemas/autonomous_pipeline.py`
- PR/deployment schemas used by `skills/deployment_monitor.py`,
  `skills/github_pr.py`, `skills/repo_workspace.py`,
  `skills/repo_change_guard.py`, and `skills/file_change_generator.py`
- Any cost/leakage robustness primitives that are used by the monthly evidence
  foundation

Recommended new homes:

- `schemas/simulation_metrics.py`
- `schemas/approval.py`
- `schemas/repo_changes.py`
- `schemas/bot_profile.py`
- `schemas/parameter_definition.py`
- `schemas/repo_task.py`

Actions:

- Move shared models to neutral schema modules.
- Update monthly, approval, and test imports.
- Leave temporary compatibility imports in old modules only until the next
  deletion phase.
- Add tests that instantiate moved models through the new canonical imports.
- Keep `schemas/autonomous_pipeline.py` only as a temporary compatibility
  exporter until all active approval, repo, config, and monthly imports are
  moved.

Checks:

- `rg "SimulationMetrics"`
- `rg "schemas.autonomous_pipeline"`
- `rg "schemas.wfo_results"`
- `rg "ParameterDefinition|BotConfigProfile|ApprovalRequest|PRRequest|PRResult|FileChange|RepoTaskContext"`
- Run approval, monthly validation, and deployment monitor tests.

Acceptance criteria:

- Monthly and approval code no longer need legacy WFO or autonomous schema
  modules for shared data types.
- Old modules can be deleted without losing active schema definitions.

Rollback:

- Keep compatibility exports in old modules for one phase if an import is missed,
  but treat them as temporary and remove them in Phase 3 or Phase 4.

## Phase 2A - Decouple Approval Infrastructure From Autonomous

Goal: make monthly approval-gated validation work without enabling the legacy
autonomous suggestion pipeline.

Current code facts:

- `ApprovalTracker`, `ApprovalHandler`, `ConfigRegistry`, `FileChangeGenerator`,
  `PRBuilder`, repo workspace/task runners, and Telegram approval callbacks are
  created only inside `if config.autonomous_enabled:` in `orchestrator/app.py`.
- `MonthlyCandidatePipeline` can create `ApprovalRequest` objects, but only when
  an `approval_tracker` is passed.
- `DeploymentMonitor` depends on helpers that are currently available only when
  autonomous is enabled.

Actions:

- Create shared approval infrastructure when either:
  - `AUTONOMOUS_ENABLED=true`, or
  - `MONTHLY_VALIDATION_MODE=approval_gated`, or
  - `DEPLOYMENT_MONITORING_ENABLED=true`.
- Keep `skills/autonomous_pipeline.AutonomousPipeline` creation behind
  `AUTONOMOUS_ENABLED`, but move approval handler/tracker, PR builder,
  config registry, file-change generator, repo workspace manager, and repo task
  runner setup outside that block.
- Register Telegram approval callbacks whenever the approval handler exists,
  not only when the autonomous pipeline exists.
- Let monthly validation pass a real `ApprovalTracker` and create approval
  requests in approval-gated mode without enabling autonomous suggestion
  processing.
- Update deployment monitoring so it can use shared PR/config/file-change
  helpers without depending on `AUTONOMOUS_ENABLED`.
- Update startup warnings so they refer to missing shared approval infrastructure,
  not missing autonomous mode.

Checks:

- `rg "config.autonomous_enabled" orchestrator/app.py`
- `rg "approval_tracker|approval_handler|deployment_monitor" orchestrator/app.py`
- Add or update tests proving `MONTHLY_VALIDATION_MODE=approval_gated` creates
  monthly approval requests with `AUTONOMOUS_ENABLED=false`.
- Add or update tests proving Telegram approval callbacks register without the
  autonomous pipeline.
- Add or update tests proving deployment monitoring can run when monthly
  approval infrastructure exists but autonomous processing is disabled.

Acceptance criteria:

- Monthly approval-gated validation no longer requires the legacy autonomous
  feature flag.
- Autonomous suggestion processing can be disabled while approvals, PR creation,
  deployment monitoring, and monthly candidate approval still work.

Rollback:

- Revert the shared setup only if monthly approvals cannot be made independent.
  Do not use `AUTONOMOUS_ENABLED` as the permanent monthly approval gate.

## Phase 3 - Disable Legacy WFO Runtime Wiring

Goal: stop the old WFO path from creating parallel strategy-change authority.

Disable first:

- WFO scheduler jobs
- WFO catchup jobs
- Brain routing to WFO tasks
- Worker dispatch to WFO handler
- Any manual `run-now wfo` path
- WFO-specific provider preference defaults that only exist for the removed path
- WFO subagent labels where they represent the old scheduled workflow

Likely files to edit:

- `orchestrator/app.py`
- `orchestrator/handlers.py`
- `orchestrator/scheduler.py`
- `orchestrator/catchup.py`
- `orchestrator/worker.py`
- `orchestrator/orchestrator_brain.py`
- `orchestrator/config.py`
- `orchestrator/agent_preferences.py`
- `schemas/agent_preferences.py`
- `schemas/agent_capabilities.py`
- `comms/telegram_renderer.py`
- `comms/discord_renderer.py`
- `comms/email_renderer.py`
- `.env.example`

Actions:

- Remove WFO task scheduling and dispatch.
- Remove unconditional per-bot `tracked_wfo_fns` construction in
  `orchestrator/app.py`.
- Remove `worker.on_wfo` setup and `ActionType.SPAWN_WFO` dispatch once no
  trigger can produce it.
- Remove `wfo_trigger` from `OrchestratorBrain._handlers`.
- Remove WFO run-history bootstrap from startup catchup, or keep it only as a
  one-time historical migration guarded by comments and tests.
- Make any deprecated manual WFO trigger return a clear unsupported message for
  one release window, or delete the CLI option if it is internal only.
- Ensure monthly validation remains the only scheduled path for material
  strategy evidence and proposal generation.
- Update docs and command examples that still mention WFO as a live path.
- Add `monthly_validation` or `monthly_model_review` workflow preferences if
  monthly model-review invocation will use agent-provider routing.

Checks:

- `rg "\bwfo\b" orchestrator analysis skills schemas tests docs .env.example`
- Run scheduler, catchup, worker, brain, agent preference, and monthly tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- No active scheduler, worker, or brain route invokes legacy WFO.
- Monthly jobs still schedule and run.
- WFO cannot produce a deployable proposal.

Rollback:

- Re-enable dispatch only if monthly scheduling cannot generate evidence
  packages. Do not re-enable WFO deployment authority.

## Phase 4 - Delete Legacy WFO Implementation

Goal: remove the old WFO code once runtime wiring is gone.

Remove candidates:

- `analysis/wfo_prompt_assembler.py`
- `analysis/wfo_report_builder.py`
- `skills/run_wfo.py`
- `schemas/wfo_config.py`
- WFO-only parts of `schemas/wfo_results.py` after `SimulationMetrics` is moved
- `Handlers._load_trades_for_wfo` and `tests/test_wfo_trade_loading.py` if no
  monthly code reuses that loader
- WFO-only tests
- WFO docs that describe the old path as active

Keep only if still used by monthly evidence:

- General robustness or leakage logic after moving it into neutral monthly
  modules
- Historical report parsers needed for migration, if any

Actions:

- Delete legacy WFO modules.
- Delete or rewrite WFO tests as monthly evidence tests.
- Remove WFO fixture data that is not used by monthly validation tests.
- Update architecture docs to describe monthly validation as the canonical
  replacement.
- Remove `memory/skills/wfo_pipeline.md` references if the file exists in a
  deployment checkout.

Checks:

- `rg "\bwfo\b" analysis skills schemas tests orchestrator`
- `rg "WFO_AGENT|wfo_status|agent_settings_.*wfo|wfo_pipeline"`
- Run all monthly candidate, validation, outcome, ledger, and scheduler tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- WFO is absent from executable runtime code.
- Any remaining `wfo` references are historical docs or explicitly named
  migration notes.

Rollback:

- Restore only the specific helper that was incorrectly classified as WFO-only,
  and move it into a neutral monthly module.

## Phase 5 - Retire Autonomous Suggestion-To-Approval Path

Goal: remove the older autonomous path that can produce strategy-change
approval work outside the monthly candidate workflow.

Remove or replace:

- `skills/autonomous_pipeline.py`
- `skills/suggestion_backtester.py`
- Legacy parameter-search approval route
- `skills/backtest_calibration_tracker.py`
- `search_reports.jsonl` feedback writers
- `search_signals.jsonl` feedback writers
- Tests that assert autonomous strategy-change finalization

Keep or move:

- Generic approval models, after Phase 2 extraction
- Generic PR/change metadata, after Phase 2 extraction
- `skills/suggestion_tracker.py` if used for monthly suggestion lifecycle,
  compatibility reads, or scorecard history
- `skills/approval_tracker.py`, `skills/approval_handler.py`,
  `skills/github_pr.py`, `skills/file_change_generator.py`,
  `skills/config_registry.py`, `skills/repo_workspace.py`,
  `orchestrator/repo_task_runner.py`, and `skills/deployment_monitor.py` as shared
  approval/deployment infrastructure after Phase 2A
- `orchestrator/app.py` approval and PR-review jobs, after they are independent
  of the autonomous pipeline

Actions:

- Route all material strategy proposals through monthly candidate generation and
  proposal ledgers.
- Remove code paths that auto-backtest, auto-score, and submit parameter-search
  approvals from daily or weekly signals.
- Preserve legacy data readers only if they feed historical scoring context.
- Update response parsing and validation docs so structured monthly candidates
  are the only deployable proposal format.
- Remove `app.state.autonomous_pipeline` only after any dashboard or API code no
  longer reads it.
- Rewrite autonomous tests into shared approval tests or monthly approval tests
  before deleting them.

Checks:

- `rg "autonomous_pipeline|suggestion_backtester|backtest_calibration|search_reports|search_signals"`
- `rg "AUTONOMOUS_ENABLED|autonomous_enabled"`
- Run approval handler, proposal ledger, strategy change ledger, monthly
  response parser, and monthly validation tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- No non-monthly path can create or advance a deployable strategy-change
  proposal.
- Historical suggestion and scorecard context can still be read where intended.

Rollback:

- Restore read-only historical adapters if reporting needs legacy records.
- Do not restore autonomous deployment or approval generation.

## Phase 6 - Delete Simplified Replay And Search Stack

Goal: remove local simulation/search modules that are weaker than the monthly
evidence foundation and no longer authoritative.

Remove candidates after references are gone:

- `skills/backtest_simulator.py`
- `skills/parameter_searcher.py`
- `skills/param_optimizer.py`
- `schemas/parameter_search.py`
- WFO-only parts of `skills/fold_generator.py`
- WFO-only parts of `skills/robustness_tester.py`
- WFO-only parts of `skills/leakage_detector.py`
- WFO-only parts of `skills/cost_model.py`
- Old parameter-search context loaders in `analysis/context_builder.py`:
  `load_search_reports`, `load_regime_parameter_analysis`,
  `load_search_signal_summary`, and `load_backtest_reliability`, unless they
  are explicitly retained as historical read-only context.

Keep or refactor:

- Any generic leakage, cost, fold, or robustness checks used by monthly replay
  artifacts.
- Monthly artifact generation and validation modules.
- `schemas/objective_weights.py`, because monthly and daily systems still need
  a shared objective vocabulary.

Actions:

- Confirm monthly evidence modules do not call the simplified simulator/search
  stack.
- Move any genuinely reusable primitives into monthly evidence modules or
  neutral utility modules.
- Delete modules whose remaining purpose is old WFO or old autonomous
  parameter-search support.
- Delete or rewrite tests to assert monthly evidence behavior instead of local
  simplified simulator behavior.
- Replace context-builder search/calibration signals with monthly outcomes,
  outcome priors, proposal outcomes, and strategy-change history.

Checks:

- `rg "backtest_simulator|parameter_searcher|param_optimizer"`
- `rg "fold_generator|robustness_tester|leakage_detector|cost_model"`
- `rg "search_reports|search_signals|backtest_calibration|regime_parameter_analysis"`
- Run monthly evidence, parity, counterfactual, leakage, and cost tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- Replay authority comes from monthly artifact schemas and validators, not the
  old simplified simulator.
- No stale optimizer/search code remains in active imports.

Rollback:

- Restore only primitives that were actually shared, and rename them so they do
  not imply old WFO/search ownership.

## Phase 7 - Simplify Early-Warning Outcome Measurement

Goal: keep daily/weekly outcome measurement useful without letting it close
material strategy loops that now belong to monthly measurement.

Simplify, do not delete outright:

- `skills/auto_outcome_measurer.py`
- `schemas/outcome_measurement.py`
- `skills/portfolio_outcome_measurer.py`

Current code facts:

- `skills/auto_outcome_measurer.py` already documents monthly validation as
  authoritative and writes proposal outcomes with `outcome_source="early_warning"`.
- It still imports and can write to `BacktestCalibrationTracker`.
- It still has `detect_parameter_changes()` that scans WFO reports.
- `skills/portfolio_outcome_measurer.py` still writes source-less
  `portfolio_outcomes.jsonl` rows that legacy scoring treats as early warning.
- `skills/suggestion_scorer.py` already has an authoritative-only path, but it
  reads monthly/follow-up suggestion outcomes through `outcomes.jsonl`, not
  directly from `monthly_outcomes.jsonl`.

Remove from these paths:

- WFO parameter-change detection
- Backtest calibration fan-out
- Strategy-change finalization for material changes
- Duplicate status transitions that monthly outcome measurement now owns

Keep:

- Early-warning alerts
- Lightweight suggestion outcome context
- Portfolio-level context for monthly prompts
- Read compatibility for existing `outcomes.jsonl` where useful

Actions:

- Reclassify outputs as early-warning or context-only.
- Ensure monthly outcome measurement writes authoritative monthly verdicts.
- Prevent daily/weekly measurers from changing strategy-change ledgers.
- Update scorecard logic to prefer authoritative monthly outcomes.
- Add `outcome_source="early_warning"` to portfolio outcomes, or ensure all
  consumers treat source-less portfolio rows as early warning.
- Decide whether `SuggestionScorer.compute_authoritative_scorecard()` should
  also read `monthly_outcomes.jsonl` directly for strategy-change records that
  do not have suggestion IDs.

Checks:

- `rg "StrategyChangeLedger|IMPLEMENTED|finalize|backtest_calibration|wfo" skills/auto_outcome_measurer.py skills/portfolio_outcome_measurer.py schemas/outcome_measurement.py`
- `rg "monthly_outcomes|portfolio_outcomes|outcome_source" skills/suggestion_scorer.py skills/_outcome_utils.py`
- Run outcome utility, suggestion scorer, monthly outcome, portfolio outcome,
  and context builder tests.
- Run `pytest tests/ -q`.

Acceptance criteria:

- Monthly outcomes are authoritative for material strategy changes.
- Daily/weekly outcomes remain available as warning/context signals.
- Scorecards do not mix early-warning signals with final monthly verdicts unless
  explicitly marked.

Rollback:

- Restore early-warning reads if prompt context becomes too sparse.
- Do not restore ledger-finalization authority to daily/weekly measurers.

## Phase 8 - Wire Or Delete Monthly Repair Prompt Assembler

Goal: resolve the currently orphaned monthly repair prompt assembler.

Candidate:

- `analysis/monthly_repair_prompt_assembler.py`

Preferred action:

- Wire it into monthly validation when repairable package gaps exist, such as
  missing artifacts, inconclusive parity, or parseable model output defects.

Current code facts:

- The assembler is tested in `tests/test_monthly_candidate_generation.py`.
- `MonthlyValidationOrchestrator` does not call it.
- `MonthlyCandidatePipeline` validates `model_review.json` only if that file
  already exists in the artifact root.
- `orchestrator/backtest_invocation.py` defaults to
  `python -m backtests.shared.monthly_repair --manifest ...`; this is a
  backtest-repo command, not an in-repo model-review invocation.

Delete instead if:

- Monthly validation already handles repair deterministically.
- No caller needs an LLM repair package.
- Tests confirm no target-state workflow requires it.

Actions if wiring:

- Add an explicit repair state to monthly validation output.
- Call the assembler only for repairable gaps, not for hard failures.
- Require repaired output to pass the same validator as first-pass output.
- Add an agent workflow/preference entry for monthly model review if the review
  should call an LLM runtime.
- Make the monthly handler own the model-review invocation and write
  `model_review.json`, rather than relying on the backtest repo to produce it
  implicitly.
- Add tests for one repairable case and one hard-fail case.

Actions if deleting:

- Delete the assembler and tests.
- Document deterministic validation as the chosen replacement.

Checks:

- `rg "monthly_repair_prompt_assembler|repair prompt|repairable"`
- Run monthly model parser, validator, and validation orchestrator tests.

Acceptance criteria:

- The repair path is either actively wired and tested or fully removed.
- No orphan module remains.

Rollback:

- If repair is deleted and later needed, recreate it around monthly validation
  states rather than as an independent path.

## Phase 9 - Documentation, Configuration, And Command Cleanup

Goal: make repo guidance match the cleaned architecture.

Update:

- `AGENTS.md` or repo-level instructions if WFO/autonomous paths are described
  as active
- `README.md`
- `CLAUDE.md`
- `.env.example`
- `docs/adr/0001-monthly-evidence-replay-foundation.md`, if the final cleanup
  changes its "WFO as sensor" wording
- `docs/2026-05-11-workflow-learning-loop-target-state.md`, only if it needs
  a post-cleanup status note
- Any quick-start or command docs that advertise removed WFO commands
- Architecture diagrams that show WFO as a live scheduled strategy-change path
- Telegram/Discord/email renderer labels that expose `wfo_status` as an active
  operating state
- Agent preference UI text/callback values that expose `wfo` as a configurable
  active workflow

Actions:

- Replace WFO/autonomous wording with monthly learning loop wording.
- Mark daily/weekly loops as warning/context providers.
- Document historical data compatibility separately from active runtime.
- Replace `WFO_AGENT_PROVIDER` and `WFO_AGENT_MODEL` with a monthly model-review
  workflow setting only if the monthly model-review invocation is retained.

Checks:

- `rg "\bwfo\b|autonomous|parameter search|backtest calibration" docs AGENTS.md README.md CLAUDE.md .env.example comms schemas orchestrator`

Acceptance criteria:

- User-facing docs do not direct operators to removed pipelines.
- Remaining historical references are clearly labeled historical or migration
  context.

Rollback:

- Not usually needed. Revert only inaccurate doc edits.

## Phase 10 - Final Verification

Goal: prove the cleaned implementation is robust and integrated.

Required checks:

- `pytest tests/ -q`
- Monthly evidence package generation test
- Monthly candidate parsing and validation tests
- Monthly approval-gated candidate test with `AUTONOMOUS_ENABLED=false`
- Proposal approval and strategy-change ledger tests
- Deployment monitor tests
- Monthly outcome measurement tests
- Context builder tests for outcomes, forecasts, hypotheses, and scorecards
- Scheduler/catchup tests proving monthly jobs still run and WFO jobs do not
- `rg` checks proving removed modules are not imported

Final acceptance criteria:

- Full test suite passes.
- Removed runtime paths cannot create strategy-change proposals.
- Monthly pipeline owns material proposal, approval, deployment, monitoring,
  outcome measurement, and next-cycle context.
- Approval, PR, and deployment infrastructure works without enabling the legacy
  autonomous suggestion pipeline.
- Daily/weekly systems are context-only for material strategy learning.
- No stale compatibility modules or orphan monthly modules remain.

## Removal Matrix

| Component | Current classification | Action | Earliest phase |
| --- | --- | --- | --- |
| `schemas/coverage_manifest.py` | Alias | Delete | Phase 1 |
| `schemas/parity_report.py` | Alias | Delete | Phase 1 |
| `schemas/run_manifest.py` | Alias | Delete | Phase 1 |
| `schemas/wfo_results.py` | Legacy plus shared type | Extract, then delete WFO parts | Phase 2-4 |
| `schemas/wfo_config.py` | Legacy WFO config | Delete after WFO runtime removal | Phase 4 |
| `analysis/wfo_prompt_assembler.py` | Legacy WFO | Delete | Phase 4 |
| `analysis/wfo_report_builder.py` | Legacy WFO | Delete | Phase 4 |
| `skills/run_wfo.py` | Legacy WFO | Delete | Phase 4 |
| WFO scheduler/catchup/worker wiring | Legacy runtime | Disable, then remove | Phase 3 |
| `schemas/agent_preferences.py` WFO workflow | Legacy runtime preference | Remove or replace with monthly model-review workflow | Phase 3 |
| `schemas/agent_capabilities.py` WFO type | Legacy runtime capability | Remove or replace with monthly validation/review type | Phase 3 |
| `comms/*` `wfo_status` labels | User-facing stale status | Rename/remove | Phase 9 |
| Approval setup under `AUTONOMOUS_ENABLED` | Mis-owned shared infrastructure | Move out of autonomous block | Phase 2A |
| `skills/autonomous_pipeline.py` | Legacy autonomous processor | Extract shared schemas, then delete | Phase 2, 5 |
| `schemas/autonomous_pipeline.py` | Mixed shared schemas plus legacy name | Split into neutral schema modules, then delete compatibility surface | Phase 2, 5 |
| `skills/suggestion_backtester.py` | Legacy autonomous | Delete | Phase 5 |
| `skills/backtest_calibration_tracker.py` | Legacy calibration | Delete or read-only migrate | Phase 5 |
| `search_reports.jsonl` writers | Legacy feedback | Stop writing | Phase 5 |
| `search_signals.jsonl` writers | Legacy feedback | Stop writing | Phase 5 |
| `analysis/context_builder.py` search/backtest loaders | Legacy context | Replace with monthly outcomes/priors or mark historical read-only | Phase 6 |
| `skills/backtest_simulator.py` | Simplified simulator | Delete after monthly independence | Phase 6 |
| `skills/parameter_searcher.py` | Legacy search | Delete | Phase 6 |
| `skills/param_optimizer.py` | Legacy search | Delete | Phase 6 |
| `schemas/parameter_search.py` | Legacy search schemas | Delete after readers/tests are removed | Phase 6 |
| WFO-only `skills/fold_generator.py` code | Legacy helper | Delete or move shared primitives | Phase 6 |
| WFO-only `skills/robustness_tester.py` code | Legacy helper | Delete or move shared primitives | Phase 6 |
| WFO-only `skills/leakage_detector.py` code | Legacy helper | Delete or move shared primitives | Phase 6 |
| WFO-only `skills/cost_model.py` code | Legacy helper | Delete or move shared primitives | Phase 6 |
| `skills/auto_outcome_measurer.py` | Useful but over-authoritative | Simplify | Phase 7 |
| `schemas/outcome_measurement.py` | Useful but broad | Simplify | Phase 7 |
| `skills/portfolio_outcome_measurer.py` | Useful context | Simplify | Phase 7 |
| `analysis/monthly_repair_prompt_assembler.py` | Orphan monthly module | Wire or delete | Phase 8 |

## Modules To Keep

Keep and strengthen these as target-state components:

- `skills/learning_cycle.py`
- `schemas/learning_ledger.py`
- `skills/suggestion_tracker.py`
- `skills/proposal_ledger.py`
- `schemas/strategy_change_ledger.py`
- `schemas/backtest_artifacts.py`
- `schemas/market_data_manifest.py`
- `schemas/monthly_candidates.py`
- `schemas/monthly_model_review.py`
- `schemas/monthly_outcome.py`
- `schemas/monthly_run_manifest.py`
- `schemas/monthly_validation.py`
- `schemas/outcome_priors.py`
- `schemas/replay_parity.py`
- `schemas/telemetry_manifest.py`
- `orchestrator/backtest_invocation.py`
- `orchestrator/lineage_audit.py`
- `orchestrator/market_data_jobs.py`
- `skills/backtest_runner_client.py`
- `skills/coverage_manifest_writer.py`
- `skills/market_data_catalog.py`
- `skills/market_data_sync.py`
- `skills/monthly_candidate_pipeline.py`
- `skills/monthly_gap_attribution.py`
- `skills/monthly_validation_orchestrator.py`
- `skills/monthly_outcome_measurer.py`
- `skills/monthly_outcome_scorer.py`
- `skills/outcome_prior_store.py`
- `skills/replay_parity_checker.py`
- `skills/rollback_advisor.py`
- `skills/search_allocation_policy.py`
- `skills/strategy_change_ledger.py`
- `analysis/monthly_model_response_parser.py`
- `analysis/monthly_model_response_validator.py`
- `analysis/response_validator.py`
- `skills/approval_handler.py`
- `skills/approval_tracker.py`
- `skills/deployment_monitor.py`
- `skills/file_change_generator.py`
- `skills/github_pr.py`
- `skills/repo_workspace.py`
- `orchestrator/repo_task_runner.py`

## Main Risks

| Risk | Mitigation |
| --- | --- |
| Shared schema deleted with legacy module | Phase 2 extraction before deletion |
| Historical outcomes disappear from context | Preserve read-only adapters where needed |
| Monthly scheduling breaks while removing WFO | Disable WFO separately from deleting WFO code |
| Scorecards mix early warnings with final verdicts | Prefer authoritative monthly outcomes and label context-only records |
| Tests pass by deleting coverage | Rewrite removed-path tests as monthly-path tests before deletion |
| Docs still advertise removed commands | Dedicated documentation cleanup phase |

## Definition Of Done

Cleanup is complete when:

- The repo has one material strategy-learning authority: the monthly learning
  loop.
- Legacy WFO and autonomous parameter-search paths are absent from runtime code.
- Simplified simulator/search modules are removed or reduced to neutral shared
  primitives.
- Daily/weekly outcome measurement cannot finalize material strategy changes.
- Monthly evidence, proposals, approval, deployment monitoring, outcome
  measurement, and next-cycle context are fully tested.
- Full test suite passes.
- `rg` confirms no stale imports remain.
