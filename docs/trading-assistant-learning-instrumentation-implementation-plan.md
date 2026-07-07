# Trading Assistant Learning Instrumentation Implementation Plan

## Purpose

This plan describes how to upgrade bot and assistant instrumentation so the trading assistant can reliably propose high-value, evidence-grounded improvements that measurably improve trading performance over time.

The goal is not to build a sophisticated reporting system. The goal is to make the system a learning loop:

1. Live and shadow bots emit enough causal evidence to explain what happened, what almost happened, and why.
2. Daily and weekly assistant reviews generate hypotheses and improvement proposals with explicit expected effects.
3. Monthly backtests, replay, shadow evidence, and live follow-up measure those proposals after costs.
4. The measured outcomes update future search, priors, prompts, gates, and strategy discovery.

The target state is a system where past assistant outputs become measurable inputs to future trading performance.

## Reference Context

The workflow explainer describes the current architecture as a fail-closed, shadow-ready learning spine. The control-plane contracts, ledgers, gates, monthly orchestration, parity reports, deployment metadata, and validation matrix are substantial. The main remaining gap is that active bridges are still `shadow_validated`, not `approval_ready`, and real monthly outcomes have not accumulated enough to make the prior layer load-bearing.

The instrumentation assessment found that the repo already has strong learning primitives:

- Rich `TradeEvent`, `MissedOpportunityEvent`, `DailySnapshot`, and canonical envelope schemas.
- Bot-side trade, missed-opportunity, filter, indicator, orderbook, order/fill, post-exit, pipeline, health, deployment, and portfolio-rule events.
- Monthly candidate gates around lineage, replay parity, cost realism, drawdown, model review, decision parity, outcome priors, and plugin contracts.
- A performance-learning ledger capable of linking proposals, strategy changes, portfolio changes, monthly verdicts, and after-cost realized deltas.

The main gap is not raw event volume. The main gap is learning sufficiency: the assistant needs to know whether the evidence for a proposed improvement is complete enough to support causal learning.

## Existing Codebase Assets to Reuse

This plan should extend the existing learning substrate. It should not create a parallel instrumentation system.

| Existing asset | Current role | Reuse rule |
|---|---|---|
| `schemas/events.py` | Defines assistant-side `TradeEvent`, `MissedOpportunityEvent`, `DailySnapshot`, `PipelineFunnelSnapshot`, and `HealthReportSnapshot`. | Extend these only when a required field is genuinely missing. Prefer derived sufficiency artifacts over expanding every event. |
| `schemas/canonical_envelope.py` | Carries join keys such as decision IDs, order IDs, fill IDs, deployment ID, `runtime_join`, and `join_completeness`. | Use it as the canonical join-key vocabulary for all new auditors. Do not invent alternate join-key names. |
| `schemas/telemetry_manifest.py` and `orchestrator/lineage_audit.py` | Provide lineage-oriented telemetry authority from curated trades and missed opportunities. | Keep them as compatibility and lineage inputs. The new learning-sufficiency manifest composes with them rather than replacing them. |
| `orchestrator/action_handlers/daily_data.py` | Loads raw event types into daily curated outputs, including filter, indicator, orderbook, order/fill, post-exit, funnel, portfolio-rule, family, concurrent-position, macro-regime, and rolling portfolio files. | Reuse only the loading paths needed by material learning-capability checks. Do not duplicate raw-file discovery rules. |
| `skills/build_daily_metrics.py` | Builds daily summaries, filter summaries, order lifecycle summaries, latency, sizing, portfolio context, funnel reports, and portfolio-level analyses. | Add after-cost and sufficiency inputs here only where the data is naturally curated. Keep learning authority in the sufficiency manifest. |
| `skills/instrumentation_scorer.py` | Scores coarse readiness from curated file/field presence. | Use it only as a baseline hint. Do not make coarse readiness or dashboard-style scoring part of monthly learning authority. |
| `skills/monthly_candidate_pipeline.py` | Already gates candidates on telemetry lineage, market data, replay parity, realistic costs, drawdown, risk constraints, decision parity, plugin contract, outcome prior, and model review. | Add learning-sufficiency gates beside the existing gates. Do not weaken existing gates. |
| `skills/performance_learning_ledger.py` | Projects proposals, strategy changes, portfolio outcomes, monthly verdicts, source weekly IDs, expected deltas, and realized after-cost deltas into learning records. | Add sufficiency manifest IDs and supported/blocked capability IDs. Preserve existing validations around expected deltas, realized after-cost deltas, and source weekly IDs. |
| `schemas/monthly_run_manifest.py` | Already carries `telemetry_manifest_path`, `proposal_ids`, `deployment_id`, and `source_weekly_signal_ids`. | Add `learning_sufficiency_manifest_path` as an optional compatibility field instead of replacing `telemetry_manifest_path`. |
| `schemas/monthly_validation.py` | Already carries monthly artifact paths, replay parity, candidate gate reports, proposals, approvals, and adopted candidate ID. | Add sufficiency artifact paths and gate statuses here so monthly result consumers do not scrape arbitrary files. |
| Bot event contracts and sidecars | IBKR, K-stock, and crypto already map deployment, filter, orderbook, fill, portfolio-rule, and funnel event directories. | Reuse existing sidecar maps and event priorities. Add new runtime event types only when an approval-relevant capability cannot be derived from existing events. |
| Bot deployment metadata emitters | IBKR, K-stock, and crypto already emit or validate live deployment metadata. | Treat deployment metadata as an existing gate input. Extend with assistant proposal lineage only when the deployment was assistant-driven. |

## Corrections From the Review Pass

The first draft of this plan was directionally correct but understated what already exists. The implementation should make these corrections:

- Build `LearningSufficiencyManifest` as a thin authority artifact over existing telemetry, curated files, bot contracts, and monthly gates. Do not build a second event-ingestion pipeline.
- Reuse `LineageAuditor` for lineage counts and missing-field summaries, but expand learning authority beyond its current trade/missed-event scope.
- Reuse `DailyMetricsBuilder` and daily raw-to-curated loading for event taxonomy, post-exit merges, order lifecycle, funnel, and portfolio summaries.
- Reuse existing monthly candidate gates and add sufficiency checks as additional gates, not as replacements.
- Reuse existing performance-learning ledger projection and validation; add sufficiency context rather than creating a disconnected learning ledger.
- Treat `PipelineFunnelSnapshot` as an existing central schema. The work is cross-bot coverage and consistency, not inventing the concept.
- Treat `proposal_ids`, `source_weekly_signal_ids`, `deployment_id`, strategy-change records, and proposal-ledger records as existing central lineage. The gap is propagation into bot runtime lineage and trade/missed/order/fill events.
- Treat manual hook manifests as advisory context only. Runtime evidence coverage in the sufficiency manifest should become the trusted source.

## Non-Goals

- Do not turn the assistant into a dashboard-first reporting system.
- Do not block live trading because a telemetry call failed.
- Do not emit a high-cardinality firehose of every possible market state.
- Do not let daily or weekly evidence masquerade as monthly approval evidence.
- Do not promote structural candidates without approval-ready bridge maturity.
- Do not add instrumentation whose only payoff is documentation freshness, nicer reporting, or diagnostics that cannot change prompts, gates, ledger updates, or bridge promotion.

## Materiality Filter

Keep an implementation item only if it directly improves at least one of:

- evidence authority for a monthly gate,
- causal attribution from assistant proposal to trading outcome,
- after-cost measurement quality,
- denominator quality for a recurring decision class,
- prompt discipline that prevents unsupported recommendations,
- performance-learning ledger updates from real outcomes,
- approval readiness for an active bridge,
- diagnostics-only new-strategy discovery grounded in existing missed-opportunity and denominator evidence.

## Core Design Principles

### 1. Instrument for causal learning, not observability

Every required field should answer one of these questions:

- What decision was made?
- What alternatives were rejected?
- Which rule, risk constraint, filter, model state, or portfolio condition caused the decision?
- What execution actually happened?
- What would likely have happened under the rejected path?
- Which assistant proposal, experiment, deployment, and config produced the behavior?
- Did the realized result improve after costs?

### 2. Keep runtime instrumentation fail-open, but make learning fail-closed

Bot instrumentation should never crash trading. Missing evidence should be handled downstream by:

- quarantine records,
- learning-sufficiency gaps,
- instrumentation work items,
- blocked monthly candidate gates,
- lower confidence in LLM suggestions.

### 3. Prefer compact denominators over noisy event firehoses

The assistant needs denominators: how often setups appeared, filters passed, filters failed, portfolio/risk rules denied entries, orders filled, and trades closed. These should be emitted as compact daily or interval snapshots where full per-event emission would be expensive or noisy.

### 4. Make assistant outputs traceable into future outcomes

Every assistant proposal that can influence trading should carry stable IDs through:

- weekly signal,
- monthly search brief,
- proposal candidate,
- model review,
- strategy/config patch,
- deployment metadata,
- live/shadow events,
- monthly or live follow-up outcome,
- performance-learning ledger.

### 5. Treat after-cost performance as canonical

Learning should optimize after fees, slippage, tax, spread, funding, borrow, and realistic execution assumptions. Gross PnL can be diagnostic, but it should not be the default learning target.

## Target Architecture

The target architecture adds a learning-sufficiency layer between raw instrumentation and assistant reasoning.

```text
bot hooks
  -> canonical envelopes
  -> raw event store
  -> daily curation
  -> learning sufficiency manifests
  -> weekly evidence packets
  -> monthly search briefs
  -> monthly replay / shadow / model review
  -> performance learning ledger
  -> updated priors, prompts, gates, and strategy discovery
```

The new layer should answer:

- Is this bot/strategy/month eligible for learning?
- Which learning capabilities are supported by complete evidence?
- Which proposals are blocked because their causal evidence is incomplete?
- Which instrumentation gaps would unlock the most learning value?

## Workstream A: Learning Sufficiency Manifest

### Objective

Create a first-class `LearningSufficiencyManifest` for each bot, strategy, portfolio, and monthly window. This manifest becomes the authority for whether the assistant can learn from a slice of telemetry.

### Why This Matters

The current telemetry manifest primarily validates trade and missed-opportunity lineage. That is necessary but not sufficient. A candidate can have valid outcome lineage while missing the causal chain needed to explain why the outcome occurred.

### Proposed Manifest Shape

Add a schema under `packages/trading_assistant/src/trading_assistant/schemas/`, for example `learning_sufficiency.py`. The schema should link to the existing telemetry manifest instead of copying all lineage details into a second source of truth.

```python
class LearningSufficiencyManifest(BaseModel):
    manifest_id: str
    bot_id: str
    strategy_id: str = ""
    family_id: str = ""
    portfolio_id: str = ""
    run_month: str
    window_start: date
    window_end: date
    telemetry_manifest_path: str = ""
    telemetry_authoritative_eligibility: str = ""

    eligibility: Literal[
        "learning_authoritative",
        "diagnostics_only",
        "insufficient_lineage",
        "insufficient_joins",
        "insufficient_denominators",
        "insufficient_after_costs",
        "insufficient_outcomes",
        "insufficient_shadow_evidence",
    ]

    event_counts_by_type: dict[str, int]
    required_event_coverage: dict[str, CoverageCheck]
    lineage_coverage: CoverageCheck
    join_coverage: dict[str, CoverageCheck]
    denominator_coverage: dict[str, CoverageCheck]
    after_cost_coverage: CoverageCheck
    counterfactual_coverage: CoverageCheck
    proposal_trace_coverage: CoverageCheck
    deployment_metadata_coverage: CoverageCheck
    artifact_paths: dict[str, str]
    known_gaps: list[LearningGap]
    blocked_learning_capabilities: list[str]
    supported_learning_capabilities: list[str]
    evidence_paths: list[str]
```

### Required Coverage Checks

At minimum, the manifest should measure:

- `trade_outcome_lineage`: strategy/config/deployment/code lineage on completed trades.
- `missed_opportunity_lineage`: same for missed opportunities.
- `decision_to_trade_join`: completed trades joined to entry/exit decisions where applicable.
- `decision_to_order_join`: entry/exit decisions joined to order intents or explicit no-order reasons.
- `order_to_fill_join`: orders joined to fills or explicit non-fill/cancel/reject reasons.
- `risk_portfolio_join`: entries and misses joined to risk or portfolio rule decisions where those gates exist.
- `filter_decision_coverage`: filter names, thresholds, actual values, pass/fail, and margins.
- `denominator_coverage`: compact funnels for setups, gate passes/fails, risk denials, orders, fills, and closes.
- `after_cost_coverage`: fees, slippage, tax/funding, realistic execution assumptions, net PnL/R.
- `counterfactual_coverage`: missed-opportunity backfills and post-exit backfills.
- `proposal_trace_coverage`: suggestion/proposal/experiment/deployment IDs for assistant-influenced changes.
- `deployment_metadata_coverage`: live runtime metadata and approval metadata checks.

### Eligibility Rules

Use capability-specific eligibility rather than one global score.

Examples:

- `filter_threshold_learning` requires filter decisions with thresholds, actual values, margins, and outcome joins.
- `execution_learning` requires order/fill joins, slippage, latency, and cost fields.
- `sizing_learning` requires sizing inputs, account equity/risk basis, portfolio state, and realized R.
- `portfolio_interaction_learning` requires portfolio rules, concurrent positions, family snapshots, and allocation context.
- `new_strategy_discovery` requires repeated missed-opportunity or denominator clusters, control slices, after-cost estimates, and a replay plan.
- `approval_grade_strategy_change` requires all of the above plus monthly replay authority, parity, realistic costs, and approval-ready bridge maturity.

### Implementation Steps

1. Add `LearningSufficiencyManifest`, `LearningCapabilityStatus`, `CoverageCheck`, and `LearningGap` schemas.
2. Add `learning_sufficiency_manifest_path: str = ""` to `MonthlyRunManifest` and monthly validation/result schemas while keeping `telemetry_manifest_path`.
3. Add `LearningSufficiencyAuditor` under `packages/trading_assistant/src/trading_assistant/orchestrator/`.
4. Have the auditor call or compose with `LineageAuditor.build_telemetry_manifest()` for trade/missed lineage authority.
5. Reuse the raw and curated event taxonomy already handled by `orchestrator/action_handlers/daily_data.py`.
6. Read only the curated and raw event types required by the capability checks, not every file daily curation can load.
7. Compute coverage by strategy, family, portfolio, and run month.
8. Persist manifests under monthly artifact directories.
9. Add manifest path fields to monthly validation and monthly result schemas.
10. Make monthly candidate gates consume this manifest.

### Acceptance Criteria

- Every monthly validation run emits a learning-sufficiency manifest.
- Monthly candidate gates fail closed if a candidate requires a capability that is not learning-authoritative.
- The manifest lists actionable gaps with bot, strategy, event type, missing field, blocked learning capability, and expected learning value.
- Existing telemetry manifests remain supported for compatibility, but no longer serve as the only learning authority.
- The new auditor has regression tests proving it reuses existing trade/missed lineage results rather than computing incompatible lineage ratios.

## Workstream B: Causal Join Completeness

### Objective

Measure whether events form a usable causal graph from signal to outcome.

### Required Graph

For each completed trade where applicable:

```text
assistant proposal / experiment
  -> strategy config / deployment
  -> market bar / signal
  -> filter decisions
  -> risk and portfolio decisions
  -> order intent
  -> order lifecycle
  -> fill lifecycle
  -> trade outcome
  -> post-exit backfill
  -> monthly/live outcome verdict
```

For each missed opportunity:

```text
market bar / signal
  -> filter decisions
  -> risk or portfolio block
  -> hypothetical entry policy
  -> missed-opportunity backfill
  -> comparison against executed opportunities
```

### Implementation Steps

1. Define canonical join key groups:
   - `decision_keys`: `entry_decision_id`, `exit_decision_id`, `decision_id`, `signal_id`, `bar_id`.
   - `order_keys`: `intent_id`, `idempotency_key`, `client_order_ids`, `order_ids`, `exchange_order_ids`.
   - `fill_keys`: `fill_ids`, `entry_fill_ids`, `exit_fill_ids`, broker fill IDs.
   - `portfolio_keys`: `portfolio_rule_event_id`, `risk_decision_id`, `portfolio_decision_ref`.
   - `assistant_keys`: `suggestion_id`, `proposal_id`, `experiment_id`, `deployment_id`.
2. Add a join auditor that reads curated events plus raw order/fill/risk/portfolio events.
3. Reuse the canonical envelope key names and bot runtime refs when building join graphs.
4. Emit `join_completeness` into the learning-sufficiency manifest.
5. Update curated trade records to include explicit `join_completeness` summaries when possible.
6. Treat explicit `not_applicable` as valid only if the strategy contract declares that a join surface is unavailable.
7. Add examples for partial fills, inferred fills, canceled orders, rejected orders, no-order decisions, and portfolio-rule denials.

### Acceptance Criteria

- At least 95% of approval-grade completed trades have decision-to-order-to-fill joins or explicit justified exceptions.
- 100% of assistant-influenced deployments carry proposal or experiment lineage through deployment metadata and trade events.
- Missing join classes are visible in the monthly gate report, not only in logs.

## Workstream C: Decision Denominator and Funnel Snapshots

### Objective

Add compact denominator instrumentation so the assistant can distinguish:

- no opportunity,
- good opportunity rejected,
- bad opportunity correctly filtered,
- opportunity accepted but poorly executed,
- opportunity accepted and then managed poorly.

### Snapshot Contract

Emit per strategy, symbol, regime, and interval:

- bars received,
- indicators ready,
- setups detected,
- setup quality bands,
- gate evaluations,
- first failing gate,
- passed all strategy filters,
- risk denials,
- portfolio denials,
- orders submitted,
- orders rejected,
- partial fills,
- full fills,
- trades closed,
- post-exit backfills completed.

### Implementation Steps

1. Treat assistant-side `PipelineFunnelSnapshot` as the canonical schema.
2. Confirm sidecar mappings for `pipeline_funnel` in IBKR stock, momentum, swing, and crypto.
3. Add or adapt K-stock runtime exporter support for compact funnel snapshots if current exported session data cannot populate the canonical schema.
4. Reuse `DailyMetricsBuilder` funnel summary methods instead of creating a separate funnel aggregator.
5. Add weekly aggregation for funnel changes over time.
6. Include funnel evidence in weekly prompt packages and monthly search briefs.

### Acceptance Criteria

- Active strategies produce denominator snapshots for at least 90% of trading sessions.
- Weekly evidence packets show conversion rates from setup to close by strategy, symbol, regime, and major gate.
- The assistant can cite denominators when proposing threshold, sizing, or portfolio-rule changes.

## Workstream D: After-Cost Outcome Canonicalization

### Objective

Make after-cost performance the default learning target everywhere.

### Required Fields

For each completed trade:

- gross PnL,
- fees/commission,
- tax where applicable,
- slippage,
- spread or execution cost estimate,
- funding or borrow where applicable,
- net PnL,
- realized net R,
- cost model version,
- execution model version,
- whether fields are observed, inferred, or unavailable.

### Implementation Steps

1. Add a canonical `after_cost_outcome` block to trade curation.
2. Preserve legacy `gross_pnl` and `net_pnl` fields, but add `net_pnl_source` and `after_cost_status` so fallbacks are visible.
3. Remove or quarantine silent fallbacks where `net_pnl` equals `gross_pnl` because fees are missing.
4. Add `after_cost_coverage` to the learning-sufficiency manifest.
5. Update weekly and monthly prompt packages to prefer after-cost deltas.
6. Extend existing performance-learning ledger validation with sufficiency manifest references; do not duplicate its current expected/realized delta checks.

### Acceptance Criteria

- Approval-grade candidates require 100% after-cost outcome coverage for included completed trades.
- Diagnostics-only runs may use inferred costs, but the manifest must label them.
- Performance-learning records cannot be considered measured unless realized after-cost deltas are present.

## Workstream E: Proposal Trace Propagation

### Objective

Make assistant outputs measurable by attaching stable proposal lineage to every downstream artifact and event they influence.

### Required IDs

Use stable IDs consistently:

- `weekly_signal_id`,
- `monthly_search_brief_id`,
- `proposal_id`,
- `suggestion_id`,
- `hypothesis_id`,
- `experiment_id`,
- `variant_id`,
- `parameter_set_id`,
- `deployment_id`,
- `strategy_change_record_id`,
- `monthly_outcome_id`.

### Implementation Steps

1. Reuse existing `proposal_ids`, `source_weekly_signal_ids`, `deployment_id`, and proposal-ledger fields in monthly manifests and candidates.
2. Add an explicit assistant lineage block only where the existing fields are too scattered for bot runtimes to consume.
3. Require proposal IDs in generated config patches or deployment manifests when a patch is assistant-driven.
4. Ensure bot runtime lineage includes proposal/experiment/deployment IDs when deployed config came from an assistant proposal.
5. Ensure trade, missed-opportunity, order/fill, portfolio-rule, and deployment events can carry the assistant lineage block or equivalent stable IDs.
6. Add proposal-trace coverage to the learning-sufficiency manifest.
7. Extend the performance-learning ledger validation to reject measured assistant-driven records that cannot link back to proposal or strategy-change records.

### Acceptance Criteria

- 100% of assistant-driven deployments have proposal trace from weekly signal through outcome measurement.
- Weekly prompt assembly can summarize which prior suggestions improved, failed, regressed, or remain unmeasured.
- Monthly search allocation can be adjusted by measured proposal family, not just generic performance.

## Workstream F: Runtime Evidence Coverage

### Objective

Make learning authority depend on observed runtime evidence and configured sidecar support, not hand-written coverage claims.

### Why This Matters

The assistant needs a trustworthy map of which strategies support which learning capabilities. Documentation drift is not a material deliverable by itself; the material outcome is that unsupported or unobserved evidence cannot be used as authority.

### Implementation Steps

1. Define the minimum evidence classes required for each active learning capability:
   - trade entry,
   - trade exit,
   - missed opportunity,
   - filter decisions,
   - order/fill lifecycle,
   - portfolio/risk decisions,
   - pipeline funnel,
   - deployment metadata.
2. Reuse sidecar maps and runtime exporter outputs to determine which evidence can be emitted.
3. Add a runtime evidence scanner that checks whether expected event types appeared during the monthly window.
4. Emit coverage summaries into the learning-sufficiency manifest.
5. Mark unobserved required evidence as an `InstrumentationGap` with blocked learning capability and expected learning value.

### Acceptance Criteria

- The assistant receives generated coverage from runtime evidence and sidecar support, not hand-written manifest prose.
- Unobserved required evidence cannot satisfy monthly learning authority.
- Every gap is mapped to blocked learning capabilities and expected learning value.

## Workstream G: Instrumentation Gap Work Items

### Objective

Turn missing evidence into prioritized improvement tasks, not passive warnings.

### Implementation Steps

1. Add `InstrumentationGap` records to the learning-sufficiency output.
2. Include:
   - bot,
   - strategy,
   - event type,
   - missing field or join,
   - blocked learning capability,
   - affected candidate IDs,
   - frequency,
   - expected learning value.
3. Feed top gaps into weekly assistant prompts.
4. Allow monthly runs to emit instrumentation proposals only when the expected learning value exceeds the best available strategy tweak.
5. Track accepted instrumentation fixes in the performance-learning ledger with before/after capability status.

### Acceptance Criteria

- Weekly reviews include a short ranked list of learning blockers.
- The assistant can propose instrumentation fixes when those fixes unlock higher-value strategy learning.
- Resolved gaps show measurable increases in supported learning capabilities.

## Workstream H: Strategy Discovery From Existing Evidence

### Objective

Enable the assistant to propose new strategies without adding speculative runtime instrumentation.

### Required Evidence

New strategy proposals need more than trade outcomes. They need samples of the opportunity space already visible through missed opportunities, denominator snapshots, and after-cost outcome context:

- non-traded setup clusters,
- rejected high-quality signals,
- market states with repeated favorable follow-through,
- market states where current strategies are inactive,
- regime/session/symbol slices with high missed edge,
- negative examples where similar states failed.

### Implementation Steps

1. Build monthly discovery packets from existing missed-opportunity evidence, denominator snapshots, market/regime context, and after-cost outcomes.
2. Cluster by a compact material feature set:
   - symbol,
   - session,
   - regime,
   - setup type,
   - filter margin,
   - follow-through outcome,
   - after-cost estimate.
3. Include control slices that show similar states without favorable follow-through.
4. Write monthly `strategy_discovery_packet.json` artifacts only when recurring clusters meet minimum sample-size and after-cost thresholds.
5. Require new strategy proposals to cite discovery packets and baseline replay evidence.
6. Add a separate gate for `new_strategy_discovery` that is diagnostics-only until replay and bridge maturity support promotion.

### Acceptance Criteria

- The assistant can identify repeated unserved opportunity clusters with sample size, after-cost estimate, and control slices.
- New strategy proposals include a falsifiable hypothesis, target slice, exclusion slice, expected delta, risk budget, and replay plan.
- No new strategy can be promoted from discovery without replay parity and approval-ready bridge maturity.

## Workstream I: Schema Evolution and Artifact Wiring

### Objective

Make the new instrumentation authority compatible with existing artifacts and monthly workflows.

### Implementation Steps

1. Add optional fields first:
   - `learning_sufficiency_manifest_path`,
   - `learning_sufficiency_status`,
   - `supported_learning_capabilities`,
   - `blocked_learning_capabilities`.
2. Keep existing artifact readers tolerant of missing sufficiency fields.
3. Add artifact-authority registry entries for learning sufficiency and strategy discovery packets.
4. Add migration tests for old monthly manifests that only have `telemetry_manifest_path`.
5. Update docs and fixtures after code support exists.

### Acceptance Criteria

- Existing monthly fixtures and validation tests still pass without learning-sufficiency fields.
- New monthly runs emit sufficiency paths in run, validation, result, and artifact-index outputs.

## Workstream J: Prompt Contract and LLM Output Discipline

### Objective

Ensure LLM outputs use sufficiency evidence correctly.

### Implementation Steps

1. Add a prompt-section contract for `supported_learning_capabilities`, `blocked_learning_capabilities`, and `sufficiency_caveats`.
2. Require high-confidence recommendations to cite at least one supported learning capability and one evidence path.
3. Require diagnostics-only recommendations to label themselves as diagnostics-only.
4. Require instrumentation proposals when missing evidence blocks higher-value learning.
5. Add parser/validator checks for evidence path references and capability labels in structured LLM output.

### Acceptance Criteria

- Weekly and monthly LLM outputs cannot present blocked or diagnostics-only evidence as approval-grade evidence.
- Model-review parsing preserves sufficiency caveats in candidate gate reports.
- Prompt fixtures cover complete authority, partial authority, and insufficient-evidence cases.

## Bot-Specific Implementation Notes

### Crypto

Crypto already has strong instrumentation for filter decisions, pipeline funnels, funding, fees, slippage, fills, missed opportunities, and portfolio state.

Priority actions:

- Ensure pipeline funnel snapshots are emitted and curated consistently.
- Audit order/fill joins from canonical order intents through execution reports.
- Add proposal trace fields into runtime lineage for assistant-driven config changes.
- Make funding and fee coverage part of after-cost approval gating.
- Reuse `crypto_trader.instrumentation.pipeline_tracker.PipelineTracker` and `PipelineFunnelSnapshot`; the missing work is consistency and gate integration.

### K-Stock

K-stock already has a facade with entry, exit, missed, indicator, filter, orderbook, and order hooks.

Priority actions:

- Ensure OLR/KALCB deployment metadata and proposal IDs flow into trade and missed events.
- Validate KIS order IDs, fill IDs, slippage, tax, and fee fields for after-cost authority.
- Generate runtime evidence coverage from runtime exporter outputs and sidecar mappings.
- Add compact denominator snapshots if full per-gate emission is too noisy.
- Reuse `instrumentation/src/runtime_exporter.py` for runtime lineage and filter/fill context exports where possible.

### IBKR Stock and Momentum

The stock and momentum families already emit enriched entries/misses and standalone filter/orderbook/order events.

Priority actions:

- Confirm all active strategy families emit portfolio/risk decisions with joinable IDs.
- Normalize futures-specific costs and slippage into the canonical after-cost block.
- Require decision-to-order-to-fill coverage for execution-learning proposals.
- Reuse stock/momentum/swing sidecar support for `pipeline_funnel`, `portfolio_rule_check`, `inferred_fill`, `deployment`, `filter_decision`, and `orderbook_context`.

### IBKR Swing

Swing instrumentation has improved beyond some stale manifest claims. Current ATRSS, AKC_HELIX, and TPC paths populate important filter and portfolio fields.

Priority actions:

- Generate swing runtime coverage from current artifacts and sidecar support.
- Ensure TPC denominator snapshots compensate for intentionally sparse passed-gate events.
- Confirm AKC_HELIX gate decisions include threshold, actual value, pass/fail, and margin where available.
- Make portfolio state fields consistent across ATRSS, AKC_HELIX, TPC, and overlay decisions.

## Monthly Candidate Gate Changes

Update monthly candidate gating so `telemetry_lineage` becomes one input, not the whole evidence authority.

Add gates:

- `learning_sufficiency_manifest_present`
- `learning_capability_authority`
- `causal_join_completeness`
- `denominator_coverage`
- `after_cost_outcome_coverage`
- `proposal_trace_coverage`
- `counterfactual_backfill_coverage`
- `runtime_evidence_coverage`
- `instrumentation_gap_impact`

Example rule:

```text
If candidate.change_kind == "filter_threshold_change":
  require filter_threshold_learning == learning_authoritative
  require denominator_coverage for relevant strategy/symbol/regime
  require missed-opportunity backfill coverage
  require completed trade outcome joins
```

Example rule:

```text
If candidate.change_kind == "sizing_change":
  require sizing_learning == learning_authoritative
  require sizing_inputs coverage
  require portfolio state at entry
  require realized net R
  require drawdown and concentration context
```

Example rule:

```text
If candidate.change_kind == "structural_change":
  require new_strategy_discovery support
  require strategy discovery packet
  require replay parity
  require plugin contract maturity
  require approval_ready bridge before adoption
```

## Weekly Prompt and Evidence Packet Changes

Weekly prompts should receive:

- top supported learning capabilities,
- top blocked learning capabilities,
- ranked instrumentation gaps,
- denominator shifts,
- after-cost outcome deltas,
- proposal outcome history,
- material discovery clusters,
- caveats from the sufficiency manifest.

The assistant should be instructed to:

- cite sufficiency status for each high-confidence recommendation,
- avoid confident causal claims when denominators or joins are missing,
- propose instrumentation improvements when they unlock more learning than a strategy tweak,
- distinguish diagnostics from approval-grade evidence.

## Performance-Learning Ledger Changes

Strengthen ledger projection and validation:

- Require measured records to include realized after-cost deltas.
- Require proposal lineage for assistant-influenced changes.
- Store the learning capability that supported each proposal.
- Store sufficiency manifest IDs and blocked capability IDs.
- Record whether a proposal was rejected because evidence was insufficient, not because the idea was bad.
- Track accepted instrumentation fixes as learning investments with before/after capability deltas.

## Implementation Acceptance Matrix

This matrix is the implementation contract. A row is complete only when its acceptance signal and validation path both pass.

| ID | Area | Existing code to reuse | Required implementation | Acceptance signal | Validation path |
|---|---|---|---|---|---|
| AM-01 | Telemetry compatibility | `TelemetryManifest`, `LineageAuditor`, `MonthlyRunManifest.telemetry_manifest_path` | Keep existing telemetry manifest generation and add sufficiency as an additional artifact. | Existing monthly validation fixtures still pass unchanged. | Monthly validation regression tests. |
| AM-02 | Sufficiency schema | `schemas/telemetry_manifest.py`, monthly schemas | Add `LearningSufficiencyManifest`, `LearningCapabilityStatus`, `CoverageCheck`, `LearningGap`. | Schema validates complete, partial, and empty manifest fixtures. | New schema unit tests. |
| AM-03 | Monthly artifact path wiring | `monthly_run_manifest.py`, `monthly_validation.py`, artifact index logic | Add optional `learning_sufficiency_manifest_path` and related status fields, and ensure phase/build tools create artifact directories before writing sufficiency sidecars or manifests. | New monthly runs expose sufficiency paths without breaking old manifests, and phase manifest generation succeeds from a clean empty output root. | Old/new manifest migration tests and a clean-output-root manifest-builder test. |
| AM-04 | Lineage composition | `LineageAuditor.build_telemetry_manifest`, `lineage_utils.py` | Have the sufficiency auditor consume existing lineage summaries; when a caller supplies an existing telemetry manifest, compose from it or write to a separate output without overwriting that artifact. | Sufficiency lineage ratios match telemetry lineage for trade/missed scopes, and supplied telemetry manifests remain stable unless explicitly regenerated. | Golden fixture comparing both outputs plus a prebuilt-telemetry-manifest fixture that verifies no accidental overwrite. |
| AM-05 | Event coverage input | `daily_data.py`, `build_daily_metrics.py` | Reuse existing raw/curated event taxonomy for capability-relevant coverage checks; do not maintain auditor-local duplicate file taxonomies that can drift from daily curation; include active runtime raw and curated `risk_decision` / `risk_decisions` evidence as portfolio/risk evidence. | Auditor sees trade, missed, filter, indicator, order/fill, orderbook, post-exit, funnel, portfolio-rule, risk-decision, and deployment events where required by a learning capability, and taxonomy drift is caught before reports can pass. Curated risk-decision summaries must not synthesize placeholder join IDs that can satisfy authority without a real canonical risk or portfolio reference. | Multi-event raw-to-curated fixture plus a drift guard comparing sufficiency-auditor inputs to daily-data curation inputs, including raw `risk_decision` and curated risk-decision summary fixtures that contribute to `risk_portfolio_join`, plus a negative curated-summary fixture with no canonical risk/portfolio ID that fails instead of joining on `unknown` or other placeholders. |
| AM-06 | Join graph | Canonical envelope keys, bot runtime refs, order/fill events | Build decision-to-order-to-fill and portfolio/risk join coverage by matching canonical IDs across event classes, not by field presence alone; `decision_to_order_join` must start from decision/order-intent surfaces rather than completed trades alone; explicit no-order reasons carried on decision records must be accepted even without synthetic order records. | Approval-grade trade slices show >=95% required joins or declared exceptions; records with mismatched decision/order/trade/fill/risk IDs fail even when all join-key fields are present; any orphan canonical join ref in decision/order/trade/risk joins fails unless recorded as an explicit justified exception, even when aggregate joined-target coverage remains >=95%; canceled, rejected, explicit no-order, and risk-denial decisions with no completed trade still satisfy their relevant join evidence when canonical IDs or explicit reasons match. Completed trade records cannot be the only source side that lets `decision_to_order_join` pass; order-matched fill records cannot satisfy fill authority without a canonical fill ID; terminal canceled/rejected/no-order records cannot satisfy order-to-fill authority without a canonical order ID; plural canonical order-ID fields count as distinct fill-required identities unless explicitly grouped as aliases or equivalents with a canonical singular order identity; mere co-presence of singular and plural order IDs is not alias proof; fills keyed by a proven singular alias identity must not be treated as orphaned; and repeated lifecycle rows for the same canonical order ID are grouped before final fill-required versus terminal classification. | Join-completeness unit and integration tests, including negative fixtures for mismatched IDs, ratio-threshold orphan joins that otherwise reach >=95%, pure orphan fills, mixed valid-plus-orphan fills, plural order-ID rows with only partial fill evidence, explicit singular-plus-plural alias rows where a fill keyed by the canonical singular order ID is accepted, parent/container singular order rows with multiple plural child order IDs where partial child fills still fail, order-matched fills missing canonical fill IDs, and terminal no-fill orders missing canonical order IDs; positive lifecycle fixtures where submitted-plus-filled rows for one order and submitted-plus-canceled rows for one order count as one canonical order; terminal-order/no-completed-trade fixtures; decision-record-only no-order fixtures; `risk_decision` join fixtures; standalone risk-denial/no-completed-trade fixtures; trade-only decision-to-order false-pass fixtures; and explicit justified exceptions. |
| AM-07 | `not_applicable` semantics | Strategy plugin contracts, candidate gates | Allow `not_applicable` only when declared in strategy capability contracts. | Undeclared missing joins fail; declared unavailable joins pass as diagnostics. | Contract fixture tests. |
| AM-08 | Denominator snapshots | `PipelineFunnelSnapshot`, crypto `PipelineTracker`, IBKR sidecar maps | Standardize cross-bot funnel summaries and compute session-level coverage against an explicit expected active-session source such as a trading calendar, scheduler output, or session manifest, not observed event presence; wire that source through phase manifest generation and monthly validation runs. | Active strategies have denominator snapshots for >=90% of expected active sessions or an explicit unsupported reason; sessions with no emitted records still count as missing, phase-generated and monthly artifacts record the expected-session source path, and a single snapshot cannot satisfy a multi-session window. | Daily, weekly, clean phase-builder, and monthly-orchestrator funnel tests, including a no-record expected-session failure fixture, a one-snapshot multi-session failure fixture, a >=90% pass fixture, and a phase-builder source-path propagation fixture. |
| AM-09 | K-stock denominator support | K-stock runtime exporter/session artifacts | Add or derive canonical funnel snapshots for OLR/KALCB if source data supports it. | K-stock sufficiency manifest reports denominator coverage, not unknown. | K-stock runtime-exporter fixture. |
| AM-10 | After-cost block | Trade schemas, `build_daily_metrics.py`, crypto funding/fee fields, KIS tax/fee fields | Add canonical `after_cost_outcome` with observed/inferred/unavailable status and an observed numeric net after-cost value when authority is claimed. | Approval-grade candidates have 100% after-cost coverage for included trades; observed status/source flags without a realized net after-cost value are diagnostics-only. | After-cost fixture tests by venue, including missing-net-PnL negative fixtures. |
| AM-11 | Net-PnL fallback visibility | Existing `net_pnl` and `gross_pnl` summary fields | Add `net_pnl_source` and `after_cost_status`; quarantine silent gross-as-net fallbacks and source/status-only records for authority. | Fallbacks and source/status-only records appear as diagnostics-only and cannot satisfy approval-grade cost gates. | Daily curation and candidate gate tests with explicit gross-as-net and observed-source-without-value failures. |
| AM-12 | Proposal trace central fields | Monthly search brief, candidate, proposal ledger, strategy-change ledger | Normalize assistant lineage block where existing IDs are too scattered. | Proposal lineage is available in monthly artifacts and candidate packets. | Monthly search/candidate fixture tests. |
| AM-13 | Proposal trace in bot events | Bot lineage contexts, deployment metadata, trade/missed/order/fill events | Carry proposal/experiment/deployment lineage into runtime event payloads for assistant-driven changes. | 100% of assistant-driven deployments trace from proposal to live/shadow event. | Bot-specific runtime emitter integration tests for trade, missed-opportunity, order, inferred-fill, and portfolio-rule JSONL outputs; the assistant-lineage source inventory artifact; scoped PTG-4 runtime evidence JSONL generated through actual IBKR stock emitters from existing ledger IDs; and PTG-4 checks that scoped generated evidence records assistant lineage when those event classes are observed. |
| AM-14 | Runtime evidence coverage | Sidecar maps, runtime exporter outputs, curated artifacts | Generate capability coverage from configured sidecar/exporter support plus observed runtime evidence, with an explicit support state for each required evidence class: unsupported, supported-but-unobserved, or observed; classify sufficiency-consumed runtime event classes as learning-authority evidence, learning-gap diagnostics, or operational-health telemetry; carry the same classification in source-side bot sidecar/exporter contracts instead of relying only on priority or scope; enforce `learning_authority` source classifications for capability-required event classes and their canonical aliases before support states, monthly gates, prompt authority, or priors can treat runtime evidence as authoritative across generated capability support, direct runtime support payload shapes (`capabilities`, `runtime_evidence_support`, `evidence_classes`, `events`, and `support`), every source side of multi-source checks such as decision-to-order and decision-to-trade joins, and prompt/prior inputs that might otherwise rely on hand-written or legacy manifests; generate phase support capability rows from the same shared runtime-authority mapping, or an equivalent learning-authority-only projection, so diagnostic-only or operational event names are not advertised as configured learning requirements; wire support artifacts through phase manifest generation and monthly validation runs. | Generated runtime evidence coverage exists for every active family, records the sidecar/exporter support source and observed evidence paths, and cannot be inferred from hand-written manifests or event observation alone. Operational-health and broad diagnostic events remain diagnostics-only and cannot satisfy monthly learning authority unless a capability requirement explicitly maps them to a learning evidence class. Source-side bot contracts expose an event-value classification for emitted event classes; canonical support derivation credits learning-authority aliases such as `inferred_fill` -> `fill`, `portfolio_rule_check` / `risk_decision` -> `portfolio_rule`, and `deployment` -> `deployment_metadata` for both generated manifests and direct support payloads; generated capability rows do not list diagnostic-only or operational event names such as `decision_event` or `post_exit` as configured learning requirements unless they are explicitly mapped to learning-authority consumed runtime classes; every event class consumed by a capability-required join or coverage check is learning-authority classified; manifests missing generated runtime support are diagnostics-only for prompts and priors; and any capability-required runtime class classified as operational-health, broad diagnostic, missing classification, or only matched by a non-authority alias blocks monthly gates and priors instead of being upgraded by observation alone. | Runtime evidence fixture tests covering configured support, observed evidence, supported-but-unobserved gaps, unsupported contracts, monthly-run support wiring, an observation-only false-pass case, source-side event-classification coverage for emitted bot event types, generated-capability and direct alias-keyed support fixtures for canonical event classes, clean generated-support capability rows that exclude diagnostic-only or operational event names from configured learning requirements, multi-source join fixtures where non-authority `filter_decision` or other side inputs cannot satisfy decision joins, missing-`runtime_evidence_support` prompt/prior fixtures, and negative fixtures proving heartbeat, health, snapshot, resource, missing-classification, or other ops-only/diagnostic emissions cannot satisfy learning capability gates, prompt authority, or priors without mapped core learning evidence. |
| AM-15 | Coverage freshness | Recent monthly artifacts and sidecar support | Treat stale or unobserved supported evidence as insufficient for learning authority unless the strategy contract declares it unsupported or unavailable, and require monthly gates to consume the generated support-state artifact. | Stale evidence and supported-but-unobserved evidence cannot satisfy capability gates; declared unavailable evidence remains diagnostics-only and records the declaring contract. | Coverage-freshness fixture tests for stale observed evidence, supported-but-unobserved evidence, declared unavailable evidence, and monthly candidate gating from generated support states. |
| AM-16 | Gap records | Lineage warnings, sufficiency known gaps | Emit ranked `InstrumentationGap` records with blocked capability and expected learning value. | Weekly prompt can list top learning blockers. | Gap-ranking unit tests and prompt fixture. |
| AM-17 | Monthly candidate gates | `monthly_candidate_pipeline.py` existing gates | Add sufficiency gates without weakening telemetry, parity, cost, drawdown, risk, plugin, prior, or model-review gates; keep the change-kind capability map aligned with the required coverage checks and shared materiality predicates. | Unsupported candidates fail closed with precise sufficiency reasons, including `filter_threshold_change` when counterfactual coverage or decision-to-trade joins are missing. | Candidate gate matrix tests for each change kind, including threshold candidates missing counterfactuals and joins. |
| AM-18 | Weekly prompt contract | Weekly evidence loaders and prompt assemblers | Add supported/blocked capability summaries and sufficiency caveats; package operational-health context under bot-scope-neutral keys unless the data is explicitly crypto-only, so non-crypto health telemetry cannot be mislabeled as crypto learning context. | LLM prompt includes authority labels and blocked-learning caveats, and operational-health summaries are labeled by actual bot scope while remaining quarantine/context signals rather than learning authority. | Prompt snapshot tests, including a non-crypto health-summary fixture proving all-bot health context is not stored only under a crypto-specific key. |
| AM-19 | LLM output validation | Model review parser and structured response validators | Require evidence paths and capability labels for high-confidence recommendations. | Diagnostics-only evidence cannot be presented as approval-grade. | Parser/validator tests. |
| AM-20 | Performance-learning ledger | `performance_learning_ledger.py`, proposal/strategy/portfolio ledgers | Add sufficiency manifest IDs, supported capabilities, blocked capabilities, and capability status to projected learning records; classify measured records as assistant-driven from material approval evidence, source records, proposal IDs, strategy-change IDs, or assistant lineage/deployment metadata that contains proposal, suggestion, experiment, or strategy-change IDs. Do not treat a bare `deployment_id` as assistant-driven by itself. | Measured records include after-cost deltas and sufficiency context; measured assistant-driven records without sufficiency context are rejected instead of updating priors, including records that carry proposal or strategy-change IDs even if `source_records` is empty; legacy or human-authored records with only `deployment_id` are not false-failed as assistant-driven. | Ledger projection and validation tests for monthly manifests with and without sufficiency context, negative fixtures for ID-bearing measured records missing sufficiency context, and negative fixtures proving `deployment_id` alone does not imply assistant-driven authority. |
| AM-21 | Strategy discovery packets | Missed opportunities, funnel summaries, monthly artifacts | Create `strategy_discovery_packet.json` from existing missed-opportunity, denominator, regime, and after-cost evidence only when clusters meet minimum sample-size and after-cost materiality thresholds, and reuse the same thresholds when monthly gates and PTG-6 checks load any packet source. | New-strategy proposals cite recurring clusters, control slices, after-cost estimates, and replay plan; weak, empty, stale, or externally supplied below-threshold packets cannot satisfy the new-strategy gate or PTG-6. | Discovery packet fixture tests for material clusters, below-threshold generated clusters, below-threshold externally supplied packets, missing controls, missing after-cost estimates, and PTG-6 weak-packet rejection. |
| AM-22 | Artifact authority | Artifact authority registry and monthly artifact index | Register sufficiency and discovery artifacts. | Artifact paths are contained, indexed, and verifier-readable. | Artifact containment tests. |
| AM-23 | Pilot evidence fixtures | Existing raw/curated directories and approval audit fixtures | Create bounded production-derived fixture windows for the selected pilot bridge. | Pilot fixtures cover complete and insufficient evidence cases needed by the approval audit. | Pilot fixture tests. |
| AM-24 | Approval-ready pilot | Existing bridge maturity and deployment metadata gates | Promote one bridge only after production fixtures, scheduled shadow evidence, metadata, parity, and sufficiency pass. | One bridge reaches `approval_ready`; structural adoption remains blocked before that. | Approval audit fixture and scheduled shadow evidence tests. |
| AM-25 | Regression safety | Existing monthly validation matrix, CI workflow, and bot instrumentation fail-open behavior | Preserve current validation results and runtime exception safety, and wire the learning-sufficiency focused tests plus PTG-3/PTG-4/PTG-5/PTG-6 scripts into CI against freshly generated clean-output artifacts and freshly written chained gate reports; run PTG-7 as a separate closeout gate. | Existing tests pass, bot instrumentation exceptions remain non-fatal, and the blocking CI path propagates PTG-3/PTG-4/PTG-5/PTG-6 script failures, gate regressions, gate-script crashes, missing reports, stale reports, focused fixture regressions, and linked checklist regressions as workflow failures; expected-blocked compatibility reports may exist only as non-promotion snapshots; PTG-4 cannot pass while required Phase 6 runtime-lineage checklist items are incomplete; PTG-7 blocks final closeout promotion only. | Full targeted regression suite plus CI workflow coverage that passes the fresh manifest index and report paths into the PTG scripts, explicitly validates each fresh PTG-3/PTG-4/PTG-5/PTG-6 report, verifies linked finite-checklist completion for PTG-3/PTG-4/PTG-5/PTG-6 instead of only recording checklist sections as metadata, verifies PTG-5 consumes the same PTG-3/PTG-4 reports and PTG-6 consumes the same PTG-5 report generated in that CI run, verifies PTG regressions fail the workflow rather than only writing blocked reports, and verifies PTG-7 consumes the same clean PTG-6 report as a closeout-only gate. |
| AM-26 | Secret and payload hygiene | Existing redaction and event validation helpers | Ensure new artifacts do not leak secrets and obey payload size constraints. | Sufficiency/discovery artifacts contain no secrets and remain bounded. | Redaction and payload-size tests. |

## Finite Implementation Checklist

Status note: checked items reflect implementation observed in the current workspace on 2026-07-04. Existing phase-gate reports are evidence snapshots, not promotion authority, when linked acceptance rows or finite checklist items below remain unchecked. A phase still cannot pass its transition gate unless the linked acceptance rows, validation paths, and repository integration all pass.

Current validation artifacts:

- Baseline capability matrix: `artifacts/learning_sufficiency/baseline_capability_matrix.json`.
- Phase 2 manifest index: `artifacts/learning_sufficiency/phase2_manifests/manifest_index.json`.
- Phase gate reports: `artifacts/learning_sufficiency/ptg3_gate_report.json`, `artifacts/learning_sufficiency/ptg4_gate_report.json`, `artifacts/learning_sufficiency/ptg5_gate_report.json`, `artifacts/learning_sufficiency/ptg6_gate_report.json`, and `artifacts/learning_sufficiency/ptg7_gate_report.json`.
- PTG-7 pilot artifacts: `artifacts/learning_sufficiency/ptg7_pilot/`.

Current readiness caveat: the generated PTG-3, PTG-5, and PTG-6 reports are expected to fail closed until denominator/funnel forwarding, after-cost authority, and prior phase readiness are complete. PTG-4 validates assistant-lineage propagation mechanics, not live learning sufficiency; active manifests with no observed core runtime event paths remain diagnostics-only learning inputs. PTG-7 checklist enforcement is closeout-only; it does not block or prove Phase 6 runtime propagation. A checked implementation item here means the contract, fixture, or CI wiring exists; it does not by itself mean the performance-learning rollout is promotion-ready.

Additional review caveat: shared runtime-authority guardrails are now wired through the auditor, monthly gate, prompt, and prior paths; remaining work is on unfinished rollout capabilities rather than a known source-class false pass in those reviewed paths. Known remaining blockers are missed-opportunity causal graph completion, denominator/funnel forwarding, venue-specific after-cost source mappings and production authority evidence, pilot closeout evidence, and final removal of expected-blocked tolerance from the blocking CI path after linked checklist sections are complete. Terminal order joins without completed trades when canonical order IDs exist, decision-record-only explicit no-order evidence, trade-only `decision_to_order_join` false-pass rejection, mixed valid-plus-orphan fill rejection in `order_to_fill_join`, generic cross-record join orphan rejection at the >=95% threshold, plural canonical order-ID fill-denominator handling, explicit order-ID alias proofing versus parent/container child-order rejection, downstream runtime telemetry value classification and source-side runtime telemetry classification metadata emission, exact-name source event-value enforcement plus generated-capability and direct alias credit in canonical support states and monthly gates, prompt/prior rejection of non-authority observed support entries, multi-source join source-class authority through the shared runtime authority resolver, prompt/prior fail-closed handling for manifests missing generated runtime support, phase runtime-support capability-row cleanup so diagnostic-only event names such as `decision_event` or `post_exit` are not advertised as configured learning requirements, cross-bot setup-to-fill and fill-to-close prompt evidence from available DailyMetricsBuilder funnel summaries, weekly prompt health-context key cleanup for all-bot `health_summary.json` inputs, order-matched fill records missing canonical fill IDs, terminal no-fill order canonical ID enforcement, repeated order-lifecycle grouping before `order_to_fill_join` classification, raw and curated `risk_decision` ingestion into portfolio/risk evidence, standalone risk-denial/no-completed-trade `risk_portfolio_join` evidence when canonical IDs exist, placeholder-only curated risk-decision join rejection, avoiding `deployment_id`-only assistant-driven ledger classification, ID-bearing measured ledger sufficiency validation from proposal/strategy-change IDs, prebuilt telemetry-manifest composition without overwrite, ID-matched completed-trade joins, observed numeric after-cost authority, daily-data taxonomy reuse, and stricter fresh-report CI assertions are now implemented in the current workspace.

### Phase 0: Baseline and Inventory

- [x] List active bot, family, strategy, portfolio, and bridge scopes.
- [x] Export current event type inventory per active scope.
- [x] Export current curated-file inventory per active scope.
- [x] Record current monthly validation matrix status before changes.
- [x] Identify strategies with no denominator/funnel evidence.
- [x] Identify strategies with no order/fill join evidence.
- [x] Identify strategies with no after-cost authority.
- [x] Write baseline capability matrix artifact.

### Phase 1: Schema and Compatibility

- [x] Add `schemas/learning_sufficiency.py`.
- [x] Add `CoverageCheck`.
- [x] Add `LearningCapabilityStatus`.
- [x] Add `LearningGap`.
- [x] Add `LearningSufficiencyManifest`.
- [x] Add optional `learning_sufficiency_manifest_path` to `MonthlyRunManifest`.
- [x] Add optional sufficiency fields to monthly validation/result schemas.
- [x] Add artifact authority entries for sufficiency manifests.
- [x] Add schema unit tests for complete manifest.
- [x] Add schema unit tests for diagnostics-only manifest.
- [x] Add schema unit tests for insufficient-lineage manifest.
- [x] Add migration test for old monthly manifest with no sufficiency field.

### Phase 2: Auditor and Coverage Inputs

- [x] Implement `LearningSufficiencyAuditor`.
- [x] Compose with `LineageAuditor.build_telemetry_manifest`.
- [x] Respect prebuilt telemetry manifests without overwriting or recomputing them when supplied by monthly orchestration.
- [x] Replace auditor-local event taxonomy/path scanning with reuse of `daily_data.py` or `DailyMetricsBuilder` curation inputs.
- [x] Add taxonomy drift guard comparing sufficiency-auditor event inputs to daily-data curation inputs.
- [x] Load curated `trades.jsonl`.
- [x] Load curated `missed.jsonl`.
- [x] Load raw/curated filter decisions.
- [x] Load raw/curated order and fill events.
- [x] Load raw/curated orderbook context.
- [x] Load raw/curated post-exit events.
- [x] Load raw/curated pipeline funnel snapshots.
- [x] Load raw/curated portfolio-rule events.
- [x] Load raw `risk_decision` / `risk_decisions` events into portfolio/risk evidence.
- [x] Load curated risk-decision summary/evidence into portfolio/risk evidence.
- [x] Reject curated risk-decision summary joins that rely on placeholder IDs such as `unknown` instead of real canonical risk or portfolio IDs.
- [x] Load deployment metadata artifacts.
- [x] Emit event counts by type.
- [x] Emit supported and blocked learning capabilities.
- [x] Emit ranked learning gaps.
- [x] Write manifest to monthly artifact root.
- [x] Create per-scope phase artifact roots before writing runtime support sidecars.
- [x] Add clean-output-root phase manifest builder regression test.

### Phase 3: Join Completeness

- [x] Define canonical decision key set.
- [x] Define canonical order key set.
- [x] Define canonical fill key set.
- [x] Define canonical portfolio/risk key set.
- [x] Define canonical assistant lineage key set.
- [x] Build completed-trade decision-to-order-to-fill join graph by matching canonical IDs, not field presence.
- [x] Wire `decision_to_order_join` from filter/entry/exit decision records to terminal order records, including canceled or rejected orders with no completed trade.
- [x] Treat explicit no-order reasons carried on decision records as valid `decision_to_order_join` evidence without requiring a synthetic order record.
- [x] Reject `decision_to_order_join` evidence when a completed trade record is the only source-side decision evidence.
- [ ] Build missed-opportunity causal graph.
- [x] Handle partial fills.
- [x] Handle inferred fills.
- [x] Handle canceled orders.
- [x] Handle rejected orders.
- [x] Handle explicit no-order decisions.
- [x] Handle portfolio-rule denials.
- [x] Handle standalone risk-denial or risk-decision records with no completed trade or missed-opportunity target.
- [x] Implement declared `not_applicable` semantics.
- [x] Add join-completeness golden fixtures.
- [x] Add negative join fixtures for mismatched decision/order/trade IDs and orphan fills.
- [x] Reject `order_to_fill_join` evidence when any fill references an unknown order, including mixed windows that also contain valid order/fill joins.
- [x] Reject `order_to_fill_join` evidence when an order-matched fill record lacks a canonical fill ID such as `fill_id` or broker execution ID.
- [x] Reject `order_to_fill_join` evidence when a terminal canceled, rejected, expired, or no-order record lacks a canonical order ID.
- [x] Group repeated order lifecycle rows by canonical order ID before classifying `order_to_fill_join` fill-required versus terminal evidence.
- [x] Reject generic cross-record join evidence when any target record carries an orphan canonical join reference, even if aggregate joined-target coverage remains >=95%, unless the orphan is recorded as an explicit justified exception.
- [x] Treat plural canonical order ID fields as multiple fill-required identities, and add a negative `order_to_fill_join` fixture where one of multiple order IDs lacks fill evidence.
- [x] Add explicit singular-plus-plural alias proof handling and fixtures: positive when canonical singular and plural IDs are declared aliases/equivalents, and negative when the singular ID is a parent/container and plural child order IDs are only partially filled.
- [x] Add monthly gate failure for insufficient joins.

### Phase 4: Denominator and Funnel Coverage

- [ ] Confirm central `PipelineFunnelSnapshot` is sufficient for target denominator fields.
- [ ] Confirm crypto funnel emission and sidecar forwarding.
- [ ] Confirm IBKR stock `pipeline_funnel` sidecar forwarding.
- [ ] Confirm IBKR momentum `pipeline_funnel` sidecar forwarding.
- [ ] Confirm IBKR swing `pipeline_funnel` sidecar forwarding.
- [ ] Add or derive K-stock funnel support.
- [x] Reuse `DailyMetricsBuilder` funnel summaries.
- [x] Add crypto weekly funnel aggregation and prompt evidence for setup-to-fill and fill-to-close conversion.
- [x] Add cross-bot weekly funnel aggregation from available IBKR/K-stock `funnel_analysis.json` summaries while bot-side forwarding remains tracked separately.
- [x] Add denominator coverage checks to sufficiency manifest.
- [x] Replace event-presence and observed-session denominator checks with explicit expected-active-session `>=90%` coverage.
- [x] Add denominator fixtures for no-record expected-session failure, one-snapshot multi-session failure, and `>=90%` pass.
- [x] Wire explicit expected-session source paths through `MonthlyValidationRequest` and `MonthlyValidationOrchestrator`.
- [x] Add monthly-orchestrator fixture proving expected-session source paths reach the sufficiency manifest.
- [x] Wire explicit expected-session source paths through phase manifest generation and add a clean-output regression proving generated manifests record the source path instead of `missing_expected_active_session_source`.
- [x] Add cross-bot prompt evidence for setup-to-fill and fill-to-close conversion by strategy/family when DailyMetricsBuilder funnel summaries are present.

### Phase 5: After-Cost Authority

- [ ] Define canonical `after_cost_outcome`.
- [ ] Add `net_pnl_source`.
- [ ] Add `after_cost_status`.
- [ ] Preserve existing gross/net fields for compatibility.
- [x] Mark gross-as-net fallback as diagnostics-only.
- [x] Require observed numeric after-cost outcome values for authority; status/source flags alone remain diagnostics-only.
- [ ] Add K-stock tax/fee source mapping.
- [ ] Add crypto funding/fee/slippage source mapping.
- [ ] Add futures/IBKR commission/slippage source mapping.
- [x] Add after-cost coverage checks to sufficiency manifest.
- [x] Add negative after-cost fixture for observed status/source with missing net after-cost value.
- [x] Add after-cost monthly candidate gate.
- [x] Add after-cost daily curation fixtures.

### Phase 6: Proposal Trace Propagation

- [x] Inventory existing proposal IDs in monthly search briefs.
- [x] Inventory existing proposal IDs in monthly candidates.
- [x] Inventory existing proposal IDs in proposal ledger.
- [x] Inventory source weekly signal IDs in monthly manifests.
- [x] Define assistant lineage block.
- [x] Add assistant lineage block to assistant-driven deployment manifests.
- [x] Generate `assistant_lineage_source_inventory.json` with proposal, weekly-signal, candidate, strategy-change, and runtime-lineage seed locations.
- [x] Propagate proposal IDs into bot runtime lineage.
- [x] Propagate assistant lineage into trade events.
- [x] Propagate assistant lineage into missed-opportunity events.
- [x] Propagate assistant lineage into order/fill events where available.
- [x] Propagate assistant lineage into portfolio-rule events where available.
- [x] Add proposal-trace coverage checks.
- [x] Add ledger validation for missing proposal trace on measured assistant-driven records.
- [x] Add sufficiency manifest IDs, supported capabilities, blocked capabilities, and capability status to performance-learning records.
- [x] Add performance-ledger validation rejecting sourced and ID-bearing measured records missing sufficiency context.
- [x] Extend performance-ledger sufficiency-context validation to measured records identified as assistant-driven by `proposal_ids`, `strategy_change_record_ids`, or source records.
- [x] Require assistant lineage/proposal metadata before treating `deployment_id`-only measured records as assistant-driven.
- [x] Extend PTG-4 to fail closed when generated runtime event evidence for active scopes lacks assistant lineage or required event evidence.
- [x] Add bot-side trade, missed, order, fill, and portfolio-rule fixture coverage proving assistant lineage propagation for assistant-driven changes, and wire that fixture signal into PTG-4.
- [x] Add bot-specific runtime emitter integration tests proving actual trade, missed-opportunity, order, inferred-fill, and portfolio-rule JSONL records carry assistant lineage.
- [x] Add scoped PTG-4 runtime evidence JSONL generated through actual IBKR stock emitters from existing source IDs, with `bot_id`, `strategy_id`, and assistant trace fields.

### Phase 7: Runtime Evidence Coverage and Gap Records

- [x] Define required evidence classes per active learning capability.
- [x] Classify sufficiency-consumed runtime event classes as learning-authority evidence, learning-gap diagnostics, or operational-health telemetry.
- [x] Add source-side bot sidecar/exporter event-value classification metadata so emitted event classes are marked as learning-authority evidence, learning-gap diagnostics, or operational-health telemetry in generated support artifacts.
- [x] Enforce exact-name source-side event-value classifications when deriving support states and before monthly gates consume runtime evidence as authoritative; observed events classified as operational-health, broad diagnostic, or missing a learning-authority source mapping must not satisfy capability authority.
- [x] Normalize generated capability source event-value classifications through canonical event aliases when deriving runtime support, so `inferred_fill`, `portfolio_rule_check`, `risk_decision`, and `deployment` can credit `fill`, `portfolio_rule`, and `deployment_metadata` only when the source alias is classified as learning-authority evidence.
- [x] Normalize direct runtime support payload aliases (`runtime_evidence_support`, `evidence_classes`, `events`, and `support`) against source event-value classifications, so alias-keyed support declarations retain the classified source alias instead of defaulting only to the canonical class name.
- [x] Add negative manifest and monthly-gate fixtures proving operational-health or missing-classification emissions cannot satisfy learning capability authority without mapped core evidence.
- [x] Add generated-capability alias-credit manifest and monthly-gate fixtures proving learning-authority aliases satisfy their canonical runtime evidence class without letting ops-only aliases satisfy authority.
- [x] Add direct alias-keyed runtime support fixtures proving `events`/`evidence_classes`/`support` payloads credit learning-authority aliases and false-block ops-only or missing-classification aliases.
- [x] Enforce source-side event-value classifications for every event class consumed by multi-source learning checks, especially `filter_decision` in `decision_to_order_join` / `decision_to_trade_join` and any counterfactual side inputs, instead of checking only representative `CHECK_EVENT_TYPE` classes.
- [x] Add monthly-gate negative fixtures proving multi-source join checks fail when any consumed source class is operational-health, broad diagnostic, or unclassified, even if the representative check event class is learning-authority observed.
- [x] Enforce source-side event-value classifications before prompts or priors consume manifests that carry `runtime_evidence_support`; non-authority observed support entries downgrade prompt/prior context to diagnostics-only.
- [x] Add negative prompt and performance-prior fixtures proving operational-health, broad diagnostic, or missing-classification `runtime_evidence_support` entries cannot satisfy prompt authority or performance-learning priors without mapped core evidence.
- [x] Fail closed prompt and performance-prior consumption when a learning-sufficiency manifest lacks `runtime_evidence_support`, so hand-written or legacy manifests cannot self-authorize supported capabilities.
- [x] Add negative prompt and performance-prior fixtures proving manifests missing `runtime_evidence_support` are diagnostics-only even when they list supported learning capabilities.
- [x] Align generated phase runtime-support capability rows with the shared runtime authority resolver, so generated support artifacts do not list diagnostic-only or operational event names such as `decision_event` or `post_exit` as configured learning requirements.
- [ ] Add a clean-output regression proving generated runtime support capability rows contain only learning-authority consumed runtime classes or approved learning-authority aliases for monthly authority checks.
- [x] Reuse sidecar maps and runtime exporter outputs for coverage support.
- [x] Add manifest fields for required evidence support state: unsupported, supported-but-unobserved, or observed.
- [x] Record sidecar/exporter support-source paths and observed evidence paths per required evidence class.
- [x] Implement runtime event-evidence scanner.
- [x] Emit runtime coverage summaries into sufficiency manifests.
- [x] Emit `InstrumentationGap` records.
- [x] Rank gaps by blocked capability and frequency.
- [x] Include top gaps in weekly prompt package.
- [x] Add fixtures that distinguish unsupported, supported-but-unobserved, and observed runtime evidence.
- [x] Add an observation-only false-pass fixture for runtime evidence coverage.
- [x] Wire explicit runtime-support artifact paths through `MonthlyValidationRequest` and `MonthlyValidationOrchestrator`.
- [x] Add monthly-orchestrator fixture proving runtime support-source paths reach the sufficiency manifest.

### Phase 8: Monthly Gates and Prompt Discipline

- [x] Add `learning_sufficiency_manifest_present` gate.
- [x] Add `learning_capability_authority` gate.
- [x] Add `causal_join_completeness` gate.
- [x] Add `denominator_coverage` gate.
- [x] Add `after_cost_outcome_coverage` gate.
- [x] Add `proposal_trace_coverage` gate.
- [x] Add `counterfactual_backfill_coverage` gate.
- [x] Add `runtime_evidence_coverage` gate.
- [x] Wire `counterfactual_coverage` and `decision_to_trade_join` into `filter_threshold_learning`.
- [x] Add prompt section for supported capabilities.
- [x] Add prompt section for blocked capabilities.
- [x] Add prompt section for sufficiency caveats.
- [x] Rename or split weekly prompt all-bot health summary data so non-crypto `health_summary.json` records are exposed under a bot-scope-neutral operational-health key, not only `crypto_health_summaries`.
- [x] Add a weekly prompt fixture proving non-crypto health summaries remain operational-health quarantine context and are not mislabeled as crypto-specific learning evidence.
- [x] Add LLM output validator for evidence paths.
- [x] Add LLM output validator for authority labels.
- [x] Make PTG-3 through PTG-7 fail when required finite-checklist sections remain incomplete, instead of only recording required checklist sections as report metadata.
- [x] Add CI job coverage for the learning-sufficiency focused pytest suite and PTG-3/PTG-4/PTG-5/PTG-6 gate scripts.
- [x] Wire CI PTG-3/PTG-4/PTG-5/PTG-6 runs to the fresh clean-output manifest index and fresh chained gate report outputs, including PTG-5 consuming the same PTG-3/PTG-4 reports and PTG-6 consuming the same PTG-5 report.
- [x] Explicitly validate the fresh PTG-4 report in CI and fail on any missing, stale, or crashed PTG-4 report instead of accepting PTG-4 only as an expected PTG-5 prior-phase failure.
- [x] Build and validate the fresh assistant-lineage source inventory and scoped PTG-4 runtime evidence in CI before PTG-5 consumes PTG-4.
- [x] Run PTG-7 in CI as a separate closeout gate with a fresh report and the same clean PTG-6 report generated in that CI run.
- [x] Assert each fresh PTG script exit code matches its report status so abnormal exits cannot be hidden by expected-blocked report handling.
- [ ] Remove expected-blocked tolerance from the blocking CI path for PTG-3/PTG-4/PTG-5/PTG-6 once linked checklist sections are complete.
- [ ] Add CI regression assertion proving a PTG-3/PTG-4/PTG-5/PTG-6 script failure fails the workflow, not only writes a blocked report.

### Phase 9: Strategy Discovery

- [x] Define discovery packet fields from existing missed-opportunity, denominator, regime, and after-cost evidence.
- [x] Build recurring opportunity clusters from missed opportunities.
- [x] Build recurring opportunity clusters from denominator snapshots.
- [x] Add control slices.
- [x] Add after-cost estimate fields.
- [x] Write `strategy_discovery_packet.json` only when recurring clusters meet sample-size, control-slice, and after-cost materiality requirements.
- [x] Enforce the same minimum sample-size and after-cost materiality thresholds in the new-strategy gate for generated and externally supplied packets.
- [x] Add new-strategy discovery gate.
- [x] Add a below-threshold externally supplied packet negative gate fixture.
- [x] Add PTG-6 weak externally supplied packet rejection check.
- [x] Add discovery prompt fixtures.

### Phase 10: Pilot and Closeout

- [ ] Select one approval-ready bridge pilot.
- [ ] Expand production-derived fixtures for pilot.
- [ ] Run scheduled shadow cycles for pilot.
- [ ] Emit approval-grade optimizer manifests for pilot.
- [ ] Rerun approval audit.
- [ ] Verify one bridge can reach `approval_ready`.
- [ ] Verify structural adoption remains blocked before approval readiness.
- [ ] Refresh performance-learning ledger from real monthly outcomes.
- [x] Wire PTG-7 to the shared finite-checklist completion check before closeout promotion.
- [x] Add `--ptg6-report` to PTG-7 so closeout enforcement consumes the fresh upstream gate report in CI.
- [x] Update this plan with implementation status and links to artifacts.

## Phase Transition Gates

The checklist above is the task-level tracker. These gates are the explicit movement criteria for the rollout phases below; the checklist references point back to the finite implementation checklist above. Downstream exploratory work may begin early, but a rollout phase is not promoted until its gate is green. A gate is green only when the linked acceptance-matrix rows pass, the listed checklist sections are complete, and the rollout exit criteria for the current phase are met.

| Gate | Move by rollout phase | Required acceptance rows | Required finite-checklist sections | Promotion criteria |
| --- | --- | --- | --- | --- |
| PTG-0 | Phase 0 -> Phase 1 | AM-01, AM-14, AM-15, AM-25 | Phase 0 | Baseline event inventory, runtime evidence coverage, material missing capabilities, and current validation matrix are captured without introducing a validation regression. |
| PTG-1 | Phase 1 -> Phase 2 | AM-02, AM-03, AM-22, AM-25 | Phase 1 | Compatible schemas, artifact registry paths, optional monthly fields, and legacy-manifest validation all work together. |
| PTG-2 | Phase 2 -> Phase 3 | AM-04, AM-05, AM-16, AM-25 | Phase 2 | Sufficiency manifests are emitted for every active scope from a clean output root, compose with the existing telemetry manifest, and produce deterministic coverage and gap records. |
| PTG-3 | Phase 3 -> Phase 4 | AM-06, AM-07, AM-08, AM-09, AM-10, AM-11, AM-25 | Phases 3, 4, and 5 | Join coverage, denominator coverage from explicit expected sessions, cross-bot funnel coverage, and after-cost authority are implemented and fixture-validated; insufficient slices are flagged as diagnostics or gate blockers instead of treated as evidence. |
| PTG-4 | Phase 4 -> Phase 5 | AM-12, AM-13, AM-20, AM-25 | Phase 6 | Proposal lineage is traceable from monthly artifacts into runtime events and the performance-learning ledger can distinguish approved, shadowed, reverted, and expired proposals. |
| PTG-5 | Phase 5 -> Phase 6 | AM-14, AM-15, AM-16, AM-17, AM-18, AM-19, AM-25 | Phases 7 and 8 | Monthly gates consume sufficiency artifacts, runtime coverage includes configured sidecar/exporter support states and freshness, gap records are emitted, and prompt/output validators enforce evidence authority labels. |
| PTG-6 | Phase 6 -> Phase 7 | AM-21, AM-22, AM-25 | Phase 9 | Strategy discovery packets are diagnostics-only, and new-strategy proposals must cite recurring clusters, control slices, after-cost estimates, and a replay or shadow plan; generated and externally supplied packets use the same materiality gate. |
| PTG-7 | Phase 7 -> Closeout | AM-23, AM-24, AM-25, AM-26, Definition of Done | Phase 10 pilot and closeout items | One bridge reaches `approval_ready` only after approval metadata, production-derived fixtures, scheduled shadow cycles, and approval-grade optimizer manifests are present. |

Universal blockers:

- AM-25 regression safety blocks every transition.
- AM-26 artifact hygiene blocks any transition that publishes or promotes new evidence artifacts.
- Clean-output-root artifact generation blocks PTG-2 and every downstream transition.
- CI gate checks must consume the fresh clean-output manifest index and freshly written chained gate reports, and blocking CI jobs must propagate PTG script failures as workflow failures; stale default artifacts or expected-blocked snapshot reports cannot satisfy AM-25.
- PTG-4 cannot satisfy AM-13 or AM-25 until it checks real bot runtime event lineage propagation and all material Phase 6 checklist items are complete.
- PTG-7 checklist enforcement is separate closeout enforcement for Phase 10; it cannot substitute for, block, or prove Phase 6 runtime event propagation.
- Operational-health and broad diagnostic telemetry cannot satisfy AM-14, AM-15, prompt authority, or performance-learning priors unless explicitly mapped to every learning capability evidence class it is used for, including each source side of multi-source joins.
- Learning-sufficiency manifests missing generated `runtime_evidence_support` cannot satisfy prompt authority or performance-learning priors.
- Generated runtime-support capability rows that advertise diagnostic-only or operational event names as configured learning requirements cannot satisfy AM-14 clean-output closeout, even if the runtime authority resolver blocks prompt or gate false passes elsewhere.
- Priority, criticality, or scope labels alone cannot satisfy AM-14 source-side event-value classification.
- Any unchecked rollout exit criterion for the current phase blocks promotion even if implementation tasks are complete.
- Any unchecked material item in a required finite-checklist section blocks the linked transition gate, even if an older gate artifact reports `pass`.
- Any missing gap record for a known coverage deficit blocks promotion, because unknown absence is worse for learning than explicit insufficiency.
- Observation-only runtime coverage cannot satisfy AM-14 or AM-15, observed-session-only denominator coverage cannot satisfy AM-08, and builder-only discovery materiality cannot satisfy AM-21.

## Rollout Plan

The finite checklist above is the task-level tracker. This rollout plan is the higher-level sequencing.

### Phase 0: Baseline Audit and Capability Matrix

Deliverables:

- Current event inventory by bot and strategy.
- Current runtime evidence coverage by bot, strategy, and learning capability.
- Current learning capability matrix.
- Baseline monthly validation matrix snapshot.

Exit criteria:

- The team can see which strategies support which learning capabilities today.
- The first version of the capability matrix is checked in as an artifact.

### Phase 1: Compatible Schema and Artifact Wiring

Deliverables:

- `LearningSufficiencyManifest` schema.
- Optional sufficiency fields in monthly run and validation schemas.
- Artifact authority registration.
- Backward-compatible tests for old manifests with only `telemetry_manifest_path`.

Exit criteria:

- Existing monthly fixtures still pass.
- New artifact paths can be emitted without breaking older monthly readers.

### Phase 2: Sufficiency Auditor and Coverage Inputs

Deliverables:

- `LearningSufficiencyAuditor`.
- Lineage composition with `LineageAuditor`.
- Event coverage across trade, missed, filter, indicator, order/fill, orderbook, post-exit, funnel, portfolio-rule, and deployment events required by learning capabilities.
- Monthly manifest output.

Exit criteria:

- Monthly validation emits a manifest for every active scope.
- Manifest capabilities and gaps are deterministic under golden fixtures.

### Phase 3: Join, Denominator, and After-Cost Authority

Deliverables:

- Join completeness auditor.
- Canonical after-cost block.
- Denominator coverage checks.
- Curated output updates.
- Bot-side or curation-side adapters for all active families.
- Weekly funnel conversion summaries where they support threshold, sizing, or portfolio proposals.

Exit criteria:

- Execution, sizing, and filter proposals are blocked when required joins or after-cost fields are missing.
- Diagnostics-only paths still run and label their evidence quality.
- Cross-bot weekly assistant prompts can compare setup-to-fill and fill-to-close conversion by strategy/family, not only crypto-only summaries.
- Filter-threshold proposals cite denominators.

### Phase 4: Proposal Trace and Ledger Context

Deliverables:

- Proposal lineage fields in monthly search briefs, candidate manifests, deployment metadata, and bot runtime lineage.
- Ledger validation for proposal trace.
- Sufficiency manifest IDs and capability status in performance-learning records.

Exit criteria:

- Assistant-driven deployments are traceable from proposal to outcome.
- Untraceable outcomes are not used to update assistant priors.

### Phase 5: Monthly Gates and Gap Work Items

Deliverables:

- Sufficiency gates added to monthly candidate pipeline.
- Runtime evidence coverage scanner.
- Coverage summaries in sufficiency manifests.
- Instrumentation gap records.
- Weekly prompt integration for top gaps.

Exit criteria:

- Unsupported candidates fail closed with precise sufficiency reasons.
- Top learning blockers are emitted as explicit work items.

### Phase 6: Prompt Contract and Strategy Discovery Packets

Deliverables:

- Prompt sections for supported/blocked learning capabilities and sufficiency caveats.
- Structured output validation for evidence paths and authority labels.
- Monthly strategy discovery packets derived from existing missed-opportunity, denominator, regime, and after-cost evidence.
- New-strategy proposal gate.

Exit criteria:

- LLM outputs cannot present diagnostics-only evidence as approval-grade evidence.
- New strategy proposals cite repeated opportunity clusters, control slices, after-cost estimates, and replay plan.
- Discovery remains diagnostics-only until replay and approval-ready bridge gates pass.

### Phase 7: Approval-Ready Bridge Pilot

Deliverables:

- Select one active bridge as the pilot.
- Expand production-derived fixtures.
- Run scheduled shadow cycles.
- Emit approval-grade optimizer manifests.
- Rerun approval audit.

Exit criteria:

- One bridge reaches `approval_ready`.
- First real monthly outcome cycle can update priors with authority.

## Validation Strategy

### Unit Tests

- Schema validation for sufficiency manifests and gap records.
- Backward-compatible validation for old monthly manifests.
- Coverage computation for each required event class.
- Join completeness for complete, partial, and explicitly-not-applicable graphs.
- After-cost coverage and fallback detection.
- Proposal trace validation.
- Prompt authority-label and evidence-path validation.

### Golden Fixtures

Create fixture windows for:

- complete learning-authoritative slice,
- missing lineage,
- missing order/fill joins,
- missing denominator snapshots,
- missing after-cost fields,
- missing runtime evidence coverage,
- assistant proposal with full trace,
- assistant proposal with broken trace.
- diagnostics-only prompt evidence,
- strategy discovery packet derived from existing evidence.

### Integration Tests

- Daily raw-to-curated rebuild includes all event types required by the sufficiency auditor.
- Weekly prompt packages include sufficiency caveats.
- Monthly validation runs can pass explicit expected-session and runtime-support artifacts into the sufficiency auditor.
- Monthly candidate pipeline blocks unsupported candidate types.
- Performance-learning ledger accepts only measured after-cost outcomes for learning authority.
- Monthly artifact indexes include sufficiency and discovery artifacts.
- Phase manifest generation succeeds from a clean empty output root.

### Regression Tests

- Existing telemetry manifests remain valid.
- Existing monthly validation matrix still passes.
- Diagnostics-only learning still produces useful summaries without approval authority.
- Live bot runtimes remain fail-open for instrumentation exceptions.
- Existing proposal, strategy-change, portfolio-outcome, and performance-learning ledgers remain readable.

## Definition of Done

The instrumentation upgrade is complete when:

- Every active bot/strategy/month has a learning-sufficiency manifest.
- Existing telemetry manifests and old monthly artifacts remain backward compatible.
- Monthly gates use learning capability authority, not only trade/missed lineage.
- Completed trades have measurable causal join coverage.
- Missed opportunities and post-exit events have tracked backfill coverage.
- After-cost outcomes are canonical for learning authority.
- Assistant-driven proposals are traceable into deployments and outcomes.
- Weekly prompts can distinguish supported learning from blocked learning.
- LLM outputs carry evidence-path and authority labels for high-confidence recommendations.
- Instrumentation gaps are ranked and actionable.
- Strategy discovery packets are derived from existing evidence and remain diagnostics-only until replay and maturity gates pass.
- At least one bridge is promoted from `shadow_validated` to `approval_ready`.
- The performance-learning ledger contains real, non-fixture monthly outcomes that update future search and priors.

## Priority Summary

Highest-value sequence:

1. Implement `LearningSufficiencyManifest`.
2. Add backward-compatible monthly artifact wiring.
3. Add causal join and after-cost coverage checks.
4. Gate monthly candidates by supported learning capability.
5. Add compact denominator snapshots.
6. Propagate proposal lineage into deployments and bot events.
7. Generate runtime evidence coverage and instrumentation gap work items.
8. Add prompt authority labels and validators.
9. Build strategy discovery packets from existing evidence.
10. Promote one bridge to `approval_ready` and let real monthly outcomes accumulate.

This sequence keeps the system focused on learning leverage. It improves the assistant's ability to make useful recommendations without turning the project into a general-purpose reporting platform.
