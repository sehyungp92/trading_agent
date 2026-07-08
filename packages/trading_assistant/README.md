# Trading Assistant

An agent system that monitors multiple trading bots across VPSes, automatically triages errors, and continuously analyses trading performance to propose parameter, structural, and portfolio-level improvements — closing the loop between observation and action over time.

## What It Does

### Monitors bot health and catches problems early

- Ingests every trade, missed opportunity, daily snapshot, and error event from each bot in real-time via a relay service
- Classifies error severity deterministically (CRITICAL/HIGH/MEDIUM/LOW) and routes by complexity — obvious fixes get a Claude-generated diagnosis and draft PR, complex issues get flagged for human investigation
- Tracks error rates per bot with sliding-window spike detection (>3/hour auto-promotes severity)
- Monitors bot heartbeats and alerts on gaps (>2h warning, >4h critical)
- Runs morning and evening proactive scans for unusual losses (>2σ from 30-day mean) and repeated error patterns

### Analyses trading performance across multiple dimensions

Every day, raw event data is transformed into a curated analysis package per bot covering:

- **Trade quality**: winners, losers, process failures (quality score < 60), and notable missed opportunities
- **Regime analysis**: PnL, win rate, and trade count broken down by market regime
- **Filter effectiveness**: per-filter block count, saved PnL, and missed PnL
- **Time-of-day patterns**: hourly PnL and win rate distributions
- **Slippage**: per-symbol and per-hour spread distributions (mean, median, p75, p95)
- **Exit efficiency**: premature exit rates by exit reason and regime
- **Signal attribution**: per signal factor win rate, PnL contribution, and trend
- **Drawdown episodes**: equity curve segmentation with duration, root cause attribution, and recovery tracking
- **Root cause distribution**: across a 21-type controlled taxonomy

The agent runtime can be switched per workflow between Claude Max, Codex Pro, Z.AI Coding Plan, and OpenRouter-backed Claude-compatible models, while reports still flow through Telegram, Discord, or email.

### Runs monthly evidence and replay validation

The monthly learning loop is the authoritative path for material strategy changes. It assembles a frozen evidence package with:

- Market-data coverage manifests and telemetry lineage
- Live-vs-backtest replay parity checks
- Objective deltas, counterfactuals, and regime slices
- Leakage and cost-realism signals from the monthly replay artifacts
- Structured candidate proposals with calibrated predictions and rollback plans

Approval-gated monthly candidates create Telegram approval cards and GitHub PRs through shared approval infrastructure. After merge, deployment monitoring tracks the bot for regressions and can create rollback PRs if performance degrades.

### Proposes structural and portfolio-level improvements

Beyond parameter tuning, the strategy engine runs 19 detectors across four levels:

- **Signal quality** (4): alpha decay, signal decay, component signal decay, factor correlation decay
- **Execution quality** (4): exit timing issues, microstructure issues (fill quality, adverse selection), stress entry patterns, position sizing mismatches
- **Regime awareness** (3): regime config effectiveness, regime transition cost, time-of-day patterns
- **Portfolio structure** (5): family imbalance, correlation concentration, drawdown tier miscalibration, coordination gaps, heat cap utilisation
- **Pattern detection** (3): drawdown patterns, filter interaction effects, correlation breakdown

These findings feed into the configured weekly analysis provider, which produces structural suggestions (e.g., "add a regime-aware exit rule", "split this strategy into two variants"). Portfolio-level analysis computes cross-bot exposure concentration, crowding alerts, and risk-parity allocation recommendations with Calmar ratio tilt. Cross-bot transfer proposals identify validated patterns from one bot that may apply to others, scored by regime distribution overlap and historical transfer success rates.

Weekly structural suggestions surface as advisory hypotheses unless they are promoted into monthly candidate validation. Production implementation still requires isolated candidate testing, parity evidence, and human approval.

### Learns from its own suggestions over time

Every suggestion the system makes is tracked through a full lifecycle:

```
                         Strategy Engine (19 detectors)
                                    ↓
                    suggestion with ID → SuggestionTracker
                          ↓                       ↓
                   parameter change          structural proposal
                          ↓                       ↓
                   monthly validation       6 validation gates
                   (replay evidence,        (hypothesis track record,
                    parity checks)           category win rate, confidence,
                          ↓                   acceptance criteria)
                   monthly ledgers                ↓
                          ↓                  report for human review
                   Telegram approval card
                          ↓
              User approves/rejects via Telegram → lifecycle update
                          ↓
  ┌─────────────────── Learning Cycle (weekly) ───────────────────┐
  │  ground truth snapshot → ledger delta → route pending items   │
  │  parameter candidates → monthly val.   structural → tracker   │
  └───────────────────────────────────────────────────────────────┘
                          ↓
         AutoOutcomeMeasurer → 7-day pre/post comparison → outcomes.jsonl
                          ↓
         Outcome-derived lessons → learning ledger entries
         Correction patterns (count ≥ 3) → synthesised lessons
                          ↓
         ExperimentManager auto-conclusion (when statistically decisive)
           → accept/reject linked suggestion
           → record hypothesis outcome (positive/negative)
                          ↓
  ┌─────────── Fed back into every subsequent prompt ─────────────┐
  │  ground truth trends · search reports · outcome measurements  │
  │  category scorecards · prediction accuracy · convergence      │
  │  loop health metrics · instrumentation readiness · bias data  │
  └───────────────────────────────────────────────────────────────┘
```

The system won't re-suggest rejected ideas. Categories with low success rates get their suggestions stripped before delivery. Confidence levels are capped based on historical prediction accuracy. Hypotheses that accumulate rejections and negative outcomes are auto-retired. Measured outcomes and persistent correction patterns are automatically synthesised into learning ledger entries, so the system's accumulated experience informs future analysis without manual curation.

Crucially, the learning system is self-correcting — it monitors whether its own optimisation process is working and adjusts accordingly:

- **Convergence tracking** synthesises composite score trends, prediction accuracy, outcome ratios, and scorecard evolution into an overall signal (improving/degrading/oscillating/stable). When oscillation is detected, the LLM is instructed to hold steady rather than reversing last week's suggestions.
- **Loop health metrics** quantify six operational KPIs per cycle: proposal-to-measurement latency, oscillation severity, transfer success rate, recalibration effectiveness, suggestions per cycle, and measurement coverage. These are injected into every prompt so the agent can see where the learning pipeline itself is underperforming.
- **Temporal decay on scorecards** applies 5%/week exponential decay to outcome weights, matching the learning ledger's existing decay rate. Categories recover from early failures as old negatives fade, making the learning loop adaptive rather than accumulative.
- **Directional bias correction** detects systematic optimism or pessimism per metric in the LLM's prediction track record and reduces confidence on predictions that match the bias pattern (capped at 20%).
- **Per-detector confidence calibration** gives each of the strategy engine's 19 detectors an empirical confidence multiplier derived from its outcome history. This is distinct from threshold learning (which adapts *when* a detector fires) — confidence calibration adapts *how much to trust* a detection once it fires.
- **Suggestion pre-validation** filters strategy engine suggestions against the scorecard at recording time, preventing category leakage when the scorecard was unavailable during report generation.
- **Structural proposal validation** runs six parity gates before any structural suggestion reaches a report: hypothesis track record (block if effectiveness ≤ 0 or retired), category track record (block if win rate < 0.3 with n ≥ 5), simplicity criterion (block marginal suggestions), acceptance criteria presence (require at least one well-formed metric criterion), low-confidence block (< 0.4), and empirical calibration adjustment.
- **Instrumentation readiness scoring** evaluates each bot across eight capability categories (basic analysis, process quality, exit analysis, regime analysis, slippage analysis, factor attribution, signal health, drawdown analysis) and injects per-bot readiness reports into prompts. The agent sees which bots have sufficient data coverage for which analysis types, preventing confident conclusions from incomplete instrumentation.
- **Experiment auto-conclusion** tracks parameter experiments to statistical resolution, then escalates through the full chain: conclude experiment → accept/reject linked suggestion → record hypothesis outcome (positive or negative). This closes the loop from "the system proposed a change" through "the change was tested" to "the hypothesis that motivated it was updated."
- **Discovery and convergence context** in prompt instructions tells the LLM how to use automated pattern discoveries and convergence status that were previously loaded into data but invisible to the instruction set.

## Design Philosophy: OpenClaw Governance + Hermes Memory + Autoresearch Optimisation + Symphony Orchestration

This system combines four reference architectures to serve a single goal: continuously improve trading bot performance while keeping a human in control of what matters.

### From OpenClaw: disposable agents, permanent knowledge

The core architectural choice from OpenClaw is that the agent is disposable and the knowledge is permanent. The CLI runtime is a stateless executor — it reads context, runs a skill, writes results, and exits. Everything that compounds value over time — outcome measurements, suggestion histories, forecast calibration, correction logs, prompt patterns — lives in local files and SQLite on your machine: owned, queryable, backed up, inspectable. The agent is the lens; the data is the asset.

This gives you three things concretely. The agent brain is swappable: when a better model arrives or a new provider launches, you swap the executor and everything continues — the skills, the memory, the learning history, the bot configurations. The system already supports four runtime providers (Claude Max, Codex Pro, ZAI, OpenRouter) with automatic fallback chains. Every interaction is auditable: because the agent reads explicit context (memory files, skill prompts, event payloads) and writes explicit outputs (run folder artifacts, JSONL records, parsed analyses), you can trace exactly what the agent saw when it made a particular recommendation. And the scheduler gives you proactive intelligence without infrastructure: APScheduler fires cron jobs for daily analysis, weekly summaries, monthly validation, heartbeat monitoring, and proactive scanning — just scheduled subprocesses on your local machine, not a server or cloud function.

This disposability depends on clear, enforceable boundaries between what the system can change and what only a human can change.

**Three-tier permission gates** (`memory/policies/v1/permission_gates.md`) classify every action the system can take. Routine data processing runs unattended. Strategy and parameter changes require explicit Telegram approval. Critical paths — the ground truth evaluation function, deployment scripts, kill switches, and the policy documents themselves — require double approval. The system enforces these at the file-path level during PR review.

**Separated memory tiers** keep human intent stable. `memory/policies/` contains the identity document (`soul.md`), trading rules, and permission definitions — versioned, human-edited only, never modified autonomously. `memory/findings/` is where the system writes its observations: outcome measurements, correction logs, prompt patterns, and failure modes. These are additive, time-stamped, and subject to 90-day decay. The system learns by accumulating findings; it cannot rewrite its own values.

**Deterministic routing** keeps the critical path predictable. The orchestrator brain classifies events and creates tasks without any LLM involvement. The LLM is invoked only for analysis and reasoning, never for routing or scheduling decisions.

### From Hermes: advisory memory and harness meta-learning

The trading assistant already had a rich learning substrate: outcome measurements, scorecards, convergence tracking, calibration, and retrospective synthesis. Hermes adds the advisory memory and harness layer that decides what the next agent run should see, and whether prompt, retrieval, validator, parser, playbook, or provider changes actually improve decision quality.

**Prompt delivery closes the last-mile gap.** The context builder assembled corrections, skill instructions, and outcome histories, but these were sidecar files in the run directory that the CLI runtime might never read. `InvocationBuilder.build_full_prompt()` now merges learning context directly into the prompt text: instructions, corrections, skill methodology, ranked learning cards, and focused recall. The agent sees accumulated experience in every run, so upstream learning can influence analysis without becoming trading authority.

**Learning cards turn past lessons into ranked retrieval.** Evidence types such as corrections, measured outcomes, discoveries, causal outcome reasonings, confidence recalibrations, spurious outcome flags, hypothesis results, validator blocks, transfer outcomes, retrospectives, and validation logs are ingested into `LearningCard` objects scored by recency, impact, confidence, and context match. `ranked_for_prompt()` injects the most relevant cards first, scoped by bot for single-bot workflows, so a daily analysis for a specific bot sees the corrections and outcomes that matter to that bot rather than an unranked global dump.

**Session history and run recall flow through assemblers.** The session store is passed through assembler paths into `base_package()`, and `RunIndex`/focused recall can surface relevant prior artifacts. This prevents redundant re-analysis and lets the agent build on prior reasoning rather than starting from a blank slate each run.

**Bounded search briefs are the bridge into monthly optimization.** Weekly synthesis, learning cards, hypotheses, and prior outcomes may be summarized into `monthly_search_brief.json` so the LLM planner and monthly runners inspect the most plausible mechanisms first. The brief can adjust experiment emphasis, phase order, seed neighborhoods, negative priors, and repair-ablation ordering after repair is triggered; it cannot choose the monthly sequence, trigger OOS repair, satisfy gates, approve changes, weaken policy memory, or command bots.

**Generated memory remains advisory.** Policy memory and monthly evidence outrank learning cards, generated playbooks, raw recall, and harness benchmark results. Those artifacts improve retrieval, summaries, validation, model review, and provider routing, but they require provenance, outcome attribution, supersession, quarantine/archive semantics, and keep/discard evidence before they are promoted into future prompts.

### From Autoresearch: an immutable objective function and a separable evidence loop

The core insight from Autoresearch is that a self-improving system needs an evaluation function it cannot modify, and an optimization loop whose scoring, gates, repair triggers, and adoption decisions do not depend on model persuasion. The LLM can design experiments and review candidates after evidence is assembled; deterministic replay, objective scoring, parity gates, and approval policy remain decisive.

**The ground truth computer** (`skills/ground_truth_computer.py`) is the system's equivalent of Autoresearch's `evaluate_bpb()` for daily/weekly learning snapshots. It computes a legacy `objective_weights_v1` helper composite from daily trading data using z-score normalized metrics with fixed weights centralized in `schemas/objective_weights.py`: expected return (30%), Calmar ratio (20%), profit factor (15%), expectancy (15%), inverse drawdown (10%), and process quality (10%). The prompt policy uses broader human-utility bands so model suggestions stay aligned with the immutable profile taxonomy without changing historical helper scoring. Replay-backed monthly/phased-auto candidate ranking is governed by strategy-family immutable score profiles recorded in artifacts as `effective_objective_version=immutable_score_profiles_v1` and `objective_profile_id=<profile>`. When replay cannot simulate process quality, helper weights are renormalized for local triage; trade frequency remains a viability and under-trading gate, not a standalone target that can override expectancy, Calmar, or drawdown. Both helper objective weights and immutable score profiles are human-owned objective contracts behind approval governance.

**The monthly evidence loop** (`skills/monthly_validation_orchestrator.py`) is the material strategy-learning path that replaces legacy WFO. In `trading_assistant`, it owns frozen manifests, telemetry and market-data coverage checks, replay-parity evidence, monthly search-brief attachment, model review, candidate gates, approval packets, ledgers, and monthly outcome measurement. The full target loop delegates full-fidelity replay to `trading_assistant_backtest`: run `round_N` diagnostics, use the bounded search brief as advisory priors, run two-fold phased-auto on the in-sample window, trigger granular OOS repair only when the latest completed month underperforms, run repair-centered confirmatory follow-up, and adopt `round_N+1` only after parity and approval gates.

**The weekly learning cycle** (`skills/learning_cycle.py`) remains a sensor, early-warning, and context provider. It computes ground-truth snapshots, records learning signals, and feeds trend context into subsequent prompts. Its influence on monthly optimization is intentionally bounded to `monthly_search_brief.json`: it may steer planner emphasis, phase order, seed neighborhoods, negative priors, and conditional repair-ablation priority, but it cannot create approval-ready candidates, change the monthly sequence, trigger OOS repair, or satisfy adoption gates.

### From Symphony: isolated candidate orchestration

Symphony contributes the runner boundary for expensive monthly optimisation work, not the trading brain. In this system, the Symphony-style layer owns deterministic per-candidate workspaces, path containment and enforced subprocess `cwd`, repo-owned optimizer workflow contracts, candidate-attempt state, retry/backoff, stall detection, drift reconciliation, and structured attempt logs.

It explicitly does not own strategy logic, replay scoring, live/backtest parity gates, approval or deployment decisions, ticket workflow, or live trading commands. `trading_assistant` remains the control plane and ledger; `trading_assistant_backtest` is the experiment lab; the actual trading repo remains the production source of truth.

### How they integrate into a single learning system

The four patterns are not just additive: they share a single objective function and a unified suggestion lifecycle, so every proposal flows through approval, deployment monitoring, and outcome measurement regardless of whether it originated in report reasoning or monthly candidate validation.

**Shared objective function.** Daily ground truth, monthly replay evidence, and approval-gated candidate scoring draw from the same objective vocabulary (`schemas/objective_weights.py`). Replay layers renormalize only when process quality is unavailable, and trade frequency remains a viability gate. When weights are updated behind double approval, evaluation, candidate scoring, and monthly outcomes move together.

**Unified suggestion lifecycle.** Suggestions and monthly candidates are tracked with deterministic IDs through `SuggestionTracker`, `ProposalLedger`, and `StrategyChangeLedger`, then flow through approval, implementation, deployment monitoring, rollback/watch decisions, and outcome measurement. Early-warning measurements remain prompt context; the next completed one-month full-fidelity validation is the primary deployed verdict for material strategy changes.

**Experiment-to-hypothesis traceability.** When a parameter or structural experiment is approved, `ExperimentManager` tracks it to statistical conclusion. On resolution, the auto-conclusion chain updates the linked suggestion (accept or reject) and records the outcome against the hypothesis that motivated it (positive or negative). This means the LLM's next analysis prompt reflects not just "this change was tested" but "the hypothesis behind it was strengthened or weakened" — connecting empirical results back to the analytical reasoning that produced them.

**Bidirectional context flow.** The weekly learning cycle computes ground truth deltas and records the period's ledger entry, while monthly validation owns material strategy evidence and approvals. Results flow back through `analysis/context_builder.py`, monthly outcome priors, learning cards, focused recall, and `monthly_search_brief.json`. The LLM sees what was already tried, what past suggestions achieved, which candidate families have poor track records, whether the system is converging or oscillating, where its directional biases lie, which bots have sufficient instrumentation, and which weak weekly signals should stay low-confidence.

**Governance prevents gaming.** OpenClaw's permission gates ensure neither the weekly/harness layer nor the monthly optimizer can drift its own objective. Because `soul.md` and the ground truth formula sit behind human-only policy controls, the system cannot shift to something easier to optimize. Structural and config candidates must pass data coverage, replay parity, objective, no-regression, model-review, approval-payload, and human approval gates before adoption. The system learns from its own history through an evaluation function it cannot change, governed by permissions it cannot override.

### Matching the change type to the right tool

The learning system routes each type of improvement to whichever tool handles it best.

Parameter-level changes — adjusting a stop-loss percentage, a signal threshold, a filter sensitivity — are numerical optimization problems with a clear objective function. These now belong to monthly evidence and replay validation, where phased-auto can compare config candidates against frozen artifacts, replay parity, objective thresholds, OOS degradation checks, and rollback requirements before approval.

Structural changes — adding a regime-aware exit rule, splitting a strategy into variants, redesigning a filter interaction — require genuine analytical reasoning: reading diagnostics in context, forming hypotheses about *why* something is failing, and proposing changes that a simple config grid would never enumerate. In the target monthly loop, they are first-class phased-auto candidates, not a separate advisory-only lane: the LLM designs candidate patches, the runner applies them in isolated workspaces across the live strategy repo and backtest adapter, unit/decision/parity tests run before fold scoring, and config neighborhoods can then be tuned around passing structural candidates. Weekly reports may still surface structural ideas for human review, but production adoption remains gated by monthly evidence and explicit approval.

Portfolio-level changes — rebalancing family allocation weights, adjusting risk caps, modifying coordination rules between strategies, changing drawdown tier multipliers — operate above individual strategies. The strategy engine runs portfolio-level detectors (correlation crowding, family imbalance, drawdown synchronisation) and produces structured `PortfolioProposal` objects. Allocation proposals are validated through a what-if analysis (`skills/portfolio_what_if.py`) that rescales historical family PnL under the proposed weights to estimate portfolio Calmar, Sharpe, and max drawdown before any change is recorded. When per-family trade-level data is available, the what-if uses individual trades instead of daily aggregates — producing intra-day max drawdown from the per-trade equity curve, Sortino ratio, profit factor, and PnL-by-regime breakdowns that daily aggregation would mask. Portfolio outcomes are measured over a 30-day observation window (vs 7 days for strategy-level), and regime shifts during the window yield INCONCLUSIVE verdicts rather than false signals. Each strategy carries an archetype profile (`data/strategy_profiles.yaml`) that sets archetype-specific detection thresholds — trend-following strategies get looser alpha decay thresholds than intraday momentum, for example — so the engine's suggestions respect each strategy's inherent characteristics rather than applying uniform rules.

## Architecture

```
Bot VPSes → Sidecar → Assistant-local relay ingress → EventQueue (SQLite)
                                                       ↓
                                              OrchestratorBrain
                                            (classify → create task)
                                                       ↓
                                                    Worker
                                              (pick task → run agent)
                                                       ↓
                                                  AgentRunner
                                          (assemble context → provider profile → CLI runtime)
                                                       ↓
                                                runs/<id>/outputs/
                                                       ↓
                                          Telegram / Discord / Email
```

### Package Layout

```
orchestrator/   — FastAPI app, brain, worker, scheduler, event queue, handlers
analysis/       — prompt assemblers, strategy engine, context builders, response validation
skills/         — data pipelines, metrics builders, simulation runners, trackers
comms/          — Telegram, Discord, Email adapters + dispatcher + renderers
schemas/        — Pydantic v2 models for all data contracts
memory/         — policies/ (human-edited) + findings/ (system-written)
tests/          — pytest, asyncio_mode=auto, ~3370 tests
```

## Quick Start

**Prerequisites:** Python 3.12+

```bash
# Clone and install
git clone <repo-url>
cd trading-assistant
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your credentials (bot tokens, relay URL, etc.)

# Run tests
pytest tests/

# Start the orchestrator
export ALLOW_UNAUTHENTICATED_LOCAL=true  # local loopback dev only
uvicorn trading_assistant.orchestrator.app:app --reload
```

## Configuration

Copy `.env.example` to `.env` and configure:

- Local runs auto-load `.env`; exported process environment variables still take precedence.
- On Windows, one-time background startup is: copy `.env`, run `powershell -ExecutionPolicy Bypass -File scripts\install-startup.ps1`, then log in. The logon task keeps the supervisor running in the background and startup catch-up replays missed cron work after downtime.

| Variable | Required | Description |
|----------|----------|-------------|
| `ENVIRONMENT` | Production | Set to `production` for unattended deployments |
| `BOT_IDS` | Production | Comma-separated bot identifiers |
| `RELAY_URL` | Production unless direct-ingest only | Assistant-local relay ingress endpoint URL |
| `RELAY_API_KEY` | Production when `RELAY_URL` is set | API key for orchestrator-to-relay polling |
| `DIRECT_INGEST_ONLY` | Production when no relay is used | Explicitly declares that `/ingest` is the only ingest path |
| `ORCHESTRATOR_API_KEY` | Production/public bind | Required `X-Api-Key` for control-plane endpoints |
| `CLAUDE_COMMAND` | No | Claude Code CLI command or absolute path |
| `CLAUDE_COMMAND_ARGS` | No | JSON array of launcher args prepended before Claude runtime args |
| `CODEX_COMMAND` | No | Codex CLI command or absolute path |
| `CODEX_COMMAND_ARGS` | No | JSON array of launcher args prepended before Codex runtime args |
| `ZAI_API_KEY` | No | Enables the Z.AI Coding Plan profile through Claude Code-compatible env overrides |
| `OPENROUTER_API_KEY` | No | Enables the OpenRouter profile through Claude Code-compatible env overrides |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for notifications and `/settings` |
| `TELEGRAM_CHAT_ID` | No | Telegram chat for reports |
| `DISCORD_BOT_TOKEN` | No | Discord bot token |
| `DISCORD_CHANNEL_ID` | No | Discord channel for reports |
| `SMTP_HOST/PORT/USER/PASS` | No | Email delivery configuration |
| `DATA_DIR` | No | Base data directory (default: `data`) |

Run `python -m trading_assistant.orchestrator.preflight` before starting a
deployment. It prints the exact orchestrator `.env` path, validates auth,
production bot/ingest settings, relay auth, data-dir writability, and scheduler
dependencies without starting the ASGI app.

Before shipping a deployment copy, run the bounded gate from the workspace root:
`python tools/run_workspace_checks.py deployment-gate`. It runs named tiers with
per-command timeouts and skips local-only slow checks unless
`TA_RUN_SLOW_LOCAL_CHECKS=1` is set. To inspect the file set that would be copied
without running tests, use
`python tools/run_workspace_checks.py deployment-package --json`; generated
runtime data and the large data package tree are excluded by default. Use
`--include-data` only for an intentional data artifact shipment.

Concrete deployment copy options:

1. Sparse checkout from the monorepo:

   ```powershell
   git clone --filter=blob:none --no-checkout <repo-url> trading-assistant-deploy
   cd trading-assistant-deploy
   git sparse-checkout init --cone
   git sparse-checkout set `
     pyproject.toml `
     tools `
     packages/trading_assistant `
     packages/trading_assistant_data/src `
     packages/trading_assistant_data/pyproject.toml `
     packages/trading_assistant_backtest/src `
     packages/trading_assistant_backtest/backtests `
     packages/trading_assistant_backtest/pyproject.toml
   git checkout <release-ref>
   python tools/run_workspace_checks.py deployment-package --json
   ```

2. Package install from a prepared source copy:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e .\packages\trading_assistant
   python -m pip install -e .\packages\trading_assistant_data
   python -m pip install -e .\packages\trading_assistant_backtest
   python -m trading_assistant.orchestrator.preflight
   ```

The deploy copy should contain source code, migrations, docs, package metadata,
and static config examples only. Keep generated runtime directories such as
`data/`, `runs/`, `artifacts/`, `.pytest_cache/`, `.ruff_cache/`, and large
canonical market-data trees out of the service copy unless the deployment is
explicitly a data artifact shipment.

See `orchestrator/config.py` for all settings.

### Agent Provider Switching

Agent selection is persisted in `data/agent_preferences.json`.

- Global default applies everywhere unless a workflow override exists.
- When no persisted preferences or env seed exists, the global default is `codex_pro`.
- Workflow overrides are available for `daily_analysis`, `weekly_analysis`, `monthly_validation`, `monthly_model_review`, and `triage`.
- Initial values can be seeded from `AGENT_PROVIDER` and the per-workflow `*_AGENT_PROVIDER` env vars, but only when no persisted preferences file exists yet.
- `GET /agent/preferences` returns the persisted default, overrides, effective workflow selections, and provider readiness.
- `PUT /agent/preferences` updates the global default or workflow overrides. A `null` model means "use the provider default model."
- Telegram exposes the same provider switching via `/settings` and the control-panel `Settings` button. Telegram changes provider only; model overrides remain HTTP-only.

Built-in provider defaults:

- `claude_max` -> `sonnet`
- `codex_pro` -> `gpt-5.4`
- `zai_coding_plan` -> `glm-5`
- `openrouter` -> `minimax/minimax-m2.5`

Claude Max setup:

- Install Claude Code locally and log in with `claude auth login`.
- Verify the CLI reports `loggedIn=true`, `authMethod=claude.ai`, and `subscriptionType=max` via `claude auth status`.
- `GET /agent/preferences` marks `claude_max` unavailable until that Max-auth check passes.

Launcher args:

- `CLAUDE_COMMAND_ARGS` and `CODEX_COMMAND_ARGS` accept JSON arrays and are inserted before the runtime-specific CLI args.
- This supports wrapper scripts and Windows launcher patterns such as `CODEX_COMMAND=wsl.exe` with `CODEX_COMMAND_ARGS=["--","codex"]`.

Provider notes:

- `claude_max` runs through the local Claude Code CLI with `stream-json` output, fresh per-run sessions, and live runtime events on the existing SSE bus. The runner clears Anthropic API/base-url/model override env vars before Claude-backed launches so Max runs cannot silently fall back to API-credit billing.
- `codex_pro` runs through the local Codex CLI in read-only sandbox mode with streamed JSONL parsing, raw run artifacts, and the same runtime SSE events as Claude-backed providers.
- `codex_pro` requires a launchable Codex command plus local ChatGPT auth in `~/.codex/auth.json`. API-key auth is rejected for this subscription-backed profile, and OpenAI API/base-url env vars are cleared before launch so runs cannot drift into API billing.
- On Windows, `CODEX_COMMAND` may need a wrapper or launcher abstraction rather than a directly executable WindowsApp binary. Use `CODEX_COMMAND_ARGS` for that instead of editing orchestrator code.
- The app surfaces Codex diagnostics when it detects high-latency local settings such as `model_reasoning_effort = "xhigh"` or recent shared-state lock/slow-write symptoms.
- `zai_coding_plan` intentionally stays Claude Code-backed so tool use still works and usage can count against the Z.AI coding plan. It only requires the Claude CLI itself plus `ZAI_API_KEY`; it does not depend on Claude Max auth.
- `openrouter` uses the Claude-compatible Anthropic surface for parity with the existing tool-using flow. It only requires the Claude CLI itself plus `OPENROUTER_API_KEY`; it does not depend on Claude Max auth.
- Daily and weekly Claude workflows use read-only tool allowlists (`Read`, `Grep`, `Glob`). Monthly model review runs without tools. Triage keeps `Bash` enabled alongside those read-only tools.

## Scheduled Tasks

| Task | Schedule | What It Does |
|------|----------|--------------|
| Daily Analysis | 06:00 UTC (configurable per bot timezone) | Per-bot daily report with quality gate |
| Weekly Summary | Sunday | Cross-bot review, strategy suggestions, structural proposals |
| Monthly Validation | Monthly | Authoritative evidence package, candidate validation, and approval-gated proposals |
| Proactive Scanner | Morning + evening | Anomaly detection, heartbeat monitoring |
| Learning Cycle | Weekly | Ground truth snapshots, suggestion routing, ledger deltas |
| Outcome Measurement | Sunday 10:00 UTC | Measures implemented suggestions against pre/post performance |
| Memory Consolidation | Sunday 09:00 UTC | Aggregates findings, generates hypothesis candidates |
| Transfer Outcome Tracking | Sunday 10:30 UTC | Measures cross-bot pattern transfer results |
| Threshold Learning | Periodic | Adapts detector firing thresholds from outcome history |
| Experiment Check | Periodic | Auto-concludes experiments that reach statistical significance |
| Discovery Analysis | Periodic | Raw JSONL pattern discovery outside detector coverage |
| Reliability Verification | Periodic | Verifies system health and data pipeline integrity |

Bug triage runs on-demand when HIGH+ severity errors arrive. The configured agent runtime is invoked per-task (not always-running), which keeps subscription-backed profiles at zero extra cost while idle.

## Human-in-the-Loop

The system is read-only with respect to bots — it never sends commands to them. All changes flow through GitHub PRs that require human action to merge:

- **Parameter changes**: backtested and safety-checked automatically, but require explicit Telegram approval before a PR is even created
- **Structural candidates**: may be implemented and scored inside the monthly phased-auto/backtest lane, but production PRs require evidence gates and human approval; weekly structural suggestions remain report-only until promoted
- **Bug fix PRs**: generated for obvious fixes, always require human review
- **Rollback PRs**: created automatically on regression detection, still require human merge
- **Portfolio allocation**: recommendations only — reallocating capital is a manual action

Three-tier permission gates (auto / requires_approval / requires_double_approval) enforce this. Trading logic changes always require approval.

## Commands

```bash
# Start the orchestrator (default: loopback only)
export ALLOW_UNAUTHENTICATED_LOCAL=true
export BIND_HOST=127.0.0.1
python -m trading_assistant.orchestrator.preflight
uvicorn trading_assistant.orchestrator.app:app --host 127.0.0.1 --port 8000

# Manual task triggers
python -m orchestrator.scheduler --run-now daily_report
python -m orchestrator.scheduler --run-now weekly_summary

# Build curated data for a date
python skills/build_daily_metrics.py --date 2026-03-01

# Build portfolio risk card
python skills/compute_portfolio_risk.py --date 2026-03-01

# Run tests
pytest tests/
pytest tests/ -k "integration"
pytest tests/test_handlers.py -v
```

### Exposing the orchestrator beyond localhost

The control plane exposes mutating endpoints (`/ingest`, `/feedback`,
`/agent/preferences`, dead-letter reprocess, cancel subagent, etc.). Binding
to a non-loopback host without authentication would let anyone reachable
inject events or change preferences.

The app **refuses to start** when bound to a non-loopback host with no API
key configured. Two env vars are involved:

- `ORCHESTRATOR_API_KEY` — required header `X-Api-Key` for all endpoints
  except `/health`, `/live`, and `/ready`. Pick a long random value.
- `ALLOW_UNAUTHENTICATED_LOCAL` — explicit local-dev escape hatch. Leave this
  `false` outside loopback-only development.
- `BIND_HOST` — declare the host you intend to pass to `uvicorn --host`.
  Without an API key, startup is allowed only when this was explicitly set to a
  loopback host and `ALLOW_UNAUTHENTICATED_LOCAL=true`.
- `ENVIRONMENT=production` — enables production validation for bot IDs, ingest
  mode, and relay API-key configuration.

```bash
export ORCHESTRATOR_API_KEY="$(openssl rand -hex 32)"
export ENVIRONMENT="production"
export BIND_HOST="0.0.0.0"
export BOT_IDS="bot1,bot2"
export RELAY_URL="http://<assistant-private-ip>:8001"
export RELAY_API_KEY="..."
python -m trading_assistant.orchestrator.preflight
uvicorn trading_assistant.orchestrator.app:app --host 0.0.0.0 --port 8000
```

Prefer running behind a reverse proxy that terminates TLS and adds the
`X-Api-Key` header from a secret store; do not put the raw orchestrator on
the public internet even with an API key set.

Use `/live` for process liveness and `/ready` for supervisor/readiness checks.
`/ready` returns HTTP 503 whenever scheduler or configured communication loops
are degraded. `/health` keeps the same JSON body for compatibility but always
returns HTTP 200.

## Tech Stack

- **Python 3.12**, FastAPI, uvicorn, APScheduler
- **SQLite** for event queue + task tracking
- **Claude Code CLI** and **Codex CLI**, with Claude-compatible provider overlays for Z.AI and OpenRouter
- **Pydantic v2** for all data contracts
- **Telegram Bot API**, discord.py, SMTP for notifications
- **HMAC-SHA256** for bot-to-assistant-relay write authentication; `RELAY_API_KEY` for orchestrator relay polling
