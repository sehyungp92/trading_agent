# Strategy Refinement Skill

## Purpose
Generate testable hypotheses for structural strategy improvements. Goes beyond parameter tuning to propose new filters, signals, regime adaptations, and cross-bot pattern transfers.

## Trigger
Weekly report flags degrading metrics, alpha decay, or structural issues (Tier 3-4 suggestions from strategy engine).

## Pipeline
1. **Strategy Engine Pre-processing** — 10 deterministic detectors run first:
   - detect_alpha_decay, detect_signal_decay, detect_exit_timing_issues
   - detect_correlation_breakdown, detect_time_of_day_patterns
   - detect_drawdown_patterns, detect_position_sizing_issues
   - detect_underperforming_strategies, detect_overperforming_strategies
   - detect_filter_issues
2. **Simulation Analysis** — FilterSensitivity, Counterfactual, ExitStrategy results available
3. **Pattern Library Review** — check cross-bot pattern library for transferable innovations
4. **Context Assembly** — 90-day findings history, quarterly patterns, rejected suggestions
5. **Prompt Assembly** — strategy refinement focus within weekly or dedicated review
6. **Claude Invocation** — agent_type="strategy_refinement"

## Hypothesis Tiers
| Tier | Example | Validation Required |
|------|---------|-------------------|
| PARAMETER | "Widen ATRSS stop from 1.0 to 1.3 ATR" | Monthly full-fidelity validation |
| FILTER | "Relax ADX filter from 25 to 20" | FilterSensitivity + monthly validation |
| STRATEGY_VARIANT | "Add time-of-day gate for Helix" | 30-day backtest |
| HYPOTHESIS | "Alpha decay detected — test new signal" | Full test plan required |

## Input Data
- Strategy engine refinement_report.json
- Simulation results (filter sensitivity, counterfactual, exit strategy)
- Pattern library entries (for cross-bot transfer candidates)
- 90-day rolling metrics per strategy
- Per-regime performance breakdown

## Output
- Ranked list of hypotheses (max 3) with:
  - What to test (specific parameter/filter/signal change)
  - Expected outcome (quantified Calmar/PnL impact range)
  - How to measure success (backtest config, sample size, time period)
  - Rollback plan (reversibility assessment)
  - Null hypothesis (what happens if we do nothing)

## Quality Criteria
- Every hypothesis must be specific and testable — no vague "consider improving"
- Evidence base: minimum 30 trades, 30 days of data
- Cross-reference rejected suggestions — do not re-propose without new evidence
- Include estimated implementation effort
- Rank by expected Calmar impact × feasibility
- Consider cross-bot learning — has another bot solved a similar problem?
