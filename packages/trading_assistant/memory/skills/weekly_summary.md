# Weekly Summary Skill

## Purpose
Produce a comprehensive weekly analysis that goes beyond daily report summaries. Identifies trends, structural patterns, portfolio allocation opportunities, and forward-looking recommendations.

## Trigger
Scheduled weekly on Sunday via APScheduler cron job.

## Pipeline
1. **Weekly Metrics Build** — aggregate 7 daily summaries into weekly portfolio summary.
2. **Strategy Engine** — run 10 deterministic detectors producing Tier 1-4 suggestions.
3. **Simulations** — run FilterSensitivity, Counterfactual, and ExitStrategy simulations for ALL bots unconditionally (not gated by strategy engine findings).
4. **Allocation Analyses** — 6 parallel analyses:
   - Synergy analysis (cross-strategy correlation + intra-bot synergy)
   - Portfolio allocation (risk-parity + Calmar tilt with correlation penalty)
   - Proportion optimization (intra-bot strategy allocation)
   - Structural analysis (lifecycle stage, architecture mismatches)
   - Regime-conditional metrics (per-regime Sharpe/win_rate)
   - Interaction analysis (swing_multi_01 coordinator effects)
5. **Weekly Retrospective** — compare last week's predictions/warnings to actual outcomes.
6. **Context Assembly** — load policies, corrections, failure log, pattern library, allocation history.
7. **Prompt Assembly** — 18-step structured analysis framework.
8. **Claude Invocation** — agent_type="weekly_analysis"
9. **Notification** — dispatch report via configured channels.

## Input Data
- weekly_summary.json (aggregated metrics)
- refinement_report.json (strategy engine output)
- All simulation results (filter sensitivity, counterfactual, exit strategy per bot)
- allocation_analysis (portfolio allocation, synergy, proportion, structural, regime, interaction)
- weekly_retrospective (past predictions vs actual outcomes)
- 7 daily reports + 7 portfolio risk cards

## Output
- weekly_report.md
- Maximum 5 actionable suggestions ranked by expected Calmar impact

## Quality Criteria
- Do NOT repeat what daily reports already said — synthesize higher-level patterns
- Each suggestion must quantify Calmar ratio impact with evidence base
- Cross-reference against rejected suggestions and recent_proposal_outcomes; do not re-suggest without new evidence
- Review allocation analysis and validate whether changes make strategic sense
- Consider second-order effects of allocation changes
- Propose structural improvements (Tier 3-4) with specific test plans
- Review retrospective data and assess prediction accuracy calibration
