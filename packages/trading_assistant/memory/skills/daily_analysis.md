# Daily Analysis Skill

## Purpose
Analyze daily trading bot performance across all bots. Synthesize 12 curated data files per bot into a concise, actionable report.

## Trigger
Scheduled daily at 06:00 UTC via APScheduler cron job.

## Pipeline
1. **Quality Gate** — verify curated data exists for all bots. Degrade gracefully if partial.
2. **Minimum Data Check** — skip Claude invocation if <3 total trades (produce deterministic summary).
3. **Context Assembly** — load policies, corrections, failure log, rejected suggestions, outcome measurements, allocation history, consolidated patterns, session history.
4. **Prompt Assembly** — 9-step structured analysis framework:
   - Portfolio overview (PnL, drawdown, exposure, crowding alerts)
   - Per-bot deep dive (winners, losers, process failures, missed opportunities)
   - Statistical anomalies
   - Factor attribution
   - Exit efficiency (MAE/MFE analysis)
   - Cross-bot patterns
   - Hourly performance
   - Slippage analysis
   - Actionable suggestions (max 3)
5. **Claude Invocation** — agent_type="daily_analysis"
6. **Notification** — dispatch report via configured channels

## Input Data (per bot)
- summary.json, winners.json, losers.json
- process_failures.json, notable_missed.json
- regime_analysis.json, filter_analysis.json, root_cause_summary.json
- factor_attribution.json, exit_efficiency.json
- hourly_performance.json, slippage_stats.json
- Portfolio: risk_card.json

## Output
- daily_report.md
- report_checklist.json (Definition of Done validation)

## Quality Criteria
- Every suggestion must include: expected return impact (range), drawdown impact, evidence base (trade count + period)
- Suggestions without quantification are rejected
- Do not re-suggest items from rejected_suggestions list
- Check recent_proposal_outcomes before proposing changes; do not repeat recent failed or inconclusive proposals without new evidence
- Maximum 3 actionable items
- Portfolio-level picture comes first, then per-bot detail
