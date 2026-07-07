---
loop_id: daily_analysis
status: active
job_key: daily_analysis
schedule:
  trigger: cron
  cadence: daily
  hour: 6
  minute: 0
  coalesce: true
  catchup_limit: 7
authority:
  may_create_approval_request: false
  may_modify_policy_memory: false
  may_modify_live_bot_state: false
  may_write_generated_memory: true
  negative_authority:
    - no_live_bot_mutation
    - no_autonomous_policy_memory_write
reads:
  - artifact_type: curated_daily_metrics
    authority: binding
  - artifact_type: loop_contract
    authority: binding
writes:
  - artifact_type: daily_report
    authority: generated
  - artifact_type: report_checklist
    authority: diagnostics_only
verification:
  - requirement_id: report_checklist_valid
    description: Daily report checklist is generated.
stopping_criteria:
  - report_written_or_deterministic_skip_recorded
  - scheduled_run_completed_or_failed
---

## Purpose
Analyze daily bot performance and produce bounded, evidence-backed observations.

## Current focus
Use scheduler truth: daily analysis runs at 06:00 UTC unless bot-specific market-close grouping overrides the variant.

## Authority boundary
May generate reports and suggestions. May not mutate live bot state, create approvals, or write policy memory.

## Inputs
Curated daily metrics, policies, findings, loop contract context, and recent work-log entries.

## Outputs
Daily report, report checklist, and scheduled-run status.

## Required checks
Minimum data checks, report checklist validation, and stale schedule-doc checks.

## Failure modes
Missing curated files, insufficient trades, provider failure, stale trigger documentation.

## Escalation path
Record blocker in scheduled-run state and surface it in loop-run projection.

## Backlog
Improve per-bot timezone contract projections for bot-specific variants.

## Timeline
Latest runs are projected into `memory/findings/loop_run_ledger.jsonl`.
