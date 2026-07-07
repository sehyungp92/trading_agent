---
loop_id: morning_scan
status: active
job_key: morning_scan
schedule:
  trigger: cron
  cadence: daily
  hour: 7
  minute: 0
  coalesce: true
  catchup_limit: 1
authority:
  may_create_approval_request: false
  may_modify_policy_memory: false
  may_modify_live_bot_state: false
  may_write_generated_memory: true
  negative_authority:
    - no_live_bot_mutation
    - no_autonomous_policy_memory_write
reads:
  - artifact_type: morning_market_context
    authority: diagnostics_only
writes:
  - artifact_type: morning_scan_report
    authority: generated
verification:
  - requirement_id: report_or_skip
    description: Report or deterministic skip is recorded.
stopping_criteria:
  - scan_sent_or_skip_recorded
---

## Purpose
Prepare morning context for bot monitoring.

## Current focus
Keep scans informational and non-mutating.

## Authority boundary
May notify and write generated context. May not mutate live bots or policy memory.

## Inputs
Market context and bot configuration.

## Outputs
Morning scan report or skip record.

## Required checks
Bot scope and schedule consistency.

## Failure modes
Missing market context or notification adapter failure.

## Escalation path
Record blocker in scheduled-run state.

## Backlog
Project bot-specific schedule variants into contract metadata.

## Timeline
Daily coalesced run projected after completion.
