---
loop_id: evening_report
status: active
job_key: evening_report
schedule:
  trigger: cron
  cadence: daily
  hour: 22
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
  - artifact_type: daily_metrics
    authority: diagnostics_only
writes:
  - artifact_type: evening_report
    authority: generated
verification:
  - requirement_id: report_or_skip
    description: Evening report or deterministic skip is recorded.
stopping_criteria:
  - report_sent_or_skip_recorded
---

## Purpose
Summarize end-of-day operational context.

## Current focus
Keep reporting separate from approval and deployment decisions.

## Authority boundary
May notify and write generated reports. May not mutate live bots or policy memory.

## Inputs
Daily metrics, bot scope, and notification preferences.

## Outputs
Evening report or skip record.

## Required checks
Scope and report generation checks.

## Failure modes
Missing metrics or notification failure.

## Escalation path
Record blocker and defer material decisions.

## Backlog
Add output artifact links to loop-run projection.

## Timeline
Projected daily after scheduled completion.
