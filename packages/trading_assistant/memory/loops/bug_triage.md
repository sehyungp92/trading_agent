---
loop_id: bug_triage
status: active
job_key: bug_triage
schedule:
  trigger: event
  cadence: on_high_or_critical_error
  coalesce: true
  catchup_limit: 0
authority:
  may_create_approval_request: false
  may_modify_policy_memory: false
  may_modify_live_bot_state: false
  may_write_generated_memory: true
  negative_authority:
    - no_live_bot_mutation
    - no_autonomous_policy_memory_write
reads:
  - artifact_type: error_event
    authority: binding
writes:
  - artifact_type: triage_report
    authority: generated
verification:
  - requirement_id: no_live_remediation
    description: Triage proposes fixes but does not mutate live systems.
stopping_criteria:
  - triage_report_written_or_event_deferred
---

## Purpose
Diagnose high-severity errors and prepare fix context.

## Current focus
Keep triage scoped to evidence, not live mutation.

## Authority boundary
May write generated triage findings. May not deploy, approve, mutate live bots, or write policy memory.

## Inputs
Error event, logs, recent work-log entries, and relevant policies.

## Outputs
Triage report and proposed repair context.

## Required checks
Preserve approval and monthly authority boundaries.

## Failure modes
Insufficient logs, duplicated failures, stale task state.

## Escalation path
Create repair context and require human or monthly approval for material trading changes.

## Backlog
Link repeated failures into loop-run repeated-blocker groups.

## Timeline
Event-driven entries may be projected when task state exists.
