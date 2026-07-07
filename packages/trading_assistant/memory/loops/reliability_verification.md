---
loop_id: reliability_verification
status: active
job_key: reliability_verification
schedule:
  trigger: cron
  cadence: daily
  hour: 6
  minute: 30
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
  - artifact_type: reliability_inputs
    authority: diagnostics_only
writes:
  - artifact_type: reliability_summary
    authority: diagnostics_only
verification:
  - requirement_id: diagnostics_only
    description: Reliability output is diagnostic and cannot satisfy approval gates.
stopping_criteria:
  - reliability_summary_written_or_blocker_recorded
---

## Purpose
Verify operational reliability signals.

## Current focus
Identify degraded instrumentation without creating strategy approvals.

## Authority boundary
May write diagnostics only. May not mutate live bots or policy memory.

## Inputs
Reliability metrics, event state, and recent failures.

## Outputs
Reliability summary and blockers.

## Required checks
Diagnostics-only authority.

## Failure modes
Missing metrics, stale tasks, telemetry gaps.

## Escalation path
Record diagnostics and route material changes through monthly validation.

## Backlog
Add explicit reliability artifact registration when schemas stabilize.

## Timeline
Daily projection after completion.
