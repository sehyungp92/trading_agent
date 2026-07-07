---
loop_id: outcome_measurement
status: active
job_key: outcome_measurement
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 10
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
  - artifact_type: suggestion_tracker
    authority: binding
writes:
  - artifact_type: outcome_measurements
    authority: diagnostics_only
verification:
  - requirement_id: early_warning_only
    description: Auto outcome measurement is context only for material changes.
stopping_criteria:
  - outcomes_recorded_or_insufficient_data_recorded
---

## Purpose
Measure short-window suggestion outcomes for context and early warning.

## Current focus
Avoid finalizing material strategy outcomes outside monthly replay-backed validation.

## Authority boundary
May write diagnostics and context. May not approve, deploy, write policy memory, or finalize material verdicts.

## Inputs
Suggestion tracker entries and recent performance evidence.

## Outputs
Early-warning outcome measurements.

## Required checks
Mark material strategy decisions as requiring monthly or follow-up authority.

## Failure modes
Sparse windows, noisy evidence, false confidence.

## Escalation path
Escalate material decisions to monthly validation.

## Backlog
Add stronger check coverage for misuse attempts.

## Timeline
Weekly entries are projected into loop-run ledger.
