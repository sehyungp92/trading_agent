---
loop_id: transfer_outcome_measurement
status: active
job_key: transfer_outcome_measurement
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 10
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
  - artifact_type: transfer_hypotheses
    authority: advisory
writes:
  - artifact_type: transfer_track_record
    authority: generated
verification:
  - requirement_id: advisory_transfer_only
    description: Transfer outcomes are advisory and cannot satisfy monthly gates.
stopping_criteria:
  - transfer_outcomes_recorded_or_noop
---

## Purpose
Measure whether ideas transfer across strategies or bot families.

## Current focus
Keep transfer evidence advisory until replay-backed monthly validation confirms material changes.

## Authority boundary
May write generated transfer records. May not approve, deploy, or write policy memory.

## Inputs
Transfer hypotheses, outcomes, and recent reports.

## Outputs
Transfer track record entries.

## Required checks
Do not use transfer evidence as approval-gate proof.

## Failure modes
Sparse samples or mismatched strategy families.

## Escalation path
Feed advisory priors to monthly validation.

## Backlog
Link transfer outcomes to monthly candidate families.

## Timeline
Projected weekly after completion.
