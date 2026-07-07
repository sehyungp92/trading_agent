---
loop_id: threshold_learning
status: active
job_key: threshold_learning
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 9
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
  - artifact_type: threshold_observations
    authority: diagnostics_only
writes:
  - artifact_type: threshold_profile
    authority: advisory
verification:
  - requirement_id: advisory_threshold_only
    description: Learned thresholds remain advisory until approved through existing gates.
stopping_criteria:
  - threshold_profile_written_or_noop
---

## Purpose
Learn advisory thresholds from accumulated observations.

## Current focus
Keep threshold updates out of live bot state.

## Authority boundary
May write advisory generated profiles. May not deploy or write policy memory.

## Inputs
Outcome observations and threshold history.

## Outputs
Advisory threshold profile.

## Required checks
No live mutation and no approval bypass.

## Failure modes
Sparse observations, overfit thresholds, stale strategy references.

## Escalation path
Feed candidates into monthly validation if material.

## Backlog
Record confidence and sample sizes in projection.

## Timeline
Weekly projection after completion.
