---
loop_id: memory_consolidation
status: active
job_key: memory_consolidation
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 9
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
  - artifact_type: findings_memory
    authority: generated
writes:
  - artifact_type: consolidated_patterns
    authority: generated
verification:
  - requirement_id: policy_memory_not_written
    description: Consolidation cannot write memory/policies.
stopping_criteria:
  - consolidation_written_or_noop_recorded
---

## Purpose
Consolidate generated findings into compact prompt context.

## Current focus
Reduce context load without changing policy memory.

## Authority boundary
May write generated findings. May not write policy memory or live bot state.

## Inputs
Findings, reports, outcomes, and recent work log.

## Outputs
Consolidated patterns and compact summaries.

## Required checks
Generated-memory-only write target.

## Failure modes
Malformed findings or stale strategy references.

## Escalation path
Skip malformed records and log blockers.

## Backlog
Annotate consolidation outputs with source loop IDs.

## Timeline
Weekly projection after completion.
