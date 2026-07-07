---
loop_id: discovery_analysis
status: active
job_key: discovery_analysis
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sat
  hour: 3
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
  - artifact_type: discovery_inputs
    authority: diagnostics_only
writes:
  - artifact_type: strategy_ideas
    authority: advisory
verification:
  - requirement_id: hypothesis_only
    description: Discoveries are hypotheses until validated by replay and approval gates.
stopping_criteria:
  - hypotheses_recorded_or_noop
---

## Purpose
Find new strategy ideas, gaps, and hypotheses.

## Current focus
Keep discovery outputs hypothesis-only.

## Authority boundary
May write generated advisory ideas. May not create approvals, deploy, or write policy memory.

## Inputs
Patterns, outcomes, regime context, and recent work log.

## Outputs
Strategy ideas and discovery notes.

## Required checks
No claim of monthly validation passage.

## Failure modes
Overfit hypotheses, stale context, unsupported evidence.

## Escalation path
Route promising ideas into monthly replay-backed validation.

## Backlog
Tie discovery hypotheses to later monthly outcomes.

## Timeline
Weekly projection after completion.
