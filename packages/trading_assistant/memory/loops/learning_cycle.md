---
loop_id: learning_cycle
status: active
job_key: learning_cycle
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 11
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
  - artifact_type: monthly_outcomes
    authority: binding
  - artifact_type: learning_cards
    authority: generated
writes:
  - artifact_type: learning_cards
    authority: generated
  - artifact_type: outcome_priors
    authority: advisory
verification:
  - requirement_id: monthly_outcome_authority_preserved
    description: Lightweight outcomes remain context, not final material verdicts.
stopping_criteria:
  - learning_cards_updated_or_no_change_recorded
---

## Purpose
Convert outcome history into future context, priors, and repair focus.

## Current focus
Preserve monthly and follow-up outcomes as the authoritative material verdicts.

## Authority boundary
May update generated learning memory and advisory priors. May not write policy memory or live bot state.

## Inputs
Monthly outcomes, follow-up outcomes, proposal history, and recent work log.

## Outputs
Learning cards, advisory outcome priors, and context summaries.

## Required checks
Do not treat early-warning outcome windows as final material approval evidence.

## Failure modes
Sparse outcome history, contradictory priors, stale strategy IDs.

## Escalation path
Mark priors as advisory and defer material decisions to monthly validation.

## Backlog
Improve repeated-blocker grouping from loop-run ledger.

## Timeline
Weekly learning-cycle runs are projected after scheduled completion.
