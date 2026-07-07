---
loop_id: approval_expiry
status: active
job_key: approval_expiry
schedule:
  trigger: cron
  cadence: daily
  hour: 0
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
  - artifact_type: approval_records
    authority: binding
writes:
  - artifact_type: approval_expiry_notice
    authority: generated
verification:
  - requirement_id: no_deployment_side_effect
    description: Expiry changes approval state only, not live deployment.
stopping_criteria:
  - expired_approvals_processed_or_noop
---

## Purpose
Expire stale approval requests and notify reviewers.

## Current focus
Keep approval lifecycle separate from live deployment.

## Authority boundary
May update approval tracking state through the approval tracker. May not mutate live bots or policy memory.

## Inputs
Approval records and notification preferences.

## Outputs
Expiry notices and approval-state updates.

## Required checks
Do not perform deployment actions.

## Failure modes
Missing tracker, notification failure, malformed approval record.

## Escalation path
Log blocker and keep approval request unchanged if uncertain.

## Backlog
Link expiry events to work-log projection.

## Timeline
Daily coalesced run projection.
