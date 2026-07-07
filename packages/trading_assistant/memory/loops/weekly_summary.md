---
loop_id: weekly_summary
status: active
job_key: weekly_summary
schedule:
  trigger: cron
  cadence: weekly
  day_of_week: sun
  hour: 8
  minute: 0
  coalesce: true
  catchup_limit: 4
authority:
  may_create_approval_request: false
  may_modify_policy_memory: false
  may_modify_live_bot_state: false
  may_write_generated_memory: true
  negative_authority:
    - no_live_bot_mutation
    - no_autonomous_policy_memory_write
reads:
  - artifact_type: weekly_metrics
    authority: binding
  - artifact_type: recent_work_log
    authority: generated
writes:
  - artifact_type: weekly_summary_report
    authority: generated
  - artifact_type: monthly_search_brief_inputs
    authority: advisory
verification:
  - requirement_id: search_prior_boundary
    description: Weekly output may steer search order but cannot satisfy monthly gates.
stopping_criteria:
  - weekly_summary_written_or_blocker_recorded
---

## Purpose
Summarize weekly portfolio and strategy evidence for human review and monthly search priors.

## Current focus
Keep weekly findings advisory for monthly candidate ordering.

## Authority boundary
May write generated weekly findings. May not create approvals, mutate live bots, or update policy memory.

## Inputs
Weekly metrics, recent outcomes, loop contracts, and projected activity history.

## Outputs
Weekly synthesis and advisory inputs to monthly search briefs.

## Required checks
Confirm advisory language and avoid claiming monthly gate passage.

## Failure modes
Incomplete weekly metrics, stale context, or repeated unsupported suggestions.

## Escalation path
Record blockers and defer material strategy decisions to monthly validation.

## Backlog
Tighten linkage from weekly focus rotation to monthly search brief attribution.

## Timeline
Latest weekly entries are projected into the generated work log.
