---
loop_id: market_data_sync
status: active
job_key: market_data_sync
schedule:
  trigger: cron
  cadence: monthly
  day: 1
  hour: 1
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
  - artifact_type: market_data_sources
    authority: binding
writes:
  - artifact_type: market_data_manifest
    authority: binding
verification:
  - requirement_id: coverage_manifest_written
    description: Market data coverage manifest is written or blocker recorded.
stopping_criteria:
  - data_bundle_synced_or_blocker_recorded
---

## Purpose
Refresh monthly market-data authority for validation.

## Current focus
Produce manifests that monthly validation can treat as authoritative or diagnostics-only.

## Authority boundary
May write generated market-data manifests. May not modify live bot state or policy memory.

## Inputs
Configured market-data roots, bot registry, and completed month.

## Outputs
Coverage and data bundle manifests.

## Required checks
Coverage ratio and manifest consistency.

## Failure modes
Missing canonical data, incomplete coverage, invalid bundle manifests.

## Escalation path
Record diagnostics-only or blocked status for monthly validation.

## Backlog
Add richer lineage links to loop-run projection.

## Timeline
Projected monthly after scheduled run state is updated.
