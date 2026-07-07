---
loop_id: lineage_audit
status: active
job_key: lineage_audit
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
  - artifact_type: telemetry_manifest
    authority: binding
writes:
  - artifact_type: lineage_audit_report
    authority: diagnostics_only
verification:
  - requirement_id: lineage_ratio_checked
    description: Telemetry lineage ratio is checked for monthly readiness.
stopping_criteria:
  - lineage_report_written_or_blocker_recorded
---

## Purpose
Audit telemetry lineage before monthly validation relies on it.

## Current focus
Protect monthly evidence from insufficient lineage.

## Authority boundary
May write diagnostics. May not mutate live bots or policy memory.

## Inputs
Curated telemetry, proposal ledger, and completed month window.

## Outputs
Lineage audit report and blockers.

## Required checks
Required lineage ratio and manifest consistency.

## Failure modes
Missing telemetry, insufficient lineage, duplicate events.

## Escalation path
Mark monthly validation diagnostics-only or blocked.

## Backlog
Register lineage report schema in artifact authority registry.

## Timeline
Daily projection while monthly validation is enabled.
