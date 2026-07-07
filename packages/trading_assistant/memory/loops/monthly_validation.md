---
loop_id: monthly_validation
status: active
job_key: monthly_validation
schedule:
  trigger: cron
  cadence: monthly
  day: 2
  hour: 3
  minute: 0
  coalesce: true
  catchup_limit: 1
authority:
  may_create_approval_request: true
  may_modify_policy_memory: false
  may_modify_live_bot_state: false
  may_write_generated_memory: true
  negative_authority:
    - no_live_bot_mutation
    - no_autonomous_policy_memory_write
reads:
  - artifact_type: monthly_run_manifest
    authority: binding
  - artifact_type: artifact_index
    authority: binding
  - artifact_type: monthly_search_brief
    authority: advisory
writes:
  - artifact_type: monthly_validation_result
    authority: approval_gate
  - artifact_type: candidate_gate_report
    authority: approval_gate
  - artifact_type: monthly_evidence_verification
    authority: approval_gate
  - artifact_type: approval_packet
    authority: approval_gate
  - artifact_type: strategy_change_ledger
    authority: binding
verification:
  - requirement_id: data_bundle_authoritative
    description: Market data must be authoritative for approval-facing output.
  - requirement_id: candidate_gates_evaluated
    description: Deterministic candidate gates must run.
  - requirement_id: independent_verifier_required_when_approval_ready
    description: Read-only verifier must pass before approval routing.
    required_for_approval: true
stopping_criteria:
  - selected_candidate_or_deterministic_no_adoption
  - blocking_reasons_recorded
  - verifier_artifact_written_for_approval_facing_packet
---

## Purpose
Run monthly full-fidelity replay-backed validation. This loop is authoritative for material strategy and config changes.

## Current focus
Keep monthly validation manifest-driven, replay-backed, approval-gated, and read-only with respect to live bots.

## Authority boundary
May create approval requests only after deterministic gates and independent verifier pass. It may not deploy or mutate live bots.

## Inputs
Run manifest, artifact index, data and telemetry manifests, replay parity, selected candidates, model-review validation, loop contract, and advisory search brief.

## Outputs
Monthly validation result, candidate gate report, verifier artifact, approval packet, proposal records, and strategy-change records.

## Required checks
Data coverage, telemetry lineage, replay parity, model-review validation, artifact authority, deployment metadata blockers, and verifier verdict.

## Failure modes
Diagnostics-only data, failed parity, invalid model review, shadow/local deployment metadata, missing evidence, failed verifier.

## Escalation path
Suppress approval, record blockers, and create repair context without changing live bot state.

## Backlog
Compare verifier shadow outputs against historical monthly fixtures before any broader gate rollout.

## Timeline
Monthly runs are projected into the loop-run ledger and generated work log.
