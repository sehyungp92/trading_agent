from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "packages" / "trading_assistant" / "src"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_contract_doc_checks import check_auto_outcome_authority, check_retired_wfo_docs
from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.config import (
    runtime_memory_dir_from_env,
    runtime_scheduler_config_from_env,
)
from trading_assistant.orchestrator.scheduler import SchedulerConfig, build_scheduled_job_specs
from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry
from trading_assistant.skills.loop_contract_store import LoopContractStore, validate_scheduler_contracts
from trading_assistant.skills.loop_run_ledger import LoopRunLedgerStore, RuntimeLoopProjectionWriter


async def _noop(_scheduled_for=None) -> None:
    return None


def _all_specs(config: SchedulerConfig | None = None):
    return build_scheduled_job_specs(
        config=config or runtime_scheduler_config_from_env(),
        worker_fn=_noop,
        monitoring_fn=_noop,
        relay_fn=_noop,
        daily_analysis_fn=_noop,
        weekly_analysis_fn=_noop,
        stale_error_sweep_fn=_noop,
        stale_event_recovery_fn=_noop,
        morning_scan_fn=_noop,
        evening_report_fn=_noop,
        outcome_measurement_fn=_noop,
        memory_consolidation_fn=_noop,
        transfer_outcome_fn=_noop,
        approval_expiry_fn=_noop,
        pr_review_check_fn=_noop,
        deployment_check_fn=_noop,
        threshold_learning_fn=_noop,
        experiment_check_fn=_noop,
        reliability_verification_fn=_noop,
        discovery_fn=_noop,
        learning_cycle_fn=_noop,
        lineage_audit_fn=_noop,
        market_data_sync_fn=_noop,
        monthly_validation_fn=_noop,
    )


def run_checks(*, memory_dir: Path | None = None) -> list[str]:
    memory = memory_dir or ROOT / "packages" / "trading_assistant" / "memory"
    issues = validate_scheduler_contracts(_all_specs(), memory_dir=memory)
    messages = [issue.format() for issue in issues]

    registry = ArtifactAuthorityRegistry.load(memory)
    required_registry = {
        ("monthly_search_brief", False, "AM-07"),
        ("memory_policies", False, "AM-06"),
        ("replay_parity_report", True, "AM-06"),
        ("approval_packet", True, "AM-09"),
        ("monthly_evidence_verification", True, "AM-09"),
        ("runner_observability", False, "AM-06"),
    }
    for artifact_type, expected_gate, am_row in sorted(required_registry):
        actual = registry.may_satisfy_approval_gate(artifact_type)
        if actual != expected_gate:
            messages.append(
                f"{am_row} artifact_registry:{artifact_type} - approval gate eligibility "
                f"is {actual!r}, expected {expected_gate!r}\n"
                "  remediation: edit memory/artifacts/registry.yaml or registry seeding."
            )
    messages.extend(_check_contract_authority_surfaces(memory, registry))
    messages.extend(_check_runtime_projection_smoke())
    messages.extend(_check_delivered_projection_artifacts(memory))
    messages.extend(_check_default_runtime_memory_alignment(memory))
    messages.extend(_check_runtime_memory_writer_paths())
    messages.extend(_check_model_review_advisory_boundary(registry))
    messages.extend(_check_model_review_diagnostics_boundary(registry))
    messages.extend(check_auto_outcome_authority(SRC))
    messages.extend(check_retired_wfo_docs(ROOT, memory))
    return messages


def _issue(am_row: str, path: str, field: str, message: str, remediation: str) -> str:
    location = f"{path}:{field}" if field else path
    return f"{am_row} {location} - {message}\n  remediation: {remediation}"


def _check_contract_authority_surfaces(
    memory_dir: Path,
    registry: ArtifactAuthorityRegistry,
) -> list[str]:
    messages: list[str] = []
    contracts = LoopContractStore(memory_dir).load_all()
    approval_required = {
        "monthly_validation_result",
        "candidate_gate_report",
        "model_review_validation",
        "monthly_evidence_verification",
        "approval_packet",
        "replay_parity_report",
    }
    for contract in contracts.values():
        for artifact in contract.writes:
            path_hint = artifact.path_hint.replace("\\", "/").lower()
            if artifact.artifact_type == "memory_policies" or "memory/policies" in path_hint:
                messages.append(_issue(
                    "AM-01",
                    contract.source_path,
                    f"writes.{artifact.artifact_type}",
                    "loop contract lists memory/policies as an autonomous write target",
                    "Remove memory/policies from loop writes; policy memory is human-owned.",
                ))
        if contract.authority.may_create_approval_request:
            if not contract.approval_verifier_required:
                messages.append(_issue(
                    "AM-09",
                    contract.source_path,
                    "verification",
                    "approval-capable loop does not require an independent verifier",
                    "Add a verification item with required_for_approval: true.",
                ))
            write_types = {artifact.artifact_type for artifact in contract.writes}
            for artifact_type in ("candidate_gate_report", "monthly_evidence_verification", "approval_packet"):
                if artifact_type not in write_types:
                    messages.append(_issue(
                        "AM-09",
                        contract.source_path,
                        "writes",
                        f"approval-capable loop does not declare {artifact_type} output",
                        f"Add {artifact_type} to the loop contract writes.",
                    ))

    for artifact_type in sorted(approval_required):
        entry = registry.get(artifact_type)
        if entry is None:
            messages.append(_issue(
                "AM-06",
                "artifact_registry",
                artifact_type,
                "approval packet artifact type is not registered",
                "Register the artifact type in ArtifactAuthorityRegistry.",
            ))
        elif not entry.may_satisfy_approval_gate:
            messages.append(_issue(
                "AM-06",
                "artifact_registry",
                artifact_type,
                "approval packet artifact type is not eligible for approval gates",
                "Set approval-gate eligibility only for deterministic approval evidence.",
            ))
    return messages


def _check_runtime_projection_smoke() -> list[str]:
    async def _run() -> list[str]:
        with TemporaryDirectory(prefix="loop_projection_check_") as root:
            base = Path(root)
            memory = base / "memory"
            store = ScheduledRunStore(
                str(base / "scheduled_runs.db"),
                final_status_observer=RuntimeLoopProjectionWriter(memory).project_record,
            )
            await store.initialize()
            try:
                scheduled_for = datetime(2026, 6, 2, 3, 0, tzinfo=timezone.utc)
                await store.mark_started("monthly_validation", "bot:check", scheduled_for)
                await store.mark_failed(
                    "monthly_validation",
                    "bot:check",
                    scheduled_for,
                    error="fixture blocker",
                )
                await store.mark_completed(
                    "weekly_summary",
                    "global",
                    scheduled_for + timedelta(hours=1),
                )
                entries = LoopRunLedgerStore(memory / "findings" / "loop_run_ledger.jsonl").read()
                work_log = memory / "work_log.md"
                if not entries or entries[0].status != "failed" or not work_log.exists():
                    return [_issue(
                        "AM-03",
                        "runtime_projection",
                        "ScheduledRunStore.final_status_observer",
                        "final scheduled run did not produce loop ledger and work-log projection",
                        "Wire RuntimeLoopProjectionWriter into ScheduledRunStore final statuses.",
                    )]
                if not any(
                    entry.loop_id == "weekly_summary" and entry.status == "completed"
                    for entry in entries
                ):
                    return [_issue(
                        "AM-03",
                        "runtime_projection",
                        "ScheduledRunStore.final_status_observer",
                        "non-monthly scheduled run did not produce a global lifecycle feed row",
                        "Keep RuntimeLoopProjectionWriter attached to all ScheduledRunStore final statuses.",
                    )]
            finally:
                await store.close()
        return []

    return asyncio.run(_run())


def _check_delivered_projection_artifacts(memory_dir: Path) -> list[str]:
    messages: list[str] = []
    ledger_path = memory_dir / "findings" / "loop_run_ledger.jsonl"
    work_log_path = memory_dir / "work_log.md"
    try:
        entries = LoopRunLedgerStore(ledger_path).read(strict=True)
    except ValueError as exc:
        return [_issue(
            "AM-17",
            str(ledger_path),
            "jsonl",
            f"loop-run ledger contains a malformed row: {exc}",
            "Regenerate memory/findings/loop_run_ledger.jsonl from scheduled-run projections.",
        )]
    work_log = work_log_path.read_text(encoding="utf-8") if work_log_path.exists() else ""

    if not entries:
        messages.append(_issue(
            "AM-03",
            str(ledger_path),
            "entries",
            "delivered loop ledger has no projected runtime or integration entries",
            "Generate a non-empty runtime/integration projection with RuntimeLoopProjectionWriter.",
        ))
    if not work_log_path.exists() or "No projected loop entries yet" in work_log or not work_log.strip():
        messages.append(_issue(
            "AM-04",
            str(work_log_path),
            "entries",
            "delivered work log is missing or still a placeholder",
            "Regenerate memory/work_log.md from a non-empty loop ledger projection.",
        ))
    monthly_entries = [entry for entry in entries if entry.loop_id == "monthly_validation"]
    if entries and not monthly_entries:
        messages.append(_issue(
            "AM-03",
            str(ledger_path),
            "loop_id",
            "delivered projection evidence does not include a monthly_validation entry",
            "Record a monthly_validation runtime/integration projection entry.",
        ))
    if monthly_entries and not any(entry.proposal_ids for entry in monthly_entries):
        messages.append(_issue(
            "AM-09",
            str(ledger_path),
            "proposal_ids",
            "monthly runtime projection evidence omits proposal IDs",
            "Propagate approval packet proposal IDs into monthly projection records.",
        ))
    if monthly_entries and not any(entry.task_id for entry in monthly_entries):
        messages.append(_issue(
            "AM-03",
            str(ledger_path),
            "task_id",
            "delivered monthly projection evidence omits task IDs",
            "Propagate action or TaskRegistry task IDs into monthly projection records when acceptance evidence supplies them.",
        ))
    if monthly_entries and not any(entry.cost_usd > 0 for entry in monthly_entries):
        messages.append(_issue(
            "AM-03",
            str(ledger_path),
            "cost_usd",
            "monthly runtime projection evidence does not preserve available model-review cost",
            "Propagate model-review AgentResult.cost_usd into MonthlyValidationResult and loop projection metadata.",
        ))
    if monthly_entries and not any(
        record.kind == "run_index" and record.id and record.path
        for entry in monthly_entries
        for record in entry.source_records
    ):
        messages.append(_issue(
            "AM-03",
            str(ledger_path),
            "source_records",
            "monthly runtime projection evidence does not link run metadata",
            "Include run_id/run_dir metadata when projecting scheduled monthly results.",
        ))
    for entry in monthly_entries:
        for raw_path in [*entry.output_artifacts, *entry.evidence_paths, *entry.approval_packet_paths]:
            path = Path(raw_path)
            if not raw_path or not path.exists() or not _projection_artifact_is_placeholder(path):
                continue
            messages.append(_issue(
                "AM-03",
                str(path),
                "payload",
                "delivered projection artifact is placeholder-only",
                "Replace acceptance projection artifacts with minimal domain payloads that describe their evidence role.",
            ))
    return messages


def _check_default_runtime_memory_alignment(memory_dir: Path) -> list[str]:
    messages: list[str] = []
    checked_memory = memory_dir.resolve()
    runtime_memory = runtime_memory_dir_from_env().resolve()
    if runtime_memory != checked_memory:
        messages.append(_issue(
            "AM-15",
            "runtime_memory",
            "MEMORY_DIR",
            (
                f"default runtime memory {runtime_memory} does not match checked "
                f"package memory {checked_memory}"
            ),
            "Unset the default MEMORY_DIR override or set MEMORY_DIR=memory so runtime context uses checked package memory.",
        ))
        return messages

    ctx = ContextBuilder(runtime_memory)
    if not ctx.load_loop_contract_context(agent_type="monthly_model_review"):
        messages.append(_issue(
            "AM-15",
            str(runtime_memory),
            "loop_contract",
            "default runtime context cannot load the checked monthly loop contract",
            "Ensure runtime ContextBuilder reads the checked memory/loops directory.",
        ))
    if not ctx.load_recent_work_log_entries(
        agent_type="monthly_model_review",
        limit=5,
    ):
        messages.append(_issue(
            "AM-15",
            str(runtime_memory),
            "recent_work_log",
            "default runtime context cannot load checked loop work-log projections",
            "Ensure runtime projections write to the checked memory/findings tree.",
        ))
    if not ctx.load_recent_performance_learning_entries(bot_id="bot1", limit=5):
        messages.append(_issue(
            "AM-15",
            str(runtime_memory),
            "performance_learning",
            "default runtime context cannot load checked performance-learning projections",
            "Ensure runtime performance-learning context reads the checked memory/findings tree.",
        ))
    return messages


def _check_runtime_memory_writer_paths() -> list[str]:
    source_root = ROOT / "packages" / "trading_assistant" / "src" / "trading_assistant" / "orchestrator"
    if not source_root.exists():
        return []
    forbidden = ('db_path / "memory"', "db_path / 'memory'")
    messages: list[str] = []
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in forbidden):
                messages.append(_issue(
                    "AM-15",
                    f"{path}:{line_no}",
                    "memory_root",
                    "runtime memory writer bypasses the resolved memory_dir",
                    "Use resolve_runtime_memory_dir()/memory_dir so checked contracts, logs, outcomes, and learning projections share one root.",
                ))
    return messages


def _projection_artifact_is_placeholder(path: Path) -> bool:
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if isinstance(payload, dict) and set(payload) <= {"fixture", "artifact"}:
            return True
    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip().lower()
        return text in {"", "# loop projection acceptance monthly report"}
    return False


def _check_model_review_advisory_boundary(registry: ArtifactAuthorityRegistry) -> list[str]:
    issues = registry.validate_model_review_evidence(["monthly_search_brief.json"])
    if any("advisory" in issue.message for issue in issues):
        return []
    return [_issue(
        "AM-07",
        "artifact_registry",
        "monthly_search_brief",
        "monthly search brief can support actionable model-review evidence",
        "Keep monthly_search_brief advisory and reject it in validate_model_review_evidence.",
    )]


def _check_model_review_diagnostics_boundary(registry: ArtifactAuthorityRegistry) -> list[str]:
    issues = registry.validate_model_review_evidence(["runner_observability.json"])
    if any("diagnostics_only" in issue.message for issue in issues):
        return []
    return [_issue(
        "AM-08",
        "artifact_registry",
        "runner_observability",
        "diagnostics-only artifacts can support actionable model-review evidence",
        "Reject diagnostics-only evidence in validate_model_review_evidence.",
    )]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check loop contracts and authority registry.")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=ROOT / "packages" / "trading_assistant" / "memory",
    )
    parser.add_argument(
        "--warning-mode",
        action="store_true",
        help="print issues as warnings and exit 0 for pre-CI rollout checks",
    )
    args = parser.parse_args(argv)
    issues = run_checks(memory_dir=args.memory_dir)
    if issues:
        print("loop-contract checks warnings:" if args.warning_mode else "loop-contract checks failed:")
        for issue in issues:
            print(issue)
        return 0 if args.warning_mode else 1
    print(
        "loop-contract checks passed: AM-01 AM-02 AM-03 AM-04 AM-06 "
        "AM-07 AM-08 AM-09 AM-12 AM-13 AM-15 AM-17 guardrails hold"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
