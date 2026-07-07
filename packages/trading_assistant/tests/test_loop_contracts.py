from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.config import runtime_scheduler_config_from_env
from trading_assistant.orchestrator.scheduler import (
    ScheduledJobClass,
    SchedulerConfig,
    build_scheduled_job_specs,
)
from trading_assistant.skills.loop_contract_store import (
    LoopContractStore,
    validate_scheduler_contracts,
    validate_skill_trigger_freshness,
)


async def _noop(_scheduled_for=None) -> None:
    return None


def _all_specs():
    return build_scheduled_job_specs(
        config=SchedulerConfig(),
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


def _repo_memory() -> Path:
    return Path(__file__).resolve().parents[1] / "memory"


def _load_loop_contract_check():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "check_loop_contracts.py"
    spec = importlib.util.spec_from_file_location("check_loop_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_stateful_or_coalesced_jobs_have_active_loop_contracts() -> None:
    specs = _all_specs()
    non_interval = [spec for spec in specs if spec.job_class != ScheduledJobClass.INTERVAL]

    store = LoopContractStore(_repo_memory())
    contracts = store.load_all()
    issues = validate_scheduler_contracts(specs, memory_dir=_repo_memory())

    assert not issues, [issue.format() for issue in issues]
    assert {spec.contract_id or spec.job_key for spec in non_interval} <= set(contracts)
    assert contracts["monthly_validation"].approval_verifier_required is True
    assert "authoritative for material strategy" in contracts["monthly_validation"].body_sections["Purpose"]
    assert all(not contract.authority.may_modify_live_bot_state for contract in contracts.values())
    assert all(not contract.authority.may_modify_policy_memory for contract in contracts.values())


def test_daily_schedule_mismatch_fixture_is_caught(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    shutil.copytree(_repo_memory() / "loops", memory / "loops")
    (memory / "skills").mkdir(parents=True)
    (memory / "skills" / "daily_analysis.md").write_text(
        "# Daily Analysis Skill\n\n## Trigger\nScheduled daily at 22:30 UTC via APScheduler cron job.\n",
        encoding="utf-8",
    )
    (memory / "skills" / "skills_index.md").write_text(
        "| Skill | File | Trigger | Description |\n"
        "|---|---|---|---|\n"
        "| `daily_analysis` | [daily_analysis.md](daily_analysis.md) | 22:30 UTC cron | stale |\n",
        encoding="utf-8",
    )

    issues = validate_skill_trigger_freshness(memory)

    assert any(issue.am_row == "AM-02" and "22:30" in issue.message for issue in issues)
    assert any("daily_analysis.md" in issue.path for issue in issues)


def test_runtime_monthly_schedule_env_drift_is_caught(monkeypatch) -> None:
    monkeypatch.setenv("MONTHLY_VALIDATION_HOUR", "4")
    specs = build_scheduled_job_specs(
        config=runtime_scheduler_config_from_env(),
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

    issues = validate_scheduler_contracts(
        specs,
        memory_dir=_repo_memory(),
        require_skill_freshness=False,
    )

    assert any(
        issue.am_row == "AM-02"
        and "monthly_validation" in issue.path
        and issue.field == "schedule.hour"
        and "4" in issue.message
        for issue in issues
    )


def test_loop_contract_check_fails_on_malformed_loop_run_ledger(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    shutil.copytree(_repo_memory(), memory)
    with (memory / "findings" / "loop_run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{bad-json}\n")

    issues = _load_loop_contract_check().run_checks(memory_dir=memory)

    assert any("AM-17" in issue and "malformed row" in issue for issue in issues)


def test_loop_contract_check_flags_unresolved_runtime_memory_writers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_loop_contract_check()
    app_path = (
        tmp_path
        / "packages"
        / "trading_assistant"
        / "src"
        / "trading_assistant"
        / "orchestrator"
        / "app.py"
    )
    app_path.parent.mkdir(parents=True)
    app_path.write_text(
        'approval_tracker = ApprovalTracker(db_path / "memory" / "findings" / "approvals.jsonl")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)

    issues = module._check_runtime_memory_writer_paths()

    assert any("AM-15" in issue and "bypasses the resolved memory_dir" in issue for issue in issues)


def test_context_builder_injects_loop_contract_and_bounded_work_log(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    shutil.copytree(_repo_memory() / "loops", memory / "loops")
    (memory / "policies" / "v1").mkdir(parents=True)
    (memory / "findings").mkdir(parents=True)
    rows = []
    for idx in range(12):
        rows.append({
            "loop_run_id": f"run-{idx}",
            "loop_id": "daily_analysis",
            "job_key": "daily_analysis",
            "status": "completed",
            "scheduled_for": f"2026-06-{idx + 1:02d}T06:00:00+00:00",
            "summary": f"daily summary {idx}",
            "blocking_reasons": [],
            "evidence_paths": [f"artifact-{idx}.json"],
        })
    (memory / "findings" / "loop_run_ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    package = ContextBuilder(memory).base_package(
        agent_type="daily_analysis",
        context_budget_items=2,
        record_retrieval=False,
    )

    assert package.data["loop_contract"]["loop_id"] == "daily_analysis"
    assert package.data["recent_work_log"][0]["loop_run_id"] == "run-11"
    assert len(package.data["recent_work_log"]) == 10
