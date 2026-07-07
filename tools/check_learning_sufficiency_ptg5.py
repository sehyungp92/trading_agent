from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from learning_sufficiency_gate_utils import checklist_completion_check


ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    ROOT / "packages" / "trading_assistant" / "src",
    ROOT / "packages" / "trading_contracts" / "src",
    ROOT / "packages" / "trading_instrumentation" / "src",
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_assistant.analysis.monthly_model_response_parser import parse_monthly_model_review  # noqa: E402
from trading_assistant.analysis.monthly_model_response_validator import MonthlyModelResponseValidator  # noqa: E402
from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler  # noqa: E402
from trading_assistant.schemas.learning_sufficiency import (  # noqa: E402
    CoverageCheck,
    CoverageStatus,
    LearningCapabilityAuthority,
    LearningCapabilityStatus,
    LearningEligibility,
    LearningGap,
    LearningSufficiencyManifest,
)
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate  # noqa: E402
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus  # noqa: E402
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline  # noqa: E402


DEFAULT_INDEX = ROOT / "artifacts" / "learning_sufficiency" / "phase2_manifests" / "manifest_index.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "ptg5_gate_report.json"
DEFAULT_PTG3_REPORT = ROOT / "artifacts" / "learning_sufficiency" / "ptg3_gate_report.json"
DEFAULT_PTG4_REPORT = ROOT / "artifacts" / "learning_sufficiency" / "ptg4_gate_report.json"
REQUIRED_GATE_NAMES = [
    "learning_sufficiency_manifest_present",
    "learning_capability_authority",
    "causal_join_completeness",
    "denominator_coverage",
    "after_cost_outcome_coverage",
    "proposal_trace_coverage",
    "counterfactual_backfill_coverage",
    "runtime_evidence_coverage",
    "instrumentation_gap_impact",
]
EVENT_COVERAGE_KEYS = {
    "trade_outcome_lineage",
    "missed_opportunity_lineage",
    "filter_decision_coverage",
    "orderbook_context_coverage",
    "portfolio_rule_coverage",
}
RUNTIME_SUPPORT_CLASSES = {
    "trade",
    "missed_opportunity",
    "filter_decision",
    "orderbook_context",
    "portfolio_rule",
    "order",
    "fill",
    "pipeline_funnel",
    "deployment_metadata",
}
RUNTIME_SUPPORT_STATES = {"unsupported", "supported_but_unobserved", "observed"}
EVENT_VALUE_CLASSIFICATIONS = {"learning_authority", "learning_gap_diagnostic", "operational_health"}
GAP_PRIORITY = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PTG-5 learning gate and prompt discipline.")
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ptg3-report", default=str(DEFAULT_PTG3_REPORT))
    parser.add_argument("--ptg4-report", default=str(DEFAULT_PTG4_REPORT))
    args = parser.parse_args(argv)

    checks = [
        checklist_completion_check(["Phase 7", "Phase 8"]),
        _check_prior_gate_reports((Path(args.ptg3_report), Path(args.ptg4_report))),
        _check_manifest_runtime_evidence_and_gaps(Path(args.index)),
        _check_monthly_gate_names_and_fail_closed(),
        _check_weekly_prompt_contract(),
        _check_model_review_authority_validation(),
    ]
    failures = [check for check in checks if not check["passed"]]
    report = {
        "schema_version": "learning_sufficiency_ptg5_gate_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate": "PTG-5",
        "required_acceptance_rows": ["AM-14", "AM-15", "AM-16", "AM-17", "AM-18", "AM-19", "AM-25"],
        "required_finite_checklist_sections": ["Phase 7", "Phase 8"],
        "status": "pass" if not failures else "blocked",
        "promotion_criteria": (
            "Monthly candidate gates consume sufficiency manifests, runtime evidence "
            "coverage and ranked gap records are emitted, and prompt/output validators "
            "enforce evidence authority labels."
        ),
        "checks": checks,
        "failures": failures,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": report["status"] == "pass",
        "gate": report["gate"],
        "status": report["status"],
        "artifact_path": _rel(output_path),
    }, indent=2))
    return 0 if report["status"] == "pass" else 1


def _check_prior_gate_reports(paths: tuple[Path, ...]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for path in paths:
        row = {"path": _rel(path), "exists": path.exists(), "status": ""}
        if path.exists():
            row["status"] = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "")
        details.append(row)
    passed = all(row["exists"] and row["status"] == "pass" for row in details)
    return _check("prior_phase_gate_reports_pass", passed, details)


def _check_manifest_runtime_evidence_and_gaps(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return _check("runtime_evidence_coverage_and_gap_records", False, {
            "index_path": _rel(index_path),
            "error": "manifest index is missing",
        })
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [row for row in index.get("manifests", []) if isinstance(row, dict)]
    failures: list[str] = []
    manifest_details: list[dict[str, Any]] = []
    for row in rows:
        manifest_path = ROOT / str(row.get("manifest_path") or "")
        if not manifest_path.exists():
            failures.append(f"{_rel(manifest_path)} is missing")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        event_coverage = manifest.get("required_event_coverage", {})
        missing_event_keys = sorted(EVENT_COVERAGE_KEYS - set(event_coverage))
        if missing_event_keys:
            failures.append(f"{_rel(manifest_path)} missing event coverage {missing_event_keys}")
        runtime_support = manifest.get("runtime_evidence_support", {})
        missing_support = sorted(RUNTIME_SUPPORT_CLASSES - set(runtime_support))
        if missing_support:
            failures.append(f"{_rel(manifest_path)} missing runtime support states {missing_support}")
        for event_class, support in runtime_support.items():
            if not isinstance(support, dict):
                failures.append(f"{_rel(manifest_path)} runtime support for {event_class} is malformed")
                continue
            state = str(support.get("support_state") or "")
            source_paths = [str(path) for path in support.get("support_source_paths", [])]
            observed_paths = [str(path) for path in support.get("observed_evidence_paths", [])]
            observed_count = int(support.get("observed_event_count") or 0)
            configured_events = [str(item) for item in support.get("configured_event_types", [])]
            value_classes = support.get("event_value_classifications", {})
            if state not in RUNTIME_SUPPORT_STATES:
                failures.append(f"{_rel(manifest_path)} runtime support for {event_class} has invalid state {state!r}")
            if not isinstance(value_classes, dict):
                failures.append(f"{_rel(manifest_path)} runtime support for {event_class} has malformed event-value classifications")
                value_classes = {}
            invalid_classes = sorted({
                str(value)
                for value in value_classes.values()
                if str(value) not in EVENT_VALUE_CLASSIFICATIONS
            })
            if invalid_classes:
                failures.append(f"{_rel(manifest_path)} runtime support for {event_class} has invalid value classes {invalid_classes}")
            missing_value_classes = sorted(event for event in configured_events if event not in value_classes)
            if missing_value_classes:
                failures.append(f"{_rel(manifest_path)} runtime support for {event_class} lacks value classes {missing_value_classes}")
            if state == "observed" and (not source_paths or not observed_paths or observed_count <= 0):
                failures.append(f"{_rel(manifest_path)} observed runtime support for {event_class} lacks source or evidence paths")
            if state in {"unsupported", "supported_but_unobserved"} and not source_paths:
                failures.append(f"{_rel(manifest_path)} {state} runtime support for {event_class} lacks declaring source")
        gaps = [gap for gap in manifest.get("known_gaps", []) if isinstance(gap, dict)]
        blocked = [str(item) for item in manifest.get("blocked_learning_capabilities", [])]
        if blocked and not gaps:
            failures.append(f"{_rel(manifest_path)} blocked capabilities have no gap records")
        if gaps:
            priorities = [
                GAP_PRIORITY.get(str(gap.get("expected_learning_value") or "").lower(), 99)
                for gap in gaps
            ]
            if priorities != sorted(priorities):
                failures.append(f"{_rel(manifest_path)} gap records are not ranked by learning value")
            for gap in gaps:
                if not gap.get("blocked_learning_capability") or not gap.get("remediation"):
                    failures.append(f"{_rel(manifest_path)} has incomplete gap record")
                    break
        manifest_details.append({
            "manifest_path": _rel(manifest_path),
            "event_coverage_keys": sorted(event_coverage),
            "runtime_support_states": {
                key: value.get("support_state", "")
                for key, value in sorted(runtime_support.items())
                if isinstance(value, dict)
            },
            "blocked_learning_capabilities": blocked,
            "known_gap_count": len(gaps),
        })
    return _check("runtime_evidence_coverage_and_gap_records", bool(rows) and not failures, {
        "index_path": _rel(index_path),
        "manifest_count": len(rows),
        "manifests": manifest_details,
        "failures": failures,
    })


def _check_monthly_gate_names_and_fail_closed() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        manifest_path = root / "learning_sufficiency_manifest.json"
        manifest = _insufficient_execution_manifest(manifest_path)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        gates = MonthlyCandidatePipeline()._learning_sufficiency_gates(
            MonthlyImprovementCandidate(candidate_id="candidate-1", change_kind="execution_change"),
            MonthlyValidationResult(
                run_id="monthly-bot1-strat1-2026-05",
                run_month="2026-05",
                bot_id="bot1",
                strategy_id="strat1",
                status=MonthlyValidationStatus.EXPERIMENT,
                learning_sufficiency_manifest_path=str(manifest_path),
            ),
        )
    by_name = {gate.name: gate for gate in gates}
    passed = (
        list(by_name) == REQUIRED_GATE_NAMES
        and by_name["learning_sufficiency_manifest_present"].passed
        and not by_name["learning_capability_authority"].passed
        and not by_name["causal_join_completeness"].passed
        and not by_name["runtime_evidence_coverage"].passed
        and not by_name["instrumentation_gap_impact"].passed
        and by_name["after_cost_outcome_coverage"].passed
    )
    return _check("monthly_sufficiency_gates_fail_closed", passed, {
        "gate_names": [gate.name for gate in gates],
        "gate_results": [gate.model_dump(mode="json") for gate in gates],
    })


def _check_weekly_prompt_contract() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        curated = root / "curated"
        memory = root / "memory"
        runs = root / "runs"
        (memory / "policies" / "v1").mkdir(parents=True)
        manifest_path = runs / "monthly-bot1-strat1-2026-05" / "learning_sufficiency_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "run_month": "2026-05",
            "eligibility": "diagnostics_only",
            "supported_learning_capabilities": ["execution_learning"],
            "blocked_learning_capabilities": ["filter_threshold_learning"],
            "known_gaps": [{
                "blocked_learning_capability": "filter_threshold_learning",
                "expected_learning_value": "critical",
                "event_type": "pipeline_funnel",
                "missing_field": "denominator_coverage",
                "remediation": "Emit funnel snapshots for bot1/strat1.",
            }],
        }), encoding="utf-8")
        package = WeeklyPromptAssembler(
            week_start="2026-05-04",
            week_end="2026-05-10",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        ).assemble()
    sufficiency = package.data.get("learning_sufficiency", {})
    instructions = package.instructions
    passed = (
        bool(sufficiency.get("supported_capabilities_by_scope"))
        and bool(sufficiency.get("blocked_capabilities_by_scope"))
        and bool(sufficiency.get("top_learning_gaps"))
        and "LEARNING SUFFICIENCY AUTHORITY" in instructions
        and "supported_learning_capabilities" in instructions
        and "blocked_learning_capabilities" in instructions
        and "diagnostics-only evidence" in instructions
    )
    return _check("weekly_prompt_sufficiency_contract", passed, {
        "learning_sufficiency": sufficiency,
        "instruction_contract_present": passed,
    })


def _check_model_review_authority_validation() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        evidence = Path(temp) / "candidate_gate_report.json"
        evidence.write_text("{}", encoding="utf-8")
        authoritative = _model_review_payload(evidence, "learning_authoritative")
        diagnostics = _model_review_payload(evidence, "diagnostics_only")
        validator = MonthlyModelResponseValidator()
        authoritative_result = validator.validate(
            parse_monthly_model_review(authoritative),
            allowed_evidence_paths=[str(evidence)],
        )
        diagnostics_result = validator.validate(
            parse_monthly_model_review(diagnostics),
            allowed_evidence_paths=[str(evidence)],
        )
    diagnostics_message = "diagnostics-only evidence cannot be presented as approval-grade"
    passed = (
        authoritative_result.valid
        and not diagnostics_result.valid
        and any(issue.message == diagnostics_message for issue in diagnostics_result.issues)
    )
    return _check("model_review_authority_label_validation", passed, {
        "authoritative_valid": authoritative_result.valid,
        "diagnostics_valid": diagnostics_result.valid,
        "diagnostics_issues": [issue.model_dump(mode="json") for issue in diagnostics_result.issues],
    })


def _insufficient_execution_manifest(path: Path) -> LearningSufficiencyManifest:
    missing_join = CoverageCheck(
        check_id="decision_to_order_join",
        status=CoverageStatus.MISSING,
        observed_count=0,
        required_count=1,
        required_fields=["decision_id", "order_id"],
        missing_fields=["decision_id", "order_id"],
    )
    return LearningSufficiencyManifest(
        bot_id="bot1",
        strategy_id="strat1",
        run_month="2026-05",
        window_start=date(2026, 5, 1),
        window_end=date(2026, 5, 31),
        eligibility=LearningEligibility.INSUFFICIENT_JOINS,
        required_event_coverage={
            "trade_outcome_lineage": CoverageCheck(check_id="trade_outcome_lineage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
            "missed_opportunity_lineage": CoverageCheck(check_id="missed_opportunity_lineage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
            "filter_decision_coverage": CoverageCheck(check_id="filter_decision_coverage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
            "orderbook_context_coverage": CoverageCheck(check_id="orderbook_context_coverage", status=CoverageStatus.MISSING, observed_count=0, required_count=1),
            "portfolio_rule_coverage": CoverageCheck(check_id="portfolio_rule_coverage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
        },
        join_coverage={
            "decision_to_trade_join": CoverageCheck(check_id="decision_to_trade_join", status=CoverageStatus.PASS, observed_count=1, required_count=1),
            "decision_to_order_join": missing_join,
            "order_to_fill_join": CoverageCheck(check_id="order_to_fill_join", status=CoverageStatus.MISSING, observed_count=0, required_count=1),
            "risk_portfolio_join": CoverageCheck(check_id="risk_portfolio_join", status=CoverageStatus.PASS, observed_count=1, required_count=1),
        },
        denominator_coverage={
            "denominator_coverage": CoverageCheck(check_id="denominator_coverage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
        },
        after_cost_coverage=CoverageCheck(check_id="after_cost_coverage", status=CoverageStatus.PASS, observed_count=1, required_count=1),
        capability_status={
            "execution_learning": LearningCapabilityStatus(
                capability_id="execution_learning",
                status=LearningCapabilityAuthority.BLOCKED,
                required_checks=[
                    "decision_to_order_join",
                    "order_to_fill_join",
                    "orderbook_context_coverage",
                    "after_cost_coverage",
                ],
                satisfied_checks=["after_cost_coverage"],
                blocking_checks=[
                    "decision_to_order_join",
                    "order_to_fill_join",
                    "orderbook_context_coverage",
                ],
                blocking_reasons=[
                    "decision_to_order_join:missing",
                    "order_to_fill_join:missing",
                    "orderbook_context_coverage:missing",
                ],
            ),
        },
        known_gaps=[
            LearningGap(
                bot_id="bot1",
                strategy_id="strat1",
                event_type="order",
                missing_field="decision_id,order_id",
                blocked_learning_capability="execution_learning",
                expected_learning_value="high",
                frequency=1,
                remediation="Emit canonical decision/order join fields.",
            ),
        ],
        artifact_paths={"learning_sufficiency_manifest": str(path)},
    )


def _model_review_payload(evidence: Path, evidence_authority: str) -> str:
    return f"""
<!-- MONTHLY_MODEL_REVIEW
{{
  "run_id": "monthly-bot1-strat1-2026-05",
  "bot_id": "bot1",
  "strategy_id": "strat1",
  "candidate_reviews": [
    {{
      "candidate_id": "candidate-1",
      "recommendation": "approval packet is coherent",
      "routing": "experiment",
      "risk_classification": "medium",
      "evidence_paths": [{json.dumps(str(evidence))}],
      "capability_labels": ["execution_learning"],
      "evidence_authority": "{evidence_authority}",
      "expected_objective_impact": {{"latest_month_oos": 0.1}},
      "replay_or_experiment_plan": "Run shadow for the next completed month.",
      "acceptance_criteria": ["positive latest OOS"],
      "rollback_plan": "restore incumbent"
    }}
  ]
}}
-->
"""


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "details": details}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
