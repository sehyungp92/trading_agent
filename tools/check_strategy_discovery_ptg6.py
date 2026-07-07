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
):
    if source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from trading_assistant.analysis.discovery_prompt_assembler import DiscoveryPromptAssembler  # noqa: E402
from trading_assistant.schemas.artifact_authority import ArtifactAuthority  # noqa: E402
from trading_assistant.schemas.monthly_candidates import MonthlyImprovementCandidate  # noqa: E402
from trading_assistant.schemas.monthly_validation import MonthlyValidationResult, MonthlyValidationStatus  # noqa: E402
from trading_assistant.skills.artifact_authority_registry import ArtifactAuthorityRegistry  # noqa: E402
from trading_assistant.skills.monthly_candidate_pipeline import MonthlyCandidatePipeline  # noqa: E402
from trading_assistant.skills.strategy_discovery_packet_builder import StrategyDiscoveryPacketBuilder  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "learning_sufficiency" / "ptg6_gate_report.json"
PTG5_REPORT = ROOT / "artifacts" / "learning_sufficiency" / "ptg5_gate_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify PTG-6 strategy discovery packet gate.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ptg5-report", default=str(PTG5_REPORT))
    args = parser.parse_args(argv)

    checks = [
        checklist_completion_check(["Phase 9"]),
        _check_ptg5_report(Path(args.ptg5_report)),
        _check_artifact_registry(),
        _check_packet_builder_contract(),
        _check_new_strategy_gate(),
        _check_external_weak_packet_rejected(),
        _check_discovery_prompt_contract(),
    ]
    failures = [check for check in checks if not check["passed"]]
    report = {
        "schema_version": "strategy_discovery_ptg6_gate_report_v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "gate": "PTG-6",
        "required_acceptance_rows": ["AM-21", "AM-22", "AM-25"],
        "required_finite_checklist_sections": ["Phase 9"],
        "status": "pass" if not failures else "blocked",
        "promotion_criteria": (
            "Strategy discovery packets are diagnostics-only, and new-strategy "
            "proposals must cite recurring clusters, control slices, after-cost "
            "estimates, and a replay or shadow plan."
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


def _check_ptg5_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _check("ptg5_report_passed", False, {"path": _rel(path), "error": "missing"})
    report = json.loads(path.read_text(encoding="utf-8"))
    return _check("ptg5_report_passed", report.get("status") == "pass", {
        "path": _rel(path),
        "status": report.get("status", ""),
    })


def _check_artifact_registry() -> dict[str, Any]:
    registry = ArtifactAuthorityRegistry.load()
    entry = registry.get("/tmp/strategy_discovery_packet.json")
    passed = (
        entry is not None
        and entry.authority == ArtifactAuthority.DIAGNOSTICS_ONLY
        and entry.may_satisfy_approval_gate is False
        and registry.may_satisfy_approval_gate("/tmp/strategy_discovery_packet.json") is False
    )
    return _check("strategy_discovery_artifact_registry_diagnostics_only", passed, {
        "entry": entry.model_dump(mode="json") if entry is not None else None,
    })


def _check_packet_builder_contract() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        curated = root / "curated"
        _write_fixture_window(curated)
        builder = StrategyDiscoveryPacketBuilder(curated, min_cluster_count=2)
        packet = builder.build(
            run_id="monthly-bot1-strat1-2026-05",
            run_month="2026-05",
            bot_id="bot1",
            strategy_id="strat1",
            window_start=date(2026, 5, 1),
            window_end=date(2026, 5, 31),
        )
        packet_path = builder.write(packet, root / "artifacts")
        payload = json.loads(packet_path.read_text(encoding="utf-8"))
    passed = (
        payload.get("authority") == "diagnostics_only"
        and payload.get("approval_gate_eligible") is False
        and bool(payload.get("missed_opportunity_clusters"))
        and bool(payload.get("denominator_clusters"))
        and bool(payload.get("control_slices"))
        and bool(payload.get("after_cost_estimates"))
        and bool(payload.get("replay_or_shadow_plan"))
    )
    return _check("strategy_discovery_packet_builder_contract", passed, {
        "packet": payload,
    })


def _check_new_strategy_gate() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        curated = root / "curated"
        _write_fixture_window(curated)
        builder = StrategyDiscoveryPacketBuilder(curated, min_cluster_count=2)
        packet = builder.build(
            run_id="monthly-bot1-strat1-2026-05",
            run_month="2026-05",
            bot_id="bot1",
            strategy_id="strat1",
            window_start=date(2026, 5, 1),
            window_end=date(2026, 5, 31),
        )
        packet_path = builder.write(packet, root / "artifacts")
        cluster_id = packet.missed_opportunity_clusters[0].cluster_id
        monthly_result = MonthlyValidationResult(
            run_id="monthly-bot1-strat1-2026-05",
            run_month="2026-05",
            bot_id="bot1",
            strategy_id="strat1",
            status=MonthlyValidationStatus.EXPERIMENT,
            strategy_discovery_packet_path=str(packet_path),
        )
        passing = MonthlyCandidatePipeline._new_strategy_discovery_gate(
            MonthlyImprovementCandidate.from_raw({
                "candidate_id": "new-1",
                "change_kind": "new_strategy",
                "strategy_discovery_cluster_ids": [cluster_id],
                "replay_or_experiment_plan": "Replay and shadow before promotion.",
            }),
            monthly_result,
        )
        missing = MonthlyCandidatePipeline._new_strategy_discovery_gate(
            MonthlyImprovementCandidate.from_raw({
                "candidate_id": "new-2",
                "change_kind": "new_strategy",
                "replay_or_experiment_plan": "Replay and shadow before promotion.",
            }),
            monthly_result,
        )
    passed = passing.passed and not missing.passed and "strategy_discovery_cluster_ids" in missing.reason
    return _check("new_strategy_discovery_gate", passed, {
        "passing": passing.model_dump(mode="json"),
        "missing_citation": missing.model_dump(mode="json"),
    })


def _check_external_weak_packet_rejected() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        packet_path = root / "strategy_discovery_packet.json"
        cluster_id = "weak-cluster"
        packet_path.write_text(json.dumps({
            "run_id": "monthly-bot1-strat1-2026-05",
            "run_month": "2026-05",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "missed_opportunity_clusters": [{
                "cluster_id": cluster_id,
                "source": "missed_opportunity",
                "bot_id": "bot1",
                "strategy_id": "strat1",
                "support_count": 1,
                "control_count": 1,
                "estimated_after_cost_pnl": 10.0,
                "control_slice": {"control_count": 1},
            }],
            "control_slices": [{"control_count": 1}],
            "after_cost_estimates": [{"cluster_id": cluster_id, "estimated_after_cost_pnl": 10.0}],
            "replay_or_shadow_plan": "Replay and shadow before promotion.",
        }), encoding="utf-8")
        gate = MonthlyCandidatePipeline._new_strategy_discovery_gate(
            MonthlyImprovementCandidate.from_raw({
                "candidate_id": "new-weak",
                "change_kind": "new_strategy",
                "strategy_discovery_cluster_ids": [cluster_id],
                "replay_or_experiment_plan": "Replay and shadow before promotion.",
            }),
            MonthlyValidationResult(
                run_id="monthly-bot1-strat1-2026-05",
                run_month="2026-05",
                bot_id="bot1",
                strategy_id="strat1",
                status=MonthlyValidationStatus.EXPERIMENT,
                strategy_discovery_packet_path=str(packet_path),
            ),
        )
    passed = not gate.passed and "no material after-cost clusters" in gate.reason
    return _check("external_weak_strategy_discovery_packet_rejected", passed, {
        "gate": gate.model_dump(mode="json"),
    })


def _check_discovery_prompt_contract() -> dict[str, Any]:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        memory = root / "memory"
        (memory / "policies" / "v1").mkdir(parents=True)
        curated = root / "curated"
        runs = root / "runs"
        packet_dir = runs / "monthly-bot1-strat1-2026-05"
        packet_dir.mkdir(parents=True)
        (packet_dir / "strategy_discovery_packet.json").write_text(json.dumps({
            "run_id": "monthly-bot1-strat1-2026-05",
            "run_month": "2026-05",
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "authority": "diagnostics_only",
            "evidence_authority": "diagnostics_only",
            "approval_gate_eligible": False,
            "missed_opportunity_clusters": [{"cluster_id": "cluster-1"}],
            "control_slices": [{"control_count": 2}],
            "after_cost_estimates": [{"estimated_after_cost_pnl": 10.0}],
            "replay_or_shadow_plan": "Replay then shadow.",
        }), encoding="utf-8")
        package = DiscoveryPromptAssembler(
            date="2026-05-31",
            bots=["bot1"],
            curated_dir=curated,
            memory_dir=memory,
            runs_dir=runs,
        ).assemble()
    passed = (
        bool(package.data.get("strategy_discovery_packets"))
        and "recurring opportunity clusters" in package.instructions
        and "approval-grade evidence" in package.instructions
    )
    return _check("discovery_prompt_packet_contract", passed, {
        "packet_count": len(package.data.get("strategy_discovery_packets", [])),
        "context_files": package.context_files,
    })


def _write_fixture_window(curated: Path) -> None:
    for day, suffix in [("2026-05-01", "a"), ("2026-05-02", "b")]:
        bot_dir = curated / day / "bot1"
        _write_jsonl(bot_dir / "missed.jsonl", [{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "opportunity_id": f"miss-{suffix}",
            "pair": "AAPL",
            "signal": "opening_breakout",
            "market_regime": "trend_up",
            "would_have_pnl": 12.0,
            "blocked_by": "no_strategy_available",
        }])
        _write_jsonl(bot_dir / "trades.jsonl", [{
            "bot_id": "bot1",
            "strategy_id": "strat1",
            "trade_id": f"trade-{suffix}",
            "pair": "AAPL",
            "market_regime": "trend_up",
            "net_pnl": 2.5,
        }])
        bot_dir.mkdir(parents=True, exist_ok=True)
        (bot_dir / "funnel_analysis.json").write_text(json.dumps({
            "stage_totals": {
                "setups_detected": 10,
                "confirmations": 4,
                "entries_attempted": 3,
                "fills": 2,
                "trades_closed": 2,
            },
        }), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _check(name: str, passed: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "passed": passed, "details": details}


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
