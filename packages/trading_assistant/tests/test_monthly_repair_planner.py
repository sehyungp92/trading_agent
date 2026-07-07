from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.backtest_artifacts import BacktestArtifactIndex
from trading_assistant.schemas.monthly_repair_request import MonthlyRepairClassification
from trading_assistant.skills.monthly_repair_planner import MonthlyRepairPlanner


def test_repair_planner_classifies_missing_market_data(tmp_path: Path) -> None:
    request = MonthlyRepairPlanner().build(
        run_id="monthly-bot-strat-2026-04",
        bot_id="bot",
        strategy_id="strat",
        run_month="2026-04",
        blocking_reasons=["market data manifest missing or malformed: x"],
        artifact_root=tmp_path,
        evidence_paths=["run_manifest.json"],
    )

    assert request.classification == MonthlyRepairClassification.DATA
    assert request.retry_eligible is True
    assert request.owner_component == "market_data_sync"


def test_repair_planner_records_missing_and_malformed_artifacts(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "coverage_manifest.json").write_text("{bad", encoding="utf-8")
    index = BacktestArtifactIndex(
        run_id="monthly-bot-strat-2026-04",
        artifact_root=str(artifact_root),
        artifacts={"coverage_manifest.json": str(artifact_root / "coverage_manifest.json")},
    )

    request = MonthlyRepairPlanner().build(
        run_id="monthly-bot-strat-2026-04",
        bot_id="bot",
        strategy_id="strat",
        run_month="2026-04",
        blocking_reasons=["backtest runner failed: missing required artifacts"],
        artifact_root=artifact_root,
        artifact_index=index,
    )

    assert request.classification == MonthlyRepairClassification.ARTIFACT_CONTRACT
    assert "coverage_manifest.json" in request.malformed_artifacts
    assert "incumbent_validation.json" in request.missing_artifact_keys


def test_repair_planner_write_outputs_request(tmp_path: Path) -> None:
    planner = MonthlyRepairPlanner()
    request = planner.build(
        run_id="monthly-bot-strat-2026-04",
        bot_id="bot",
        strategy_id="strat",
        run_month="2026-04",
        blocking_reasons=["replay parity report missing or malformed"],
        artifact_root=tmp_path,
    )

    path = planner.write(request, tmp_path)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["classification"] == "replay_parity"


def test_repair_planner_classifies_missing_selected_candidates(tmp_path: Path) -> None:
    request = MonthlyRepairPlanner().build(
        run_id="monthly-bot-strat-2026-04",
        bot_id="bot",
        strategy_id="strat",
        run_month="2026-04",
        blocking_reasons=["selected_candidates artifact contains no candidates for repair monthly status"],
        artifact_root=tmp_path,
        evidence_paths=["selected_candidates.json"],
    )

    assert request.classification == MonthlyRepairClassification.CANDIDATE_GENERATION
    assert request.owner_component == "monthly_candidate_pipeline"


def test_repair_planner_classifies_diagnostic_only_data_bundle(tmp_path: Path) -> None:
    request = MonthlyRepairPlanner().build(
        run_id="monthly-bot-strat-2026-04",
        bot_id="bot",
        strategy_id="strat",
        run_month="2026-04",
        blocking_reasons=["data bundle is not authoritative: diagnostics_only"],
        artifact_root=tmp_path,
        evidence_paths=["data_bundle_manifest.json"],
    )

    assert request.classification == MonthlyRepairClassification.DATA
    assert request.owner_component == "market_data_sync"
