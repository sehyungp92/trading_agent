from __future__ import annotations

from pathlib import Path

import pytest

from trading_assistant.schemas.benchmark_case import BenchmarkCase, BenchmarkSeverity, BenchmarkSource
from trading_assistant.skills.harness_case_materializer import (
    HarnessCaseMaterializationError,
    HarnessCaseMaterializer,
)


def test_materializer_refuses_cases_without_provenance(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    case = BenchmarkCase(
        case_id="case-1",
        source=BenchmarkSource.VALIDATION_BLOCK,
        source_id="validation:1",
        severity=BenchmarkSeverity.HIGH,
    )

    with pytest.raises(HarnessCaseMaterializationError):
        HarnessCaseMaterializer(findings).materialize_case(case)


def test_materializer_writes_execution_input_with_artifact_refs(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    evidence = findings / "source.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    case = BenchmarkCase(
        case_id="case-2",
        source=BenchmarkSource.NEGATIVE_OUTCOME,
        source_id="outcome:2",
        severity=BenchmarkSeverity.CRITICAL,
        agent_type="weekly_analysis",
        bot_id="bot1",
        artifact_refs=["memory/findings/source.json"],
        input_snapshot={"strategy_id": "strat1"},
    )

    case_dir = HarnessCaseMaterializer(findings).materialize_case(case)

    assert (case_dir / "execution_input.json").exists()
    assert "memory/findings/source.json" in (case_dir / "execution_input.json").read_text(encoding="utf-8")


def test_materializer_rejects_existing_absolute_path_outside_workspace(tmp_path: Path) -> None:
    findings = tmp_path / "memory" / "findings"
    outside = tmp_path.parent / f"outside-materializer-{tmp_path.name}.json"
    outside.write_text("{}", encoding="utf-8")
    case = BenchmarkCase(
        case_id="case-outside",
        source=BenchmarkSource.NEGATIVE_OUTCOME,
        source_id="outcome:outside",
        severity=BenchmarkSeverity.HIGH,
        artifact_refs=[str(outside)],
        input_snapshot={"strategy_id": "strat1"},
    )

    try:
        with pytest.raises(HarnessCaseMaterializationError):
            HarnessCaseMaterializer(findings).materialize_case(case)
    finally:
        outside.unlink(missing_ok=True)
