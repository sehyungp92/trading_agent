"""Decision parity report helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from trading_assistant_backtest.contract_models import (
    DECISION_PARITY_DIMENSIONS,
    DecisionParityCheck,
    DecisionParityReport,
    DecisionParityStatus,
    MonthlyRunManifest,
)
from trading_assistant_backtest.replay.decision_trace import (
    DecisionTraceEvent,
    event_signature,
)


@dataclass(frozen=True)
class DimensionParity:
    dimension: str
    status: DecisionParityStatus
    match_rate: float
    mismatch_count: int
    live_count: int
    adapter_count: int
    notes: str = ""
    evidence_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionTraceComparison:
    dimensions: dict[str, DimensionParity]

    @property
    def status(self) -> DecisionParityStatus:
        statuses = {item.status for item in self.dimensions.values()}
        if statuses == {DecisionParityStatus.PASS}:
            return DecisionParityStatus.PASS
        if DecisionParityStatus.FAIL in statuses:
            return DecisionParityStatus.FAIL
        return DecisionParityStatus.INSUFFICIENT_DATA


def insufficient_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    evidence_path: Path,
) -> DecisionParityReport:
    return DecisionParityReport(
        run_id=manifest.run_id,
        candidate_id=candidate_id,
        strategy_plugin_id=manifest.strategy_plugin_id,
        live_repo_commit_sha=manifest.trading_repo_commit_sha,
        backtest_adapter_commit_sha=manifest.backtest_repo_commit_sha,
        status=DecisionParityStatus.INSUFFICIENT_DATA,
        evidence_paths=[str(evidence_path)],
        checks=[
            DecisionParityCheck(
                dimension=dimension,
                status=DecisionParityStatus.INSUFFICIENT_DATA,
                match_rate=0.0,
                mismatch_count=0,
                notes="No structural candidate was adopted.",
                evidence_paths=[],
            )
            for dimension in sorted(DECISION_PARITY_DIMENSIONS)
        ],
    )


def compare_decision_traces(
    live_events: Iterable[DecisionTraceEvent],
    adapter_events: Iterable[DecisionTraceEvent],
    *,
    evidence_paths: list[str] | None = None,
    evidence_paths_by_dimension: dict[str, list[str]] | None = None,
    min_required_match_rate: float = 1.0,
) -> DecisionTraceComparison:
    live_by_dimension = _events_by_dimension(live_events)
    adapter_by_dimension = _events_by_dimension(adapter_events)
    fallback_evidence = evidence_paths or []
    dimension_evidence = evidence_paths_by_dimension or {}
    dimensions: dict[str, DimensionParity] = {}
    for dimension in sorted(DECISION_PARITY_DIMENSIONS):
        live_counter = Counter(live_by_dimension.get(dimension, []))
        adapter_counter = Counter(adapter_by_dimension.get(dimension, []))
        live_count = sum(live_counter.values())
        adapter_count = sum(adapter_counter.values())
        denominator = max(live_count, adapter_count)
        evidence = dimension_evidence.get(dimension, fallback_evidence)
        if denominator == 0:
            dimensions[dimension] = DimensionParity(
                dimension=dimension,
                status=DecisionParityStatus.INSUFFICIENT_DATA,
                match_rate=0.0,
                mismatch_count=0,
                live_count=0,
                adapter_count=0,
                notes="No live or adapter decision events covered this dimension.",
                evidence_paths=evidence,
            )
            continue
        matches = sum((live_counter & adapter_counter).values())
        mismatch_count = denominator - matches
        match_rate = matches / denominator
        status = (
            DecisionParityStatus.PASS
            if mismatch_count == 0 and match_rate >= min_required_match_rate
            else DecisionParityStatus.FAIL
        )
        dimensions[dimension] = DimensionParity(
            dimension=dimension,
            status=status,
            match_rate=match_rate,
            mismatch_count=mismatch_count,
            live_count=live_count,
            adapter_count=adapter_count,
            notes=f"live_events={live_count}; adapter_events={adapter_count}",
            evidence_paths=evidence,
        )
    return DecisionTraceComparison(dimensions=dimensions)


def decision_parity_report_from_traces(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    live_events: Iterable[DecisionTraceEvent],
    adapter_events: Iterable[DecisionTraceEvent],
    evidence_paths: list[str],
    evidence_paths_by_dimension: dict[str, list[str]] | None = None,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
    min_required_match_rate: float = 1.0,
) -> DecisionParityReport:
    comparison = compare_decision_traces(
        live_events,
        adapter_events,
        evidence_paths=evidence_paths,
        evidence_paths_by_dimension=evidence_paths_by_dimension,
        min_required_match_rate=min_required_match_rate,
    )
    live_sha = live_repo_commit_sha or manifest.trading_repo_commit_sha
    adapter_sha = backtest_adapter_commit_sha or manifest.backtest_repo_commit_sha
    lineage_complete = bool(
        manifest.strategy_plugin_id and live_sha and adapter_sha and evidence_paths
    )
    status = comparison.status if lineage_complete else DecisionParityStatus.FAIL
    return DecisionParityReport(
        run_id=manifest.run_id,
        candidate_id=candidate_id,
        strategy_plugin_id=manifest.strategy_plugin_id,
        live_repo_commit_sha=live_sha,
        backtest_adapter_commit_sha=adapter_sha,
        status=status,
        min_required_match_rate=min_required_match_rate,
        evidence_paths=evidence_paths,
        checks=[
            DecisionParityCheck(
                dimension=dimension,
                status=item.status,
                match_rate=item.match_rate,
                mismatch_count=item.mismatch_count,
                notes=item.notes,
                evidence_paths=item.evidence_paths,
            )
            for dimension, item in sorted(comparison.dimensions.items())
        ],
    )


def _events_by_dimension(
    events: Iterable[DecisionTraceEvent],
) -> dict[str, list[tuple[str, str, str, tuple[tuple[str, object], ...]]]]:
    by_dimension: dict[str, list[tuple[str, str, str, tuple[tuple[str, object], ...]]]] = {}
    for event in events:
        signature = event_signature(event)
        by_dimension.setdefault(signature[0], []).append(signature)
    return by_dimension
