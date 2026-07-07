"""Decision parity harness seam for structural candidates."""

from __future__ import annotations

from collections.abc import Iterable

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces


def build_structural_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    live_events: Iterable[DecisionTraceEvent],
    adapter_events: Iterable[DecisionTraceEvent],
    evidence_paths: list[str],
    evidence_paths_by_dimension: dict[str, list[str]] | None = None,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        evidence_paths_by_dimension=evidence_paths_by_dimension,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )
