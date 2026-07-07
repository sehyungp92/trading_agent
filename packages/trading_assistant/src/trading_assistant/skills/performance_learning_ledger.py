"""Append-only performance-learning projection over trading source ledgers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.learning_sufficiency_audit import (
    manifest_runtime_support_source_authoritative,
)
from trading_assistant.schemas.performance_learning_ledger import (
    AuthorityLevel,
    DecisionStage,
    IntendedLearningEffects,
    PerformanceLearningRecord,
    PerformanceMetricDeltas,
    PerformanceRecordType,
    PerformanceSourceRecord,
    PortfolioInteractionContext,
    SourceCadence,
    StrategySliceContext,
    authority_for_cadence,
)
from trading_assistant.schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalEvaluation,
    ProposalKind,
    ProposalOutcome,
    ProposalRecord,
    ProposalSource,
)
from trading_assistant.schemas.strategy_change_ledger import (
    RollbackStatus,
    StrategyChangeRecord,
    StrategyChangeRecordType,
)
from trading_assistant.skills.performance_learning_relevance import (
    matches_bot_scope as _matches_bot_scope,
    rank_bot_scoped_records as _rank_bot_scoped_records,
)

PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME = "performance_learning_refresh_error.json"


class PerformanceLearningRefreshMarkerError(RuntimeError):
    """Raised when a failed refresh cannot write its quarantine marker."""


class PerformanceLearningLedgerStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read(self, *, strict: bool = True) -> list[PerformanceLearningRecord]:
        """Return the latest valid projection row for each record id.

        The JSONL file can contain repeated record ids when a source row is
        updated. Every non-empty line is still parsed in strict mode so malformed
        or authority-invalid historical rows cannot become invisible to checks.
        """

        records = self.read_history(strict=strict)
        latest: dict[str, PerformanceLearningRecord] = {}
        for record in records:
            latest[record.record_id] = record
        return list(latest.values())

    def read_history(self, *, strict: bool = True) -> list[PerformanceLearningRecord]:
        if not self.path.exists():
            return []
        records: list[PerformanceLearningRecord] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(PerformanceLearningRecord.model_validate(payload))
                elif strict:
                    raise ValueError("line is not a JSON object")
            except Exception as exc:
                if strict:
                    raise ValueError(f"{self.path}:{line_no}: invalid performance-learning record: {exc}") from exc
        return records

    def append_records(self, records: Iterable[PerformanceLearningRecord]) -> int:
        existing = {record.record_id: record for record in self.read(strict=True)}
        new_records = [
            record for record in records
            if _record_fingerprint(existing.get(record.record_id)) != _record_fingerprint(record)
        ]
        if not new_records:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in new_records:
                handle.write(record.model_dump_json(by_alias=True) + "\n")
        return len(new_records)

    def recent_summaries(
        self,
        *,
        bot_id: str = "",
        strategy_id: str = "",
        portfolio_id: str = "",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        records = self.read(strict=False)
        if bot_id and not strategy_id and not portfolio_id:
            records = _rank_bot_scoped_records(records, bot_id, limit)
            return [record.summary_line() for record in records]
        if strategy_id:
            records = [
                record for record in records
                if record.strategy_id == strategy_id or record.scope == strategy_id
            ]
        if portfolio_id:
            records = [
                record for record in records
                if record.portfolio_id == portfolio_id or record.scope == portfolio_id
            ]
        if bot_id and not portfolio_id:
            records = [record for record in records if _matches_bot_scope(record, bot_id)]
        records.sort(key=lambda record: record.event_time, reverse=True)
        return [record.summary_line() for record in records[:limit]]


class PerformanceLearningProjector:
    """Build performance-learning records without mutating source ledgers."""

    def __init__(self, findings_dir: Path) -> None:
        self.findings_dir = Path(findings_dir)
        self.ledger_path = self.findings_dir / "performance_learning_ledger.jsonl"

    def project_to_ledger(self, *, strict: bool = True) -> list[PerformanceLearningRecord]:
        records = self.build_records(strict=strict)
        PerformanceLearningLedgerStore(self.ledger_path).append_records(records)
        return records

    def build_records(self, *, strict: bool = True) -> list[PerformanceLearningRecord]:
        loop_links = _loop_links_by_proposal(self.findings_dir / "loop_run_ledger.jsonl", strict=strict)
        source_context = _SourceContext.from_findings(self.findings_dir, strict=strict)
        records: list[PerformanceLearningRecord] = []
        for proposal in _read_proposal_records(self.findings_dir / "proposal_ledger.jsonl", strict=strict):
            records.extend(_records_from_proposal(proposal, loop_links, source_context))
        for strategy_change in _read_strategy_change_records(
            self.findings_dir / "strategy_change_ledger.jsonl",
            strict=strict,
        ):
            records.extend(_records_from_strategy_change(strategy_change, loop_links, source_context))
        for portfolio_outcome in _read_portfolio_outcome_rows(
            self.findings_dir / "portfolio_outcomes.jsonl",
            strict=strict,
        ):
            records.append(_record_from_portfolio_outcome(portfolio_outcome, loop_links, source_context))
        return records


def refresh_performance_learning_projection(findings_dir: Path) -> int:
    """Best-effort runtime hook for source-ledger writers."""

    findings = Path(findings_dir)
    marker = findings / PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME
    try:
        records = PerformanceLearningProjector(findings).build_records()
        appended = PerformanceLearningLedgerStore(
            findings / "performance_learning_ledger.jsonl"
        ).append_records(records)
    except Exception as exc:
        try:
            _write_refresh_error(marker, exc)
        except OSError as marker_exc:
            raise PerformanceLearningRefreshMarkerError(
                "performance-learning refresh failed and could not write "
                f"{PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME}: {marker_exc}; "
                f"original refresh error: {exc}"
            ) from marker_exc
        raise
    _clear_refresh_error(marker)
    return appended


def _write_refresh_error(path: Path, exc: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")


def _clear_refresh_error(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def validate_performance_learning_records(
    records: Iterable[PerformanceLearningRecord],
) -> list[str]:
    items = list(records)
    messages: list[str] = []
    if not items:
        return [
            "AM-14 performance_learning_ledger: no records found; run "
            "PerformanceLearningProjector.project_to_ledger() from fixture or integration inputs."
        ]
    if not any(record.record_type == PerformanceRecordType.STRATEGY for record in items):
        messages.append("AM-14 performance_learning_ledger: missing strategy learning records.")
    if not any(record.record_type == PerformanceRecordType.PORTFOLIO for record in items):
        messages.append("AM-14 performance_learning_ledger: missing portfolio learning records.")
    if not any(record.expected_deltas.has_any() for record in items):
        messages.append("AM-14 performance_learning_ledger: no expected delta fields are preserved.")
    if not any(record.realized_after_cost_deltas.has_any() for record in items):
        messages.append("AM-14 performance_learning_ledger: no realized after-cost deltas are preserved.")
    if not any(record.source_weekly_signal_ids for record in items):
        messages.append("AM-14 performance_learning_ledger: weekly-to-monthly signal attribution is absent.")
    if not any(record.intended_learning_effects.has_any() for record in items):
        messages.append("AM-14 performance_learning_ledger: intended learning effects are absent.")
    if not any(record.strategy_slice.has_any() for record in items):
        messages.append("AM-14 performance_learning_ledger: strategy slice context is absent.")
    if not any(record.portfolio_context.has_any() for record in items):
        messages.append("AM-14 performance_learning_ledger: portfolio interaction context is absent.")
    for record in items:
        if record.material_approval_evidence and record.source_cadence != SourceCadence.MONTHLY:
            messages.append(
                "AM-14 performance_learning_ledger: non-monthly record "
                f"{record.record_id} is marked material approval evidence."
            )
        if (
            record.source_cadence in {SourceCadence.DAILY, SourceCadence.WEEKLY, SourceCadence.HARNESS}
            and record.authority_level
            in {AuthorityLevel.MONTHLY_REPLAY_AUTHORITY, AuthorityLevel.PERSISTENCE_CONFIRMATION}
        ):
            messages.append(
                "AM-14 performance_learning_ledger: record "
                f"{record.record_id} conflates {record.source_cadence.value} cadence with "
                f"{record.authority_level.value} authority."
            )
        assistant_measured = (
            record.decision_stage in {DecisionStage.MEASURED, DecisionStage.FOLLOW_UP}
            and record.realized_after_cost_deltas.has_any()
            and _is_assistant_driven_measured_record(record)
        )
        if assistant_measured and not (record.proposal_ids or record.strategy_change_record_ids):
            messages.append(
                "AM-20 performance_learning_ledger: measured assistant-driven record "
                f"{record.record_id} is missing proposal or strategy-change trace."
            )
        if (
            assistant_measured
            and record.source_cadence == SourceCadence.MONTHLY
            and not _has_sufficiency_context(record)
        ):
            messages.append(
                "AM-20 performance_learning_ledger: measured assistant-driven record "
                f"{record.record_id} is missing learning-sufficiency context."
            )
    return messages


class _SourceContext:
    def __init__(self, findings_dir: Path) -> None:
        self.findings_dir = Path(findings_dir)
        self.by_run_id: dict[str, dict[str, Any]] = {}
        self.by_proposal_id: dict[str, dict[str, Any]] = {}
        self.by_strategy_change_id: dict[str, dict[str, Any]] = {}
        self.by_portfolio_id: dict[str, dict[str, Any]] = {}
        self.by_path: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_findings(cls, findings_dir: Path, *, strict: bool = True) -> "_SourceContext":
        context = cls(findings_dir)
        proposal_records = _read_proposal_records(
            context.findings_dir / "proposal_ledger.jsonl",
            strict=strict,
        )
        strategy_records = _read_strategy_change_records(
            context.findings_dir / "strategy_change_ledger.jsonl",
            strict=strict,
        )
        portfolio_rows = _read_portfolio_outcome_rows(
            context.findings_dir / "portfolio_outcomes.jsonl",
            strict=strict,
        )
        loop_rows = _read_jsonl_dicts(
            context.findings_dir / "loop_run_ledger.jsonl",
            strict=strict,
        )

        for proposal in proposal_records:
            paths = [
                *proposal.candidate.linked_diagnostics,
                *[path for evaluation in proposal.evaluations for path in evaluation.evidence_paths],
                *[outcome.measurement_path for outcome in proposal.outcomes if outcome.measurement_path],
            ]
            payload = context._context_from_paths(paths)
            context._merge_keyed(context.by_proposal_id, proposal.candidate.proposal_id, payload)
            if proposal.candidate.linked_run_id:
                context._merge_keyed(context.by_run_id, proposal.candidate.linked_run_id, payload)

        for record in strategy_records:
            paths = list(record.evidence_paths)
            if record.monthly_verdict:
                paths.extend(_list_from_payload(record.monthly_verdict, "evidence_paths"))
            if record.follow_up_verdict:
                paths.extend(_list_from_payload(record.follow_up_verdict, "evidence_paths"))
            payload = context._context_from_paths(paths)
            context._merge_keyed(context.by_strategy_change_id, record.record_id, payload)
            context._merge_keyed(context.by_run_id, record.run_id, payload)
            for proposal_id in record.source_proposal_ids:
                context._merge_keyed(context.by_proposal_id, proposal_id, payload)

        for row in portfolio_rows:
            paths = [
                *(_list_from_payload(row, "evidence_paths")),
                str(row.get("monthly_search_brief_path") or ""),
                str(row.get("portfolio_metrics_path") or ""),
            ]
            payload = context._context_from_paths(paths)
            portfolio_id = str(row.get("portfolio_id") or row.get("bot_id") or "portfolio")
            context._merge_keyed(context.by_portfolio_id, portfolio_id, payload)
            for proposal_id in _list_from_payload(row, "proposal_ids") or _list_value(row.get("proposal_id", "")):
                context._merge_keyed(context.by_proposal_id, proposal_id, payload)

        for row in loop_rows:
            paths = [
                *[str(path) for path in row.get("evidence_paths") or []],
                *[str(path) for path in row.get("output_artifacts") or []],
                *[str(path) for path in row.get("approval_packet_paths") or []],
            ]
            for source in row.get("source_records") or []:
                if isinstance(source, dict):
                    paths.append(str(source.get("path") or ""))
            payload = context._context_from_paths(paths)
            run_id = str(row.get("run_id") or row.get("agent_run_id") or "")
            context._merge_keyed(context.by_run_id, run_id, payload)
            for proposal_id in row.get("proposal_ids") or []:
                context._merge_keyed(context.by_proposal_id, str(proposal_id), payload)
        return context

    def for_proposal(self, proposal_id: str, run_id: str = "") -> dict[str, Any]:
        return self._merged(
            self.by_proposal_id.get(proposal_id, {}),
            self.by_run_id.get(run_id, {}),
        )

    def for_strategy_change(self, record: StrategyChangeRecord) -> dict[str, Any]:
        payload = self.by_strategy_change_id.get(record.record_id, {})
        for proposal_id in record.source_proposal_ids:
            payload = self._merged(payload, self.by_proposal_id.get(proposal_id, {}))
        return self._merged(payload, self.by_run_id.get(record.run_id, {}))

    def for_portfolio(self, row: dict[str, Any]) -> dict[str, Any]:
        portfolio_id = str(row.get("portfolio_id") or row.get("bot_id") or "portfolio")
        payload = self.by_portfolio_id.get(portfolio_id, {})
        for proposal_id in _list_from_payload(row, "proposal_ids") or _list_value(row.get("proposal_id", "")):
            payload = self._merged(payload, self.by_proposal_id.get(proposal_id, {}))
        return payload

    def for_paths(self, paths: list[str]) -> dict[str, Any]:
        return self._context_from_paths(paths)

    def _context_from_paths(self, paths: list[str]) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for raw in _dedupe(paths):
            path = _resolve_source_path(raw, self.findings_dir)
            if path is None:
                continue
            key = str(path)
            if key in self.by_path:
                context = self._merged(context, self.by_path[key])
                continue
            payload = _context_from_source_path(path)
            self.by_path[key] = payload
            context = self._merged(context, payload)
        return context

    @staticmethod
    def _merge_keyed(target: dict[str, dict[str, Any]], key: str, payload: dict[str, Any]) -> None:
        if not key or not payload:
            return
        target[key] = _SourceContext._merged(target.get(key, {}), payload)

    @staticmethod
    def _merged(*payloads: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for payload in payloads:
            for key, value in payload.items():
                if value in ("", None, [], {}):
                    continue
                if key == "source_records" and isinstance(value, list):
                    current = merged.get(key) if isinstance(merged.get(key), list) else []
                    merged[key] = [
                        item.model_dump()
                        for item in _dedupe_source_records([
                            *_source_records_from_payloads(current),
                            *_source_records_from_payloads(value),
                        ])
                    ]
                elif isinstance(value, list):
                    merged[key] = _dedupe([*(merged.get(key) or []), *[str(item) for item in value]])
                elif isinstance(value, dict):
                    current = merged.get(key) if isinstance(merged.get(key), dict) else {}
                    merged[key] = {**current, **value}
                elif key not in merged or merged[key] in ("", None):
                    merged[key] = value
        return merged


def _records_from_proposal(
    proposal: ProposalRecord,
    loop_links: dict[str, dict[str, Any]],
    source_context: _SourceContext,
) -> list[PerformanceLearningRecord]:
    candidate = proposal.candidate
    record_type = _record_type_for_proposal_kind(candidate.kind)
    scope = _scope(record_type, candidate.bot_id, candidate.strategy_id, candidate.lifecycle_stage)
    cadence = _proposal_source_cadence(candidate.source, candidate.evaluation_method)
    link = _first_loop_link(loop_links, [candidate.proposal_id])
    candidate_context = source_context.for_paths(candidate.linked_diagnostics)
    records = [
        _enrich_record(PerformanceLearningRecord(
            record_type=record_type,
            scope=scope,
            bot_id=candidate.bot_id,
            strategy_id=candidate.strategy_id,
            portfolio_id=scope if record_type == PerformanceRecordType.PORTFOLIO else "",
            source_cadence=cadence,
            learning_layer=authority_for_cadence(cadence)[0],
            authority_level=authority_for_cadence(cadence)[1],
            decision_stage=DecisionStage.PROPOSED,
            loop_run_id=str(link.get("loop_run_id", "")),
            run_id=candidate.linked_run_id or str(link.get("run_id", "")),
            task_id=str(link.get("task_id", "")),
            agent_run_id=str(link.get("agent_run_id", "")),
            proposal_ids=[candidate.proposal_id],
            deployment_id=candidate.deployment_id,
            source_weekly_signal_ids=_weekly_signal_ids(candidate.linked_diagnostics),
            strategy_config_diff={
                "affected_parameters": candidate.affected_parameters,
                "affected_files": candidate.affected_files,
                "hypothesis_id": candidate.hypothesis_id,
                "lifecycle_stage": candidate.lifecycle_stage,
            },
            intended_learning_effects=_learning_effects_for_cadence(cadence, candidate.kind.value),
            evidence_paths=_paths(candidate.linked_diagnostics),
            summary=candidate.title,
            event_time=_aware(candidate.proposed_at),
            source_records=[
                PerformanceSourceRecord(
                    kind="proposal_candidate",
                    id=candidate.proposal_id,
                    path="proposal_ledger.jsonl",
                )
            ],
        ), candidate_context)
    ]
    for evaluation_index, evaluation in enumerate(proposal.evaluations, start=1):
        cadence = _evaluation_cadence(candidate, evaluation)
        evaluation_context = source_context.for_paths([
            *candidate.linked_diagnostics,
            *evaluation.evidence_paths,
        ])
        records.append(_enrich_record(PerformanceLearningRecord(
            record_type=record_type,
            scope=scope,
            bot_id=candidate.bot_id,
            strategy_id=candidate.strategy_id,
            portfolio_id=scope if record_type == PerformanceRecordType.PORTFOLIO else "",
            source_cadence=cadence,
            learning_layer=authority_for_cadence(cadence)[0],
            authority_level=authority_for_cadence(cadence)[1],
            decision_stage=_stage_for_evaluation_decision(evaluation.decision),
            material_approval_evidence=(
                cadence == SourceCadence.MONTHLY
                and evaluation.decision.lower() in {"approve", "approved"}
            ),
            loop_run_id=str(link.get("loop_run_id", "")),
            run_id=candidate.linked_run_id or str(link.get("run_id", "")),
            task_id=str(link.get("task_id", "")),
            agent_run_id=str(link.get("agent_run_id", "")),
            proposal_ids=[candidate.proposal_id],
            deployment_id=candidate.deployment_id,
            objective_version=evaluation.objective_version,
            source_weekly_signal_ids=_weekly_signal_ids(candidate.linked_diagnostics),
            expected_deltas=PerformanceMetricDeltas(
                objective=evaluation.objective_score,
                confidence=evaluation.confidence,
            ),
            intended_learning_effects=_learning_effects_for_decision(evaluation.decision),
            evidence_paths=_dedupe(evaluation.evidence_paths),
            blocker_reasons=[] if evaluation.decision.lower() != "defer" else [evaluation.decision_reason],
            summary=evaluation.summary or evaluation.decision_reason or candidate.title,
            event_time=_aware(evaluation.evaluated_at),
            source_records=[
                PerformanceSourceRecord(
                    kind="proposal_evaluation",
                    id=_proposal_evaluation_source_id(candidate.proposal_id, evaluation, evaluation_index),
                    path="proposal_ledger.jsonl",
                )
            ],
        ), evaluation_context))
    for outcome_index, outcome in enumerate(proposal.outcomes, start=1):
        cadence = _outcome_cadence(outcome.outcome_source)
        outcome_context = source_context.for_proposal(candidate.proposal_id, candidate.linked_run_id)
        records.append(_enrich_record(PerformanceLearningRecord(
            record_type=record_type,
            scope=scope,
            bot_id=candidate.bot_id,
            strategy_id=candidate.strategy_id,
            portfolio_id=scope if record_type == PerformanceRecordType.PORTFOLIO else "",
            source_cadence=cadence,
            learning_layer=authority_for_cadence(cadence)[0],
            authority_level=authority_for_cadence(cadence)[1],
            decision_stage=DecisionStage.MEASURED
            if cadence != SourceCadence.FOLLOW_UP else DecisionStage.FOLLOW_UP,
            material_approval_evidence=cadence == SourceCadence.MONTHLY,
            proposal_ids=[candidate.proposal_id],
            strategy_change_record_ids=_list_value(outcome.strategy_change_record_id),
            deployment_id=outcome.deployment_id,
            objective_version=outcome.objective_version,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=outcome.objective_delta),
            verdict=outcome.verdict,
            intended_learning_effects=_learning_effects_for_verdict(outcome.verdict),
            evidence_paths=_dedupe(_list_value(outcome.measurement_path)),
            summary=f"{candidate.title}: {outcome.verdict}",
            event_time=_aware(outcome.measured_at),
            source_records=[
                PerformanceSourceRecord(
                    kind="proposal_outcome",
                    id=_proposal_outcome_source_id(candidate.proposal_id, outcome, outcome_index),
                    path="proposal_ledger.jsonl",
                )
            ],
        ), outcome_context))
    return records


def _records_from_strategy_change(
    record: StrategyChangeRecord,
    loop_links: dict[str, dict[str, Any]],
    source_context: _SourceContext,
) -> list[PerformanceLearningRecord]:
    link = _first_loop_link(loop_links, record.source_proposal_ids)
    context = source_context.for_strategy_change(record)
    cadence = _strategy_change_cadence(record.record_type)
    stage = _strategy_change_stage(record)
    records = [
        _enrich_record(PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope=record.strategy_id or record.bot_id,
            bot_id=record.bot_id,
            strategy_id=record.strategy_id,
            source_cadence=cadence,
            learning_layer=authority_for_cadence(cadence)[0],
            authority_level=authority_for_cadence(cadence)[1],
            decision_stage=stage,
            material_approval_evidence=(
                cadence == SourceCadence.MONTHLY
                and stage in {DecisionStage.APPROVED, DecisionStage.EVALUATED}
            ),
            loop_run_id=str(link.get("loop_run_id", "")),
            run_id=record.run_id or str(link.get("run_id", "")),
            run_month=record.run_month,
            task_id=str(link.get("task_id", "")),
            agent_run_id=str(link.get("agent_run_id", "")),
            proposal_ids=record.source_proposal_ids,
            strategy_change_record_ids=[record.record_id],
            approval_request_id=record.approval_request_id or "",
            deployment_id=record.deployment_id or "",
            monthly_search_brief_path=_first_path_named(record.evidence_paths, "monthly_search_brief.json"),
            source_weekly_signal_ids=_weekly_signal_ids_from_payload(record.mutation_diff),
            brief_attribution_ids=_list_from_payload(record.mutation_diff, "brief_attribution_ids"),
            strategy_config_diff=record.mutation_diff,
            expected_deltas=_metric_deltas(record.objective_deltas),
            intended_learning_effects=_learning_effects_from_strategy_change(record),
            strategy_slice=_strategy_slice_from_payload(record.mutation_diff),
            evidence_paths=_dedupe(record.evidence_paths),
            rollback_status=record.rollback_status.value,
            summary=record.decision_reason or record.monthly_status or record.record_type.value,
            event_time=_aware(record.updated_at),
            source_records=[
                PerformanceSourceRecord(
                    kind="strategy_change",
                    id=record.record_id,
                    path="strategy_change_ledger.jsonl",
                )
            ],
        ), context)
    ]
    if record.monthly_verdict:
        records.append(_enrich_record(
            _strategy_verdict_record(record, record.monthly_verdict, SourceCadence.MONTHLY),
            context,
        ))
    if record.follow_up_verdict:
        records.append(_enrich_record(
            _strategy_verdict_record(record, record.follow_up_verdict, SourceCadence.FOLLOW_UP),
            context,
        ))
    return records


def _strategy_verdict_record(
    record: StrategyChangeRecord,
    verdict_payload: dict[str, Any],
    cadence: SourceCadence,
) -> PerformanceLearningRecord:
    return PerformanceLearningRecord(
        record_type=PerformanceRecordType.STRATEGY,
        scope=record.strategy_id or record.bot_id,
        bot_id=record.bot_id,
        strategy_id=record.strategy_id,
        source_cadence=cadence,
        learning_layer=authority_for_cadence(cadence)[0],
        authority_level=authority_for_cadence(cadence)[1],
        decision_stage=DecisionStage.MEASURED if cadence == SourceCadence.MONTHLY else DecisionStage.FOLLOW_UP,
        material_approval_evidence=cadence == SourceCadence.MONTHLY,
        run_id=record.run_id,
        run_month=record.run_month,
        proposal_ids=record.source_proposal_ids,
        strategy_change_record_ids=[record.record_id],
        approval_request_id=record.approval_request_id or "",
        deployment_id=record.deployment_id or "",
        realized_after_cost_deltas=_metric_deltas(verdict_payload),
        verdict=str(verdict_payload.get("verdict") or verdict_payload.get("status") or ""),
        intended_learning_effects=_learning_effects_for_verdict(
            str(verdict_payload.get("verdict") or verdict_payload.get("status") or "")
        ),
        strategy_slice=_strategy_slice_from_payload(verdict_payload),
        evidence_paths=_dedupe(_list_from_payload(verdict_payload, "evidence_paths")),
        rollback_status=record.rollback_status.value,
        summary=str(verdict_payload.get("summary") or verdict_payload.get("verdict") or ""),
        event_time=_aware(record.updated_at),
        source_records=[
            PerformanceSourceRecord(
                kind=f"strategy_{cadence.value}_verdict",
                id=record.record_id,
                path="strategy_change_ledger.jsonl",
            )
        ],
    )


def _record_from_portfolio_outcome(
    row: dict[str, Any],
    loop_links: dict[str, dict[str, Any]],
    source_context: _SourceContext,
) -> PerformanceLearningRecord:
    proposal_ids = _list_from_payload(row, "proposal_ids") or _list_value(row.get("proposal_id", ""))
    link = _first_loop_link(loop_links, proposal_ids)
    cadence = _outcome_cadence(str(row.get("outcome_source") or "early_warning"))
    portfolio_id = str(row.get("portfolio_id") or row.get("bot_id") or "portfolio")
    return _enrich_record(PerformanceLearningRecord(
        record_type=PerformanceRecordType.PORTFOLIO,
        scope=portfolio_id,
        bot_id=str(row.get("bot_id") or "PORTFOLIO"),
        portfolio_id=portfolio_id,
        source_cadence=cadence,
        learning_layer=authority_for_cadence(cadence)[0],
        authority_level=authority_for_cadence(cadence)[1],
        decision_stage=DecisionStage.MEASURED if cadence != SourceCadence.FOLLOW_UP else DecisionStage.FOLLOW_UP,
        material_approval_evidence=cadence == SourceCadence.MONTHLY,
        loop_run_id=str(link.get("loop_run_id", "")),
        run_id=str(row.get("run_id") or link.get("run_id", "")),
        task_id=str(link.get("task_id", "")),
        agent_run_id=str(link.get("agent_run_id", "")),
        proposal_ids=proposal_ids,
        deployment_id=str(row.get("deployment_id") or ""),
        monthly_search_brief_path=str(row.get("monthly_search_brief_path") or ""),
        source_weekly_signal_ids=_list_from_payload(row, "source_weekly_signal_ids"),
        brief_attribution_ids=_list_from_payload(row, "brief_attribution_ids"),
        portfolio_allocation_diff=dict(row.get("portfolio_allocation_diff") or {}),
        realized_after_cost_deltas=_portfolio_metric_deltas(row),
        verdict=str(row.get("verdict") or ""),
        intended_learning_effects=_learning_effects_from_portfolio(row),
        portfolio_context=_portfolio_context(row),
        evidence_paths=_dedupe(_list_from_payload(row, "evidence_paths")),
        summary=str(row.get("summary") or row.get("category") or row.get("verdict") or "portfolio outcome"),
        event_time=_parse_time(row.get("measured_at") or row.get("deployed_at")),
        source_records=[
            PerformanceSourceRecord(
                kind="portfolio_outcome",
                id=str(row.get("outcome_id") or row.get("suggestion_id") or row.get("proposal_id") or ""),
                path="portfolio_outcomes.jsonl",
            )
        ],
    ), source_context.for_portfolio(row))


def _enrich_record(
    record: PerformanceLearningRecord,
    context: dict[str, Any],
) -> PerformanceLearningRecord:
    if not context:
        return record
    context_evidence_paths = _context_evidence_paths_for_stage(record, context)
    context_source_records = _context_source_records_for_stage(record, context)
    updates = {
        "data_bundle_id": record.data_bundle_id or str(context.get("data_bundle_id") or ""),
        "objective_version": record.objective_version or str(context.get("objective_version") or ""),
        "scoring_profile": record.scoring_profile or str(context.get("scoring_profile") or ""),
        "verifier_version": record.verifier_version or str(context.get("verifier_version") or ""),
        "artifact_authority_version": (
            record.artifact_authority_version
            or str(context.get("artifact_authority_version") or "")
        ),
        "monthly_search_brief_path": (
            record.monthly_search_brief_path
            or str(context.get("monthly_search_brief_path") or "")
        ),
        "learning_sufficiency_manifest_id": (
            record.learning_sufficiency_manifest_id
            or str(context.get("learning_sufficiency_manifest_id") or "")
        ),
        "learning_sufficiency_manifest_path": (
            record.learning_sufficiency_manifest_path
            or str(context.get("learning_sufficiency_manifest_path") or "")
        ),
        "learning_sufficiency_status": (
            record.learning_sufficiency_status
            or str(context.get("learning_sufficiency_status") or "")
        ),
        "learning_capability_status": {
            **dict(context.get("learning_capability_status") or {}),
            **record.learning_capability_status,
        },
        "supported_learning_capabilities": _dedupe([
            *record.supported_learning_capabilities,
            *[str(item) for item in context.get("supported_learning_capabilities", [])],
        ]),
        "blocked_learning_capability_ids": _dedupe([
            *record.blocked_learning_capability_ids,
            *[str(item) for item in context.get("blocked_learning_capability_ids", [])],
        ]),
        "source_weekly_signal_ids": _dedupe([
            *record.source_weekly_signal_ids,
            *[str(item) for item in context.get("source_weekly_signal_ids", [])],
        ]),
        "brief_attribution_ids": _dedupe([
            *record.brief_attribution_ids,
            *[str(item) for item in context.get("brief_attribution_ids", [])],
        ]),
        "evidence_paths": _dedupe([
            *record.evidence_paths,
            *context_evidence_paths,
        ]),
        "source_records": _dedupe_source_records([
            *record.source_records,
            *context_source_records,
        ]),
    }
    effects = record.intended_learning_effects.model_dump()
    context_effects = context.get("intended_learning_effects")
    if not _is_decision_stage(record) and isinstance(context_effects, dict):
        effects.update({
            key: value for key, value in context_effects.items()
            if value not in ("", None, [], {})
        })
        updates["intended_learning_effects"] = IntendedLearningEffects.model_validate(effects)
    if (
        _allows_hindsight_context(record)
        and not record.portfolio_context.has_any()
        and isinstance(context.get("portfolio_context"), dict)
    ):
        updates["portfolio_context"] = PortfolioInteractionContext.model_validate(context["portfolio_context"])
    if (
        _allows_hindsight_context(record)
        and not record.strategy_slice.has_any()
        and isinstance(context.get("strategy_slice"), dict)
    ):
        updates["strategy_slice"] = StrategySliceContext.model_validate(context["strategy_slice"])
    return record.model_copy(update=updates)


def _context_from_source_path(path: Path) -> dict[str, Any]:
    name = path.name.lower()
    payload = _read_json_file(path)
    if not isinstance(payload, dict):
        return {}
    source = {
        "source_records": [
            {"kind": _source_kind_for_name(name), "id": _source_id(payload), "path": _path_text(path)}
        ]
    }
    if name in {"run_manifest.json", "monthly_run_manifest.json"}:
        return _SourceContext._merged(source, _context_from_monthly_manifest(payload, path))
    if name == "learning_sufficiency_manifest.json":
        return _SourceContext._merged(source, _context_from_learning_sufficiency_manifest(payload, path))
    if name == "artifact_index.json":
        return _SourceContext._merged(source, _context_from_artifact_index(payload, path))
    if name == "monthly_search_brief.json":
        return _SourceContext._merged(source, _context_from_monthly_search_brief(payload, path))
    if name in {"outcome_priors_snapshot.json", "outcome_priors.json"}:
        return _SourceContext._merged(source, _context_from_outcome_priors(payload, path))
    if name in {
        "monthly_evidence_verification.json",
        "monthly_evidence_verifier.json",
    } or name.startswith("monthly_evidence_verification"):
        return _SourceContext._merged(source, _context_from_evidence_verification(payload))
    if name in {
        "portfolio_rolling_metrics.json",
        "portfolio_metrics.json",
        "portfolio_synergy.json",
        "portfolio_risk_card.json",
    }:
        return _SourceContext._merged(source, {"portfolio_context": _portfolio_context(payload).model_dump()})
    return source


def _context_from_monthly_manifest(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    data_bundle_id = (
        str(payload.get("data_bundle_checksum") or "")
        or str(payload.get("data_bundle_manifest_path") or "")
        or str(payload.get("data_manifest_checksum") or "")
    )
    evidence_paths = [
        _path_text(path),
        str(payload.get("monthly_search_brief_path") or ""),
        str(payload.get("outcome_prior_snapshot_path") or ""),
        str(payload.get("learning_sufficiency_manifest_path") or ""),
    ]
    context = {
        "data_bundle_id": data_bundle_id,
        "objective_version": str(payload.get("objective_version") or ""),
        "scoring_profile": str(payload.get("workflow_contract_version") or payload.get("mode") or ""),
        "monthly_search_brief_path": str(payload.get("monthly_search_brief_path") or ""),
        "learning_sufficiency_manifest_path": str(payload.get("learning_sufficiency_manifest_path") or ""),
        "learning_sufficiency_status": str(payload.get("learning_sufficiency_status") or ""),
        "supported_learning_capabilities": _list_from_payload(payload, "supported_learning_capabilities"),
        "blocked_learning_capability_ids": _list_from_payload(payload, "blocked_learning_capabilities"),
        "source_weekly_signal_ids": _list_from_payload(payload, "source_weekly_signal_ids"),
        "brief_attribution_ids": _dedupe(_list_value(payload.get("monthly_search_brief_id", ""))),
        "evidence_paths": _dedupe(evidence_paths),
    }
    sufficiency_path = _resolve_artifact_index_path(
        str(payload.get("learning_sufficiency_manifest_path") or ""),
        path.parent,
    )
    sufficiency_payload = _read_json_file(sufficiency_path) if sufficiency_path else None
    if isinstance(sufficiency_payload, dict):
        context = _SourceContext._merged(
            context,
            _context_from_learning_sufficiency_manifest(sufficiency_payload, sufficiency_path),
        )
    return context


def _context_from_learning_sufficiency_manifest(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    capability_status = payload.get("capability_status") if isinstance(payload.get("capability_status"), dict) else {}
    source_authoritative = manifest_runtime_support_source_authoritative(payload)
    status_by_capability = {
        str(capability_id): str(status.get("status") or "")
        for capability_id, status in capability_status.items()
        if isinstance(status, dict)
    } if source_authoritative else {}
    return {
        "learning_sufficiency_manifest_id": str(payload.get("manifest_id") or ""),
        "learning_sufficiency_manifest_path": _path_text(path),
        "learning_sufficiency_status": str(payload.get("eligibility") or "") if source_authoritative else "diagnostics_only",
        "learning_capability_status": status_by_capability,
        "supported_learning_capabilities": (
            _list_from_payload(payload, "supported_learning_capabilities") if source_authoritative else []
        ),
        "blocked_learning_capability_ids": (
            _list_from_payload(payload, "blocked_learning_capabilities") if source_authoritative else []
        ),
        "evidence_paths": [_path_text(path)],
    }


def _context_from_artifact_index(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    evidence_paths = [str(path)]
    for key in (
        "monthly_search_brief.json",
        "outcome_priors_snapshot.json",
        "monthly_evidence_verification.json",
        "learning_sufficiency_manifest.json",
        "portfolio_synergy.json",
    ):
        if artifacts.get(key):
            evidence_paths.append(str(artifacts[key]))
    context = {
        "scoring_profile": str(payload.get("index_version") or ""),
        "artifact_authority_version": "artifact_authority_registry_v1",
        "evidence_paths": _dedupe(evidence_paths),
    }
    for raw in evidence_paths[1:]:
        nested = _resolve_artifact_index_path(raw, path.parent)
        if nested is not None and nested != path:
            context = _SourceContext._merged(context, _context_from_source_path(nested))
    return context


def _context_from_monthly_search_brief(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    attribution = payload.get("attribution") if isinstance(payload.get("attribution"), dict) else {}
    attribution_ids: list[str] = []
    for key, values in attribution.items():
        attribution_ids.append(str(key))
        attribution_ids.extend(str(value) for value in values or [])
    return {
        "monthly_search_brief_path": _path_text(path),
        "source_weekly_signal_ids": _list_from_payload(payload, "source_weekly_signal_ids"),
        "brief_attribution_ids": _dedupe([
            *attribution_ids,
            *(_list_value(payload.get("monthly_search_brief_id", ""))),
        ]),
        "evidence_paths": [_path_text(path)],
    }


def _context_from_outcome_priors(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    prior_ids = _list_from_payload(payload, "source_outcome_ids")
    if payload.get("prior_id"):
        prior_ids.append(str(payload["prior_id"]))
    return {
        "evidence_paths": [_path_text(path), *(_list_from_payload(payload, "evidence_paths"))],
        "intended_learning_effects": {
            "outcome_prior_update": ";".join(_dedupe(prior_ids)) or "outcome prior snapshot linked",
            "search_allocation_change": str(payload.get("allocation_multiplier") or ""),
            "evidence_gate_calibration": str(payload.get("gate_strictness") or ""),
            "rollback_priority": str(payload.get("rollback_priority") or ""),
        },
    }


def _context_from_evidence_verification(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "verifier_version": str(
            payload.get("verifier_version")
            or payload.get("schema_version")
            or payload.get("verification_version")
            or "monthly_evidence_verifier_v1"
        )
    }


def _resolve_source_path(raw: str, findings_dir: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidates = [Path(text)]
    if not Path(text).is_absolute():
        candidates.extend([
            findings_dir / text,
            findings_dir.parent / text,
            findings_dir.parent.parent / text,
        ])
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _resolve_artifact_index_path(raw: str, artifact_root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    candidates = [Path(text)]
    if not Path(text).is_absolute():
        candidates.append(artifact_root / text)
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _read_json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _source_kind_for_name(name: str) -> str:
    aliases = {
        "run_manifest.json": "monthly_run_manifest",
        "monthly_run_manifest.json": "monthly_run_manifest",
        "artifact_index.json": "artifact_index",
        "monthly_search_brief.json": "monthly_search_brief",
        "outcome_priors_snapshot.json": "outcome_priors_snapshot",
        "outcome_priors.json": "outcome_priors_snapshot",
        "learning_sufficiency_manifest.json": "learning_sufficiency_manifest",
        "portfolio_rolling_metrics.json": "portfolio_metrics",
        "portfolio_metrics.json": "portfolio_metrics",
        "portfolio_synergy.json": "portfolio_metrics",
        "portfolio_risk_card.json": "portfolio_metrics",
    }
    if name.startswith("monthly_evidence_verification"):
        return "monthly_evidence_verification"
    return aliases.get(name, Path(name).stem)


def _source_id(payload: dict[str, Any]) -> str:
    for key in (
        "manifest_id",
        "monthly_search_brief_id",
        "run_id",
        "prior_id",
        "outcome_id",
        "verification_id",
        "id",
    ):
        if payload.get(key):
            return str(payload[key])
    return ""


def _read_proposal_records(path: Path, *, strict: bool = True) -> list[ProposalRecord]:
    records: dict[str, ProposalRecord] = {}
    for event in _read_jsonl_dicts(path, strict=strict):
        etype = event.get("type")
        payload = event.get("payload") or {}
        try:
            if etype == "candidate":
                candidate = ProposalCandidate.model_validate(payload)
                records.setdefault(candidate.proposal_id, ProposalRecord(candidate=candidate))
            elif etype == "evaluation":
                proposal_id = str(payload.get("proposal_id") or "")
                if proposal_id in records:
                    records[proposal_id].evaluations.append(ProposalEvaluation.model_validate(payload))
            elif etype == "outcome":
                proposal_id = str(payload.get("proposal_id") or "")
                if proposal_id in records:
                    records[proposal_id].outcomes.append(ProposalOutcome.model_validate(payload))
        except Exception as exc:
            if strict:
                raise ValueError(
                    f"{path}: invalid proposal ledger {etype or 'unknown'} event: {exc}"
                ) from exc
            continue
    return list(records.values())


def _read_strategy_change_records(path: Path, *, strict: bool = True) -> list[StrategyChangeRecord]:
    records: dict[str, StrategyChangeRecord] = {}
    for event in _read_jsonl_dicts(path, strict=strict):
        try:
            if "record_id" in event and "record_type" in event:
                record = StrategyChangeRecord.model_validate(event)
                records[record.record_id] = record
                continue
            etype = event.get("type")
            payload = event.get("payload") or {}
            if etype == "record":
                record = StrategyChangeRecord.model_validate(payload)
                records[record.record_id] = record
            elif etype == "update":
                record_id = str(payload.get("record_id") or "")
                if record_id in records:
                    merged = records[record_id].model_dump(mode="json")
                    merged.update(payload)
                    records[record_id] = StrategyChangeRecord.model_validate(merged)
        except Exception as exc:
            if strict:
                raise ValueError(
                    f"{path}: invalid strategy-change ledger event: {exc}"
                ) from exc
            continue
    return sorted(records.values(), key=lambda item: item.updated_at)


def _read_jsonl_dicts(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            continue
        if not isinstance(payload, dict):
            if strict:
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            continue
        rows.append(payload)
    return rows


def _read_portfolio_outcome_rows(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_read_jsonl_dicts(path, strict=strict), start=1):
        try:
            _validate_portfolio_outcome_row(row)
        except ValueError as exc:
            if strict:
                raise ValueError(f"{path}:{index}: invalid portfolio outcome row: {exc}") from exc
            continue
        rows.append(row)
    return rows


def _validate_portfolio_outcome_row(row: dict[str, Any]) -> None:
    if not any(str(row.get(key) or "").strip() for key in ("outcome_id", "suggestion_id", "proposal_id")):
        raise ValueError("outcome_id, suggestion_id, or proposal_id is required")
    if not str(row.get("portfolio_id") or row.get("bot_id") or "").strip():
        raise ValueError("portfolio_id or bot_id is required")
    if not str(row.get("outcome_source") or "").strip():
        raise ValueError("outcome_source is required")
    if not str(row.get("verdict") or "").strip():
        raise ValueError("verdict is required")
    if not str(row.get("measured_at") or "").strip():
        raise ValueError("measured_at is required")
    _parse_required_time(row.get("measured_at"))
    has_delta = False
    for key in (
        "objective_delta",
        "composite_delta",
        "after_cost_objective_delta",
        "return_delta",
        "drawdown_delta",
        "cost_delta",
    ):
        if key not in row or row.get(key) in ("", None):
            continue
        if not _has_number(row.get(key)):
            raise ValueError(f"{key} must be finite numeric")
        has_delta = True
    if not has_delta:
        raise ValueError("at least one realized delta field is required")


def _has_number(value: Any) -> bool:
    return _float_or_none(value) is not None


def _parse_required_time(value: Any) -> None:
    raw = str(value or "").strip()
    datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _record_fingerprint(record: PerformanceLearningRecord | None) -> str:
    if record is None:
        return ""
    payload = record.model_dump(mode="json")
    payload.pop("generated_at", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _loop_links_by_proposal(path: Path, *, strict: bool = True) -> dict[str, dict[str, Any]]:
    links: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl_dicts(path, strict=strict):
        for proposal_id in row.get("proposal_ids") or []:
            if proposal_id and proposal_id not in links:
                links[str(proposal_id)] = {
                    "loop_run_id": row.get("loop_run_id", ""),
                    "task_id": row.get("task_id", ""),
                    "run_id": row.get("run_id") or row.get("agent_run_id", ""),
                    "agent_run_id": row.get("agent_run_id", ""),
                }
    return links


def _first_loop_link(loop_links: dict[str, dict[str, Any]], proposal_ids: list[str]) -> dict[str, Any]:
    for proposal_id in proposal_ids:
        if proposal_id in loop_links:
            return loop_links[proposal_id]
    return {}


def _proposal_evaluation_source_id(
    proposal_id: str,
    evaluation: ProposalEvaluation,
    event_index: int,
) -> str:
    return ":".join([proposal_id, "evaluation", str(event_index)])


def _proposal_outcome_source_id(
    proposal_id: str,
    outcome: ProposalOutcome,
    event_index: int,
) -> str:
    event_key = (
        outcome.monthly_outcome_id
        or outcome.measurement_path
        or outcome.strategy_change_record_id
        or outcome.outcome_source
        or str(event_index)
    )
    return ":".join([proposal_id, "outcome", str(event_index), _source_token(event_key)])


def _source_token(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").replace(" ", "_")


def _record_type_for_proposal_kind(kind: ProposalKind) -> PerformanceRecordType:
    if kind == ProposalKind.PORTFOLIO_CHANGE:
        return PerformanceRecordType.PORTFOLIO
    return PerformanceRecordType.STRATEGY


def _scope(record_type: PerformanceRecordType, bot_id: str, strategy_id: str, lifecycle_stage: str) -> str:
    if record_type == PerformanceRecordType.PORTFOLIO:
        return bot_id or lifecycle_stage or "portfolio"
    return strategy_id or bot_id or lifecycle_stage or "global"


def _proposal_source_cadence(source: ProposalSource, evaluation_method: str = "") -> SourceCadence:
    value = source.value
    method = evaluation_method.lower()
    if "harness" in value or "benchmark" in method:
        return SourceCadence.HARNESS
    if value == ProposalSource.LLM_DAILY.value:
        return SourceCadence.DAILY
    if value in {ProposalSource.LLM_WEEKLY.value, ProposalSource.PORTFOLIO.value}:
        return SourceCadence.WEEKLY
    if value.startswith("monthly") or "monthly" in method or "replay" in method:
        return SourceCadence.MONTHLY
    return SourceCadence.DAILY


def _evaluation_cadence(candidate: ProposalCandidate, evaluation: ProposalEvaluation) -> SourceCadence:
    method = evaluation.method.lower()
    if "harness" in method or "benchmark" in method:
        return SourceCadence.HARNESS
    if "monthly" in method or "replay" in method or "approval" in method:
        return SourceCadence.MONTHLY
    return _proposal_source_cadence(candidate.source, candidate.evaluation_method)


def _outcome_cadence(outcome_source: str) -> SourceCadence:
    source = outcome_source.lower()
    if "monthly" in source or "replay" in source:
        return SourceCadence.MONTHLY
    if "follow" in source or "persistence" in source:
        return SourceCadence.FOLLOW_UP
    if "weekly" in source or "search" in source:
        return SourceCadence.WEEKLY
    if "harness" in source or "benchmark" in source:
        return SourceCadence.HARNESS
    return SourceCadence.DAILY


def _stage_for_evaluation_decision(decision: str) -> DecisionStage:
    value = decision.lower()
    if value in {"approve", "approved", "adopt", "accepted"}:
        return DecisionStage.APPROVED
    if value in {"reject", "rejected"}:
        return DecisionStage.REJECTED
    return DecisionStage.EVALUATED


def _strategy_change_cadence(record_type: StrategyChangeRecordType) -> SourceCadence:
    if record_type == StrategyChangeRecordType.FOLLOW_UP_VERDICT:
        return SourceCadence.FOLLOW_UP
    if record_type in {StrategyChangeRecordType.WATCH, StrategyChangeRecordType.QUARANTINE}:
        return SourceCadence.DAILY
    return SourceCadence.MONTHLY


def _strategy_change_stage(record: StrategyChangeRecord) -> DecisionStage:
    if record.record_type == StrategyChangeRecordType.PROPOSED_CHANGE:
        return DecisionStage.PROPOSED
    if record.record_type == StrategyChangeRecordType.DEPLOYED_CHANGE:
        return DecisionStage.DEPLOYED
    if record.record_type == StrategyChangeRecordType.ROLLBACK:
        return DecisionStage.ROLLBACK
    if record.record_type == StrategyChangeRecordType.FOLLOW_UP_VERDICT:
        return DecisionStage.FOLLOW_UP
    if record.monthly_status.lower() in {"approved", "approve", "accepted", "adopted"}:
        return DecisionStage.APPROVED
    if record.monthly_status.lower() in {"rejected", "reject", "no_change"}:
        return DecisionStage.REJECTED
    return DecisionStage.EVALUATED


def _metric_deltas(payload: dict[str, Any]) -> PerformanceMetricDeltas:
    return PerformanceMetricDeltas(
        objective=_float_first(payload, "objective", "objective_delta", "objective_score", "composite_delta"),
        return_=_float_first(payload, "return", "return_delta", "net_return_delta"),
        drawdown=_float_first(payload, "drawdown", "drawdown_delta", "max_drawdown_delta"),
        turnover=_float_first(payload, "turnover", "turnover_delta"),
        cost=_float_first(payload, "cost", "cost_delta", "cost_usd", "cost_bps"),
        slippage=_float_first(payload, "slippage", "slippage_delta"),
        confidence=_float_first(payload, "confidence", "confidence_delta"),
    )


def _portfolio_metric_deltas(row: dict[str, Any]) -> PerformanceMetricDeltas:
    return PerformanceMetricDeltas(
        objective=_float_first(row, "objective_delta", "composite_delta", "after_cost_objective_delta"),
        return_=_float_first(row, "return_delta", "portfolio_return_delta"),
        drawdown=_float_first(row, "drawdown_delta", "max_drawdown_delta"),
        turnover=_float_first(row, "turnover_delta"),
        cost=_float_first(row, "cost_delta", "cost_usd", "cost_bps"),
        slippage=_float_first(row, "slippage_delta"),
        confidence=_float_first(row, "confidence_delta", "direction_accuracy"),
    )


def _strategy_slice_from_payload(payload: dict[str, Any]) -> StrategySliceContext:
    source = dict(payload.get("strategy_slice") or payload.get("slice_context") or payload)
    return StrategySliceContext(
        regime=str(source.get("regime") or ""),
        symbol=str(source.get("symbol") or ""),
        session=str(source.get("session") or ""),
        side=str(source.get("side") or source.get("long_short") or ""),
        liquidity=str(source.get("liquidity") or ""),
        sample_size=_int_or_none(source.get("sample_size")),
        trade_count=_int_or_none(source.get("trade_count")),
        cost_bps=_float_or_none(source.get("cost_bps")),
        failure_mode=str(source.get("failure_mode") or ""),
    )


def _portfolio_context(row: dict[str, Any]) -> PortfolioInteractionContext:
    source = dict(row.get("portfolio_context") or row)
    return PortfolioInteractionContext(
        allocation_weights=_float_map(source.get("allocation_weights")),
        risk_budgets=_float_map(source.get("risk_budgets")),
        exposure=_float_map(source.get("exposure")),
        correlation=_float_map(source.get("correlation")),
        drawdown_overlap=_float_map(source.get("drawdown_overlap")),
        crowding=str(source.get("crowding") or ""),
        cannibalization=str(source.get("cannibalization") or ""),
        marginal_contribution=_float_map(source.get("marginal_contribution")),
        concentration=str(source.get("concentration") or ""),
        liquidity_constraints=[str(item) for item in source.get("liquidity_constraints") or []],
    )


def _learning_effects_for_cadence(cadence: SourceCadence, category: str = "") -> IntendedLearningEffects:
    if cadence == SourceCadence.WEEKLY:
        return IntendedLearningEffects(search_allocation_change=f"bounded search prior: {category}")
    if cadence == SourceCadence.MONTHLY:
        return IntendedLearningEffects(evidence_gate_calibration=f"monthly replay gate: {category}")
    if cadence == SourceCadence.HARNESS:
        return IntendedLearningEffects(notes=["benchmark-only prompt or validator learning"])
    return IntendedLearningEffects(notes=[f"diagnostic context: {category}"] if category else [])


def _learning_effects_for_decision(decision: str) -> IntendedLearningEffects:
    value = decision.lower()
    if value in {"approve", "approved", "adopt", "accepted"}:
        return IntendedLearningEffects(evidence_gate_calibration="approval gate supported by source evidence")
    if value in {"reject", "rejected"}:
        return IntendedLearningEffects(search_allocation_change="reduce priority for rejected neighborhood")
    if value in {"defer", "watch"}:
        return IntendedLearningEffects(watch="insufficient evidence; keep under review")
    return IntendedLearningEffects(notes=[decision] if decision else [])


def _learning_effects_for_verdict(verdict: str) -> IntendedLearningEffects:
    value = verdict.lower()
    if value in {"improved", "positive"}:
        return IntendedLearningEffects(outcome_prior_update="reinforce similar future candidates")
    if value in {"regressed", "negative"}:
        return IntendedLearningEffects(
            outcome_prior_update="penalize similar future candidates",
            rollback_priority="review rollback priority",
        )
    return IntendedLearningEffects(watch="outcome inconclusive or insufficient data")


def _learning_effects_from_strategy_change(record: StrategyChangeRecord) -> IntendedLearningEffects:
    if record.rollback_status != RollbackStatus.NONE or record.record_type == StrategyChangeRecordType.ROLLBACK:
        return IntendedLearningEffects(rollback_priority=record.rollback_status.value)
    if record.record_type == StrategyChangeRecordType.QUARANTINE:
        return IntendedLearningEffects(quarantine="strategy quarantined")
    if record.record_type == StrategyChangeRecordType.WATCH:
        return IntendedLearningEffects(watch="strategy watch state")
    return _learning_effects_for_decision(record.monthly_status or record.record_type.value)


def _learning_effects_from_portfolio(row: dict[str, Any]) -> IntendedLearningEffects:
    verdict = str(row.get("verdict") or "")
    effect = _learning_effects_for_verdict(verdict)
    if row.get("risk_budget_change"):
        effect.search_allocation_change = "portfolio risk-budget calibration"
    return effect


def _weekly_signal_ids(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        raw = str(value)
        if "weekly" in raw.lower() or "signal" in raw.lower():
            result.append(raw)
    return _dedupe(result)


def _weekly_signal_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    return _dedupe(
        _list_from_payload(payload, "source_weekly_signal_ids")
        or _list_from_payload(payload, "weekly_signal_ids")
    )


def _first_path_named(paths: list[str], filename: str) -> str:
    for path in paths:
        if Path(path).name == filename:
            return path
    return ""


def _paths(values: list[str]) -> list[str]:
    return _dedupe([value for value in values if "/" in str(value) or "\\" in str(value)])


def _list_value(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _list_from_payload(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, list | tuple | set):
        return _dedupe([str(item) for item in value if str(item)])
    if isinstance(value, str) and value:
        return [value]
    return []


def _float_first(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if value in ("", None) or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        parsed = _float_or_none(raw)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, str) and value:
        try:
            return _aware(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip().replace("\\", "/")
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _context_evidence_paths_for_stage(
    record: PerformanceLearningRecord,
    context: dict[str, Any],
) -> list[str]:
    paths = [str(item) for item in context.get("evidence_paths", [])]
    if _is_decision_stage(record):
        paths = [path for path in paths if not _is_hindsight_context_path(path)]
    return _dedupe(paths)


def _context_source_records_for_stage(
    record: PerformanceLearningRecord,
    context: dict[str, Any],
) -> list[PerformanceSourceRecord]:
    records = _source_records_from_payloads([
        item for item in context.get("source_records", []) if isinstance(item, dict)
    ])
    if _is_decision_stage(record):
        records = [
            source for source in records
            if source.kind not in _HINDSIGHT_CONTEXT_SOURCE_KINDS
            and not _is_hindsight_context_path(source.path)
        ]
    return _dedupe_source_records(records)


def _is_decision_stage(record: PerformanceLearningRecord) -> bool:
    return record.decision_stage in {
        DecisionStage.PROPOSED,
        DecisionStage.EVALUATED,
        DecisionStage.APPROVED,
        DecisionStage.DEPLOYED,
        DecisionStage.REJECTED,
    }


def _allows_hindsight_context(record: PerformanceLearningRecord) -> bool:
    return record.decision_stage in {
        DecisionStage.MEASURED,
        DecisionStage.FOLLOW_UP,
        DecisionStage.ROLLBACK,
    }


def _has_sufficiency_context(record: PerformanceLearningRecord) -> bool:
    return bool(
        (record.learning_sufficiency_manifest_id or record.learning_sufficiency_manifest_path)
        and record.learning_sufficiency_status
        and (
            record.learning_capability_status
            or record.supported_learning_capabilities
            or record.blocked_learning_capability_ids
        )
    )


def _is_assistant_driven_measured_record(record: PerformanceLearningRecord) -> bool:
    return bool(
        record.material_approval_evidence
        or record.proposal_ids
        or record.strategy_change_record_ids
        or any(
            source.kind in {"proposal_outcome", "strategy_change", "portfolio_outcome"}
            for source in record.source_records
        )
    )


def _is_hindsight_context_path(raw: str) -> bool:
    return Path(str(raw or "").replace("\\", "/")).name.lower() in _HINDSIGHT_CONTEXT_FILENAMES


def _dedupe_source_records(records: list[PerformanceSourceRecord]) -> list[PerformanceSourceRecord]:
    result: list[PerformanceSourceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for record in records:
        normalized = record.model_copy(update={"path": record.path.replace("\\", "/")})
        key = (normalized.kind, normalized.id, normalized.path)
        if key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _source_records_from_payloads(values: list[Any]) -> list[PerformanceSourceRecord]:
    records: list[PerformanceSourceRecord] = []
    for value in values:
        try:
            records.append(PerformanceSourceRecord.model_validate(value))
        except Exception:
            continue
    return records


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


_HINDSIGHT_CONTEXT_SOURCE_KINDS = {
    "outcome_priors_snapshot",
    "portfolio_follow_up_outcome",
    "portfolio_metrics",
    "strategy_follow_up_verdict",
    "strategy_monthly_verdict",
}


_HINDSIGHT_CONTEXT_FILENAMES = {
    "outcome_priors.json",
    "outcome_priors_snapshot.json",
    "portfolio_follow_up_outcome.json",
    "portfolio_metrics.json",
    "portfolio_rolling_metrics.json",
    "portfolio_risk_card.json",
    "portfolio_synergy.json",
    "strategy_monthly_verdict.json",
}
