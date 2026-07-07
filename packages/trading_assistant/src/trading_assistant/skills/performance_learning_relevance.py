"""Prompt relevance ranking for performance-learning records.

Relevance is intentionally derived from relation keys already projected into
records; this module does not infer portfolio membership from external config.
"""

from __future__ import annotations

from trading_assistant.schemas.performance_learning_ledger import (
    PerformanceLearningRecord,
    PerformanceRecordType,
)


def matches_bot_scope(record: PerformanceLearningRecord, bot_id: str) -> bool:
    if not bot_id or record.bot_id == bot_id:
        return True
    if not record.bot_id and record.record_type != PerformanceRecordType.PORTFOLIO:
        return True
    return False


def rank_bot_scoped_records(
    records: list[PerformanceLearningRecord],
    bot_id: str,
    limit: int,
) -> list[PerformanceLearningRecord]:
    if limit <= 0:
        return []
    primary = [record for record in records if matches_bot_scope(record, bot_id)]
    relevance_keys = performance_learning_bot_relevance_keys(primary, bot_id)
    portfolio = [
        record for record in records
        if record.record_type == PerformanceRecordType.PORTFOLIO
        and _portfolio_relevant_to_bot(record, bot_id, relevance_keys)
    ]
    primary.sort(key=lambda record: (_bot_scope_priority(record, bot_id), record.event_time), reverse=True)
    portfolio.sort(key=lambda record: record.event_time, reverse=True)
    portfolio_limit = min(len(portfolio), _portfolio_context_budget(limit, len(primary)))
    primary_limit = max(0, limit - portfolio_limit)
    selected = [*primary[:primary_limit], *portfolio[:portfolio_limit]]
    selected.sort(key=lambda record: (_bot_scope_priority(record, bot_id), record.event_time), reverse=True)
    return selected[:limit]


def performance_learning_bot_relevance_keys(
    records: list[PerformanceLearningRecord],
    bot_id: str,
) -> set[str]:
    keys: set[str] = set()
    _add_relation_key(keys, bot_id)
    for record in records:
        _add_record_relation_keys(keys, record)
    return keys


def performance_learning_record_relation_keys(
    record: PerformanceLearningRecord,
    *,
    include_portfolio_context: bool = False,
) -> set[str]:
    keys: set[str] = set()
    _add_record_relation_keys(keys, record)
    if include_portfolio_context:
        _add_portfolio_context_keys(keys, record)
    return keys


def _portfolio_context_budget(limit: int, primary_count: int) -> int:
    if limit <= 1:
        return 0
    if primary_count <= 0:
        return min(3, limit)
    return max(1, min(3, limit // 3, limit - 1))


def _bot_scope_priority(record: PerformanceLearningRecord, bot_id: str) -> int:
    if record.bot_id == bot_id:
        return 2
    if record.bot_id == "":
        return 1
    return 0


def _portfolio_relevant_to_bot(
    record: PerformanceLearningRecord,
    bot_id: str,
    relevance_keys: set[str],
) -> bool:
    if record.bot_id == bot_id:
        return True
    return bool(
        performance_learning_record_relation_keys(
            record,
            include_portfolio_context=True,
        )
        & relevance_keys
    )


def _add_record_relation_keys(keys: set[str], record: PerformanceLearningRecord) -> None:
    for value in (
        record.bot_id,
        record.strategy_id,
        record.portfolio_id,
        record.scope if record.record_type != PerformanceRecordType.PORTFOLIO else "",
        record.approval_request_id,
        record.deployment_id,
    ):
        _add_relation_key(keys, value)
    for values in (
        record.proposal_ids,
        record.strategy_change_record_ids,
        record.source_weekly_signal_ids,
        record.brief_attribution_ids,
    ):
        for value in values:
            _add_relation_key(keys, value)


def _add_portfolio_context_keys(keys: set[str], record: PerformanceLearningRecord) -> None:
    for value in record.portfolio_allocation_diff:
        _add_relation_key(keys, value)
    context = record.portfolio_context.model_dump()
    for field in (
        "allocation_weights",
        "risk_budgets",
        "exposure",
        "correlation",
        "drawdown_overlap",
        "marginal_contribution",
    ):
        values = context.get(field)
        if isinstance(values, dict):
            for key in values:
                _add_relation_key(keys, key)


def _add_relation_key(keys: set[str], value: object) -> None:
    text = str(value or "").strip()
    if not text:
        return
    keys.add(text.lower())
    for separator in (":", "|", ",", ";", ">", "<"):
        text = text.replace(separator, " ")
    keys.update(part.strip().lower() for part in text.split() if part.strip())
