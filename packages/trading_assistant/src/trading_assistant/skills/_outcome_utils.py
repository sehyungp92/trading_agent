# skills/_outcome_utils.py
"""Shared outcome evaluation utilities.

Centralises is_positive / is_conclusive logic so that learning_ledger,
suggestion_scorer, and convergence_tracker stay consistent.
"""
from __future__ import annotations

_POSITIVE_VERDICTS = frozenset({"positive", "keep"})
_NEGATIVE_VERDICTS = frozenset({"negative", "repair", "rollback", "quarantine"})
_INCONCLUSIVE_VERDICTS = frozenset({"inconclusive", "insufficient_data", "neutral", "watch"})
_AUTHORITATIVE_SOURCES = frozenset({"monthly", "follow_up"})


def outcome_source(outcome: dict) -> str:
    """Return normalized source, treating legacy source-less rows as early warning."""
    source = str(outcome.get("outcome_source") or outcome.get("source") or "").strip().lower()
    return source or "early_warning"


def is_authoritative_outcome(outcome: dict) -> bool:
    return outcome_source(outcome) in _AUTHORITATIVE_SOURCES


def is_positive_outcome(outcome: dict) -> bool:
    """Check if an outcome is positive, preferring verdict field."""
    verdict = str(outcome.get("verdict") or "").strip().lower()
    if verdict:
        if verdict in _POSITIVE_VERDICTS:
            return True
        if verdict in _NEGATIVE_VERDICTS or verdict in _INCONCLUSIVE_VERDICTS:
            return False
    delta = _preferred_delta(outcome)
    try:
        return float(delta) > 0
    except (TypeError, ValueError):
        return False


def is_conclusive_outcome(outcome: dict) -> bool:
    """Returns False for inconclusive/insufficient_data verdicts.

    These outcomes should be excluded from both numerator AND denominator
    when computing win rates, since the learning system added these
    verdicts specifically for ambiguous outcomes.
    """
    verdict = str(outcome.get("verdict") or "").strip().lower()
    return verdict not in _INCONCLUSIVE_VERDICTS


def _preferred_delta(outcome: dict):
    for key in ("pnl_delta", "composite_delta", "objective_delta", "live_vs_expected_objective_delta"):
        value = outcome.get(key)
        if value is not None:
            return value
    delta_30d = outcome.get("pnl_delta_30d")
    delta_7d = outcome.get("pnl_delta_7d")
    try:
        if float(delta_30d) != 0.0 or delta_7d is None:
            return delta_30d
    except (TypeError, ValueError):
        if delta_30d is not None:
            return delta_30d
    return delta_7d if delta_7d is not None else 0
