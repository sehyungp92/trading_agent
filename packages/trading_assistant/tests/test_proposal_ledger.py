"""Tests for ProposalLedger — JSONL-backed unified proposal record store."""
from __future__ import annotations

from pathlib import Path

from trading_assistant.schemas.proposal_ledger import (
    ProposalCandidate,
    ProposalEvaluation,
    ProposalKind,
    ProposalOutcome,
    ProposalSource,
)
from trading_assistant.schemas.objective_weights import OBJECTIVE_WEIGHTS_VERSION
from trading_assistant.skills.proposal_ledger import ProposalLedger, make_proposal_id


def _candidate(tmp_path: Path, **overrides) -> ProposalCandidate:
    base = {
        "proposal_id": "test_id_0001",
        "source": ProposalSource.DETERMINISTIC,
        "kind": ProposalKind.PARAMETER_CHANGE,
        "bot_id": "bot_a",
        "title": "lower stop_loss to 1.5R",
    }
    base.update(overrides)
    return ProposalCandidate(**base)


def test_record_candidate_writes_and_dedupes(tmp_path: Path) -> None:
    ledger = ProposalLedger(tmp_path)
    cand = _candidate(tmp_path)

    assert ledger.record_candidate(cand) is True
    assert ledger.record_candidate(cand) is False  # dedup

    records = ledger.list_all()
    assert len(records) == 1
    assert records[0].candidate.proposal_id == "test_id_0001"


def test_evaluation_and_outcome_chain(tmp_path: Path) -> None:
    ledger = ProposalLedger(tmp_path)
    cand = _candidate(tmp_path)
    ledger.record_candidate(cand)

    ok = ledger.record_evaluation(
        cand.proposal_id,
        ProposalEvaluation(
            proposal_id=cand.proposal_id,
            method="parameter_search",
            decision="approve",
            objective_score=0.7,
            confidence=0.8,
        ),
    )
    assert ok is True

    ok = ledger.record_outcome(
        cand.proposal_id,
        ProposalOutcome(
            proposal_id=cand.proposal_id,
            objective_delta=0.12,
            verdict="positive",
        ),
    )
    assert ok is True

    rec = ledger.get_by_id(cand.proposal_id)
    assert rec is not None
    assert len(rec.evaluations) == 1
    assert rec.evaluations[0].decision == "approve"
    assert len(rec.outcomes) == 1
    assert rec.outcomes[0].verdict == "positive"


def test_evaluation_and_outcome_reject_unknown_id(tmp_path: Path) -> None:
    ledger = ProposalLedger(tmp_path)
    assert ledger.record_evaluation("missing", ProposalEvaluation(
        proposal_id="missing", method="x", decision="approve",
    )) is False
    assert ledger.record_outcome("missing", ProposalOutcome(
        proposal_id="missing", verdict="positive",
    )) is False


def test_list_by_bot_filters(tmp_path: Path) -> None:
    ledger = ProposalLedger(tmp_path)
    ledger.record_candidate(_candidate(tmp_path,
        proposal_id="a1", bot_id="bot_a", lifecycle_stage="exit",
    ))
    ledger.record_candidate(_candidate(tmp_path,
        proposal_id="a2", bot_id="bot_a", lifecycle_stage="entry",
    ))
    ledger.record_candidate(_candidate(tmp_path,
        proposal_id="b1", bot_id="bot_b", lifecycle_stage="exit",
    ))

    assert {r.candidate.proposal_id for r in ledger.list_by_bot("bot_a")} == {"a1", "a2"}
    assert {r.candidate.proposal_id for r in ledger.list_by_bot("bot_a", lifecycle_stage="exit")} == {"a1"}
    assert ledger.list_by_bot("nope") == []


def test_list_open_excludes_terminal_outcomes(tmp_path: Path) -> None:
    ledger = ProposalLedger(tmp_path)
    ledger.record_candidate(_candidate(tmp_path, proposal_id="open_one", title="open"))
    ledger.record_candidate(_candidate(tmp_path, proposal_id="closed_one", title="closed"))
    ledger.record_outcome("closed_one", ProposalOutcome(
        proposal_id="closed_one", verdict="improved",
    ))

    open_ids = {r.candidate.proposal_id for r in ledger.list_open()}
    assert open_ids == {"open_one"}


def test_make_proposal_id_is_deterministic() -> None:
    a = make_proposal_id(ProposalSource.PARAMETER_SEARCH, "bot_x", ProposalKind.PARAMETER_CHANGE, "Tighten stop")
    b = make_proposal_id(ProposalSource.PARAMETER_SEARCH, "bot_x", ProposalKind.PARAMETER_CHANGE, "tighten stop  ")
    assert a == b  # case + whitespace insensitive
    assert len(a) == 16


def test_make_proposal_id_includes_strategy_and_stable_link() -> None:
    base = make_proposal_id(
        ProposalSource.LLM_DAILY,
        "bot_x",
        ProposalKind.PARAMETER_CHANGE,
        "Tighten stop",
        strategy_id="alpha",
        link_key="suggestion-1",
    )
    different_strategy = make_proposal_id(
        ProposalSource.LLM_DAILY,
        "bot_x",
        ProposalKind.PARAMETER_CHANGE,
        "Tighten stop",
        strategy_id="beta",
        link_key="suggestion-1",
    )
    different_link = make_proposal_id(
        ProposalSource.LLM_DAILY,
        "bot_x",
        ProposalKind.PARAMETER_CHANGE,
        "Tighten stop",
        strategy_id="alpha",
        link_key="suggestion-2",
    )

    assert base != different_strategy
    assert base != different_link


def test_objective_version_defaults_on_evaluation_and_outcome() -> None:
    assert ProposalEvaluation(
        proposal_id="p1",
        method="parameter_search",
        decision="approve",
    ).objective_version == OBJECTIVE_WEIGHTS_VERSION
    assert ProposalOutcome(
        proposal_id="p1",
        verdict="positive",
    ).objective_version == OBJECTIVE_WEIGHTS_VERSION


def test_malformed_lines_are_tolerated(tmp_path: Path) -> None:
    ledger_path = tmp_path / "proposal_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("not json\n", encoding="utf-8")

    ledger = ProposalLedger(tmp_path)
    cand = _candidate(tmp_path, proposal_id="recovered")
    assert ledger.record_candidate(cand) is True
    rec = ledger.get_by_id("recovered")
    assert rec is not None
