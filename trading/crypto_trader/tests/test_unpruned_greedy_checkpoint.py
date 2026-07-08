"""Tests for the no-pruning greedy helper used by phased runners."""

from __future__ import annotations

import pytest

from crypto_trader.optimize.breakout_round4_trade_frequency import run_greedy_without_pruning
from crypto_trader.optimize.types import Experiment, ScoredCandidate


def _score(name: str, value: float) -> ScoredCandidate:
    return ScoredCandidate(
        experiment=Experiment(name=name, mutations={name.lower(): True}),
        score=value,
        metrics={"total_trades": 20.0},
    )


def test_unpruned_greedy_resumes_from_checkpoint(tmp_path):
    candidates = [
        Experiment("A", {"a": True}),
        Experiment("B", {"b": True}),
    ]
    checkpoint = tmp_path / "phase_4_greedy_checkpoint.json"
    calls = {"count": 0}

    def interrupted_evaluate(batch, current_mutations):
        del current_mutations
        calls["count"] += 1
        if calls["count"] == 1:
            return [_score("__baseline__", 0.0)]
        if calls["count"] == 2:
            return [_score("A", 1.0), _score("B", 0.2)]
        raise RuntimeError("interrupted after checkpoint")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_greedy_without_pruning(
            candidates,
            {},
            interrupted_evaluate,
            min_delta=0.001,
            max_rounds=3,
            checkpoint_path=checkpoint,
            checkpoint_context="same-context",
        )

    assert checkpoint.exists()

    seen_batches: list[list[str]] = []

    def resumed_evaluate(batch, current_mutations):
        seen_batches.append([candidate.name for candidate in batch])
        if batch[0].name == "__baseline__":
            assert current_mutations == {"a": True}
            return [_score("__baseline__", 1.0)]
        return [_score("B", 1.2)]

    result = run_greedy_without_pruning(
        candidates,
        {},
        resumed_evaluate,
        min_delta=0.001,
        max_rounds=3,
        checkpoint_path=checkpoint,
        checkpoint_context="same-context",
    )

    assert result.kept_features == ["A", "B"]
    assert seen_batches == [["__baseline__"], ["B"]]
    assert not checkpoint.exists()
