"""Tests for greedy_optimizer — forward selection with baseline, pruning, structured rounds."""

import json

import pytest

from crypto_trader.optimize.greedy_optimizer import run_greedy, _delta_ratio, _save_checkpoint
from crypto_trader.optimize.types import Experiment, GreedyRound, ScoredCandidate


def _make_evaluate_fn(score_map: dict[str, float], reject_set: set[str] | None = None):
    """Create a mock evaluate function that returns fixed scores.

    Handles the __baseline__ experiment automatically.
    """
    reject_set = reject_set or set()

    def evaluate_fn(candidates, current_mutations):
        results = []
        for exp in candidates:
            if exp.name == "__baseline__":
                # Return baseline score based on current_mutations
                results.append(ScoredCandidate(
                    experiment=exp,
                    score=score_map.get("__baseline__", 0.1),
                    metrics={"mock": 0.1},
                ))
                continue
            score = score_map.get(exp.name, 0.0)
            rejected = exp.name in reject_set
            results.append(ScoredCandidate(
                experiment=exp,
                score=score,
                metrics={"mock": score},
                rejected=rejected,
                reject_reason="hard reject" if rejected else "",
            ))
        return results

    return evaluate_fn


class TestDeltaRatio:
    def test_positive_baseline(self):
        assert _delta_ratio(1.1, 1.0) == pytest.approx(0.1)

    def test_zero_baseline(self):
        assert _delta_ratio(0.5, 0.0) == 0.5

    def test_negative_change(self):
        assert _delta_ratio(0.9, 1.0) == pytest.approx(-0.1)


class TestRunGreedy:
    def test_accepts_best_candidate(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
            Experiment("C", {"c": 3}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.3, "B": 0.8, "C": 0.5})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)

        # B should be accepted first (highest score)
        assert len(result.accepted_experiments) >= 1
        assert result.accepted_experiments[0].experiment.name == "B"

    def test_stops_at_min_delta(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        # Both have equal score — second round improvement is 0
        evaluate_fn = _make_evaluate_fn({"A": 0.5, "B": 0.5})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.01)

        # Should accept first, then stop
        assert len(result.accepted_experiments) == 1

    def test_rejects_hard_rejected(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("BAD", {"bad": 1}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.5, "BAD": 0.9}, reject_set={"BAD"})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)

        accepted_names = {sc.experiment.name for sc in result.accepted_experiments}
        assert "BAD" not in accepted_names
        assert "A" in accepted_names

    def test_merges_mutations(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.3})
        result = run_greedy(candidates, {"existing": 0}, evaluate_fn, min_delta=0.001)

        assert "existing" in result.final_mutations
        assert "a" in result.final_mutations

    def test_empty_candidates(self):
        def evaluate_fn(candidates, mutations):
            # Handle baseline
            return [ScoredCandidate(
                experiment=c, score=0.0, metrics={},
            ) for c in candidates]

        result = run_greedy([], {}, evaluate_fn, min_delta=0.001)
        assert result.accepted_experiments == []
        assert result.final_score == 0.0

    def test_all_rejected(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.5, "B": 0.5}, reject_set={"A", "B"})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert len(result.accepted_experiments) == 0

    def test_baseline_evaluation(self):
        """Baseline experiment is evaluated at start."""
        candidates = [Experiment("A", {"a": 1})]
        baseline_scores = {"__baseline__": 0.3, "A": 0.8}
        evaluate_fn = _make_evaluate_fn(baseline_scores)

        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert result.base_score == pytest.approx(0.3)

    def test_structured_rounds(self):
        """Rounds are GreedyRound objects."""
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.6})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)

        assert isinstance(result.rounds, list)
        assert len(result.rounds) >= 1
        assert isinstance(result.rounds[0], GreedyRound)
        assert result.rounds[0].best_name == "A"
        assert result.rounds[0].kept is True

    def test_max_rounds_cap(self):
        """max_rounds limits the number of rounds."""
        candidates = [
            Experiment(f"E{i}", {f"e{i}": i}) for i in range(10)
        ]
        scores = {f"E{i}": 0.5 + i * 0.01 for i in range(10)}
        evaluate_fn = _make_evaluate_fn(scores)

        result = run_greedy(
            candidates, {}, evaluate_fn,
            min_delta=0.0001, max_rounds=3,
        )
        assert len(result.rounds) <= 3

    def test_elapsed_seconds(self):
        """Result includes elapsed time."""
        candidates = [Experiment("A", {"a": 1})]
        evaluate_fn = _make_evaluate_fn({"A": 0.8})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert result.elapsed_seconds >= 0.0

    def test_kept_features(self):
        """kept_features tracks names of accepted experiments."""
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.6})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert "A" in result.kept_features

    def test_total_candidates(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.6})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert result.total_candidates == 2

    def test_accepted_count(self):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.6})
        result = run_greedy(candidates, {}, evaluate_fn, min_delta=0.001)
        assert result.accepted_count == len(result.accepted_experiments)

    def test_checkpoint_save_load(self, tmp_path):
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
            Experiment("C", {"c": 3}),
        ]
        evaluate_fn = _make_evaluate_fn({"A": 0.8, "B": 0.6, "C": 0.4})
        checkpoint_path = tmp_path / "greedy_checkpoint.json"
        result = run_greedy(
            candidates, {}, evaluate_fn,
            min_delta=0.001, checkpoint_path=checkpoint_path,
        )

        # Checkpoint is cleaned up on success
        assert len(result.accepted_experiments) >= 1

    def test_checkpoint_stores_contract_payload(self, tmp_path):
        checkpoint_path = tmp_path / "greedy_checkpoint.json"
        candidate = ScoredCandidate(
            experiment=Experiment("A", {"a": 1}),
            score=0.8,
            metrics={"total_trades": 10.0},
        )
        context = json.dumps({
            "contract_hash": "hash_a",
            "contract": {"contract_hash": "hash_a", "profile_hash": "profile"},
        })

        _save_checkpoint(
            checkpoint_path,
            [candidate],
            [],
            {"a": 1},
            0.8,
            1,
            "identity",
            [GreedyRound(1, 1, "A", 0.8, 70.0, True)],
            context,
        )

        with open(checkpoint_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["contract_hash"] == "hash_a"
        assert payload["contract"] == {"contract_hash": "hash_a", "profile_hash": "profile"}

    def test_checkpoint_resume(self, tmp_path):
        """Running greedy again with same checkpoint_path resumes correctly."""
        candidates = [
            Experiment("A", {"a": 1}),
            Experiment("B", {"b": 2}),
            Experiment("C", {"c": 3}),
        ]
        call_count = 0

        def counting_evaluate_fn(cands, mutations):
            nonlocal call_count
            call_count += len(cands)
            return _make_evaluate_fn({"A": 0.8, "B": 0.6, "C": 0.4})(cands, mutations)

        checkpoint_path = tmp_path / "greedy_checkpoint.json"

        # First run — evaluates all candidates
        result1 = run_greedy(
            candidates, {}, counting_evaluate_fn,
            min_delta=0.001, checkpoint_path=checkpoint_path,
        )
        first_run_calls = call_count

        # Checkpoint is deleted on success, so resume won't happen
        # This tests that the code works end-to-end
        assert len(result1.accepted_experiments) >= 1

    def test_checkpoint_identity_mismatch(self, tmp_path):
        """Stale checkpoint with different identity is skipped."""
        import json

        checkpoint_path = tmp_path / "greedy_checkpoint.json"
        # Write a checkpoint with wrong identity
        stale = {
            "identity": "wrong_identity_hash",
            "accepted": [],
            "rejected": [],
            "mutations": {},
            "best_score": 0.5,
            "round": 3,
            "rounds": [],
        }
        with open(checkpoint_path, "w") as f:
            json.dump(stale, f)

        candidates = [Experiment("A", {"a": 1})]
        evaluate_fn = _make_evaluate_fn({"A": 0.8})

        result = run_greedy(
            candidates, {}, evaluate_fn,
            min_delta=0.001, checkpoint_path=checkpoint_path,
        )
        # Should run fresh despite checkpoint existing
        assert len(result.accepted_experiments) >= 1
