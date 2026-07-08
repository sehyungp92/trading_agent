from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .phase_state import _atomic_write_json
from .types import Experiment, GreedyResult, GreedyRound, ScoredCandidate

EvaluateFn = Callable[[list[Experiment], dict[str, object]], list[ScoredCandidate]]
ProgressFn = Callable[[dict[str, object]], None]


def _delta_ratio(score: float, baseline: float) -> float:
    """Relative score change when possible, absolute fallback near zero."""
    if baseline > 0:
        return (score - baseline) / baseline
    return score - baseline


def _delta_pct_display(score: float, baseline: float) -> float:
    return _delta_ratio(score, baseline) * 100.0 if baseline > 0 else _delta_ratio(score, baseline)


def _checkpoint_identity(
    base_mutations: dict[str, object],
    candidate_names: list[str],
    checkpoint_context: dict[str, object] | None,
) -> str:
    payload = {
        "base_mutations": base_mutations,
        "candidate_names": sorted(candidate_names),
        "checkpoint_context": checkpoint_context or {},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def save_greedy_checkpoint(
    current_mutations: dict[str, object],
    kept_features: list[str],
    current_score: float,
    remaining_names: list[str],
    rounds: list[GreedyRound],
    checkpoint_identity: str,
    path: Path,
) -> None:
    payload = {
        "current_mutations": current_mutations,
        "kept_features": kept_features,
        "current_score": current_score,
        "remaining_names": remaining_names,
        "rounds": [asdict(round_result) for round_result in rounds],
        "checkpoint_identity": checkpoint_identity,
    }
    _atomic_write_json(payload, path)


def load_greedy_checkpoint(path: Path) -> tuple[dict[str, object], list[str], float, set[str], list[GreedyRound], str] | None:
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    return (
        data.get("current_mutations", {}),
        data.get("kept_features", []),
        float(data.get("current_score", 0.0)),
        set(data.get("remaining_names", [])),
        [GreedyRound(**round_data) for round_data in data.get("rounds", [])],
        data.get("checkpoint_identity") or data.get("base_mutations_hash", ""),
    )


def save_greedy_result(result: GreedyResult, path: Path) -> None:
    _atomic_write_json(asdict(result), path)


def run_greedy(
    candidates: list[Experiment],
    base_mutations: dict[str, object],
    evaluate_batch: EvaluateFn,
    *,
    max_rounds: int = 20,
    min_delta: float = 0.001,
    prune_threshold: float = 0.05,
    reject_streak_limit: int = 2,
    checkpoint_path: Path | None = None,
    checkpoint_context: dict[str, object] | None = None,
    logger: logging.Logger | None = None,
    progress_callback: ProgressFn | None = None,
) -> GreedyResult:
    start = time.time()
    log = logger or logging.getLogger(__name__)
    total_candidates = len(candidates)
    current_mutations = dict(base_mutations)
    kept_features: list[str] = []
    rounds: list[GreedyRound] = []
    remaining = list(candidates)
    reject_streak: dict[str, int] = {}
    checkpoint_identity = _checkpoint_identity(
        base_mutations,
        [candidate.name for candidate in candidates],
        checkpoint_context,
    )

    baseline = evaluate_batch([Experiment("__baseline__", {})], current_mutations)
    base_score = baseline[0].score if baseline else 0.0
    if baseline and baseline[0].rejected:
        log.warning(
            "Baseline rejected by hard rejects: %s. Candidates must still "
            "beat this score AND pass hard rejects to be accepted.",
            baseline[0].reject_reason,
        )
    current_score = base_score
    _emit_progress(
        progress_callback,
        {
            "event": "baseline_complete",
            "base_score": base_score,
            "current_score": current_score,
            "candidate_count": total_candidates,
            "accepted_count": len(kept_features),
        },
    )

    if checkpoint_path:
        resumed = load_greedy_checkpoint(checkpoint_path)
        if resumed and resumed[-1] == checkpoint_identity:
            current_mutations, kept_features, current_score, remaining_names, rounds, _ = resumed
            remaining = [candidate for candidate in candidates if candidate.name in remaining_names]
            log.info(
                "Resumed greedy checkpoint: kept=%d score=%.4f remaining=%d",
                len(kept_features),
                current_score,
                len(remaining),
            )
            _emit_progress(
                progress_callback,
                {
                    "event": "checkpoint_resumed",
                    "current_score": current_score,
                    "accepted_count": len(kept_features),
                    "remaining_candidates": len(remaining),
                    "kept_features": list(kept_features),
                },
            )
        elif resumed:
            log.info("Checkpoint identity mismatch; starting greedy search fresh.")

    try:
        for round_num in range(len(rounds) + 1, max_rounds + 1):
            if not remaining:
                break

            _emit_progress(
                progress_callback,
                {
                    "event": "round_start",
                    "round_num": round_num,
                    "candidate_count": len(remaining),
                    "current_score": current_score,
                    "accepted_count": len(kept_features),
                    "kept_features": list(kept_features),
                },
            )
            scored = evaluate_batch(remaining, current_mutations)
            rejected_count = sum(1 for item in scored if item.rejected)
            valid = [item for item in scored if not item.rejected]

            # Track consecutive hard-rejection streaks for pruning.
            _rejected_names = {item.name for item in scored if item.rejected}
            for candidate in remaining:
                if candidate.name in _rejected_names:
                    reject_streak[candidate.name] = reject_streak.get(candidate.name, 0) + 1
                else:
                    reject_streak[candidate.name] = 0

            if not valid:
                rounds.append(
                    GreedyRound(
                        round_num=round_num,
                        candidates_tested=len(remaining),
                        best_name="",
                        best_score=current_score,
                        best_delta_pct=0.0,
                        kept=False,
                        rejected_count=rejected_count,
                    )
                )
                log.info("Round %d produced no valid candidates; stopping.", round_num)
                _emit_progress(
                    progress_callback,
                    {
                        "event": "round_complete",
                        "round_num": round_num,
                        "candidate_count": len(remaining),
                        "rejected_count": rejected_count,
                        "valid_count": 0,
                        "current_score": current_score,
                        "accepted_count": len(kept_features),
                        "kept_features": list(kept_features),
                        "kept": False,
                        "stop_reason": "no_valid_candidates",
                    },
                )
                break

            best = max(valid, key=lambda item: item.score)
            best_delta_ratio = _delta_ratio(best.score, current_score)
            delta_pct = _delta_pct_display(best.score, current_score)
            kept = best.score > current_score and best_delta_ratio >= min_delta
            rounds.append(
                GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(remaining),
                    best_name=best.name,
                    best_score=best.score,
                    best_delta_pct=delta_pct,
                    kept=kept,
                    rejected_count=rejected_count,
                )
            )

            if not kept:
                log.info(
                    "Round %d best candidate %s did not clear min_delta %.2f%%; stopping.",
                    round_num,
                    best.name,
                    min_delta * 100.0,
                )
                _emit_progress(
                    progress_callback,
                    {
                        "event": "round_complete",
                        "round_num": round_num,
                        "candidate_count": len(remaining),
                        "rejected_count": rejected_count,
                        "valid_count": len(valid),
                        "best_name": best.name,
                        "best_score": best.score,
                        "best_delta_pct": delta_pct,
                        "current_score": current_score,
                        "accepted_count": len(kept_features),
                        "kept_features": list(kept_features),
                        "kept": False,
                        "stop_reason": "min_delta_not_cleared",
                    },
                )
                break

            round_delta_by_name = {
                item.name: _delta_ratio(item.score, current_score)
                for item in valid
            }
            chosen = next(candidate for candidate in remaining if candidate.name == best.name)
            current_mutations.update(chosen.mutations)
            current_score = best.score
            kept_features.append(best.name)

            next_remaining: list[Experiment] = []
            streak_pruned = 0
            for candidate in remaining:
                if candidate.name == best.name:
                    continue
                if prune_threshold > 0:
                    candidate_delta = round_delta_by_name.get(candidate.name)
                    if candidate_delta is not None and candidate_delta < -prune_threshold:
                        continue
                if reject_streak.get(candidate.name, 0) >= reject_streak_limit:
                    streak_pruned += 1
                    continue
                next_remaining.append(candidate)
            remaining = next_remaining
            if streak_pruned:
                log.info("  Pruned %d persistently-rejected candidates (%d+ consecutive).", streak_pruned, reject_streak_limit)
            _emit_progress(
                progress_callback,
                {
                    "event": "round_complete",
                    "round_num": round_num,
                    "candidate_count": rounds[-1].candidates_tested,
                    "rejected_count": rejected_count,
                    "valid_count": len(valid),
                    "best_name": best.name,
                    "best_score": best.score,
                    "best_delta_pct": delta_pct,
                    "current_score": current_score,
                    "accepted_count": len(kept_features),
                    "kept_features": list(kept_features),
                    "remaining_candidates": len(remaining),
                    "streak_pruned": streak_pruned,
                    "kept": True,
                },
            )

            if checkpoint_path:
                save_greedy_checkpoint(
                    current_mutations=current_mutations,
                    kept_features=kept_features,
                    current_score=current_score,
                    remaining_names=[candidate.name for candidate in remaining],
                    rounds=rounds,
                    checkpoint_identity=checkpoint_identity,
                    path=checkpoint_path,
                )
    finally:
        close_fn = getattr(evaluate_batch, "close", None)
        if callable(close_fn):
            close_fn()

    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()

    # final_metrics={} intentionally empty -- populated by PhaseRunner after
    # calling plugin.compute_final_metrics() on the winning mutation set.
    return GreedyResult(
        base_score=base_score,
        final_score=current_score,
        final_mutations=current_mutations,
        kept_features=kept_features,
        rounds=rounds,
        final_metrics={},
        total_candidates=total_candidates,
        accepted_count=len(kept_features),
        elapsed_seconds=time.time() - start,
    )


def _emit_progress(callback: ProgressFn | None, payload: dict[str, object]) -> None:
    if callable(callback):
        callback(payload)
