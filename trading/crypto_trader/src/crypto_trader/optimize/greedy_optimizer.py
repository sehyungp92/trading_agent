"""Greedy forward-selection optimizer with checkpointing, pruning, and structured rounds."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import structlog

from crypto_trader.optimize.config_mutator import merge_mutations
from crypto_trader.optimize.types import (
    EvaluateFn,
    Experiment,
    GreedyResult,
    GreedyRound,
    ScoredCandidate,
)

log = structlog.get_logger("optimize.greedy")


def _delta_ratio(score: float, baseline: float) -> float:
    """Compute relative score change from baseline."""
    if baseline > 0:
        return (score - baseline) / baseline
    return score - baseline


def _compute_identity(
    base_mutations: dict[str, Any],
    candidate_names: list[str],
    context: str | None,
) -> str:
    """Compute MD5 identity hash for checkpoint validation."""
    payload = json.dumps(
        {"base": base_mutations, "candidates": sorted(candidate_names),
         "context": context or ""},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _contract_hash_from_context(context: str | None) -> str:
    payload = _payload_from_context(context)
    return str(payload.get("contract_hash") or "")


def _contract_from_context(context: str | None) -> dict[str, Any]:
    payload = _payload_from_context(context)
    contract = payload.get("contract")
    return contract if isinstance(contract, dict) else {}


def _payload_from_context(context: str | None) -> dict[str, Any]:
    if not context:
        return {}
    try:
        payload = json.loads(context)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_greedy(
    candidates: list[Experiment],
    current_mutations: dict[str, Any],
    evaluate_fn: EvaluateFn,
    *,
    min_delta: float = 0.005,
    max_rounds: int = 20,
    prune_threshold: float = 0.05,
    checkpoint_path: Path | None = None,
    checkpoint_context: str | None = None,
    logger: Any = None,
) -> GreedyResult:
    """Run greedy forward selection over candidates.

    Each round:
    1. Evaluate all remaining candidates with current mutations
    2. Pick the best-scoring candidate
    3. If improvement >= min_delta (relative), accept it and merge its mutations
    4. Otherwise stop

    Features:
    - Baseline evaluation at start
    - Structured GreedyRound tracking
    - Delta pruning: remove candidates far below best
    - Streak pruning: remove candidates rejected 2+ consecutive rounds
    - Checkpoint identity validation
    """
    start_time = time.time()
    remaining = list(candidates)
    accepted: list[ScoredCandidate] = []
    rejected: list[ScoredCandidate] = []
    active_mutations = dict(current_mutations)
    rounds: list[GreedyRound] = []
    round_num = 0
    total_candidates = len(candidates)
    rejection_streaks: dict[str, int] = {}

    # Compute checkpoint identity
    identity = _compute_identity(
        current_mutations, [c.name for c in candidates], checkpoint_context
    )

    # Try to resume from checkpoint
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path, identity)
        if checkpoint:
            accepted = checkpoint["accepted"]
            rejected = checkpoint["rejected"]
            active_mutations = checkpoint["mutations"]
            round_num = checkpoint["round"]
            rounds = checkpoint.get("rounds", [])
            accepted_names = {sc.experiment.name for sc in accepted}
            rejected_names = {sc.experiment.name for sc in rejected}
            remaining = [
                c for c in remaining
                if c.name not in accepted_names and c.name not in rejected_names
            ]
            log.info(
                "greedy.resumed",
                round=round_num,
                remaining=len(remaining),
                accepted=len(accepted),
            )

    # Baseline evaluation
    baseline_candidates = [Experiment("__baseline__", {})]
    baseline_results = evaluate_fn(baseline_candidates, active_mutations)
    base_score = 0.0
    if baseline_results and not baseline_results[0].rejected:
        base_score = baseline_results[0].score
    best_score = base_score if not accepted else max(
        base_score, *(sc.score for sc in accepted)
    )

    try:
        while remaining and round_num < max_rounds:
            round_num += 1
            log.info("greedy.round_start", round=round_num, remaining=len(remaining))

            scored = evaluate_fn(remaining, active_mutations)

            # Separate rejects from viable
            viable = [sc for sc in scored if not sc.rejected]
            round_rejected = [sc for sc in scored if sc.rejected]
            rejected_count = len(round_rejected)
            rejected.extend(round_rejected)

            # Remove hard-rejected from remaining
            rejected_names = {sc.experiment.name for sc in round_rejected}
            remaining = [c for c in remaining if c.name not in rejected_names]

            if not viable:
                log.info("greedy.no_viable_candidates", round=round_num)
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(scored),
                    best_name="(none)",
                    best_score=0.0,
                    best_delta_pct=0.0,
                    kept=False,
                    rejected_count=rejected_count,
                ))
                break

            # Pick best
            viable.sort(key=lambda sc: sc.score, reverse=True)
            best = viable[0]

            delta_pct = _delta_ratio(best.score, best_score) * 100.0

            log.info(
                "greedy.round_result",
                round=round_num,
                best=best.experiment.name,
                score=best.score,
                delta_pct=f"{delta_pct:.2f}%",
            )

            kept = delta_pct >= (min_delta * 100.0)

            if not kept:
                log.info("greedy.converged", round=round_num, delta_pct=delta_pct)
                rounds.append(GreedyRound(
                    round_num=round_num,
                    candidates_tested=len(scored),
                    best_name=best.experiment.name,
                    best_score=best.score,
                    best_delta_pct=delta_pct,
                    kept=False,
                    rejected_count=rejected_count,
                ))
                break

            # Accept best
            accepted.append(best)
            active_mutations = merge_mutations(
                active_mutations, best.experiment.mutations
            )
            best_score = best.score

            # Record round
            rounds.append(GreedyRound(
                round_num=round_num,
                candidates_tested=len(scored),
                best_name=best.experiment.name,
                best_score=best.score,
                best_delta_pct=delta_pct,
                kept=True,
                rejected_count=rejected_count,
            ))

            # Remove accepted from remaining
            remaining = [c for c in remaining if c.name != best.experiment.name]

            # Update rejection streaks for remaining viable (non-best)
            for sc in viable[1:]:
                rejection_streaks[sc.experiment.name] = (
                    rejection_streaks.get(sc.experiment.name, 0) + 1
                )

            # Delta pruning: remove candidates far below best
            if prune_threshold > 0 and len(viable) > 1:
                prune_cutoff = best.score * (1.0 - prune_threshold)
                pruned = {
                    sc.experiment.name for sc in viable[1:]
                    if sc.score < prune_cutoff
                }
                if pruned:
                    remaining = [c for c in remaining if c.name not in pruned]
                    log.info("greedy.delta_pruned", count=len(pruned))

            # Streak pruning: remove candidates rejected 2+ consecutive rounds
            streak_pruned = {
                name for name, streak in rejection_streaks.items()
                if streak >= 2
            }
            if streak_pruned:
                remaining = [c for c in remaining if c.name not in streak_pruned]
                log.info("greedy.streak_pruned", count=len(streak_pruned))

            # Checkpoint
            if checkpoint_path:
                _save_checkpoint(
                    checkpoint_path, accepted, rejected, active_mutations,
                    best_score, round_num, identity, rounds, checkpoint_context,
                )
    finally:
        # Cleanup evaluate_fn if it has a close method
        close_fn = getattr(evaluate_fn, "close", None)
        if callable(close_fn):
            close_fn()

    elapsed = time.time() - start_time

    # Delete checkpoint on successful completion
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
        except OSError:
            pass

    return GreedyResult(
        accepted_experiments=accepted,
        rejected_experiments=rejected,
        final_mutations=active_mutations,
        final_score=best_score,
        rounds=rounds,
        base_score=base_score,
        kept_features=[sc.experiment.name for sc in accepted],
        total_candidates=total_candidates,
        accepted_count=len(accepted),
        elapsed_seconds=elapsed,
    )


def _save_checkpoint(
    checkpoint_path: Path,
    accepted: list[ScoredCandidate],
    rejected: list[ScoredCandidate],
    mutations: dict[str, Any],
    best_score: float,
    round_num: int,
    identity: str,
    rounds: list[GreedyRound],
    checkpoint_context: str | None = None,
) -> None:
    """Save greedy progress to checkpoint file."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "identity": identity,
        "checkpoint_context": checkpoint_context or "",
        "contract_hash": _contract_hash_from_context(checkpoint_context),
        "contract": _contract_from_context(checkpoint_context),
        "accepted": [
            {
                "name": sc.experiment.name,
                "mutations": sc.experiment.mutations,
                "score": sc.score,
                "metrics": sc.metrics,
            }
            for sc in accepted
        ],
        "rejected": [
            {
                "name": sc.experiment.name,
                "mutations": sc.experiment.mutations,
                "score": sc.score,
                "metrics": sc.metrics,
                "reject_reason": sc.reject_reason,
            }
            for sc in rejected
        ],
        "mutations": mutations,
        "best_score": best_score,
        "round": round_num,
        "rounds": [
            {
                "round_num": r.round_num,
                "candidates_tested": r.candidates_tested,
                "best_name": r.best_name,
                "best_score": r.best_score,
                "best_delta_pct": r.best_delta_pct,
                "kept": r.kept,
                "rejected_count": r.rejected_count,
            }
            for r in rounds
        ],
    }
    # Atomic write: write to temp, then os.replace
    tmp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(str(tmp_path), str(checkpoint_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_checkpoint(
    checkpoint_path: Path,
    expected_identity: str,
) -> dict | None:
    """Load greedy checkpoint if it exists and identity matches."""
    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, encoding="utf-8") as f:
            data = json.load(f)

        # Validate identity
        stored_identity = data.get("identity", "")
        if stored_identity and stored_identity != expected_identity:
            log.warning(
                "greedy.checkpoint_stale",
                path=str(checkpoint_path),
                expected=expected_identity[:8],
                found=stored_identity[:8],
            )
            return None

        accepted = [
            ScoredCandidate(
                experiment=Experiment(name=item["name"], mutations=item["mutations"]),
                score=item["score"],
                metrics=item.get("metrics", {}),
            )
            for item in data.get("accepted", [])
        ]
        rejected = [
            ScoredCandidate(
                experiment=Experiment(name=item["name"], mutations=item["mutations"]),
                score=item["score"],
                metrics=item.get("metrics", {}),
                rejected=True,
                reject_reason=item.get("reject_reason", ""),
            )
            for item in data.get("rejected", [])
        ]

        # Restore rounds
        rounds = [
            GreedyRound(
                round_num=r["round_num"],
                candidates_tested=r["candidates_tested"],
                best_name=r["best_name"],
                best_score=r["best_score"],
                best_delta_pct=r["best_delta_pct"],
                kept=r["kept"],
                rejected_count=r.get("rejected_count", 0),
            )
            for r in data.get("rounds", [])
        ]

        return {
            "accepted": accepted,
            "rejected": rejected,
            "mutations": data["mutations"],
            "best_score": data["best_score"],
            "round": data["round"],
            "rounds": rounds,
        }
    except (json.JSONDecodeError, KeyError):
        log.warning("greedy.checkpoint_corrupt", path=str(checkpoint_path))
        return None
