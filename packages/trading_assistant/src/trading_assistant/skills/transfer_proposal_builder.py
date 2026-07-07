# skills/transfer_proposal_builder.py
"""TransferProposalBuilder — identifies cross-bot pattern transfer candidates.

For each VALIDATED/IMPLEMENTED pattern, finds eligible target bots and scores compatibility.
Also measures outcomes of completed transfers and feeds back into scoring.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

from trading_assistant.schemas.pattern_library import PatternEntry, PatternStatus
from trading_assistant.schemas.transfer_proposals import TransferOutcome, TransferProposal

logger = logging.getLogger(__name__)


class TransferProposalBuilder:
    def __init__(
        self,
        pattern_library,  # PatternLibrary instance
        curated_dir: Path,
        bots: list[str],
        findings_dir: Path | None = None,
        strategy_registry=None,
    ) -> None:
        self._library = pattern_library
        self._curated_dir = curated_dir
        self._bots = bots
        self._findings_dir = findings_dir
        self._outcomes_path = (findings_dir / "transfer_outcomes.jsonl") if findings_dir else None
        self._strategy_registry = strategy_registry

    def build_proposals(self) -> list[TransferProposal]:
        """Build transfer proposals for active patterns (PROPOSED, VALIDATED, IMPLEMENTED)."""
        active = self._library.load_active()
        transferable = active  # load_active() already excludes REJECTED

        # Load track record for scoring adjustment
        track_record = self.compute_transfer_track_record()

        proposals: list[TransferProposal] = []
        for pattern in transferable:
            eligible = self._find_eligible_bots(pattern)
            for bot_id in eligible:
                score = self._compute_compatibility(pattern, bot_id)

                # Adjust score based on track record
                pattern_track = track_record.get(pattern.pattern_id, {})
                if pattern_track:
                    success_rate = pattern_track.get("success_rate", 0.5)
                    if success_rate > 0.5:
                        score = min(1.0, score + 0.1)
                    elif success_rate < 0.3 and pattern_track.get("total", 0) >= 2:
                        score = max(0.0, score - 0.2)

                proposals.append(TransferProposal(
                    pattern_id=pattern.pattern_id,
                    source_bot=pattern.source_bot,
                    target_bot=bot_id,
                    pattern_title=pattern.title,
                    category=pattern.category.value if hasattr(pattern.category, "value") else str(pattern.category),
                    compatibility_score=round(score, 2),
                    rationale=f"Pattern '{pattern.title}' validated on {pattern.source_bot}, "
                              f"compatibility score {score:.2f} for {bot_id}.",
                ))

        # Sort by score descending
        proposals.sort(key=lambda p: p.compatibility_score, reverse=True)
        return proposals

    def measure_transfer_outcomes(self, curated_dir: Path | None = None) -> list[TransferOutcome]:
        """Measure outcomes of transferred patterns by comparing before/after metrics.

        For each TRANSFERRED pattern (target_bots populated), compares the target
        bot's 7-day metrics before and after the transfer date.
        """
        curated = curated_dir or self._curated_dir
        active = self._library.load_active()
        transferred = [
            p for p in active
            if p.target_bots and p.status in (PatternStatus.VALIDATED, PatternStatus.IMPLEMENTED)
        ]

        outcomes: list[TransferOutcome] = []
        existing = self._load_outcomes()
        measured_pairs = {(o.get("pattern_id"), o.get("target_bot")) for o in existing}

        for pattern in transferred:
            transfer_date = pattern.validated_at or (
                pattern.updated_at.strftime("%Y-%m-%d")
                if hasattr(pattern.updated_at, "strftime") else ""
            )
            for target_bot in pattern.target_bots:
                if (pattern.pattern_id, target_bot) in measured_pairs:
                    continue

                before = self._load_bot_metrics(target_bot, transfer_date, curated, before=True)
                after = self._load_bot_metrics(target_bot, transfer_date, curated, before=False)

                if before is None or after is None:
                    continue

                before_trades = before.get("trade_count", 0)
                after_trades = after.get("trade_count", 0)

                # Quality gate: require minimum trade counts
                if before_trades < 5 or after_trades < 5:
                    continue

                pnl_delta = after.get("pnl", 0) - before.get("pnl", 0)
                wr_delta = after.get("win_rate", 0) - before.get("win_rate", 0)

                # Regime matching
                before_regime = before.get("dominant_regime", "")
                after_regime = after.get("dominant_regime", "")
                regime_matched = (
                    before_regime == after_regime
                    or not before_regime
                    or not after_regime
                )

                if not regime_matched:
                    verdict = "inconclusive"
                elif pnl_delta > 0 and wr_delta >= 0:
                    verdict = "positive"
                elif pnl_delta < 0:
                    verdict = "negative"
                else:
                    verdict = "neutral"

                quality = "high" if regime_matched and before_trades >= 10 and after_trades >= 10 else "medium"

                outcome = TransferOutcome(
                    pattern_id=pattern.pattern_id,
                    source_bot=pattern.source_bot,
                    target_bot=target_bot,
                    transferred_at=transfer_date,
                    pnl_delta_7d=round(pnl_delta, 4),
                    win_rate_delta_7d=round(wr_delta, 4),
                    verdict=verdict,
                    regime_matched=regime_matched,
                    measurement_quality=quality,
                    before_trade_count=before_trades,
                    after_trade_count=after_trades,
                )
                outcomes.append(outcome)

        # Persist
        if outcomes and self._outcomes_path:
            self._outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._outcomes_path, "a", encoding="utf-8") as f:
                for o in outcomes:
                    f.write(json.dumps(o.model_dump(mode="json"), default=str) + "\n")

        return outcomes

    def compute_transfer_track_record(self) -> dict[str, dict]:
        """Compute per-pattern success rates across all measured transfers.

        Excludes "inconclusive" verdicts from success rate calculation.
        """
        return self._build_track_record(self._load_outcomes())

    def create_from_reasoning(
        self, reasoning: dict, source_bot: str,
    ) -> list[str]:
        """Create transfer proposals from a transferable outcome reasoning.

        Args:
            reasoning: Outcome reasoning dict with transferable=True.
            source_bot: The bot where the original suggestion was applied.

        Returns:
            List of target bot IDs that received proposals.
        """
        mechanism = reasoning.get("mechanism", "")
        suggestion_id = reasoning.get("suggestion_id", "")
        title = reasoning.get("title", mechanism[:80] if mechanism else "Transferable pattern")

        import hashlib as _hashlib
        pattern_id = _hashlib.sha256(
            f"reasoning:{suggestion_id}:{source_bot}".encode()
        ).hexdigest()[:12]

        targets: list[str] = []
        for bot_id in self._bots:
            if bot_id == source_bot:
                continue
            try:
                proposal = TransferProposal(
                    pattern_id=pattern_id,
                    source_bot=source_bot,
                    target_bot=bot_id,
                    pattern_title=title,
                    category=reasoning.get("category", "structural"),
                    compatibility_score=0.5,  # default — no pattern entry to score against
                    rationale=(
                        f"Outcome reasoning for suggestion {suggestion_id} identified "
                        f"transferable mechanism: {mechanism[:200]}"
                    ),
                )
                # Persist proposal
                if self._findings_dir:
                    proposals_path = self._findings_dir / "transfer_proposals.jsonl"
                    proposals_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(proposals_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(proposal.model_dump(mode="json"), default=str) + "\n")
                targets.append(bot_id)
            except Exception:
                logger.warning("Failed to create transfer proposal for %s → %s", source_bot, bot_id)
        return targets

    @staticmethod
    def load_track_record_from_file(findings_dir: Path) -> dict[str, dict]:
        """Load transfer track record directly from outcomes file, no builder needed."""
        outcomes_path = findings_dir / "transfer_outcomes.jsonl"
        if not outcomes_path.exists():
            return {}
        records: list[dict] = []
        for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return TransferProposalBuilder._build_track_record(records)

    @staticmethod
    def _build_track_record(outcomes: list[dict]) -> dict[str, dict]:
        """Compute per-pattern success rates, excluding inconclusive verdicts."""
        if not outcomes:
            return {}
        by_pattern: dict[str, list[dict]] = defaultdict(list)
        for o in outcomes:
            by_pattern[o.get("pattern_id", "")].append(o)
        track_record: dict[str, dict] = {}
        for pattern_id, pattern_outcomes in by_pattern.items():
            decisive = [o for o in pattern_outcomes if o.get("verdict") != "inconclusive"]
            positive = sum(1 for o in decisive if o.get("verdict") == "positive")
            total_decisive = len(decisive)
            track_record[pattern_id] = {
                "total": len(pattern_outcomes),
                "decisive": total_decisive,
                "positive": positive,
                "negative": sum(1 for o in decisive if o.get("verdict") == "negative"),
                "inconclusive": len(pattern_outcomes) - total_decisive,
                "success_rate": round(positive / total_decisive, 3) if total_decisive > 0 else 0.0,
            }
        return track_record

    def _load_outcomes(self) -> list[dict]:
        if not self._outcomes_path or not self._outcomes_path.exists():
            return []
        records: list[dict] = []
        for line in self._outcomes_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def _find_eligible_bots(self, pattern: PatternEntry) -> list[str]:
        """Find bots eligible for receiving a pattern transfer."""
        already_targeted = set(pattern.target_bots or [])
        source = pattern.source_bot

        return [
            bot for bot in self._bots
            if bot != source and bot not in already_targeted
        ]

    # Asset classes that operate in isolated capital pools / markets.
    _ISOLATED_ASSET_CLASSES = frozenset({"k_equity"})

    def _compute_compatibility(self, pattern: PatternEntry, target_bot: str) -> float:
        """Score compatibility between a pattern and a target bot.

        Uses archetype-aware scoring when strategy_registry is available,
        falls back to regime-based scoring otherwise.
        """
        if not self._strategy_registry or not getattr(self._strategy_registry, "strategies", None):
            return self._compute_compatibility_legacy(pattern, target_bot)

        source_arch = self._strategy_registry.archetype_for_strategy(
            getattr(pattern, "source_strategy_id", "") or ""
        )
        target_strategies = self._strategy_registry.strategies_for_bot(target_bot)

        if not source_arch or not target_strategies:
            return self._compute_compatibility_legacy(pattern, target_bot)

        # Hard gate: isolated markets (e.g., k_equity) cannot transfer to/from other markets
        if self._cross_market_incompatible(pattern.source_bot, target_bot):
            return 0.0

        best_score = 0.0
        for _strat_id, _profile in target_strategies.items():
            target_arch = self._strategy_registry.archetype_for_strategy(_strat_id)
            if source_arch == target_arch:
                base = 0.75
            elif self._same_family(pattern.source_bot, target_bot):
                base = 0.60
            else:
                base = 0.30
            if self._same_asset_class(pattern.source_bot, target_bot):
                base = min(1.0, base + 0.10)
            best_score = max(best_score, base)

        # Blend with regime overlap if available
        regime_score = self._regime_overlap_score(pattern.source_bot, target_bot)
        if regime_score is not None:
            return round(0.6 * best_score + 0.4 * regime_score, 4)
        return best_score

    def _compute_compatibility_legacy(self, pattern: PatternEntry, target_bot: str) -> float:
        """Legacy regime-based compatibility scoring."""
        score = self._regime_overlap_score(pattern.source_bot, target_bot)
        return score if score is not None else 0.5

    def _cross_market_incompatible(self, source_bot: str, target_bot: str) -> bool:
        """Check if source and target are in incompatible isolated markets."""
        if not self._strategy_registry:
            return False
        source_strats = self._strategy_registry.strategies_for_bot(source_bot)
        target_strats = self._strategy_registry.strategies_for_bot(target_bot)
        source_isolated = any(
            p.asset_class in self._ISOLATED_ASSET_CLASSES
            for p in source_strats.values()
        )
        target_isolated = any(
            p.asset_class in self._ISOLATED_ASSET_CLASSES
            for p in target_strats.values()
        )
        return source_isolated != target_isolated

    def _same_family(self, source_bot: str, target_bot: str) -> bool:
        """Check if two bots belong to the same strategy family."""
        if not self._strategy_registry:
            return False
        source_strats = self._strategy_registry.strategies_for_bot(source_bot)
        target_strats = self._strategy_registry.strategies_for_bot(target_bot)
        source_families = {p.family for p in source_strats.values() if p.family}
        target_families = {p.family for p in target_strats.values() if p.family}
        return bool(source_families & target_families)

    def _same_asset_class(self, source_bot: str, target_bot: str) -> bool:
        """Check if two bots trade the same asset class."""
        if not self._strategy_registry:
            return False
        source_strats = self._strategy_registry.strategies_for_bot(source_bot)
        target_strats = self._strategy_registry.strategies_for_bot(target_bot)
        source_classes = {p.asset_class for p in source_strats.values() if p.asset_class}
        target_classes = {p.asset_class for p in target_strats.values() if p.asset_class}
        return bool(source_classes & target_classes)

    def _regime_overlap_score(self, source_bot: str, target_bot: str) -> float | None:
        """Compute regime overlap score, or None if data unavailable."""
        source_regimes = self._load_regime_distribution(source_bot)
        target_regimes = self._load_regime_distribution(target_bot)
        if not source_regimes or not target_regimes:
            return None
        source_keys = set(source_regimes.keys())
        target_keys = set(target_regimes.keys())
        if not source_keys or not target_keys:
            return None
        intersection = source_keys & target_keys
        union = source_keys | target_keys
        regime_overlap = len(intersection) / len(union) if union else 0.0
        overlap_weight = 0.0
        for regime in intersection:
            overlap_weight += min(source_regimes.get(regime, 0), target_regimes.get(regime, 0))
        return min(1.0, regime_overlap * 0.5 + overlap_weight * 0.5)

    def _load_regime_distribution(self, bot_id: str) -> dict[str, float]:
        """Load the most recent regime_analysis.json for a bot."""
        from datetime import datetime, timedelta, timezone

        today = datetime.now(timezone.utc)
        for days_back in range(7):
            date_str = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            path = self._curated_dir / date_str / bot_id / "regime_analysis.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data.get("regime_trade_count"), dict):
                        counts = {
                            str(regime): int(count or 0)
                            for regime, count in data["regime_trade_count"].items()
                        }
                    else:
                        counts = {
                            str(regime): int(
                                (value.get("trade_count", 1) if isinstance(value, dict) else 1) or 0
                            )
                            for regime, value in data.items()
                        }
                    total = sum(counts.values())
                    if total > 0:
                        return {regime: count / total for regime, count in counts.items() if count > 0}
                except (json.JSONDecodeError, OSError):
                    pass
        return {}

    def _load_bot_metrics(
        self, bot_id: str, date: str, curated_dir: Path, before: bool,
    ) -> dict | None:
        """Load aggregated bot metrics for 7 days before or after a date.

        Returns dict with pnl, win_rate, trade_count, and dominant_regime.
        """
        from datetime import datetime, timedelta

        if not date:
            return None

        try:
            base = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return None

        if before:
            dates = [(base - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]
        else:
            dates = [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 8)]

        pnl_total = 0.0
        wr_values: list[float] = []
        trade_count = 0
        regime_counts: dict[str, int] = {}
        found = 0

        for d in dates:
            summary_path = curated_dir / d / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    pnl_total += data.get("net_pnl", 0)
                    wr = data.get("win_rate")
                    if wr is not None:
                        wr_values.append(float(wr))
                    trade_count += data.get("total_trades", 0)
                    found += 1
                except (json.JSONDecodeError, OSError, ValueError):
                    pass

            # Load regime data
            regime_path = curated_dir / d / bot_id / "regime_analysis.json"
            if regime_path.exists():
                try:
                    rdata = json.loads(regime_path.read_text(encoding="utf-8"))
                    regime = rdata.get("dominant_regime", "") or rdata.get("regime", "")
                    if regime:
                        regime_counts[regime] = regime_counts.get(regime, 0) + 1
                except (json.JSONDecodeError, OSError):
                    pass

        if found < 3:
            return None

        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else ""  # type: ignore[arg-type]
        return {
            "pnl": pnl_total,
            "win_rate": sum(wr_values) / len(wr_values) if wr_values else 0.0,
            "trade_count": trade_count,
            "dominant_regime": dominant_regime,
        }
