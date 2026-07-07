# skills/suggestion_scorer.py
"""SuggestionScorer — computes per-category success rates from measured outcomes.

Reads outcomes.jsonl, portfolio_outcomes.jsonl, and suggestions.jsonl to build
a CategoryScorecard that tells the system which suggestion categories have
positive track records.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER
from trading_assistant.schemas.suggestion_scoring import CategoryScore, CategoryScorecard
from trading_assistant.skills._outcome_utils import (
    is_authoritative_outcome,
    is_conclusive_outcome,
    is_positive_outcome,
)


# Build reverse mapping: tier → category, plus identity mappings for categories.
# "parameter" is an ambiguous tier (maps to exit_timing, stop_loss, signal, position_sizing)
# — we default it to "filter_threshold" for backward compatibility with existing data.
_TIER_TO_CATEGORY: dict[str, str] = {"parameter": "filter_threshold"}
for _cat, _tier in CATEGORY_TO_TIER.items():
    _TIER_TO_CATEGORY.setdefault(_tier, _cat)  # first category for that tier wins
    _TIER_TO_CATEGORY[_cat] = _cat  # identity for categories that are also tier names
# Additional legacy mappings
_TIER_TO_CATEGORY.setdefault("strategy_variant", "signal")
_TIER_TO_CATEGORY.setdefault("hypothesis", "structural")


_DECAY_RATE = 0.95  # 5%/week, consistent with learning_ledger
_HIGH_QUALITY = frozenset({"high", "medium"})

# Non-portfolio categories eligible for exploration-exploitation tracking
_NON_PORTFOLIO_CATEGORIES: frozenset[str] = frozenset(
    cat for cat, tier in CATEGORY_TO_TIER.items()
    if not cat.startswith("portfolio_")
)


class SuggestionScorer:
    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = findings_dir

    @staticmethod
    def _compute_age_weight(outcome: dict) -> float:
        """Exponential decay weight based on outcome age.

        Uses 5%/week decay (same rate as learning_ledger) so old outcomes
        gradually lose influence, allowing categories to recover.
        """
        ts = outcome.get("measurement_date") or outcome.get("measured_at") or outcome.get("timestamp", "")
        if not ts:
            return 0.5  # unknown age → half weight
        try:
            measured = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            weeks_ago = (datetime.now(timezone.utc) - measured).days / 7
            return _DECAY_RATE ** max(0, weeks_ago)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _target_metric_weight(outcome: dict) -> float:
        """Return weight multiplier based on whether the targeted metric improved.

        1.2 if target improved, 0.8 if target worsened, 1.0 if no info.
        """
        improved = outcome.get("target_metric_improved")
        if improved is True:
            return 1.2
        elif improved is False:
            return 0.8
        return 1.0

    def compute_scorecard(self, *, authoritative_only: bool = False) -> CategoryScorecard:
        """Compute (bot_id, [strategy_id], category) win rates from outcomes.

        Emits two row variants per data point:
          - bot-wide aggregate (strategy_id=None) — preserves existing callers
          - per-strategy stratified (bot_id, strategy_id, category) — for
            records that carry a non-null strategy_id

        Applies category_overrides.jsonl confidence multipliers when present
        (overrides remain bot-aggregate-only). Weights outcomes by target
        metric improvement signal and applies temporal decay.
        """
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()
        if authoritative_only:
            outcomes = [outcome for outcome in outcomes if is_authoritative_outcome(outcome)]
        category_overrides = self._load_category_overrides()

        if not outcomes:
            return CategoryScorecard()

        # Build suggestion_id → (bot_id, strategy_id_or_None, category) mapping
        id_to_info: dict[str, tuple[str, str | None, str]] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            bot_id = s.get("bot_id", "")
            strategy_id = s.get("strategy_id") or None
            category = s.get("category", "")
            if not category:
                tier = s.get("tier", "")
                category = _TIER_TO_CATEGORY.get(tier, tier)
            if sid:
                id_to_info[sid] = (bot_id, strategy_id, category)

        # Group outcomes by (bot_id, None, category) AND (bot_id, strategy_id, category).
        # Each outcome contributes to both the bot-aggregate row and (if it has a
        # strategy_id) the per-strategy row — same outcome, two views.
        raw_groups: dict[tuple[str, str | None, str], dict[str, dict]] = defaultdict(dict)
        for outcome in outcomes:
            quality = outcome.get("measurement_quality", "high")  # legacy defaults to high
            if quality not in _HIGH_QUALITY:
                continue
            if not is_conclusive_outcome(outcome):
                continue
            sid = outcome.get("suggestion_id", "")
            info = id_to_info.get(sid)
            if not info:
                continue
            bot_id, strategy_id, category = info
            # Bot-wide aggregate (always)
            raw_groups[(bot_id, None, category)][sid] = outcome
            # Per-strategy row (only if attributable)
            if strategy_id:
                raw_groups[(bot_id, strategy_id, category)][sid] = outcome
        groups: dict[tuple[str, str | None, str], list[dict]] = {
            k: list(v.values()) for k, v in raw_groups.items()
        }

        scores: list[CategoryScore] = []
        for (bot_id, strategy_id, category), group_outcomes in groups.items():
            # Apply temporal decay × target metric weight
            weights = [
                self._compute_age_weight(o) * self._target_metric_weight(o)
                for o in group_outcomes
            ]
            total_weight = sum(weights)
            total = len(group_outcomes)

            if total_weight > 0:
                positive_weight = sum(
                    w for o, w in zip(group_outcomes, weights, strict=False) if is_positive_outcome(o)
                )
                win_rate = positive_weight / total_weight
                avg_pnl = (
                    sum(self._outcome_pnl_delta(o) * w for o, w in zip(group_outcomes, weights, strict=False))
                    / total_weight
                )
            else:
                positive_weight = 0.0
                win_rate = 0.0
                avg_pnl = 0.0

            # confidence_multiplier: Bayesian posterior mean with Beta(1,1) prior.
            # Overrides are bot-aggregate-only — only apply to strategy_id=None rows.
            override = (
                category_overrides.get((bot_id, category))
                if strategy_id is None else None
            )
            if override is not None:
                multiplier = override
            elif total_weight > 0:
                posterior = (positive_weight + 1) / (total_weight + 2)
                multiplier = max(0.3, min(1.0, posterior * 2.0))
            else:
                multiplier = 1.0

            scores.append(CategoryScore(
                bot_id=bot_id,
                strategy_id=strategy_id,
                category=category,
                win_rate=round(win_rate, 3),
                avg_pnl_delta=round(avg_pnl, 4),
                sample_size=total,
                confidence_multiplier=round(multiplier, 3),
            ))

        return CategoryScorecard(scores=scores)

    def compute_authoritative_scorecard(self) -> CategoryScorecard:
        """Scorecard built only from monthly/follow-up outcomes."""
        return self.compute_scorecard(authoritative_only=True)

    def compute_regime_stratified_scores(self) -> dict[str, dict[str, float]]:
        """Compute win rates stratified by macro regime at implementation time.

        Returns: {macro_regime: {category: win_rate}} for regimes with >=3 outcomes.
        """
        outcomes = self._load_outcomes()
        suggestions = self._load_suggestions()
        if not outcomes:
            return {}

        id_to_cat: dict[str, str] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            category = s.get("category", "")
            if not category:
                tier = s.get("tier", "")
                category = _TIER_TO_CATEGORY.get(tier, tier)
            if sid and category:
                id_to_cat[sid] = category

        # Group by (macro_regime, category)
        regime_groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for o in outcomes:
            quality = o.get("measurement_quality", "high")
            if quality not in _HIGH_QUALITY:
                continue
            macro = o.get("macro_regime_at_implementation", "")
            if not macro:
                continue
            sid = o.get("suggestion_id", "")
            cat = id_to_cat.get(sid, "")
            if cat:
                regime_groups[macro][cat].append(o)

        result: dict[str, dict[str, float]] = {}
        for regime, cat_outcomes in regime_groups.items():
            cat_scores: dict[str, float] = {}
            for cat, outcomes_list in cat_outcomes.items():
                if len(outcomes_list) < 3:
                    continue
                positive = sum(1 for o in outcomes_list if is_positive_outcome(o))
                cat_scores[cat] = round(positive / len(outcomes_list), 3)
            if cat_scores:
                result[regime] = cat_scores
        return result

    @staticmethod
    def apply_regime_confidence_adjustment(
        confidence: float,
        category: str,
        current_macro_regime: str,
    ) -> float:
        """Adjust suggestion confidence based on current macro regime.

        In S/D regimes: boost defensive suggestions (reduce sizing, tighter caps),
        reduce aggressive ones (relax filters, wider caps).
        """
        if not current_macro_regime or current_macro_regime not in ("S", "D"):
            return confidence

        defensive_categories = {"stop_loss", "position_sizing", "regime_gate"}
        aggressive_categories = {"filter_threshold", "signal"}

        if category in defensive_categories:
            return round(min(1.0, confidence * 1.15), 3)
        elif category in aggressive_categories:
            return round(confidence * 0.85, 3)
        return confidence

    def compute_category_value_map(self) -> dict:
        """Per-(bot_id, category): avg_composite_delta, suggestion_count, value_per_suggestion.

        Also generates recommendations for shifting effort between categories.
        """
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()

        # Count suggestions per (bot_id, category)
        suggestion_counts: dict[str, int] = defaultdict(int)
        id_to_key: dict[str, str] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            bot_id = s.get("bot_id", "")
            category = s.get("category", "")
            if not category:
                tier = s.get("tier", "")
                category = _TIER_TO_CATEGORY.get(tier, tier)
            key = f"{bot_id}:{category}"
            suggestion_counts[key] += 1
            if sid:
                id_to_key[sid] = key

        # Group outcome deltas by (bot_id, category)
        outcome_deltas: dict[str, list[float]] = defaultdict(list)
        positive_counts: dict[str, int] = defaultdict(int)
        for outcome in outcomes:
            quality = outcome.get("measurement_quality", "high")
            if quality not in _HIGH_QUALITY:
                continue
            if not is_conclusive_outcome(outcome):
                continue
            sid = outcome.get("suggestion_id", "")
            key = id_to_key.get(sid)
            if key:
                delta = self._outcome_pnl_delta(outcome)
                outcome_deltas[key].append(delta)
                if is_positive_outcome(outcome):
                    positive_counts[key] += 1

        # Build value map
        all_keys = set(suggestion_counts) | set(outcome_deltas)
        value_map: dict[str, dict] = {}
        for key in all_keys:
            deltas = outcome_deltas.get(key, [])
            count = suggestion_counts.get(key, 0)
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            vps = avg_delta / count if count > 0 else 0.0
            value_map[key] = {
                "suggestion_count": count,
                "measured_count": len(deltas),
                "positive_count": positive_counts.get(key, 0),
                "avg_composite_delta": round(avg_delta, 4),
                "value_per_suggestion": round(vps, 4),
            }

        # Add unexplored entries for bot:category combos with no data
        seen_bots = {key.split(":")[0] for key in all_keys if ":" in key}
        for bot_id in seen_bots:
            for category in _NON_PORTFOLIO_CATEGORIES:
                key = f"{bot_id}:{category}"
                if key not in value_map:
                    value_map[key] = {
                        "suggestion_count": 0,
                        "measured_count": 0,
                        "positive_count": 0,
                        "avg_composite_delta": 0.0,
                        "value_per_suggestion": 0.0,
                        "unexplored": True,
                    }

        # Generate recommendations
        recommendations: list[str] = []
        explored = {k: v for k, v in value_map.items() if not v.get("unexplored")}
        sorted_keys = sorted(explored, key=lambda k: explored[k]["value_per_suggestion"])
        high_value = [k for k in sorted_keys if explored[k]["value_per_suggestion"] > 0]
        low_value = [k for k in sorted_keys if explored[k]["value_per_suggestion"] < 0]
        for lv in low_value[:2]:
            if high_value:
                recommendations.append(
                    f"shift effort from {lv} to {high_value[-1]}"
                )
        for key, info in explored.items():
            if info["measured_count"] >= 3 and info["positive_count"] == 0:
                recommendations.append(
                    f"deprioritize {key} with 0 positive outcomes in {info['measured_count']} attempts"
                )
        unexplored_keys = [k for k, v in value_map.items() if v.get("unexplored")]
        if unexplored_keys:
            recommendations.append(
                f"consider exploring unexplored categories: {', '.join(sorted(unexplored_keys)[:5])}"
            )

        value_map["_recommendations"] = recommendations  # type: ignore[assignment]
        return value_map

    def compute_detector_confidence(self) -> dict[str, float]:
        """Compute per-detector confidence multipliers from outcome data.

        Groups outcomes by detector_name (from detection_context in suggestions.jsonl),
        applies temporal decay, and returns {detector_name: confidence_multiplier}.
        Detectors with no outcome data return no entry (caller uses 1.0 default).
        """
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()

        if not outcomes or not suggestions:
            return {}

        # Map suggestion_id → detector_name
        sid_to_detector: dict[str, str] = {}
        for s in suggestions:
            sid = s.get("suggestion_id", "")
            ctx = s.get("detection_context") or {}
            detector = ctx.get("detector_name", "")
            if sid and detector:
                sid_to_detector[sid] = detector

        # Group outcomes by detector, deduplicating by suggestion_id
        raw_groups: dict[str, dict[str, dict]] = defaultdict(dict)
        for outcome in outcomes:
            quality = outcome.get("measurement_quality", "high")
            if quality not in _HIGH_QUALITY:
                continue
            if not is_conclusive_outcome(outcome):
                continue
            sid = outcome.get("suggestion_id", "")
            detector = sid_to_detector.get(sid)
            if detector:
                raw_groups[detector][sid] = outcome

        result: dict[str, float] = {}
        for detector, oc_by_sid in raw_groups.items():
            group_outcomes = list(oc_by_sid.values())
            weights = [self._compute_age_weight(o) for o in group_outcomes]
            total_weight = sum(weights)
            if total_weight <= 0:
                continue

            positive_weight = sum(
                w for o, w in zip(group_outcomes, weights, strict=False) if is_positive_outcome(o)
            )
            # Bayesian posterior → confidence multiplier (same formula as scorecard)
            posterior = (positive_weight + 1) / (total_weight + 2)
            multiplier = max(0.3, min(1.0, posterior * 2.0))
            result[detector] = round(multiplier, 3)

        return result

    @staticmethod
    def _outcome_pnl_delta(outcome: dict) -> float:
        """Extract PnL delta from outcome, supporting bot + portfolio schemas."""
        delta = outcome.get("pnl_delta")
        if delta is None:
            delta = outcome.get("composite_delta")
        if delta is None:
            delta = outcome.get("objective_delta")
        if delta is None:
            delta = outcome.get("live_vs_expected_objective_delta")
        if delta is None:
            delta_30d = outcome.get("pnl_delta_30d")
            delta_7d = outcome.get("pnl_delta_7d")
            try:
                delta = delta_30d if float(delta_30d) != 0.0 or delta_7d is None else delta_7d
            except (TypeError, ValueError):
                delta = delta_30d if delta_30d is not None else delta_7d
        if delta is None:
            delta = 0
        try:
            return float(delta)
        except (TypeError, ValueError):
            return 0.0

    def compute_suggestion_quality_trend(
        self, weeks: int = 8, *, value_map: dict | None = None,
    ) -> dict:
        """Compute per-week suggestion quality metrics with rolling average.

        Args:
            weeks: Number of weeks to analyze.
            value_map: Pre-computed category value map (avoids redundant I/O
                when called from ContextBuilder which already computed it).

        Returns:
            {
                "weekly_metrics": [{"week": "...", "hit_rate": 0.X, "high_value_ratio": 0.X}, ...],
                "rolling_avg_hit_rate": 0.X,
                "rolling_avg_high_value_ratio": 0.X,
                "trend": "improving" | "degrading" | "stable"
            }
        """
        suggestions = self._load_suggestions()
        outcomes = self._load_outcomes()

        if not suggestions:
            return {}

        # Map suggestion_id → outcome verdict (quality + conclusiveness filtered)
        sid_to_positive: dict[str, bool] = {}
        for o in outcomes:
            quality = o.get("measurement_quality", "high")
            if quality not in _HIGH_QUALITY:
                continue
            if not is_conclusive_outcome(o):
                continue
            sid = o.get("suggestion_id", "")
            if sid:
                sid_to_positive[sid] = is_positive_outcome(o)

        # Compute value map for top-3 category identification
        if value_map is None:
            value_map = self.compute_category_value_map()
        explored = {
            k: v for k, v in value_map.items()
            if k != "_recommendations" and not (isinstance(v, dict) and v.get("unexplored"))
        }
        sorted_cats = sorted(
            explored.items(),
            key=lambda x: x[1].get("value_per_suggestion", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        top_categories = {k for k, _ in sorted_cats[:3]}

        # Group suggestions by ISO week
        from datetime import datetime
        weekly: dict[str, list[dict]] = {}
        for s in suggestions:
            ts = s.get("proposed_at", "") or s.get("timestamp", "") or s.get("created_at", "")
            if not ts or len(ts) < 10:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                week_key = dt.strftime("%G-W%V")
            except (ValueError, TypeError):
                continue
            status = s.get("status", "")
            if status in ("deployed", "implemented", "measured"):
                weekly.setdefault(week_key, []).append(s)

        sorted_weeks = sorted(weekly.keys())[-weeks:]
        if not sorted_weeks:
            return {}

        weekly_metrics: list[dict] = []
        for wk in sorted_weeks:
            items = weekly[wk]
            total = len(items)
            positive = sum(
                1 for s in items
                if sid_to_positive.get(s.get("suggestion_id", ""), False)
            )
            hit_rate = positive / total if total > 0 else 0.0

            # High-value ratio: suggestions in top-3 categories
            in_top = sum(
                1 for s in items
                if f"{s.get('bot_id', '')}:{s.get('category', '')}" in top_categories
            )
            high_value_ratio = in_top / total if total > 0 else 0.0

            weekly_metrics.append({
                "week": wk,
                "hit_rate": round(hit_rate, 3),
                "high_value_ratio": round(high_value_ratio, 3),
                "total_implemented": total,
            })

        # 4-week rolling averages
        recent = weekly_metrics[-4:]
        if recent:
            avg_hr = sum(m["hit_rate"] for m in recent) / len(recent)
            avg_hvr = sum(m["high_value_ratio"] for m in recent) / len(recent)
        else:
            avg_hr = avg_hvr = 0.0

        # Trend: compare first half vs second half hit rates
        if len(weekly_metrics) >= 4:
            mid = len(weekly_metrics) // 2
            first_half = sum(m["hit_rate"] for m in weekly_metrics[:mid]) / mid
            second_half = sum(m["hit_rate"] for m in weekly_metrics[mid:]) / (len(weekly_metrics) - mid)
            if second_half > first_half + 0.05:
                trend = "improving"
            elif second_half < first_half - 0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "weekly_metrics": weekly_metrics,
            "rolling_avg_hit_rate": round(avg_hr, 3),
            "rolling_avg_high_value_ratio": round(avg_hvr, 3),
            "trend": trend,
        }

    def apply_recalibration(self, discard_items: list) -> None:
        """Write category overrides from discard items.

        Discarded categories get confidence_multiplier = 0.3, which
        compute_scorecard() will pick up on next call.
        """
        overrides_path = self._findings_dir / "category_overrides.jsonl"
        overrides_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing overrides
        existing: dict[tuple[str, str], dict] = {}
        if overrides_path.exists():
            for line in overrides_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    try:
                        entry = json.loads(line)
                        key = (entry.get("bot_id", ""), entry.get("category", ""))
                        existing[key] = entry
                    except json.JSONDecodeError:
                        pass

        # Update/add overrides from discard items
        from datetime import datetime, timezone
        for item in discard_items:
            bot_id = item.bot_id if hasattr(item, "bot_id") else item.get("bot_id", "")
            category = item.category if hasattr(item, "category") else item.get("category", "")
            key = (bot_id, category)
            existing[key] = {
                "bot_id": bot_id,
                "category": category,
                "confidence_multiplier": 0.3,
                "reason": item.reason if hasattr(item, "reason") else item.get("reason", "discarded"),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        # Rewrite file
        with open(overrides_path, "w", encoding="utf-8") as f:
            for entry in existing.values():
                f.write(json.dumps(entry) + "\n")

    def _load_category_overrides(self) -> dict[tuple[str, str], float]:
        """Load category confidence multiplier overrides."""
        path = self._findings_dir / "category_overrides.jsonl"
        if not path.exists():
            return {}
        overrides: dict[tuple[str, str], float] = {}
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    key = (entry.get("bot_id", ""), entry.get("category", ""))
                    overrides[key] = entry.get("confidence_multiplier", 1.0)
                except json.JSONDecodeError:
                    pass
        return overrides

    def _load_suggestions(self) -> list[dict]:
        return self._read_jsonl(self._findings_dir / "suggestions.jsonl")

    def _load_outcomes(self) -> list[dict]:
        outcomes = self._read_jsonl(self._findings_dir / "outcomes.jsonl")
        outcomes.extend(self._read_jsonl(self._findings_dir / "portfolio_outcomes.jsonl"))
        return outcomes

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records
