# analysis/response_validator.py
"""Response validator — enforces constraints on agent suggestions and predictions.

Strips blocked suggestions, adjusts confidence based on calibration and track record,
and appends Validator Notes to the report.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from trading_assistant.schemas.agent_response import (
    CATEGORY_TO_TIER,
    AgentPrediction,
    AgentSuggestion,
    ParsedAnalysis,
    StructuralProposal,
)
from trading_assistant.schemas.suggestion_scoring import CategoryScorecard

logger = logging.getLogger(__name__)


@dataclass
class BlockedSuggestion:
    suggestion: AgentSuggestion
    reason: str


@dataclass
class BlockedStructuralProposal:
    proposal: StructuralProposal
    reason: str


@dataclass
class BlockedPortfolioProposal:
    proposal: object  # PortfolioProposal
    reason: str


@dataclass
class ValidationResult:
    approved_suggestions: list[AgentSuggestion] = field(default_factory=list)
    blocked_suggestions: list[BlockedSuggestion] = field(default_factory=list)
    approved_predictions: list[AgentPrediction] = field(default_factory=list)
    approved_structural_proposals: list[StructuralProposal] = field(default_factory=list)
    blocked_structural_proposals: list[BlockedStructuralProposal] = field(default_factory=list)
    approved_portfolio_proposals: list = field(default_factory=list)
    blocked_portfolio_proposals: list[BlockedPortfolioProposal] = field(default_factory=list)
    validator_notes: str = ""


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class ResponseValidator:
    # Recognized metric names for acceptance criteria validation (Gate 4).
    # Aligned with _METRIC_FIELD_MAP in app.py and GroundTruthSnapshot fields.
    _RECOGNIZED_METRICS: frozenset[str] = frozenset({
        "sharpe", "sharpe_ratio", "sharpe_ratio_30d",
        "win_rate",
        "drawdown", "max_drawdown", "max_drawdown_pct",
        "process_quality", "avg_process_quality",
        "composite", "composite_score",
        "pnl", "pnl_total", "net_pnl", "expected_r", "expected_total_r",
        "calmar", "calmar_ratio", "calmar_ratio_30d",
        "profit_factor",
        "expectancy",
        "trade_count",
    })
    _RECOGNIZED_DIRECTIONS: frozenset[str] = frozenset({"improve", "not_degrade"})

    # Portfolio guardrail constants
    _MAX_ALLOC_CHANGE_PCT = 0.15  # 15% max per family per cycle
    _MIN_ALLOC_FLOOR = 0.05  # 5% minimum per family
    _MAX_RISK_CAP_CHANGE_PCT = 0.20  # 20% max heat_cap change
    _MIN_EVIDENCE_ALLOC = 60  # days for allocation changes
    _MIN_EVIDENCE_RISK = 90  # days for risk/drawdown changes
    _MIN_PREDICTION_ACCURACY = 0.40  # block all portfolio proposals if below
    _MIN_PORTFOLIO_CONFIDENCE = 0.3  # minimum confidence for portfolio proposals

    def __init__(
        self,
        rejected_suggestions: list[dict] | None = None,
        forecast_meta: dict | None = None,
        category_scorecard: CategoryScorecard | None = None,
        hypothesis_track_record: dict | None = None,
        prediction_accuracy: dict | None = None,
        recalibrations: list[dict] | None = None,
        outcome_priors: list[dict] | None = None,
        current_macro_regime: str = "",
        strategy_registry: object | None = None,
        safety_critical_params: set[str] | None = None,
    ) -> None:
        self._rejected = rejected_suggestions or []
        self._forecast_meta = forecast_meta or {}
        self._scorecard = category_scorecard or CategoryScorecard()
        self._hypothesis_track_record = hypothesis_track_record or {}
        self._prediction_accuracy = prediction_accuracy or {}
        self._outcome_priors = outcome_priors or []
        self._current_macro_regime = current_macro_regime
        self._strategy_registry = strategy_registry  # StrategyRegistry or None
        self._safety_critical_params: set[str] = safety_critical_params or set()
        # Build recalibration index: (bot_id, category) → mean revised_confidence
        self._recalib_by_key: dict[tuple[str, str], float] = {}
        if recalibrations:
            from collections import defaultdict
            _groups: dict[tuple[str, str], list[float]] = defaultdict(list)
            for rec in recalibrations:
                bot = rec.get("bot_id", "")
                cat = rec.get("category", "")
                rev = rec.get("revised_confidence")
                if bot and cat and rev is not None:
                    _groups[(bot, cat)].append(float(rev))
            self._recalib_by_key = {
                k: sum(v) / len(v) for k, v in _groups.items()
            }

    def validate(self, parsed: ParsedAnalysis) -> ValidationResult:
        """Validate a parsed analysis, filtering suggestions and adjusting confidence."""
        approved: list[AgentSuggestion] = []
        blocked: list[BlockedSuggestion] = []
        notes_lines: list[str] = []

        for suggestion in parsed.suggestions:
            # 1. Rejection check — fuzzy match against rejected suggestions
            rejection_match = self._check_rejection(suggestion)
            if rejection_match:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=f"Similar to rejected suggestion: {rejection_match}",
                ))
                continue

            # 1b. Safety-critical ablation check
            if suggestion.ablation_flag and suggestion.ablation_flag in self._safety_critical_params:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=(
                        f"Safety-critical ablation flag '{suggestion.ablation_flag}' "
                        f"cannot be toggled autonomously — requires manual review"
                    ),
                ))
                continue

            # 1c. Engine existence check
            if suggestion.engine and self._strategy_registry:
                engine_valid = self._validate_engine(suggestion)
                if not engine_valid:
                    blocked.append(BlockedSuggestion(
                        suggestion=suggestion,
                        reason=(
                            f"Engine '{suggestion.engine}' not found in strategy "
                            f"sub_engines for bot '{suggestion.bot_id}'"
                        ),
                    ))
                    continue

            # 1c-bis. Inactive strategy_id check — block suggestions targeting
            # a strategy that is no longer in the registry. Suggestions for
            # retired strategies waste implementation cycles and pollute
            # learning signal for whatever replaced them.
            if suggestion.strategy_id and self._strategy_registry is not None:
                is_active = getattr(self._strategy_registry, "is_active", None)
                if callable(is_active) and not is_active(suggestion.strategy_id):
                    blocked.append(BlockedSuggestion(
                        suggestion=suggestion,
                        reason=(
                            f"Suggestion targets retired strategy "
                            f"'{suggestion.strategy_id}' — strategy is not in the "
                            f"current registry"
                        ),
                    ))
                    continue

            # 1d. Crypto safety gates
            if self._strategy_registry and self._is_crypto_bot(suggestion.bot_id):
                crypto_block = self._check_crypto_safety_gates(suggestion)
                if crypto_block:
                    blocked.append(BlockedSuggestion(
                        suggestion=suggestion,
                        reason=crypto_block,
                    ))
                    continue

            # 2. Category track record check — prefer per-strategy row when
            # we have one, fall back to bot-wide aggregate (handled by
            # CategoryScorecard.get_score itself).
            score = self._scorecard.get_score(
                suggestion.bot_id, suggestion.category, suggestion.strategy_id,
            )
            if score and score.sample_size >= 5 and score.win_rate < 0.3:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=f"Poor category track record: {suggestion.category} win_rate={score.win_rate:.0%} (n={score.sample_size})",
                ))
                continue

            # 2b. Simplicity criterion — marginal track record not worth implementation cost
            if (
                score
                and score.sample_size >= 3
                and score.win_rate < 0.5
                and score.avg_pnl_delta < 0.001
            ):
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=(
                        f"Blocked: marginal track record in {suggestion.category} "
                        f"(win_rate={score.win_rate:.0%}, avg_pnl={score.avg_pnl_delta:.4f}) "
                        f"— not worth the implementation cost"
                    ),
                ))
                continue

            prior_block = self._check_negative_monthly_prior(suggestion)
            if prior_block:
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=prior_block,
                ))
                continue

            # 2c. Low-confidence structural suggestions blocked
            if (
                suggestion.category == "structural"
                and suggestion.confidence < 0.4
            ):
                blocked.append(BlockedSuggestion(
                    suggestion=suggestion,
                    reason=(
                        "Blocked: low-confidence structural changes carry "
                        "high risk for uncertain benefit"
                    ),
                ))
                continue

            # 3. Calibration adjustment
            adjusted = self._apply_calibration(suggestion)
            approved.append(adjusted)

        # Adjust prediction confidence based on calibration
        approved_predictions = [
            self._adjust_prediction_confidence(p) for p in parsed.predictions
        ]

        # Validate structural proposals
        approved_proposals, blocked_proposals = (
            self._validate_structural_proposals(parsed.structural_proposals)
        )

        # Validate portfolio proposals
        approved_portfolio, blocked_portfolio = (
            self._validate_portfolio_proposals(
                getattr(parsed, "portfolio_proposals", [])
            )
        )

        # Build validator notes
        if blocked:
            notes_lines.append(f"**{len(blocked)} suggestion(s) blocked:**")
            for b in blocked:
                notes_lines.append(f"- \"{b.suggestion.title}\" — {b.reason}")

        self._forecast_meta.get("calibration_adjustment", 0)
        rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
        if rolling_acc and rolling_acc < 0.5:
            notes_lines.append(
                f"\n**Calibration warning:** Rolling 4-week accuracy is {rolling_acc:.0%}. "
                f"Confidence scores have been adjusted downward."
            )

        if blocked_proposals:
            notes_lines.append(f"\n**{len(blocked_proposals)} structural proposal(s) blocked:**")
            for bp in blocked_proposals:
                notes_lines.append(f"- \"{bp.proposal.title}\" — {bp.reason}")

        if blocked_portfolio:
            notes_lines.append(f"\n**{len(blocked_portfolio)} portfolio proposal(s) blocked:**")
            for bp in blocked_portfolio:
                title = getattr(bp.proposal, "proposal_type", "unknown")
                notes_lines.append(f"- {title} — {bp.reason}")

        return ValidationResult(
            approved_suggestions=approved,
            blocked_suggestions=blocked,
            approved_predictions=approved_predictions,
            approved_structural_proposals=approved_proposals,
            blocked_structural_proposals=blocked_proposals,
            approved_portfolio_proposals=approved_portfolio,
            blocked_portfolio_proposals=blocked_portfolio,
            validator_notes="\n".join(notes_lines),
        )

    def _check_rejection(self, suggestion: AgentSuggestion) -> str | None:
        """Check if a suggestion is too similar to a rejected one."""
        for rejected in self._rejected:
            rej_bot = rejected.get("bot_id", "")
            rej_title = rejected.get("title", "")
            rejected.get("tier", "")

            # Must be same bot
            if rej_bot and suggestion.bot_id and rej_bot != suggestion.bot_id:
                continue

            # Check title similarity
            if _jaccard_similarity(suggestion.title, rej_title) >= 0.6:
                return rej_title

        return None

    def _check_negative_monthly_prior(self, suggestion: AgentSuggestion) -> str | None:
        """Require stronger evidence after repeated negative monthly outcomes."""
        prior = self._matching_negative_monthly_prior(
            suggestion.bot_id,
            suggestion.strategy_id or "",
            suggestion.category,
        )
        if prior is None:
            return None
        if suggestion.confidence >= 0.85:
            return None
        category = prior.get("category") or prior.get("mutation_family") or suggestion.category
        return (
            f"Negative monthly prior for {category or 'this family'} "
            "requires confidence >= 0.85 and stronger replay evidence"
        )

    def _matching_negative_monthly_prior(
        self,
        bot_id: str,
        strategy_id: str,
        category: str,
    ) -> dict | None:
        tier = CATEGORY_TO_TIER.get(category, category)
        tokens = {value for value in (category, tier) if value}
        for prior in self._outcome_priors:
            prior_bot = prior.get("bot_id") or ""
            if prior_bot and prior_bot != bot_id:
                continue
            prior_strategy = prior.get("strategy_id") or ""
            if prior_strategy and prior_strategy != strategy_id:
                continue
            prior_tokens = {
                value
                for value in (
                    prior.get("category") or "",
                    prior.get("mutation_family") or "",
                )
                if value
            }
            if prior_tokens and tokens and not (prior_tokens & tokens):
                continue
            if prior.get("gate_strictness") != "stricter":
                continue
            if int(prior.get("negative_count") or 0) < 1:
                continue
            return prior
        return None

    def _apply_calibration(self, suggestion: AgentSuggestion) -> AgentSuggestion:
        """Apply calibration-based confidence adjustments.

        When empirical calibration buckets are available, uses bucket-matched
        observed accuracy to blend confidence. Falls back to heuristic when
        no bucket data exists.
        """
        confidence = suggestion.confidence

        # Apply causal recalibration if available for this (bot_id, category)
        recalib_conf = self._recalib_by_key.get(
            (suggestion.bot_id, suggestion.category),
        )
        if recalib_conf is not None:
            confidence = confidence * 0.6 + recalib_conf * 0.4

        # Try empirical bucket-based adjustment first
        bucket_adjusted = self._bucket_adjust_confidence(confidence)
        if bucket_adjusted is not None:
            confidence = bucket_adjusted
        else:
            # Heuristic fallback
            rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
            rolling_factor = rolling_acc if (rolling_acc and rolling_acc < 0.5) else 1.0
            score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
            category_factor = score.confidence_multiplier if (score and score.confidence_multiplier < 1.0) else 1.0
            confidence *= min(rolling_factor, category_factor)

        calibration = self._forecast_meta.get("calibration_adjustment", 0)
        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        # Still apply category factor even when using bucket adjustment
        if bucket_adjusted is not None:
            score = self._scorecard.get_score(suggestion.bot_id, suggestion.category)
            if score and score.confidence_multiplier < 1.0:
                confidence *= score.confidence_multiplier

        # Apply macro regime confidence adjustment
        if self._current_macro_regime:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer
            regime_conf = SuggestionScorer.apply_regime_confidence_adjustment(
                confidence=confidence,
                category=suggestion.category,
                current_macro_regime=self._current_macro_regime,
            )
            confidence = regime_conf

        confidence = max(0.0, min(1.0, round(confidence, 3)))

        return suggestion.model_copy(update={"confidence": confidence})

    def _adjust_prediction_confidence(self, prediction: AgentPrediction) -> AgentPrediction:
        """Adjust prediction confidence based on forecast meta."""
        confidence = prediction.confidence

        # Try empirical bucket-based adjustment first
        bucket_adjusted = self._bucket_adjust_confidence(confidence)
        if bucket_adjusted is not None:
            confidence = bucket_adjusted
        else:
            rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
            if rolling_acc and rolling_acc < 0.5:
                confidence *= rolling_acc

        calibration = self._forecast_meta.get("calibration_adjustment", 0)
        if calibration and calibration < -0.2:
            confidence = min(confidence, 0.6)

        # Apply directional bias correction from prediction track record
        directional_bias = self._forecast_meta.get("directional_bias", {})
        if directional_bias:
            metric_bias = directional_bias.get(prediction.metric, {})
            bias_direction = metric_bias.get("bias", "balanced")
            gap_pct = metric_bias.get("gap_pct", 0)

            # Systematically optimistic about "improve" → reduce confidence
            if bias_direction == "optimistic" and prediction.direction == "improve":
                penalty = min(0.2, gap_pct / 100)
                confidence *= (1.0 - penalty)
            # Systematically pessimistic about "decline" → reduce confidence
            elif bias_direction == "pessimistic" and prediction.direction == "decline":
                penalty = min(0.2, gap_pct / 100)
                confidence *= (1.0 - penalty)

        confidence = max(0.0, min(1.0, round(confidence, 3)))
        return prediction.model_copy(update={"confidence": confidence})

    def _validate_structural_proposals(
        self,
        proposals: list[StructuralProposal],
    ) -> tuple[list[StructuralProposal], list[BlockedStructuralProposal]]:
        """Validate structural proposals with parity gates matching parameter validation.

        Gates applied (mirrors parameter suggestion validation):
        1. Hypothesis track record — block if effectiveness <= 0 or retired
        2. Category track record — block if win_rate < 0.3 with sufficient data
        3. Simplicity criterion — block marginal track record suggestions
        4. Acceptance criteria presence — must have at least one metric-bearing criterion
        5. Low-confidence block — block structural proposals below 0.4 confidence
        6. Calibration adjustment — adjust confidence based on empirical data

        Returns (approved, blocked).
        """
        approved: list[StructuralProposal] = []
        blocked_list: list[BlockedStructuralProposal] = []

        for proposal in proposals:
            # Gate 1: Hypothesis track record
            hyp_id = proposal.hypothesis_id
            if hyp_id and self._hypothesis_track_record:
                hyp_data = self._hypothesis_track_record.get(hyp_id)
                if hyp_data:
                    effectiveness = hyp_data.get("effectiveness", 0.0)
                    status = hyp_data.get("status", "")
                    if effectiveness <= 0 or status == "retired":
                        reason = (
                            f"Hypothesis {hyp_id} has "
                            f"effectiveness={effectiveness:.3f}, status={status}"
                        )
                        blocked_list.append(BlockedStructuralProposal(
                            proposal=proposal, reason=reason,
                        ))
                        logger.info(
                            "Blocked structural proposal '%s': %s",
                            proposal.title, reason,
                        )
                        continue

            # Gate 2: Category track record (parity with parameter suggestions)
            bot_id = getattr(proposal, "bot_id", "") or ""
            strategy_id = getattr(proposal, "affected_strategy_id", "") or getattr(proposal, "strategy_id", "") or ""
            category = self._infer_structural_category(proposal)
            if bot_id and category:
                score = self._scorecard.get_score(bot_id, category)
                if score and score.sample_size >= 5 and score.win_rate < 0.3:
                    reason = (
                        f"Poor category track record: {category} "
                        f"win_rate={score.win_rate:.0%} (n={score.sample_size})"
                    )
                    blocked_list.append(BlockedStructuralProposal(
                        proposal=proposal, reason=reason,
                    ))
                    logger.info(
                        "Blocked structural proposal '%s': %s",
                        proposal.title, reason,
                    )
                    continue

                # Gate 3: Simplicity criterion — marginal track record
                if (
                    score
                    and score.sample_size >= 3
                    and score.win_rate < 0.5
                    and score.avg_pnl_delta < 0.001
                ):
                    reason = (
                        f"Marginal track record in {category} "
                        f"(win_rate={score.win_rate:.0%}, avg_pnl={score.avg_pnl_delta:.4f})"
                    )
                    blocked_list.append(BlockedStructuralProposal(
                        proposal=proposal, reason=reason,
                    ))
                    logger.info(
                        "Blocked structural proposal '%s': %s",
                        proposal.title, reason,
                    )
                    continue

            # Gate 4: Must have at least one well-formed criterion with recognized
            # metric name and valid direction. Rejects malformed criteria like
            # {"metric": "foo"} that would pass a presence-only check.
            criteria = proposal.acceptance_criteria
            well_formed_count = 0
            gate4_issues: list[str] = []
            for c in criteria:
                if not isinstance(c, dict):
                    continue
                metric = c.get("metric", "")
                if not metric:
                    continue
                if metric not in self._RECOGNIZED_METRICS:
                    gate4_issues.append(f"unrecognized metric '{metric}'")
                    continue
                direction = c.get("direction", "")
                if direction and direction not in self._RECOGNIZED_DIRECTIONS:
                    gate4_issues.append(
                        f"invalid direction '{direction}' for {metric}"
                    )
                    continue
                well_formed_count += 1
            if well_formed_count == 0:
                detail = "; ".join(gate4_issues) if gate4_issues else "no metric field"
                blocked_list.append(BlockedStructuralProposal(
                    proposal=proposal,
                    reason=f"No valid acceptance criteria ({detail})",
                ))
                continue

            # Gate 5: Low-confidence structural proposals blocked
            confidence = getattr(proposal, "confidence", 0.5) or 0.5
            if confidence < 0.4:
                reason = f"Low confidence: {confidence:.2f} < 0.4"
                blocked_list.append(BlockedStructuralProposal(
                    proposal=proposal, reason=reason,
                ))
                logger.info(
                    "Blocked structural proposal '%s': %s",
                    proposal.title, reason,
                )
                continue

            prior = self._matching_negative_monthly_prior(bot_id, strategy_id, category)
            if prior is not None and confidence < 0.85:
                prior_category = prior.get("category") or prior.get("mutation_family") or category
                reason = (
                    f"Negative monthly prior for {prior_category} requires "
                    "confidence >= 0.85 and stronger replay evidence"
                )
                blocked_list.append(BlockedStructuralProposal(
                    proposal=proposal, reason=reason,
                ))
                logger.info(
                    "Blocked structural proposal '%s': %s",
                    proposal.title, reason,
                )
                continue

            # Gate 6: Calibration adjustment on confidence
            if bot_id and category:
                adjusted_confidence = self._calibrate_structural_confidence(
                    confidence, bot_id, category,
                )
                if adjusted_confidence < 0.4:
                    reason = f"Calibrated confidence too low: {adjusted_confidence:.2f} < 0.4"
                    blocked_list.append(BlockedStructuralProposal(
                        proposal=proposal, reason=reason,
                    ))
                    logger.info(
                        "Blocked structural proposal '%s': %s",
                        proposal.title, reason,
                    )
                    continue
                if hasattr(proposal, "model_copy"):
                    proposal = proposal.model_copy(
                        update={"confidence": adjusted_confidence},
                    )

            approved.append(proposal)

        return approved, blocked_list

    def _infer_structural_category(self, proposal: StructuralProposal) -> str:
        """Infer suggestion category from structural proposal metadata.

        Checks acceptance_criteria metrics and proposal title/description
        to map to the category scorecard taxonomy.
        """
        # Check acceptance criteria for metric hints
        for c in proposal.acceptance_criteria:
            if isinstance(c, dict):
                metric = c.get("metric", "")
                if metric in ("pnl", "expected_r", "net_pnl"):
                    return "signal"
                if metric in ("calmar", "sharpe", "drawdown", "max_drawdown"):
                    return "stop_loss"
                if metric in ("win_rate", "profit_factor"):
                    return "signal"
                if metric in ("process_quality",):
                    return "structural"

        # Fall back to title keyword matching
        title_lower = (getattr(proposal, "title", "") or "").lower()
        if any(kw in title_lower for kw in ("stop", "drawdown", "risk")):
            return "stop_loss"
        if any(kw in title_lower for kw in ("filter", "threshold")):
            return "filter_threshold"
        if any(kw in title_lower for kw in ("regime", "gate")):
            return "regime_gate"
        if any(kw in title_lower for kw in ("exit", "trailing", "take profit")):
            return "exit_timing"
        if any(kw in title_lower for kw in ("position", "sizing")):
            return "position_sizing"

        return "structural"

    def _calibrate_structural_confidence(
        self,
        confidence: float,
        bot_id: str,
        category: str,
    ) -> float:
        """Apply calibration adjustments to structural proposal confidence.

        Uses the same recalibration and empirical bucket logic as parameter
        suggestions for parity.
        """
        # Causal recalibration if available
        recalib_conf = self._recalib_by_key.get((bot_id, category))
        if recalib_conf is not None:
            confidence = confidence * 0.6 + recalib_conf * 0.4

        # Empirical bucket adjustment
        bucket_adjusted = self._bucket_adjust_confidence(confidence)
        if bucket_adjusted is not None:
            confidence = bucket_adjusted
        else:
            rolling_acc = self._forecast_meta.get("rolling_accuracy_4w", 0)
            if rolling_acc and rolling_acc < 0.5:
                confidence *= rolling_acc

        # Category factor
        score = self._scorecard.get_score(bot_id, category)
        if score and score.confidence_multiplier < 1.0:
            confidence *= score.confidence_multiplier

        # Macro regime adjustment
        if self._current_macro_regime:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer
            confidence = SuggestionScorer.apply_regime_confidence_adjustment(
                confidence=confidence,
                category=category,
                current_macro_regime=self._current_macro_regime,
            )

        return max(0.0, min(1.0, round(confidence, 3)))

    def _bucket_adjust_confidence(self, confidence: float) -> float | None:
        """Adjust confidence using empirical calibration buckets.

        Returns adjusted confidence, or None if no bucket data available.
        """
        buckets = self._forecast_meta.get("calibration_buckets", [])
        if not buckets:
            return None

        # Find matching bucket
        for bucket in buckets:
            lo = bucket.get("bucket_lower", 0)
            hi = bucket.get("bucket_upper", 1)
            count = bucket.get("prediction_count", 0)
            in_range = (lo <= confidence < hi) if hi < 1.0 else (lo <= confidence <= hi)
            if in_range and count >= 5:
                gap = bucket.get("gap", 0)
                if abs(gap) > 0.1:
                    observed = bucket.get("observed_accuracy", confidence)
                    # Blend: 70% original, 30% observed accuracy
                    return confidence * 0.7 + observed * 0.3
                return confidence  # Bucket matched, gap small — preserve original

        return None  # No reliable bucket matched

    def _validate_portfolio_proposals(
        self,
        proposals: list,
    ) -> tuple[list, list["BlockedPortfolioProposal"]]:
        """Validate portfolio proposals against hard guardrails.

        Returns (approved, blocked).
        """
        approved: list = []
        blocked: list[BlockedPortfolioProposal] = []

        # Global gate: block ALL portfolio proposals if prediction accuracy < 40%
        overall_acc = self._prediction_accuracy.get("overall_accuracy")
        if overall_acc is not None and overall_acc < self._MIN_PREDICTION_ACCURACY:
            for p in proposals:
                blocked.append(BlockedPortfolioProposal(
                    proposal=p,
                    reason=(
                        f"Blocked: overall prediction accuracy {overall_acc:.0%} "
                        f"< {self._MIN_PREDICTION_ACCURACY:.0%} minimum"
                    ),
                ))
            return approved, blocked

        for proposal in proposals:
            block_reason = self._check_portfolio_guardrails(proposal)
            if block_reason:
                blocked.append(BlockedPortfolioProposal(
                    proposal=proposal, reason=block_reason,
                ))
            else:
                approved.append(proposal)

        return approved, blocked

    def _validate_engine(self, suggestion: AgentSuggestion) -> bool:
        """Check if the engine tag in a suggestion exists in the strategy's sub_engines.

        Returns True if valid (or if we can't check), False if definitively invalid.
        """
        registry = self._strategy_registry
        if not registry or not suggestion.engine:
            return True

        bot_id = suggestion.bot_id
        engine_upper = suggestion.engine.upper()

        # Check all strategies for this bot
        strategies_for_bot = {}
        if hasattr(registry, "strategies_for_bot"):
            strategies_for_bot = registry.strategies_for_bot(bot_id)
        elif hasattr(registry, "strategies"):
            strategies_for_bot = {
                sid: p for sid, p in registry.strategies.items()
                if getattr(p, "bot_id", "") == bot_id
            }

        if not strategies_for_bot:
            return True  # Can't verify — allow through

        for _sid, profile in strategies_for_bot.items():
            sub_engines = getattr(profile, "sub_engines", []) or []
            for eng in sub_engines:
                if eng.upper() == engine_upper:
                    return True

        return False

    def _is_crypto_bot(self, bot_id: str) -> bool:
        """Check if bot_id is associated with crypto perpetual strategies."""
        registry = self._strategy_registry
        if not registry or not hasattr(registry, "strategies"):
            return False
        for _sid, profile in registry.strategies.items():
            if (
                getattr(profile, "bot_id", "") == bot_id
                and getattr(profile, "asset_class", "") == "crypto_perpetual"
            ):
                return True
        return False

    def _check_crypto_safety_gates(self, suggestion: AgentSuggestion) -> str | None:
        """Apply crypto-specific safety gates. Returns block reason or None."""
        target = (suggestion.target_param or "").lower()

        # Gate 1: Leverage cap changes require 60d evidence + high confidence
        if target in ("max_leverage_major", "max_leverage_alt"):
            evidence = suggestion.evidence_summary or ""
            conf = suggestion.confidence
            # Check for evidence_days mention in evidence text
            has_sufficient_evidence = "60" in evidence or "90" in evidence
            if conf < 0.9 or not has_sufficient_evidence:
                return (
                    f"Leverage cap change on '{target}' requires confidence >= 0.9 "
                    f"and 60+ days of evidence (got confidence={conf:.2f})"
                )

        # Gate 2: Funding threshold loosening requires evidence of blocked profitable trades
        if target in ("funding_extreme", "funding_extreme_threshold"):
            proposed = suggestion.proposed_value
            if proposed is not None:
                evidence = (suggestion.evidence_summary or "").lower()
                if "blocked" not in evidence and "profitable" not in evidence:
                    return (
                        "Changing funding extreme threshold requires evidence "
                        "of profitable trades blocked by the current threshold"
                    )

        # Gate 3: Risk-pct on crypto always treated as safety-critical (10% threshold)
        if target in ("risk_pct_a_plus", "risk_pct_a", "risk_pct_b"):
            # Enforce that crypto risk_pct suggestions have strong evidence
            conf = suggestion.confidence
            if conf < 0.7:
                return (
                    f"Crypto perpetual risk_pct change requires confidence >= 0.7 "
                    f"(got {conf:.2f}) — leverage amplifies risk sizing errors"
                )

        return None

    def _check_portfolio_guardrails(self, proposal) -> str | None:
        """Check a single portfolio proposal against all hard guardrails.

        Returns block reason string, or None if proposal passes.
        """
        proposal_type = getattr(proposal, "proposal_type", "")
        ptype = proposal_type.value if hasattr(proposal_type, "value") else str(proposal_type)
        current = getattr(proposal, "current_config", {}) or {}
        proposed = getattr(proposal, "proposed_config", {}) or {}
        getattr(proposal, "evidence_summary", "") or ""
        obs_window = getattr(proposal, "observation_window_days", 0) or 0
        confidence = getattr(proposal, "confidence", 0.5) or 0.5

        # Confidence floor — reject very low confidence portfolio proposals
        if confidence < self._MIN_PORTFOLIO_CONFIDENCE:
            return (
                f"Portfolio proposal confidence {confidence:.2f} "
                f"below {self._MIN_PORTFOLIO_CONFIDENCE:.2f} minimum"
            )

        # --- Allocation rebalance guardrails ---
        if ptype == "allocation_rebalance":
            # Check each family for max change and minimum floor
            for family, new_weight in proposed.items():
                old_weight = current.get(family)
                if old_weight is None:
                    continue
                change = abs(new_weight - old_weight)
                if change > self._MAX_ALLOC_CHANGE_PCT:
                    return (
                        f"Allocation change for '{family}' is {change:.0%}, "
                        f"exceeds {self._MAX_ALLOC_CHANGE_PCT:.0%} max per cycle"
                    )
                if new_weight < self._MIN_ALLOC_FLOOR:
                    return (
                        f"Proposed allocation for '{family}' is {new_weight:.0%}, "
                        f"below {self._MIN_ALLOC_FLOOR:.0%} minimum floor"
                    )
            # Evidence minimum: 60 days
            if obs_window < self._MIN_EVIDENCE_ALLOC:
                return (
                    f"Allocation proposal observation_window_days={obs_window}, "
                    f"requires minimum {self._MIN_EVIDENCE_ALLOC} days"
                )

        # --- Risk cap change guardrails ---
        elif ptype == "risk_cap_change":
            old_cap = current.get("heat_cap_R", 0)
            new_cap = proposed.get("heat_cap_R", 0)
            if old_cap and new_cap:
                change_pct = abs(new_cap - old_cap) / old_cap
                if change_pct > self._MAX_RISK_CAP_CHANGE_PCT:
                    return (
                        f"Risk cap change {change_pct:.0%} exceeds "
                        f"{self._MAX_RISK_CAP_CHANGE_PCT:.0%} maximum"
                    )
            if obs_window < self._MIN_EVIDENCE_RISK:
                return (
                    f"Risk cap proposal observation_window_days={obs_window}, "
                    f"requires minimum {self._MIN_EVIDENCE_RISK} days"
                )

        # --- Drawdown tier change guardrails ---
        elif ptype == "drawdown_tier_change":
            old_tiers = current.get("drawdown_tiers", [])
            new_tiers = proposed.get("drawdown_tiers", [])

            # Block tier removal entirely
            if len(new_tiers) < len(old_tiers):
                return "Drawdown tier removal is blocked — tiers can only narrow"

            # Block loosening (higher thresholds or higher multipliers)
            for i, (old_t, new_t) in enumerate(zip(old_tiers, new_tiers, strict=False)):
                old_thresh = old_t[0] if isinstance(old_t, (list, tuple)) else 0
                new_thresh = new_t[0] if isinstance(new_t, (list, tuple)) else 0
                old_mult = old_t[1] if isinstance(old_t, (list, tuple)) and len(old_t) > 1 else 1.0
                new_mult = new_t[1] if isinstance(new_t, (list, tuple)) and len(new_t) > 1 else 1.0

                if new_thresh > old_thresh:
                    return (
                        f"Drawdown tier {i} threshold loosened "
                        f"({old_thresh} → {new_thresh}) — only narrowing allowed"
                    )
                if new_mult > old_mult:
                    return (
                        f"Drawdown tier {i} multiplier loosened "
                        f"({old_mult} → {new_mult}) — only tightening allowed"
                    )

            if obs_window < self._MIN_EVIDENCE_RISK:
                return (
                    f"Drawdown tier proposal observation_window_days={obs_window}, "
                    f"requires minimum {self._MIN_EVIDENCE_RISK} days"
                )

        # --- Coordination change --- (lighter guardrails, mainly evidence check)
        elif ptype == "coordination_change":
            if obs_window < self._MIN_EVIDENCE_ALLOC:
                return (
                    f"Coordination proposal observation_window_days={obs_window}, "
                    f"requires minimum {self._MIN_EVIDENCE_ALLOC} days"
                )

        else:
            # Unknown proposal type — block by default for safety
            return f"Unknown portfolio proposal type '{ptype}' — blocked"

        return None
