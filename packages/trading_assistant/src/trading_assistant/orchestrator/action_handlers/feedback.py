"""Feedback, validation, and learning-card action support."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)


class FeedbackActions:
    """Feedback, validation, and learning-card action support."""

    async def handle_feedback(self, action: Action) -> None:
        """Process user feedback from Telegram/Discord callbacks."""
        details = action.details or {}
        text = details.get("text", "")
        report_id = details.get("report_id", "unknown")
        if not text:
            return

        from trading_assistant.analysis.feedback_handler import FeedbackHandler, UnsafeFeedbackError

        handler = FeedbackHandler(report_id=report_id)
        try:
            correction = handler.parse(text, source="handler")
        except UnsafeFeedbackError as exc:
            logger.warning("Rejected unsafe feedback for %s: %s", report_id, exc)
            self._event_stream.broadcast("feedback_rejected", {
                "report_id": report_id,
                "reason": str(exc),
            })
            return
        corrections_path = self._memory_dir / "findings" / "corrections.jsonl"
        handler.write_correction(correction, corrections_path)

        # Record allocation changes when user approves allocation recommendations
        from trading_assistant.schemas.corrections import CorrectionType

        if correction.correction_type == CorrectionType.ALLOCATION_CHANGE:
            self._record_allocation_change(correction, details)

        # Route suggestion accept/reject to SuggestionTracker
        if self._suggestion_tracker and correction.target_id:
            if correction.correction_type == CorrectionType.SUGGESTION_ACCEPT:
                accepted = False
                if self._approval_tracker and self._approval_handler:
                    pending = self._approval_tracker.find_pending_for_suggestion(
                        correction.target_id,
                    )
                    if pending is not None:
                        await self._approval_handler.handle_approve(pending.request_id)
                        updated = self._approval_tracker.get_by_id(pending.request_id)
                        accepted = (
                            updated is not None
                            and updated.status.value == "APPROVED"
                        )
                if not accepted and not (
                    self._approval_tracker
                    and self._approval_tracker.find_pending_for_suggestion(correction.target_id)
                ):
                    self._suggestion_tracker.accept(correction.target_id)
                    accepted = True
                if accepted:
                    self._event_stream.broadcast("suggestion_accepted", {
                        "suggestion_id": correction.target_id,
                    })
                    self._update_hypothesis_from_feedback(
                        correction.target_id, accepted=True,
                    )
            elif correction.correction_type == CorrectionType.SUGGESTION_REJECT:
                routed = False
                if self._approval_tracker and self._approval_handler:
                    pending = self._approval_tracker.find_pending_for_suggestion(
                        correction.target_id,
                    )
                    if pending is not None:
                        await self._approval_handler.handle_reject(
                            pending.request_id,
                            reason=text[:200],
                        )
                        routed = True
                if not routed:
                    self._suggestion_tracker.reject(correction.target_id, text[:200])
                self._event_stream.broadcast("suggestion_rejected", {
                    "suggestion_id": correction.target_id,
                })
                self._update_hypothesis_from_feedback(
                    correction.target_id, accepted=False,
                )

        await self._notify(
            notification_type="feedback_received",
            priority=NotificationPriority.LOW,
            title="Feedback recorded",
            body=f"Correction type: {correction.correction_type.value}",
        )

    def _record_allocation_change(self, correction, details: dict) -> None:
        """Persist an approved allocation change via AllocationTracker."""
        try:
            from trading_assistant.skills.allocation_tracker import AllocationTracker
            from trading_assistant.schemas.allocation_history import AllocationRecord, AllocationSource

            tracker = AllocationTracker(self._memory_dir / "findings")

            # Extract allocation details from the action payload if provided
            allocations = details.get("allocations", [])
            date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

            if allocations:
                for alloc in allocations:
                    tracker.record(AllocationRecord(
                        date=date,
                        bot_id=alloc.get("bot_id", ""),
                        strategy_id=alloc.get("strategy_id", ""),
                        allocation_pct=alloc.get("allocation_pct", 0.0),
                        source=AllocationSource.MANUAL,
                        reason=f"Approved via feedback: {correction.raw_text[:100]}",
                    ))
            else:
                # Record the approval event even without specific allocations
                tracker.record(AllocationRecord(
                    date=date,
                    bot_id=details.get("bot_id", "unknown"),
                    allocation_pct=0.0,
                    source=AllocationSource.MANUAL,
                    reason=f"Allocation approved: {correction.raw_text[:200]}",
                ))
        except Exception:
            logger.error("Failed to record allocation change from feedback", exc_info=True)

    def _extract_and_record_patterns(
        self, parsed, bots: list[str], suggestion_ids: dict[str, str] | None = None,
    ) -> None:
        """Extract patterns from structural proposals and record in PatternLibrary."""
        if not parsed.structural_proposals:
            return
        try:
            from trading_assistant.schemas.pattern_library import PatternCategory
            from trading_assistant.skills.pattern_library import PatternLibrary, PatternEntry, PatternStatus

            lib = PatternLibrary(self._memory_dir / "findings")

            # Map proposal categories to PatternCategory
            category_map = {
                "signal": PatternCategory.ENTRY_SIGNAL,
                "signal_decay": PatternCategory.ENTRY_SIGNAL,
                "filter": PatternCategory.FILTER,
                "filter_over_blocking": PatternCategory.FILTER,
                "exit_timing": PatternCategory.EXIT_RULE,
                "exit": PatternCategory.EXIT_RULE,
                "adverse_fills": PatternCategory.RISK_MANAGEMENT,
                "regime_breakdown": PatternCategory.REGIME_GATE,
                "regime": PatternCategory.REGIME_GATE,
                "correlation_crowding": PatternCategory.COORDINATION,
                "position_sizing": PatternCategory.POSITION_SIZING,
                "structural": PatternCategory.ENTRY_SIGNAL,
            }

            existing = lib.load_all()
            existing_titles = {e.title for e in existing}

            for proposal in parsed.structural_proposals:
                # Dedup by title
                if proposal.title in existing_titles:
                    continue

                # Determine category from proposal fields
                raw_cat = ""
                if proposal.hypothesis_id:
                    # Try to infer from hypothesis category
                    try:
                        from trading_assistant.skills.hypothesis_library import HypothesisLibrary
                        hyp_lib = HypothesisLibrary(self._memory_dir / "findings")
                        for h in hyp_lib.get_all_records():
                            if h.id == proposal.hypothesis_id:
                                raw_cat = h.category
                                break
                    except Exception:
                        pass
                if not raw_cat:
                    # Infer from title keywords
                    title_lower = proposal.title.lower()
                    for keyword, cat_str in [
                        ("filter", "filter"), ("exit", "exit"),
                        ("signal", "signal"), ("regime", "regime"),
                        ("sizing", "position_sizing"), ("stop", "exit"),
                    ]:
                        if keyword in title_lower:
                            raw_cat = cat_str
                            break

                cat = category_map.get(raw_cat, PatternCategory.ENTRY_SIGNAL)
                target_bots = [b for b in bots if b != proposal.bot_id]

                linked_sid = proposal.linked_suggestion_id or ""

                entry = PatternEntry(
                    title=proposal.title,
                    category=cat,
                    status=PatternStatus.PROPOSED,
                    source_bot=proposal.bot_id,
                    target_bots=target_bots,
                    description=proposal.description,
                    evidence=proposal.evidence,
                    linked_suggestion_id=linked_sid,
                )
                lib.add(entry)
                existing_titles.add(proposal.title)
                logger.info("Recorded pattern from structural proposal: %s", proposal.title)
        except Exception:
            logger.error("Failed to extract and record patterns", exc_info=True)

    def _update_hypothesis_from_feedback(
        self, suggestion_id: str, accepted: bool,
    ) -> None:
        """Update HypothesisLibrary when a suggestion linked to a hypothesis is accepted/rejected."""
        if not self._suggestion_tracker:
            return
        try:
            from trading_assistant.skills.hypothesis_library import HypothesisLibrary

            # Find the suggestion to get its hypothesis_id
            all_suggestions = self._suggestion_tracker.load_all()
            hypothesis_id = None
            for s in all_suggestions:
                if s.get("suggestion_id") == suggestion_id:
                    hypothesis_id = s.get("hypothesis_id")
                    break

            if not hypothesis_id:
                return

            lib = HypothesisLibrary(self._memory_dir / "findings")
            if accepted:
                lib.record_acceptance(hypothesis_id)
            else:
                lib.record_rejection(hypothesis_id)
        except Exception:
            logger.error("Failed to update hypothesis lifecycle for suggestion %s", suggestion_id, exc_info=True)

    def _validate_and_annotate(
        self,
        parsed,
        date_or_week: str,
        provider: str = "",
        model: str = "",
        run_id: str = "",
        agent_type: str = "",
        bot_ids: str | list[str] | None = None,
    ):
        """Run response validation and return annotated report text + validation result.

        Returns:
            tuple[str, ValidationResult | None]: (annotated_report, validation_result)
        """
        try:
            from trading_assistant.analysis.response_validator import ResponseValidator
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer

            ctx = ContextBuilder(self._memory_dir, curated_dir=self._curated_dir)
            rejected = ctx.load_rejected_suggestions()
            forecast_meta = ctx.load_forecast_meta()

            scorer = SuggestionScorer(self._memory_dir / "findings")
            scorecard = scorer.compute_scorecard()

            hypothesis_tr = ctx.load_hypothesis_track_record()
            prediction_accuracy = ctx.load_prediction_accuracy()
            recalibrations = ctx.load_recalibrations()
            if isinstance(bot_ids, str):
                prior_bot_id = bot_ids
            elif isinstance(bot_ids, list) and len(bot_ids) == 1:
                prior_bot_id = bot_ids[0]
            else:
                prior_bot_id = ""
            outcome_priors = ctx.load_outcome_priors(bot_id=prior_bot_id)

            # Load current macro regime for confidence adjustment
            macro_regime_ctx = ctx.load_macro_regime_context()
            current_macro_regime = macro_regime_ctx.get("macro_regime", "") if macro_regime_ctx else ""

            validator = ResponseValidator(
                rejected_suggestions=rejected,
                forecast_meta=forecast_meta,
                category_scorecard=scorecard,
                hypothesis_track_record=hypothesis_tr,
                prediction_accuracy=prediction_accuracy,
                recalibrations=recalibrations,
                outcome_priors=outcome_priors,
                current_macro_regime=current_macro_regime,
                strategy_registry=self._strategy_registry,
            )
            validation = validator.validate(parsed)

            final_report = parsed.raw_report
            if validation.validator_notes:
                final_report += "\n\n---\n## Validator Notes\n" + validation.validator_notes

            # Log validation results (with blocked details for learning signal)
            try:
                log_path = self._memory_dir / "findings" / "validation_log.jsonl"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                import json as _json
                blocked_details = [
                    {
                        "title": b.suggestion.title,
                        "reason": b.reason,
                        "bot_id": b.suggestion.bot_id,
                        "category": getattr(b.suggestion, "category", "") or "",
                    }
                    for b in validation.blocked_suggestions
                ]
                blocked_proposal_details = [
                    {
                        "title": getattr(b.proposal, "title", ""),
                        "reason": b.reason,
                        "bot_id": getattr(b.proposal, "bot_id", "") or "",
                        "hypothesis_id": getattr(b.proposal, "hypothesis_id", "") or "",
                    }
                    for b in validation.blocked_structural_proposals
                ]
                blocked_portfolio_details = [
                    {
                        "title": getattr(b.proposal, "title", "")
                        or getattr(b.proposal, "proposal_type", ""),
                        "reason": b.reason,
                    }
                    for b in validation.blocked_portfolio_proposals
                ]
                entry = {
                    "date": date_or_week,
                    "approved_count": len(validation.approved_suggestions),
                    "blocked_count": len(validation.blocked_suggestions),
                    "blocked_details": blocked_details,
                    "approved_proposal_count": len(validation.approved_structural_proposals),
                    "blocked_proposal_count": len(validation.blocked_structural_proposals),
                    "blocked_proposal_details": blocked_proposal_details,
                    "approved_portfolio_count": len(validation.approved_portfolio_proposals),
                    "blocked_portfolio_count": len(validation.blocked_portfolio_proposals),
                    "blocked_portfolio_details": blocked_portfolio_details,
                    "provider": provider,
                    "model": model,
                    "run_id": run_id,
                    "agent_type": agent_type,
                    "bot_ids": bot_ids or "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(entry) + "\n")
            except Exception:
                logger.error("Failed to log validation results", exc_info=True)

            self._record_rejected_validation_proposals(
                validation=validation,
                run_id=run_id,
                source="llm_daily" if agent_type == "daily_analysis" else "llm_weekly",
            )

            return final_report, validation
        except Exception:
            logger.warning("Response validation failed - using raw report")
            return parsed.raw_report, None

    def _persist_validator_notes(self, run_dir: Path, validation_result) -> None:
        """Persist validator notes as a run artifact for RunIndex FTS."""
        if validation_result is None:
            return
        notes = getattr(validation_result, "validator_notes", "") or ""
        if not notes:
            return
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            (Path(run_dir) / "validator_notes.md").write_text(
                notes,
                encoding="utf-8",
            )
        except Exception:
            logger.error("Failed to persist validator notes for %s", run_dir, exc_info=True)

    def _refresh_run_index_entry(
        self,
        *,
        run_id: str,
        agent_type: str,
        run_dir: Path,
        provider: str = "",
        model: str = "",
        prompt_package=None,
        success: bool = True,
        duration_ms: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Refresh RunIndex after handlers persist parsed output and validator notes."""
        try:
            self._agent_runner.refresh_run_index(
                run_id=run_id,
                agent_type=agent_type,
                run_dir=Path(run_dir),
                provider=provider,
                model=model,
                prompt_package=prompt_package,
                success=success,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
            )
        except Exception:
            logger.debug("Failed to refresh RunIndex entry for %s", run_id)

    def _record_learning_card_feedback(
        self, validation_result, prompt_package,
    ) -> None:
        """Compatibility shim for older callers; delegates to targeted feedback."""
        self._record_learning_card_feedback_targeted(validation_result, prompt_package)

    def _record_learning_card_feedback_targeted(
        self, validation_result, prompt_package,
    ) -> None:
        """Record feedback only for retrieved cards that match validated tags."""
        if validation_result is None:
            return
        card_ids = (prompt_package.metadata or {}).get("_learning_card_ids")
        if not card_ids:
            return

        approved_tags = self._feedback_structured_tags(validation_result.approved_suggestions)
        blocked_tags = self._feedback_structured_tags(
            [b.suggestion for b in validation_result.blocked_suggestions],
            reasons=[b.reason for b in validation_result.blocked_suggestions],
        )
        if not approved_tags and not blocked_tags:
            return

        try:
            from trading_assistant.skills.learning_card_store import LearningCardStore

            store = LearningCardStore(self._memory_dir / "findings")
            index = store.load()
            for card_id in card_ids:
                card = index.get(card_id)
                if card is None:
                    continue
                card_tags = set(card.tags)
                matched_approved = bool(card_tags & approved_tags)
                matched_blocked = bool(card_tags & blocked_tags)
                if matched_approved and not matched_blocked:
                    index.record_feedback(card_id, True)
                elif matched_blocked and not matched_approved:
                    index.record_feedback(card_id, False)
            store.save(index)
        except Exception:
            logger.error("Failed to record targeted learning card feedback", exc_info=True)

    @staticmethod
    def _feedback_structured_tags(suggestions: list, reasons: list[str] | None = None) -> set[str]:
        tags: set[str] = set()
        for suggestion in suggestions or []:
            category = str(getattr(suggestion, "category", "") or "").strip()
            if category:
                tag = FeedbackActions._feedback_tag("category", category)
                if tag:
                    tags.add(tag)
        for reason in reasons or []:
            tag = FeedbackActions._feedback_tag("reason", reason)
            if tag:
                tags.add(tag)
        return tags

    @staticmethod
    def _feedback_tag(prefix: str, value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        chars: list[str] = []
        prev_sep = False
        for char in text:
            if char.isalnum():
                chars.append(char)
                prev_sep = False
            elif not prev_sep:
                chars.append("_")
                prev_sep = True
        slug = "".join(chars).strip("_")
        return f"{prefix}:{slug}" if slug else ""

    def _record_predictions(self, date_or_week: str, predictions: list) -> None:
        """Record structured predictions from a parsed response."""
        if not predictions:
            return
        try:
            from trading_assistant.skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            tracker.record_predictions(date_or_week, predictions)
        except Exception:
            logger.error("Failed to record predictions for %s", date_or_week, exc_info=True)

    def _update_hypothesis_lifecycle(self, parsed, suggestion_ids: dict) -> None:
        """Update hypothesis lifecycle based on parsed structural proposals."""
        if not parsed.structural_proposals:
            return
        try:
            from trading_assistant.skills.hypothesis_library import HypothesisLibrary

            lib = HypothesisLibrary(self._memory_dir / "findings")
            for proposal in parsed.structural_proposals:
                if proposal.hypothesis_id:
                    lib.record_proposal(proposal.hypothesis_id)
        except Exception:
            logger.error("Failed to update hypothesis lifecycle", exc_info=True)
