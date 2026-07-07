"""Suggestion and proposal ledger write support."""

from __future__ import annotations

import logging

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ProposalLedgerActions:
    """Suggestion and proposal ledger write support."""

    def _ledger_write_candidate(
        self,
        *,
        source: str,
        kind_hint: str,
        bot_id: str,
        title: str,
        description: str = "",
        suggestion_id: str = "",
        experiment_id: str = "",
        deployment_id: str = "",
        hypothesis_id: str = "",
        strategy_id: str = "",
        stable_link_key: str = "",
        run_id: str = "",
        detector_name: str = "",
        evaluation_method: str = "",
        lifecycle_stage: str = "",
        affected_parameters: list[str] | None = None,
        affected_files: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
    ) -> str:
        """Write a ProposalCandidate to the unified ledger.

        Returns the proposal_id (empty string if no ledger configured). This
        helper centralizes ledger writes so each proposal-producing handler
        site can insert one consistent record.
        """
        # Use getattr for resilience: tests sometimes construct via __new__
        # without going through __init__.
        ledger = getattr(self, "_proposal_ledger", None)
        if not ledger:
            return ""
        try:
            from trading_assistant.schemas.proposal_ledger import (
                ProposalCandidate,
                ProposalKind,
                ProposalSource,
            )
            from trading_assistant.skills.proposal_ledger import make_proposal_id
        except Exception:
            logger.debug("ProposalLedger schemas unavailable", exc_info=True)
            return ""

        try:
            src = ProposalSource(source)
        except ValueError:
            src = ProposalSource.DETERMINISTIC

        kind_map = {
            "parameter": ProposalKind.PARAMETER_CHANGE,
            "filter": ProposalKind.PARAMETER_CHANGE,
            "parameter_change": ProposalKind.PARAMETER_CHANGE,
            "strategy_variant": ProposalKind.STRUCTURAL_CHANGE,
            "hypothesis": ProposalKind.STRUCTURAL_CHANGE,
            "structural": ProposalKind.STRUCTURAL_CHANGE,
            "structural_change": ProposalKind.STRUCTURAL_CHANGE,
            "new_strategy": ProposalKind.NEW_STRATEGY,
            "portfolio": ProposalKind.PORTFOLIO_CHANGE,
            "portfolio_change": ProposalKind.PORTFOLIO_CHANGE,
            "search_space_change": ProposalKind.SEARCH_SPACE_CHANGE,
            "instrumentation_request": ProposalKind.INSTRUMENTATION_REQUEST,
            "bug_fix": ProposalKind.BUG_FIX,
        }
        kind = kind_map.get(kind_hint.lower(), ProposalKind.PARAMETER_CHANGE)

        link_key = stable_link_key or suggestion_id or experiment_id or deployment_id
        proposal_id = make_proposal_id(
            src,
            bot_id,
            kind,
            title or "untitled",
            strategy_id=strategy_id or "",
            link_key=link_key or "",
        )
        candidate = ProposalCandidate(
            proposal_id=proposal_id,
            source=src,
            kind=kind,
            bot_id=bot_id,
            strategy_id=strategy_id or "",
            title=title or "untitled",
            description=description or "",
            hypothesis_id=hypothesis_id or "",
            suggestion_id=suggestion_id or "",
            experiment_id=experiment_id or "",
            deployment_id=deployment_id or "",
            linked_run_id=run_id or "",
            evaluation_method=evaluation_method or "",
            lifecycle_stage=lifecycle_stage or "",
            linked_diagnostics=[detector_name] if detector_name else [],
            affected_parameters=affected_parameters or [],
            affected_files=affected_files or [],
            acceptance_criteria=acceptance_criteria or [],
        )
        try:
            ledger.record_candidate(candidate)
        except Exception:
            logger.warning("Failed to record proposal candidate", exc_info=True)
        return proposal_id

    def _ledger_write_evaluation(
        self,
        proposal_id: str,
        method: str,
        decision: str,
        decision_reason: str = "",
        objective_score: float = 0.0,
        confidence: float = 0.0,
        summary: str = "",
    ) -> None:
        ledger = getattr(self, "_proposal_ledger", None)
        if not ledger or not proposal_id:
            return
        try:
            from trading_assistant.schemas.proposal_ledger import ProposalEvaluation
            ledger.record_evaluation(
                proposal_id,
                ProposalEvaluation(
                    proposal_id=proposal_id,
                    method=method,
                    decision=decision,
                    decision_reason=decision_reason,
                    objective_score=objective_score,
                    confidence=confidence,
                    summary=summary,
                ),
            )
        except Exception:
            logger.warning("Failed to record proposal evaluation", exc_info=True)

    def _record_rejected_validation_proposals(
        self,
        validation,
        run_id: str,
        source: str,
    ) -> None:
        """Mirror validator-rejected proposals into ProposalLedger."""
        if validation is None or not getattr(self, "_proposal_ledger", None):
            return
        try:
            from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER
        except Exception:
            CATEGORY_TO_TIER = {}

        for idx, blocked in enumerate(getattr(validation, "blocked_suggestions", []) or []):
            suggestion = blocked.suggestion
            category = getattr(suggestion, "category", "") or ""
            tier = CATEGORY_TO_TIER.get(category, "parameter")
            strategy_id = getattr(suggestion, "strategy_id", None) or ""
            suggestion_id = getattr(suggestion, "suggestion_id", "") or ""
            proposal_id = self._ledger_write_candidate(
                source=source,
                kind_hint="structural_change" if tier in ("strategy_variant", "hypothesis") else tier,
                bot_id=getattr(suggestion, "bot_id", "") or "",
                strategy_id=strategy_id,
                title=getattr(suggestion, "title", "") or "rejected suggestion",
                description=getattr(suggestion, "evidence_summary", "") or "",
                suggestion_id=suggestion_id,
                run_id=run_id,
                evaluation_method="validator",
                affected_parameters=[
                    getattr(suggestion, "target_param", "") or ""
                ] if getattr(suggestion, "target_param", None) else [],
                stable_link_key=(
                    suggestion_id or
                    f"rejected:suggestion:{run_id}:{idx}:{getattr(suggestion, 'title', '')}"
                ),
            )
            self._ledger_write_evaluation(
                proposal_id,
                method="validator",
                decision="reject",
                decision_reason=getattr(blocked, "reason", "") or "",
                confidence=float(getattr(suggestion, "confidence", 0.0) or 0.0),
                summary="Validator rejected LLM suggestion",
            )

        for idx, blocked in enumerate(getattr(validation, "blocked_structural_proposals", []) or []):
            proposal = blocked.proposal
            linked_suggestion_id = getattr(proposal, "linked_suggestion_id", "") or ""
            proposal_id = self._ledger_write_candidate(
                source=source,
                kind_hint="structural_change",
                bot_id=getattr(proposal, "bot_id", "") or "",
                title=getattr(proposal, "title", "") or "rejected structural proposal",
                description=getattr(proposal, "description", "") or getattr(proposal, "evidence", "") or "",
                suggestion_id=linked_suggestion_id,
                hypothesis_id=getattr(proposal, "hypothesis_id", "") or "",
                run_id=run_id,
                evaluation_method="validator",
                acceptance_criteria=[
                    str(c) for c in (getattr(proposal, "acceptance_criteria", []) or [])
                ],
                stable_link_key=(
                    linked_suggestion_id or
                    getattr(proposal, "hypothesis_id", "") or
                    f"rejected:structural:{run_id}:{idx}:{getattr(proposal, 'title', '')}"
                ),
            )
            self._ledger_write_evaluation(
                proposal_id,
                method="validator",
                decision="reject",
                decision_reason=getattr(blocked, "reason", "") or "",
                confidence=float(getattr(proposal, "confidence", 0.0) or 0.0),
                summary="Validator rejected structural proposal",
            )

        for idx, blocked in enumerate(getattr(validation, "blocked_portfolio_proposals", []) or []):
            proposal = blocked.proposal
            ptype = getattr(proposal, "proposal_type", "") or ""
            ptype_str = ptype.value if hasattr(ptype, "value") else str(ptype)
            proposal_id = self._ledger_write_candidate(
                source=source,
                kind_hint="portfolio_change",
                bot_id="PORTFOLIO",
                title=f"Portfolio: {ptype_str or 'rejected'}",
                description=getattr(proposal, "evidence_summary", "") or "",
                run_id=run_id,
                evaluation_method="validator",
                acceptance_criteria=[
                    str(c) for c in (getattr(proposal, "acceptance_criteria", []) or [])
                ],
                stable_link_key=f"rejected:portfolio:{run_id}:{idx}:{ptype_str}",
            )
            self._ledger_write_evaluation(
                proposal_id,
                method="validator",
                decision="reject",
                decision_reason=getattr(blocked, "reason", "") or "",
                confidence=float(getattr(proposal, "confidence", 0.0) or 0.0),
                summary="Validator rejected portfolio proposal",
            )

    def _record_suggestions(
        self, suggestions: list, run_id: str,
        category_scorecard=None,
    ) -> dict[str, str]:
        """Convert StrategySuggestions to SuggestionRecords and persist via tracker.

        Applies scorecard pre-validation: skips suggestions in categories with
        poor track records (win_rate < 0.3, sample_size >= 5) to prevent
        category leakage when scorecard was unavailable during build_report().

        Returns a mapping of suggestion_id - title for metadata injection.
        """
        if not self._suggestion_tracker or not suggestions:
            return {}

        import hashlib
        from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER as _C2T
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(suggestions):
            title = getattr(suggestion, "title", "") or ""
            bot_id = getattr(suggestion, "bot_id", "") or ""
            tier = getattr(suggestion, "tier", "parameter")
            description = getattr(suggestion, "description", "") or ""

            # Pre-validation: skip suggestions in categories with poor track record
            if category_scorecard is not None:
                tier_val = str(tier.value) if hasattr(tier, "value") else str(tier)
                _skip = False
                for _score in getattr(category_scorecard, "scores", []):
                    if _score.bot_id != bot_id:
                        continue
                    _cat_tier = _C2T.get(_score.category, _score.category)
                    if _cat_tier == tier_val and _score.sample_size >= 5 and _score.win_rate < 0.3:
                        logger.info(
                            "Skipping strategy suggestion in poor category %s/%s (win_rate=%.0f%%, n=%d)",
                            bot_id, _score.category, _score.win_rate * 100, _score.sample_size,
                        )
                        _skip = True
                        break
                if _skip:
                    continue

            # Deterministic ID: SHA256(run_id + index + title)[:12]
            raw = f"{run_id}:{idx}:{title}"
            suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            category_str = str(getattr(suggestion, "category", "") or "")
            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)
            det_ctx = getattr(suggestion, "detection_context", None)
            det_ctx_dict = None
            if isinstance(det_ctx, BaseModel):
                det_ctx_dict = det_ctx.model_dump(mode="json")
            # Derive category from detector_name when not set on suggestion
            if not category_str and det_ctx:
                from trading_assistant.analysis.strategy_engine import StrategyEngine
                _det_name = ""
                if isinstance(det_ctx, BaseModel):
                    _det_name = getattr(det_ctx, "detector_name", "")
                elif isinstance(det_ctx, dict):
                    _det_name = det_ctx.get("detector_name", "")
                if _det_name:
                    category_str = StrategyEngine._DETECTOR_TO_CATEGORY.get(_det_name, "")
            # Extract target_param and proposed_value from detection context
            target_param = None
            proposed_value = None
            expected_impact = ""
            if det_ctx:
                if isinstance(det_ctx, BaseModel):
                    target_param = getattr(det_ctx, "threshold_name", None)
                elif isinstance(det_ctx, dict):
                    target_param = det_ctx.get("threshold_name")
            raw_suggested = getattr(suggestion, "suggested_value", None)
            if raw_suggested is not None:
                try:
                    proposed_value = float(raw_suggested)
                except (TypeError, ValueError):
                    pass
            if description:
                expected_impact = description[:200]

            tier_val = str(tier.value) if hasattr(tier, "value") else str(tier)
            detector_name = ""
            if isinstance(det_ctx, BaseModel):
                detector_name = getattr(det_ctx, "detector_name", "") or ""
            elif isinstance(det_ctx, dict):
                detector_name = det_ctx.get("detector_name", "") or ""
            # Preserve strategy attribution that StrategySuggestion already carries.
            _raw_sid = getattr(suggestion, "strategy_id", None)
            strategy_id_val = _raw_sid if isinstance(_raw_sid, str) and _raw_sid else None
            proposal_id = self._ledger_write_candidate(
                source="deterministic",
                kind_hint=tier_val,
                bot_id=bot_id,
                strategy_id=strategy_id_val or "",
                title=title,
                description=description,
                suggestion_id=suggestion_id,
                run_id=run_id,
                detector_name=detector_name,
                evaluation_method="parameter_search" if tier_val in ("parameter", "filter") else "approval",
                affected_parameters=[target_param] if target_param else [],
            )

            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                strategy_id=strategy_id_val,
                title=title,
                tier=tier_val,
                category=category_str,
                source_report_id=run_id,
                description=description,
                confidence=confidence,
                target_param=target_param,
                proposed_value=proposed_value,
                expected_impact=expected_impact,
                detection_context=det_ctx_dict,
                proposal_id=proposal_id or None,
            )

            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:  # record() returns None (old) or bool (new)
                id_map[suggestion_id] = title
                logger.info("Recorded suggestion %s: %s", suggestion_id, title)

        if id_map:
            self._event_stream.broadcast("suggestions_recorded", {
                "run_id": run_id, "count": len(id_map),
            })

        return id_map

    def _record_agent_suggestions(
        self, validation_result, run_id: str, parsed=None,
        provider: str = "", model: str = "",
        source: str = "llm_weekly",
    ) -> dict[str, str]:
        """Record approved suggestions from validation into SuggestionTracker.

        Maps AgentSuggestion - SuggestionRecord with deterministic IDs and hypothesis linking.
        Only approved (not blocked) suggestions are recorded. ``source`` controls
        the ProposalSource label written to the unified ledger - daily handlers
        should pass ``"llm_daily"``.

        Returns a mapping of suggestion_id - title.
        """
        if validation_result is None:
            return {}
        approved_structural = (
            getattr(validation_result, "approved_structural_proposals", []) or []
        )
        blocked_structural = (
            getattr(validation_result, "blocked_structural_proposals", []) or []
        )
        if not approved_structural and not blocked_structural and parsed is not None:
            approved_structural = getattr(parsed, "structural_proposals", []) or []
        if not self._suggestion_tracker or not validation_result.approved_suggestions:
            if approved_structural:
                self._record_structural_experiments(approved_structural, run_id)
            return {}

        import hashlib
        from trading_assistant.schemas.suggestion_tracking import SuggestionRecord

        # Build structural proposal lookup by explicit linked suggestion id.
        structural_context_map: dict[str, dict] = {}
        fallback_context_by_bot: dict[str, list[dict]] = {}
        for proposal in approved_structural:
            context = {
                "notes": proposal.description,
                "file_changes": [
                    fc.model_dump(mode="json")
                    for fc in getattr(proposal, "file_changes", [])
                ],
                "verification_commands": list(
                    getattr(proposal, "verification_commands", []) or []
                ),
                "hypothesis_id": proposal.hypothesis_id,
            }
            if proposal.linked_suggestion_id:
                structural_context_map[proposal.linked_suggestion_id] = context
                continue
            fallback_context_by_bot.setdefault(proposal.bot_id or "", []).append(context)

        approved_counts_by_bot: dict[str, int] = {}
        for suggestion in validation_result.approved_suggestions:
            bot_id = suggestion.bot_id or ""
            approved_counts_by_bot[bot_id] = approved_counts_by_bot.get(bot_id, 0) + 1

        # Validate suggestions with backtesting before recording
        from trading_assistant.skills.suggestion_validator import SuggestionValidator

        validator = SuggestionValidator(curated_dir=self._curated_dir)

        id_map: dict[str, str] = {}
        for idx, suggestion in enumerate(validation_result.approved_suggestions):
            title = suggestion.title or ""
            bot_id = suggestion.bot_id or ""
            category = suggestion.category or "parameter"

            raw = f"{run_id}:agent:{idx}:{title}"
            existing_id = getattr(suggestion, "suggestion_id", "")
            suggestion_id = existing_id if isinstance(existing_id, str) and existing_id else hashlib.sha256(raw.encode()).hexdigest()[:12]

            # Map category to tier using shared mapping
            from trading_assistant.schemas.agent_response import CATEGORY_TO_TIER

            tier = CATEGORY_TO_TIER.get(category, "parameter")

            confidence = float(getattr(suggestion, "confidence", 0.0) or 0.0)

            # Run suggestion validation (backtest replay)
            validation_evidence = None
            val_result = None
            try:
                val_result = validator.validate(
                    suggestion_id=suggestion_id,
                    bot_id=bot_id,
                    category=category,
                    target_param=getattr(suggestion, "target_param", None),
                    proposed_value=getattr(suggestion, "proposed_value", None),
                    title=title,
                )
                validation_evidence = val_result.evidence.model_dump(mode="json")
                if val_result.degradation_detected:
                    logger.warning(
                        "Suggestion %s shows degradation in backtest: improvement=%.1f%%",
                        suggestion_id, val_result.evidence.improvement_pct,
                    )
            except Exception:
                logger.warning("Suggestion validation failed for %s - recording anyway", suggestion_id)

            structural_context = structural_context_map.get(suggestion_id)
            if structural_context is None:
                fallback_contexts = fallback_context_by_bot.get(bot_id, [])
                # Preserve precise explicit links, but still allow the legacy
                # one-suggestion/one-proposal hypothesis mapping for same-bot runs.
                if fallback_contexts and approved_counts_by_bot.get(bot_id, 0) == 1 and len(fallback_contexts) == 1:
                    structural_context = fallback_contexts[0]

            # Merge validation evidence into detection_context
            detection_ctx = {}
            if provider:
                detection_ctx["source_provider"] = provider
            if model:
                detection_ctx["source_model"] = model
            if validation_evidence:
                detection_ctx["validation_evidence"] = validation_evidence
            if val_result and val_result.requires_review:
                detection_ctx["requires_review"] = True

            # Extract target fields from AgentSuggestion
            _raw_tp = getattr(suggestion, "target_param", None)
            agent_target_param = str(_raw_tp) if isinstance(_raw_tp, str) else None
            agent_proposed_value = None
            raw_pv = getattr(suggestion, "proposed_value", None)
            if raw_pv is not None:
                try:
                    agent_proposed_value = float(raw_pv)
                except (TypeError, ValueError):
                    pass
            _raw_ei = getattr(suggestion, "expected_impact", None)
            agent_expected_impact = str(_raw_ei) if isinstance(_raw_ei, str) else ""

            agent_hypothesis_id = (
                structural_context.get("hypothesis_id")
                if structural_context is not None
                else None
            )
            _raw_agent_sid = getattr(suggestion, "strategy_id", None)
            agent_strategy_id = (
                _raw_agent_sid if isinstance(_raw_agent_sid, str) and _raw_agent_sid else None
            )
            ledger_proposal_id = self._ledger_write_candidate(
                source=source,
                kind_hint="structural_change" if tier in ("strategy_variant", "hypothesis") else tier,
                bot_id=bot_id,
                strategy_id=agent_strategy_id or "",
                title=title,
                description=suggestion.evidence_summary or "",
                suggestion_id=suggestion_id,
                run_id=run_id,
                hypothesis_id=agent_hypothesis_id or "",
                evaluation_method="approval",
                affected_parameters=[agent_target_param] if agent_target_param else [],
            )

            record = SuggestionRecord(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                strategy_id=agent_strategy_id,
                title=title,
                tier=tier,
                category=category,
                source_report_id=run_id,
                description=suggestion.evidence_summary or "",
                confidence=confidence,
                target_param=agent_target_param,
                proposed_value=agent_proposed_value,
                expected_impact=agent_expected_impact,
                hypothesis_id=agent_hypothesis_id,
                implementation_context=structural_context,
                proposal_id=ledger_proposal_id or None,
                detection_context=detection_ctx if detection_ctx else None,
            )

            recorded = self._suggestion_tracker.record(record)
            if recorded is not False:
                id_map[suggestion_id] = title
                logger.info("Recorded agent suggestion %s: %s", suggestion_id, title)

        if id_map:
            self._event_stream.broadcast("agent_suggestions_recorded", {
                "run_id": run_id, "count": len(id_map),
            })

        # Record structural experiments from approved proposals with acceptance_criteria
        if approved_structural:
            self._record_structural_experiments(approved_structural, run_id)

        return id_map
