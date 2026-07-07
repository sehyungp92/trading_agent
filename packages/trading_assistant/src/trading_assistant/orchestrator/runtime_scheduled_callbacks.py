"""Runtime-owned scheduled callback implementations."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from trading_assistant.orchestrator.orchestrator_brain import Action, ActionType
from trading_assistant.orchestrator.runtime_jobs import RuntimeJobCallbacks
from trading_assistant.schemas.suggestion_tracking import SuggestionStatus

logger = logging.getLogger(__name__)


def _run_coroutine_in_thread(coro):
    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover - defensive bridge for optional LLM review
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("result", {"actions": []})


def _extract_json_object(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {"actions": []}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"actions": []}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {"actions": []}
            except json.JSONDecodeError:
                return {"actions": []}
    return {"actions": []}


def requires_monthly_outcome(suggestion: dict) -> bool:
    """Return True for material strategy/config changes."""
    tier = str(suggestion.get("tier") or "").lower()
    category = str(suggestion.get("category") or "").lower()
    change_kind = str(suggestion.get("change_kind") or suggestion.get("kind") or "").lower()
    risk = str(
        (suggestion.get("implementation_context") or {}).get("risk_classification")
        or suggestion.get("risk_classification")
        or ""
    ).lower()
    has_config_change = bool(
        suggestion.get("param_name")
        or suggestion.get("param_changes")
        or suggestion.get("planned_files")
        or suggestion.get("strategy_id")
    )
    material_tokens = {
        "parameter",
        "parameter_change",
        "filter",
        "filter_threshold",
        "strategy_variant",
        "structural",
        "position_sizing",
        "regime_gate",
        "stop_loss",
        "exit_timing",
        "signal",
    }
    return (
        has_config_change
        or tier in material_tokens
        or category in material_tokens
        or change_kind in material_tokens
        or risk in {"medium", "high", "critical"}
    )


async def expire_approvals_with_notification(
    approval_tracker,
    telegram_bot,
    dispatcher,
    notification_prefs,
) -> None:
    """Expire old approvals and send notification for each expired request."""
    expired_ids = approval_tracker.expire_old(max_age_days=7)
    if not expired_ids and telegram_bot is None:
        return
    for rid in expired_ids:
        logger.info("Approval request %s expired", rid)
        if telegram_bot is not None:
            try:
                req = approval_tracker.get_by_id(rid)
                params = ", ".join(
                    pc.get("param_name", "?") for pc in (req.param_changes if req else [])
                )
                text = (
                    f"\u23f0 Approval Expired: {rid}\n"
                    f"Bot: {req.bot_id if req else '?'}\n"
                    f"Params: {params}\n"
                    "Request was pending for >7 days."
                )
                await telegram_bot.send_message(text)
                if req and req.message_id:
                    try:
                        await telegram_bot.edit_message(
                            req.message_id,
                            f"Suggestion {rid}\nBot: {req.bot_id}\n"
                            "\u23f0 EXPIRED - pending >7 days",
                        )
                    except Exception:
                        pass
            except Exception:
                logger.warning("Failed to send expiry notification for %s", rid)


def build_runtime_job_callbacks(
    runtime: Any,
    *,
    reconcile_linked_subagent_tasks,
) -> RuntimeJobCallbacks:
    config = runtime.config
    db_path = runtime.db_path
    curated_dir = runtime.curated_dir
    memory_dir = runtime.memory_dir
    handlers = runtime.handlers
    event_stream = runtime.event_stream
    agent_runner = runtime.agent_runner
    suggestion_tracker = runtime.suggestion_tracker
    proposal_ledger = runtime.proposal_ledger
    hypothesis_library = runtime.hypothesis_library
    structural_experiment_tracker = runtime.structural_experiment_tracker
    calibration_tracker = runtime.calibration_tracker
    reliability_tracker = runtime.reliability_tracker
    playbook_generator = runtime.playbook_generator
    write_coordinator = runtime.write_coordinator

    from trading_assistant.skills.auto_outcome_measurer import AutoOutcomeMeasurer

    outcome_measurer = AutoOutcomeMeasurer(
        curated_dir=curated_dir,
        findings_dir=memory_dir / "findings",
        hypothesis_library=hypothesis_library,
        proposal_ledger=proposal_ledger,
    )

    def lookup_proposal_id(
        suggestion_id: str = "",
        experiment_id: str = "",
    ) -> str:
        if suggestion_id:
            try:
                for suggestion in suggestion_tracker.load_all():
                    if (
                        suggestion.get("suggestion_id") == suggestion_id
                        and suggestion.get("proposal_id")
                    ):
                        return suggestion["proposal_id"]
            except Exception:
                logger.warning("Failed to lookup proposal id for suggestion %s", suggestion_id)
        if experiment_id or suggestion_id:
            try:
                for record in proposal_ledger.list_all():
                    candidate = record.candidate
                    if experiment_id and candidate.experiment_id == experiment_id:
                        return candidate.proposal_id
                    if suggestion_id and candidate.suggestion_id == suggestion_id:
                        return candidate.proposal_id
            except Exception:
                logger.warning("Failed to lookup proposal id for experiment %s", experiment_id)
        return ""

    def record_proposal_outcome(
        *,
        proposal_id: str,
        objective_delta: float,
        verdict: str,
        measurement_path: Path,
        outcome_source: str = "early_warning",
    ) -> None:
        if not proposal_id:
            return
        try:
            from trading_assistant.schemas.proposal_ledger import ProposalOutcome

            proposal_ledger.record_outcome(
                proposal_id,
                ProposalOutcome(
                    proposal_id=proposal_id,
                    objective_delta=float(objective_delta or 0.0),
                    verdict=verdict,
                    measurement_path=str(measurement_path),
                    outcome_source=outcome_source,
                ),
            )
        except Exception:
            logger.warning("Failed to record proposal outcome for %s", proposal_id)

    async def measure_outcomes(scheduled_for: datetime | None = None) -> None:
        suggestions = suggestion_tracker.load_all()
        deployed = [
            s for s in suggestions
            if s.get("status") == SuggestionStatus.DEPLOYED.value
        ]
        existing_outcomes = suggestion_tracker.load_outcomes()
        measured_ids = {o.get("suggestion_id") for o in existing_outcomes}

        for suggestion in deployed:
            sid = suggestion.get("suggestion_id", "")
            if sid in measured_ids or suggestion.get("bot_id") == "PORTFOLIO":
                continue
            anchor_date = (suggestion.get("deployed_at") or "")[:10]
            if not anchor_date:
                continue
            try:
                result = outcome_measurer.measure_progressive(
                    suggestion_id=sid,
                    bot_id=suggestion.get("bot_id", ""),
                    implemented_date=anchor_date,
                )
                if result:
                    detection_context = suggestion.get("detection_context") or {}
                    if not isinstance(detection_context, dict):
                        detection_context = {}
                    result = result.model_copy(update={
                        "bot_id": suggestion.get("bot_id", ""),
                        "category": suggestion.get("category", ""),
                        "source_run_id": suggestion.get("source_report_id", ""),
                        "source_provider": detection_context.get("source_provider", ""),
                        "source_model": detection_context.get("source_model", ""),
                    })
                    enhanced_path = memory_dir / "findings" / "outcomes.jsonl"
                    enhanced_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(enhanced_path, "a", encoding="utf-8") as handle:
                        handle.write(result.model_dump_json() + "\n")

                    outcome_measurer.record_measurement_feedback(result)
                    suggestion_tracker.mark_measured(
                        sid,
                        source="early_warning",
                        final=not requires_monthly_outcome(suggestion),
                    )
                    verdict_str = (
                        result.verdict.value
                        if hasattr(result.verdict, "value")
                        else str(result.verdict)
                    )
                    if verdict_str == "positive":
                        try:
                            from trading_assistant.schemas.pattern_library import PatternStatus
                            from trading_assistant.skills.pattern_library import PatternLibrary

                            pattern_library = PatternLibrary(memory_dir / "findings")
                            for pattern in pattern_library.load_all():
                                if (
                                    pattern.linked_suggestion_id == sid
                                    and pattern.status == PatternStatus.PROPOSED
                                ):
                                    pattern_library.validate_pattern(pattern.pattern_id)
                                    logger.info("Promoted pattern %s to VALIDATED", pattern.pattern_id)
                        except Exception:
                            logger.warning("Failed to promote pattern for suggestion %s", sid)
            except Exception:
                logger.warning("Outcome measurement failed for %s", sid)

        try:
            from trading_assistant.skills.portfolio_outcome_measurer import PortfolioOutcomeMeasurer

            portfolio_measurer = PortfolioOutcomeMeasurer(
                findings_dir=memory_dir / "findings",
                curated_dir=curated_dir,
            )
            portfolio_outcomes = portfolio_measurer.measure_deployed()
            for outcome in portfolio_outcomes:
                outcome_sid = outcome.get("suggestion_id", "")
                if outcome_sid:
                    suggestion_tracker.mark_measured(
                        outcome_sid,
                        source="early_warning",
                        final=False,
                    )
                    outcome_verdict = outcome.get("verdict", "")
                    outcome_hypothesis_id = None
                    outcome_proposal_id = None
                    for suggestion in deployed:
                        if suggestion.get("suggestion_id") == outcome_sid:
                            outcome_hypothesis_id = suggestion.get("hypothesis_id")
                            outcome_proposal_id = suggestion.get("proposal_id")
                            break
                    if outcome_hypothesis_id and outcome_verdict in ("positive", "negative"):
                        try:
                            hypothesis_library.record_outcome(
                                outcome_hypothesis_id,
                                positive=(outcome_verdict == "positive"),
                            )
                        except Exception:
                            logger.warning(
                                "Failed to record hypothesis outcome for portfolio %s",
                                outcome_hypothesis_id,
                            )
                    record_proposal_outcome(
                        proposal_id=outcome_proposal_id or "",
                        objective_delta=float(outcome.get("composite_delta", 0.0) or 0.0),
                        verdict=outcome_verdict or "inconclusive",
                        measurement_path=memory_dir / "findings" / "portfolio_outcomes.jsonl",
                        outcome_source="early_warning",
                    )
            if portfolio_outcomes:
                logger.info("Measured %d portfolio outcomes", len(portfolio_outcomes))
        except Exception:
            logger.warning("Portfolio outcome measurement failed")

        try:
            from trading_assistant.skills.prediction_tracker import PredictionTracker

            prediction_tracker = PredictionTracker(memory_dir / "findings")
            predictions = prediction_tracker.load_predictions()
            if predictions:
                for week in sorted({prediction.week for prediction in predictions}):
                    evaluation = prediction_tracker.evaluate_predictions(week, curated_dir)
                    if evaluation.total > 0:
                        logger.info(
                            "Prediction evaluation for %s: %d/%d correct (%.0f%%)",
                            week,
                            evaluation.correct,
                            evaluation.total,
                            evaluation.accuracy * 100,
                        )
        except Exception:
            logger.warning("Prediction evaluation failed during outcome measurement")

        try:
            from trading_assistant.analysis.outcome_reasoning_prompt import OutcomeReasoningAssembler

            outcomes_path = memory_dir / "findings" / "outcomes.jsonl"
            reasoning_path = memory_dir / "findings" / "outcome_reasonings.jsonl"
            if outcomes_path.exists():
                recent_outcomes = []
                reasoned_ids: set[str] = set()
                if reasoning_path.exists():
                    for line in reasoning_path.read_text(encoding="utf-8").strip().splitlines():
                        if line.strip():
                            try:
                                reasoned_ids.add(json.loads(line).get("suggestion_id", ""))
                            except json.JSONDecodeError:
                                pass
                for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                    if line.strip():
                        try:
                            outcome = json.loads(line)
                            if outcome.get("suggestion_id", "") not in reasoned_ids:
                                recent_outcomes.append(outcome)
                        except json.JSONDecodeError:
                            pass

                if recent_outcomes:
                    assembler = OutcomeReasoningAssembler(
                        memory_dir=memory_dir,
                        curated_dir=curated_dir,
                        bot_configs=config.bot_configs,
                    )
                    reasoning_result = await agent_runner.invoke(
                        agent_type="outcome_reasoning",
                        prompt_package=assembler.assemble(
                            recent_outcomes,
                            session_store=agent_runner.session_store,
                        ),
                        run_id=f"reasoning-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
                        allowed_tools=["Read"],
                        max_turns=5,
                    )
                    if reasoning_result.success:
                        from trading_assistant.analysis.response_parser import parse_response

                        parsed_reasoning = parse_response(reasoning_result.response)
                        if (
                            parsed_reasoning.raw_structured
                            and "reasonings" in parsed_reasoning.raw_structured
                        ):
                            reasonings = parsed_reasoning.raw_structured["reasonings"]
                            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            enriched = [
                                {**reasoning, "reasoned_at": datetime.now(timezone.utc).isoformat()}
                                for reasoning in reasonings
                            ]
                            suggestion_lookup = {
                                suggestion.get("suggestion_id", ""): suggestion
                                for suggestion in deployed
                            }
                            spurious_records: list[dict] = []
                            recalib_records: list[dict] = []
                            transfer_builder = None
                            findings = memory_dir / "findings"
                            for reasoning in reasonings:
                                sid = reasoning.get("suggestion_id", "")
                                if reasoning.get("transferable") and sid:
                                    try:
                                        if transfer_builder is None:
                                            from trading_assistant.skills.pattern_library import PatternLibrary
                                            from trading_assistant.skills.transfer_proposal_builder import (
                                                TransferProposalBuilder,
                                            )

                                            transfer_builder = TransferProposalBuilder(
                                                pattern_library=PatternLibrary(findings),
                                                curated_dir=curated_dir,
                                                bots=config.bot_ids,
                                                findings_dir=findings,
                                            )
                                        transfer_builder.create_from_reasoning(
                                            reasoning,
                                            source_bot=reasoning.get("bot_id", ""),
                                        )
                                    except Exception:
                                        logger.warning("Transfer proposal from reasoning failed for %s", sid)
                                if reasoning.get("genuine_effect") is False and sid:
                                    spurious_records.append({
                                        "suggestion_id": sid,
                                        "bot_id": suggestion_lookup.get(sid, {}).get("bot_id", ""),
                                        "mechanism": reasoning.get("mechanism", ""),
                                        "confounders": reasoning.get("confounders", []),
                                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                                    })
                                revised = reasoning.get("revised_confidence")
                                if revised is not None and sid:
                                    suggestion = suggestion_lookup.get(sid, {})
                                    recalib_records.append({
                                        "suggestion_id": sid,
                                        "bot_id": suggestion.get("bot_id", ""),
                                        "category": suggestion.get("category", ""),
                                        "revised_confidence": revised,
                                        "lessons_learned": reasoning.get("lessons_learned", ""),
                                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                                    })
                            group = write_coordinator.begin(
                                source_workflow="outcome_reasoning",
                                source_run_id=f"reasoning-{date_str}",
                            )
                            write_coordinator.add_jsonl_append(
                                group,
                                "record_reasonings",
                                "outcome_reasonings.jsonl",
                                enriched,
                            )
                            if spurious_records:
                                write_coordinator.add_jsonl_append(
                                    group,
                                    "record_spurious",
                                    "spurious_outcomes.jsonl",
                                    spurious_records,
                                )
                            if recalib_records:
                                write_coordinator.add_jsonl_append(
                                    group,
                                    "record_recalibrations",
                                    "recalibrations.jsonl",
                                    recalib_records,
                                )
                            result = write_coordinator.execute(group)
                            logger.info(
                                "Outcome reasoning writes: group=%s, all_ok=%s, ops=%d",
                                result.group_id,
                                result.all_succeeded,
                                len(result.operations),
                            )
        except Exception:
            logger.warning("Outcome reasoning failed - skipping", exc_info=True)

        _ingest_learning_cards(memory_dir, "after outcome reasoning")

    async def measure_transfer_outcomes(scheduled_for: datetime | None = None) -> None:
        try:
            from trading_assistant.skills.pattern_library import PatternLibrary
            from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

            outcomes = TransferProposalBuilder(
                pattern_library=PatternLibrary(memory_dir / "findings"),
                curated_dir=curated_dir,
                bots=config.bot_ids,
                findings_dir=memory_dir / "findings",
            ).measure_transfer_outcomes()
            if outcomes:
                logger.info("Measured %d transfer outcomes", len(outcomes))
        except Exception:
            logger.exception("Transfer outcome measurement failed")

    async def verify_reliability(scheduled_for: datetime | None = None) -> None:
        try:
            verified = reliability_tracker.verify_completed()
            if verified:
                logger.info("Auto-verified %d reliability interventions", len(verified))
                summary = reliability_tracker.compute_summary()
                if summary.chronic_bug_classes:
                    created = hypothesis_library.create_from_reliability(summary)
                    if created:
                        logger.info("Created %d reliability hypothesis candidates", len(created))
        except Exception:
            logger.exception("Reliability verification failed")

    async def discovery_analysis(scheduled_for: datetime | None = None) -> None:
        try:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            await handlers.handle_discovery_analysis(Action(
                type=ActionType.LOG_UNKNOWN,
                event_id=f"discovery-{date}",
                bot_id="",
                details={"date": date, "bots": config.bot_ids},
            ))
        except Exception:
            logger.exception("Discovery analysis failed")

    def learning_review_structured_reviewer():
        if config.learning_review_mode != "llm_review":
            return None

        def reviewer(payload: dict) -> dict:
            async def invoke() -> dict:
                from trading_assistant.schemas.prompt_package import PromptPackage

                package = PromptPackage(
                    system_prompt=(
                        "You are a bounded post-run learning reviewer. Return JSON only. "
                        "You may propose advisory learning actions, never policy edits, "
                        "approval, deployment, or trading-logic changes."
                    ),
                    task_prompt=json.dumps(payload, indent=2, default=str),
                    instructions=(
                        "Return an object shaped as {\"actions\": [...]}. Every action must use "
                        "one of allowed_action_types and cite safe evidence_paths from the payload."
                    ),
                    metadata={"workflow": "learning_review", "mode": "llm_review"},
                )
                result = await agent_runner.invoke(
                    "weekly_analysis",
                    package,
                    run_id=f"learning-review-{payload.get('week_end') or 'latest'}",
                    max_turns=3,
                    allowed_tools=[],
                )
                if not result.success:
                    return {"actions": []}
                return _extract_json_object(result.response)

            return _run_coroutine_in_thread(invoke())

        return reviewer

    async def run_learning_cycle(scheduled_for: datetime | None = None) -> None:
        try:
            from trading_assistant.skills.learning_cycle import LearningCycle

            now = datetime.now(timezone.utc)
            week_end = now.strftime("%Y-%m-%d")
            week_start = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            entry = await LearningCycle(
                curated_dir=curated_dir,
                memory_dir=memory_dir,
                runs_dir=db_path / "runs",
                bots=config.bot_ids,
                suggestion_tracker=suggestion_tracker,
                hypothesis_library=hypothesis_library,
                experiment_tracker=structural_experiment_tracker,
                prediction_tracker=None,
                calibration_tracker=calibration_tracker,
                learning_review_mode=config.learning_review_mode,
                learning_review_disabled_workflows=config.learning_review_disabled_workflows,
                learning_review_structured_reviewer=learning_review_structured_reviewer(),
            ).run(week_start, week_end)
            event_stream.broadcast("learning_cycle_completed", {
                "week_start": week_start,
                "week_end": week_end,
                "net_improvement": entry.net_improvement,
                "composite_delta": entry.composite_delta,
                "lessons": entry.lessons_for_next_week[:3],
            })
            logger.info("Learning cycle complete: net_improvement=%s", entry.net_improvement)
        except Exception:
            logger.exception("Learning cycle failed")

    async def consolidate_memory(scheduled_for: datetime | None = None) -> None:
        try:
            runtime.consolidator.rebuild_index()
            if runtime.consolidator.needs_consolidation("corrections.jsonl"):
                runtime.consolidator.consolidate("corrections.jsonl")
            if runtime.consolidator.needs_consolidation("failure-log.jsonl"):
                runtime.consolidator.consolidate("failure-log.jsonl")
            promoted = hypothesis_library.promote_candidates()
            if promoted:
                logger.info("Promoted %d candidate hypotheses to active", promoted)
        except Exception:
            logger.exception("Memory consolidation failed")

        _ingest_learning_cards(memory_dir, "after consolidation")
        try:
            retired = playbook_generator.retire_ineffective()
            if retired:
                logger.info("Retired %d ineffective playbooks", retired)
        except Exception:
            logger.warning("Playbook retirement failed", exc_info=True)

        try:
            from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
            from trading_assistant.skills.archive_retired_strategies import archive_retired

            archive_summary = archive_retired(
                memory_dir / "findings",
                load_strategy_registry(),
            )
            if archive_summary:
                logger.info(
                    "Archived retired-strategy records: %s",
                    ", ".join(f"{name}={count}" for name, count in archive_summary.items()),
                )
        except Exception:
            logger.warning("Retired-strategy archive sweep failed", exc_info=True)

    return RuntimeJobCallbacks(
        outcome_measurement=measure_outcomes,
        memory_consolidation=consolidate_memory,
        transfer_outcome=measure_transfer_outcomes,
        reliability_verification=verify_reliability,
        discovery_analysis=discovery_analysis,
        learning_cycle=run_learning_cycle,
        lookup_proposal_id=lookup_proposal_id,
        record_proposal_outcome=record_proposal_outcome,
        expire_approvals=expire_approvals_with_notification,
        reconcile_linked_subagent_tasks=reconcile_linked_subagent_tasks,
    )


def _ingest_learning_cards(memory_dir: Path, label: str) -> None:
    try:
        from trading_assistant.skills.learning_card_store import LearningCardStore

        count = LearningCardStore(memory_dir / "findings").ingest_from_existing()
        if count:
            logger.info("Ingested %d new learning cards %s", count, label)
    except Exception:
        logger.warning("Learning card ingestion %s failed", label, exc_info=True)
