"""Weekly analysis loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator
from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeeklyAnalysisDependencies:
    agent_runner: Any
    event_stream: Any
    curated_dir: Path
    memory_dir: Path
    runs_dir: Path
    bots: list[str]
    bot_configs: dict | None
    strategy_registry: object | None
    run_index: object | None
    threshold_learner: object | None
    suggestion_tracker: object | None
    experiment_manager: object | None
    brain: Any | None
    record_run: Callable[..., None]
    load_bot_dailies: Callable[[str, str, str], list]
    load_weekly_strategy_evidence: Callable[..., dict]
    run_portfolio_detectors: Callable[..., list]
    run_weekly_simulations: Callable[..., dict]
    run_allocation_analyses: Callable[..., dict]
    record_suggestions: Callable[..., dict]
    run_autonomous_pipeline: Callable[[dict[str, str], str], Awaitable[None]]
    ledger_write_candidate: Callable[..., Any]
    validate_and_annotate: Callable[..., tuple[str, object]]
    persist_validator_notes: Callable[..., None]
    refresh_run_index_entry: Callable[..., None]
    record_agent_suggestions: Callable[..., dict]
    record_learning_card_feedback_targeted: Callable[..., None]
    record_portfolio_proposals: Callable[..., None]
    record_predictions: Callable[..., None]
    update_hypothesis_lifecycle: Callable[..., None]
    extract_and_record_patterns: Callable[..., None]
    write_run_report: Callable[..., None]
    notify: Callable[..., Awaitable[None]]
    signal_scheduled_result: Callable[..., Awaitable[None]]


class WeeklyAnalysisLoop:
    """Run weekly analysis behind a loop-shaped module."""

    def __init__(self, dependencies: WeeklyAnalysisDependencies) -> None:
        self._deps = dependencies

    async def handle(self, action: Action) -> None:
        deps = self._deps
        details = action.details or {}
        week_start = details.get("week_start", "")
        week_end = details.get("week_end", "")
        run_id = f"weekly-{week_start}"
        start_time = datetime.now(timezone.utc)
        deps.record_run(run_id, "weekly_analysis", "running", started_at=start_time.isoformat())
        scheduled_success = False
        scheduled_error = ""

        try:
            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "weekly_analysis",
            })

            index = MemoryConsolidator.load_index(deps.runs_dir.parent)
            if index:
                start = datetime.strptime(week_start, "%Y-%m-%d")
                dates_in_week = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                for bot in deps.bots:
                    avail_dates = index.curated_dates_by_bot.get(bot, [])
                    count = sum(1 for d in dates_in_week if d in avail_dates)
                    if count < 5:
                        logger.warning(
                            "Only %d/7 daily data days for %s in week %s", count, bot, week_start,
                        )

            from trading_assistant.skills.build_weekly_metrics import WeeklyMetricsBuilder

            builder = WeeklyMetricsBuilder(
                week_start=week_start,
                week_end=week_end,
                bots=deps.bots,
            )

            dailies_by_bot = {
                bot: deps.load_bot_dailies(bot, week_start, week_end)
                for bot in deps.bots
            }
            portfolio_summary = builder.build_portfolio_summary(dailies_by_bot)
            builder.write_weekly_curated(portfolio_summary, deps.curated_dir)

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "metrics_build", "handler": "weekly_analysis",
            })

            signal_health_data: dict[str, dict] = {}
            start_dt = datetime.strptime(week_start, "%Y-%m-%d")
            dates_in_week = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
            for bot in deps.bots:
                for date_str in reversed(dates_in_week):
                    sh_path = deps.curated_dir / date_str / bot / "signal_health.json"
                    if sh_path.exists():
                        signal_health_data[bot] = json.loads(sh_path.read_text())
                        break

            factor_rolling_data: dict[str, list[dict]] = {}
            findings_dir = deps.memory_dir / "findings"
            if findings_dir.exists():
                try:
                    from trading_assistant.skills.signal_factor_tracker import SignalFactorTracker

                    tracker = SignalFactorTracker(findings_dir)
                    for bot in deps.bots:
                        report = tracker.compute_rolling(bot, week_end)
                        if report.factors:
                            factor_rolling_data[bot] = [
                                f.model_dump(mode="json") for f in report.factors
                            ]
                except Exception:
                    logger.warning("Failed to load factor rolling data", exc_info=True)

            weekly_evidence = deps.load_weekly_strategy_evidence(
                week_start=week_start,
                week_end=week_end,
                bot_summaries=portfolio_summary.bot_summaries,
                signal_health_data=signal_health_data,
                factor_rolling_data=factor_rolling_data,
            )

            from trading_assistant.analysis.strategy_engine import StrategyEngine

            scorecard = None
            scorer = None
            detector_confidence: dict[str, float] = {}
            try:
                from trading_assistant.skills.suggestion_scorer import SuggestionScorer
                scorer = SuggestionScorer(deps.memory_dir / "findings")
                scorecard = scorer.compute_scorecard()
                detector_confidence = scorer.compute_detector_confidence()
            except Exception:
                logger.debug("Could not load category scorecard / detector confidence")

            recent_suggestions: list[dict] = []
            if deps.suggestion_tracker:
                grouped = deps.suggestion_tracker.get_recent_grouped(
                    list(portfolio_summary.bot_summaries.keys()),
                    weeks=4,
                )
                for items in grouped.values():
                    recent_suggestions.extend(items)

            convergence_report: dict = {}
            try:
                convergence_report = ContextBuilder(deps.memory_dir).load_convergence_report()
            except Exception:
                pass

            category_value_map: dict = {}
            if scorer:
                try:
                    category_value_map = scorer.compute_category_value_map()
                except Exception:
                    pass

            engine = StrategyEngine(
                week_start=week_start, week_end=week_end,
                threshold_learner=deps.threshold_learner,
                strategy_registry=deps.strategy_registry,
                category_scorecard=scorecard,
                detector_confidence=detector_confidence,
                recent_suggestions=recent_suggestions,
                convergence_report=convergence_report,
                category_value_map=category_value_map,
            )
            refinement_report = engine.build_report(
                portfolio_summary.bot_summaries,
                **weekly_evidence,
            )

            try:
                portfolio_suggestions = deps.run_portfolio_detectors(
                    engine, week_start, week_end, portfolio_summary,
                )
                if portfolio_suggestions:
                    refinement_report.suggestions.extend(portfolio_suggestions)
                    logger.info(
                        "Portfolio detectors produced %d suggestions", len(portfolio_suggestions),
                    )
            except Exception:
                logger.warning("Portfolio detectors failed - skipping", exc_info=True)

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "strategy_engine", "handler": "weekly_analysis",
            })

            simulation_results = deps.run_weekly_simulations(
                refinement_report, week_start, week_end,
            )
            allocation_results = deps.run_allocation_analyses(
                portfolio_summary, week_start, week_end,
            )

            if simulation_results:
                weekly_evidence = deps.load_weekly_strategy_evidence(
                    week_start=week_start,
                    week_end=week_end,
                    bot_summaries=portfolio_summary.bot_summaries,
                    signal_health_data=signal_health_data,
                    factor_rolling_data=factor_rolling_data,
                    simulation_results=simulation_results,
                )
                refinement_report = engine.build_report(
                    portfolio_summary.bot_summaries,
                    **weekly_evidence,
                )

            weekly_dir = deps.curated_dir / "weekly" / week_start
            weekly_dir.mkdir(parents=True, exist_ok=True)
            (weekly_dir / "refinement_report.json").write_text(
                json.dumps(refinement_report.model_dump(mode="json"), indent=2, default=str),
                encoding="utf-8",
            )

            suggestion_ids = deps.record_suggestions(
                getattr(refinement_report, "suggestions", []), run_id,
                category_scorecard=scorecard,
            )
            await deps.run_autonomous_pipeline(suggestion_ids, run_id)

            experiment_results = []
            if deps.experiment_manager is not None:
                start_exp = datetime.strptime(week_start, "%Y-%m-%d")
                dates_in_week_exp = [(start_exp + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
                for bot in deps.bots:
                    for date_str in dates_in_week_exp:
                        exp_path = deps.curated_dir / date_str / bot / "experiment_data.json"
                        if not exp_path.exists():
                            continue
                        try:
                            exp_data = json.loads(exp_path.read_text())
                            for exp_id, variants in exp_data.items():
                                for variant_name, variant_data in variants.items():
                                    trades = variant_data.get("trades", [])
                                    if trades:
                                        deps.experiment_manager.ingest_variant_data(
                                            exp_id, variant_name, trades,
                                        )
                        except Exception:
                            logger.warning("Failed to ingest experiment data from %s", exp_path)

                active_experiments = deps.experiment_manager.get_active()
                for exp in active_experiments:
                    try:
                        if deps.experiment_manager.check_auto_conclusion(exp.experiment_id):
                            result = deps.experiment_manager.analyze_experiment(exp.experiment_id)
                            deps.experiment_manager.conclude_experiment(exp.experiment_id, result)
                            experiment_results.append(result)
                            deps.event_stream.broadcast("experiment_concluded", {
                                "experiment_id": exp.experiment_id,
                                "recommendation": result.recommendation,
                            })
                    except Exception:
                        logger.warning("Experiment check failed for %s", exp.experiment_id)

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "simulations", "handler": "weekly_analysis",
            })

            from trading_assistant.analysis.weekly_triage import WeeklyTriage
            from trading_assistant.analysis.weekly_prompt_assembler import WeeklyPromptAssembler

            ctx_weekly = ContextBuilder(deps.memory_dir, curated_dir=deps.curated_dir)
            reliable_outcomes, _low_q = ctx_weekly.load_outcome_measurements()
            weekly_triage = WeeklyTriage(
                curated_dir=deps.curated_dir,
                end_date=week_end,
                bots=deps.bots,
                active_suggestions=ctx_weekly.load_active_suggestions(),
                outcome_measurements=reliable_outcomes,
                prediction_accuracy=ctx_weekly.load_prediction_accuracy(),
            )
            weekly_triage_report = weekly_triage.run()

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "triage", "handler": "weekly_analysis",
                "anomalies": len(weekly_triage_report.anomalies),
            })

            assembler = WeeklyPromptAssembler(
                week_start=week_start,
                week_end=week_end,
                bots=deps.bots,
                curated_dir=deps.curated_dir,
                memory_dir=deps.memory_dir,
                runs_dir=deps.runs_dir,
                bot_configs=deps.bot_configs,
                strategy_registry=deps.strategy_registry,
                run_index=deps.run_index,
            )
            package = assembler.assemble(
                triage_report=weekly_triage_report,
                session_store=deps.agent_runner.session_store,
            )

            if simulation_results:
                package.data.update({"simulation_results": simulation_results})
            if allocation_results:
                package.data.update({"allocation_analysis": allocation_results})
            if suggestion_ids:
                package.metadata["suggestion_ids"] = suggestion_ids

            retro_builder = None
            try:
                from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder

                retro_builder = RetrospectiveBuilder(
                    runs_dir=deps.runs_dir,
                    curated_dir=deps.curated_dir,
                    memory_dir=deps.memory_dir,
                )
                retrospective = retro_builder.build(week_start, week_end)
                if retrospective.predictions_reviewed > 0:
                    package.data["weekly_retrospective"] = retrospective.model_dump(mode="json")
            except Exception:
                logger.warning("Retrospective builder failed - skipping")

            try:
                if retro_builder is None:
                    from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder
                    retro_builder = RetrospectiveBuilder(
                        runs_dir=deps.runs_dir,
                        curated_dir=deps.curated_dir,
                        memory_dir=deps.memory_dir,
                    )
                synthesis = retro_builder.build_synthesis(week_start, week_end)
                if synthesis.what_worked or synthesis.what_failed or synthesis.discard or synthesis.lessons:
                    package.data["last_week_synthesis"] = synthesis.model_dump(mode="json")

                if synthesis.discard:
                    try:
                        from trading_assistant.skills.suggestion_scorer import SuggestionScorer
                        SuggestionScorer(deps.memory_dir / "findings").apply_recalibration(synthesis.discard)
                        logger.info("Recalibrated %d categories from synthesis", len(synthesis.discard))
                    except Exception:
                        logger.warning("Category recalibration failed")
            except Exception:
                logger.warning("Retrospective synthesis failed - skipping")

            try:
                from trading_assistant.skills.forecast_tracker import ForecastTracker
                from trading_assistant.schemas.forecast_tracking import ForecastRecord

                forecast_tracker = ForecastTracker(deps.memory_dir / "findings")
                retro_data = package.data.get("weekly_retrospective")
                if retro_data:
                    by_bot = {}
                    by_type: dict[str, list[float]] = {}
                    for pred in retro_data.get("predictions", []):
                        bid = pred.get("bot_id", "")
                        accuracy = pred.get("accuracy", "")
                        score = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}.get(accuracy)
                        if score is None:
                            continue
                        if bid:
                            by_bot.setdefault(bid, []).append(score)
                        metric = pred.get("metric", "") or pred.get("prediction_type", "")
                        if metric:
                            by_type.setdefault(metric, []).append(score)
                    forecast_tracker.record_week(ForecastRecord(
                        week_start=week_start,
                        week_end=week_end,
                        predictions_reviewed=retro_data.get("predictions_reviewed", 0),
                        correct_predictions=retro_data.get("correct", 0),
                        accuracy=retro_data.get("accuracy_pct", 0.0) / 100.0,
                        by_bot={b: sum(v) / len(v) for b, v in by_bot.items() if v},
                        by_type={m: sum(v) / len(v) for m, v in by_type.items() if v},
                    ))
                meta = forecast_tracker.compute_meta_analysis()
                if meta.weeks_analyzed > 0:
                    package.data["forecast_meta_analysis"] = meta.model_dump(mode="json")
            except Exception:
                logger.error("Forecast tracking failed - skipping", exc_info=True)

            try:
                from trading_assistant.skills.learning_ledger import LearningLedger

                outcome_lessons: list[str] = []
                for om in package.data.get("outcome_measurements", []):
                    verdict = om.get("verdict", "")
                    sid = om.get("suggestion_id", "")
                    if verdict in ("positive", "negative"):
                        outcome_lessons.append(
                            f"Suggestion {sid}: {verdict} outcome "
                            f"(PnL delta={om.get('pnl_delta', 0):.2f})"
                        )
                if outcome_lessons:
                    LearningLedger(deps.memory_dir / "findings").record_outcome_lessons(
                        week_start, outcome_lessons,
                    )
            except Exception:
                logger.debug("Outcome-derived lesson injection failed")

            try:
                from trading_assistant.skills.correction_pattern_extractor import CorrectionPatternExtractor

                corrections = ContextBuilder(deps.memory_dir).load_corrections()
                if corrections:
                    pattern_report = CorrectionPatternExtractor(min_occurrences=2).extract(corrections)
                    if pattern_report.patterns:
                        package.data["correction_patterns"] = [
                            p.model_dump(mode="json") for p in pattern_report.patterns
                        ]
                        patterns_path = deps.memory_dir / "findings" / "correction_patterns.jsonl"
                        patterns_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(patterns_path, "w", encoding="utf-8") as f:
                            for p in pattern_report.patterns:
                                f.write(json.dumps(p.model_dump(mode="json")) + "\n")

                        try:
                            from trading_assistant.skills.learning_ledger import LearningLedger
                            correction_lessons = []
                            for p in pattern_report.patterns[:5]:
                                if p.count >= 3:
                                    target = getattr(p, "target", "") or ""
                                    correction_lessons.append(
                                        f"[correction] {p.description}. "
                                        f"Adjust analysis approach for {target}."
                                    )
                            if correction_lessons:
                                LearningLedger(deps.memory_dir / "findings").record_outcome_lessons(
                                    week_start, correction_lessons, source="corrections",
                                )
                        except Exception:
                            logger.debug("Correction lesson synthesis failed")
            except Exception:
                logger.error("Correction pattern extraction failed - skipping", exc_info=True)

            try:
                from trading_assistant.skills.hypothesis_library import HypothesisLibrary

                active_hypotheses = HypothesisLibrary(deps.memory_dir / "findings").get_active()
                keyword_map = {
                    "signal": "signal_decay", "decay": "signal_decay", "alpha": "signal_decay",
                    "filter": "filter_over_blocking", "block": "filter_over_blocking",
                    "exit": "exit_timing", "premature": "exit_timing", "stop": "exit_timing",
                    "slippage": "adverse_fills", "fill": "adverse_fills",
                    "regime": "regime_breakdown",
                    "correlation": "correlation_crowding", "crowding": "correlation_crowding",
                    "diversif": "correlation_crowding",
                }
                matched_categories: set[str] = set()
                for suggestion in getattr(refinement_report, "suggestions", []):
                    text = f"{getattr(suggestion, 'title', '') or ''} {getattr(suggestion, 'description', '') or ''}".lower()
                    for keyword, category in keyword_map.items():
                        if keyword in text:
                            matched_categories.add(category)

                seen_ids: set[str] = set()
                merged: list[dict] = []
                for h in active_hypotheses:
                    if h.category in matched_categories or h.effectiveness > 0.3 or h.status == "candidate":
                        if h.id not in seen_ids:
                            seen_ids.add(h.id)
                            merged.append({
                                "id": h.id, "title": h.title, "category": h.category,
                                "description": h.description, "evidence_required": h.evidence_required,
                                "reversibility": h.reversibility, "estimated_complexity": h.estimated_complexity,
                                "effectiveness": round(h.effectiveness, 3),
                                "times_proposed": h.times_proposed,
                            })
                if merged:
                    package.data["structural_hypotheses"] = merged
            except Exception:
                logger.error("Hypothesis library matching failed - skipping", exc_info=True)

            try:
                from trading_assistant.skills.pattern_library import PatternLibrary
                from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

                proposals = TransferProposalBuilder(
                    pattern_library=PatternLibrary(deps.memory_dir / "findings"),
                    curated_dir=deps.curated_dir,
                    bots=deps.bots,
                    findings_dir=deps.memory_dir / "findings",
                    strategy_registry=deps.strategy_registry,
                ).build_proposals()
                if proposals:
                    package.data["transfer_proposals"] = [
                        p.model_dump(mode="json") for p in proposals
                    ]
                    for proposal in proposals:
                        pattern_id = getattr(proposal, "pattern_id", "") or ""
                        target_bot = getattr(proposal, "target_bot", "") or ""
                        target_strategy_id = getattr(proposal, "target_strategy_id", None) or ""
                        deps.ledger_write_candidate(
                            source="transfer",
                            kind_hint="structural_change",
                            bot_id=target_bot,
                            strategy_id=target_strategy_id,
                            title=f"Transfer pattern: {getattr(proposal, 'pattern_title', '') or pattern_id}",
                            description=getattr(proposal, "rationale", "") or "",
                            run_id=run_id,
                            lifecycle_stage=getattr(proposal, "category", "") or "",
                            evaluation_method="approval",
                            stable_link_key=(
                                f"transfer:{pattern_id}:{getattr(proposal, 'source_bot', '')}:"
                                f"{target_bot}:{target_strategy_id}"
                            ),
                        )
            except Exception:
                logger.error("Transfer proposal building failed - skipping", exc_info=True)

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "weekly_analysis",
            })

            result = await deps.agent_runner.invoke(
                agent_type="weekly_analysis",
                prompt_package=package,
                run_id=run_id,
                allowed_tools=["Read", "Grep", "Glob"],
            )

            final_report = result.response
            if result.success:
                from trading_assistant.analysis.response_parser import parse_response

                parsed = parse_response(result.response)
                try:
                    run_dir = result.run_dir or (deps.runs_dir / run_id)
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    (Path(run_dir) / "parsed_analysis.json").write_text(
                        parsed.model_dump_json(indent=2), encoding="utf-8",
                    )
                except Exception:
                    logger.error("Failed to save parsed analysis for %s", run_id, exc_info=True)

                if not parsed.parse_success:
                    logger.warning("No structured output block found in %s response", run_id)
                elif parsed.fallback_used:
                    logger.info("Fallback markdown extraction used for %s - structured block was missing", run_id)

                final_report, validation = deps.validate_and_annotate(
                    parsed,
                    week_start,
                    provider=result.provider,
                    model=result.effective_model,
                    run_id=run_id,
                    agent_type="weekly_analysis",
                    bot_ids=package.metadata.get("bot_ids", ""),
                )
                deps.persist_validator_notes(result.run_dir or (deps.runs_dir / run_id), validation)
                deps.refresh_run_index_entry(
                    run_id=run_id,
                    agent_type="weekly_analysis",
                    run_dir=result.run_dir or (deps.runs_dir / run_id),
                    provider=result.provider,
                    model=result.effective_model,
                    prompt_package=package,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    cost_usd=result.cost_usd,
                )

                if validation is None and (parsed.suggestions or parsed.portfolio_proposals):
                    from trading_assistant.analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                        approved_portfolio_proposals=parsed.portfolio_proposals,
                    )
                    logger.warning("Validation failed for %s - recording unvalidated suggestions", run_id)

                weekly_agent_ids = deps.record_agent_suggestions(
                    validation, run_id, parsed,
                    provider=result.provider, model=result.effective_model,
                )
                deps.record_learning_card_feedback_targeted(validation, package)
                deps.record_portfolio_proposals(validation, run_id)
                await deps.run_autonomous_pipeline(weekly_agent_ids, run_id)
                deps.record_predictions(week_start, parsed.predictions)
                deps.update_hypothesis_lifecycle(parsed, suggestion_ids)
                deps.extract_and_record_patterns(parsed, deps.bots, suggestion_ids)

            if deps.brain:
                deps.brain.record_weekly_analysis(datetime.now(timezone.utc).isoformat())

            deps.event_stream.broadcast("weekly_analysis_complete", {
                "week_start": week_start,
                "success": result.success,
            })

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            scheduled_success = result.success
            scheduled_error = result.error if not result.success else ""
            deps.record_run(
                run_id, "weekly_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
                error=scheduled_error,
            )

            if result.success:
                deps.write_run_report(run_id, "weekly_report.md", final_report)
                await deps.notify(
                    notification_type="weekly_report",
                    priority=NotificationPriority.NORMAL,
                    title=f"Weekly Report - {week_start} to {week_end}",
                    body=final_report[:2000],
                )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            deps.record_run(
                run_id, "weekly_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Weekly analysis handler failed for %s", run_id)
            deps.event_stream.broadcast("weekly_analysis_error", {
                "week_start": week_start,
                "error": str(exc),
            })
            scheduled_error = str(exc)
        finally:
            await deps.signal_scheduled_result(
                action,
                success=scheduled_success,
                error=scheduled_error,
            )
