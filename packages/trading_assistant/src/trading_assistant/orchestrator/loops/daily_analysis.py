"""Daily analysis loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
from typing import Any

from trading_assistant.analysis.context_builder import ContextBuilder
from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator
from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)

MIN_TRADES_FOR_ANALYSIS = 3
INSTRUMENTATION_READINESS_THRESHOLD = 0.4


@dataclass(frozen=True)
class DailyAnalysisDependencies:
    agent_runner: Any
    event_stream: Any
    curated_dir: Path
    memory_dir: Path
    runs_dir: Path
    bots: list[str]
    bot_configs: dict | None
    strategy_registry: object | None
    run_index: object | None
    worker: Any | None
    brain: Any | None
    record_run: Callable[..., None]
    rebuild_daily_curated_from_raw: Callable[[str, list[str]], None]
    count_daily_trades: Callable[[str], int]
    check_strategy_registry_drift: Callable[[str], None]
    validate_and_annotate: Callable[..., tuple[str, object]]
    persist_validator_notes: Callable[..., None]
    refresh_run_index_entry: Callable[..., None]
    record_agent_suggestions: Callable[..., dict]
    record_learning_card_feedback_targeted: Callable[..., None]
    run_autonomous_pipeline: Callable[[dict[str, str], str], Awaitable[None]]
    record_predictions: Callable[..., None]
    write_run_report: Callable[..., None]
    notify: Callable[..., Awaitable[None]]
    signal_scheduled_result: Callable[..., Awaitable[None]]


class DailyAnalysisLoop:
    """Run the daily analysis pipeline behind a loop-shaped module."""

    def __init__(self, dependencies: DailyAnalysisDependencies) -> None:
        self._deps = dependencies

    async def handle(self, action: Action) -> None:
        deps = self._deps
        details = action.details or {}
        date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        bots = details.get("bots", deps.bots)
        run_scope = details.get("run_scope", "")
        run_id = f"daily-{date}"
        if run_scope:
            run_id = f"{run_id}-{hashlib.sha256(run_scope.encode('utf-8')).hexdigest()[:8]}"
        start_time = datetime.now(timezone.utc)
        deps.record_run(run_id, "daily_analysis", "running", started_at=start_time.isoformat())
        scheduled_success = False
        scheduled_error = ""

        try:
            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "daily_analysis",
            })

            deps.rebuild_daily_curated_from_raw(date, bots)

            index = MemoryConsolidator.load_index(deps.runs_dir.parent)
            for bot in bots:
                avail = ContextBuilder.check_data_availability(index, bot, date)
                if avail["has_curated"] is False:
                    logger.warning("No curated data for %s on %s - analysis may be incomplete", bot, date)

            from trading_assistant.analysis.quality_gate import QualityGate

            gate = QualityGate(
                report_id=run_id,
                date=date,
                expected_bots=bots,
                curated_dir=deps.curated_dir,
            )
            checklist = gate.run()

            if not checklist.can_proceed:
                scheduled_error = f"Quality gate blocked: {', '.join(checklist.blocking_issues)}"
                logger.warning("Quality gate blocked for %s: %s", run_id, checklist.blocking_issues)
                deps.event_stream.broadcast("daily_analysis_blocked", {
                    "date": date,
                    "blocking_issues": checklist.blocking_issues,
                })
                await deps.notify(
                    notification_type="daily_report_blocked",
                    priority=NotificationPriority.LOW,
                    title=f"Daily report {date} - blocked",
                    body=f"Quality gate blocked: {', '.join(checklist.blocking_issues)}",
                )
                return

            if checklist.overall == "FAIL":
                logger.warning("Quality gate FAIL (degraded) for %s: %s", run_id, checklist.blocking_issues)
                deps.event_stream.broadcast("daily_analysis_degraded", {
                    "date": date,
                    "blocking_issues": checklist.blocking_issues,
                    "data_completeness": checklist.data_completeness,
                })

            total_trades = deps.count_daily_trades(date)
            if total_trades < MIN_TRADES_FOR_ANALYSIS:
                logger.info(
                    "Only %d trades on %s (min %d) - producing deterministic summary",
                    total_trades, date, MIN_TRADES_FOR_ANALYSIS,
                )
                completeness = getattr(checklist, "data_completeness", 0.0)
                try:
                    completeness_str = f"{completeness:.0%}"
                except (TypeError, ValueError):
                    completeness_str = str(completeness)
                body = (
                    f"Daily summary for {date}: {total_trades} trade(s) across {len(bots)} bot(s). "
                    f"Insufficient data for full analysis (minimum {MIN_TRADES_FOR_ANALYSIS} trades required). "
                    f"Data completeness: {completeness_str}."
                )
                deps.record_run(
                    run_id, "daily_analysis", "skipped",
                    started_at=start_time.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    duration_ms=int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                )
                deps.write_run_report(run_id, "daily_report.md", body, mirror_response=True)
                await deps.notify(
                    notification_type="daily_report",
                    priority=NotificationPriority.LOW,
                    title=f"Daily Summary - {date} (light day)",
                    body=body,
                )
                scheduled_success = True
                return

            event_counts: dict = {}
            if deps.worker:
                event_counts = deps.worker.get_and_reset_daily_counts()

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "quality_gate", "handler": "daily_analysis",
            })

            deps.check_strategy_registry_drift(run_id)

            from trading_assistant.analysis.daily_triage import DailyTriage
            from trading_assistant.analysis.prompt_assembler import DailyPromptAssembler

            ctx = ContextBuilder(deps.memory_dir, curated_dir=deps.curated_dir)
            try:
                readiness = ctx.load_instrumentation_readiness(bots)
                low_readiness_bots = {
                    bid: r.get("overall_score", 0)
                    for bid, r in readiness.items()
                    if r.get("overall_score", 0) < INSTRUMENTATION_READINESS_THRESHOLD
                }
                if low_readiness_bots:
                    logger.warning(
                        "Low instrumentation readiness: %s",
                        ", ".join(f"{b}={s:.0%}" for b, s in low_readiness_bots.items()),
                    )
                    deps.event_stream.broadcast("instrumentation_readiness_low", {
                        "date": date, "bots": low_readiness_bots,
                    })
            except Exception:
                logger.debug("Instrumentation readiness check skipped", exc_info=True)

            triage = DailyTriage(
                curated_dir=deps.curated_dir,
                date=date,
                bots=bots,
                active_suggestions=ctx.load_active_suggestions(),
            )
            triage_report = triage.run()

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "triage", "handler": "daily_analysis",
                "significant_events": len(triage_report.significant_events),
                "focus_questions": len(triage_report.focus_questions),
            })

            assembler = DailyPromptAssembler(
                date=date,
                bots=bots,
                curated_dir=deps.curated_dir,
                memory_dir=deps.memory_dir,
                bot_configs=deps.bot_configs,
                strategy_registry=deps.strategy_registry,
                run_index=deps.run_index,
            )
            package = assembler.assemble(
                triage_report=triage_report,
                session_store=deps.agent_runner.session_store,
            )
            if event_counts:
                package.metadata["event_counts"] = event_counts

            deps.event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "daily_analysis",
            })

            result = await deps.agent_runner.invoke(
                agent_type="daily_analysis",
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
                    date,
                    provider=result.provider,
                    model=result.effective_model,
                    run_id=run_id,
                    agent_type="daily_analysis",
                    bot_ids=package.metadata.get("bot_ids", ""),
                )
                deps.persist_validator_notes(result.run_dir or (deps.runs_dir / run_id), validation)
                deps.refresh_run_index_entry(
                    run_id=run_id,
                    agent_type="daily_analysis",
                    run_dir=result.run_dir or (deps.runs_dir / run_id),
                    provider=result.provider,
                    model=result.effective_model,
                    prompt_package=package,
                    success=result.success,
                    duration_ms=result.duration_ms,
                    cost_usd=result.cost_usd,
                )

                if validation is None and parsed.suggestions:
                    from trading_assistant.analysis.response_validator import ValidationResult
                    validation = ValidationResult(
                        approved_suggestions=parsed.suggestions,
                        approved_predictions=parsed.predictions,
                    )
                    logger.warning("Validation failed for %s - recording unvalidated suggestions", run_id)

                agent_suggestion_ids = deps.record_agent_suggestions(
                    validation, run_id, parsed,
                    provider=result.provider, model=result.effective_model,
                    source="llm_daily",
                )
                deps.record_learning_card_feedback_targeted(validation, package)
                await deps.run_autonomous_pipeline(agent_suggestion_ids, run_id)
                deps.record_predictions(date, parsed.predictions)

            if deps.brain:
                deps.brain.record_daily_analysis(datetime.now(timezone.utc).isoformat())

            deps.event_stream.broadcast("daily_analysis_complete", {
                "date": date,
                "success": result.success,
                "run_dir": str(result.run_dir),
            })

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            scheduled_success = result.success
            scheduled_error = result.error if not result.success else ""
            deps.record_run(
                run_id, "daily_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
                error=result.error if not result.success else "",
                metadata={
                    "degraded": checklist.overall == "FAIL",
                    "data_completeness": checklist.data_completeness,
                },
            )

            if result.success:
                deps.write_run_report(run_id, "daily_report.md", final_report)
                degraded = checklist.overall == "FAIL"
                title_prefix = "[DEGRADED] " if degraded else ""
                body_prefix = "DEGRADED: quality gate reported incomplete coverage.\n\n" if degraded else ""
                await deps.notify(
                    notification_type="daily_report",
                    priority=NotificationPriority.NORMAL,
                    title=f"{title_prefix}Daily Report - {date}",
                    body=body_prefix + final_report[:2000],
                )
            else:
                await deps.notify(
                    notification_type="daily_report_error",
                    priority=NotificationPriority.HIGH,
                    title=f"Daily report {date} - error",
                    body=f"Agent failed: {result.error}",
                )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            deps.record_run(
                run_id, "daily_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Daily analysis handler failed for %s", run_id)
            deps.event_stream.broadcast("daily_analysis_error", {
                "date": date,
                "error": str(exc),
            })
            scheduled_error = str(exc)
        finally:
            await deps.signal_scheduled_result(
                action,
                success=scheduled_success,
                error=scheduled_error,
            )
