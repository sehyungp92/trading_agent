"""Monthly validation loop."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from trading_assistant.orchestrator.loops.monthly_services import (
    bool_detail,
    monthly_stage_status_for_result,
    optional_date,
    optional_path,
    positive_int,
    string_list,
)
from trading_assistant.orchestrator.orchestrator_brain import Action

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonthlyValidationDependencies:
    agent_runner: Any
    event_stream: Any
    curated_dir: Path
    memory_dir: Path
    market_data_root: Path
    backtest_repo_path: Path | str
    backtest_artifact_root: Path
    strategy_registry: Any | None
    proposal_ledger: Any | None
    approval_tracker: Any | None
    monthly_validation_mode: str
    monthly_approval_scope_allowlist: list[str]
    monthly_optimizer_sequence_enabled: bool
    monthly_backtest_command: list[str]
    monthly_workflow_contract_path: str
    monthly_workflow_contract_version: str
    monthly_strategy_plugin_contract_path: str
    market_data_required_coverage_ratio: float
    telemetry_required_lineage_ratio: float
    backtest_command_timeout_seconds: int
    backtest_max_parallel_strategies: int
    record_run: Callable[..., None]
    write_artifact_index: Callable[..., Path]
    signal_scheduled_result: Callable[..., Awaitable[None]]
    project_scheduled_results: Callable[..., Awaitable[None]]


class MonthlyValidationLoop:
    """Own the monthly validation action flow behind the handler facade."""

    def __init__(self, dependencies: MonthlyValidationDependencies) -> None:
        self._deps = dependencies

    async def handle(self, action: Action) -> None:
        """Run Phase-1 monthly validation in shadow/fail-closed mode."""
        deps = self._deps
        details = action.details or {}
        bot_id = details.get("bot_id") or action.bot_id
        strategy_id = details.get("strategy_id", "")
        run_month = details.get("run_month", "")
        run_label = run_month or "latest"
        run_id = f"monthly-{bot_id}-{strategy_id or 'all'}-{run_label}"
        start_time = datetime.now(timezone.utc)
        deps.record_run(run_id, "monthly_validation", "running", started_at=start_time.isoformat())
        scheduled_success = False
        scheduled_error = ""
        results = []
        monthly_run_dir: Path | None = None
        monthly_duration_ms = 0
        try:
            if strategy_id:
                strategies = [strategy_id]
            elif deps.strategy_registry is not None:
                lookup = getattr(deps.strategy_registry, "strategies_for_bot", None)
                strategies = list(lookup(bot_id).keys()) if callable(lookup) else []
            else:
                strategies = []
            if not strategies:
                strategies = [bot_id]

            from trading_assistant.skills.monthly_validation_orchestrator import (
                MonthlyValidationOrchestrator,
                MonthlyValidationRequest,
            )

            orchestrator = MonthlyValidationOrchestrator(
                curated_dir=deps.curated_dir,
                findings_dir=deps.memory_dir / "findings",
                market_data_root=deps.market_data_root,
                backtest_repo_path=deps.backtest_repo_path,
                backtest_artifact_root=deps.backtest_artifact_root,
                required_market_coverage_ratio=deps.market_data_required_coverage_ratio,
                required_lineage_ratio=deps.telemetry_required_lineage_ratio,
                timeout_seconds=deps.backtest_command_timeout_seconds,
                proposal_ledger=deps.proposal_ledger,
                approval_tracker=deps.approval_tracker,
                model_review_invoker=self._monthly_model_review_invoker(),
            )
            requested_strategy_parallelism = positive_int(
                details.get("max_parallel_strategies"),
                default=deps.backtest_max_parallel_strategies,
            )
            strategy_parallelism = min(
                deps.backtest_max_parallel_strategies,
                requested_strategy_parallelism,
            )
            strategy_semaphore = asyncio.Semaphore(strategy_parallelism)
            approval_allowlist = set(deps.monthly_approval_scope_allowlist or [])

            async def _run_strategy(sid: str):
                async with strategy_semaphore:
                    approval_scope = str(details.get("approval_scope") or sid)
                    approval_scope_allowed = bool(
                        approval_scope and approval_scope in approval_allowlist
                    )
                    requested_shadow = bool_detail(
                        details,
                        "shadow",
                        deps.monthly_validation_mode != "approval_gated",
                    )
                    shadow = (
                        True
                        if deps.monthly_validation_mode == "approval_gated"
                        and not approval_scope_allowed
                        else requested_shadow
                    )
                    approval_evidence_mode = (
                        bool_detail(
                            details,
                            "approval_evidence_mode",
                            False,
                        )
                        and approval_scope_allowed
                    )
                    deps.event_stream.broadcast(
                        "monthly_validation_progress",
                        {
                            "bot_id": bot_id,
                            "strategy_id": sid,
                            "run_month": run_month,
                            "stage": "started",
                        },
                    )
                    result = await asyncio.to_thread(
                        orchestrator.run,
                        MonthlyValidationRequest(
                            bot_id=bot_id,
                            strategy_id=sid,
                            run_month=run_month,
                            strategy_version=str(details.get("strategy_version", "")),
                            config_version=str(details.get("config_version", "")),
                            config_hash=str(details.get("config_hash", "")),
                            deployment_id=str(details.get("deployment_id", "")),
                            parameter_set_id=str(details.get("parameter_set_id", "")),
                            market_data_manifest_path=optional_path(
                                details.get("market_data_manifest_path")
                            ),
                            data_bundle_manifest_path=optional_path(
                                details.get("data_bundle_manifest_path")
                            ),
                            data_bundle_checksum=str(details.get("data_bundle_checksum") or ""),
                            telemetry_manifest_path=optional_path(
                                details.get("telemetry_manifest_path")
                            ),
                            backtest_command=(
                                string_list(details.get("backtest_command"))
                                or deps.monthly_backtest_command
                                or None
                            ),
                            optimizer_sequence_enabled=bool_detail(
                                details,
                                "optimizer_sequence_enabled",
                                deps.monthly_optimizer_sequence_enabled,
                            ),
                            in_sample_start=optional_date(details.get("in_sample_start")),
                            in_sample_end=optional_date(details.get("in_sample_end")),
                            strategy_plugin_id=str(details.get("strategy_plugin_id", "")),
                            strategy_plugin_contract_path=(
                                optional_path(details.get("strategy_plugin_contract_path"))
                                or optional_path(deps.monthly_strategy_plugin_contract_path)
                            ),
                            round_id=str(details.get("round_id", "")),
                            prior_round_id=str(details.get("prior_round_id", "")),
                            next_round_id=str(details.get("next_round_id", "")),
                            round_n_strategy_config_path=str(
                                details.get("round_n_strategy_config_path", "")
                            ),
                            round_n_strategy_config_version=str(
                                details.get("round_n_strategy_config_version", "")
                            ),
                            round_n_portfolio_config_path=str(
                                details.get("round_n_portfolio_config_path", "")
                            ),
                            round_n_portfolio_config_version=str(
                                details.get("round_n_portfolio_config_version", "")
                            ),
                            trading_repo_path=str(details.get("trading_repo_path", "")),
                            trading_repo_branch=str(details.get("trading_repo_branch", "")),
                            trading_repo_commit_sha=str(details.get("trading_repo_commit_sha", "")),
                            deployment_metadata_path=optional_path(
                                details.get("deployment_metadata_path")
                            ),
                            deployment_metadata_install_report_paths=[
                                path
                                for path in (
                                    optional_path(item)
                                    for item in details.get(
                                        "deployment_metadata_install_report_paths",
                                        [],
                                    )
                                )
                                if path is not None
                            ],
                            operational_evidence_path=optional_path(
                                details.get("operational_evidence_path")
                            ),
                            relay_ingest_evidence_path=optional_path(
                                details.get("relay_ingest_evidence_path")
                            ),
                            vps_host_id=str(details.get("vps_host_id", "")),
                            assistant_host_id=str(
                                details.get("assistant_host_id", "local")
                            ),
                            workflow_contract_path=(
                                str(details.get("workflow_contract_path", ""))
                                or deps.monthly_workflow_contract_path
                            ),
                            workflow_contract_version=(
                                str(details.get("workflow_contract_version", ""))
                                or deps.monthly_workflow_contract_version
                            ),
                            max_workers=positive_int(details.get("max_workers"), default=2),
                            shadow=shadow,
                            approval_evidence_mode=approval_evidence_mode,
                        ),
                    )
                    deps.event_stream.broadcast(
                        "monthly_validation_progress",
                        {
                            "bot_id": bot_id,
                            "strategy_id": sid,
                            "run_month": result.run_month,
                            "status": result.status.value,
                            "approval_ready_candidate_count": result.approval_ready_candidate_count,
                            "approval_request_ids": result.approval_request_ids,
                            "monthly_stage_status": monthly_stage_status_for_result(result),
                            "stage": "completed",
                        },
                    )
                    return result

            strategy_outcomes = await asyncio.gather(
                *(_run_strategy(sid) for sid in strategies),
                return_exceptions=True,
            )
            results.extend(
                outcome for outcome in strategy_outcomes if not isinstance(outcome, Exception)
            )
            first_error = next(
                (outcome for outcome in strategy_outcomes if isinstance(outcome, Exception)),
                None,
            )
            if first_error is not None:
                raise first_error

            finished_at = datetime.now(timezone.utc)
            elapsed = int((finished_at - start_time).total_seconds() * 1000)
            artifact_index_run_dir = deps.write_artifact_index(
                run_id=run_id,
                started_at=start_time.isoformat(),
                finished_at=finished_at.isoformat(),
                results=results,
            )
            monthly_run_dir = artifact_index_run_dir
            monthly_duration_ms = elapsed
            deps.record_run(
                run_id,
                "monthly_validation",
                "completed",
                started_at=start_time.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_ms=elapsed,
                metadata={
                    "results": [r.model_dump(mode="json") for r in results],
                    "artifact_index_run_dir": str(artifact_index_run_dir),
                },
            )
            scheduled_success = True
        except Exception as exc:
            scheduled_error = str(exc)
            finished_at = datetime.now(timezone.utc)
            elapsed = int((finished_at - start_time).total_seconds() * 1000)
            metadata = None
            if results:
                try:
                    artifact_index_run_dir = deps.write_artifact_index(
                        run_id=run_id,
                        started_at=start_time.isoformat(),
                        finished_at=finished_at.isoformat(),
                        results=results,
                    )
                    monthly_run_dir = artifact_index_run_dir
                    monthly_duration_ms = elapsed
                    metadata = {
                        "results": [r.model_dump(mode="json") for r in results],
                        "artifact_index_run_dir": str(artifact_index_run_dir),
                        "partial_results": True,
                    }
                except Exception:
                    logger.warning("Failed to write partial monthly artifact index for %s", run_id)
            deps.record_run(
                run_id,
                "monthly_validation",
                "failed",
                started_at=start_time.isoformat(),
                finished_at=finished_at.isoformat(),
                duration_ms=elapsed,
                error=scheduled_error,
                metadata=metadata,
            )
            logger.exception("Monthly validation handler failed for %s", run_id)
            raise
        finally:
            await deps.signal_scheduled_result(
                action,
                success=scheduled_success,
                error=scheduled_error,
            )
            await deps.project_scheduled_results(
                action,
                results=results,
                run_id=run_id,
                run_dir=monthly_run_dir,
                duration_ms=monthly_duration_ms,
                error=scheduled_error,
            )

    def _monthly_model_review_invoker(self):
        deps = self._deps
        """Return a sync bridge used by the threaded monthly orchestrator."""
        from trading_assistant.skills.monthly_model_review_runner import (
            MonthlyModelReviewInvocationResult,
        )

        loop = asyncio.get_running_loop()
        timeout_seconds = max(60, min(deps.backtest_command_timeout_seconds, 1800))

        def _invoke(prompt_package, run_id: str):
            future = asyncio.run_coroutine_threadsafe(
                deps.agent_runner.invoke(
                    agent_type="monthly_model_review",
                    prompt_package=prompt_package,
                    run_id=run_id,
                    allowed_tools=[],
                ),
                loop,
            )
            try:
                result = future.result(timeout=timeout_seconds)
            except FutureTimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"monthly model review timed out after {timeout_seconds}s"
                ) from exc
            if not result.success:
                raise RuntimeError(result.error or "monthly model review failed")
            return MonthlyModelReviewInvocationResult(
                response=result.response,
                provider=result.provider,
                model=result.effective_model,
                runtime=result.runtime,
                cost_usd=result.cost_usd,
            )

        return _invoke
