"""Compatibility composition for orchestrator action handlers."""

from __future__ import annotations

from pathlib import Path

from trading_assistant.orchestrator.action_handlers import (
    AutonomousSupportActions,
    BugTriageActions,
    CoreHandlerSupport,
    DailyDataSupport,
    DeploymentMonitoringActions,
    DiscoveryActions,
    FeedbackActions,
    PortfolioProposalActions,
    ProposalLedgerActions,
    StructuralExperimentActions,
    WeeklyAggregateSupport,
    WeeklyAllocationSupport,
    WeeklyEventLoadingSupport,
    WeeklyEvidenceLoadingSupport,
    WeeklySimulationSupport,
)
from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.event_stream import EventStream
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.schemas.notifications import NotificationPreferences


class Handlers(
    CoreHandlerSupport,
    BugTriageActions,
    DiscoveryActions,
    FeedbackActions,
    PortfolioProposalActions,
    ProposalLedgerActions,
    StructuralExperimentActions,
    AutonomousSupportActions,
    WeeklySimulationSupport,
    WeeklyAllocationSupport,
    WeeklyEventLoadingSupport,
    WeeklyEvidenceLoadingSupport,
    WeeklyAggregateSupport,
    DailyDataSupport,
    DeploymentMonitoringActions,
):
    """Compatibility adapter that composes extracted action-handler modules."""

    def __init__(
        self,
        agent_runner: AgentRunner,
        event_stream: EventStream,
        dispatcher: object,  # NotificationDispatcher
        notification_prefs: NotificationPreferences,
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        source_root: Path,
        bots: list[str],
        raw_data_dir: Path | None = None,
        heartbeat_dir: Path | None = None,
        failure_log_path: Path | None = None,
        worker: Worker | None = None,
        brain: OrchestratorBrain | None = None,
        run_history_path: Path | None = None,
        suggestion_tracker: object | None = None,
        autonomous_pipeline: object | None = None,
        approval_handler: object | None = None,
        approval_tracker: object | None = None,
        pr_builder: object | None = None,
        config_registry: object | None = None,
        repo_workspace_manager: object | None = None,
        deployment_monitor: object | None = None,
        threshold_learner: object | None = None,
        experiment_manager: object | None = None,
        experiment_config_gen: object | None = None,
        bot_configs: dict | None = None,
        reliability_tracker: object | None = None,
        structural_experiment_tracker: object | None = None,
        strategy_registry: object | None = None,
        run_index: object | None = None,
        proposal_ledger: object | None = None,
        calibration_tracker: object | None = None,
        scheduled_run_store: object | None = None,
        market_data_root: Path | str = "",
        backtest_repo_path: Path | str = "",
        backtest_artifact_root: Path | str = "",
        monthly_validation_mode: str = "disabled",
        monthly_optimizer_sequence_enabled: bool = True,
        monthly_backtest_command: list[str] | None = None,
        monthly_workflow_contract_path: str = "",
        monthly_workflow_contract_version: str = "",
        monthly_strategy_plugin_contract_path: str = "",
        market_data_required_coverage_ratio: float = 0.95,
        telemetry_required_lineage_ratio: float = 0.95,
        backtest_command_timeout_seconds: int = 3600,
        backtest_max_parallel_strategies: int = 1,
    ) -> None:
        self._agent_runner = agent_runner
        self._event_stream = event_stream
        self._dispatcher = dispatcher
        self._notification_prefs = notification_prefs
        self._curated_dir = Path(curated_dir)
        self._raw_data_dir = (
            Path(raw_data_dir) if raw_data_dir is not None else self._curated_dir.parent / "raw"
        )
        self._memory_dir = Path(memory_dir)
        self._runs_dir = Path(runs_dir)
        self._source_root = Path(source_root)
        self._bots = bots
        self._heartbeat_dir = Path(heartbeat_dir) if heartbeat_dir else self._runs_dir.parent / "heartbeats"
        self._failure_log_path = failure_log_path or (self._runs_dir.parent / "data" / "failure_log.jsonl")
        self._worker = worker
        self._brain = brain
        self._run_history_path = run_history_path or (self._runs_dir.parent / "data" / "run_history.jsonl")
        self._suggestion_tracker = suggestion_tracker
        self._autonomous_pipeline = autonomous_pipeline
        self._approval_handler = approval_handler
        self._approval_tracker = approval_tracker
        self._pr_builder = pr_builder
        self._config_registry = config_registry
        self._repo_workspace_manager = repo_workspace_manager
        self._deployment_monitor = deployment_monitor
        self._threshold_learner = threshold_learner
        self._experiment_manager = experiment_manager
        self._experiment_config_gen = experiment_config_gen
        self._bot_configs = bot_configs
        self._reliability_tracker = reliability_tracker
        self._structural_experiment_tracker = structural_experiment_tracker
        self._strategy_registry = strategy_registry
        self._run_index = run_index
        self._proposal_ledger = proposal_ledger
        self._calibration_tracker = calibration_tracker
        self._scheduled_run_store = scheduled_run_store
        self._market_data_root = (
            Path(market_data_root)
            if str(market_data_root)
            else self._runs_dir.parent / "market_data"
        )
        backtest_repo_raw = str(backtest_repo_path).strip()
        self._backtest_repo_path = Path(backtest_repo_raw) if backtest_repo_raw else ""
        self._backtest_artifact_root = (
            Path(backtest_artifact_root)
            if str(backtest_artifact_root)
            else self._runs_dir.parent / "backtest_artifacts"
        )
        self._monthly_validation_mode = monthly_validation_mode
        self._monthly_optimizer_sequence_enabled = monthly_optimizer_sequence_enabled
        self._monthly_backtest_command = list(monthly_backtest_command or [])
        self._monthly_workflow_contract_path = monthly_workflow_contract_path
        self._monthly_workflow_contract_version = monthly_workflow_contract_version
        self._monthly_strategy_plugin_contract_path = monthly_strategy_plugin_contract_path
        self._market_data_required_coverage_ratio = market_data_required_coverage_ratio
        self._telemetry_required_lineage_ratio = telemetry_required_lineage_ratio
        self._backtest_command_timeout_seconds = backtest_command_timeout_seconds
        self._backtest_max_parallel_strategies = max(1, int(backtest_max_parallel_strategies or 1))
        self._daily_analysis_loop = self._build_daily_analysis_loop()
        self._weekly_analysis_loop = self._build_weekly_analysis_loop()
        self._monthly_validation_loop = self._build_monthly_validation_loop()
