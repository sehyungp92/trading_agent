"""Control-plane runtime composition surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from trading_assistant.comms.dispatcher import NotificationDispatcher
from trading_assistant.comms.telegram_handlers import TelegramCallbackRouter
from trading_assistant.comms.telegram_renderer import TelegramRenderer
from trading_assistant.orchestrator.adapters.vps_receiver import VPSReceiver
from trading_assistant.orchestrator.agent_runner import AgentRunner
from trading_assistant.orchestrator.config import (
    AppConfig,
    resolve_runtime_memory_dir,
)
from trading_assistant.orchestrator.conversation_tracker import ConversationTracker
from trading_assistant.orchestrator.cost_tracker import CostTracker
from trading_assistant.orchestrator.data_paths import resolve_data_dirs
from trading_assistant.orchestrator.db.queue import EventQueue
from trading_assistant.orchestrator.event_stream import AuditTrailConsumer, EventStream
from trading_assistant.orchestrator.handlers import Handlers
from trading_assistant.orchestrator.latency_tracker import LatencyTracker
from trading_assistant.orchestrator.memory_consolidator import MemoryConsolidator
from trading_assistant.orchestrator.monitoring import MonitoringCheck, MonitoringLoop
from trading_assistant.orchestrator.orchestrator_brain import OrchestratorBrain
from trading_assistant.orchestrator.runtime_validation import (
    RuntimeConfigError,
    validate_auth_config,
    validate_production_runtime_config,
)
from trading_assistant.orchestrator.scheduled_runs import ScheduledRunStore
from trading_assistant.orchestrator.session_store import SessionStore
from trading_assistant.orchestrator.skills_registry import SkillsRegistry
from trading_assistant.orchestrator.subagent import SubagentManager
from trading_assistant.orchestrator.task_registry import TaskRegistry
from trading_assistant.orchestrator.worker import Worker
from trading_assistant.schemas.agent_preferences import AgentPreferences
from trading_assistant.schemas.notifications import (
    ChannelConfig,
    NotificationChannel,
    NotificationPreferences,
)
from trading_assistant.skills.loop_run_ledger import RuntimeLoopProjectionWriter


@dataclass(frozen=True)
class RuntimeBuildHooks:
    load_notification_preferences: Callable[[Path], NotificationPreferences]
    save_notification_preferences: Callable[[NotificationPreferences, Path], None]
    load_agent_preferences: Callable[[Path, AppConfig], AgentPreferences]
    register_channel_adapters: Callable[[AppConfig, NotificationDispatcher], list[Any]]


@dataclass
class ControlPlaneRuntime:
    """Runtime collaborators exposed to HTTP and worker adapters."""

    config: AppConfig
    db_path: Path
    memory_dir: Path
    collaborators: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.collaborators[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def attach_state(self, app: Any) -> None:
        app.state.control_plane_runtime = self
        app.state.config = self.config
        app.state.memory_dir = self.memory_dir
        for name, collaborator in self.collaborators.items():
            setattr(app.state, name, collaborator)


def build_control_plane_runtime(
    *,
    config: AppConfig | None = None,
    db_dir: str | Path | None = None,
    hooks: RuntimeBuildHooks,
) -> ControlPlaneRuntime:
    """Build runtime collaborators without binding FastAPI routes."""
    config = config or AppConfig.from_env()
    try:
        validate_auth_config(config)
        validate_production_runtime_config(config)
    except RuntimeConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    db_dir_explicit = db_dir is not None
    db_path = Path(db_dir or config.data_dir)
    memory_dir = resolve_runtime_memory_dir(
        config,
        db_dir=db_path,
        db_dir_explicit=db_dir_explicit,
    )
    data_dirs = resolve_data_dirs(db_path)
    raw_data_dir = data_dirs.raw_data_dir
    curated_dir = data_dirs.curated_dir

    queue = EventQueue(db_path=str(db_path / "events.db"))
    registry = TaskRegistry(db_path=str(db_path / "tasks.db"))
    loop_projection_writer = RuntimeLoopProjectionWriter(memory_dir)
    scheduled_run_store = ScheduledRunStore(
        db_path=str(db_path / "scheduled_runs.db"),
        final_status_observer=loop_projection_writer.project_record,
    )
    brain = OrchestratorBrain()
    event_stream = EventStream()
    conversation_tracker = ConversationTracker()
    worker = Worker(
        queue=queue,
        registry=registry,
        brain=brain,
        event_stream=event_stream,
        conversation_tracker=conversation_tracker,
        raw_data_dir=raw_data_dir,
        bot_configs=config.bot_configs,
    )
    session_store = SessionStore(base_dir=str(db_path / ".assistant" / "sessions"))
    subagent_mgr = SubagentManager()
    latency_tracker = LatencyTracker()

    prefs_path = db_path / "data" / "notification_prefs.json"
    notification_prefs = hooks.load_notification_preferences(prefs_path)
    agent_prefs_path = db_path / "data" / "agent_preferences.json"
    agent_preferences = hooks.load_agent_preferences(agent_prefs_path, config)

    dispatcher = NotificationDispatcher()
    skills_registry = SkillsRegistry()
    cost_tracker = CostTracker(db_path / "data" / "cost_log.jsonl")
    from trading_assistant.orchestrator.run_index import RunIndex
    from trading_assistant.skills.learning_write_coordinator import LearningWriteCoordinator

    run_index = RunIndex(db_path / "data" / "run_index.db")
    write_coordinator = LearningWriteCoordinator(
        findings_dir=memory_dir / "findings",
        event_stream=event_stream,
    )
    agent_runner = AgentRunner(
        runs_dir=db_path / "runs",
        session_store=session_store,
        claude_command=config.claude_command,
        claude_command_args=config.claude_command_args,
        codex_command=config.codex_command,
        codex_command_args=config.codex_command_args,
        skills_registry=skills_registry,
        preferences=agent_preferences,
        zai_api_key=config.zai_api_key,
        openrouter_api_key=config.openrouter_api_key,
        event_stream=event_stream,
        cost_tracker=cost_tracker,
        run_index=run_index,
    )
    monitoring_loop = MonitoringLoop(
        checks=[MonitoringCheck(
            registry=registry,
            heartbeat_dir=str(db_path / "heartbeats"),
            queue=queue,
            brain=brain,
            heartbeat_md_path=str(memory_dir / "heartbeat.md"),
            relay_url=config.relay_url,
            latency_tracker=latency_tracker,
        )],
        event_stream=event_stream,
    )

    channel_adapters = hooks.register_channel_adapters(config, dispatcher)
    if not notification_prefs.channels and dispatcher.adapters:
        seeded_channels = []
        for channel in dispatcher.adapters:
            chat_id = ""
            if channel == NotificationChannel.TELEGRAM:
                chat_id = config.telegram_chat_id or ""
            elif channel == NotificationChannel.EMAIL:
                chat_id = config.email_to or ""
            seeded_channels.append(ChannelConfig(channel=channel, enabled=True, chat_id=chat_id))
        notification_prefs = NotificationPreferences(channels=seeded_channels)
        hooks.save_notification_preferences(notification_prefs, prefs_path)

    telegram_adapter = None
    for adapter in channel_adapters:
        from trading_assistant.comms.telegram_bot import TelegramBotAdapter

        if isinstance(adapter, TelegramBotAdapter):
            telegram_adapter = adapter
            break
    callback_router = TelegramCallbackRouter() if telegram_adapter is not None else None
    telegram_renderer = TelegramRenderer() if telegram_adapter is not None else None

    from trading_assistant.skills.proposal_ledger import ProposalLedger
    from trading_assistant.skills.suggestion_tracker import SuggestionTracker

    suggestion_tracker = SuggestionTracker(store_dir=memory_dir / "findings")
    proposal_ledger = ProposalLedger(store_dir=memory_dir / "findings")

    autonomous_pipeline = None
    approval_tracker = None
    approval_handler = None
    config_registry = None
    file_change_gen = None
    pr_builder = None
    repo_workspace_manager = None
    repo_task_runner = None
    calibration_tracker = None
    telegram_bot = telegram_adapter
    shared_approval_enabled = (
        config.autonomous_enabled
        or config.monthly_validation_mode == "approval_gated"
        or config.deployment_monitoring_enabled
    )
    if shared_approval_enabled:
        from trading_assistant.orchestrator.repo_task_runner import RepoTaskRunner
        from trading_assistant.skills.approval_handler import ApprovalHandler
        from trading_assistant.skills.approval_tracker import ApprovalTracker
        from trading_assistant.skills.config_registry import ConfigRegistry
        from trading_assistant.skills.file_change_generator import FileChangeGenerator
        from trading_assistant.skills.github_pr import PRBuilder
        from trading_assistant.skills.repo_workspace import RepoWorkspaceManager

        config_registry = ConfigRegistry(Path(config.bot_config_dir))
        approval_tracker = ApprovalTracker(memory_dir / "findings" / "approvals.jsonl")
        file_change_gen = FileChangeGenerator()
        pr_builder = PRBuilder(dry_run=False, github_token=config.github_token)
        repo_workspace_manager = RepoWorkspaceManager(
            cache_root=Path(config.bot_repo_cache_dir),
            task_root=db_path / "runs" / "repo_tasks",
        )
        repo_task_runner = RepoTaskRunner(agent_runner)
        approval_handler = ApprovalHandler(
            approval_tracker=approval_tracker,
            suggestion_tracker=suggestion_tracker,
            file_change_generator=file_change_gen,
            pr_builder=pr_builder,
            config_registry=config_registry,
            event_stream=event_stream,
            telegram_bot=telegram_bot,
            repo_workspace_manager=repo_workspace_manager,
            repo_task_runner=repo_task_runner,
        )

    if config.autonomous_enabled:
        from trading_assistant.skills.autonomous_pipeline import AutonomousPipeline
        from trading_assistant.skills.suggestion_backtester import SuggestionBacktester

        if shared_approval_enabled and approval_tracker is not None and config_registry is not None:
            backtester = SuggestionBacktester(config_registry, db_path)
            from trading_assistant.skills.backtest_calibration_tracker import BacktestCalibrationTracker
            from trading_assistant.skills.backtest_simulator import BacktestSimulator
            from trading_assistant.skills.cost_model import CostModel
            from trading_assistant.skills.parameter_searcher import ParameterSearcher

            cost_cfg = getattr(config, "cost_model_config", None)
            if cost_cfg:
                cost_model = CostModel(cost_cfg)
                simulator = BacktestSimulator(cost_model)
                parameter_searcher = ParameterSearcher(
                    config_registry=config_registry,
                    simulator=simulator,
                    cost_model=cost_model,
                )
            else:
                parameter_searcher = None
            calibration_tracker = BacktestCalibrationTracker(store_dir=memory_dir / "findings")
            autonomous_pipeline = AutonomousPipeline(
                config_registry=config_registry,
                backtester=backtester,
                approval_tracker=approval_tracker,
                suggestion_tracker=suggestion_tracker,
                telegram_bot=telegram_bot,
                telegram_renderer=telegram_renderer,
                event_stream=event_stream,
                parameter_searcher=parameter_searcher,
                experiment_config_generator=None,
                proposal_ledger=proposal_ledger,
                calibration_tracker=calibration_tracker,
                search_log_dir=memory_dir / "findings",
                curated_dir=curated_dir,
            )

    threshold_learner = None
    if config.adaptive_thresholds_enabled:
        from trading_assistant.skills.threshold_learner import ThresholdLearner

        threshold_learner = ThresholdLearner(findings_dir=memory_dir / "findings")

    deployment_monitor = None
    if config.deployment_monitoring_enabled:
        from trading_assistant.skills.deployment_monitor import DeploymentMonitor

        deployment_monitor = DeploymentMonitor(
            findings_dir=memory_dir / "findings",
            curated_dir=curated_dir,
            pr_builder=pr_builder,
            config_registry=config_registry,
            event_stream=event_stream,
            file_change_generator=file_change_gen,
        )

    experiment_manager = None
    experiment_config_gen = None
    if config.ab_testing_enabled:
        from trading_assistant.skills.experiment_config_generator import ExperimentConfigGenerator
        from trading_assistant.skills.experiment_manager import ExperimentManager

        experiment_manager = ExperimentManager(findings_dir=memory_dir / "findings")
        experiment_config_gen = ExperimentConfigGenerator(config_registry=config_registry)

    from trading_assistant.skills.reliability_tracker import ReliabilityTracker
    from trading_assistant.skills.structural_experiment_tracker import StructuralExperimentTracker

    reliability_tracker = ReliabilityTracker(store_dir=memory_dir / "findings")
    structural_experiment_tracker = StructuralExperimentTracker(
        store_dir=memory_dir / "findings",
    )
    if approval_handler is not None:
        approval_handler._structural_experiment_tracker = structural_experiment_tracker

    handlers = Handlers(
        agent_runner=agent_runner,
        event_stream=event_stream,
        dispatcher=dispatcher,
        notification_prefs=notification_prefs,
        curated_dir=curated_dir,
        raw_data_dir=raw_data_dir,
        memory_dir=memory_dir,
        runs_dir=db_path / "runs",
        source_root=db_path,
        bots=config.bot_ids,
        heartbeat_dir=db_path / "heartbeats",
        failure_log_path=db_path / "data" / "failure_log.jsonl",
        worker=worker,
        brain=brain,
        suggestion_tracker=suggestion_tracker,
        autonomous_pipeline=autonomous_pipeline,
        approval_handler=approval_handler,
        approval_tracker=approval_tracker,
        pr_builder=pr_builder,
        config_registry=config_registry,
        repo_workspace_manager=repo_workspace_manager,
        deployment_monitor=deployment_monitor,
        threshold_learner=threshold_learner,
        experiment_manager=experiment_manager,
        experiment_config_gen=experiment_config_gen,
        bot_configs=config.bot_configs,
        reliability_tracker=reliability_tracker,
        structural_experiment_tracker=structural_experiment_tracker,
        strategy_registry=config.strategy_registry,
        run_index=run_index,
        proposal_ledger=proposal_ledger,
        calibration_tracker=calibration_tracker,
        scheduled_run_store=scheduled_run_store,
        market_data_root=config.market_data_root,
        backtest_repo_path=config.backtest_repo_path,
        backtest_artifact_root=config.backtest_artifact_root,
        monthly_validation_mode=config.monthly_validation_mode,
        monthly_optimizer_sequence_enabled=config.monthly_optimizer_sequence_enabled,
        monthly_backtest_command=config.monthly_backtest_command,
        monthly_workflow_contract_path=config.monthly_workflow_contract_path,
        monthly_workflow_contract_version=config.monthly_workflow_contract_version,
        monthly_strategy_plugin_contract_path=config.monthly_strategy_plugin_contract_path,
        market_data_required_coverage_ratio=config.market_data_required_coverage_ratio,
        telemetry_required_lineage_ratio=config.telemetry_required_lineage_ratio,
        backtest_command_timeout_seconds=config.backtest_command_timeout_seconds,
        backtest_max_parallel_strategies=config.backtest_max_parallel_strategies,
    )
    if config.autonomous_enabled and autonomous_pipeline is not None:
        autonomous_pipeline._experiment_tracker = structural_experiment_tracker
        if experiment_config_gen is not None:
            autonomous_pipeline._experiment_config_generator = experiment_config_gen
        if experiment_manager is not None:
            autonomous_pipeline._experiment_manager = experiment_manager
    from trading_assistant.orchestrator.loops import MonthlyValidationLoop
    from trading_assistant.orchestrator.loops.monthly_services import (
        MonthlyRunRecorder,
        ScheduledMonthlyProjection,
    )
    from trading_assistant.orchestrator.loops.monthly_validation import (
        MonthlyValidationDependencies,
    )

    monthly_recorder = MonthlyRunRecorder(
        run_history_path=db_path / "data" / "run_history.jsonl",
        runs_dir=db_path / "runs",
    )
    monthly_projection = ScheduledMonthlyProjection(
        scheduled_run_store=scheduled_run_store,
        memory_dir=memory_dir,
    )
    monthly_market_data_root = (
        Path(config.market_data_root)
        if str(config.market_data_root)
        else db_path / "market_data"
    )
    monthly_backtest_repo_raw = str(config.backtest_repo_path).strip()
    monthly_backtest_repo_path = (
        Path(monthly_backtest_repo_raw) if monthly_backtest_repo_raw else ""
    )
    monthly_backtest_artifact_root = (
        Path(config.backtest_artifact_root)
        if str(config.backtest_artifact_root)
        else db_path / "backtest_artifacts"
    )
    monthly_validation_loop = MonthlyValidationLoop(
        MonthlyValidationDependencies(
            agent_runner=agent_runner,
            event_stream=event_stream,
            curated_dir=curated_dir,
            memory_dir=memory_dir,
            market_data_root=monthly_market_data_root,
            backtest_repo_path=monthly_backtest_repo_path,
            backtest_artifact_root=monthly_backtest_artifact_root,
            strategy_registry=config.strategy_registry,
            proposal_ledger=proposal_ledger,
            approval_tracker=approval_tracker,
            monthly_validation_mode=config.monthly_validation_mode,
            monthly_approval_scope_allowlist=list(config.monthly_approval_scope_allowlist),
            monthly_optimizer_sequence_enabled=config.monthly_optimizer_sequence_enabled,
            monthly_backtest_command=list(config.monthly_backtest_command or []),
            monthly_workflow_contract_path=config.monthly_workflow_contract_path,
            monthly_workflow_contract_version=config.monthly_workflow_contract_version,
            monthly_strategy_plugin_contract_path=config.monthly_strategy_plugin_contract_path,
            market_data_required_coverage_ratio=config.market_data_required_coverage_ratio,
            telemetry_required_lineage_ratio=config.telemetry_required_lineage_ratio,
            backtest_command_timeout_seconds=config.backtest_command_timeout_seconds,
            backtest_max_parallel_strategies=max(1, int(config.backtest_max_parallel_strategies or 1)),
            record_run=monthly_recorder.record_run,
            write_artifact_index=monthly_recorder.write_artifact_index,
            signal_scheduled_result=monthly_projection.signal_result,
            project_scheduled_results=monthly_projection.project_results,
        )
    )
    daily_analysis_loop = handlers.daily_analysis_loop
    weekly_analysis_loop = handlers.weekly_analysis_loop
    handlers.set_monthly_validation_loop(monthly_validation_loop)

    if config.relay_url:
        vps_receiver = VPSReceiver(
            relay_url=config.relay_url,
            local_queue=queue,
            api_key=config.relay_api_key,
            latency_tracker=latency_tracker,
            allowed_bot_ids=set(config.bot_ids),
        )
    else:
        vps_receiver = None

    audit_consumer = AuditTrailConsumer(log_dir=db_path / "logs")
    consolidator = MemoryConsolidator(
        findings_dir=memory_dir / "findings",
        base_dir=db_path,
    )
    from trading_assistant.skills.hypothesis_library import HypothesisLibrary
    from trading_assistant.skills.playbook_generator import PlaybookGenerator

    hypothesis_library = HypothesisLibrary(memory_dir / "findings")
    if approval_handler is not None:
        approval_handler._hypothesis_library = hypothesis_library
    playbook_generator = PlaybookGenerator(memory_dir=memory_dir)

    return ControlPlaneRuntime(
        config=config,
        db_path=db_path,
        memory_dir=memory_dir,
        collaborators={
            "queue": queue,
            "registry": registry,
            "scheduled_run_store": scheduled_run_store,
            "worker": worker,
            "event_stream": event_stream,
            "session_store": session_store,
            "subagent_mgr": subagent_mgr,
            "dispatcher": dispatcher,
            "agent_runner": agent_runner,
            "agent_preferences": agent_runner.get_preferences(),
            "brain": brain,
            "handlers": handlers,
            "daily_analysis_loop": daily_analysis_loop,
            "weekly_analysis_loop": weekly_analysis_loop,
            "monthly_validation_loop": monthly_validation_loop,
            "monitoring_loop": monitoring_loop,
            "notification_preferences": notification_prefs,
            "prefs_path": prefs_path,
            "agent_prefs_path": agent_prefs_path,
            "channel_adapters": channel_adapters,
            "telegram_adapter": telegram_adapter,
            "telegram_callback_router": callback_router,
            "telegram_renderer": telegram_renderer,
            "vps_receiver": vps_receiver,
            "latency_tracker": latency_tracker,
            "consolidator": consolidator,
            "audit_consumer": audit_consumer,
            "conversation_tracker": conversation_tracker,
            "skills_registry": skills_registry,
            "autonomous_pipeline": autonomous_pipeline,
            "approval_tracker": approval_tracker,
            "approval_handler": approval_handler,
            "config_registry": config_registry,
            "file_change_gen": file_change_gen,
            "pr_builder": pr_builder,
            "repo_workspace_manager": repo_workspace_manager,
            "repo_task_runner": repo_task_runner,
            "deployment_monitor": deployment_monitor,
            "threshold_learner": threshold_learner,
            "experiment_manager": experiment_manager,
            "experiment_config_gen": experiment_config_gen,
            "reliability_tracker": reliability_tracker,
            "structural_experiment_tracker": structural_experiment_tracker,
            "suggestion_tracker": suggestion_tracker,
            "proposal_ledger": proposal_ledger,
            "calibration_tracker": calibration_tracker,
            "write_coordinator": write_coordinator,
            "run_index": run_index,
            "curated_dir": curated_dir,
            "raw_data_dir": raw_data_dir,
            "hypothesis_library": hypothesis_library,
            "playbook_generator": playbook_generator,
            "data_dirs": data_dirs,
        },
    )
