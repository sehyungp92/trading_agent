from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from trading_assistant.analysis.context_sources import LoopContractContextSource
from trading_assistant.analysis.detectors import DEFAULT_DETECTOR_CATALOG
from trading_assistant.orchestrator.config import AppConfig
from trading_assistant.orchestrator.runtime import (
    ControlPlaneRuntime,
    RuntimeBuildHooks,
    build_control_plane_runtime,
)
from trading_assistant.schemas.agent_preferences import AgentPreferences
from trading_assistant.schemas.notifications import NotificationPreferences


def test_control_plane_runtime_attaches_compatible_state(tmp_path: Path) -> None:
    app = SimpleNamespace(state=SimpleNamespace())
    runtime = ControlPlaneRuntime(
        config=AppConfig(data_dir=str(tmp_path)),
        db_path=tmp_path,
        memory_dir=tmp_path / "memory",
        collaborators={"queue": object(), "handlers": object()},
    )

    runtime.attach_state(app)

    assert app.state.control_plane_runtime is runtime
    assert app.state.config is runtime.config
    assert app.state.memory_dir == tmp_path / "memory"
    assert hasattr(app.state, "queue")
    assert hasattr(app.state, "handlers")


def test_control_plane_runtime_builder_owns_core_collaborators(tmp_path: Path) -> None:
    runtime = build_control_plane_runtime(
        config=AppConfig(
            data_dir=str(tmp_path),
            allow_unauthenticated_local=True,
        ),
        hooks=RuntimeBuildHooks(
            load_notification_preferences=lambda path: NotificationPreferences(),
            save_notification_preferences=lambda prefs, path: None,
            load_agent_preferences=lambda path, config: AgentPreferences(),
            register_channel_adapters=lambda config, dispatcher: [],
        ),
    )

    assert runtime.queue is not None
    assert runtime.registry is not None
    assert runtime.worker is not None
    assert runtime.handlers is not None
    assert runtime.daily_analysis_loop is not None
    assert runtime.weekly_analysis_loop is not None
    assert runtime.monthly_validation_loop is not None
    assert runtime.dispatcher is not None


def test_app_delegates_worker_and_callback_wiring() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "packages/trading_assistant/src/trading_assistant/orchestrator/app.py").read_text(
        encoding="utf-8"
    )
    source = source[source.index("def create_app"):source.index("# Default app instance")]

    assert "wire_worker_dispatch(runtime)" in source
    assert "register_runtime_callbacks(" in source
    assert "build_runtime_scheduler_wiring(" in source
    assert "build_runtime_lifespan(runtime, scheduler_wiring)" in source
    assert "create_scheduler(" not in source
    assert "run_morning_scan(" not in source
    assert "run_evening_report(" not in source
    assert "run_experiment_checks(" not in source
    assert "run_market_data_sync(" not in source
    assert "run_lineage_audit(" not in source
    assert "def _create_scheduler" not in source
    assert "ProactiveScanner(" not in source
    assert "MarketDataSyncJob(" not in source
    assert "LineageAuditor(" not in source
    assert "AutoOutcomeMeasurer(" not in source
    assert "OutcomeReasoningAssembler(" not in source
    assert "LearningCycle(" not in source
    assert "TransferProposalBuilder(" not in source
    assert "GroundTruthComputer(" not in source
    assert "worker.on_triage" not in source
    assert "approve_suggestion_" not in source
    assert "agent_settings_set_" not in source


def test_monthly_validation_loop_uses_dependency_boundary() -> None:
    from trading_assistant.orchestrator.loops import monthly_validation

    source = inspect.getsource(monthly_validation)

    assert "orchestrator.handlers" not in source
    assert "Handlers" not in source
    assert "._record_run" not in source
    assert "._write_monthly_artifact_index" not in source
    assert "._project_monthly_scheduled_results" not in source


def test_daily_weekly_worker_dispatch_uses_loop_boundaries() -> None:
    from trading_assistant.orchestrator import runtime, runtime_dispatch
    from trading_assistant.orchestrator.loops import daily_analysis, weekly_analysis

    dispatch_source = inspect.getsource(runtime_dispatch)
    assert "daily_analysis_loop.handle" in dispatch_source
    assert "weekly_analysis_loop.handle" in dispatch_source
    assert "handlers.handle_daily_analysis" not in dispatch_source
    assert "handlers.handle_weekly_analysis" not in dispatch_source

    runtime_source = inspect.getsource(runtime)
    assert "handlers.handle_daily_analysis" not in runtime_source
    assert "handlers.handle_weekly_analysis" not in runtime_source
    assert "DailyAnalysisDependencies(run=" not in runtime_source
    assert "WeeklyAnalysisDependencies(run=" not in runtime_source

    for module in (daily_analysis, weekly_analysis):
        source = inspect.getsource(module)
        assert "orchestrator.handlers" not in source
        assert "Handlers" not in source
        assert "run: Callable[[Action]" not in source
        assert "self._deps.run" not in source
    assert "DailyTriage" in inspect.getsource(daily_analysis)
    assert "StrategyEngine" in inspect.getsource(weekly_analysis)


def test_handlers_module_is_logic_free_facade() -> None:
    from trading_assistant.orchestrator import handlers

    source = inspect.getsource(handlers)
    assert "_HandlerImplementation" in source
    assert "async def handle_daily_analysis" not in source
    assert "async def handle_weekly_analysis" not in source
    assert "def _run_weekly_simulations" not in source


def test_handler_implementation_is_composition_adapter() -> None:
    from trading_assistant.orchestrator import _handler_implementation
    from trading_assistant.orchestrator import action_handlers

    source = inspect.getsource(_handler_implementation)
    assert len(source.splitlines()) < 220
    for forbidden in (
        "async def handle_triage",
        "async def handle_feedback",
        "async def handle_discovery_analysis",
        "def _run_allocation_analyses",
        "def _check_deployments",
        "def _run_weekly_simulations",
        "def _record_agent_suggestions",
    ):
        assert forbidden not in source

    for module_name in action_handlers.__all__:
        module_source = inspect.getsource(getattr(action_handlers, module_name))
        assert "orchestrator._handler_implementation" not in module_source


def test_daily_prompt_assembler_consumes_evidence_memory() -> None:
    from trading_assistant.analysis import prompt_assembler

    source = inspect.getsource(prompt_assembler.DailyPromptAssembler)

    assert "EvidenceMemory(" in source
    assert "self._evidence.findings.load_corrections(" in source


def test_prompt_assemblers_inject_evidence_memory() -> None:
    from trading_assistant.analysis import (
        discovery_prompt_assembler,
        outcome_reasoning_prompt,
        prompt_assembler,
        triage_prompt_assembler,
        weekly_prompt_assembler,
    )

    for cls in (
        prompt_assembler.DailyPromptAssembler,
        weekly_prompt_assembler.WeeklyPromptAssembler,
        triage_prompt_assembler.TriagePromptAssembler,
        discovery_prompt_assembler.DiscoveryPromptAssembler,
        outcome_reasoning_prompt.OutcomeReasoningAssembler,
    ):
        source = inspect.getsource(cls)
        assert "EvidenceMemory(" in source
        assert "evidence_memory=self._evidence" in source


def test_signal_decay_detectors_delegate_out_of_strategy_engine() -> None:
    from trading_assistant.analysis.strategy_engine import StrategyEngine

    assert "return evaluate_alpha_decay(" in inspect.getsource(StrategyEngine.detect_alpha_decay)
    assert "return evaluate_signal_decay(" in inspect.getsource(StrategyEngine.detect_signal_decay)
    assert "return evaluate_component_signal_decay(" in inspect.getsource(
        StrategyEngine.detect_component_signal_decay
    )
    assert "return evaluate_factor_correlation_decay(" in inspect.getsource(
        StrategyEngine.detect_factor_correlation_decay
    )


def test_detector_catalog_exposes_engine_metadata() -> None:
    assert DEFAULT_DETECTOR_CATALOG.category_for("tight_stop") == "stop_loss"
    assert DEFAULT_DETECTOR_CATALOG.archetype_default(
        "alpha_decay",
        "trend_follow",
        "decay_threshold",
    ) == 0.55


def test_loop_contract_context_source_handles_empty_agent(tmp_path: Path) -> None:
    assert LoopContractContextSource(tmp_path).load("") == {}
