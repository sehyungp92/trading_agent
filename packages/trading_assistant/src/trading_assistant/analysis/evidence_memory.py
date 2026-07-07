"""Workflow-facing prompt evidence memory sources."""

from __future__ import annotations

from pathlib import Path

from trading_assistant.analysis.context_sources import LoopContractContextSource
from trading_assistant.analysis.context_sources.finding_memory import FindingMemorySource
from trading_assistant.analysis.context_sources.performance_learning_context import (
    PerformanceLearningContextSource,
)
from trading_assistant.analysis.context_sources.policy_memory import PolicyMemorySource
from trading_assistant.analysis.context_sources.run_recall import RunRecallSource


class EvidenceMemory:
    """Compatibility facade over source-specific prompt context loaders."""

    def __init__(
        self,
        memory_dir: Path,
        *,
        run_index: object | None = None,
        strategy_registry: object | None = None,
    ) -> None:
        self.policies = PolicyMemorySource(memory_dir)
        self.findings = FindingMemorySource(
            memory_dir,
            strategy_registry=strategy_registry,
        )
        self.run_recall = RunRecallSource(memory_dir, run_index=run_index)
        self.loop_contracts = LoopContractContextSource(memory_dir)
        self.performance_learning = PerformanceLearningContextSource(memory_dir)
