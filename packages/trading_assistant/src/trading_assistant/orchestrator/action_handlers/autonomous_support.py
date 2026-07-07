"""Autonomous pipeline and registry drift support."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class AutonomousSupportActions:
    """Autonomous pipeline and registry drift support."""

    async def _run_autonomous_pipeline(self, suggestion_ids: dict[str, str], run_id: str) -> None:
        """Run the autonomous pipeline on newly recorded suggestions (if enabled)."""
        if not self._autonomous_pipeline or not suggestion_ids:
            return
        try:
            await self._autonomous_pipeline.process_new_suggestions(
                suggestion_ids=list(suggestion_ids.keys()),
                run_id=run_id,
            )
        except Exception:
            logger.exception("Autonomous pipeline failed - analysis unaffected")

    def _check_strategy_registry_drift(self, run_id: str) -> None:
        """Compare data/strategy_profiles.yaml against live reference dirs.

        Records the diff to memory/findings/strategy_registry_drift.jsonl so
        ContextBuilder can surface it in subsequent prompts. Drift never blocks
        the run - it's purely informational.
        """
        try:
            from trading_assistant.orchestrator.strategy_registry_loader import load_strategy_registry
            from trading_assistant.skills.strategy_registry_drift import check_drift, record_drift

            registry = load_strategy_registry()
            drift = check_drift(registry)
            record_drift(drift, self._memory_dir / "findings")
            if drift.has_drift:
                logger.warning(
                    "Strategy registry drift (run %s): %s", run_id, drift.summary(),
                )
                self._event_stream.broadcast("strategy_registry_drift", {
                    "run_id": run_id,
                    "registered_but_missing": drift.registered_but_missing,
                    "present_but_unregistered": drift.present_but_unregistered,
                    "empty_shells": drift.empty_shells,
                })
        except Exception:
            logger.exception("Strategy registry drift check failed (run %s)", run_id)
