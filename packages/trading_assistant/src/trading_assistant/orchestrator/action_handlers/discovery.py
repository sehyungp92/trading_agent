"""Discovery action orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from trading_assistant.orchestrator.orchestrator_brain import Action
from trading_assistant.schemas.notifications import NotificationPriority

logger = logging.getLogger(__name__)


class DiscoveryActions:
    """Discovery action orchestration."""

    async def handle_discovery_analysis(self, action: Action) -> None:
        """Run the discovery analysis: raw data exploration for novel patterns."""
        details = action.details or {}
        date = details.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        bots = details.get("bots", self._bots)
        run_id = f"discovery-{date}"
        start_time = datetime.now(timezone.utc)
        self._record_run(run_id, "discovery_analysis", "running", started_at=start_time.isoformat())

        try:
            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "started", "handler": "discovery_analysis",
            })

            from trading_assistant.analysis.discovery_prompt_assembler import DiscoveryPromptAssembler

            assembler = DiscoveryPromptAssembler(
                date=date,
                bots=bots,
                curated_dir=self._curated_dir,
                memory_dir=self._memory_dir,
                bot_configs=self._bot_configs,
            )
            package = assembler.assemble(session_store=self._agent_runner.session_store)

            self._event_stream.broadcast("handler_progress", {
                "run_id": run_id, "stage": "prompt_assembly", "handler": "discovery_analysis",
            })

            # Discovery agent gets higher max_turns and file access tools
            result = await self._agent_runner.invoke(
                agent_type="discovery_analysis",
                prompt_package=package,
                run_id=run_id,
                allowed_tools=["Read", "Grep", "Glob"],
                max_turns=15,
            )

            if result.success:
                from trading_assistant.analysis.response_parser import parse_response

                parsed = parse_response(result.response)

                # Parse discoveries from structured output
                discoveries = []
                if parsed.raw_structured and "discoveries" in parsed.raw_structured:
                    discoveries = parsed.raw_structured["discoveries"]

                # Persist discoveries to findings
                if discoveries:
                    discoveries_path = self._memory_dir / "findings" / "discoveries.jsonl"
                    discoveries_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(discoveries_path, "a", encoding="utf-8") as f:
                        for d in discoveries:
                            d["run_id"] = run_id
                            d["date"] = date
                            d["discovered_at"] = datetime.now(timezone.utc).isoformat()
                            f.write(json.dumps(d) + "\n")
                            # Mirror into the unified proposal ledger so weekly
                            # learning sees discoveries as comparable proposals.
                            self._ledger_write_candidate(
                                source="discovery",
                                kind_hint="structural_change",
                                bot_id=d.get("bot_id", "") or "",
                                title=d.get("pattern_description", "")[:120] or "discovery",
                                description=d.get("testable_hypothesis", "") or "",
                                run_id=run_id,
                                evaluation_method="discovery_review",
                                lifecycle_stage=d.get("lifecycle_stage", "") or "",
                                stable_link_key=f"discovery:{run_id}:{d.get('pattern_description', '')[:120]}",
                            )

                    # Add novel hypotheses to hypothesis library
                    try:
                        from trading_assistant.skills.hypothesis_library import HypothesisLibrary

                        hypothesis_lib = HypothesisLibrary(self._memory_dir / "findings")
                        for d in discoveries:
                            if d.get("testable_hypothesis") and d.get("confidence", 0) >= 0.5:
                                hypothesis_lib.add_candidate(
                                    title=d.get("pattern_description", "")[:100],
                                    description=d.get("testable_hypothesis", ""),
                                    category=d.get("proposed_root_cause", "novel"),
                                )
                    except Exception:
                        logger.warning("Failed to add discovery hypotheses")

                    self._event_stream.broadcast("discoveries_recorded", {
                        "run_id": run_id, "count": len(discoveries),
                    })

                # Parse strategy ideas from structured output
                strategy_ideas = []
                if parsed.raw_structured and "strategy_ideas" in parsed.raw_structured:
                    strategy_ideas = parsed.raw_structured["strategy_ideas"]

                if strategy_ideas:
                    import hashlib as _hashlib
                    ideas_path = self._memory_dir / "findings" / "strategy_ideas.jsonl"
                    ideas_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(ideas_path, "a", encoding="utf-8") as f:
                        for idea in strategy_ideas:
                            # Deterministic ID from description
                            desc = idea.get("description", "")
                            idea["idea_id"] = _hashlib.sha256(desc.encode()).hexdigest()[:12]
                            idea["proposed_at"] = datetime.now(timezone.utc).isoformat()
                            idea["run_id"] = run_id
                            idea["status"] = "proposed"
                            f.write(json.dumps(idea) + "\n")
                            self._ledger_write_candidate(
                                source="discovery",
                                kind_hint="new_strategy",
                                bot_id=idea.get("bot_id", "") or "",
                                strategy_id=idea.get("strategy_id", "") or "",
                                title=idea.get("title", "") or desc[:80] or "strategy_idea",
                                description=desc,
                                run_id=run_id,
                                evaluation_method="experiment",
                                lifecycle_stage=idea.get("lifecycle_stage", "") or "",
                                stable_link_key=idea.get("idea_id", "") or "",
                            )

                    # High-confidence ideas → structural experiment records
                    if self._structural_experiment_tracker is not None:
                        from trading_assistant.schemas.structural_experiment import (
                            AcceptanceCriteria,
                            ExperimentRecord,
                        )

                        for idea in strategy_ideas:
                            if idea.get("confidence", 0) >= 0.7:
                                try:
                                    exp_id = "exp_" + _hashlib.sha256(
                                        f"{run_id}:{idea.get('bot_id', 'unknown')}:{idea.get('title', '')}".encode()
                                    ).hexdigest()[:12]

                                    # Extract criteria from the idea itself if available
                                    idea_criteria: list[AcceptanceCriteria] = []
                                    for raw_c in idea.get("acceptance_criteria", []):
                                        if isinstance(raw_c, dict) and raw_c.get("metric"):
                                            try:
                                                idea_criteria.append(AcceptanceCriteria(**raw_c))
                                            except Exception:
                                                pass
                                    if not idea_criteria:
                                        # Fallback: infer from applicable_regimes and edge_hypothesis
                                        default_metric = "pnl"
                                        default_window = 14
                                        default_min_trades = 20
                                        # Longer window for regime-specific strategies
                                        if idea.get("applicable_regimes") and len(idea.get("applicable_regimes", [])) <= 2:
                                            default_window = 30
                                            default_min_trades = 15
                                        idea_criteria = [
                                            AcceptanceCriteria(
                                                metric=default_metric,
                                                direction="improve",
                                                minimum_change=0.0,
                                                observation_window_days=default_window,
                                                minimum_trade_count=default_min_trades,
                                            ),
                                        ]

                                    experiment = ExperimentRecord(
                                        experiment_id=exp_id,
                                        bot_id=idea.get("bot_id", "unknown"),
                                        title=idea.get("title", "Strategy idea"),
                                        description=idea.get("description", ""),
                                        proposal_run_id=run_id,
                                        acceptance_criteria=idea_criteria,
                                    )
                                    self._structural_experiment_tracker.record_experiment(experiment)
                                    logger.info("Recorded structural experiment %s for strategy idea %s", exp_id, idea.get("idea_id"))
                                    await self._notify(
                                        "structural_experiment_proposed",
                                        NotificationPriority.NORMAL,
                                        f"Structural Experiment Proposed: {experiment.title}",
                                        (f"Bot: {experiment.bot_id}\n"
                                         f"ID: {exp_id}\n"
                                         f"Description: {experiment.description[:200]}"),
                                    )
                                except Exception:
                                    logger.warning("Failed to create experiment for strategy idea %s", idea.get("idea_id"))

                    self._event_stream.broadcast("strategy_ideas_proposed", {
                        "run_id": run_id, "count": len(strategy_ideas),
                    })

                # Process structural proposals from discovery
                if parsed.structural_proposals:
                    self._update_hypothesis_lifecycle(parsed, {})
                    self._record_structural_experiments(parsed.structural_proposals, run_id)

                self._write_run_report(run_id, "discovery_report.md", result.response)

            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            status = "completed" if result.success else "failed"
            self._record_run(
                run_id, "discovery_analysis", status,
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._record_run(
                run_id, "discovery_analysis", "failed",
                started_at=start_time.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=elapsed, error=str(exc),
            )
            logger.exception("Discovery analysis handler failed for %s", run_id)
