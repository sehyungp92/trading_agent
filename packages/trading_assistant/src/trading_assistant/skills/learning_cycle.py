# skills/learning_cycle.py
"""LearningCycle — autonomous weekly learning loop.

Autoresearch's infinite loop adapted for trading:
measure → synthesize → keep/discard → recalibrate → propose next.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from trading_assistant.schemas.learning_ledger import LearningLedgerEntry
from trading_assistant.skills._outcome_utils import is_conclusive_outcome, is_positive_outcome

logger = logging.getLogger(__name__)

_BLACKLIST_EXPIRY_WEEKS = 12  # allow re-testing after ~1 market regime cycle


class LearningCycle:
    """Autonomous weekly learning cycle.

    Weekly sequence: ground truth → synthesis → hypothesis lifecycle →
    category recalibration → experiment selection → ledger record.
    """

    def __init__(
        self,
        curated_dir: Path,
        memory_dir: Path,
        runs_dir: Path,
        bots: list[str],
        suggestion_tracker=None,
        hypothesis_library=None,
        experiment_tracker=None,
        prediction_tracker=None,
        calibration_tracker=None,
        learning_review_mode: str = "deterministic",
        learning_review_disabled_workflows: list[str] | None = None,
        learning_review_structured_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._curated_dir = curated_dir
        self._memory_dir = memory_dir
        self._runs_dir = runs_dir
        self._bots = bots
        self._suggestion_tracker = suggestion_tracker
        self._hypothesis_library = hypothesis_library
        self._experiment_tracker = experiment_tracker
        self._prediction_tracker = prediction_tracker
        self._calibration_tracker = calibration_tracker
        self._learning_review_mode = learning_review_mode
        self._learning_review_disabled_workflows = list(learning_review_disabled_workflows or [])
        self._learning_review_structured_reviewer = learning_review_structured_reviewer

    async def run(
        self, week_start: str, week_end: str,
    ) -> LearningLedgerEntry:
        """Execute the full weekly learning cycle.

        1. Compute ground truth snapshots (start and end of week)
        2. Build retrospective synthesis (keep/discard verdicts)
        3. Update hypothesis lifecycle from verdicts
        4. Recalibrate suggestion categories
        5. Select next experiments (max 2 per bot)
        6. Record to learning ledger
        """
        from trading_assistant.skills.ground_truth_computer import GroundTruthComputer
        from trading_assistant.skills.learning_ledger import LearningLedger
        from trading_assistant.skills.retrospective_builder import RetrospectiveBuilder
        from trading_assistant.skills.suggestion_scorer import SuggestionScorer

        findings_dir = self._memory_dir / "findings"
        computer = GroundTruthComputer(self._curated_dir)
        ledger = LearningLedger(findings_dir)

        # 1. Ground truth snapshots
        gt_start = computer.compute_all_bots(self._bots, week_start)
        gt_end = computer.compute_all_bots(self._bots, week_end)

        composite_delta: dict[str, float] = {}
        for bot_id in self._bots:
            start_score = gt_start.get(bot_id)
            end_score = gt_end.get(bot_id)
            if start_score and end_score:
                composite_delta[bot_id] = round(
                    end_score.composite_score - start_score.composite_score, 4,
                )

        # 2. Retrospective synthesis
        retro = RetrospectiveBuilder(
            runs_dir=self._runs_dir,
            curated_dir=self._curated_dir,
            memory_dir=self._memory_dir,
        )
        synthesis = retro.build_synthesis(week_start, week_end)

        # 3. Update hypothesis lifecycle from verdicts
        if self._hypothesis_library:
            for item, positive in [
                *((i, True) for i in synthesis.what_worked),
                *((i, False) for i in synthesis.what_failed),
            ]:
                sid = getattr(item, "suggestion_id", "") or ""
                if not sid:
                    continue
                try:
                    self._hypothesis_library.record_outcome(sid, positive=positive)
                except Exception:
                    logger.debug("No hypothesis for suggestion %s", sid)

        # 4. Recalibrate suggestion categories
        scorer = SuggestionScorer(findings_dir)
        if synthesis.discard:
            scorer.apply_recalibration(synthesis.discard)
            logger.info(
                "Recalibrated %d categories", len(synthesis.discard),
            )

        benchmark_case_count = 0
        provider_scores_count = 0
        generated_playbooks_count = 0
        harness_results_count = 0
        learning_review_action_count = 0

        benchmark_suite = None

        try:
            from trading_assistant.skills.benchmark_compiler import BenchmarkCompiler

            benchmark_compiler = BenchmarkCompiler(findings_dir, runs_dir=self._runs_dir)
            benchmark_suite = benchmark_compiler.compile()
            benchmark_case_count = benchmark_compiler.save_suite(benchmark_suite)
        except Exception:
            logger.exception("Failed to compile benchmark suite")

        try:
            from trading_assistant.skills.provider_route_scorer import ProviderRouteScorer

            provider_scores = ProviderRouteScorer(findings_dir).recompute()
            provider_scores_count = len(provider_scores)
        except Exception:
            logger.exception("Failed to recompute provider route scores")

        try:
            from trading_assistant.skills.harness_eval_runner import HarnessEvalRunner
            from trading_assistant.skills.harness_execution_runner import HarnessExecutionRunner, load_execution_inputs

            if benchmark_suite is not None:
                case_dirs = _materialize_strict_harness_cases(
                    findings_dir=findings_dir,
                    benchmark_suite=benchmark_suite,
                )
                execution_inputs = load_execution_inputs(case_dirs)
                executable_suite = _executable_benchmark_suite(
                    benchmark_suite=benchmark_suite,
                    execution_inputs=execution_inputs,
                )
                execution_runner = HarnessExecutionRunner(
                    memory_dir=self._memory_dir,
                    output_root=findings_dir / "harness_cases" / "executions",
                )
                if executable_suite.cases:
                    harness_results = HarnessEvalRunner(findings_dir).evaluate_and_save(
                        executable_suite,
                        execution_runner=execution_runner,
                        execution_inputs=execution_inputs,
                    )
                    harness_results_count = len(harness_results)
        except Exception:
            logger.exception("Failed to run harness evaluation")

        try:
            from trading_assistant.skills.playbook_generator import PlaybookGenerator

            generated_playbooks = PlaybookGenerator(self._memory_dir).generate()
            generated_playbooks_count = len(generated_playbooks)
        except Exception:
            logger.exception("Failed to generate advisory playbooks")

        try:
            from trading_assistant.skills.learning_review_orchestrator import LearningReviewOrchestrator

            review = LearningReviewOrchestrator(
                findings_dir,
                review_mode=self._learning_review_mode,
                runs_dir=self._runs_dir,
                structured_reviewer=self._learning_review_structured_reviewer,
                disabled_workflows=self._learning_review_disabled_workflows,
            ).run(
                week_start=week_start,
                week_end=week_end,
            )
            learning_review_action_count = len(review.get("actions", []))
        except Exception:
            logger.exception("Failed to run post-cycle learning review")

        # 5. Select next experiments (max 2 per bot) and record them
        selected_experiments = self._select_next_experiments()
        self._record_selected_experiments(selected_experiments)

        # 6. Count activity for ledger
        suggestions_proposed, suggestions_accepted, suggestions_implemented = (
            self._count_suggestions(week_start, week_end)
        )
        experiments_concluded = self._count_experiments(week_start, week_end)
        discoveries_found = self._count_jsonl_in_range(
            findings_dir / "discoveries.jsonl",
            week_start, week_end,
            ts_keys=("discovered_at", "timestamp"),
        )

        # Net improvement: majority of bots improved (not just sum > 0)
        if composite_delta:
            improved = sum(1 for d in composite_delta.values() if d > 0)
            net_improvement = improved > len(composite_delta) / 2
        else:
            net_improvement = False

        # 7. Loop source attribution — classify suggestions by origin
        inner_proposed, outer_proposed, inner_pos, outer_pos, inner_total, outer_total = (
            self._classify_loop_sources(week_start, week_end, findings_dir)
        )

        # 8. Cycle effectiveness score
        cycle_effectiveness = ledger.compute_cycle_effectiveness(
            composite_delta=composite_delta,
            suggestions_proposed=suggestions_proposed,
            suggestions_implemented=suggestions_implemented,
            lessons=synthesis.lessons,
            week_start=week_start,
            week_end=week_end,
        )

        entry = ledger.record_week(
            week_start=week_start,
            week_end=week_end,
            bots=self._bots,
            gt_start=gt_start,
            gt_end=gt_end,
            composite_delta=composite_delta,
            net_improvement=net_improvement,
            suggestions_proposed=suggestions_proposed,
            suggestions_accepted=suggestions_accepted,
            suggestions_implemented=suggestions_implemented,
            experiments_concluded=experiments_concluded,
            discoveries_found=discoveries_found,
            what_worked=[item.title for item in synthesis.what_worked],
            what_failed=[item.title for item in synthesis.what_failed],
            lessons_for_next_week=synthesis.lessons,
            cycle_effectiveness=cycle_effectiveness,
            inner_suggestions_proposed=inner_proposed,
            outer_suggestions_proposed=outer_proposed,
            inner_positive_outcomes=inner_pos,
            outer_positive_outcomes=outer_pos,
            inner_total_outcomes=inner_total,
            outer_total_outcomes=outer_total,
        )

        # Log calibration reliability summary (informational only)
        if self._calibration_tracker:
            self._log_calibration_summary(week_start)

        logger.info(
            "Learning cycle complete: net_improvement=%s, composite_deltas=%s, benchmarks=%d, provider_scores=%d, harness_results=%d, playbooks=%d, learning_review_actions=%d",
            net_improvement, composite_delta, benchmark_case_count, provider_scores_count,
            harness_results_count, generated_playbooks_count, learning_review_action_count,
        )

        return entry

    def _log_calibration_summary(self, week_start: str) -> None:
        """Log per-(bot, category) backtest reliability summary."""
        import json
        summaries: list[dict] = []
        cal_path = self._memory_dir / "findings" / "backtest_calibration.jsonl"
        if not cal_path.exists():
            return
        # Collect unique (bot, category) pairs
        seen: set[tuple[str, str]] = set()
        for line in cal_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec.get("bot_id", ""), rec.get("param_category", ""))
                seen.add(key)
            except Exception:
                pass
        for bot_id, category in sorted(seen):
            reliability, n = self._calibration_tracker.get_reliability(bot_id, category)
            if n > 0:
                summaries.append({
                    "bot_id": bot_id,
                    "category": category,
                    "reliability": round(reliability, 2),
                    "sample_count": n,
                })
        if summaries:
            summary_path = self._memory_dir / "findings" / "calibration_summary.jsonl"
            try:
                with summary_path.open("w", encoding="utf-8") as f:
                    for s in summaries:
                        f.write(json.dumps(s) + "\n")
                logger.info("Calibration summary: %d entries written", len(summaries))
            except Exception:
                logger.exception("Failed to write calibration summary")

    # ── Loop source classification ──

    @staticmethod
    def classify_suggestion_source(suggestion: dict) -> str:
        """Classify a suggestion as 'inner' or 'outer' loop origin.

        Inner loop: has detection_context with a detector_name (strategy engine).
        Outer loop: everything else (LLM structural proposals).
        """
        ctx = suggestion.get("detection_context") or {}
        if isinstance(ctx, dict) and ctx.get("detector_name"):
            return "inner"
        return "outer"

    def _classify_loop_sources(
        self, week_start: str, week_end: str, findings_dir: Path,
    ) -> tuple[int, int, int, int, int, int]:
        """Count inner/outer suggestions and outcomes for the week.

        Returns: (inner_proposed, outer_proposed,
                  inner_positive, outer_positive,
                  inner_total, outer_total)
        """
        inner_proposed = outer_proposed = 0
        inner_pos = outer_pos = inner_total = outer_total = 0

        # Build suggestion_id → source mapping from ALL suggestions (not just
        # current week) so that outcomes measured this week for older suggestions
        # get correct source attribution.  Only count proposals for current week.
        sid_to_source: dict[str, str] = {}
        if self._suggestion_tracker:
            try:
                for s in self._suggestion_tracker.load_all():
                    sid = s.get("suggestion_id", "")
                    source = self.classify_suggestion_source(s)
                    sid_to_source[sid] = source  # ALL suggestions → mapping
                    ts = s.get("proposed_at", "") or s.get("timestamp", "") or s.get("created_at", "")
                    if ts and week_start <= ts[:10] <= week_end:
                        if source == "inner":
                            inner_proposed += 1
                        else:
                            outer_proposed += 1
            except Exception:
                logger.warning("Failed to classify suggestion sources")

        # Count outcomes by source
        outcomes_path = findings_dir / "outcomes.jsonl"
        if outcomes_path.exists():
            try:
                for line in outcomes_path.read_text(encoding="utf-8").strip().splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = entry.get("measurement_date") or entry.get("measured_at") or entry.get("timestamp", "")
                    if not ts or not (week_start <= ts[:10] <= week_end):
                        continue
                    if not is_conclusive_outcome(entry):
                        continue
                    sid = entry.get("suggestion_id", "")
                    source = sid_to_source.get(sid, "outer")
                    if source == "inner":
                        inner_total += 1
                        if is_positive_outcome(entry):
                            inner_pos += 1
                    else:
                        outer_total += 1
                        if is_positive_outcome(entry):
                            outer_pos += 1
            except Exception:
                logger.warning("Failed to classify outcome sources")

        return inner_proposed, outer_proposed, inner_pos, outer_pos, inner_total, outer_total

    # ── Counting helpers ──

    def _count_suggestions(
        self, week_start: str, week_end: str,
    ) -> tuple[int, int, int]:
        """Count suggestions proposed/accepted/implemented in the week."""
        proposed = accepted = implemented = 0
        if not self._suggestion_tracker:
            return proposed, accepted, implemented
        try:
            for s in self._suggestion_tracker.load_all():
                ts = s.get("proposed_at", "") or s.get("timestamp", "") or s.get("created_at", "")
                if not ts or not (week_start <= ts[:10] <= week_end):
                    continue
                proposed += 1
                status = s.get("status", "")
                if status in ("accepted", "merged", "deployed", "implemented", "measured"):
                    accepted += 1
                if status in ("deployed", "implemented", "measured"):
                    implemented += 1
        except Exception:
            logger.warning("Failed to count suggestions for ledger")
        return proposed, accepted, implemented

    def _count_experiments(self, week_start: str, week_end: str) -> int:
        """Count experiments concluded in the week."""
        if not self._experiment_tracker:
            return 0
        try:
            loader = getattr(
                self._experiment_tracker, "load_all",
                getattr(self._experiment_tracker, "_load_all", None),
            )
            if not loader:
                return 0
            count = 0
            for exp in loader():
                concluded = str(getattr(exp, "resolved_at", "") or getattr(exp, "concluded_at", "") or "")
                if concluded and week_start <= concluded[:10] <= week_end:
                    count += 1
            return count
        except Exception:
            logger.warning("Failed to count experiments for ledger")
            return 0

    @staticmethod
    def _count_jsonl_in_range(
        path: Path, week_start: str, week_end: str,
        *, ts_keys: tuple[str, ...] = ("timestamp",),
    ) -> int:
        """Count JSONL entries whose timestamp falls in [week_start, week_end]."""
        if not path.exists():
            return 0
        count = 0
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = ""
                for k in ts_keys:
                    ts = d.get(k, "") or ""
                    if ts:
                        break
                if ts and len(ts) >= 10 and week_start <= ts[:10] <= week_end:
                    count += 1
        except Exception:
            logger.warning("Failed to count entries in %s", path.name)
        return count

    # ── Experiment selection ──

    def _select_next_experiments(self) -> list[dict]:
        """Select next experiments: max 2 per bot, positive-effectiveness only.

        Skips hypotheses in discarded categories and hypothesis+bot combos
        that already failed an experiment.
        """
        if not self._hypothesis_library or not self._experiment_tracker:
            return []

        # Load discarded categories
        discarded = self._load_discarded_categories()

        # Load failed experiment combos (hypothesis_id, bot_id),
        # allowing re-testing after expiry (market regime cycle)
        expiry_cutoff = datetime.now(timezone.utc) - timedelta(weeks=_BLACKLIST_EXPIRY_WEEKS)
        failed_combos: set[tuple[str, str]] = set()
        try:
            for exp in self._experiment_tracker.get_failed_experiments():
                resolved_at = getattr(exp, "resolved_at", None)
                if resolved_at is not None:
                    if isinstance(resolved_at, str):
                        try:
                            resolved_at = datetime.fromisoformat(
                                resolved_at.replace("Z", "+00:00"),
                            )
                        except (ValueError, TypeError):
                            resolved_at = None
                    if resolved_at is not None:
                        if resolved_at.tzinfo is None:
                            resolved_at = resolved_at.replace(tzinfo=timezone.utc)
                        if resolved_at < expiry_cutoff:
                            continue  # expired — allow re-testing
                hyp_id = getattr(exp, "hypothesis_id", "") or ""
                bot_id = getattr(exp, "bot_id", "") or ""
                if hyp_id and bot_id:
                    failed_combos.add((hyp_id, bot_id))
        except Exception:
            logger.warning("Failed to load failed experiments for blacklist")

        try:
            hypotheses = self._hypothesis_library.get_active()
        except Exception:
            return []

        hypotheses.sort(
            key=lambda h: getattr(h, "effectiveness", 0.0), reverse=True,
        )

        # Count already-active experiments per bot
        bot_exp_count: dict[str, int] = {}
        for exp in self._experiment_tracker.get_active_experiments():
            bid = getattr(exp, "bot_id", "") or ""
            if bid:
                bot_exp_count[bid] = bot_exp_count.get(bid, 0) + 1

        selected: list[dict] = []
        for hyp in hypotheses:
            effectiveness = getattr(hyp, "effectiveness", 0.0)
            if effectiveness <= 0:
                continue

            category = getattr(hyp, "category", "")
            hyp_id = getattr(hyp, "hypothesis_id", "") or getattr(hyp, "id", "")

            for bot_id in self._bots:
                if bot_exp_count.get(bot_id, 0) >= 2:
                    continue
                if (bot_id, category) in discarded:
                    continue
                if (hyp_id, bot_id) in failed_combos:
                    continue

                selected.append({
                    "hypothesis_id": hyp_id,
                    "bot_id": bot_id,
                    "category": category,
                    "effectiveness": effectiveness,
                })
                bot_exp_count[bot_id] = bot_exp_count.get(bot_id, 0) + 1

        if selected:
            logger.info("Selected %d next experiments", len(selected))
        return selected

    def _record_selected_experiments(self, selected: list[dict]) -> None:
        """Record selected experiments via experiment_tracker."""
        if not selected or not self._experiment_tracker:
            return
        from trading_assistant.schemas.structural_experiment import ExperimentRecord
        import hashlib

        for exp_dict in selected:
            experiment_id = hashlib.sha256(
                f"{exp_dict['hypothesis_id']}:{exp_dict['bot_id']}".encode(),
            ).hexdigest()[:12]
            record = ExperimentRecord(
                experiment_id=experiment_id,
                bot_id=exp_dict["bot_id"],
                title=f"Hypothesis test: {exp_dict.get('category', 'unknown')}",
                hypothesis_id=exp_dict.get("hypothesis_id"),
            )
            try:
                self._experiment_tracker.record_experiment(record)
                logger.info(
                    "Recorded experiment %s for hypothesis %s on bot %s",
                    experiment_id, exp_dict["hypothesis_id"], exp_dict["bot_id"],
                )
            except Exception:
                logger.warning(
                    "Failed to record experiment for hypothesis %s",
                    exp_dict.get("hypothesis_id"),
                )

    def _load_discarded_categories(self) -> set[tuple[str, str]]:
        """Load (bot_id, category) pairs with confidence_multiplier <= 0.3."""
        discarded: set[tuple[str, str]] = set()
        overrides_path = self._memory_dir / "findings" / "category_overrides.jsonl"
        if not overrides_path.exists():
            return discarded
        try:
            for line in overrides_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("confidence_multiplier", 1.0) <= 0.3:
                    discarded.add((
                        entry.get("bot_id", ""),
                        entry.get("category", ""),
                    ))
        except Exception:
            logger.warning("Failed to load category overrides")
        return discarded


def _materialize_strict_harness_cases(*, findings_dir: Path, benchmark_suite) -> list[Path]:
    """Materialize only benchmark cases with strict, root-contained provenance."""
    from trading_assistant.skills.harness_case_materializer import (
        HarnessCaseMaterializationError,
        HarnessCaseMaterializer,
    )

    materializer = HarnessCaseMaterializer(findings_dir)
    case_dirs: list[Path] = []
    for case in getattr(benchmark_suite, "cases", []):
        try:
            case_dirs.append(materializer.materialize_case(case, strict=True))
        except HarnessCaseMaterializationError as exc:
            logger.warning(
                "Skipping non-executable harness case %s: %s",
                getattr(case, "case_id", "unknown"),
                exc,
            )
    return case_dirs


def _executable_benchmark_suite(*, benchmark_suite, execution_inputs):
    """Return the benchmark subset that has strict materialized replay inputs."""
    executable_case_ids = {
        execution_input.case_id
        for execution_input in execution_inputs
        if getattr(execution_input, "case_id", "")
    }
    return benchmark_suite.model_copy(update={
        "cases": [
            case for case in getattr(benchmark_suite, "cases", [])
            if getattr(case, "case_id", "") in executable_case_ids
        ],
    })
