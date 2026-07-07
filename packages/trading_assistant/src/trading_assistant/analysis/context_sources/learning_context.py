"""Learning and findings-derived prompt context loaders."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.analysis.context_sources.common import (
    apply_temporal_window as _apply_temporal_window,
    filter_inactive_strategies as _filter_inactive_strategies,
    safe_jsonl as _safe_jsonl,
)

logger = logging.getLogger(__name__)



class LearningContextMixin:
    def load_generated_playbooks(self, workflow: str, tags: list[str], limit: int = 3) -> list[dict]:
        """Load top matching generated playbooks for prompt context."""
        path = self._memory_dir / "playbooks" / "generated" / "playbooks.jsonl"
        if not path.exists():
            return []
        try:
            from trading_assistant.schemas.generated_playbook import GeneratedPlaybook
            from trading_assistant.skills.generated_playbook_guard import GeneratedPlaybookGuard

            guard = GeneratedPlaybookGuard(self._memory_dir)
            matches: list[tuple[float, GeneratedPlaybook]] = []
            for record in _safe_jsonl(path):
                try:
                    playbook = GeneratedPlaybook.model_validate(record)
                except Exception:
                    continue
                if not guard.is_safe(playbook):
                    continue
                score = playbook.match_score(workflow, tags)
                if score > 0:
                    matches.append((score, playbook))
            matches.sort(key=lambda item: item[0], reverse=True)
            return [
                {
                    "playbook_id": playbook.playbook_id,
                    "title": playbook.title,
                    "text": playbook.to_prompt_text(),
                }
                for _, playbook in matches[:limit]
            ]
        except Exception:
            logger.debug("Generated playbook loading failed; skipping")
            return []

    def load_pattern_library(self, bot_id: str = "") -> list[dict]:
        """Load cross-bot pattern library entries.

        If bot_id is provided, only returns patterns relevant to that bot.
        """
        try:
            from trading_assistant.skills.pattern_library import PatternLibrary

            lib = PatternLibrary(self._memory_dir / "findings")
            if bot_id:
                entries = lib.load_for_bot(bot_id)
            else:
                entries = lib.load_active()
            return [e.model_dump(mode="json") for e in entries]
        except Exception:
            return []

    def load_contradictions(
        self, date: str, bots: list[str], curated_dir: Path,
    ) -> list[dict]:
        """Load temporal contradictions across recent daily reports.

        Returns list of ContradictionItem dicts for prompt injection.
        """
        try:
            from trading_assistant.skills.contradiction_detector import ContradictionDetector

            detector = ContradictionDetector(
                date=date, bots=bots, curated_dir=curated_dir,
            )
            report = detector.detect()
            return [item.model_dump(mode="json") for item in report.items]
        except Exception:
            return []

    def load_signal_factor_history(
        self, bot_id: str, date: str, findings_dir: Path,
    ) -> dict:
        """Load rolling signal factor analysis for a bot.

        Returns SignalFactorRollingReport as dict, or empty dict if insufficient data.
        """
        try:
            from trading_assistant.skills.signal_factor_tracker import SignalFactorTracker

            tracker = SignalFactorTracker(findings_dir)
            report = tracker.compute_rolling(bot_id, date)
            if not report.factors:
                return {}
            return report.model_dump(mode="json")
        except Exception:
            return {}

    def load_correction_patterns(self) -> list[dict]:
        """Load extracted correction patterns from findings/correction_patterns.jsonl."""
        path = self._memory_dir / "findings" / "correction_patterns.jsonl"
        if not path.exists():
            return []
        patterns = _safe_jsonl(path)
        return _apply_temporal_window(patterns)

    def load_forecast_meta(self) -> dict:
        """Load forecast meta-analysis from findings/forecast_history.jsonl.

        When prediction verdicts are available, includes empirical calibration
        buckets, ECE, and Brier score. Also includes directional bias analysis.
        """
        try:
            from trading_assistant.skills.forecast_tracker import ForecastTracker

            tracker = ForecastTracker(self._memory_dir / "findings")
            records = tracker.load_all()
            if not records:
                return {}

            # Load prediction verdicts for empirical calibration
            verdicts = self._load_prediction_verdicts()

            # Compute directional bias from prediction tracker
            dir_bias: dict[str, dict] = {}
            try:
                from trading_assistant.skills.prediction_tracker import PredictionTracker as _PT
                if self._curated_dir:
                    pt = _PT(self._memory_dir / "findings")
                    dir_bias = pt.compute_directional_bias(self._curated_dir)
            except Exception:
                pass

            meta = tracker.compute_meta_analysis(
                prediction_verdicts=verdicts if verdicts else None,
                directional_bias=dir_bias if dir_bias else None,
            )
            return meta.model_dump(mode="json")
        except Exception:
            return {}

    def _load_prediction_verdicts(self) -> list:
        """Load prediction verdicts from the prediction tracker."""
        try:
            from trading_assistant.skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            path = self._memory_dir / "findings" / "predictions.jsonl"
            if not path.exists():
                return []
            predictions = tracker.load_predictions()
            if not predictions or not self._curated_dir:
                return []
            evaluation = tracker.evaluate_predictions(predictions, self._curated_dir)
            return evaluation.verdicts if evaluation else []
        except Exception:
            return []

    def load_active_suggestions(self) -> list[dict]:
        """Load non-rejected suggestions from findings/suggestions.jsonl.

        Returns suggestions with unresolved status, applying temporal window
        (90d, 30-entry cap) and dropping entries pinned to retired strategies.
        """
        active_statuses = {"proposed", "accepted", "merged", "deployed"}
        active = [
            rec for rec in _safe_jsonl(self._memory_dir / "findings" / "suggestions.jsonl")
            if rec.get("status", "") in active_statuses
        ]
        active = _filter_inactive_strategies(active, self._get_strategy_registry())
        return _apply_temporal_window(active, max_entries=30)

    def load_recent_proposal_outcomes(
        self, bot_id: str = "", days: int = 30, max_entries: int = 30,
    ) -> list[dict]:
        """Load recent ProposalLedger outcomes (lightweight summary view).

        Returns one dict per proposal with measured outcome inside ``days`` window.
        Filters by ``bot_id`` when provided. Empty list if the ledger file is
        missing or malformed.
        """
        try:
            from trading_assistant.skills.proposal_ledger import ProposalLedger

            ledger = ProposalLedger(self._memory_dir / "findings")
            recs = ledger.list_all()
        except Exception:
            return []

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        out: list[dict] = []
        for rec in recs:
            if bot_id and rec.candidate.bot_id != bot_id:
                continue
            if not rec.outcomes:
                continue
            latest = max(
                rec.outcomes,
                key=lambda o: (
                    o.measured_at.replace(tzinfo=timezone.utc)
                    if o.measured_at.tzinfo is None
                    else o.measured_at
                ),
            )
            measured_at = latest.measured_at
            if measured_at.tzinfo is None:
                measured_at = measured_at.replace(tzinfo=timezone.utc)
            if measured_at < cutoff:
                continue
            out.append({
                "proposal_id": rec.candidate.proposal_id,
                "bot_id": rec.candidate.bot_id,
                "source": rec.candidate.source.value,
                "kind": rec.candidate.kind.value,
                "title": rec.candidate.title,
                "verdict": latest.verdict,
                "objective_delta": latest.objective_delta,
                "measured_at": measured_at.isoformat(),
            })
        out.sort(key=lambda r: r["measured_at"], reverse=True)
        return out[:max_entries]

    def load_strategy_change_ledger(
        self, bot_id: str = "", days: int = 180, max_entries: int = 30,
    ) -> list[dict]:
        """Load recent strategy-level monthly/change decisions."""
        try:
            from trading_assistant.skills.strategy_change_ledger import StrategyChangeLedger

            ledger = StrategyChangeLedger(self._memory_dir / "findings")
            records = ledger.get_recent(days=days)
        except Exception:
            return []
        out: list[dict] = []
        for record in records:
            if bot_id and record.bot_id != bot_id:
                continue
            out.append({
                "record_id": record.record_id,
                "bot_id": record.bot_id,
                "strategy_id": record.strategy_id,
                "record_type": record.record_type.value,
                "run_month": record.run_month,
                "monthly_status": record.monthly_status,
                "decision_reason": record.decision_reason,
                "evidence_paths": record.evidence_paths[:10],
                "objective_deltas": record.objective_deltas,
                "updated_at": record.updated_at.isoformat(),
            })
        return out[:max_entries]

    def load_category_scorecard(self) -> dict:
        """Load category-level suggestion success rates."""
        try:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer

            scorer = SuggestionScorer(self._memory_dir / "findings")
            scorecard = scorer.compute_scorecard()
            if scorecard.scores:
                return scorecard.model_dump(mode="json")
        except Exception:
            pass
        return {}

    def load_optimization_allocation(self) -> dict:
        """Load per-category value analysis for optimization direction guidance."""
        try:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer

            scorer = SuggestionScorer(self._memory_dir / "findings")
            value_map = scorer.compute_category_value_map()
            if value_map and len(value_map) > 1:  # more than just _recommendations
                return value_map
        except Exception:
            pass
        return {}

    def load_regime_stratified_scores(self) -> dict | None:
        """Load category win rates stratified by macro regime."""
        try:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer
            scorer = SuggestionScorer(self._memory_dir / "findings")
            scores = scorer.compute_regime_stratified_scores()
            return scores if scores else None
        except Exception:
            return None

    def load_search_signal_summary(self) -> dict:
        """Load historical search signal approve/discard summary."""
        path = self._memory_dir / "findings" / "search_signals.jsonl"
        if not path.exists():
            return {}
        from collections import defaultdict
        counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"approve": 0, "discard": 0}
        )
        try:
            for rec in _safe_jsonl(path):
                bot_id = rec.get("bot_id", "")
                category = rec.get("category", "")
                key = (bot_id, category)
                if rec.get("positive"):
                    counts[key]["approve"] += 1
                else:
                    counts[key]["discard"] += 1
        except OSError:
            return {}

        if not counts:
            return {}

        summary: dict[str, dict] = {}
        for (bot_id, category), c in counts.items():
            total = c["approve"] + c["discard"]
            summary[f"{bot_id}:{category}"] = {
                "approve_count": c["approve"],
                "discard_count": c["discard"],
                "approve_rate": round(c["approve"] / total, 3) if total > 0 else 0.0,
            }
        return summary

    def load_prediction_accuracy(self) -> dict:
        """Load per-metric prediction accuracy from the prediction tracker.

        When curated_dir is available, computes real accuracy by evaluating predictions
        against actual curated data. Otherwise returns prediction count metadata.
        """
        try:
            from trading_assistant.skills.prediction_tracker import PredictionTracker

            tracker = PredictionTracker(self._memory_dir / "findings")
            if not (self._memory_dir / "findings" / "predictions.jsonl").exists():
                return {}
            predictions = tracker.load_predictions()
            if not predictions:
                return {}

            # When curated_dir available, compute real per-metric accuracy
            if self._curated_dir and self._curated_dir.exists():
                accuracy_by_metric = tracker.get_accuracy_by_metric(self._curated_dir)
                if accuracy_by_metric:
                    return {
                        "has_predictions": True,
                        "count": len(predictions),
                        "accuracy_by_metric": accuracy_by_metric,
                    }

            return {"has_predictions": True, "count": len(predictions)}
        except Exception:
            return {}

    def load_hypothesis_track_record(self) -> dict:
        """Load hypothesis effectiveness scores for prompt injection."""
        try:
            from trading_assistant.skills.hypothesis_library import HypothesisLibrary

            lib = HypothesisLibrary(self._memory_dir / "findings")
            track = lib.get_track_record()
            if track:
                return track
        except Exception:
            pass
        return {}

    def load_transfer_track_record(self) -> dict:
        """Load transfer outcome success rates for prompt injection."""
        try:
            from trading_assistant.skills.transfer_proposal_builder import TransferProposalBuilder

            return TransferProposalBuilder.load_track_record_from_file(
                self._memory_dir / "findings",
            )
        except Exception:
            return {}

    def load_experiment_track_record(self) -> dict:
        """Load structural experiment pass/fail track record."""
        try:
            from trading_assistant.skills.structural_experiment_tracker import StructuralExperimentTracker

            tracker = StructuralExperimentTracker(self._memory_dir / "findings")
            record = tracker.compute_track_record()
            if record.get("total", 0) > 0:
                return record
        except Exception:
            pass
        return {}

    def load_recalibrations(self) -> list[dict]:
        """Load causal recalibrations from findings/recalibrations.jsonl.

        Returns recalibrations with bot_id, category, revised_confidence,
        and lessons_learned, filtered by temporal window (90d, 30-entry cap).
        """
        path = self._memory_dir / "findings" / "recalibrations.jsonl"
        if not path.exists():
            return []
        entries = _safe_jsonl(path)
        return _apply_temporal_window(entries, max_entries=30)

    def load_outcome_reasonings(self) -> list[dict]:
        """Load causal outcome reasonings from findings/outcome_reasonings.jsonl.

        Returns recent reasonings with lessons learned, mechanisms, and
        transferability assessments for injection into prompts. Drops entries
        pinned to retired strategies.
        """
        path = self._memory_dir / "findings" / "outcome_reasonings.jsonl"
        if not path.exists():
            return []
        reasonings = _safe_jsonl(path)
        reasonings = _filter_inactive_strategies(reasonings, self._get_strategy_registry())
        return _apply_temporal_window(reasonings, max_entries=20)

    def load_discoveries(self) -> list[dict]:
        """Load discoveries from findings/discoveries.jsonl. Drops entries
        pinned to retired strategies.
        """
        path = self._memory_dir / "findings" / "discoveries.jsonl"
        if not path.exists():
            return []
        discoveries = _safe_jsonl(path)
        discoveries = _filter_inactive_strategies(discoveries, self._get_strategy_registry())
        return _apply_temporal_window(discoveries, max_entries=20)

    def load_active_experiments(self) -> list[dict]:
        """Load active structural experiments for prompt injection."""
        try:
            from trading_assistant.skills.structural_experiment_tracker import StructuralExperimentTracker

            tracker = StructuralExperimentTracker(self._memory_dir / "findings")
            active = tracker.get_active_experiments()
            return [e.model_dump(mode="json") for e in active]
        except Exception:
            return []

    def load_reliability_summary(self) -> dict:
        """Load reliability tracking summary from findings."""
        try:
            from trading_assistant.skills.reliability_tracker import ReliabilityTracker

            tracker = ReliabilityTracker(self._memory_dir / "findings")
            summary = tracker.compute_summary()
            if summary.scorecards_by_class:
                return summary.model_dump(mode="json")
        except Exception:
            pass
        return {}

    def load_validation_patterns(self, bot_id: str = "") -> dict:
        """Load aggregated validation patterns from findings/validation_log.jsonl.

        Groups blocked suggestions by category over the last 30 days.
        Returns summary dict: {"category": {"blocked_count": N, "common_reasons": [...]}}
        """
        path = self._memory_dir / "findings" / "validation_log.jsonl"
        if not path.exists():
            return {}
        try:
            from collections import defaultdict

            rows = _safe_jsonl(path)

            def _collect(days: int) -> dict[str, list[str]]:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                category_blocks: dict[str, list[str]] = defaultdict(list)
                for entry in rows:
                    ts = entry.get("timestamp", "")
                    if ts:
                        try:
                            entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if entry_time.tzinfo is None:
                                entry_time = entry_time.replace(tzinfo=timezone.utc)
                            if entry_time < cutoff:
                                continue
                        except (ValueError, TypeError):
                            pass
                    for detail in entry.get("blocked_details", []):
                        detail_bot_id = detail.get("bot_id", "")
                        if bot_id and detail_bot_id and detail_bot_id != bot_id:
                            continue
                        reason = detail.get("reason", "")
                        category = detail.get("category", "")
                        if category:
                            category_blocks[category].append(reason)
                            continue
                        # Infer category from reason or title
                        title = detail.get("title", "").lower()
                        for keyword, cat in [
                            ("exit", "exit_timing"), ("filter", "filter_threshold"),
                            ("stop", "stop_loss"), ("signal", "signal"),
                            ("regime", "regime_gate"), ("sizing", "position_sizing"),
                        ]:
                            if keyword in title:
                                category_blocks[cat].append(reason)
                                break
                        else:
                            category_blocks["other"].append(reason)
                return category_blocks

            category_blocks = _collect(30)
            stale_window_days = 0
            if not category_blocks:
                category_blocks = _collect(90)
                stale_window_days = 90
            if not category_blocks:
                return {}

            result: dict = {}
            for cat, reasons in category_blocks.items():
                # Deduplicate and count
                unique_reasons = list(set(reasons))[:5]
                result[cat] = {
                    "blocked_count": len(reasons),
                    "common_reasons": unique_reasons,
                }
                if stale_window_days:
                    result[cat]["stale_window_days"] = stale_window_days
            return result
        except Exception:
            return {}

    def load_threshold_profile(self) -> dict:
        """Load learned threshold profiles from findings/learned_thresholds.jsonl.

        Returns a dict with per-bot threshold data when available.
        """
        path = self._memory_dir / "findings" / "learned_thresholds.jsonl"
        if not path.exists():
            return {}
        try:
            profiles = _safe_jsonl(path)
            if profiles:
                return {"profiles": profiles, "count": len(profiles)}
        except Exception:
            pass
        return {}

    def load_ground_truth_trend(self) -> dict:
        """Load ground truth composite score trend from learning_ledger.jsonl.

        Returns last 12 weeks of composite scores per bot, recent lessons,
        and curated analysis notes (deduplicated, relevance-decayed, outcome-boosted).
        """
        try:
            from trading_assistant.skills.learning_ledger import LearningLedger

            ledger = LearningLedger(self._memory_dir / "findings")
            trend = ledger.get_trend(weeks=12)
            lessons = ledger.get_lessons(weeks=4)
            curated_notes = ledger.get_curated_notes(max_notes=30)
            if not trend and not lessons and not curated_notes:
                return {}
            result: dict = {}
            if trend:
                result["composite_trend"] = trend
            if lessons:
                result["recent_lessons"] = lessons
            if curated_notes:
                result["curated_analysis_notes"] = curated_notes
            try:
                latest = ledger.get_latest()
                if latest:
                    result["net_improvement"] = latest.net_improvement
                    result["composite_delta"] = latest.composite_delta
            except Exception:
                # Graceful: latest may fail if ledger entries have incomplete GT data
                # Still return trend + lessons
                pass
            return result
        except Exception:
            return {}

    def load_cycle_effectiveness(self) -> list[dict]:
        """Load last 8 cycle effectiveness entries from learning_ledger.jsonl."""
        path = self._memory_dir / "findings" / "learning_ledger.jsonl"
        if not path.exists():
            return []
        try:
            entries = _safe_jsonl(path)
        except Exception:
            return []
        entries.sort(key=lambda e: e.get("week_start", ""))
        recent = entries[-8:]
        result = []
        for e in recent:
            ce = e.get("cycle_effectiveness", 0.0)
            if ce > 0 or e.get("suggestions_proposed", 0) > 0:
                result.append({
                    "week": e.get("week_start", ""),
                    "effectiveness": ce,
                    "net_improvement": e.get("net_improvement", False),
                    "suggestions_proposed": e.get("suggestions_proposed", 0),
                    "suggestions_implemented": e.get("suggestions_implemented", 0),
                })
        return result

    def load_suggestion_quality_trend(self, value_map: dict | None = None) -> dict:
        """Load suggestion quality trend from SuggestionScorer."""
        try:
            from trading_assistant.skills.suggestion_scorer import SuggestionScorer
            scorer = SuggestionScorer(self._memory_dir / "findings")
            return scorer.compute_suggestion_quality_trend(value_map=value_map)
        except Exception:
            return {}

    def load_convergence_report(self) -> dict:
        """Load convergence report synthesising learning loop health."""
        try:
            from trading_assistant.skills.convergence_tracker import ConvergenceTracker

            tracker = ConvergenceTracker(self._memory_dir / "findings")
            report = tracker.compute_report(weeks=12)
            # Only include if we have real data (not all insufficient_data)
            if report.overall_status.value == "insufficient_data":
                return {}
            return report.model_dump(mode="json")
        except Exception:
            return {}

    def load_instrumentation_readiness(self, bots: list[str]) -> dict:
        """Load per-bot instrumentation readiness scorecards."""
        if not self._curated_dir or not bots:
            return {}
        try:
            from trading_assistant.skills.instrumentation_scorer import InstrumentationScorer

            scorer = InstrumentationScorer(self._curated_dir, lookback_days=30)
            reports = scorer.score_all_bots(bots)
            return {
                bot_id: report.model_dump(mode="json")
                for bot_id, report in reports.items()
                if report.days_with_data > 0
            }
        except Exception:
            return {}

    def load_retrospective_synthesis(self) -> dict:
        """Load most recent retrospective synthesis from findings."""
        path = self._memory_dir / "findings" / "retrospective_synthesis.jsonl"
        if not path.exists():
            return {}
        try:
            entries = _safe_jsonl(path)
            if entries:
                return entries[-1]  # most recent
        except Exception:
            pass
        return {}

    def load_spurious_outcomes(self) -> list[dict]:
        """Load outcomes determined to be spurious (not genuinely caused by the suggestion)."""
        path = self._memory_dir / "findings" / "spurious_outcomes.jsonl"
        if not path.exists():
            return []
        try:
            entries = _safe_jsonl(path)
            return _apply_temporal_window(entries)
        except Exception:
            return []

    def load_strategy_ideas(self) -> list[dict]:
        """Load strategy ideas from findings/strategy_ideas.jsonl.

        Returns active (non-retired) strategy ideas with temporal window.
        """
        path = self._memory_dir / "findings" / "strategy_ideas.jsonl"
        if not path.exists():
            return []
        try:
            ideas = [
                entry for entry in _safe_jsonl(path)
                if entry.get("status", "proposed") != "retired"
            ]
            return _apply_temporal_window(ideas, max_entries=10)
        except Exception:
            return []

    def build_self_assessment(
        self,
        forecast_meta: dict | None = None,
        category_scorecard: dict | None = None,
        correction_patterns: list[dict] | None = None,
        recalibrations: list[dict] | None = None,
    ) -> str:
        """Synthesize a plain-text self-assessment from multiple learning signals.

        Combines directional biases, calibration state, category strengths/weaknesses,
        recurring corrections, and causal lessons into a narrative summary.
        Returns empty string if fewer than 2 signals are available.

        When called from base_package(), pre-loaded data is passed to avoid
        duplicate I/O. When called standalone, loads data on demand.
        """
        if forecast_meta is None:
            forecast_meta = self.load_forecast_meta()
        if category_scorecard is None:
            category_scorecard = self.load_category_scorecard()
        if correction_patterns is None:
            correction_patterns = self.load_correction_patterns()
        if recalibrations is None:
            recalibrations = self.load_recalibrations()

        signals: list[str] = []

        # 1. Directional biases from forecast meta
        dir_bias = forecast_meta.get("directional_bias", {})
        if dir_bias:
            bias_lines = []
            for metric, info in dir_bias.items():
                bias = info.get("bias", "balanced")
                if bias != "balanced":
                    mag = info.get("bias_magnitude", 0)
                    bias_lines.append(
                        f"  - {metric}: {bias} (magnitude {mag:.2f})"
                    )
            if bias_lines:
                signals.append(
                    "Directional biases:\n" + "\n".join(bias_lines)
                )

        # 2. Calibration state
        ece = forecast_meta.get("expected_calibration_error")
        if ece is not None:
            cal_adj = forecast_meta.get("calibration_adjustment", 0)
            if cal_adj < -0.1:
                direction = "overconfident"
            elif cal_adj > 0.1:
                direction = "underconfident"
            else:
                direction = "reasonably calibrated"
            signals.append(f"Calibration: ECE={ece:.3f}, {direction}")

        # 3. Category strengths/weaknesses from scorecard
        # Per-strategy rows (strategy_id non-null) carry tighter signal than the
        # bot-wide aggregate — surface them in their own labels so Claude can
        # cite specific strategy track records, not just bot averages.
        scores = category_scorecard.get("scores", [])
        if scores:
            strong = []
            weak = []
            strong_per_strat = []
            weak_per_strat = []
            for s in scores:
                wr = s.get("win_rate", 0)
                n = s.get("sample_size", 0)
                if n < 3:
                    continue
                strat_id = s.get("strategy_id")
                if strat_id:
                    label = f"{s.get('bot_id', '?')}/{strat_id}/{s.get('category', '?')} ({wr:.0%}, n={n})"
                    if wr >= 0.6:
                        strong_per_strat.append(label)
                    elif wr < 0.4:
                        weak_per_strat.append(label)
                else:
                    label = f"{s.get('bot_id', '?')}/{s.get('category', '?')} ({wr:.0%}, n={n})"
                    if wr >= 0.6:
                        strong.append(label)
                    elif wr < 0.4:
                        weak.append(label)
            if strong:
                signals.append("Strong categories (bot-wide): " + ", ".join(strong[:5]))
            if weak:
                signals.append("Weak categories (avoid or justify): " + ", ".join(weak[:5]))
            if strong_per_strat:
                signals.append("Strong per-strategy: " + ", ".join(strong_per_strat[:5]))
            if weak_per_strat:
                signals.append("Weak per-strategy (avoid or justify): " + ", ".join(weak_per_strat[:5]))

        # 4. Recurring corrections (top 3 by count)
        if correction_patterns:
            sorted_patterns = sorted(correction_patterns, key=lambda p: p.get("count", 0), reverse=True)
            top = sorted_patterns[:3]
            lines = [
                f"  - {p.get('description', '?')} (count={p.get('count', 0)})"
                for p in top
            ]
            signals.append("Recurring corrections:\n" + "\n".join(lines))

        # 5. Causal lessons from recalibrations
        if recalibrations:
            all_lessons: list[str] = []
            seen: set[str] = set()
            for r in recalibrations:
                raw_lessons = r.get("lessons_learned", [])
                if isinstance(raw_lessons, str):
                    items = [raw_lessons] if raw_lessons.strip() else []
                else:
                    items = [
                        lesson_text
                        for lesson_text in (raw_lessons or [])
                        if isinstance(lesson_text, str) and lesson_text.strip()
                    ]
                for lesson in items:
                    if lesson not in seen:
                        seen.add(lesson)
                        all_lessons.append(lesson)
            if all_lessons:
                signals.append(
                    "Causal lessons learned:\n"
                    + "\n".join(f"  - {lesson}" for lesson in all_lessons[:5])
                )

        if len(signals) < 2:
            return ""

        return "SELF-ASSESSMENT (auto-synthesized from learning data):\n\n" + "\n\n".join(signals)
