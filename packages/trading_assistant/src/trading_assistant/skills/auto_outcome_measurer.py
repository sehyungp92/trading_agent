"""Automated outcome measurement for implemented suggestions.

Regime-controlled measurement: compares before/after performance with
regime matching, volatility controls, concurrent change detection,
and statistical significance estimation.

Monthly full-fidelity validation is authoritative for material strategy/config
changes. This module supplies early-warning/context signals and should not
finalize material changes once they are linked to monthly validation.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from trading_assistant.schemas.outcome_measurement import (
    OutcomeMeasurement,
    compute_measurement_quality,
    compute_significance,
)

if TYPE_CHECKING:
    from trading_assistant.skills.hypothesis_library import HypothesisLibrary
    from trading_assistant.skills.proposal_ledger import ProposalLedger


class AutoOutcomeMeasurer:
    """Measures suggestion outcomes by comparing pre/post performance."""

    # Maps suggestion category → the metric that category primarily targets.
    CATEGORY_TO_TARGET_METRIC: dict[str, str] = {
        "stop_loss": "drawdown",
        "exit_timing": "pnl",
        "signal": "win_rate",
        "filter_threshold": "win_rate",
        "position_sizing": "pnl",
        "regime_gate": "drawdown",
        "structural": "pnl",
        "funding_threshold": "pnl",
        "leverage_cap": "drawdown",
        "confluence_count": "win_rate",
        "setup_grade_filter": "win_rate",
    }

    def __init__(
        self,
        curated_dir: Path,
        findings_dir: Path | None = None,
        calibration_tracker: object | None = None,
        hypothesis_library: "HypothesisLibrary | None" = None,
        proposal_ledger: "ProposalLedger | None" = None,
    ) -> None:
        self._curated_dir = curated_dir
        self._findings_dir = findings_dir
        # Accepted for compatibility with older wiring, but early-warning
        # measurement no longer writes backtest-calibration feedback.
        self._calibration_tracker = None
        self._hypothesis_library = hypothesis_library
        self._proposal_ledger = proposal_ledger

    def measure(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
        before_days: int = 7,
        after_days: int = 7,
        record_feedback: bool = True,
    ) -> OutcomeMeasurement | None:
        """Compare performance before and after a suggestion was implemented.

        Now includes regime matching, volatility controls, concurrent change
        detection, and measurement quality assessment.
        """
        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        today = datetime.now(timezone.utc)

        before_start = impl_date - timedelta(days=before_days)
        after_end = impl_date + timedelta(days=after_days)

        before_summaries = self._load_summaries(bot_id, before_start, impl_date)
        after_summaries = self._load_summaries(bot_id, impl_date, after_end)

        if not before_summaries or not after_summaries:
            return None

        # Core metrics
        before_pnl = sum(self._summary_pnl(s) for s in before_summaries)
        after_pnl = sum(self._summary_pnl(s) for s in after_summaries)

        before_wins = sum(s.get("win_count", 0) for s in before_summaries)
        before_total = sum(s.get("total_trades", 0) for s in before_summaries)
        after_wins = sum(s.get("win_count", 0) for s in after_summaries)
        after_total = sum(s.get("total_trades", 0) for s in after_summaries)

        before_wr = before_wins / before_total if before_total > 0 else 0
        after_wr = after_wins / after_total if after_total > 0 else 0

        before_dd = max(
            (s.get("max_drawdown_pct", 0) for s in before_summaries), default=0
        )
        after_dd = max(
            (s.get("max_drawdown_pct", 0) for s in after_summaries), default=0
        )

        # Regime analysis
        before_regimes = self._load_regimes(bot_id, before_start, impl_date)
        after_regimes = self._load_regimes(bot_id, impl_date, after_end)
        before_regime = self._dominant_regime(before_regimes)
        after_regime = self._dominant_regime(after_regimes)
        regime_matched = before_regime == after_regime or not before_regime or not after_regime

        # Volatility
        before_vol = self._compute_volatility(before_summaries)
        after_vol = self._compute_volatility(after_summaries)
        vol_ratio = after_vol / before_vol if before_vol > 0 else 1.0

        # Concurrent changes
        concurrent = self._find_concurrent_changes(
            suggestion_id, bot_id, implemented_date, after_days
        )

        # Macro regime stability check
        macro_regime_at_impl = ""
        macro_regime_stable = True
        try:
            impl_regime_path = (
                self._curated_dir / implemented_date / "portfolio" / "macro_regime_analysis.json"
            )
            if impl_regime_path.exists():
                regime_data = json.loads(impl_regime_path.read_text(encoding="utf-8"))
                macro_regime_at_impl = regime_data.get("macro_regime", "")

            if macro_regime_at_impl:
                # Check if macro regime changed during measurement window
                end_date_str = after_end.strftime("%Y-%m-%d")
                end_regime_path = (
                    self._curated_dir / end_date_str / "portfolio" / "macro_regime_analysis.json"
                )
                if end_regime_path.exists():
                    end_data = json.loads(end_regime_path.read_text(encoding="utf-8"))
                    end_regime = end_data.get("macro_regime", "")
                    if end_regime and end_regime != macro_regime_at_impl:
                        macro_regime_stable = False
        except Exception:
            pass

        # Quality assessment
        quality = compute_measurement_quality(
            regime_matched=regime_matched,
            before_trade_count=before_total,
            after_trade_count=after_total,
            volatility_ratio=vol_ratio,
            concurrent_changes=concurrent,
            window_days=after_days,
            macro_regime_stable=macro_regime_stable,
        )

        # Effect significance
        daily_pnls_before = [self._summary_pnl(s) for s in before_summaries]
        daily_pnls_after = [self._summary_pnl(s) for s in after_summaries]
        effect = (after_pnl / max(len(after_summaries), 1)) - (
            before_pnl / max(len(before_summaries), 1)
        )
        noise = self._estimate_noise(daily_pnls_before + daily_pnls_after)
        sig = compute_significance(
            effect, noise, len(before_summaries), len(after_summaries),
        )

        measurement = OutcomeMeasurement(
            suggestion_id=suggestion_id,
            implemented_date=implemented_date,
            measurement_date=today.strftime("%Y-%m-%d"),
            window_days=after_days,
            pnl_before=before_pnl,
            pnl_after=after_pnl,
            win_rate_before=before_wr,
            win_rate_after=after_wr,
            drawdown_before=before_dd,
            drawdown_after=after_dd,
            before_regime=before_regime,
            after_regime=after_regime,
            regime_matched=regime_matched,
            before_volatility=round(before_vol, 6),
            after_volatility=round(after_vol, 6),
            volatility_ratio=round(vol_ratio, 4),
            before_trade_count=before_total,
            after_trade_count=after_total,
            concurrent_changes=concurrent,
            measurement_quality=quality,
            effect_size=round(effect, 4),
            noise_estimate=round(noise, 4),
            significance_score=round(sig, 4),
            macro_regime_at_implementation=macro_regime_at_impl,
            macro_regime_stable=macro_regime_stable,
        )

        # Evaluate targeted metric improvement
        measurement = self._evaluate_target_metric(measurement, suggestion_id)

        # Stamp hypothesis/proposal cross-links onto the measurement
        hypothesis_id, proposal_id = self._load_suggestion_links(suggestion_id)
        if hypothesis_id or proposal_id:
            measurement = measurement.model_copy(update={
                "hypothesis_id": hypothesis_id or None,
                "proposal_id": proposal_id or None,
            })

        if record_feedback:
            self.record_measurement_feedback(measurement)

        return measurement

    def record_measurement_feedback(self, measurement: OutcomeMeasurement) -> None:
        """Record learning feedback once for a selected persisted measurement."""
        suggestion_id = measurement.suggestion_id
        hypothesis_id = measurement.hypothesis_id or ""
        proposal_id = measurement.proposal_id or ""
        if not hypothesis_id or not proposal_id:
            loaded_hypothesis_id, loaded_proposal_id = self._load_suggestion_links(suggestion_id)
            hypothesis_id = hypothesis_id or loaded_hypothesis_id
            proposal_id = proposal_id or loaded_proposal_id

        delta = self._estimate_composite_delta(measurement)
        verdict = measurement.verdict
        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)

        # A/B and structural experiment outcomes are recorded separately by app.py.
        if self._hypothesis_library and hypothesis_id and verdict_str in ("positive", "negative"):
            try:
                positive = self._is_positive_outcome(measurement, delta)
                self._hypothesis_library.record_outcome(
                    hypothesis_id,
                    positive=positive,
                )
            except Exception:
                pass

        if self._proposal_ledger and proposal_id:
            try:
                from trading_assistant.schemas.proposal_ledger import ProposalOutcome
                measurement_path = ""
                if self._findings_dir:
                    measurement_path = str(self._findings_dir / "outcomes.jsonl")
                self._proposal_ledger.record_outcome(
                    proposal_id,
                    ProposalOutcome(
                        proposal_id=proposal_id,
                        objective_delta=float(delta),
                        verdict=verdict_str,
                        measurement_path=measurement_path,
                        outcome_source="early_warning",
                    ),
                )
            except Exception:
                pass

    @staticmethod
    def _is_positive_outcome(measurement: OutcomeMeasurement, delta: float) -> bool:
        """Decide hypothesis-library outcome polarity.

        Use the same target-metric logic as _evaluate_target_metric when
        available; otherwise fall back to composite delta sign.
        """
        if measurement.target_metric_improved is not None:
            return bool(measurement.target_metric_improved)
        return delta > 0

    def _lookup_suggestion(self, suggestion_id: str) -> dict | None:
        """Find a SuggestionRecord by id in suggestions.jsonl. Tolerates malformed lines."""
        if not self._findings_dir:
            return None
        path = self._findings_dir / "suggestions.jsonl"
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate one bad line — keep scanning
            if rec.get("suggestion_id") == suggestion_id:
                return rec
        return None

    def _load_suggestion_links(self, suggestion_id: str) -> tuple[str, str]:
        """Return (hypothesis_id, proposal_id) for a suggestion, blank strings if missing."""
        rec = self._lookup_suggestion(suggestion_id)
        if not rec:
            return "", ""
        return rec.get("hypothesis_id") or "", rec.get("proposal_id") or ""

    def _load_suggestion_category(self, suggestion_id: str) -> tuple[str, str | None]:
        """Load category and target_param for a suggestion from suggestions.jsonl."""
        rec = self._lookup_suggestion(suggestion_id)
        if not rec:
            return "", None
        return rec.get("category", ""), rec.get("target_param")

    def _evaluate_target_metric(
        self, measurement: OutcomeMeasurement, suggestion_id: str,
    ) -> OutcomeMeasurement:
        """Check whether the specific targeted metric improved.

        For drawdown, improvement means decrease. For pnl/win_rate, improvement
        means increase. Updates measurement with target_metric fields.
        """
        category, target_param = self._load_suggestion_category(suggestion_id)
        if not category:
            return measurement

        target_metric = self.CATEGORY_TO_TARGET_METRIC.get(category)
        if not target_metric:
            return measurement

        # Compute delta for the targeted metric
        if target_metric == "pnl":
            delta = measurement.pnl_after - measurement.pnl_before
            improved = delta > 0
        elif target_metric == "win_rate":
            delta = measurement.win_rate_after - measurement.win_rate_before
            improved = delta > 0
        elif target_metric == "drawdown":
            # For drawdown, decrease = improvement
            delta = measurement.drawdown_after - measurement.drawdown_before
            improved = delta < 0
        else:
            return measurement

        return measurement.model_copy(update={
            "target_metric": target_metric,
            "target_metric_improved": improved,
            "target_metric_delta": round(delta, 6),
        })

    _PROGRESSIVE_WINDOWS = [7, 14, 30]
    _QUALITY_RANK = {"high": 3, "medium": 2, "low": 1, "insufficient": 0}

    def measure_progressive(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
    ) -> OutcomeMeasurement | None:
        """Try progressively wider measurement windows, returning the best quality result.

        Windows: 7d, 14d, 30d. Only attempts a window if sufficient calendar
        time has elapsed since implemented_date. Returns the result with the
        highest measurement quality, or the widest attempted window on tie.
        """
        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        today = datetime.now(timezone.utc)
        elapsed_days = (today - impl_date.replace(tzinfo=timezone.utc)).days

        best: OutcomeMeasurement | None = None
        best_rank = -1

        for window in self._PROGRESSIVE_WINDOWS:
            if elapsed_days < window:
                break
            result = self.measure(
                suggestion_id=suggestion_id,
                bot_id=bot_id,
                implemented_date=implemented_date,
                before_days=window,
                after_days=window,
                record_feedback=False,
            )
            if result is None:
                continue
            rank = self._QUALITY_RANK.get(result.measurement_quality.value, 0)
            if (
                rank > best_rank
                or (
                    best is not None
                    and rank == best_rank
                    and result.window_days > best.window_days
                )
            ):
                best = result
                best_rank = rank

        return best

    @staticmethod
    def _estimate_composite_delta(m: OutcomeMeasurement) -> float:
        """Approximate composite delta aligned with inner loop formula.

        Mirrors: 0.4*calmar + 0.3*profit_factor - 0.3*drawdown.
        Uses ratio-based comparison (after/before) to avoid scale explosion.
        NOT GroundTruthComputer — clearly an approximate calibration signal.
        """
        window = max(m.window_days, 1)

        # Calmar proxy: annualized daily PnL / drawdown (ratio-based)
        before_daily = m.pnl_before / window
        after_daily = m.pnl_after / window
        _DD_FLOOR = 0.001  # floor to avoid division by near-zero
        before_dd = max(m.drawdown_before, _DD_FLOOR)
        after_dd = max(m.drawdown_after, _DD_FLOOR)
        calmar_before = (before_daily * 365) / before_dd
        calmar_after = (after_daily * 365) / after_dd
        # Ratio-based improvement (matches inner loop style), clamped
        calmar_ratio = calmar_after / calmar_before if calmar_before != 0 else 1.0
        calmar_imp = max(-3.0, min(3.0, calmar_ratio - 1.0))

        # Profit factor proxy: PnL efficiency relative to drawdown (ratio-based)
        pnl_eff_before = m.pnl_before / (before_dd * window)
        pnl_eff_after = m.pnl_after / (after_dd * window)
        pf_ratio = pnl_eff_after / pnl_eff_before if pnl_eff_before != 0 else 1.0
        pf_imp = max(-3.0, min(3.0, pf_ratio - 1.0))

        # Drawdown increase (higher drawdown is worse)
        dd_inc = m.drawdown_after - m.drawdown_before

        return 0.4 * calmar_imp + 0.3 * pf_imp - 0.3 * max(0.0, dd_inc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summary_pnl(summary: dict) -> float:
        """Prefer net PnL because it reflects fees/slippage in live trading."""
        value = summary.get("net_pnl")
        if value is None:
            value = summary.get("gross_pnl", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _load_summaries(
        self, bot_id: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load daily summaries for a date range."""
        summaries = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            summary_path = self._curated_dir / date_str / bot_id / "summary.json"
            if summary_path.exists():
                try:
                    summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
            current += timedelta(days=1)
        return summaries

    def _load_regimes(
        self, bot_id: str, start: datetime, end: datetime
    ) -> list[dict]:
        """Load regime_analysis.json for each day in the window."""
        regimes = []
        current = start
        while current < end:
            date_str = current.strftime("%Y-%m-%d")
            path = self._curated_dir / date_str / bot_id / "regime_analysis.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    regimes.append(data)
                except (json.JSONDecodeError, OSError):
                    pass
            current += timedelta(days=1)
        return regimes

    @staticmethod
    def _dominant_regime(regime_data: list[dict]) -> str:
        """Determine the dominant regime from a list of daily regime analyses."""
        if not regime_data:
            return ""
        counts: dict[str, int] = {}
        for day in regime_data:
            regime = day.get("dominant_regime", "") or day.get("regime", "")
            if regime:
                counts[regime] = counts.get(regime, 0) + 1
        if not counts:
            return ""
        return max(counts, key=counts.get)  # type: ignore[arg-type]

    @classmethod
    def _compute_volatility(cls, summaries: list[dict]) -> float:
        """Compute daily PnL standard deviation as a volatility proxy."""
        pnls = [cls._summary_pnl(s) for s in summaries]
        if len(pnls) < 2:
            return 0.0
        return statistics.stdev(pnls)

    def _find_concurrent_changes(
        self,
        suggestion_id: str,
        bot_id: str,
        implemented_date: str,
        window_days: int,
    ) -> list[str]:
        """Find other suggestions deployed within the measurement window."""
        if not self._findings_dir:
            return []
        suggestions_path = self._findings_dir / "suggestions.jsonl"
        if not suggestions_path.exists():
            return []

        impl_date = datetime.strptime(implemented_date, "%Y-%m-%d")
        window_start = impl_date - timedelta(days=window_days)
        window_end = impl_date + timedelta(days=window_days)

        concurrent: list[str] = []
        try:
            for line in suggestions_path.read_text(encoding="utf-8").strip().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("suggestion_id") == suggestion_id:
                    continue
                if rec.get("bot_id") != bot_id:
                    continue
                status = rec.get("status", "")
                if status not in ("deployed", "implemented", "measured"):
                    continue
                deployed_str = rec.get("deployed_at") or rec.get("resolved_at", "")
                if not deployed_str:
                    continue
                try:
                    deployed_dt = datetime.fromisoformat(
                        deployed_str.replace("Z", "+00:00")
                    )
                    if deployed_dt.tzinfo:
                        deployed_dt = deployed_dt.replace(tzinfo=None)
                    if window_start <= deployed_dt <= window_end:
                        concurrent.append(rec.get("suggestion_id", "unknown"))
                except (ValueError, TypeError):
                    pass
        except (OSError, json.JSONDecodeError):
            pass
        return concurrent

    @staticmethod
    def _estimate_noise(daily_pnls: list[float]) -> float:
        """Estimate noise as the standard deviation of daily PnLs."""
        if len(daily_pnls) < 2:
            return 0.0
        return statistics.stdev(daily_pnls)
