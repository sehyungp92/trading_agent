# analysis/daily_triage.py
"""Deterministic daily triage — pre-processes curated data to identify
significant events and generate focused questions for Claude.

No LLM calls. Produces a TriageReport that replaces the 29-item instruction
checklist with 3-5 focused analytical questions.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class SignificantEvent:
    """An event worth Claude's analytical attention."""
    event_type: str  # pnl_anomaly, regime_shift, outcome_conflict, pattern_break, drawdown_spike
    bot_id: str
    severity: str  # high, medium
    description: str
    relevant_data_keys: list[str] = field(default_factory=list)


@dataclass
class TriageReport:
    """Output of deterministic triage: routine summary + significant events + focus questions."""
    significant_events: list[SignificantEvent] = field(default_factory=list)
    routine_summary: str = ""
    focus_questions: list[str] = field(default_factory=list)
    relevant_data_keys: list[str] = field(default_factory=list)


class DailyTriage:
    """Deterministic pre-processor that identifies what deserves Claude's attention."""

    _ALWAYS_VISIBLE_KEYS = (
        "order_lifecycle.json",
        "process_quality.json",
        "parameter_changes.json",
    )

    def __init__(
        self,
        curated_dir: Path,
        date: str,
        bots: list[str],
        active_suggestions: list[dict] | None = None,
        trailing_days: int = 5,
    ) -> None:
        self._curated_dir = curated_dir
        self._date = date
        self._bots = bots
        self._active_suggestions = active_suggestions or []
        self._trailing_days = trailing_days

    def run(self) -> TriageReport:
        """Run all triage checks and produce a focused report."""
        events: list[SignificantEvent] = []
        relevant_keys: set[str] = set()
        bot_summaries: list[str] = []

        for bot_id in self._bots:
            summary = self._load_json(bot_id, "summary.json")
            if not summary:
                bot_summaries.append(f"{bot_id}: no data")
                continue

            # Basic stats for routine summary
            pnl = self._get_pnl(summary)
            trades = summary.get("total_trades", 0)
            wr = summary.get("win_count", 0) / trades if trades > 0 else 0
            bot_summaries.append(
                f"{bot_id}: {trades} trades, PnL={pnl:+.2f}, WR={wr:.0%}"
            )

            # Load trailing data for context
            trailing = self._load_trailing_summaries(bot_id)

            # Check 1: PnL anomaly (> 2σ from trailing mean)
            pnl_event = self._check_pnl_anomaly(bot_id, pnl, trailing)
            if pnl_event:
                events.append(pnl_event)
                relevant_keys.update(pnl_event.relevant_data_keys)

            # Check 2: Drawdown spike
            dd_event = self._check_drawdown_spike(bot_id, summary, trailing)
            if dd_event:
                events.append(dd_event)
                relevant_keys.update(dd_event.relevant_data_keys)

            # Check 3: Regime shift vs trailing
            regime_event = self._check_regime_shift(bot_id)
            if regime_event:
                events.append(regime_event)
                relevant_keys.update(regime_event.relevant_data_keys)

            # Check 4: Active suggestion contradicted by today's data
            suggestion_events = self._check_suggestion_conflicts(bot_id, summary)
            for e in suggestion_events:
                events.append(e)
                relevant_keys.update(e.relevant_data_keys)

            # Check 5: Filter blocking winners
            filter_event = self._check_filter_anomaly(bot_id)
            if filter_event:
                events.append(filter_event)
                relevant_keys.update(filter_event.relevant_data_keys)

        # Check 6: Macro regime shift (portfolio-level, outside per-bot loop)
        macro_event = self._check_macro_regime_shift()
        if macro_event:
            events.append(macro_event)
            relevant_keys.update(macro_event.relevant_data_keys)

        routine = (
            f"Daily summary for {self._date}:\n"
            + "\n".join(bot_summaries)
        )

        # Generate focus questions from significant events
        questions = self._generate_focus_questions(events)

        # Always include summary.json for all bots
        relevant_keys.add("summary.json")
        self._include_always_visible_keys(relevant_keys)

        return TriageReport(
            significant_events=events,
            routine_summary=routine,
            focus_questions=questions,
            relevant_data_keys=sorted(relevant_keys),
        )

    def _include_always_visible_keys(self, relevant_keys: set[str]) -> None:
        """Keep compact execution/process evidence visible even on triage-filtered runs."""
        for bot_id in self._bots:
            bot_dir = self._curated_dir / self._date / bot_id
            for filename in self._ALWAYS_VISIBLE_KEYS:
                if (bot_dir / filename).exists():
                    relevant_keys.add(filename)

    def _check_pnl_anomaly(
        self, bot_id: str, today_pnl: float, trailing: list[dict]
    ) -> SignificantEvent | None:
        """Detect PnL > 2σ from trailing mean."""
        pnls = [self._get_pnl(s) for s in trailing]
        if len(pnls) < 3:
            return None
        mean = statistics.mean(pnls)
        stdev = statistics.stdev(pnls)
        if stdev == 0:
            return None
        z_score = (today_pnl - mean) / stdev
        if abs(z_score) < 2.0:
            return None
        direction = "gain" if z_score > 0 else "loss"
        return SignificantEvent(
            event_type="pnl_anomaly",
            bot_id=bot_id,
            severity="high" if abs(z_score) >= 3.0 else "medium",
            description=f"Unusual {direction}: PnL={today_pnl:+.2f} ({z_score:+.1f}σ vs trailing {self._trailing_days}d mean={mean:.2f})",
            relevant_data_keys=["summary.json", "winners.json", "losers.json", "root_cause_summary.json"],
        )

    def _check_drawdown_spike(
        self, bot_id: str, summary: dict, trailing: list[dict]
    ) -> SignificantEvent | None:
        """Detect drawdown exceeding trailing max."""
        today_dd = abs(summary.get("max_drawdown_pct", 0))
        if today_dd == 0:
            return None
        trailing_dds = [abs(s.get("max_drawdown_pct", 0)) for s in trailing]
        if not trailing_dds:
            return None
        max_trailing = max(trailing_dds)
        if max_trailing == 0 or today_dd <= max_trailing:
            return None
        return SignificantEvent(
            event_type="drawdown_spike",
            bot_id=bot_id,
            severity="high" if today_dd > max_trailing * 1.5 else "medium",
            description=f"Drawdown {today_dd:.1f}% exceeds trailing max {max_trailing:.1f}%",
            relevant_data_keys=["summary.json", "losers.json", "process_failures.json"],
        )

    def _check_regime_shift(self, bot_id: str) -> SignificantEvent | None:
        """Detect regime change vs trailing days."""
        today_regime = self._load_json(bot_id, "regime_analysis.json")
        if not today_regime:
            return None
        today_dominant = today_regime.get("dominant_regime", "") or today_regime.get("regime", "")
        if not today_dominant:
            return None

        # Check trailing days for different regime
        date_obj = datetime.strptime(self._date, "%Y-%m-%d")
        prev_regimes = []
        for d in range(1, self._trailing_days + 1):
            prev_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            prev = self._load_json_for_date(bot_id, prev_date, "regime_analysis.json")
            if prev:
                r = prev.get("dominant_regime", "") or prev.get("regime", "")
                if r:
                    prev_regimes.append(r)

        if not prev_regimes:
            return None

        # Majority of trailing days had different regime
        counts = Counter(prev_regimes)
        most_common = counts.most_common(1)[0][0]
        if most_common != today_dominant:
            return SignificantEvent(
                event_type="regime_shift",
                bot_id=bot_id,
                severity="high",
                description=f"Regime shift: {most_common} → {today_dominant}",
                relevant_data_keys=["regime_analysis.json", "summary.json", "filter_analysis.json"],
            )
        return None

    def _check_suggestion_conflicts(
        self, bot_id: str, summary: dict
    ) -> list[SignificantEvent]:
        """Check if active suggestions are contradicted by today's data.

        Only flags when today's loss is significant (> 2× average daily loss
        from trailing data), not on every negative day.
        """
        events = []
        today_pnl = self._get_pnl(summary)
        if today_pnl >= 0:
            return events

        # Only flag if today's loss is significant vs trailing
        trailing = self._load_trailing_summaries(bot_id)
        trailing_losses = [self._get_pnl(s) for s in trailing if self._get_pnl(s) < 0]
        avg_loss = statistics.mean(trailing_losses) if trailing_losses else 0
        # Only flag when today's loss exceeds 2× average loss magnitude
        if avg_loss != 0 and today_pnl > avg_loss * 2:
            return events  # Normal loss day, not worth flagging

        for suggestion in self._active_suggestions:
            if suggestion.get("bot_id") != bot_id:
                continue
            status = suggestion.get("status", "")
            if status not in ("deployed", "implemented"):
                continue
            sid = suggestion.get("suggestion_id", "?")[:8]
            title = suggestion.get("title", "unknown")
            events.append(SignificantEvent(
                event_type="outcome_conflict",
                bot_id=bot_id,
                severity="medium",
                description=f"Deployed suggestion #{sid} ('{title}') — significant loss today ({today_pnl:+.2f})",
                relevant_data_keys=["summary.json", "root_cause_summary.json"],
            ))
        return events

    def _check_filter_anomaly(self, bot_id: str) -> SignificantEvent | None:
        """Detect filters blocking an unusual number of winners."""
        filter_data = self._load_json(bot_id, "filter_analysis.json")
        if not filter_data:
            return None
        filters = filter_data if isinstance(filter_data, list) else filter_data.get("filters", [])
        # Collect all problematic filters, report worst one
        worst: SignificantEvent | None = None
        worst_score = 0
        for f in filters:
            blocked = f.get("blocked_count", 0) or f.get("block_count", 0)
            would_have_won = f.get("would_have_won", 0) or f.get("blocked_winners", 0)
            if blocked >= 3 and would_have_won >= 3:
                score = would_have_won * blocked
                if score > worst_score:
                    worst_score = score
                    name = f.get("name", f.get("filter_name", "unknown"))
                    worst = SignificantEvent(
                        event_type="pattern_break",
                        bot_id=bot_id,
                        severity="high",
                        description=f"Filter '{name}' blocked {blocked} trades, {would_have_won} would have been winners",
                        relevant_data_keys=["filter_analysis.json", "notable_missed.json"],
                    )
        return worst

    def _check_macro_regime_shift(self) -> SignificantEvent | None:
        """Detect macro regime (G/R/S/D) change vs trailing days."""
        today = self._load_json_portfolio("macro_regime_analysis.json")
        if not today or not isinstance(today, dict):
            return None
        today_regime = today.get("macro_regime", "")
        if not today_regime:
            return None

        date_obj = datetime.strptime(self._date, "%Y-%m-%d")
        prev_regimes = []
        for d in range(1, self._trailing_days + 1):
            prev_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            prev = self._load_json_portfolio_for_date(prev_date, "macro_regime_analysis.json")
            if prev and isinstance(prev, dict):
                r = prev.get("macro_regime", "")
                if r:
                    prev_regimes.append(r)

        if not prev_regimes:
            return None

        most_common = Counter(prev_regimes).most_common(1)[0][0]
        if most_common != today_regime:
            # Severity based on regime destination (high-confidence signal),
            # not stress_level (41% FPR, observational only per reliability guide)
            severity = "high" if today_regime in ("S", "D") else "medium"
            return SignificantEvent(
                event_type="macro_regime_shift",
                bot_id="portfolio",
                severity=severity,
                description=f"Macro regime shift: {most_common} → {today_regime}",
                relevant_data_keys=["macro_regime_analysis.json", "applied_regime_config.json"],
            )
        return None

    def _load_json_portfolio(self, filename: str) -> dict | list | None:
        """Load a JSON file from the portfolio curated directory for today."""
        return self._load_json_portfolio_for_date(self._date, filename)

    def _load_json_portfolio_for_date(self, date_str: str, filename: str) -> dict | list | None:
        """Load a JSON file from the portfolio curated directory for a specific date."""
        path = self._curated_dir / date_str / "portfolio" / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _generate_focus_questions(self, events: list[SignificantEvent]) -> list[str]:
        """Generate 3-5 focused analytical questions from significant events."""
        questions: list[str] = []

        # Group events by type for more specific questions
        pnl_anomalies = [e for e in events if e.event_type == "pnl_anomaly"]
        regime_shifts = [e for e in events if e.event_type == "regime_shift"]
        macro_shifts = [e for e in events if e.event_type == "macro_regime_shift"]
        outcome_conflicts = [e for e in events if e.event_type == "outcome_conflict"]
        filter_anomalies = [e for e in events if e.event_type == "pattern_break"]
        drawdown_spikes = [e for e in events if e.event_type == "drawdown_spike"]

        for e in pnl_anomalies[:2]:
            questions.append(
                f"[{e.bot_id}] What caused the unusual {e.description.split(':')[0].lower()}? "
                f"Examine root causes, regime context, and whether this represents a systematic change or a single outlier."
            )

        for e in regime_shifts[:1]:
            questions.append(
                f"[{e.bot_id}] {e.description}. How should the bot's strategy adapt? "
                f"Review filter behavior and win rates by regime to determine if parameter adjustments are warranted."
            )

        for e in macro_shifts[:1]:
            new_regime = e.description.split("\u2192")[-1].split("(")[0].strip()
            questions.append(
                f"Macro regime shifted to {new_regime}. "
                f"Review applied_regime_config for each bot: are sizing multipliers and strategy disables correct for the new regime? "
                f"Check if any trades entered during the transition were adversely affected."
            )

        for e in outcome_conflicts[:1]:
            questions.append(
                f"[{e.bot_id}] {e.description}. Is this a data point against the suggestion's thesis, "
                f"or is the negative day explainable by other factors? Examine root causes."
            )

        for e in filter_anomalies[:1]:
            questions.append(
                f"[{e.bot_id}] {e.description}. Should this filter's threshold be adjusted? "
                f"Analyze the blocked trades' characteristics vs the filter's intended purpose."
            )

        for e in drawdown_spikes[:1]:
            questions.append(
                f"[{e.bot_id}] {e.description}. Is this a process failure (bad entries/exits) "
                f"or a legitimate market condition? Review the losing trades' execution quality."
            )

        # Default question if nothing significant
        if not questions:
            questions.append(
                "Today was routine. Are there any subtle patterns across bots that the automated "
                "detectors might be missing? Look for correlations between bots or gradual trends."
            )

        return questions[:5]

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_json(self, bot_id: str, filename: str) -> dict | list | None:
        return self._load_json_for_date(bot_id, self._date, filename)

    def _load_json_for_date(
        self, bot_id: str, date: str, filename: str
    ) -> dict | list | None:
        path = self._curated_dir / date / bot_id / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _load_trailing_summaries(self, bot_id: str) -> list[dict]:
        date_obj = datetime.strptime(self._date, "%Y-%m-%d")
        summaries = []
        for d in range(1, self._trailing_days + 1):
            prev_date = (date_obj - timedelta(days=d)).strftime("%Y-%m-%d")
            s = self._load_json_for_date(bot_id, prev_date, "summary.json")
            if s and isinstance(s, dict):
                summaries.append(s)
        return summaries

    @staticmethod
    def _get_pnl(summary: dict) -> float:
        val = summary.get("net_pnl")
        if val is None:
            val = summary.get("gross_pnl", 0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
