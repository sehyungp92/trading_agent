# skills/contradiction_detector.py
"""Contradiction detector — flags temporal inconsistencies across daily reports.

Loads the last N days of curated JSON files per bot, compares deterministically
across days, and flags contradictions. Output is injected into the daily
PromptPackage so Claude addresses inconsistencies.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from trading_assistant.schemas.contradiction import (
    ContradictionItem,
    ContradictionReport,
    ContradictionType,
)


class ContradictionDetector:
    """Detect contradictions across multi-day curated data."""

    def __init__(
        self,
        date: str,
        bots: list[str],
        curated_dir: Path,
        lookback_days: int = 3,
    ) -> None:
        self.date = date
        self.bots = bots
        self.curated_dir = Path(curated_dir)
        self.lookback_days = lookback_days

    def detect(self) -> ContradictionReport:
        """Run all contradiction checks and return a report."""
        items: list[ContradictionItem] = []

        for bot in self.bots:
            multi_day = self._load_multi_day(bot)
            if len(multi_day) < 2:
                continue

            items.extend(self._check_regime_direction_conflict(bot, multi_day))
            items.extend(self._check_factor_quality_divergence(bot, multi_day))
            items.extend(self._check_exit_process_conflict(bot, multi_day))
            items.extend(self._check_risk_exposure_conflict(bot, multi_day))

        return ContradictionReport(
            date=self.date,
            lookback_days=self.lookback_days,
            items=items,
            bots_analyzed=self.bots,
        )

    def _load_multi_day(self, bot: str) -> dict[str, dict[str, dict]]:
        """Load key JSON files for the last N days.

        Returns {date_str: {file_key: data}} traversing backwards from self.date.
        Reuses the directory traversal pattern from retrospective_builder.py.
        """
        result: dict[str, dict[str, dict]] = {}
        base_date = datetime.strptime(self.date, "%Y-%m-%d")
        files_to_load = [
            "summary.json",
            "regime_analysis.json",
            "factor_attribution.json",
            "exit_efficiency.json",
            "process_failures.json",
        ]

        for i in range(self.lookback_days):
            date_str = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
            bot_dir = self.curated_dir / date_str / bot
            if not bot_dir.is_dir():
                continue

            day_data: dict[str, dict] = {}
            for filename in files_to_load:
                path = bot_dir / filename
                if path.exists():
                    try:
                        day_data[filename.replace(".json", "")] = json.loads(
                            path.read_text(encoding="utf-8")
                        )
                    except (json.JSONDecodeError, OSError):
                        pass
            if day_data:
                result[date_str] = day_data

        return result

    def _check_regime_direction_conflict(
        self, bot: str, multi_day: dict[str, dict[str, dict]],
    ) -> list[ContradictionItem]:
        """Regime X profitable day A, unprofitable day B with no regime change."""
        items: list[ContradictionItem] = []
        days = sorted(multi_day.keys())

        for i in range(len(days) - 1):
            day_a, day_b = days[i], days[i + 1]
            regime_a = multi_day[day_a].get("regime_analysis", {})
            regime_b = multi_day[day_b].get("regime_analysis", {})
            summary_a = multi_day[day_a].get("summary", {})
            summary_b = multi_day[day_b].get("summary", {})

            # Get current regimes from summaries
            current_regime_a = summary_a.get("market_regime", "")
            current_regime_b = summary_b.get("market_regime", "")

            if not current_regime_a or current_regime_a != current_regime_b:
                continue  # regime changed — not a contradiction

            # Compare PnL by regime across days
            regimes_a = self._extract_regime_pnl(regime_a)
            regimes_b = self._extract_regime_pnl(regime_b)

            for regime, pnl_a in regimes_a.items():
                pnl_b = regimes_b.get(regime)
                if pnl_b is None:
                    continue
                # Flag if profitable on one day and significantly unprofitable on the other
                if pnl_a > 0 and pnl_b < 0 and abs(pnl_b) > pnl_a * 0.5:
                    items.append(ContradictionItem(
                        type=ContradictionType.REGIME_DIRECTION_CONFLICT,
                        bot_id=bot,
                        description=(
                            f"Regime '{regime}' was profitable on {day_a} "
                            f"(${pnl_a:.0f}) but unprofitable on {day_b} "
                            f"(${pnl_b:.0f}) despite no regime change."
                        ),
                        day_a=day_a,
                        day_b=day_b,
                        severity="medium",
                        evidence={"regime": regime, "pnl_a": pnl_a, "pnl_b": pnl_b},
                    ))

        return items

    def _check_factor_quality_divergence(
        self, bot: str, multi_day: dict[str, dict[str, dict]],
    ) -> list[ContradictionItem]:
        """Factor win_rate drops >20pp while avg_contribution stays high."""
        items: list[ContradictionItem] = []
        days = sorted(multi_day.keys())

        for i in range(len(days) - 1):
            day_a, day_b = days[i], days[i + 1]
            factors_a = self._extract_factor_stats(multi_day[day_a].get("factor_attribution", {}))
            factors_b = self._extract_factor_stats(multi_day[day_b].get("factor_attribution", {}))

            for name, stats_a in factors_a.items():
                stats_b = factors_b.get(name)
                if stats_b is None:
                    continue

                wr_a = stats_a.get("win_rate", 0)
                wr_b = stats_b.get("win_rate", 0)
                contrib_a = stats_a.get("avg_contribution", 0)
                contrib_b = stats_b.get("avg_contribution", 0)

                # Win rate dropped >20pp while contribution stayed high
                wr_drop = wr_a - wr_b
                if wr_drop > 0.20 and contrib_b > 0 and abs(contrib_b) >= abs(contrib_a) * 0.7:
                    items.append(ContradictionItem(
                        type=ContradictionType.FACTOR_QUALITY_DIVERGENCE,
                        bot_id=bot,
                        description=(
                            f"Factor '{name}' win rate dropped from "
                            f"{wr_a:.0%} to {wr_b:.0%} ({wr_drop:.0%} drop) "
                            f"between {day_a} and {day_b}, but avg contribution "
                            f"remained high ({contrib_b:.3f} vs {contrib_a:.3f})."
                        ),
                        day_a=day_a,
                        day_b=day_b,
                        severity="high" if wr_drop > 0.30 else "medium",
                        evidence={
                            "factor": name,
                            "win_rate_a": wr_a,
                            "win_rate_b": wr_b,
                            "contribution_a": contrib_a,
                            "contribution_b": contrib_b,
                        },
                    ))

        return items

    def _check_exit_process_conflict(
        self, bot: str, multi_day: dict[str, dict[str, dict]],
    ) -> list[ContradictionItem]:
        """Exit efficiency improving while process_failures count increasing."""
        items: list[ContradictionItem] = []
        days = sorted(multi_day.keys())

        for i in range(len(days) - 1):
            day_a, day_b = days[i], days[i + 1]
            exit_a = multi_day[day_a].get("exit_efficiency", {})
            exit_b = multi_day[day_b].get("exit_efficiency", {})
            failures_a = multi_day[day_a].get("process_failures", [])
            failures_b = multi_day[day_b].get("process_failures", [])

            eff_a = exit_a.get("avg_efficiency", 0)
            eff_b = exit_b.get("avg_efficiency", 0)
            fail_count_a = len(failures_a) if isinstance(failures_a, list) else 0
            fail_count_b = len(failures_b) if isinstance(failures_b, list) else 0

            # Efficiency improving but failures increasing
            if eff_b > eff_a and fail_count_b > fail_count_a and fail_count_b >= 3:
                items.append(ContradictionItem(
                    type=ContradictionType.EXIT_PROCESS_CONFLICT,
                    bot_id=bot,
                    description=(
                        f"Exit efficiency improved from {eff_a:.0%} to {eff_b:.0%} "
                        f"between {day_a} and {day_b}, but process failures "
                        f"increased from {fail_count_a} to {fail_count_b}."
                    ),
                    day_a=day_a,
                    day_b=day_b,
                    severity="medium",
                    evidence={
                        "efficiency_a": eff_a,
                        "efficiency_b": eff_b,
                        "failures_a": fail_count_a,
                        "failures_b": fail_count_b,
                    },
                ))

        return items

    def _check_risk_exposure_conflict(
        self, bot: str, multi_day: dict[str, dict[str, dict]],
    ) -> list[ContradictionItem]:
        """Exposure rising while max_drawdown deepening."""
        items: list[ContradictionItem] = []
        days = sorted(multi_day.keys())

        for i in range(len(days) - 1):
            day_a, day_b = days[i], days[i + 1]
            summary_a = multi_day[day_a].get("summary", {})
            summary_b = multi_day[day_b].get("summary", {})

            exposure_a = summary_a.get("exposure_pct", 0)
            exposure_b = summary_b.get("exposure_pct", 0)
            dd_a = abs(summary_a.get("max_drawdown_pct", 0))
            dd_b = abs(summary_b.get("max_drawdown_pct", 0))

            # Exposure increasing while drawdown deepening
            if (
                exposure_b > exposure_a
                and dd_b > dd_a
                and exposure_b > 0
                and dd_b > 0
                and (exposure_b - exposure_a) > 5  # >5pp increase
                and (dd_b - dd_a) > 1  # >1pp deeper drawdown
            ):
                items.append(ContradictionItem(
                    type=ContradictionType.RISK_EXPOSURE_CONFLICT,
                    bot_id=bot,
                    description=(
                        f"Exposure increased from {exposure_a:.1f}% to {exposure_b:.1f}% "
                        f"between {day_a} and {day_b}, while max drawdown deepened "
                        f"from {dd_a:.1f}% to {dd_b:.1f}%. Increasing risk into losses."
                    ),
                    day_a=day_a,
                    day_b=day_b,
                    severity="high",
                    evidence={
                        "exposure_a": exposure_a,
                        "exposure_b": exposure_b,
                        "drawdown_a": dd_a,
                        "drawdown_b": dd_b,
                    },
                ))

        return items

    @staticmethod
    def _extract_regime_pnl(regime_data: dict) -> dict[str, float]:
        """Extract regime -> PnL from regime_analysis data."""
        result: dict[str, float] = {}
        # Handle both list and dict formats
        regimes = regime_data.get("regimes", regime_data.get("by_regime", []))
        if isinstance(regimes, list):
            for entry in regimes:
                name = entry.get("regime", entry.get("name", ""))
                pnl = entry.get("total_pnl", entry.get("pnl", 0))
                if name:
                    result[name] = pnl
        elif isinstance(regimes, dict):
            for name, data in regimes.items():
                if isinstance(data, dict):
                    result[name] = data.get("total_pnl", data.get("pnl", 0))
                elif isinstance(data, (int, float)):
                    result[name] = data
        return result

    @staticmethod
    def _extract_factor_stats(factor_data: dict) -> dict[str, dict]:
        """Extract factor_name -> stats from factor_attribution data."""
        result: dict[str, dict] = {}
        factors = factor_data.get("factors", [])
        if isinstance(factors, list):
            for entry in factors:
                name = entry.get("factor_name", "")
                if name:
                    result[name] = entry
        return result
