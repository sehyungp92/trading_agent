"""Report quality gate — Definition of Done for daily reports.

Validates curated data completeness. Degrades gracefully when some bots
or files are missing — returns a confidence score instead of blocking.
"""
from __future__ import annotations

import json
from pathlib import Path

from trading_assistant.schemas.report_checklist import ReportChecklist, CheckResult

_EXPECTED_BOT_FILES = [
    "summary.json",
    "trades.jsonl",
    "missed.jsonl",
    "winners.json",
    "losers.json",
    "process_failures.json",
    "notable_missed.json",
    "regime_analysis.json",
    "filter_analysis.json",
    "root_cause_summary.json",
    "hourly_performance.json",
    "slippage_stats.json",
    "factor_attribution.json",
    "exit_efficiency.json",
]


class QualityGate:
    """Runs all quality checks for a daily report. Always allows proceeding."""

    def __init__(
        self,
        report_id: str,
        date: str,
        expected_bots: list[str],
        curated_dir: Path,
    ) -> None:
        self.report_id = report_id
        self.date = date
        self.expected_bots = expected_bots
        self.curated_dir = curated_dir

    def run(self) -> ReportChecklist:
        """Run all checks and return checklist with completeness score."""
        checks: list[CheckResult] = []

        checks.append(self._check_all_bots_reported())
        checks.extend(self._check_curated_files())
        checks.append(self._check_portfolio_risk_card())

        available, missing = self._partition_bots()
        completeness = self._compute_file_completeness()

        return ReportChecklist(
            report_id=self.report_id,
            checks=checks,
            can_proceed=True,
            data_completeness=completeness,
            available_bots=available,
            missing_bots=missing,
        )

    def write_checklist(self, checklist: ReportChecklist, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(checklist.model_dump(mode="json"), indent=2, default=str)
        )

    def _partition_bots(self) -> tuple[list[str], list[str]]:
        date_dir = self.curated_dir / self.date
        available = [b for b in self.expected_bots if (date_dir / b).is_dir()]
        missing = [b for b in self.expected_bots if b not in available]
        return available, missing

    def _compute_file_completeness(self) -> float:
        if not self.expected_bots:
            return 0.0
        total_expected = len(self.expected_bots) * len(_EXPECTED_BOT_FILES)
        if total_expected == 0:
            return 0.0
        date_dir = self.curated_dir / self.date
        found = 0
        for bot in self.expected_bots:
            bot_dir = date_dir / bot
            if not bot_dir.is_dir():
                continue
            for f in _EXPECTED_BOT_FILES:
                if (bot_dir / f).exists():
                    found += 1
        return found / total_expected

    def _check_all_bots_reported(self) -> CheckResult:
        available, missing = self._partition_bots()
        if missing:
            return CheckResult(
                name="all_bots_reported",
                passed=False,
                detail=f"Missing: {', '.join(missing)} ({len(available)}/{len(self.expected_bots)} bots)",
            )
        return CheckResult(
            name="all_bots_reported",
            passed=True,
            detail=f"{len(available)}/{len(self.expected_bots)} bots",
        )

    def _check_curated_files(self) -> list[CheckResult]:
        results: list[CheckResult] = []
        date_dir = self.curated_dir / self.date
        for bot in self.expected_bots:
            bot_dir = date_dir / bot
            if not bot_dir.is_dir():
                continue
            missing = [f for f in _EXPECTED_BOT_FILES if not (bot_dir / f).exists()]
            if missing:
                results.append(CheckResult(
                    name=f"curated_files_{bot}",
                    passed=False,
                    detail=f"Missing: {', '.join(missing)}",
                ))
            else:
                results.append(CheckResult(
                    name=f"curated_files_{bot}",
                    passed=True,
                    detail="All files present",
                ))
        return results

    def _check_portfolio_risk_card(self) -> CheckResult:
        risk_path = self.curated_dir / self.date / "portfolio_risk_card.json"
        if risk_path.exists():
            return CheckResult(name="portfolio_risk_card", passed=True, detail="Computed")
        return CheckResult(name="portfolio_risk_card", passed=False, detail="Portfolio risk card not computed")
