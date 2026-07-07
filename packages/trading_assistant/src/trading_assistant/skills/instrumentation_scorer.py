# skills/instrumentation_scorer.py
"""Per-bot instrumentation readiness scoring.

Checks data completeness across the curated directory to surface which bots
have sufficient instrumentation for each analysis capability. Purely
deterministic — reads curated files and produces a readiness scorecard.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field


class FieldReadiness(BaseModel):
    """Readiness status for a single data field."""

    field_name: str
    present_days: int = 0
    total_days: int = 0
    coverage: float = 0.0  # present_days / total_days


class CapabilityReadiness(BaseModel):
    """Readiness for a specific analysis capability."""

    capability: str  # e.g. "exit_analysis", "regime_analysis", "slippage"
    ready: bool = False
    score: float = 0.0  # 0.0-1.0
    missing_fields: list[str] = Field(default_factory=list)
    detail: str = ""


class BotReadinessReport(BaseModel):
    """Per-bot instrumentation readiness scorecard."""

    bot_id: str
    overall_score: float = 0.0  # 0.0-1.0
    days_with_data: int = 0
    days_checked: int = 0
    capabilities: list[CapabilityReadiness] = Field(default_factory=list)
    field_coverage: list[FieldReadiness] = Field(default_factory=list)


# Required fields per analysis capability
_CAPABILITY_FIELDS: dict[str, list[str]] = {
    "basic_analysis": [
        "net_pnl", "total_trades", "winning_trades",
    ],
    "process_quality": [
        "avg_process_quality", "process_quality_avg",
    ],
    "exit_analysis": [
        "exit_efficiency",
    ],
    "regime_analysis": [
        "regime",
    ],
    "slippage_analysis": [
        "slippage_stats",
    ],
    "factor_attribution": [
        "factor_attribution",
    ],
    "signal_health": [
        "signal_strength",
    ],
    "drawdown_analysis": [
        "max_drawdown_pct",
    ],
    "indicator_snapshots": [
        "indicator_snapshot", "indicator_snapshots",
    ],
    "filter_decisions": [
        "filter_decision", "filter_decisions",
    ],
    "parameter_tracking": [
        "parameter_change", "parameter_changes",
    ],
    "order_book_analysis": [
        "orderbook_stats", "order_book", "orderbook",
    ],
    "sizing_analysis": [
        "sizing_analysis", "position_size", "sizing_inputs",
    ],
    "portfolio_context": [
        "portfolio_context", "portfolio_correlation",
    ],
    "fill_quality": [
        "fill_quality", "execution_quality",
    ],
}

# Curated files expected per day per bot
_EXPECTED_FILES = [
    "summary.json",
    "trades.jsonl",
]

_OPTIONAL_FILES = [
    "missed.jsonl",
    "hourly_performance.json",
    "slippage_stats.json",
    "regime_analysis.json",
    "factor_attribution.json",
    "exit_efficiency.json",
    "indicator_snapshots.json",
    "filter_decisions.json",
    "parameter_changes.json",
    "orderbook_stats.json",
    "sizing_analysis.json",
    "portfolio_context.json",
    "fill_quality.json",
    "signal_health.json",
]


class InstrumentationScorer:
    """Scores per-bot data completeness for analysis readiness."""

    def __init__(self, curated_dir: Path, lookback_days: int = 30) -> None:
        self._curated_dir = curated_dir
        self._lookback_days = lookback_days

    def score_bot(self, bot_id: str, as_of_date: str | None = None) -> BotReadinessReport:
        """Compute readiness scorecard for a single bot."""
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        end = datetime.strptime(as_of_date, "%Y-%m-%d")
        summaries: list[dict] = []
        days_with_data = 0
        file_presence: dict[str, int] = {f: 0 for f in _EXPECTED_FILES + _OPTIONAL_FILES}

        for d in range(self._lookback_days):
            date_str = (end - timedelta(days=d)).strftime("%Y-%m-%d")
            bot_dir = self._curated_dir / date_str / bot_id

            if not bot_dir.exists():
                continue

            days_with_data += 1

            # Check file presence
            for fname in list(file_presence.keys()):
                if (bot_dir / fname).exists():
                    file_presence[fname] += 1

            # Load summary for field checks
            summary_path = bot_dir / "summary.json"
            if summary_path.exists():
                try:
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        summaries.append(data)
                except (json.JSONDecodeError, OSError):
                    pass

        # Compute field coverage
        field_coverage = self._compute_field_coverage(
            summaries, file_presence, days_with_data,
        )

        # Compute capability readiness
        capabilities = self._compute_capabilities(
            field_coverage, file_presence, days_with_data,
        )

        # Overall score: weighted average of capability scores
        if capabilities:
            overall = sum(c.score for c in capabilities) / len(capabilities)
        else:
            overall = 0.0

        return BotReadinessReport(
            bot_id=bot_id,
            overall_score=round(overall, 3),
            days_with_data=days_with_data,
            days_checked=self._lookback_days,
            capabilities=capabilities,
            field_coverage=field_coverage,
        )

    def score_all_bots(
        self, bots: list[str], as_of_date: str | None = None,
    ) -> dict[str, BotReadinessReport]:
        """Compute readiness scorecards for all bots."""
        return {bot: self.score_bot(bot, as_of_date) for bot in bots}

    @staticmethod
    def _compute_field_coverage(
        summaries: list[dict],
        file_presence: dict[str, int],
        days_with_data: int,
    ) -> list[FieldReadiness]:
        """Check which fields are consistently present in summaries."""
        if not days_with_data:
            return []

        # Core summary fields
        key_fields = [
            "net_pnl", "total_trades", "winning_trades", "losing_trades",
            "avg_win", "avg_loss", "max_drawdown_pct",
            "avg_process_quality", "process_quality_avg",
            "signal_strength", "regime",
        ]

        results: list[FieldReadiness] = []
        for field in key_fields:
            present = sum(
                1 for s in summaries
                if s.get(field) is not None
            )
            results.append(FieldReadiness(
                field_name=field,
                present_days=present,
                total_days=len(summaries),
                coverage=round(present / len(summaries), 3) if summaries else 0.0,
            ))

        # File-level coverage
        for fname, count in file_presence.items():
            results.append(FieldReadiness(
                field_name=f"file:{fname}",
                present_days=count,
                total_days=days_with_data,
                coverage=round(count / days_with_data, 3) if days_with_data else 0.0,
            ))

        return results

    @staticmethod
    def _compute_capabilities(
        field_coverage: list[FieldReadiness],
        file_presence: dict[str, int],
        days_with_data: int,
    ) -> list[CapabilityReadiness]:
        """Determine which analysis capabilities are ready."""
        # Build lookup: field_name → coverage
        cov_map: dict[str, float] = {
            f.field_name: f.coverage for f in field_coverage
        }

        capabilities: list[CapabilityReadiness] = []
        for cap_name, required_fields in _CAPABILITY_FIELDS.items():
            # Check coverage of required fields (OR logic for alternatives)
            best_coverage = 0.0
            missing: list[str] = []
            for field in required_fields:
                # Check both summary field and file-level
                field_cov = cov_map.get(field, 0.0)
                file_cov = cov_map.get(f"file:{field}.json", 0.0)
                best = max(field_cov, file_cov)
                if best > best_coverage:
                    best_coverage = best
                if best < 0.5:
                    missing.append(field)

            ready = best_coverage >= 0.7 and days_with_data >= 7
            score = best_coverage * min(1.0, days_with_data / 14)

            detail = f"{best_coverage:.0%} field coverage over {days_with_data} days"
            if missing:
                detail += f"; missing: {', '.join(missing)}"

            capabilities.append(CapabilityReadiness(
                capability=cap_name,
                ready=ready,
                score=round(score, 3),
                missing_fields=missing,
                detail=detail,
            ))

        return capabilities
