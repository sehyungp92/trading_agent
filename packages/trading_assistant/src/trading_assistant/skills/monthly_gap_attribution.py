"""Deterministic monthly gap attribution for shadow validation."""
from __future__ import annotations

from trading_assistant.schemas.monthly_validation import GapAttribution, GapAttributionCategory
from trading_assistant.schemas.replay_parity import ReplayParityReport, ReplayParityStatus


class MonthlyGapAttributor:
    """Classify the first-order reason a monthly run is not clean."""

    def attribute(
        self,
        *,
        coverage_blocking_reasons: list[str] | None = None,
        telemetry_known_gaps: list[str] | None = None,
        parity_report: ReplayParityReport | None = None,
        objective_delta: float | None = None,
    ) -> GapAttribution:
        coverage_blocking_reasons = coverage_blocking_reasons or []
        telemetry_known_gaps = telemetry_known_gaps or []
        if coverage_blocking_reasons:
            return GapAttribution(
                primary_category=GapAttributionCategory.DATA_GAP,
                summary="Market-data coverage is insufficient for authoritative replay.",
                confidence=1.0,
            )
        if telemetry_known_gaps:
            return GapAttribution(
                primary_category=GapAttributionCategory.DATA_GAP,
                summary="Telemetry lineage is incomplete for authoritative live-vs-replay mapping.",
                confidence=1.0,
            )
        if parity_report is not None:
            if parity_report.status == ReplayParityStatus.FAIL:
                if abs(parity_report.fee_slippage_delta_bps) > 0:
                    category = GapAttributionCategory.SLIPPAGE_COST_DRIFT
                elif abs(parity_report.drawdown_delta_pct) > 0:
                    category = GapAttributionCategory.EXECUTION_DRIFT
                else:
                    category = GapAttributionCategory.EXIT_MISMATCH
                return GapAttribution(
                    primary_category=category,
                    summary="Replay does not match observed incumbent behavior closely enough.",
                    confidence=0.8,
                    evidence_paths=parity_report.evidence_paths,
                )
            if parity_report.status == ReplayParityStatus.INSUFFICIENT_DATA:
                return GapAttribution(
                    primary_category=GapAttributionCategory.OPPORTUNITY_SCARCITY,
                    summary="Insufficient live and replay trades for a stable monthly verdict.",
                    confidence=0.7,
                    evidence_paths=parity_report.evidence_paths,
                )
        if objective_delta is not None and objective_delta < -0.05:
            return GapAttribution(
                primary_category=GapAttributionCategory.BROAD_DEGRADATION,
                summary="Latest-month objective deteriorated materially versus expectation.",
                confidence=0.7,
            )
        return GapAttribution(
            primary_category=GapAttributionCategory.NONE,
            summary="No material monthly gap detected.",
            confidence=0.6,
        )
