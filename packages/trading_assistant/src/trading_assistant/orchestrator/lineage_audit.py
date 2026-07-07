"""Lineage and telemetry coverage audit for monthly validation."""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

from trading_assistant.schemas.lineage_audit import LineageGapReport, LineageSeverity
from trading_assistant.schemas.proposal_ledger import ProposalCandidate, ProposalKind, ProposalSource
from trading_assistant.schemas.telemetry_manifest import TelemetryEligibility, TelemetryManifest
from trading_assistant.skills.lineage_utils import build_lineage_summary, event_strategy_id, event_value

logger = logging.getLogger(__name__)


class LineageAuditor:
    """Scans curated event JSONL and writes lineage gap findings."""

    def __init__(
        self,
        curated_dir: Path,
        findings_dir: Path,
        *,
        required_lineage_ratio: float = 0.95,
        proposal_ledger: object | None = None,
    ) -> None:
        self._curated_dir = Path(curated_dir)
        self._findings_dir = Path(findings_dir)
        self._required_lineage_ratio = required_lineage_ratio
        self._proposal_ledger = proposal_ledger

    def audit(
        self,
        *,
        bot_id: str,
        window_start: date,
        window_end: date,
        strategy_id: str = "",
        emit_instrumentation_request: bool = True,
    ) -> list[LineageGapReport]:
        events = self._load_events(bot_id, window_start, window_end)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for event in events:
            sid = event_strategy_id(event)
            if strategy_id and sid != strategy_id:
                continue
            grouped[sid].append(event)
        if strategy_id and strategy_id not in grouped:
            grouped[strategy_id] = []
        if not grouped:
            grouped[""] = []

        reports = [
            self._build_report(bot_id, sid, window_start, window_end, grouped_events)
            for sid, grouped_events in sorted(grouped.items())
        ]
        for report in reports:
            if report.severity != LineageSeverity.OK:
                self._append_gap(report)
                if emit_instrumentation_request:
                    self._emit_instrumentation_request(report)
        return reports

    def build_telemetry_manifest(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        run_month: str,
        window_start: date,
        window_end: date,
        output_path: Path | None = None,
    ) -> TelemetryManifest:
        events = [
            event for event in self._load_events(bot_id, window_start, window_end)
            if not strategy_id or event_strategy_id(event) == strategy_id
        ]
        summary = build_lineage_summary(events)
        event_ids = [str(event_value(event, "event_id") or "") for event in events]
        duplicate_count = len(event_ids) - len({eid for eid in event_ids if eid})
        counts = Counter(str(event.get("event_type") or event.get("_event_file") or "unknown") for event in events)
        missing_counts = summary["missing_field_counts"]
        known_gaps = [
            f"{field}: {count} missing"
            for field, count in sorted(missing_counts.items())
            if count
        ]
        if not events:
            eligibility = TelemetryEligibility.INSUFFICIENT_DATA
        elif summary["lineage_coverage_ratio"] < self._required_lineage_ratio:
            eligibility = TelemetryEligibility.INSUFFICIENT_LINEAGE
        else:
            eligibility = TelemetryEligibility.AUTHORITATIVE
        manifest = TelemetryManifest(
            bot_id=bot_id,
            strategy_id=strategy_id,
            run_month=run_month,
            window_start=window_start,
            window_end=window_end,
            event_counts_by_type=dict(counts),
            lineage_coverage_ratio=summary["lineage_coverage_ratio"],
            missing_field_counts=missing_counts,
            duplicate_count=max(0, duplicate_count),
            known_gaps=known_gaps,
            authoritative_eligibility=eligibility,
            total_events=len(events),
        )
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def _build_report(
        self,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
        events: list[dict],
    ) -> LineageGapReport:
        summary = build_lineage_summary(events)
        ratio = summary["lineage_coverage_ratio"]
        if not events:
            severity = LineageSeverity.WARNING
            action = "No monthly-critical telemetry found for this window."
        elif ratio < self._required_lineage_ratio:
            severity = LineageSeverity.BLOCKING
            action = "Add missing strategy/config/deployment lineage before authoritative validation."
        elif summary["lineage_gap"]:
            severity = LineageSeverity.WARNING
            action = "Lineage is mostly complete; close residual gaps before approval-gated rollout."
        else:
            severity = LineageSeverity.OK
            action = ""
        return LineageGapReport(
            bot_id=bot_id,
            strategy_id=strategy_id,
            window_start=window_start,
            window_end=window_end,
            total_events=len(events),
            missing_field_counts=summary["missing_field_counts"],
            lineage_coverage_ratio=ratio,
            severity=severity,
            recommended_action=action,
        )

    def _load_events(self, bot_id: str, window_start: date, window_end: date) -> list[dict]:
        events: list[dict] = []
        if not self._curated_dir.exists():
            return events
        for date_dir in sorted(self._curated_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            try:
                day = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < window_start or day > window_end:
                continue
            bot_dir = date_dir / bot_id
            if not bot_dir.is_dir():
                continue
            for filename, event_type in (("trades.jsonl", "trade"), ("missed.jsonl", "missed_opportunity")):
                path = bot_dir / filename
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        payload.setdefault("bot_id", bot_id)
                        payload.setdefault("event_type", event_type)
                        payload["_event_file"] = filename
                        events.append(payload)
        return events

    def _append_gap(self, report: LineageGapReport) -> None:
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        path = self._findings_dir / "lineage_gaps.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(report.model_dump_json() + "\n")

    def _emit_instrumentation_request(self, report: LineageGapReport) -> None:
        if self._proposal_ledger is None or report.severity != LineageSeverity.BLOCKING:
            return
        try:
            from trading_assistant.skills.proposal_ledger import make_proposal_id

            proposal_id = make_proposal_id(
                ProposalSource.INSTRUMENTATION,
                report.bot_id,
                ProposalKind.INSTRUMENTATION_REQUEST,
                "Fill monthly lineage fields",
                strategy_id=report.strategy_id,
                link_key=f"{report.window_start}:{report.window_end}",
            )
            candidate = ProposalCandidate(
                proposal_id=proposal_id,
                source=ProposalSource.INSTRUMENTATION,
                kind=ProposalKind.INSTRUMENTATION_REQUEST,
                bot_id=report.bot_id,
                strategy_id=report.strategy_id,
                title="Fill monthly lineage fields",
                description=report.recommended_action,
                evaluation_method="lineage_audit",
            )
            self._proposal_ledger.record_candidate(candidate)
        except Exception:
            logger.debug("Could not emit lineage instrumentation request", exc_info=True)
