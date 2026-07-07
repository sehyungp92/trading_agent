"""Golden day loader — reads frozen test datasets for regression testing.

Each golden day directory has a fixed structure:
  <date>/
    raw_events/
      trades.json              # list of trade event dicts
      missed.json              # list of missed opportunity dicts
    expected_classifications.json  # trade_id → {root_causes, process_quality_score}
    expected_curated/
      summary.json             # what the pipeline should produce
    human_feedback.json        # corrections from that day
    reference_report.md        # the report rated "good"
    metadata.json              # date, bots, top_anomaly, biggest_loss_driver, crowding_alerts
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GoldenDay:
    """A frozen test dataset with known-good outcomes."""

    date: str
    bots: list[str]
    trades: list[dict]
    missed: list[dict]
    expected_classifications: dict[str, dict]
    expected_curated: dict[str, dict]  # filename → content
    human_feedback: list[dict]
    reference_report: str
    top_anomaly: str = ""
    biggest_loss_driver: str = ""
    crowding_alerts: list[str] = field(default_factory=list)


def load_golden_days(base_dir: Path) -> list[GoldenDay]:
    """Load all golden days from the base directory."""
    days: list[GoldenDay] = []

    for day_dir in sorted(base_dir.iterdir()):
        if not day_dir.is_dir():
            continue

        metadata_path = day_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())

        # Raw events
        raw_dir = day_dir / "raw_events"
        trades = json.loads((raw_dir / "trades.json").read_text()) if (raw_dir / "trades.json").exists() else []
        missed = json.loads((raw_dir / "missed.json").read_text()) if (raw_dir / "missed.json").exists() else []

        # Expected classifications
        class_path = day_dir / "expected_classifications.json"
        expected_classifications = json.loads(class_path.read_text()) if class_path.exists() else {}

        # Expected curated
        expected_curated: dict[str, dict] = {}
        curated_dir = day_dir / "expected_curated"
        if curated_dir.is_dir():
            for f in curated_dir.iterdir():
                if f.suffix == ".json":
                    expected_curated[f.name] = json.loads(f.read_text())

        # Human feedback
        feedback_path = day_dir / "human_feedback.json"
        human_feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else []

        # Reference report
        report_path = day_dir / "reference_report.md"
        reference_report = report_path.read_text() if report_path.exists() else ""

        days.append(GoldenDay(
            date=metadata["date"],
            bots=metadata.get("bots", []),
            trades=trades,
            missed=missed,
            expected_classifications=expected_classifications,
            expected_curated=expected_curated,
            human_feedback=human_feedback,
            reference_report=reference_report,
            top_anomaly=metadata.get("top_anomaly", ""),
            biggest_loss_driver=metadata.get("biggest_loss_driver", ""),
            crowding_alerts=metadata.get("crowding_alerts", []),
        ))

    return days
