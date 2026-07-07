"""TSV logging and routing for auto-backtesting results."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backtests.stock.auto.robustness import RobustnessReport
from backtests.stock.auto.scoring import CompositeScore

logger = logging.getLogger(__name__)

_TSV_COLUMNS = [
    "experiment_id", "strategy", "tier", "type",
    "baseline_score", "experiment_score", "delta_pct",
    "robust", "status", "description", "timestamp",
]


@dataclass
class ExperimentResult:
    experiment_id: str
    strategy: str
    tier: int
    type: str
    baseline_score: float
    experiment_score: float
    delta_pct: float
    robust: bool
    status: str
    description: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")


class ResultsTracker:
    """TSV-based results logger with routing logic."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tsv_path = self.output_dir / "results.tsv"
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.tsv_path.exists():
            with open(self.tsv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(_TSV_COLUMNS)

    def record(self, result: ExperimentResult) -> None:
        """Append a result row to the TSV file."""
        with open(self.tsv_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                result.experiment_id,
                result.strategy,
                result.tier,
                result.type,
                f"{result.baseline_score:.6f}",
                f"{result.experiment_score:.6f}",
                f"{result.delta_pct:.4f}",
                result.robust,
                result.status,
                result.description,
                result.timestamp,
            ])
        logger.info("Recorded: %s → %s (delta=%.2f%%)",
                     result.experiment_id, result.status, result.delta_pct * 100)

    def load_all(self) -> list[ExperimentResult]:
        """Read all results from the TSV file."""
        if not self.tsv_path.exists():
            return []

        results = []
        with open(self.tsv_path, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                results.append(ExperimentResult(
                    experiment_id=row["experiment_id"],
                    strategy=row["strategy"],
                    tier=int(row["tier"]),
                    type=row["type"],
                    baseline_score=float(row["baseline_score"]),
                    experiment_score=float(row["experiment_score"]),
                    delta_pct=float(row["delta_pct"]),
                    robust=row["robust"] == "True",
                    status=row["status"],
                    description=row["description"],
                    timestamp=row["timestamp"],
                ))
        return results

    def completed_ids(self) -> set[str]:
        """Get set of completed experiment IDs for --resume support."""
        return {r.experiment_id for r in self.load_all()}

    def decide(
        self,
        baseline: CompositeScore,
        experiment: CompositeScore,
        robustness: RobustnessReport | None,
        is_ablation: bool = False,
    ) -> str:
        """Route an experiment to a status.

        Status values:
          APPROVE: delta >= +5% AND robustness passes_all
          TEST_FURTHER: delta >= +2%, or delta >= +5% but robustness partial
          DISCARD: delta < +2% or hard reject
          UNWIRED: delta == 0.0 exactly for ablation (flag not checked in engine)
          CRASH: set by harness on exception (not here)
        """
        if experiment.rejected:
            return "DISCARD"

        if baseline.total == 0:
            delta = 0.0
        else:
            delta = (experiment.total - baseline.total) / baseline.total

        # Check for unwired ablation flags
        if is_ablation and abs(delta) < 1e-10:
            return "UNWIRED"

        if delta >= 0.05:
            if robustness and robustness.passes_all:
                return "APPROVE"
            return "TEST_FURTHER"

        if delta >= 0.02:
            return "TEST_FURTHER"

        return "DISCARD"
