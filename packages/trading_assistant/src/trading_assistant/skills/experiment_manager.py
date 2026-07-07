# skills/experiment_manager.py
"""Experiment manager — A/B testing lifecycle with statistical analysis.

Manages experiments from creation through conclusion with Welch's t-test
for statistical significance testing.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    VariantMetrics,
)

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages experiment lifecycle: create, monitor, analyze, conclude."""

    def __init__(self, findings_dir: Path, min_trades: int = 30) -> None:
        self._path = findings_dir / "experiments.jsonl"
        self._results_path = findings_dir / "experiment_results.jsonl"
        self._variant_data_dir = findings_dir / "experiment_data"
        self._min_trades = min_trades
        self._lock = threading.Lock()

    def create_experiment(self, config: ExperimentConfig) -> ExperimentConfig:
        """Persist a new experiment. Deduplicates by experiment_id."""
        existing = self.get_by_id(config.experiment_id)
        if existing is not None:
            return existing
        self._append_experiment(config)
        return config

    def activate_experiment(self, experiment_id: str) -> None:
        """Transition DRAFT -> ACTIVE, set started_at."""
        experiments = self._load_all()
        for exp in experiments:
            if exp.experiment_id == experiment_id:
                if exp.status != ExperimentStatus.DRAFT:
                    raise ValueError(
                        f"Cannot activate experiment in {exp.status} status"
                    )
                exp.status = ExperimentStatus.ACTIVE
                exp.started_at = datetime.now(timezone.utc)
                self._save_all(experiments)
                return
        raise ValueError(f"Experiment {experiment_id} not found")

    def ingest_variant_data(
        self,
        experiment_id: str,
        variant_name: str,
        trades: list[dict],
    ) -> None:
        """Record trade data for a specific variant. Appends to per-experiment file."""
        data_dir = self._variant_data_dir / experiment_id
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{variant_name}.jsonl"
        with data_path.open("a") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")

    def _extract_metric_values(
        self, experiment_id: str, variant_name: str, metric: str,
    ) -> list[float]:
        """Extract metric values for a variant based on the success_metric.

        Falls back to PnL if the requested metric isn't available in the data.
        """
        data_path = self._variant_data_dir / experiment_id / f"{variant_name}.jsonl"
        if not data_path.exists():
            return []

        values = []
        for line in data_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                if metric == "pnl":
                    values.append(float(data.get("pnl", 0.0)))
                elif metric == "sharpe":
                    values.append(float(data.get("sharpe", data.get("pnl", 0.0))))
                elif metric == "win_rate":
                    values.append(float(data.get("win", 1.0 if data.get("pnl", 0) > 0 else 0.0)))
                elif metric == "profit_factor":
                    values.append(float(data.get("pnl", 0.0)))
                else:
                    values.append(float(data.get("pnl", 0.0)))
            except (json.JSONDecodeError, ValueError):
                continue
        return values

    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Compute variant metrics and run statistical test."""
        experiment = self.get_by_id(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        variant_metrics = []
        variant_pnls: dict[str, list[float]] = {}
        variant_metric_values: dict[str, list[float]] = {}

        for variant in experiment.variants:
            pnls = self._load_variant_pnls(experiment_id, variant.name)
            variant_pnls[variant.name] = pnls
            # Extract success_metric values for statistical test
            metric_vals = self._extract_metric_values(
                experiment_id, variant.name, experiment.success_metric,
            )
            variant_metric_values[variant.name] = metric_vals if metric_vals else pnls

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0.0
            avg_pnl = statistics.mean(pnls) if pnls else 0.0
            total_pnl = sum(pnls)

            # Sharpe (simplified: annualized)
            sharpe = 0.0
            if len(pnls) >= 2:
                std = statistics.stdev(pnls)
                if std > 0:
                    sharpe = (statistics.mean(pnls) / std) * math.sqrt(252)

            # Profit factor — 10.0 sentinel for all-win (no losses)
            gross_wins = sum(wins) if wins else 0.0
            gross_losses = abs(sum(losses)) if losses else 0.0
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else (10.0 if wins else 0.0)

            # Max drawdown
            max_dd = 0.0
            cumsum = 0.0
            peak = 0.0
            for p in pnls:
                cumsum += p
                if cumsum > peak:
                    peak = cumsum
                dd = (peak - cumsum) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd

            variant_metrics.append(
                VariantMetrics(
                    variant_name=variant.name,
                    trade_count=len(pnls),
                    total_pnl=round(total_pnl, 2),
                    avg_pnl=round(avg_pnl, 4),
                    win_rate=round(win_rate, 4),
                    sharpe=round(sharpe, 4),
                    profit_factor=round(profit_factor, 4),
                    max_drawdown_pct=round(max_dd * 100, 2),
                )
            )

        # Statistical test (first two variants = control vs treatment)
        p_value = None
        effect_size = None
        ci_95 = None
        winner = None
        recommendation = "extend"

        variant_names = [v.name for v in experiment.variants]
        if len(variant_names) >= 2:
            control_vals = variant_metric_values.get(variant_names[0], [])
            treatment_vals = variant_metric_values.get(variant_names[1], [])

            # Empty data → recommend extending
            if not control_vals or not treatment_vals:
                recommendation = "extend"
            elif (
                len(control_vals) >= self._min_trades
                and len(treatment_vals) >= self._min_trades
            ):
                p_value, ci_95 = self._welch_t_test(control_vals, treatment_vals)
                effect_size = self._cohens_d(control_vals, treatment_vals)

                if p_value is not None and p_value < experiment.significance_level:
                    control_mean = statistics.mean(control_vals)
                    treatment_mean = statistics.mean(treatment_vals)
                    if treatment_mean > control_mean:
                        winner = variant_names[1]
                        recommendation = "adopt_treatment"
                    else:
                        winner = variant_names[0]
                        recommendation = "keep_control"
                else:
                    recommendation = "inconclusive"

        return ExperimentResult(
            experiment_id=experiment_id,
            variant_metrics=variant_metrics,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval_95=ci_95,
            winner=winner,
            recommendation=recommendation,
        )

    def check_auto_conclusion(self, experiment_id: str) -> bool:
        """Check if experiment should auto-conclude (significance or duration)."""
        experiment = self.get_by_id(experiment_id)
        if experiment is None or experiment.status != ExperimentStatus.ACTIVE:
            return False

        # Check duration
        if experiment.started_at is not None:
            elapsed = datetime.now(timezone.utc) - experiment.started_at
            if elapsed > timedelta(days=experiment.max_duration_days):
                return True

        # Check if enough trades for significance test
        variant_names = [v.name for v in experiment.variants]
        if len(variant_names) >= 2:
            control_pnls = self._load_variant_pnls(experiment_id, variant_names[0])
            treatment_pnls = self._load_variant_pnls(
                experiment_id, variant_names[1]
            )
            if (
                len(control_pnls) >= self._min_trades
                and len(treatment_pnls) >= self._min_trades
            ):
                p_value, _ = self._welch_t_test(control_pnls, treatment_pnls)
                if p_value is not None and p_value < experiment.significance_level:
                    return True

        return False

    def conclude_experiment(
        self,
        experiment_id: str,
        result: ExperimentResult,
    ) -> None:
        """Mark experiment CONCLUDED with result. Idempotent if already concluded."""
        experiments = self._load_all()
        for exp in experiments:
            if exp.experiment_id == experiment_id:
                if exp.status == ExperimentStatus.CONCLUDED:
                    logger.info("Experiment %s already concluded — skipping", experiment_id)
                    return
                exp.status = ExperimentStatus.CONCLUDED
                exp.concluded_at = datetime.now(timezone.utc)
                self._save_all(experiments)
                self._append_result(result)
                return
        raise ValueError(f"Experiment {experiment_id} not found")

    def cancel_experiment(self, experiment_id: str) -> None:
        """Mark experiment CANCELLED."""
        experiments = self._load_all()
        for exp in experiments:
            if exp.experiment_id == experiment_id:
                exp.status = ExperimentStatus.CANCELLED
                self._save_all(experiments)
                return
        raise ValueError(f"Experiment {experiment_id} not found")

    def get_active(self) -> list[ExperimentConfig]:
        """Return all ACTIVE experiments."""
        return [
            e for e in self._load_all() if e.status == ExperimentStatus.ACTIVE
        ]

    def get_by_id(self, experiment_id: str) -> ExperimentConfig | None:
        """Retrieve experiment by ID."""
        for exp in self._load_all():
            if exp.experiment_id == experiment_id:
                return exp
        return None

    def get_results(self) -> list[ExperimentResult]:
        """Return all experiment results."""
        if not self._results_path.exists():
            return []
        results = []
        for line in self._results_path.read_text().strip().splitlines():
            try:
                results.append(
                    ExperimentResult.model_validate(json.loads(line))
                )
            except (json.JSONDecodeError, Exception):
                continue
        return results

    # ── Statistical helpers ──────────────────────────────────────────

    def _welch_t_test(
        self,
        group_a: list[float],
        group_b: list[float],
    ) -> tuple[float | None, tuple[float, float] | None]:
        """Welch's t-test for unequal variances.

        Returns (p_value, confidence_interval_95).
        """
        if len(group_a) < 2 or len(group_b) < 2:
            return None, None

        n_a, n_b = len(group_a), len(group_b)
        mean_a = statistics.mean(group_a)
        mean_b = statistics.mean(group_b)
        var_a = statistics.variance(group_a)
        var_b = statistics.variance(group_b)

        # Pooled standard error
        se_sq = var_a / n_a + var_b / n_b
        if se_sq == 0:
            # Both groups have zero variance — identical values, no meaningful test
            return 1.0, None
        se = math.sqrt(se_sq)

        t_stat = (mean_b - mean_a) / se

        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (
            n_b - 1
        )
        if denom == 0:
            return None, None
        df = num / denom

        # Approximate p-value using t-distribution
        p_value = self._t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed

        # 95% CI for difference in means
        t_crit = self._t_critical_value(df)
        margin = t_crit * se
        ci = (
            round(mean_b - mean_a - margin, 4),
            round(mean_b - mean_a + margin, 4),
        )

        return round(min(p_value, 1.0), 6), ci

    def _cohens_d(self, group_a: list[float], group_b: list[float]) -> float:
        """Cohen's d effect size."""
        if len(group_a) < 2 or len(group_b) < 2:
            return 0.0
        mean_a = statistics.mean(group_a)
        mean_b = statistics.mean(group_b)
        var_a = statistics.variance(group_a)
        var_b = statistics.variance(group_b)
        n_a, n_b = len(group_a), len(group_b)

        pooled_std = math.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )
        if pooled_std == 0:
            return 0.0
        return round((mean_b - mean_a) / pooled_std, 4)

    def _t_distribution_p_value(self, t: float, df: float) -> float:
        """Approximate p-value from t-distribution.

        Uses normal approximation for large df and regularized incomplete
        beta function for small df.
        """
        if df <= 0 or t <= 0:
            return 0.5

        # Hill's approximation
        x = df / (df + t * t)
        if df >= 20:
            # Normal approximation for large df
            z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
            # Standard normal CDF approximation
            return 0.5 * math.erfc(z / math.sqrt(2))
        else:
            # Regularized incomplete beta function approximation
            a = df / 2
            b = 0.5
            return self._incomplete_beta(x, a, b)

    def _incomplete_beta(self, x: float, a: float, b: float) -> float:
        """Regularized incomplete beta function (rough approximation)."""
        n_steps = 100
        dx = x / n_steps
        result = 0.0
        for i in range(n_steps):
            t = (i + 0.5) * dx
            if t > 0 and t < 1:
                try:
                    result += t ** (a - 1) * (1 - t) ** (b - 1) * dx
                except (OverflowError, ValueError):
                    continue

        # Normalize by Beta(a, b)
        try:
            beta_val = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        except (OverflowError, ValueError):
            return 0.5

        if beta_val > 0:
            return min(1.0, result / beta_val)
        return 0.5

    def _t_critical_value(self, df: float, alpha: float = 0.05) -> float:
        """Approximate t critical value for given df and alpha."""
        if df >= 120:
            return 1.96
        if df >= 60:
            return 2.0
        if df >= 30:
            return 2.04
        if df >= 20:
            return 2.09
        if df >= 10:
            return 2.23
        if df >= 5:
            return 2.57
        return 2.78

    # ── JSONL persistence ────────────────────────────────────────────

    def _load_variant_pnls(
        self, experiment_id: str, variant_name: str
    ) -> list[float]:
        """Load PnL values for a variant from its data file."""
        data_path = (
            self._variant_data_dir / experiment_id / f"{variant_name}.jsonl"
        )
        if not data_path.exists():
            return []
        pnls = []
        for line in data_path.read_text().strip().splitlines():
            try:
                data = json.loads(line)
                pnl = data.get("pnl", 0.0)
                pnls.append(float(pnl))
            except (json.JSONDecodeError, ValueError):
                continue
        return pnls

    def _load_all(self) -> list[ExperimentConfig]:
        """Load all experiments from JSONL."""
        if not self._path.exists():
            return []
        experiments = []
        for line in self._path.read_text().strip().splitlines():
            try:
                experiments.append(
                    ExperimentConfig.model_validate(json.loads(line))
                )
            except (json.JSONDecodeError, Exception):
                continue
        return experiments

    def _save_all(self, experiments: list[ExperimentConfig]) -> None:
        """Overwrite all experiments to JSONL."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(e.model_dump(mode="json"), default=str)
                for e in experiments
            ]
            self._path.write_text("\n".join(lines) + "\n" if lines else "")

    def _append_experiment(self, config: ExperimentConfig) -> None:
        """Append a single experiment to JSONL."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as f:
                f.write(
                    json.dumps(config.model_dump(mode="json"), default=str) + "\n"
                )

    def _append_result(self, result: ExperimentResult) -> None:
        """Append a result to results JSONL."""
        with self._lock:
            self._results_path.parent.mkdir(parents=True, exist_ok=True)
            with self._results_path.open("a") as f:
                f.write(
                    json.dumps(result.model_dump(mode="json"), default=str) + "\n"
                )
