from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.shared.auto.provenance import build_phase_auto_provenance
from backtests.shared.auto.round_manager import RoundManager
from backtests.shared.auto.types import Experiment, GateCriterion
from backtests.stock.auto.iaric.phase_candidates import V4R1_BASE_MUTATIONS
from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin
from strategies.stock.iaric.config import StrategySettings


DATA_DIR = Path("backtests/stock/data/raw")
START_DATE = "2024-01-01"
END_DATE = "2026-03-01"
INITIAL_EQUITY = 10_000.0
ROUND_NAME = "live_aligned_ablation_perturbation"
NUM_PHASES = 2

_LEGACY_PARAM_ALIASES = {
    "max_per_sector": "param_overrides.max_positions_per_sector",
    "max_positions_tier_a": "param_overrides.max_positions_tier_a",
    "max_positions_tier_b": "param_overrides.max_positions_tier_b",
    "sector_risk_cap_pct": "param_overrides.sector_risk_cap_pct",
}

_PERTURB_FIELDS = {
    "max_positions_per_sector",
    "pb_atr_stop_mult",
    "pb_cdd_max",
    "pb_daily_rescue_min_score",
    "pb_daily_signal_min_score",
    "pb_delayed_confirm_after_bar",
    "pb_delayed_confirm_score_min",
    "pb_max_hold_days",
    "pb_max_positions",
    "pb_open_scored_max_hold_days",
    "pb_v2_flatten_loss_r",
    "pb_v2_flow_grace_days",
    "pb_v2_gap_max_pct",
    "pb_v2_mfe_stage1_trigger",
    "pb_v2_mfe_stage2_trigger",
    "pb_v2_mfe_stage3_trail_atr",
    "pb_v2_mfe_stage3_trigger",
    "pb_v2_open_scored_max_slots",
    "pb_v2_open_scored_min_score",
    "pb_v2_partial_profit_remainder_stop_r",
    "pb_v2_partial_profit_trigger_r",
    "pb_v2_signal_floor",
    "pb_v2_stale_bars",
    "pb_v2_stale_mfe_thresh",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", type=Path, default=None)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--profile", choices=["mainline", "aggressive"], default="mainline")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--start-phase", type=int, choices=range(1, NUM_PHASES + 1), default=None)
    return parser.parse_args()


def _load_mutations(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        if isinstance(payload.get("mutations"), dict):
            return dict(payload["mutations"])
        if isinstance(payload.get("cumulative_mutations"), dict):
            return dict(payload["cumulative_mutations"])
        return dict(payload)
    raise TypeError(f"Unexpected optimized config payload in {path}")


def _normalize_mutations(mutations: dict[str, Any]) -> dict[str, Any]:
    settings_fields = set(StrategySettings.__dataclass_fields__)
    normalized: dict[str, Any] = {}
    for key, value in mutations.items():
        mapped_key = _LEGACY_PARAM_ALIASES.get(key, key)
        if mapped_key.startswith("param_overrides."):
            field = mapped_key.split(".", 1)[1]
            if field not in settings_fields:
                raise ValueError(f"Mutation {mapped_key!r} does not target StrategySettings.")
        normalized[mapped_key] = value
    return normalized


def _mutation_count(path: Path) -> int:
    try:
        return len(_load_mutations(path))
    except Exception:
        return -1


def _find_default_baseline() -> Path:
    root = Path("backtests/output/stock/iaric/archived_rounds")
    candidates = list(root.glob("*/round_3/optimized_config.json"))
    if not candidates:
        raise FileNotFoundError(f"No archived round_3 optimized config found under {root}.")
    return max(candidates, key=lambda path: (_mutation_count(path), path.stat().st_mtime))


def _candidate_name(prefix: str, field: str, suffix: str = "") -> str:
    safe = field.removeprefix("pb_").replace(".", "_").replace("-", "_")
    return f"{prefix}_{safe}{suffix}"[:96]


def _ablation_candidates(
    baseline: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    reference = reference or {}
    defaults = StrategySettings()
    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, value in sorted(baseline.items()):
        if not key.startswith("param_overrides."):
            continue
        field = key.split(".", 1)[1]
        ablated_value = reference.get(key, getattr(defaults, field, None))
        if value == ablated_value:
            continue
        candidates.append((_candidate_name("ablate", field), {key: ablated_value}))
    return candidates


def _bounded(field: str, value: float) -> float:
    if field == "pb_v2_partial_profit_trigger_r":
        return max(0.05, value)
    if field == "pb_v2_partial_profit_remainder_stop_r":
        return max(0.0, value)
    if "score" in field or "floor" in field or "rank_pct" in field:
        return max(0.0, min(100.0, value))
    if field.endswith("_pct") or "_pct_" in field:
        return max(0.0, min(100.0, value))
    return value


def _numeric_step(field: str, value: int | float) -> int | float:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(1, round(abs(value) * 0.15))
    if "score" in field or "floor" in field:
        return 2.0
    if field.endswith("_bars") or field.endswith("_days") or field.endswith("_slots") or field.endswith("_positions"):
        return 1.0
    if field.endswith("_r") or "_mfe_" in field or field.endswith("_mult") or field.endswith("_atr"):
        return 0.10
    return max(abs(float(value)) * 0.10, 0.05)


def _perturbation_candidates(baseline: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, value in sorted(baseline.items()):
        if not key.startswith("param_overrides."):
            continue
        field = key.split(".", 1)[1]
        if field not in _PERTURB_FIELDS:
            continue
        step = _numeric_step(field, value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
        if not step:
            continue
        for label, candidate_value in (
            ("_down", float(value) - float(step)),
            ("_up", float(value) + float(step)),
        ):
            adjusted = _bounded(field, candidate_value)
            if field == "pb_v2_partial_profit_trigger_r":
                remainder = float(baseline.get("param_overrides.pb_v2_partial_profit_remainder_stop_r", 0.0) or 0.0)
                if adjusted <= 0 or adjusted < min(remainder, float(value)):
                    continue
            if isinstance(value, int):
                adjusted = int(round(adjusted))
                if adjusted == value:
                    continue
            candidates.append((_candidate_name("perturb", field, label), {key: adjusted}))
    if baseline.get("param_overrides.pb_open_scored_fill_timing") == "next_5m_open":
        candidates.append(
            (
                "perturb_open_scored_fill_same_bar_close",
                {"param_overrides.pb_open_scored_fill_timing": "same_bar_close"},
            )
        )
    return candidates


class LiveAlignedAblationPlugin(IARICPullbackPlugin):
    def __init__(self, *args, baseline_source: Path, **kwargs) -> None:
        super().__init__(*args, round_name=ROUND_NAME, num_phases=NUM_PHASES, **kwargs)
        self.baseline_source = baseline_source
        self.ablation_reference = _normalize_mutations(dict(V4R1_BASE_MUTATIONS))
        self.ultimate_targets = {
            "profit_factor": 2.0,
            "expected_total_r": 40.0,
            "total_trades": 500.0,
            "sharpe": 1.0,
            "max_drawdown_pct": 0.08,
        }
        self._phase_scoring_weights = {
            1: {
                "expected_total_r": 0.28,
                "profit_factor": 0.22,
                "sharpe": 0.18,
                "inv_dd": 0.16,
                "total_trades": 0.10,
                "avg_r": 0.06,
            },
            2: {
                "expected_total_r": 0.30,
                "profit_factor": 0.20,
                "sharpe": 0.18,
                "inv_dd": 0.14,
                "total_trades": 0.12,
                "avg_r": 0.06,
            },
        }
        self._phase_hard_rejects = {
            1: {"min_trades": 350, "min_pf": 1.15, "max_dd_pct": 0.08, "min_expected_total_r": 15.0},
            2: {"min_trades": 350, "min_pf": 1.15, "max_dd_pct": 0.08, "min_expected_total_r": 15.0},
        }

    def build_provenance(self):
        if self._provenance is None:
            repo_root = Path(__file__).resolve().parents[4]
            self._provenance = build_phase_auto_provenance(
                self.name,
                repo_root=repo_root,
                code_dirs=(repo_root / "backtests/stock/auto/iaric",),
                code_paths=(
                    Path(__file__).resolve(),
                    repo_root / "backtests/stock/engine/iaric_pullback_engine.py",
                    repo_root / "backtests/stock/config_iaric.py",
                    repo_root / "backtests/stock/auto/config_mutator.py",
                    repo_root / "backtests/stock/data/replay_cache.py",
                    repo_root / "strategies/stock/iaric/core/logic.py",
                    repo_root / "strategies/stock/iaric/artifact_store.py",
                ),
                source_artifacts={"baseline_config": self.baseline_source},
                data_dir=self.data_dir,
                selection_context={
                    "round_name": ROUND_NAME,
                    "profile": self.profile,
                    "start_date": self.start_date,
                    "end_date": self.end_date,
                    "initial_equity": self.initial_equity,
                    "num_phases": self.num_phases,
                    "phase_scoring_weights": self._phase_scoring_weights,
                    "phase_hard_rejects": self._phase_hard_rejects,
                    "ultimate_targets": self.ultimate_targets,
                    "baseline_policy": "archived_round_3_normalized",
                },
            )
        return self._provenance

    def get_phase_spec(self, phase: int, state) -> PhaseSpec:
        if phase == 1:
            focus = "Ablate archived cumulative mutations on live-aligned engine"
            candidates = _ablation_candidates(dict(self.initial_mutations or {}), self.ablation_reference)
        elif phase == 2:
            focus = "Perturb surviving live-aligned cumulative config"
            candidates = _perturbation_candidates(dict(state.cumulative_mutations or self.initial_mutations or {}))
        else:
            raise ValueError(f"Unsupported phase {phase} for {ROUND_NAME}")

        return PhaseSpec(
            focus=focus,
            candidates=[Experiment(name=name, mutations=mutations) for name, mutations in candidates],
            gate_criteria_fn=lambda metrics: self._relative_gate_criteria(phase, metrics, state),
            scoring_weights=self._phase_scoring_weights[phase],
            hard_rejects=self._phase_hard_rejects[phase],
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=["expected_total_r", "profit_factor", "sharpe", "inv_dd", "total_trades"],
                min_effective_score_delta_pct=0.0,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                build_extra_analysis_fn=self.build_analysis_extra,
                format_extra_analysis_fn=self.format_analysis_extra,
            ),
            max_rounds=1 if phase == 2 else len(candidates),
            prune_threshold=0.0,
            reject_streak_limit=None,
        )

    def _relative_gate_criteria(self, phase: int, metrics: dict[str, float], state) -> list[GateCriterion]:
        reference = self._reference_metrics(phase, state)
        expected = float(metrics.get("expected_total_r", 0.0))
        if expected == 0.0:
            expected = float(metrics.get("avg_r", 0.0)) * float(metrics.get("total_trades", 0.0))
        ref_expected = float(reference.get("expected_total_r", 0.0))
        if ref_expected == 0.0:
            ref_expected = float(reference.get("avg_r", 0.0)) * float(reference.get("total_trades", 0.0))
        ref_pf = float(reference.get("profit_factor", 0.0))
        ref_dd = float(reference.get("max_drawdown_pct", 0.0))

        def min_gate(name: str, threshold: float, actual: float) -> GateCriterion:
            return GateCriterion(name, threshold, actual, actual >= threshold)

        def max_gate(name: str, threshold: float, actual: float) -> GateCriterion:
            return GateCriterion(name, threshold, actual, actual <= threshold)

        return [
            min_gate("total_trades_floor", max(350.0, float(reference.get("total_trades", 0.0)) * 0.85), float(metrics.get("total_trades", 0.0))),
            min_gate("profit_factor_reference_floor", max(1.15, ref_pf * 0.98), float(metrics.get("profit_factor", 0.0))),
            min_gate("expected_total_r_reference_floor", max(15.0, ref_expected * 0.98), expected),
            max_gate("drawdown_reference_ceiling", min(0.08, max(ref_dd * 1.15, ref_dd + 0.005)), float(metrics.get("max_drawdown_pct", 0.0))),
        ]


def main() -> None:
    args = _parse_args()
    baseline_config = (args.baseline_config or _find_default_baseline()).resolve()
    baseline_mutations = _normalize_mutations(_load_mutations(baseline_config))

    if args.output_dir:
        output_dir = args.output_dir
        round_manager = None
        round_num = None
    else:
        round_manager = RoundManager("stock", "iaric")
        round_num, output_dir = round_manager.resolve_round(
            args.round,
            for_write=True,
            expected_phases=NUM_PHASES,
        )

    plugin = LiveAlignedAblationPlugin(
        DATA_DIR,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=INITIAL_EQUITY,
        max_workers=args.max_workers,
        profile=args.profile,
        baseline_source=baseline_config,
    )
    plugin.initial_mutations = baseline_mutations
    plugin.initial_mutations_source = str(baseline_config)

    print("=" * 72)
    print("IARIC Pullback -- Live-Aligned Ablation + Perturbation")
    print("=" * 72)
    print(f"Output dir: {output_dir}")
    print(f"Baseline: {baseline_config}")
    print(f"Date range: {args.start_date} -> {args.end_date}")
    print(f"Profile: {args.profile}")
    print(f"Baseline mutations: {len(baseline_mutations)}")
    ablation_reference = _normalize_mutations(dict(V4R1_BASE_MUTATIONS))
    print(f"Phase 1: {len(_ablation_candidates(baseline_mutations, ablation_reference)):>3} candidates | ablation")
    print(f"Phase 2: {len(_perturbation_candidates(baseline_mutations)):>3} candidates | perturbation")
    print(flush=True)

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name=ROUND_NAME,
        max_rounds=args.max_rounds,
        min_delta=args.min_delta,
        max_retries=0,
        max_diagnostic_retries=0,
        round_manager=round_manager,
        round_num=round_num,
    )
    runner.run_all_phases(start_phase=args.start_phase)


if __name__ == "__main__":
    main()
