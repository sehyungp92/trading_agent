from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .round_manager import RoundManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "backtests" / "output"

_TOP_LEVEL_METRIC_KEYS = {
    "avg_r",
    "calmar",
    "calmar_r",
    "calmar_ratio",
    "max_dd_pct",
    "max_drawdown_pct",
    "net_profit",
    "net_return_pct",
    "profit_factor",
    "return_pct",
    "sharpe",
    "sharpe_ratio",
    "sortino",
    "total_pnl",
    "total_trades",
    "trades",
    "win_rate",
}


@dataclass(frozen=True)
class StrategyBootstrapSpec:
    family: str
    strategy: str
    diagnostics: Path
    phase_state: Path | None = None
    summary_json: Path | None = None
    artifacts_dir: Path | None = None
    mutations_override: dict[str, Any] | None = None


def _path(relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else PROJECT_ROOT / path


STRATEGIES: list[StrategyBootstrapSpec] = [
    StrategyBootstrapSpec(
        family="momentum",
        strategy="downturn",
        diagnostics=_path("backtests/momentum/auto/downturn/output/r8_full_diagnostics.txt"),
        phase_state=_path("backtests/momentum/auto/downturn/output/phase_state.json"),
        artifacts_dir=_path("backtests/momentum/auto/downturn/output"),
    ),
    StrategyBootstrapSpec(
        family="momentum",
        strategy="nqdtc",
        diagnostics=_path("backtests/momentum/auto/nqdtc/output/post_audit/post_audit_full_diagnostics.txt"),
        phase_state=_path("backtests/momentum/auto/nqdtc/output/post_audit/phase_state.json"),
        artifacts_dir=_path("backtests/momentum/auto/nqdtc/output/post_audit"),
    ),
    StrategyBootstrapSpec(
        family="momentum",
        strategy="vdubus",
        diagnostics=_path("backtests/momentum/auto/vdubus/output/phase_seed_full_diagnostics.txt"),
        phase_state=_path("backtests/momentum/auto/vdubus/output/phase_state.json"),
        artifacts_dir=_path("backtests/momentum/auto/vdubus/output"),
    ),
    StrategyBootstrapSpec(
        family="momentum",
        strategy="helix",
        diagnostics=_path("backtests/momentum/auto/output/helix_optimized_seed_full_diagnostics.txt"),
        summary_json=_path("backtests/momentum/auto/output/helix_optimized_seed_summary.json"),
        mutations_override={
            "flags.use_momentum_stall": True,
            "flags.use_drawdown_throttle": False,
            "flags.vol_50_80_sizing_mult": 0.85,
        },
    ),
    StrategyBootstrapSpec(
        family="swing",
        strategy="helix",
        diagnostics=_path("backtests/swing/auto/output/helix_exit_alpha_optimized_baseline_full_diagnostics.txt"),
        phase_state=_path("backtests/swing/auto/helix/exit_alpha/output/phase_state.json"),
        summary_json=_path("backtests/swing/auto/output/helix_exit_alpha_optimized_baseline_summary.json"),
        artifacts_dir=_path("backtests/swing/auto/helix/exit_alpha/output"),
    ),
    StrategyBootstrapSpec(
        family="swing",
        strategy="atrss",
        diagnostics=_path("backtests/swing/auto/atrss/output/r9_phase1_full_diagnostics.txt"),
        phase_state=_path("backtests/swing/auto/atrss/output/phase_state.json"),
        artifacts_dir=_path("backtests/swing/auto/atrss/output"),
    ),
    StrategyBootstrapSpec(
        family="stock",
        strategy="iaric",
        diagnostics=_path("backtests/stock/auto/iaric/output_v4r1/phase_4_optimal_starting_baseline_full_diagnostics.txt"),
        phase_state=_path("backtests/stock/auto/iaric/output_v4r1/phase_state.json"),
        summary_json=_path("backtests/stock/auto/iaric/output_v4r1/phase_4_optimal_starting_baseline_summary.json"),
        artifacts_dir=_path("backtests/stock/auto/iaric/output_v4r1"),
    ),
    StrategyBootstrapSpec(
        family="stock",
        strategy="alcb",
        diagnostics=_path("backtests/stock/auto/alcb/output_targeted_entry_repair_v2/round_final_diagnostics.txt"),
        phase_state=_path("backtests/stock/auto/alcb/output_targeted_entry_repair_v2/phase_state.json"),
        artifacts_dir=_path("backtests/stock/auto/alcb/output_targeted_entry_repair_v2"),
    ),
]


def _cleanup_malformed_output_root() -> None:
    momentum_dir = OUTPUT_ROOT / "momentum"
    swing_dir = momentum_dir / "swing"
    stock_dir = swing_dir / "stock"
    if not stock_dir.exists():
        return

    stock_entries = list(stock_dir.iterdir())
    swing_entries = list(swing_dir.iterdir())
    momentum_entries = list(momentum_dir.iterdir())

    if stock_entries:
        raise RuntimeError(f"Refusing to remove non-empty malformed directory: {stock_dir}")
    if any(path != stock_dir for path in swing_entries):
        raise RuntimeError(f"Refusing to remove malformed directory with extra contents: {swing_dir}")
    if any(path != swing_dir for path in momentum_entries):
        raise RuntimeError(f"Refusing to remove malformed directory with extra contents: {momentum_dir}")

    stock_dir.rmdir()
    swing_dir.rmdir()
    momentum_dir.rmdir()


def _load_phase_state(spec: StrategyBootstrapSpec) -> dict[str, Any] | None:
    if spec.phase_state is None:
        return None
    return json.loads(spec.phase_state.read_text(encoding="utf-8"))


def _load_mutations(spec: StrategyBootstrapSpec, phase_state: dict[str, Any] | None) -> dict[str, Any]:
    if spec.mutations_override is not None:
        return dict(spec.mutations_override)
    if phase_state is None:
        raise ValueError(f"{spec.family}/{spec.strategy} needs either phase_state or mutations_override.")
    mutations = phase_state.get("cumulative_mutations")
    if not isinstance(mutations, dict):
        raise ValueError(f"Could not load cumulative_mutations for {spec.family}/{spec.strategy}.")
    return dict(mutations)


def _extract_phase_state_metrics(phase_state: dict[str, Any] | None) -> tuple[dict[str, Any], list[int]]:
    if not phase_state:
        return {}, []

    completed_raw = phase_state.get("completed_phases", [])
    completed_phases = [int(phase) for phase in completed_raw]
    phase_results = phase_state.get("phase_results", {}) or {}

    candidate_phase_ids = list(reversed(sorted(completed_phases)))
    if not candidate_phase_ids:
        candidate_phase_ids = sorted(int(key) for key in phase_results)

    for phase in candidate_phase_ids:
        phase_result = phase_results.get(str(phase), phase_results.get(phase, {}))
        if isinstance(phase_result, dict):
            metrics = phase_result.get("final_metrics")
            if isinstance(metrics, dict):
                return dict(metrics), completed_phases
    return {}, completed_phases


def _extract_summary_metrics(summary_path: Path | None) -> dict[str, Any]:
    if summary_path is None or not summary_path.exists():
        return {}

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("final_metrics", "metrics", "live_metrics", "headline_metrics"):
        nested = data.get(key)
        if isinstance(nested, dict):
            return dict(nested)

    metrics = {key: value for key, value in data.items() if key in _TOP_LEVEL_METRIC_KEYS}
    if "net_profit" not in metrics and "total_pnl" in metrics:
        metrics["net_profit"] = metrics["total_pnl"]
    if "total_trades" not in metrics and "trades" in metrics:
        metrics["total_trades"] = metrics["trades"]
    return metrics


def _parse_metrics_from_diagnostics(diagnostics_path: Path) -> dict[str, Any]:
    text = diagnostics_path.read_text(encoding="utf-8")
    patterns = {
        "total_trades": [
            r"(?im)^\s*(?:Total trades|Total Trades|Trades)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "win_rate": [
            r"(?im)^\s*Win rate\s*:\s*([0-9]+(?:\.[0-9]+)?)%",
            r"(?im)^\s*Win Rate\s*:\s*([0-9]+(?:\.[0-9]+)?)%",
        ],
        "profit_factor": [
            r"(?im)^\s*Profit factor\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?im)^\s*Profit Factor\s*:\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?im)\bPF\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ],
        "max_drawdown_pct": [
            r"(?im)^\s*Max drawdown\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)%",
            r"(?im)^\s*Max DD\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)%",
        ],
        "net_return_pct": [
            r"(?im)^\s*Net return\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)%",
            r"(?im)^\s*Return\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)%",
        ],
        "sharpe_ratio": [
            r"(?im)^\s*Sharpe ratio\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
            r"(?im)^\s*Sharpe\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
        ],
        "calmar_ratio": [
            r"(?im)^\s*Calmar ratio\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
            r"(?im)^\s*Calmar\s*:\s*([+\-]?[0-9]+(?:\.[0-9]+)?)",
        ],
    }

    metrics: dict[str, Any] = {}
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text)
            if match:
                raw = float(match.group(1))
                metrics[key] = int(raw) if key == "total_trades" and raw.is_integer() else raw
                break
    return metrics


def _collect_metrics(spec: StrategyBootstrapSpec, phase_state: dict[str, Any] | None) -> tuple[dict[str, Any], list[int]]:
    phase_metrics, completed_phases = _extract_phase_state_metrics(phase_state)
    if phase_metrics:
        return phase_metrics, completed_phases

    summary_metrics = _extract_summary_metrics(spec.summary_json)
    if summary_metrics:
        return summary_metrics, completed_phases

    return _parse_metrics_from_diagnostics(spec.diagnostics), completed_phases


def bootstrap_round_1() -> list[Path]:
    _cleanup_malformed_output_root()

    created_round_dirs: list[Path] = []
    for spec in STRATEGIES:
        if not spec.diagnostics.exists():
            raise FileNotFoundError(f"Missing diagnostics source for {spec.family}/{spec.strategy}: {spec.diagnostics}")
        if spec.phase_state is not None and not spec.phase_state.exists():
            raise FileNotFoundError(f"Missing phase_state source for {spec.family}/{spec.strategy}: {spec.phase_state}")

        phase_state = _load_phase_state(spec)
        mutations = _load_mutations(spec, phase_state)
        metrics, completed_phases = _collect_metrics(spec, phase_state)

        round_dir = RoundManager.bootstrap_round_1(
            spec.family,
            spec.strategy,
            mutations,
            spec.diagnostics,
            spec.phase_state,
            base_dir=OUTPUT_ROOT,
            diagnostics_summary_src_path=spec.summary_json if spec.summary_json and spec.summary_json.exists() else None,
            final_metrics=metrics,
            completed_phases=completed_phases,
            artifacts_dir=spec.artifacts_dir,
        )
        _verify_bootstrap(spec, round_dir, mutations)
        created_round_dirs.append(round_dir)
    return created_round_dirs


def _verify_bootstrap(spec: StrategyBootstrapSpec, round_dir: Path, expected_mutations: dict[str, Any]) -> None:
    diagnostics_copy = round_dir / "round_final_diagnostics.txt"
    if diagnostics_copy.read_bytes() != spec.diagnostics.read_bytes():
        raise RuntimeError(f"Diagnostics copy mismatch for {spec.family}/{spec.strategy}")

    optimized_config_path = round_dir / "optimized_config.json"
    actual_mutations = json.loads(optimized_config_path.read_text(encoding="utf-8"))
    if actual_mutations != expected_mutations:
        raise RuntimeError(f"Optimized config mismatch for {spec.family}/{spec.strategy}")

    if spec.phase_state is not None:
        copied_phase_state = round_dir / "phase_state.json"
        if copied_phase_state.read_bytes() != spec.phase_state.read_bytes():
            raise RuntimeError(f"Phase state copy mismatch for {spec.family}/{spec.strategy}")

    if spec.summary_json is not None and spec.summary_json.exists():
        copied_summary = round_dir / "diagnostics_summary.json"
        if copied_summary.read_bytes() != spec.summary_json.read_bytes():
            raise RuntimeError(f"Diagnostics summary copy mismatch for {spec.family}/{spec.strategy}")


def main() -> None:
    created = bootstrap_round_1()
    print("Bootstrapped round_1 for:")
    for round_dir in created:
        print(f"  {round_dir}")


if __name__ == "__main__":
    main()
