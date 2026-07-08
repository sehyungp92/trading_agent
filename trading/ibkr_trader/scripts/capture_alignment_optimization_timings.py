from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.shared.auto.phase_state import PhaseState


@dataclass(frozen=True)
class BenchmarkTarget:
    name: str
    family: str
    phase: int
    sample_size: int
    factory: Callable[[], Any]


@dataclass
class BenchmarkResult:
    name: str
    family: str
    phase: int
    sample_size: int
    candidate_names: list[str]
    cold_seconds: float
    warm_seconds: float
    warm_speedup_ratio: float | None


def _build_targets() -> dict[str, BenchmarkTarget]:
    from backtests.momentum.auto.downturn.plugin import DownturnPlugin
    from backtests.momentum.auto.nqdtc.plugin import NQDTCPlugin
    from backtests.momentum.auto.vdubus.plugin import VdubusPlugin
    from backtests.stock.auto.alcb.plugin import ALCBP16Plugin
    from backtests.stock.auto.iaric.plugin import IARICPullbackPlugin
    from backtests.swing.auto.atrss.plugin import ATRSSPlugin

    swing_data = ROOT / "backtests" / "swing" / "data" / "raw"
    momentum_data = ROOT / "backtests" / "momentum" / "data" / "raw"
    stock_data = ROOT / "backtests" / "stock" / "data" / "raw"

    return {
        "swing_atrss": BenchmarkTarget(
            name="swing_atrss",
            family="swing",
            phase=1,
            sample_size=2,
            factory=lambda: ATRSSPlugin(swing_data, max_workers=1),
        ),
        "momentum_downturn": BenchmarkTarget(
            name="momentum_downturn",
            family="momentum",
            phase=1,
            sample_size=2,
            factory=lambda: DownturnPlugin(momentum_data, max_workers=1),
        ),
        "momentum_nqdtc": BenchmarkTarget(
            name="momentum_nqdtc",
            family="momentum",
            phase=1,
            sample_size=2,
            factory=lambda: NQDTCPlugin(momentum_data, max_workers=1),
        ),
        "momentum_vdubus": BenchmarkTarget(
            name="momentum_vdubus",
            family="momentum",
            phase=1,
            sample_size=2,
            factory=lambda: VdubusPlugin(momentum_data, max_workers=1),
        ),
        "stock_alcb": BenchmarkTarget(
            name="stock_alcb",
            family="stock",
            phase=1,
            sample_size=2,
            factory=lambda: ALCBP16Plugin(stock_data, max_workers=1),
        ),
        "stock_iaric": BenchmarkTarget(
            name="stock_iaric",
            family="stock",
            phase=1,
            sample_size=2,
            factory=lambda: IARICPullbackPlugin(stock_data, max_workers=1, round_name="v4r1"),
        ),
    }


def _time_phase_sample(plugin: Any, phase: int, sample_size: int) -> tuple[float, list[str]]:
    mutations = dict(getattr(plugin, "initial_mutations", None) or {})
    state = PhaseState(cumulative_mutations=dict(mutations))

    started = time.perf_counter()
    spec = plugin.get_phase_spec(phase, state)
    candidates = list(spec.candidates[:sample_size])
    if not candidates:
        raise RuntimeError(f"{plugin.__class__.__name__} phase {phase} produced no candidates")

    evaluator = plugin.create_evaluate_batch(
        phase,
        mutations,
        scoring_weights=spec.scoring_weights,
        hard_rejects=spec.hard_rejects,
    )
    try:
        results = evaluator(candidates, mutations)
    finally:
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()
    for result in results:
        reject_reason = getattr(result, "reject_reason", "") or ""
        if reject_reason.startswith("error:"):
            raise RuntimeError(f"{plugin.__class__.__name__} benchmark failed: {reject_reason}")
    elapsed = time.perf_counter() - started
    return elapsed, [candidate.name for candidate in candidates]


def _run_target(target: BenchmarkTarget) -> BenchmarkResult:
    plugin = target.factory()
    try:
        cold_seconds, candidate_names = _time_phase_sample(plugin, target.phase, target.sample_size)
        warm_seconds, _ = _time_phase_sample(plugin, target.phase, target.sample_size)
    finally:
        close_pool = getattr(plugin, "close_pool", None)
        if callable(close_pool):
            close_pool()

    warm_speedup_ratio = None
    if warm_seconds > 0:
        warm_speedup_ratio = cold_seconds / warm_seconds
    return BenchmarkResult(
        name=target.name,
        family=target.family,
        phase=target.phase,
        sample_size=target.sample_size,
        candidate_names=candidate_names,
        cold_seconds=round(cold_seconds, 3),
        warm_seconds=round(warm_seconds, 3),
        warm_speedup_ratio=round(warm_speedup_ratio, 3) if warm_speedup_ratio is not None else None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture cold-cache and warm-cache timings for representative alignment optimization paths.",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Optional subset of benchmark targets to run. Defaults to the six completion-gate targets plus Downturn.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    targets = _build_targets()
    selected_names = args.targets or [
        "swing_atrss",
        "momentum_downturn",
        "momentum_nqdtc",
        "momentum_vdubus",
        "stock_iaric",
        "stock_alcb",
    ]

    unknown = [name for name in selected_names if name not in targets]
    if unknown:
        raise SystemExit(f"Unknown benchmark targets: {', '.join(unknown)}")

    results = [asdict(_run_target(targets[name])) for name in selected_names]
    json_kwargs = {"indent": 2, "sort_keys": True} if args.pretty else {}
    payload = json.dumps(results, **json_kwargs)
    print(payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + ("\n" if not payload.endswith("\n") else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
