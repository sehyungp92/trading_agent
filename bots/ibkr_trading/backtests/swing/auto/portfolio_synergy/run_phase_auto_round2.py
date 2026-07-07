"""Run swing portfolio-synergy phase-auto round 2 from the round 1 winner.

Round 1 repaired the hard-gate saturation, removed the idle overlay, added a
strict drawdown throttle, restored TPC participation, and improved ATRSS source
risk. Round 2 starts from that optimized portfolio config and searches the
remaining controlled-aggressive space: guarded dynamic-risk expansion, sleeve
risk balance, blocked-entry unlocks, broad quality discrimination, and final
combo checks. The scoring profile keeps alpha and frequency in front while the
16% hard drawdown gate prevents simply buying more heat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.round_manager import RoundManager
from backtests.swing.auto.portfolio_synergy import run_phase_auto_from_latest as round1


ROUND2_NAME = "round_2_phase_auto"
ROUND2_BASE_CONFIG = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "round_1" / "optimized_config.json"

ROUND2_SCORING_KWARGS: dict[str, Any] = {
    **round1.PHASE_SCORING_KWARGS,
    "alpha_return_target_pct": 560.0,
    "trade_count_target": 710,
    "trades_per_month_target": 11.5,
    "total_r_per_month_target": 7.5,
    "pf_quality_target": 3.90,
    "min_trades": 620,
    "min_required_strategy_trades": 25,
    "strategy_active_trade_floor": 25.0,
    "strategy_min_trade_target": 125.0,
    "max_single_strategy_static_pnl_share": 0.78,
    "score_weights": {
        "alpha_quality": 0.34,
        "frequency_quality": 0.22,
        "drawdown_quality": 0.12,
        "pf_quality": 0.10,
        "balance_quality": 0.10,
        "capture_quality": 0.08,
        "robustness_quality": 0.04,
    },
}

ROUND2_PHASES: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
    (
        "phase_1_guarded_dynamic_risk_expansion",
        [
            (
                "dd_tiers_growth_16",
                {
                    "dynamic_risk_enabled": True,
                    "drawdown_risk_tiers": (
                        (0.05, 1.00),
                        (0.08, 0.85),
                        (0.11, 0.60),
                        (0.145, 0.30),
                        (0.18, 0.00),
                    ),
                },
            ),
            (
                "dd_tiers_late_cut",
                {
                    "dynamic_risk_enabled": True,
                    "drawdown_risk_tiers": (
                        (0.06, 1.00),
                        (0.09, 0.90),
                        (0.12, 0.65),
                        (0.145, 0.35),
                        (0.16, 0.10),
                        (0.18, 0.00),
                    ),
                },
            ),
            ("daily_stop3_5", {"portfolio_daily_stop_R": 3.50}),
            (
                "daily_stop3_5_growth_tiers",
                {
                    "portfolio_daily_stop_R": 3.50,
                    "dynamic_risk_enabled": True,
                    "drawdown_risk_tiers": (
                        (0.05, 1.00),
                        (0.08, 0.85),
                        (0.11, 0.60),
                        (0.145, 0.30),
                        (0.18, 0.00),
                    ),
                },
            ),
            (
                "daily_stop3_75_late_cut",
                {
                    "portfolio_daily_stop_R": 3.75,
                    "dynamic_risk_enabled": True,
                    "drawdown_risk_tiers": (
                        (0.06, 1.00),
                        (0.09, 0.90),
                        (0.12, 0.65),
                        (0.145, 0.35),
                        (0.16, 0.10),
                        (0.18, 0.00),
                    ),
                },
            ),
        ],
    ),
    (
        "phase_2_sleeve_risk_barbell",
        [
            ("helix_unit_010", {"helix.unit_risk_pct": 0.010}),
            (
                "helix_unit_0105_add_trim",
                {
                    "helix.unit_risk_pct": 0.0105,
                    "helix_param.ADD_RISK_FRAC": 1.65,
                },
            ),
            (
                "tpc_unit_0035_heat3_25",
                {
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                    "tpc_param.all.max_risk_pct": 0.014,
                    "tpc_param.all.risk_a_plus_pct": 0.014,
                    "tpc_param.all.risk_a_pct": 0.009,
                    "tpc_param.all.risk_b_pct": 0.006,
                },
            ),
            (
                "atrss_unit_017_heat2_05",
                {
                    "atrss.unit_risk_pct": 0.017,
                    "atrss.max_heat_R": 2.05,
                },
            ),
            (
                "balanced_units_plus",
                {
                    "heat_cap_R": 5.25,
                    "portfolio_daily_stop_R": 3.50,
                    "atrss.unit_risk_pct": 0.017,
                    "atrss.max_heat_R": 2.05,
                    "helix.unit_risk_pct": 0.010,
                    "helix.max_heat_R": 1.65,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                },
            ),
            (
                "helix_tpc_plus_atrss_cap",
                {
                    "atrss.unit_risk_pct": 0.0155,
                    "helix.unit_risk_pct": 0.0105,
                    "helix.max_heat_R": 1.70,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                },
            ),
        ],
    ),
    (
        "phase_3_selective_blocked_entry_unlocks",
        [
            ("atrss_heat2_15", {"atrss.max_heat_R": 2.15}),
            (
                "atrss_smaller_unit_heat2_35",
                {
                    "atrss.unit_risk_pct": 0.0145,
                    "atrss.max_heat_R": 2.35,
                },
            ),
            ("helix_heat1_75", {"helix.max_heat_R": 1.75}),
            ("tpc_heat3_25", {"tpc.max_heat_R": 3.25}),
            ("tpc_daily_stop2_5_heat3_25", {"tpc.daily_stop_R": 2.50, "tpc.max_heat_R": 3.25}),
            ("portfolio_heat5_5", {"heat_cap_R": 5.50}),
        ],
    ),
    (
        "phase_4_broad_winner_loser_discrimination",
        [
            ("tpc_quality_threshold_plus1", {"tpc_param.all.score_a_min": 11, "tpc_param.all.score_b_min": 10}),
            (
                "tpc_quality_plus1_unit0035",
                {
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                    "tpc_param.all.score_a_min": 11,
                    "tpc_param.all.score_b_min": 10,
                },
            ),
            (
                "tpc_low_mfe_guard_plus_risk",
                {
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                    "tpc_param.all.confirmation_required": 2,
                    "tpc_param.all.score_a_plus_min": 14,
                    "tpc_param.all.second_entry_score_min": 16,
                },
            ),
            ("helix_add_risk_1_65", {"helix_param.ADD_RISK_FRAC": 1.65}),
            ("helix_add_risk_1_35", {"helix_param.ADD_RISK_FRAC": 1.35}),
            ("helix_trail_stall_3", {"helix_param.TRAIL_STALL_ONSET": 3}),
            ("atrss_source_risk_016", {"atrss_param.base_risk_pct": 0.016}),
            ("atrss_source_risk_020", {"atrss_param.base_risk_pct": 0.020}),
        ],
    ),
    (
        "phase_5_final_combo_refinement",
        [
            (
                "growth_tiers_balanced_units",
                {
                    "heat_cap_R": 5.25,
                    "portfolio_daily_stop_R": 3.50,
                    "drawdown_risk_tiers": (
                        (0.05, 1.00),
                        (0.08, 0.85),
                        (0.11, 0.60),
                        (0.145, 0.30),
                        (0.18, 0.00),
                    ),
                    "atrss.unit_risk_pct": 0.017,
                    "helix.unit_risk_pct": 0.010,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                },
            ),
            (
                "late_cut_helix_tpc_plus",
                {
                    "portfolio_daily_stop_R": 3.50,
                    "drawdown_risk_tiers": (
                        (0.06, 1.00),
                        (0.09, 0.90),
                        (0.12, 0.65),
                        (0.145, 0.35),
                        (0.16, 0.10),
                        (0.18, 0.00),
                    ),
                    "atrss.unit_risk_pct": 0.0155,
                    "helix.unit_risk_pct": 0.0105,
                    "helix.max_heat_R": 1.70,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                },
            ),
            (
                "quality_plus_balanced",
                {
                    "helix.unit_risk_pct": 0.010,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                    "tpc_param.all.score_a_min": 11,
                    "tpc_param.all.score_b_min": 10,
                },
            ),
            (
                "atrss_heat_unlock_guarded",
                {
                    "atrss.unit_risk_pct": 0.0145,
                    "atrss.max_heat_R": 2.35,
                    "helix.unit_risk_pct": 0.010,
                    "tpc.unit_risk_pct": 0.0035,
                },
            ),
            (
                "max_alpha_guarded",
                {
                    "heat_cap_R": 5.50,
                    "portfolio_daily_stop_R": 3.75,
                    "atrss.unit_risk_pct": 0.017,
                    "atrss.max_heat_R": 2.15,
                    "helix.unit_risk_pct": 0.0105,
                    "helix.max_heat_R": 1.75,
                    "tpc.unit_risk_pct": 0.0035,
                    "tpc.max_heat_R": 3.25,
                },
            ),
        ],
    ),
]


def _install_round2_design() -> None:
    round1.PHASES = ROUND2_PHASES
    round1.PHASE_SCORING_KWARGS = ROUND2_SCORING_KWARGS
    round1.ROUND_NAME = ROUND2_NAME
    round1.PortfolioSynergyPhasePlugin.num_phases = len(ROUND2_PHASES)


def _load_initial_mutations(base_config: str | Path | None) -> tuple[dict[str, Any], str]:
    path = Path(base_config) if base_config else ROUND2_BASE_CONFIG
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing round 2 base config: {path}")
    return json.loads(path.read_text()), str(path)


def build_phase_runner(args: argparse.Namespace) -> PhaseRunner:
    _install_round2_design()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    initial_mutations, base_source = _load_initial_mutations(getattr(args, "base_config", None))
    plugin = round1.PortfolioSynergyPhasePlugin(
        data_dir,
        initial_equity=float(args.equity),
        max_workers=int(args.max_workers),
        initial_mutations=initial_mutations,
        base_source=base_source,
    )
    manager = RoundManager("swing", "portfolio_synergy")
    round_num, round_dir = manager.resolve_round(
        getattr(args, "round", 2),
        for_write=True,
        expected_phases=plugin.num_phases,
    )
    if round_num != 2:
        raise ValueError("Swing portfolio synergy round 2 runner must write round_2.")
    return PhaseRunner(
        plugin=plugin,
        output_dir=round_dir,
        round_name="Swing portfolio synergy round 2 phase auto",
        max_rounds=getattr(args, "max_rounds", 12),
        min_delta=getattr(args, "min_delta", 0.001),
        max_retries=getattr(args, "max_retries", 2),
        round_manager=manager,
        round_num=round_num,
    )


def run_phase_auto(args: argparse.Namespace) -> Path:
    runner = build_phase_runner(args)
    state = runner.run_all_phases(start_phase=getattr(args, "start_phase", None))
    print(f"\nPhase-auto synergy round 2 complete: {runner.output_dir}", flush=True)
    print(f"Completed phases: {state.completed_phases}", flush=True)
    return runner.output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional optimized_config.json to continue from; defaults to portfolio_synergy round_1.",
    )
    parser.add_argument("--data-dir", default=ROOT / "backtests" / "swing" / "data" / "raw")
    parser.add_argument("--equity", type=float, default=round1.STARTING_EQUITY)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument("--start-phase", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--output-root", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.output_root:
        print("Ignoring --output-root; standard phased auto output uses backtests/output/swing/portfolio_synergy/round_2.", flush=True)
    run_phase_auto(args)


if __name__ == "__main__":
    main()
