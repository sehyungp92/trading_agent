"""Smoke-test more balanced round 2 swing portfolio allocations.

This is not a full phase-auto round. It starts from the round 2 portfolio
winner and evaluates broad allocation variants that try to reduce ATRSS static
PnL concentration while preserving the round 2 return, frequency, and DD
profile.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from backtests.swing.auto.portfolio_synergy.run_latest_two_rounds import _evaluate, _json_default
from backtests.swing.config_unified import UnifiedBacktestConfig
from backtests.swing.engine.unified_portfolio_engine import load_unified_data


STARTING_EQUITY = 50_000.0
BASE_CONFIG = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "round_2" / "optimized_config.json"
DEFAULT_DATA_DIR = ROOT / "backtests" / "swing" / "data" / "raw"

MATERIAL_RETURN_FLOOR_FRAC = 0.95
MATERIAL_DD_BUFFER_PCT = 1.00

SMOKE_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "mild_rebalance_helix_tpc",
        {
            "atrss.unit_risk_pct": 0.0165,
            "helix.unit_risk_pct": 0.0115,
            "helix.max_heat_R": 1.85,
            "tpc.unit_risk_pct": 0.0040,
            "tpc.max_heat_R": 3.50,
            "tpc_param.all.max_risk_pct": 0.016,
            "tpc_param.all.risk_a_plus_pct": 0.016,
            "tpc_param.all.risk_a_pct": 0.010,
            "tpc_param.all.risk_b_pct": 0.007,
        },
    ),
    (
        "balanced_soft_shift",
        {
            "atrss.unit_risk_pct": 0.0160,
            "atrss.max_heat_R": 2.00,
            "helix.unit_risk_pct": 0.0120,
            "helix.max_heat_R": 1.90,
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.018,
            "tpc_param.all.risk_a_plus_pct": 0.018,
            "tpc_param.all.risk_a_pct": 0.011,
            "tpc_param.all.risk_b_pct": 0.008,
        },
    ),
    (
        "helix_tpc_plus_atrss_cap",
        {
            "atrss.unit_risk_pct": 0.0155,
            "atrss.max_heat_R": 1.85,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "helix_param.ADD_RISK_FRAC": 1.65,
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.018,
            "tpc_param.all.risk_a_plus_pct": 0.018,
            "tpc_param.all.risk_a_pct": 0.011,
            "tpc_param.all.risk_b_pct": 0.008,
        },
    ),
    (
        "risk_parity_soft",
        {
            "atrss.unit_risk_pct": 0.0150,
            "atrss.max_heat_R": 1.85,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "helix_plus_only",
        {
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "helix_param.ADD_RISK_FRAC": 1.65,
        },
    ),
    (
        "tpc_plus_only",
        {
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "tpc_quality_plus_risk",
        {
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.018,
            "tpc_param.all.risk_a_plus_pct": 0.018,
            "tpc_param.all.risk_a_pct": 0.011,
            "tpc_param.all.risk_b_pct": 0.008,
            "tpc_param.all.score_a_min": 11,
            "tpc_param.all.score_b_min": 10,
        },
    ),
    (
        "atrss_trim_quality_hold",
        {
            "atrss.unit_risk_pct": 0.0150,
            "atrss.max_heat_R": 1.85,
            "atrss_param.base_risk_pct": 0.014,
            "helix.unit_risk_pct": 0.0120,
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
        },
    ),
    (
        "helix_tpc_plus_keep_atrss_alpha",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
        },
    ),
]

FOCUSED_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "focused_top_plus_tpc",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "focused_top_tpc_mid",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.00475,
            "tpc.max_heat_R": 3.90,
            "tpc_param.all.max_risk_pct": 0.019,
            "tpc_param.all.risk_a_plus_pct": 0.019,
            "tpc_param.all.risk_a_pct": 0.0115,
            "tpc_param.all.risk_b_pct": 0.0085,
        },
    ),
    (
        "focused_soft_atrss_plus_others",
        {
            "atrss.unit_risk_pct": 0.0160,
            "atrss.max_heat_R": 2.10,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "focused_mid_atrss_plus_others",
        {
            "atrss.unit_risk_pct": 0.01625,
            "atrss.max_heat_R": 2.10,
            "helix.unit_risk_pct": 0.01275,
            "helix.max_heat_R": 2.05,
            "tpc.unit_risk_pct": 0.00475,
            "tpc.max_heat_R": 3.90,
            "tpc_param.all.max_risk_pct": 0.019,
            "tpc_param.all.risk_a_plus_pct": 0.019,
            "tpc_param.all.risk_a_pct": 0.0115,
            "tpc_param.all.risk_b_pct": 0.0085,
        },
    ),
    (
        "focused_helix_heavier",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0130,
            "helix.max_heat_R": 2.10,
            "tpc.unit_risk_pct": 0.0045,
            "tpc.max_heat_R": 3.75,
        },
    ),
    (
        "focused_helix_tpc_heavier",
        {
            "atrss.unit_risk_pct": 0.0165,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0130,
            "helix.max_heat_R": 2.10,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "focused_atrss_1675_others",
        {
            "atrss.unit_risk_pct": 0.01675,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 2.00,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "focused_low_dd_balance",
        {
            "atrss.unit_risk_pct": 0.01625,
            "atrss.max_heat_R": 2.00,
            "helix.unit_risk_pct": 0.0125,
            "helix.max_heat_R": 1.90,
            "tpc.unit_risk_pct": 0.00475,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.019,
            "tpc_param.all.risk_a_plus_pct": 0.019,
            "tpc_param.all.risk_a_pct": 0.0115,
            "tpc_param.all.risk_b_pct": 0.0085,
        },
    ),
]

TARGET_50_60_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "target60_soft_shift",
        {
            "atrss.unit_risk_pct": 0.0145,
            "atrss.max_heat_R": 1.95,
            "helix.unit_risk_pct": 0.0160,
            "helix.max_heat_R": 2.35,
            "tpc.unit_risk_pct": 0.0080,
            "tpc.max_heat_R": 4.50,
            "tpc_param.all.max_risk_pct": 0.032,
            "tpc_param.all.risk_a_plus_pct": 0.032,
            "tpc_param.all.risk_a_pct": 0.020,
            "tpc_param.all.risk_b_pct": 0.014,
        },
    ),
    (
        "target58_core",
        {
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 1.90,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.50,
            "tpc.unit_risk_pct": 0.0090,
            "tpc.max_heat_R": 4.75,
            "tpc_param.all.max_risk_pct": 0.036,
            "tpc_param.all.risk_a_plus_pct": 0.036,
            "tpc_param.all.risk_a_pct": 0.022,
            "tpc_param.all.risk_b_pct": 0.016,
        },
    ),
    (
        "target55_core",
        {
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 1.85,
            "helix.unit_risk_pct": 0.0180,
            "helix.max_heat_R": 2.65,
            "tpc.unit_risk_pct": 0.0110,
            "tpc.max_heat_R": 5.00,
            "tpc_param.all.max_risk_pct": 0.044,
            "tpc_param.all.risk_a_plus_pct": 0.044,
            "tpc_param.all.risk_a_pct": 0.027,
            "tpc_param.all.risk_b_pct": 0.019,
        },
    ),
    (
        "target52_aggressive",
        {
            "atrss.unit_risk_pct": 0.0130,
            "atrss.max_heat_R": 1.80,
            "helix.unit_risk_pct": 0.0195,
            "helix.max_heat_R": 2.85,
            "tpc.unit_risk_pct": 0.0125,
            "tpc.max_heat_R": 5.25,
            "tpc_param.all.max_risk_pct": 0.050,
            "tpc_param.all.risk_a_plus_pct": 0.050,
            "tpc_param.all.risk_a_pct": 0.031,
            "tpc_param.all.risk_b_pct": 0.022,
        },
    ),
    (
        "target55_dd_guarded",
        {
            "heat_cap_R": 5.25,
            "portfolio_daily_stop_R": 3.25,
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 1.75,
            "helix.unit_risk_pct": 0.0180,
            "helix.max_heat_R": 2.45,
            "tpc.unit_risk_pct": 0.0110,
            "tpc.max_heat_R": 4.75,
            "tpc_param.all.max_risk_pct": 0.044,
            "tpc_param.all.risk_a_plus_pct": 0.044,
            "tpc_param.all.risk_a_pct": 0.027,
            "tpc_param.all.risk_b_pct": 0.019,
        },
    ),
    (
        "target58_dd_guarded",
        {
            "heat_cap_R": 5.25,
            "portfolio_daily_stop_R": 3.25,
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 1.80,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.35,
            "tpc.unit_risk_pct": 0.0090,
            "tpc.max_heat_R": 4.50,
            "tpc_param.all.max_risk_pct": 0.036,
            "tpc_param.all.risk_a_plus_pct": 0.036,
            "tpc_param.all.risk_a_pct": 0.022,
            "tpc_param.all.risk_b_pct": 0.016,
        },
    ),
    (
        "target55_quality_guarded",
        {
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 1.85,
            "helix.unit_risk_pct": 0.0180,
            "helix.max_heat_R": 2.65,
            "tpc.unit_risk_pct": 0.0110,
            "tpc.max_heat_R": 5.00,
            "tpc_param.all.max_risk_pct": 0.044,
            "tpc_param.all.risk_a_plus_pct": 0.044,
            "tpc_param.all.risk_a_pct": 0.027,
            "tpc_param.all.risk_b_pct": 0.019,
            "tpc_param.all.score_a_min": 11,
            "tpc_param.all.score_b_min": 10,
            "tpc_param.all.second_entry_score_min": 16,
        },
    ),
    (
        "target58_quality_guarded",
        {
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 1.90,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.50,
            "tpc.unit_risk_pct": 0.0090,
            "tpc.max_heat_R": 4.75,
            "tpc_param.all.max_risk_pct": 0.036,
            "tpc_param.all.risk_a_plus_pct": 0.036,
            "tpc_param.all.risk_a_pct": 0.022,
            "tpc_param.all.risk_b_pct": 0.016,
            "tpc_param.all.score_a_min": 11,
            "tpc_param.all.score_b_min": 10,
            "tpc_param.all.second_entry_score_min": 16,
        },
    ),
    (
        "target60_helix_led",
        {
            "atrss.unit_risk_pct": 0.0145,
            "atrss.max_heat_R": 1.95,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.80,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "target58_tpc_led",
        {
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 1.90,
            "helix.unit_risk_pct": 0.0155,
            "helix.max_heat_R": 2.30,
            "tpc.unit_risk_pct": 0.0115,
            "tpc.max_heat_R": 5.00,
            "tpc_param.all.max_risk_pct": 0.046,
            "tpc_param.all.risk_a_plus_pct": 0.046,
            "tpc_param.all.risk_a_pct": 0.028,
            "tpc_param.all.risk_b_pct": 0.020,
        },
    ),
]

TARGET_50_60_REFINED_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "refined60_restore_atrss_heat",
        {
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.60,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "refined58_restore_atrss_heat",
        {
            "atrss.unit_risk_pct": 0.0130,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0195,
            "helix.max_heat_R": 2.70,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "refined60_helix_led_tpc_low",
        {
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0195,
            "helix.max_heat_R": 2.70,
            "tpc.unit_risk_pct": 0.0050,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.020,
            "tpc_param.all.risk_a_plus_pct": 0.020,
            "tpc_param.all.risk_a_pct": 0.012,
            "tpc_param.all.risk_b_pct": 0.009,
        },
    ),
    (
        "refined60_scaled95",
        {
            "atrss.unit_risk_pct": 0.01375,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0175,
            "helix.max_heat_R": 2.50,
            "tpc.unit_risk_pct": 0.0062,
            "tpc.max_heat_R": 4.10,
            "tpc_param.all.max_risk_pct": 0.025,
            "tpc_param.all.risk_a_plus_pct": 0.025,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "refined60_scaled92_ddtiers",
        {
            "drawdown_risk_tiers": [
                [0.035, 0.85],
                [0.055, 0.65],
                [0.075, 0.45],
                [0.095, 0.20],
                [0.120, 0.0],
            ],
            "atrss.unit_risk_pct": 0.0134,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.45,
            "tpc.unit_risk_pct": 0.0060,
            "tpc.max_heat_R": 4.00,
            "tpc_param.all.max_risk_pct": 0.024,
            "tpc_param.all.risk_a_plus_pct": 0.024,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "refined60_restore_ddtiers",
        {
            "drawdown_risk_tiers": [
                [0.035, 0.85],
                [0.055, 0.65],
                [0.075, 0.45],
                [0.095, 0.20],
                [0.120, 0.0],
            ],
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.60,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
    (
        "refined58_core_ddtiers",
        {
            "drawdown_risk_tiers": [
                [0.035, 0.85],
                [0.055, 0.65],
                [0.075, 0.45],
                [0.095, 0.20],
                [0.120, 0.0],
            ],
            "atrss.unit_risk_pct": 0.0140,
            "atrss.max_heat_R": 1.90,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.50,
            "tpc.unit_risk_pct": 0.0090,
            "tpc.max_heat_R": 4.75,
            "tpc_param.all.max_risk_pct": 0.036,
            "tpc_param.all.risk_a_plus_pct": 0.036,
            "tpc_param.all.risk_a_pct": 0.022,
            "tpc_param.all.risk_b_pct": 0.016,
        },
    ),
    (
        "refined60_daily_stop",
        {
            "portfolio_daily_stop_R": 3.00,
            "atrss.unit_risk_pct": 0.0135,
            "atrss.max_heat_R": 2.15,
            "helix.unit_risk_pct": 0.0185,
            "helix.max_heat_R": 2.60,
            "tpc.unit_risk_pct": 0.0065,
            "tpc.max_heat_R": 4.25,
            "tpc_param.all.max_risk_pct": 0.026,
            "tpc_param.all.risk_a_plus_pct": 0.026,
            "tpc_param.all.risk_a_pct": 0.016,
            "tpc_param.all.risk_b_pct": 0.012,
        },
    ),
]

TARGET_50_60_NARROW_CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    (
        "narrow60_scaled95_heatcap",
        {
            "heat_cap_R": 5.00,
            "atrss.unit_risk_pct": 0.01375,
            "atrss.max_heat_R": 2.05,
            "helix.unit_risk_pct": 0.0175,
            "helix.max_heat_R": 2.20,
            "tpc.unit_risk_pct": 0.0062,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.025,
            "tpc_param.all.risk_a_plus_pct": 0.025,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "narrow58_lowheat",
        {
            "heat_cap_R": 5.00,
            "atrss.unit_risk_pct": 0.0132,
            "atrss.max_heat_R": 2.05,
            "helix.unit_risk_pct": 0.0175,
            "helix.max_heat_R": 2.20,
            "tpc.unit_risk_pct": 0.0062,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.025,
            "tpc_param.all.risk_a_plus_pct": 0.025,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "narrow57_lowheat_plus_helix",
        {
            "heat_cap_R": 5.00,
            "atrss.unit_risk_pct": 0.0130,
            "atrss.max_heat_R": 2.05,
            "helix.unit_risk_pct": 0.0180,
            "helix.max_heat_R": 2.25,
            "tpc.unit_risk_pct": 0.0062,
            "tpc.max_heat_R": 3.75,
            "tpc_param.all.max_risk_pct": 0.025,
            "tpc_param.all.risk_a_plus_pct": 0.025,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "narrow60_scaled92_heatcap",
        {
            "heat_cap_R": 4.75,
            "atrss.unit_risk_pct": 0.0134,
            "atrss.max_heat_R": 2.00,
            "helix.unit_risk_pct": 0.0170,
            "helix.max_heat_R": 2.15,
            "tpc.unit_risk_pct": 0.0060,
            "tpc.max_heat_R": 3.60,
            "tpc_param.all.max_risk_pct": 0.024,
            "tpc_param.all.risk_a_plus_pct": 0.024,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "narrow58_scaled92_heatcap",
        {
            "heat_cap_R": 4.75,
            "atrss.unit_risk_pct": 0.0130,
            "atrss.max_heat_R": 2.00,
            "helix.unit_risk_pct": 0.0173,
            "helix.max_heat_R": 2.15,
            "tpc.unit_risk_pct": 0.0061,
            "tpc.max_heat_R": 3.60,
            "tpc_param.all.max_risk_pct": 0.024,
            "tpc_param.all.risk_a_plus_pct": 0.024,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
    (
        "narrow58_lowheat_ddtiers",
        {
            "heat_cap_R": 4.75,
            "drawdown_risk_tiers": [
                [0.030, 0.85],
                [0.050, 0.65],
                [0.070, 0.45],
                [0.090, 0.20],
                [0.115, 0.0],
            ],
            "atrss.unit_risk_pct": 0.0130,
            "atrss.max_heat_R": 2.00,
            "helix.unit_risk_pct": 0.0173,
            "helix.max_heat_R": 2.15,
            "tpc.unit_risk_pct": 0.0061,
            "tpc.max_heat_R": 3.60,
            "tpc_param.all.max_risk_pct": 0.024,
            "tpc_param.all.risk_a_plus_pct": 0.024,
            "tpc_param.all.risk_a_pct": 0.015,
            "tpc_param.all.risk_b_pct": 0.011,
        },
    ),
]

_DATA = None
_DATA_DIR: Path | None = None
_EQUITY = STARTING_EQUITY


def _init_worker(data_dir: str, equity: float) -> None:
    global _DATA, _DATA_DIR, _EQUITY
    _DATA_DIR = Path(data_dir)
    _EQUITY = float(equity)
    seed = UnifiedBacktestConfig(initial_equity=_EQUITY, data_dir=_DATA_DIR)
    _DATA = load_unified_data(seed)


def _strategy_shares(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary = metrics.get("strategy_summary", {}) or {}
    total = sum(max(float(item.get("static_risk_pnl", 0.0) or 0.0), 0.0) for item in summary.values())
    shares: dict[str, dict[str, float]] = {}
    for sid, item in summary.items():
        static_pnl = float(item.get("static_risk_pnl", 0.0) or 0.0)
        shares[sid] = {
            "trades": float(item.get("trades", 0.0) or 0.0),
            "signals": float(item.get("entry_signals_fired", 0.0) or 0.0),
            "accepted": float(item.get("entries_accepted", 0.0) or 0.0),
            "blocked": float(item.get("entries_blocked", 0.0) or 0.0),
            "total_r": float(item.get("total_r", 0.0) or 0.0),
            "unit_risk": float(item.get("initial_unit_risk_dollars", 0.0) or 0.0),
            "static_pnl": static_pnl,
            "share_pct": (static_pnl / total * 100.0) if total > 0.0 else 0.0,
        }
    return shares


def _balance_stats(shares: dict[str, dict[str, float]]) -> dict[str, float]:
    positive = [max(item["static_pnl"], 0.0) for item in shares.values()]
    total = sum(positive)
    proportions = [value / total for value in positive if total > 0.0 and value > 0.0]
    entropy = -sum(p * math.log(p) for p in proportions) / math.log(len(positive)) if len(proportions) > 1 else 0.0
    return {
        "max_share_pct": max((item["share_pct"] for item in shares.values()), default=0.0),
        "entropy": entropy,
        "atrss_share_pct": shares.get("ATRSS", {}).get("share_pct", 0.0),
        "non_atrss_share_pct": sum(
            item["share_pct"]
            for sid, item in shares.items()
            if sid != "ATRSS"
        ),
    }


def _summarize(name: str, mutations: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    shares = _strategy_shares(metrics)
    return {
        "name": name,
        "mutations": mutations,
        "net_return_pct": float(metrics.get("net_return_pct", 0.0) or 0.0),
        "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0) or 0.0),
        "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
        "total_trades": int(metrics.get("total_trades", 0) or 0),
        "heat_max_R": float(metrics.get("heat_max_R", 0.0) or 0.0),
        "strategy_shares": shares,
        **_balance_stats(shares),
    }


def _evaluate_named(task: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    if _DATA is None or _DATA_DIR is None:
        raise RuntimeError("Smoke worker was not initialized.")
    name, base, overrides = task
    mutations = dict(base)
    mutations.update(overrides)
    _, _, metrics = _evaluate(_DATA, mutations, equity=_EQUITY, data_dir=_DATA_DIR)
    return _summarize(name, overrides, metrics)


def _materiality_flags(row: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    return_floor = baseline["net_return_pct"] * MATERIAL_RETURN_FLOOR_FRAC
    dd_ceiling = baseline["max_drawdown_pct"] + MATERIAL_DD_BUFFER_PCT
    return {
        "return_ok": row["net_return_pct"] >= return_floor,
        "dd_ok": row["max_drawdown_pct"] <= dd_ceiling,
        "trades_ok": row["total_trades"] >= baseline["total_trades"] - 10,
        "more_balanced": row["max_share_pct"] < baseline["max_share_pct"],
    }


def _score_smoke(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    return_delta = (row["net_return_pct"] - baseline["net_return_pct"]) / max(abs(baseline["net_return_pct"]), 1e-9)
    dd_delta = (row["max_drawdown_pct"] - baseline["max_drawdown_pct"]) / 100.0
    trade_delta = (row["total_trades"] - baseline["total_trades"]) / max(float(baseline["total_trades"]), 1.0)
    balance_gain = (baseline["max_share_pct"] - row["max_share_pct"]) / 100.0
    entropy_gain = row["entropy"] - baseline["entropy"]
    return 0.48 * return_delta + 0.22 * trade_delta + 0.20 * balance_gain + 0.10 * entropy_gain - 0.25 * max(dd_delta, 0.0)


def _score_target_50_60(row: dict[str, Any], baseline: dict[str, Any]) -> float:
    smoke_score = _score_smoke(row, baseline)
    atrss_share = row.get("atrss_share_pct", row["max_share_pct"])
    if 50.0 <= atrss_share <= 60.0:
        target_bonus = 0.065
    else:
        target_bonus = -abs(atrss_share - 55.0) / 100.0
    pf_delta = (row["profit_factor"] - baseline["profit_factor"]) / max(baseline["profit_factor"], 1e-9)
    heat_delta = (row["heat_max_R"] - baseline["heat_max_R"]) / 10.0
    return smoke_score + target_bonus + 0.06 * pf_delta - 0.05 * max(heat_delta, 0.0)


def run_smoke(args: argparse.Namespace) -> Path:
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = json.loads(base_path.read_text())

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "backtests" / "output" / "swing" / "portfolio_synergy" / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"round2_balance_smoke_{timestamp}.json"
    report_path = output_dir / f"round2_balance_smoke_{timestamp}.txt"

    if args.mode == "focused":
        candidates = FOCUSED_CANDIDATES
    elif args.mode == "target50":
        candidates = TARGET_50_60_CANDIDATES
    elif args.mode == "target50refined":
        candidates = TARGET_50_60_REFINED_CANDIDATES
    elif args.mode == "target50narrow":
        candidates = TARGET_50_60_NARROW_CANDIDATES
    elif args.mode == "both":
        candidates = SMOKE_CANDIDATES + FOCUSED_CANDIDATES
    else:
        candidates = SMOKE_CANDIDATES
    tasks = [("__baseline__", base, {})] + [(name, base, mutations) for name, mutations in candidates]
    workers = max(1, int(args.max_workers))
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(data_dir), float(args.equity)),
    ) as pool:
        future_map = {pool.submit(_evaluate_named, task): task[0] for task in tasks}
        for future in as_completed(future_map):
            row = future.result()
            print(
                f"{row['name']}: return={row['net_return_pct']:.2f}% "
                f"dd={row['max_drawdown_pct']:.2f}% pf={row['profit_factor']:.2f} "
                f"trades={row['total_trades']} max_share={row['max_share_pct']:.2f}%",
                flush=True,
            )
            results.append(row)

    baseline = next(row for row in results if row["name"] == "__baseline__")
    for row in results:
        flags = _materiality_flags(row, baseline)
        if args.mode in {"target50", "target50refined", "target50narrow"}:
            flags["atrss_share_50_60"] = 50.0 <= row["atrss_share_pct"] <= 60.0
        row["materiality"] = flags
        row["smoke_score"] = _score_smoke(row, baseline)
        row["target_50_60_score"] = _score_target_50_60(row, baseline)
        row["passes_materiality"] = all(flags.values())
    rank_key = "target_50_60_score" if args.mode in {"target50", "target50refined", "target50narrow"} else "smoke_score"
    ranked = sorted(results, key=lambda item: item[rank_key], reverse=True)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_path),
        "data_dir": str(data_dir),
        "equity": float(args.equity),
        "materiality": {
            "return_floor_frac": MATERIAL_RETURN_FLOOR_FRAC,
            "dd_buffer_pct": MATERIAL_DD_BUFFER_PCT,
            "trade_slack": 10,
        },
        "baseline": baseline,
        "ranked": ranked,
        "candidates": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    lines = [
        "ROUND 2 BALANCE SMOKE TEST",
        "=" * 70,
        f"Base config: {base_path}",
        f"Material floor: return >= {baseline['net_return_pct'] * MATERIAL_RETURN_FLOOR_FRAC:.2f}%, "
        f"DD <= {baseline['max_drawdown_pct'] + MATERIAL_DD_BUFFER_PCT:.2f}%, "
        f"trades >= {baseline['total_trades'] - 10}",
        "ATRSS target band: 50.00% to 60.00%" if args.mode in {"target50", "target50refined", "target50narrow"} else "",
        "",
        "Ranked Candidates",
    ]
    for row in ranked:
        shares = row["strategy_shares"]
        lines.append(
            f"  {row['name']:<30} score={row['smoke_score']:+.4f} "
            f"target_score={row['target_50_60_score']:+.4f} "
            f"return={row['net_return_pct']:+.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"pf={row['profit_factor']:.2f} trades={row['total_trades']} "
            f"atrss={row['atrss_share_pct']:.2f}% max_share={row['max_share_pct']:.2f}% "
            f"non_atrss={row['non_atrss_share_pct']:.2f}% "
            f"passes={row['passes_materiality']}"
        )
        lines.append(
            "    "
            + ", ".join(
                f"{sid}:{item['share_pct']:.1f}%/{int(item['trades'])}tr/${item['static_pnl']:,.0f}"
                for sid, item in shares.items()
            )
        )
    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=BASE_CONFIG)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--equity", type=float, default=STARTING_EQUITY)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("all", "focused", "target50", "target50refined", "target50narrow", "both"),
        default="all",
    )
    args = parser.parse_args(argv)
    run_smoke(args)


if __name__ == "__main__":
    main()
