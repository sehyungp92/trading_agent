"""ATRSS phase-specific scoring -- weights are immutable, gates are progressive.

Two scoring regimes:
  - R1 (r1_independent): Original high thresholds for independent-account mode.
  - R9 (r9_synchronized): Rescaled for honest synchronized/fee-net conditions.

Active regime is selected via SCORING_REGIME module variable.
"""
from __future__ import annotations

from .scoring import ATRSSCompositeScore, ATRSSMetrics, composite_score

# ---------------------------------------------------------------------------
# Active scoring regime -- set to "r9" for honest conditions
# ---------------------------------------------------------------------------
SCORING_REGIME: str = "r9"

# Weights are immutable across all phases (None = use defaults)
PHASE_WEIGHTS: dict[int, dict[str, float] | None] = {
    1: None,
    2: None,
    3: None,
    4: None,
}

# Progressive hard rejects per phase -- keyed by regime
_PHASE_HARD_REJECTS_R1: dict[int, dict[str, float]] = {
    1: {"min_trades": 100, "max_dd_pct": 0.07, "min_pf": 2.0, "min_wr": 0.55},
    2: {"min_trades": 120, "max_dd_pct": 0.06, "min_pf": 2.5, "min_wr": 0.58},
    3: {"min_trades": 140, "max_dd_pct": 0.055, "min_pf": 3.0, "min_wr": 0.60},
    4: {"min_trades": 150, "max_dd_pct": 0.05, "min_pf": 3.5, "min_wr": 0.65},
}

_PHASE_HARD_REJECTS_R9: dict[int, dict[str, float]] = {
    1: {"min_trades": 220, "max_dd_pct": 0.06, "min_pf": 1.8, "min_wr": 0.58},
    2: {"min_trades": 230, "max_dd_pct": 0.055, "min_pf": 2.0, "min_wr": 0.60},
    3: {"min_trades": 230, "max_dd_pct": 0.055, "min_pf": 2.0, "min_wr": 0.60},
    4: {"min_trades": 240, "max_dd_pct": 0.05, "min_pf": 2.2, "min_wr": 0.62},
}

RISK_ALLOCATION_PHASE_HARD_REJECTS: dict[int, dict[str, float]] = {
    1: {"min_trades": 250, "max_dd_pct": 0.075, "min_pf": 4.5, "min_wr": 0.78},
    2: {"min_trades": 250, "max_dd_pct": 0.078, "min_pf": 4.5, "min_wr": 0.78},
    3: {"min_trades": 250, "max_dd_pct": 0.080, "min_pf": 4.3, "min_wr": 0.76},
    4: {"min_trades": 250, "max_dd_pct": 0.080, "min_pf": 4.5, "min_wr": 0.78},
}


def _get_phase_hard_rejects() -> dict[int, dict[str, float]]:
    return _PHASE_HARD_REJECTS_R9 if SCORING_REGIME == "r9" else _PHASE_HARD_REJECTS_R1


# Module-level export -- set once from SCORING_REGIME at import time.
# Internal callers should use _get_phase_hard_rejects() for runtime dispatch.
PHASE_HARD_REJECTS: dict[int, dict[str, float]] = (
    _PHASE_HARD_REJECTS_R9 if SCORING_REGIME == "r9" else _PHASE_HARD_REJECTS_R1
)

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Opportunity Surface", ["total_trades", "trades_per_month", "total_r"]),
    2: ("Signal Geometry", ["total_trades", "win_rate", "trades_per_month"]),
    3: ("Execution & Stops", ["profit_factor", "total_r", "net_return_pct"]),
    4: ("Exits, Add-ons & Allocation", ["mfe_capture", "calmar_r", "total_r"]),
}

RISK_ALLOCATION_PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: ("Dynamic Risk Exposure Sweep", ["net_return_pct", "max_dd_pct", "profit_factor"]),
    2: ("Risk & Heat Calibration", ["net_return_pct", "max_dd_pct", "calmar_r"]),
    3: ("Winner Lean-In Add-ons", ["net_return_pct", "mfe_capture", "profit_factor"]),
    4: ("Aggression Guardrails", ["calmar_r", "max_dd_pct", "net_return_pct"]),
}

_ULTIMATE_TARGETS_R1 = {
    "total_r": 300.0,
    "profit_factor": 8.0,
    "max_dd_pct": 0.015,
    "calmar_r": 70.0,
    "total_trades": 300,
    "mfe_capture": 0.80,
    "win_rate": 0.80,
    "trades_per_month": 5.0,
}

_ULTIMATE_TARGETS_R9 = {
    # Aspirational targets above Phase 0 vanilla baseline (2026-04-27):
    # Vanilla actuals: R=190.8, PF=4.47, DD=2.07%, cal_r=40.9, n=290,
    #                  MFE=0.654, WR=74.1%, TPM=5.0
    "total_r": 300.0,
    "profit_factor": 6.0,
    "max_dd_pct": 0.012,
    "calmar_r": 60.0,
    "total_trades": 400,
    "mfe_capture": 0.80,
    "win_rate": 0.85,
    "trades_per_month": 7.5,
}

ULTIMATE_TARGETS = _ULTIMATE_TARGETS_R9 if SCORING_REGIME == "r9" else _ULTIMATE_TARGETS_R1

RISK_ALLOCATION_ULTIMATE_TARGETS = {
    "total_r": 225.0,
    "profit_factor": 5.5,
    "max_dd_pct": 0.070,
    "calmar_r": 70.0,
    "total_trades": 270,
    "mfe_capture": 0.70,
    "win_rate": 0.82,
    "trades_per_month": 4.7,
    "net_return_pct": 140.0,
}


def _scoring_profile() -> str:
    """Return the composite_score profile name for the active regime."""
    return "r9_synchronized" if SCORING_REGIME == "r9" else "r1_independent"


def score_phase_metrics(
    phase: int,
    metrics: ATRSSMetrics,
    weight_overrides: dict[str, float] | None = None,
    hard_rejects: dict[str, float] | None = None,
    profile: str | None = None,
) -> ATRSSCompositeScore:
    """Score metrics for a specific phase with phase-appropriate hard rejects."""
    rejects = hard_rejects or _get_phase_hard_rejects().get(phase, {})
    scoring_profile = profile or _scoring_profile()
    # ATRSS uses fixed weights -- weight_overrides ignored
    return composite_score(metrics, weights=None, hard_rejects=rejects, profile=scoring_profile)
