"""Leakage-control candidate schedule for Helix round 3."""
from __future__ import annotations

from typing import Any


def get_phase_candidates(
    phase: int,
    prior_mutations: dict[str, Any] | None = None,
    suggested_experiments: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    del prior_mutations
    if phase == 1:
        candidates = _phase_1_candidates()
    elif phase == 2:
        candidates = _phase_2_candidates()
    else:
        candidates = []

    if suggested_experiments:
        seen = {name for name, _ in candidates}
        for name, mutations in suggested_experiments:
            if name not in seen:
                candidates.append((name, mutations))
                seen.add(name)
    return candidates


def _rts_guard(
    *,
    mfe: float,
    giveback: float,
    floor: float,
    bars: int,
    fade_bars: int,
    max_mfe: float,
    flatten_r: float | None = None,
) -> dict[str, Any]:
    mutations: dict[str, Any] = {
        "param_overrides.RTS_GUARD_MFE_R": mfe,
        "param_overrides.RTS_GUARD_MIN_GIVEBACK_R": giveback,
        "param_overrides.RTS_GUARD_FLOOR_R": floor,
        "param_overrides.RTS_GUARD_MIN_BARS": bars,
        "param_overrides.RTS_GUARD_FADE_BARS": fade_bars,
        "param_overrides.RTS_GUARD_MAX_MFE_R": max_mfe,
    }
    if flatten_r is not None:
        mutations["param_overrides.RTS_FAIL_FLATTEN_R"] = flatten_r
    return mutations


def _phase_1_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("p1_reset_light_guard_be105", {
            **_rts_guard(
                mfe=0.75, giveback=0.40, floor=-0.10, bars=8, fade_bars=2, max_mfe=1.75,
            ),
            "param_overrides.R_BE_1H": 1.05,
        }),
        ("p1_light_guard_be095", {
            **_rts_guard(
                mfe=0.75, giveback=0.40, floor=-0.10, bars=8, fade_bars=2, max_mfe=1.75,
            ),
            "param_overrides.R_BE_1H": 0.95,
        }),
        ("p1_disable_rts_be105", {
            "param_overrides.RTS_GUARD_MFE_R": 0.0,
            "param_overrides.RTS_FAIL_FLATTEN_R": -999.0,
            "param_overrides.R_BE_1H": 1.05,
        }),
        ("p1_disable_rts_be095", {
            "param_overrides.RTS_GUARD_MFE_R": 0.0,
            "param_overrides.RTS_FAIL_FLATTEN_R": -999.0,
            "param_overrides.R_BE_1H": 0.95,
        }),
        ("p1_mid_guard_be105", {
            **_rts_guard(
                mfe=0.75, giveback=0.35, floor=0.00, bars=4, fade_bars=2, max_mfe=1.75,
            ),
            "param_overrides.R_BE_1H": 1.05,
        }),
        ("p1_late_guard_be105", {
            **_rts_guard(
                mfe=1.00, giveback=0.50, floor=0.20, bars=8, fade_bars=1, max_mfe=2.25,
            ),
            "param_overrides.R_BE_1H": 1.05,
        }),
        ("p1_heavy_guard_be095", {
            **_rts_guard(
                mfe=0.75, giveback=0.50, floor=0.20, bars=6, fade_bars=1, max_mfe=1.75,
            ),
            "param_overrides.R_BE_1H": 0.95,
        }),
    ]


def _phase_2_candidates() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("p2_reset_be105", {"param_overrides.R_BE_1H": 1.05}),
        ("p2_be_1h_090", {"param_overrides.R_BE_1H": 0.90}),
        ("p2_be_1h_095", {"param_overrides.R_BE_1H": 0.95}),
        ("p2_be_1h_110", {"param_overrides.R_BE_1H": 1.10}),
        ("p2_light_floor_m005", {"param_overrides.RTS_GUARD_FLOOR_R": -0.05}),
        ("p2_light_giveback_045", {"param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.45}),
        ("p2_light_max_mfe_150", {"param_overrides.RTS_GUARD_MAX_MFE_R": 1.50}),
        ("p2_light_max_mfe_200", {"param_overrides.RTS_GUARD_MAX_MFE_R": 2.00}),
        ("p2_light_giveback_035", {"param_overrides.RTS_GUARD_MIN_GIVEBACK_R": 0.35}),
    ]
