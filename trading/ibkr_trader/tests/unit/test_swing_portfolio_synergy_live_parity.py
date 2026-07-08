from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _norm(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_norm(item) for item in value]
    if isinstance(value, list):
        return [_norm(item) for item in value]
    return value


def _optimized_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "backtests" / "output" / "swing" / "portfolio_synergy" / "round_3" / "optimized_config.json"
    return json.loads(path.read_text())


def test_swing_live_configs_match_round3_portfolio_synergy() -> None:
    from strategies.swing import coordinator
    from strategies.swing.akc_helix import config as helix_config
    from strategies.swing.atrss import config as atrss_config
    from strategies.swing.overlay.config import OverlayConfig
    from strategies.swing.tpc import config as tpc_config

    optimized = _optimized_config()

    portfolio_values = {
        "heat_cap_R": coordinator._SWING_FAMILY_HEAT_CAP_R,
        "portfolio_daily_stop_R": coordinator._SWING_FAMILY_DAILY_STOP_R,
        "overlay_enabled": OverlayConfig().enabled,
        "atrss.unit_risk_pct": coordinator._RISK_PARAMS["ATRSS"]["unit_risk_pct"],
        "atrss.max_heat_R": coordinator._RISK_PARAMS["ATRSS"]["max_heat_R"],
        "atrss.daily_stop_R": coordinator._RISK_PARAMS["ATRSS"]["daily_stop_R"],
        "helix.unit_risk_pct": coordinator._RISK_PARAMS["AKC_HELIX"]["unit_risk_pct"],
        "helix.max_heat_R": coordinator._RISK_PARAMS["AKC_HELIX"]["max_heat_R"],
        "helix.daily_stop_R": coordinator._RISK_PARAMS["AKC_HELIX"]["daily_stop_R"],
        "tpc.unit_risk_pct": coordinator._RISK_PARAMS["TPC"]["unit_risk_pct"],
        "tpc.max_heat_R": coordinator._RISK_PARAMS["TPC"]["max_heat_R"],
        "tpc.daily_stop_R": coordinator._RISK_PARAMS["TPC"]["daily_stop_R"],
    }
    for key, live_value in portfolio_values.items():
        assert live_value == optimized[key], key
    assert coordinator._DD_TIERS == tuple(tuple(row) for row in optimized["drawdown_risk_tiers"])

    atrss_values = {
        "atrss_flags.early_stall_exit": atrss_config.EARLY_STALL_ENABLED,
        "atrss_flags.addon_b": atrss_config.ADDON_B_ENABLED,
        "atrss_param.early_stall_check_hours": atrss_config.EARLY_STALL_CHECK_HOURS,
        "atrss_param.early_stall_mfe_threshold": atrss_config.EARLY_STALL_MFE_THRESHOLD,
        "atrss_param.max_hold_hours": atrss_config.MAX_HOLD_HOURS,
        "atrss_param.recovery_tolerance_atr_trend": atrss_config.RECOVERY_TOLERANCE_ATR_TREND,
        "atrss_param.dynamic_risk_strong_trend_mult": atrss_config.DYNAMIC_RISK_STRONG_TREND_MULT,
        "atrss_param.dynamic_risk_weak_trend_mult": atrss_config.DYNAMIC_RISK_WEAK_TREND_MULT,
        "atrss_param.addon_a_r": atrss_config.ADDON_A_R,
        "atrss_param.addon_a_size_mult": atrss_config.ADDON_A_SIZE_MULT,
        "atrss_param.pullback_touch_tolerance_pct": atrss_config.PULLBACK_TOUCH_TOLERANCE_PCT,
    }
    for key, live_value in atrss_values.items():
        assert live_value == optimized[key], key
    for symbol in ("QQQ", "GLD"):
        assert atrss_config.ALL_SYMBOL_CONFIGS[symbol].adx_on == optimized["atrss_param.adx_on"]
        assert atrss_config.ALL_SYMBOL_CONFIGS[symbol].base_risk_pct == optimized["atrss_param.base_risk_pct"]

    helix_flag_map = {
        "disable_class_a": "DISABLE_CLASS_A",
        "disable_class_c": "DISABLE_CLASS_C",
        "disable_circuit_breaker": "DISABLE_CIRCUIT_BREAKER",
    }
    for key, expected in optimized.items():
        if key.startswith("helix_param."):
            assert getattr(helix_config, key.split(".", 1)[1]) == expected, key
        elif key.startswith("helix_flags."):
            attr = helix_flag_map[key.split(".", 1)[1]]
            assert getattr(helix_config, attr) == expected, key

    for key, expected in optimized.items():
        if not key.startswith("tpc_param."):
            continue
        _, scope, attr = key.split(".", 2)
        live_value = (
            getattr(tpc_config.TPCSymbolConfig, attr)
            if scope == "all"
            else getattr(tpc_config.SYMBOL_CONFIGS[scope], attr)
        )
        assert _norm(live_value) == _norm(expected), key
