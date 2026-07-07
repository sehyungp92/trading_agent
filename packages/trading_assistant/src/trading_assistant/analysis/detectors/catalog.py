"""Detector metadata catalog used by strategy analysis."""

from __future__ import annotations

from dataclasses import dataclass

from trading_assistant.analysis.detectors.base import DetectorMetadata


_ARCHETYPE_DEFAULTS: dict[str, dict[str, dict[str, float]]] = {
    "alpha_decay": {
        "trend_follow": {"decay_threshold": 0.55},
        "divergence_swing": {"decay_threshold": 0.55},
        "breakout": {"decay_threshold": 0.55},
        "box_breakout": {"decay_threshold": 0.55},
        "multi_tf_momentum": {"decay_threshold": 0.55},
        "pullback": {"decay_threshold": 0.80},
        "intraday_momentum": {"decay_threshold": 0.80},
        "opening_range_breakout": {"decay_threshold": 0.80},
        "vwap_pullback": {"decay_threshold": 0.80},
        "flow_following": {"decay_threshold": 0.80},
        "bear_regime_swing": {"decay_threshold": 0.50},
        "multi_engine_bear": {"decay_threshold": 0.55},
        "mean_reversion_pullback": {"decay_threshold": 0.80},
        "momentum_pullback_crypto": {"decay_threshold": 0.50},
        "institutional_anchor": {"decay_threshold": 0.50},
        "volume_profile_breakout": {"decay_threshold": 0.45},
    },
    "exit_timing": {
        "trend_follow": {"efficiency_threshold": 0.20},
        "divergence_swing": {"efficiency_threshold": 0.25},
        "multi_tf_momentum": {"efficiency_threshold": 0.35},
        "intraday_momentum": {"efficiency_threshold": 0.45},
        "opening_range_breakout": {"efficiency_threshold": 0.45},
        "vwap_pullback": {"efficiency_threshold": 0.45},
        "flow_following": {"efficiency_threshold": 0.40},
        "bear_regime_swing": {"efficiency_threshold": 0.25},
        "multi_engine_bear": {"efficiency_threshold": 0.35},
        "mean_reversion_pullback": {"efficiency_threshold": 0.45},
        "momentum_pullback_crypto": {"efficiency_threshold": 0.25},
        "institutional_anchor": {"efficiency_threshold": 0.20},
        "volume_profile_breakout": {"efficiency_threshold": 0.30},
    },
    "funding_impact": {
        "momentum_pullback_crypto": {"cost_threshold": 0.15},
        "institutional_anchor": {"cost_threshold": 0.20},
        "volume_profile_breakout": {"cost_threshold": 0.10},
    },
    "liquidation_proximity": {
        "momentum_pullback_crypto": {"proximity_threshold": 0.70},
        "institutional_anchor": {"proximity_threshold": 0.70},
        "volume_profile_breakout": {"proximity_threshold": 0.65},
    },
    "funding_trend": {
        "momentum_pullback_crypto": {"cost_threshold": 0.15},
        "institutional_anchor": {"cost_threshold": 0.20},
        "volume_profile_breakout": {"cost_threshold": 0.10},
    },
}


_DETECTOR_TO_CATEGORY: dict[str, str] = {
    "tight_stop": "stop_loss",
    "wide_stop": "stop_loss",
    "filter_cost": "filter_threshold",
    "regime_loss": "regime_gate",
    "alpha_decay": "signal",
    "signal_decay": "signal",
    "component_signal_decay": "signal",
    "factor_decay": "signal",
    "exit_timing": "exit_timing",
    "correlation": "signal",
    "time_of_day": "signal",
    "drawdown_concentration": "stop_loss",
    "position_sizing": "position_sizing",
    "filter_interactions": "filter_threshold",
    "microstructure": "signal",
    "regime_config_effectiveness": "regime_gate",
    "regime_transition_cost": "regime_gate",
    "stress_entry_pattern": "regime_gate",
    "execution_bottleneck": "signal",
    "sizing_methodology": "position_sizing",
    "portfolio_crowding": "position_sizing",
    "detect_family_imbalance": "position_sizing",
    "detect_correlation_concentration": "signal",
    "detect_drawdown_tier_miscalibration": "stop_loss",
    "detect_coordination_gaps": "position_sizing",
    "detect_heat_cap_utilization": "position_sizing",
    "funding_impact": "filter_threshold",
    "grade_selectivity": "signal",
    "confluence_quality": "filter_threshold",
    "leverage_utilization": "position_sizing",
    "mtf_alignment_drift": "signal",
    "liquidation_proximity": "leverage_cap",
    "symbol_concentration": "position_sizing",
    "session_patterns_24_7": "signal",
    "funding_trend": "funding_threshold",
}


_THRESHOLD_DEFAULTS: dict[str, dict[str, float]] = {
    "tight_stop": {"tight_stop_ratio": 0.3},
    "filter_cost": {"filter_cost_threshold": 0.0},
    "regime_loss": {"regime_loss_threshold": 0.0, "regime_min_weeks": 3.0},
    "alpha_decay": {"decay_threshold": 0.3},
    "signal_decay": {"win_rate_drop_threshold": 0.15},
    "component_signal_decay": {"stability_threshold": 0.3, "correlation_threshold": 0.05},
    "exit_timing": {"efficiency_threshold": 0.5, "premature_threshold": 0.4},
    "correlation": {"threshold": 0.7},
    "time_of_day": {"loss_threshold": 0.35, "min_trades": 10.0},
    "drawdown_concentration": {"concentration_threshold": 3.0},
    "position_sizing": {"loss_win_ratio_threshold": 1.5},
    "filter_interactions": {"redundancy_threshold": 0.5},
    "microstructure": {"spread_threshold_bps": 5.0, "imbalance_threshold": 2.0},
    "regime_config_effectiveness": {"min_trades": 10.0},
    "regime_transition_cost": {"window_days": 5.0},
    "stress_entry_pattern": {"min_trades_per_bucket": 5.0},
    "detect_correlation_concentration": {"threshold": 0.7, "weight_threshold": 0.4},
    "funding_impact": {"cost_threshold": 0.15},
    "grade_selectivity": {"min_trades": 20.0},
    "confluence_quality": {"lift_threshold": 0.1},
    "leverage_utilization": {"utilization_warning": 0.8},
    "mtf_alignment_drift": {"min_mismatched": 5.0, "win_rate_gap_threshold": 0.15},
    "liquidation_proximity": {"proximity_threshold": 0.7, "systemic_count": 3.0},
    "symbol_concentration": {"concentration_threshold": 0.7, "min_trades": 10.0},
    "session_patterns_24_7": {"min_trades": 10.0, "negative_avg_pnl_threshold": 0.0},
    "funding_trend": {"cost_threshold": 0.15, "rising_weeks": 3.0},
}


_DECREASE_KEYWORDS = frozenset({
    "tighten", "reduce", "lower", "decrease", "narrow", "less", "smaller",
    "cut", "shrink", "restrict", "shorten",
})
_INCREASE_KEYWORDS = frozenset({
    "widen", "increase", "raise", "expand", "more", "larger", "bigger",
    "extend", "loosen", "relax", "lengthen",
})


@dataclass(frozen=True)
class DetectorCatalog:
    metadata: dict[str, DetectorMetadata]
    decrease_keywords: frozenset[str] = _DECREASE_KEYWORDS
    increase_keywords: frozenset[str] = _INCREASE_KEYWORDS

    @property
    def categories(self) -> dict[str, str]:
        return {name: item.category for name, item in self.metadata.items()}

    @property
    def archetype_defaults(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            name: item.archetype_defaults
            for name, item in self.metadata.items()
            if item.archetype_defaults
        }

    @property
    def threshold_defaults(self) -> dict[str, dict[str, float]]:
        return {
            name: item.threshold_defaults
            for name, item in self.metadata.items()
            if item.threshold_defaults
        }

    def category_for(self, detector_name: str) -> str:
        return self.metadata.get(detector_name, DetectorMetadata(detector_name, "")).category

    def archetype_default(self, detector_name: str, archetype: str, param: str) -> float | None:
        return (
            self.metadata
            .get(detector_name, DetectorMetadata(detector_name, ""))
            .archetype_defaults
            .get(archetype, {})
            .get(param)
        )

    def threshold_default(self, detector_name: str, param: str) -> float | None:
        return (
            self.metadata
            .get(detector_name, DetectorMetadata(detector_name, ""))
            .threshold_defaults
            .get(param)
        )


DEFAULT_DETECTOR_CATALOG = DetectorCatalog(
    metadata={
        name: DetectorMetadata(
            name=name,
            category=category,
            threshold_defaults=_THRESHOLD_DEFAULTS.get(name, {}),
            archetype_defaults=_ARCHETYPE_DEFAULTS.get(name, {}),
        )
        for name, category in _DETECTOR_TO_CATEGORY.items()
    },
)
