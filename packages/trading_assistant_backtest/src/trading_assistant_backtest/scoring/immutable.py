"""Immutable replay objective profiles.

The reference optimizers use family-specific score contracts.  This module keeps the
same governance shape in the monthly runner: hard rejects first, then a capped,
renormalized weighted score with the exact profile persisted alongside each replay.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, pstdev
from typing import Any

from trading_assistant_backtest.replay.types import WindowSpec

IMMUTABLE_OBJECTIVE_VERSION = "immutable_score_profiles_v1"
DEFAULT_SCORE_COMPONENT_CAP = 7


@dataclass(frozen=True)
class Normalizer:
    metric: str
    kind: str = "ceiling"
    floor: float = 0.0
    ceiling: float = 1.0

    def normalize(self, metrics: dict[str, float]) -> float:
        value = _finite(metrics.get(self.metric, 0.0))
        if self.kind == "identity":
            return _clip(value)
        if self.kind == "ceiling":
            return _clip(value / self.ceiling if self.ceiling else 0.0)
        if self.kind == "floor_ceiling":
            span = self.ceiling - self.floor
            return _clip((value - self.floor) / span if span else 0.0)
        if self.kind == "inverse_ceiling":
            return _clip(1.0 - (value / self.ceiling if self.ceiling else 0.0))
        if self.kind == "inverse_floor_ceiling":
            span = self.ceiling - self.floor
            return _clip((self.ceiling - value) / span if span else 0.0)
        if self.kind == "binary_min":
            return 1.0 if value >= self.floor else 0.0
        if self.kind == "binary_max":
            return 1.0 if value <= self.ceiling else 0.0
        raise ValueError(f"unknown normalizer kind: {self.kind}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "kind": self.kind,
            "floor": self.floor,
            "ceiling": self.ceiling,
        }


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    weight: float
    normalizer: Normalizer
    source_components: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "component": self.name,
            "weight": self.weight,
            "normalizer": self.normalizer.to_payload(),
            "source_components": list(self.source_components or (self.name,)),
        }


@dataclass(frozen=True)
class HardReject:
    metric: str
    op: str
    threshold: float
    description: str = ""

    def passes(self, metrics: dict[str, float]) -> bool:
        value = _finite(metrics.get(self.metric, 0.0))
        if self.op == ">=":
            return value >= self.threshold
        if self.op == ">":
            return value > self.threshold
        if self.op == "<=":
            return value <= self.threshold
        if self.op == "<":
            return value < self.threshold
        raise ValueError(f"unknown hard-reject operator: {self.op}")

    def reason(self, metrics: dict[str, float]) -> str:
        value = _finite(metrics.get(self.metric, 0.0))
        label = self.description or self.metric
        return f"{label}: {value:.8f} must be {self.op} {self.threshold:.8f}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "op": self.op,
            "threshold": self.threshold,
            "description": self.description,
        }


@dataclass(frozen=True)
class ScoreProfile:
    profile_id: str
    family: str
    scope: str
    components: tuple[ComponentSpec, ...]
    hard_rejects: tuple[HardReject, ...] = ()
    base_profile_id: str = ""
    overlay_id: str = ""
    source_reference: str = ""
    source_round: str = ""
    reference_hard_rejects: tuple[str, ...] = ()
    version: str = IMMUTABLE_OBJECTIVE_VERSION
    component_cap: int = DEFAULT_SCORE_COMPONENT_CAP

    def capped_components(self, cap: int | None = None) -> tuple[ComponentSpec, ...]:
        effective_cap = self.component_cap if cap is None else max(1, min(cap, self.component_cap))
        return self.components[:effective_cap]

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "base_profile_id": self.base_profile_id,
            "overlay_id": self.overlay_id,
            "family": self.family,
            "scope": self.scope,
            "version": self.version,
            "component_cap": self.component_cap,
            "source_reference": self.source_reference,
            "source_round": self.source_round,
            "components": [component.to_payload() for component in self.components],
            "hard_rejects": [reject.to_payload() for reject in self.hard_rejects],
            "reference_hard_rejects": list(self.reference_hard_rejects),
        }


@dataclass(frozen=True)
class ScoreResult:
    objective_score: float
    profile: ScoreProfile
    components: list[dict[str, Any]]
    selected_components: list[dict[str, Any]]
    rejected: bool
    reject_reasons: list[str]
    metrics: dict[str, float]
    component_cap: int
    missing_components: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "immutable_score_result_v1",
            "profile_id": self.profile.profile_id,
            "profile_version": self.profile.version,
            "objective_score": self.objective_score,
            "rejected": self.rejected,
            "reject_reasons": self.reject_reasons,
            "score_component_cap": self.component_cap,
            "components": self.components,
            "renormalized_components": self.selected_components,
            "missing_components": self.missing_components,
            "metrics": self.metrics,
            "profile": self.profile.to_payload(),
        }


def resolve_score_profile(
    *,
    family: str = "",
    plugin_id: str = "",
    strategy_id: str = "",
) -> ScoreProfile:
    """Resolve the immutable family profile plus any strategy overlay."""

    plugin_key = _key(plugin_id)
    family_key = _key(family) or family_for_plugin(plugin_id)
    strategy_key = _key(strategy_id)
    profile_id = _PROFILE_ID_BY_PLUGIN.get(plugin_key)
    if profile_id is None:
        profile_id = _profile_id_for_family_strategy(family_key, strategy_key)
    return SCORE_PROFILES[profile_id]


def family_for_plugin(plugin_id: str) -> str:
    return _FAMILY_BY_PLUGIN.get(_key(plugin_id), "")


def score_replay(
    *,
    profile: ScoreProfile,
    trades: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    window: WindowSpec,
    net_return: float,
    max_drawdown: float,
    profit_factor: float,
    component_cap: int = DEFAULT_SCORE_COMPONENT_CAP,
) -> ScoreResult:
    metrics = replay_metrics(
        trades=trades,
        coverage=coverage,
        window=window,
        net_return=net_return,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
    )
    return score_metrics(profile=profile, metrics=metrics, component_cap=component_cap)


def score_metrics(
    *,
    profile: ScoreProfile,
    metrics: dict[str, float],
    component_cap: int = DEFAULT_SCORE_COMPONENT_CAP,
) -> ScoreResult:
    cap = max(1, min(component_cap, profile.component_cap, len(profile.components)))
    selected_specs = profile.capped_components(cap)
    rejected = False
    reject_reasons: list[str] = []
    for hard_reject in profile.hard_rejects:
        if not hard_reject.passes(metrics):
            rejected = True
            reject_reasons.append(hard_reject.reason(metrics))

    selected_names = {component.name for component in selected_specs}
    all_components = [
        _component_payload(
            component,
            metrics,
            selected=component.name in selected_names,
        )
        for component in profile.components
    ]
    selected_components = [
        _component_payload(component, metrics, selected=True) for component in selected_specs
    ]
    weight_sum = sum(item["weight"] for item in selected_components) or 1.0
    objective_score = (
        0.0
        if rejected
        else _clip(sum(item["weighted_value"] for item in selected_components) / weight_sum)
    )
    selected_components = [
        {
            **item,
            "renormalized_weight": item["weight"] / weight_sum,
            "renormalized_weighted_value": item["weighted_value"] / weight_sum,
        }
        for item in selected_components
    ]
    missing = [
        component.normalizer.metric
        for component in selected_specs
        if component.normalizer.metric not in metrics
    ]
    return ScoreResult(
        objective_score=objective_score,
        profile=profile,
        components=all_components,
        selected_components=selected_components,
        rejected=rejected,
        reject_reasons=reject_reasons,
        metrics={key: _finite(value) for key, value in sorted(metrics.items())},
        component_cap=cap,
        missing_components=sorted(set(missing)),
    )


def replay_metrics(
    *,
    trades: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    window: WindowSpec,
    net_return: float,
    max_drawdown: float,
    profit_factor: float,
) -> dict[str, float]:
    returns = [_finite(trade.get("return_pct", 0.0)) for trade in trades]
    total_return = sum(returns)
    trade_count = len(returns)
    wins = [value for value in returns if value > 0.0]
    losses = [-value for value in returns if value < 0.0]
    avg_return = mean(returns) if returns else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    max_abs_return = max([abs(value) for value in returns], default=0.0)
    effective_profit_factor = _effective_profit_factor(
        profit_factor=profit_factor,
        wins=wins,
        losses=losses,
    )
    months = _window_months(window.start, window.end)
    coverage_rows = _coverage_rows(coverage)
    covered_symbols = {
        str(row.get("symbol", "")).upper()
        for row in coverage
        if isinstance(row, dict) and row.get("symbol")
    }
    traded_symbols = [
        str(trade.get("symbol", "")).upper()
        for trade in trades
        if isinstance(trade, dict) and trade.get("symbol")
    ]
    symbol_balance = _balance_score(traded_symbols)
    active_symbol_ratio = (
        len(set(traded_symbols)) / len(covered_symbols) if covered_symbols else float(bool(trades))
    )
    win_rate = len(wins) / trade_count if trade_count else 0.0
    sharpe = _sharpe(returns)
    max_drawdown_pct = abs(_finite(max_drawdown)) * 100.0
    total_return_pct = total_return * 100.0
    avg_return_pct = avg_return * 100.0
    calmar = total_return_pct / max(max_drawdown_pct, 0.01)
    risk_quality = _clip(1.0 - max_drawdown_pct / 20.0)
    pf_quality = _clip((effective_profit_factor - 1.0) / 3.0)
    expectancy_quality = _clip(avg_return_pct / 1.0)
    frequency_quality = _clip(trade_count / max(1.0, len(covered_symbols) or 1.0))
    capture_quality = _clip(
        0.45 * win_rate
        + 0.25 * (avg_win / max(max_abs_return, 0.000001))
        + 0.20 * pf_quality
        + 0.10 * risk_quality
    )
    edge_quality = _clip(0.55 * pf_quality + 0.45 * expectancy_quality)
    stability = _clip(0.55 * (sharpe / 3.0) + 0.45 * risk_quality)
    process_quality = _clip(trade_count / max(coverage_rows, 1) * 50.0)
    balance_quality = _clip(0.65 * symbol_balance + 0.35 * active_symbol_ratio)
    selection_quality = _clip(0.55 * edge_quality + 0.45 * capture_quality)
    signal_quality = _clip(0.50 * edge_quality + 0.25 * frequency_quality + 0.25 * stability)
    false_positive_control = _clip(0.65 * edge_quality + 0.35 * risk_quality)
    return {
        "active_symbol_ratio": _finite(active_symbol_ratio),
        "avg_loss_pct": avg_loss * 100.0,
        "avg_return_pct": avg_return_pct,
        "avg_r_proxy": avg_return_pct,
        "avg_win_pct": avg_win * 100.0,
        "balance_quality_proxy": balance_quality,
        "calmar": calmar,
        "capture_quality_proxy": capture_quality,
        "coverage_ratio": _clip(active_symbol_ratio),
        "drawdown_resilience_proxy": risk_quality,
        "edge_quality_proxy": edge_quality,
        "entry_quality_proxy": _clip(0.60 * win_rate + 0.40 * capture_quality),
        "exit_efficiency_proxy": capture_quality,
        "expectancy_pct": avg_return_pct,
        "false_positive_control_proxy": false_positive_control,
        "frequency_quality_proxy": frequency_quality,
        "max_drawdown_pct": max_drawdown_pct,
        "net_return": _finite(net_return),
        "net_return_pct": total_return_pct,
        "official_mtm_net_return_pct": total_return_pct,
        "process_quality_proxy": process_quality,
        "profit_factor": _finite(profit_factor),
        "profit_factor_effective": effective_profit_factor,
        "rule_efficiency_proxy": process_quality,
        "selection_quality_proxy": selection_quality,
        "sharpe": sharpe,
        "signal_quality_proxy": signal_quality,
        "stability_proxy": stability,
        "strategy_balance_proxy": balance_quality,
        "total_return_pct": total_return_pct,
        "total_r_per_month": total_return_pct / months,
        "trade_count": float(trade_count),
        "trades_per_month": trade_count / months,
        "win_rate": win_rate,
        "winning_trades": float(len(wins)),
    }


def component_names_for_profile(profile: ScoreProfile, cap: int) -> list[str]:
    return [component.name for component in profile.capped_components(cap)]


def compact_score_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    reject_reasons = payload.get("reject_reasons", [])
    if not isinstance(reject_reasons, list):
        reject_reasons = [str(reject_reasons)] if reject_reasons else []
    components = payload.get("renormalized_components", [])
    if not isinstance(components, list):
        components = []
    return {
        "profile_id": str(payload.get("profile_id") or ""),
        "profile_version": str(payload.get("profile_version") or ""),
        "objective_score": _finite(payload.get("objective_score", 0.0)),
        "rejected": bool(payload.get("rejected", False)),
        "reject_reasons": reject_reasons,
        "score_component_cap": int(_finite(payload.get("score_component_cap", 0))),
        "renormalized_components": components,
    }


def _component_payload(
    component: ComponentSpec,
    metrics: dict[str, float],
    *,
    selected: bool,
) -> dict[str, Any]:
    raw_value = _finite(metrics.get(component.normalizer.metric, 0.0))
    normalized = component.normalizer.normalize(metrics)
    return {
        "component": component.name,
        "metric": component.normalizer.metric,
        "raw_value": raw_value,
        "normalized_value": normalized,
        "weight": component.weight,
        "weighted_value": component.weight * normalized,
        "selected": selected,
        "normalizer": component.normalizer.to_payload(),
        "source_components": list(component.source_components or (component.name,)),
    }


def _profile_id_for_family_strategy(family_key: str, strategy_key: str) -> str:
    if family_key == "crypto_portfolio":
        if "breakout" in strategy_key:
            return "crypto.single.breakout"
        if "momentum" in strategy_key:
            return "crypto.single.momentum"
        if "portfolio" in strategy_key:
            return "crypto.portfolio"
        return "crypto.single.trend"
    if family_key == "k_stock_olr_kalcb":
        if "portfolio" in strategy_key or "synergy" in strategy_key:
            return "k_stock.portfolio"
        if "kalcb" in strategy_key and "olr" not in strategy_key:
            return "k_stock.kalcb"
        if "olr" in strategy_key and "kalcb" not in strategy_key:
            return "k_stock.olr"
        return "k_stock.olr_kalcb"
    if family_key == "trading_stock_family":
        if "portfolio" in strategy_key or "synergy" in strategy_key:
            return "trading.stock.portfolio"
        if "iaric" in strategy_key:
            return "trading.stock.iaric"
        return "trading.stock.alcb" if "alcb" in strategy_key else "trading.stock.family"
    if family_key == "trading_momentum_family":
        if "portfolio" in strategy_key or "synergy" in strategy_key:
            return "trading.momentum.portfolio"
        if "downturn" in strategy_key:
            return "trading.momentum.downturn"
        if "nqdtc" in strategy_key:
            return "trading.momentum.nqdtc"
        if "regime" in strategy_key:
            return "trading.momentum.nq_regime"
        if "vdubus" in strategy_key:
            return "trading.momentum.vdubus"
        return "trading.momentum.family"
    if family_key == "trading_swing_family":
        if "portfolio" in strategy_key or "synergy" in strategy_key:
            return "trading.swing.portfolio"
        if "helix" in strategy_key:
            return "trading.swing.helix"
        if "tpc" in strategy_key:
            return "trading.swing.tpc"
        if "atrss" in strategy_key:
            return "trading.swing.atrss"
        return "trading.swing.family"
    if "crisis" in strategy_key or "regime" in family_key:
        return "trading.regime.crisis"
    return "generic.pnl"


def _hard_rejects(
    max_drawdown_pct: float,
    *,
    min_trades: float = 1.0,
    min_profit_factor: float = 0.8,
) -> tuple[HardReject, ...]:
    return (
        HardReject("trade_count", ">=", min_trades, "replay trade count"),
        HardReject(
            "profit_factor_effective",
            ">=",
            min_profit_factor,
            "replay profit factor",
        ),
        HardReject("max_drawdown_pct", "<=", max_drawdown_pct, "replay max drawdown pct"),
    )


def _component(
    name: str,
    weight: float,
    metric: str,
    kind: str,
    ceiling: float,
    floor: float = 0.0,
) -> ComponentSpec:
    return ComponentSpec(
        name=name,
        weight=weight,
        normalizer=Normalizer(metric=metric, kind=kind, floor=floor, ceiling=ceiling),
    )


def _identity(name: str, weight: float, metric: str) -> ComponentSpec:
    return _component(name, weight, metric, "identity", 1.0)


def _profile(
    profile_id: str,
    family: str,
    scope: str,
    components: tuple[ComponentSpec, ...],
    *,
    max_drawdown_pct: float,
    base_profile_id: str = "",
    overlay_id: str = "",
    source_reference: str = "",
    source_round: str = "",
    reference_hard_rejects: tuple[str, ...] = (),
) -> ScoreProfile:
    return ScoreProfile(
        profile_id=profile_id,
        family=family,
        scope=scope,
        components=components,
        hard_rejects=_hard_rejects(max_drawdown_pct),
        base_profile_id=base_profile_id,
        overlay_id=overlay_id,
        source_reference=source_reference,
        source_round=source_round,
        reference_hard_rejects=reference_hard_rejects,
    )


SCORE_PROFILES: dict[str, ScoreProfile] = {
    "generic.pnl": _profile(
        "generic.pnl",
        "generic",
        "strategy",
        (
            _component("expected_return", 0.24, "net_return_pct", "ceiling", 25.0),
            _component("trade_frequency", 0.16, "trade_count", "ceiling", 12.0),
            _component("edge_quality", 0.16, "profit_factor_effective", "floor_ceiling", 3.0, 1.0),
            _component("expectancy", 0.14, "expectancy_pct", "ceiling", 1.0),
            _component("drawdown_resilience", 0.12, "max_drawdown_pct", "inverse_ceiling", 20.0),
            _identity("capture_quality", 0.10, "capture_quality_proxy"),
            _identity("robustness", 0.08, "stability_proxy"),
        ),
        max_drawdown_pct=20.0,
    ),
    "crypto.single.breakout": _profile(
        "crypto.single.breakout",
        "crypto_portfolio",
        "strategy",
        (
            _component("returns", 0.22, "net_return_pct", "ceiling", 45.0),
            _component("coverage", 0.20, "trade_count", "ceiling", 26.0),
            _component("expectancy", 0.16, "expectancy_pct", "ceiling", 1.0),
            _component("edge", 0.13, "profit_factor_effective", "floor_ceiling", 9.0, 1.0),
            _identity("capture", 0.12, "capture_quality_proxy"),
            _component("sharpe", 0.10, "sharpe", "ceiling", 5.0),
            _component("risk", 0.07, "max_drawdown_pct", "inverse_ceiling", 8.0),
        ),
        max_drawdown_pct=8.0,
        base_profile_id="crypto.single",
        overlay_id="breakout",
        source_reference="backtests/baselines/crypto/breakout/round_3",
        source_round="round_3_copied_from_round_2",
        reference_hard_rejects=(
            "total_trades >= 12",
            "profit_factor >= 2.0",
            "expectancy_r >= 0.25",
            "exit_efficiency >= 0.5",
            "max_drawdown_pct <= 8",
        ),
    ),
    "crypto.single.momentum": _profile(
        "crypto.single.momentum",
        "crypto_portfolio",
        "strategy",
        (
            _component("returns", 0.24, "net_return_pct", "ceiling", 20.0),
            _component("coverage", 0.18, "trade_count", "ceiling", 42.0),
            _component("expectancy", 0.16, "expectancy_pct", "ceiling", 0.65),
            _component("edge", 0.13, "profit_factor_effective", "floor_ceiling", 4.5, 1.0),
            _identity("capture", 0.12, "capture_quality_proxy"),
            _identity("entry_quality", 0.09, "entry_quality_proxy"),
            _component("risk", 0.08, "max_drawdown_pct", "inverse_ceiling", 8.0),
        ),
        max_drawdown_pct=8.0,
        base_profile_id="crypto.single",
        overlay_id="momentum",
        source_reference="backtests/baselines/crypto/momentum/round_3",
        source_round="round_3_copied_from_round_2",
        reference_hard_rejects=(
            "total_trades >= 18",
            "profit_factor >= 1.5",
            "expectancy_r >= 0.2",
            "exit_efficiency >= 0.42",
            "avg_mae_r >= -0.45",
            "max_drawdown_pct <= 8",
        ),
    ),
    "crypto.single.trend": _profile(
        "crypto.single.trend",
        "crypto_portfolio",
        "strategy",
        (
            _component("returns", 0.22, "net_return_pct", "ceiling", 85.0),
            _component("coverage", 0.17, "trade_count", "ceiling", 85.0),
            _component("edge", 0.16, "profit_factor_effective", "floor_ceiling", 5.0, 1.0),
            _component("expectancy", 0.14, "expectancy_pct", "ceiling", 0.60),
            _identity("capture", 0.13, "capture_quality_proxy"),
            _identity("entry_quality", 0.10, "entry_quality_proxy"),
            _component("risk", 0.08, "max_drawdown_pct", "inverse_ceiling", 12.0),
        ),
        max_drawdown_pct=12.0,
        base_profile_id="crypto.single",
        overlay_id="trend",
        source_reference="backtests/baselines/crypto/trend/round_3",
        source_round="round_3_copied_from_round_2",
        reference_hard_rejects=(
            "total_trades >= 30",
            "profit_factor >= 1.5",
            "net_return_pct >= 20",
            "expectancy_r >= 0.1",
            "max_drawdown_pct <= 12",
        ),
    ),
    "crypto.portfolio": _profile(
        "crypto.portfolio",
        "crypto_portfolio",
        "portfolio",
        (
            _component("return", 0.30, "net_return_pct", "ceiling", 75.0),
            _component("frequency", 0.20, "trade_count", "ceiling", 58.0),
            _identity("edge_quality", 0.18, "edge_quality_proxy"),
            _identity("capture", 0.10, "capture_quality_proxy"),
            _component("drawdown_resilience", 0.10, "max_drawdown_pct", "inverse_ceiling", 8.5),
            _identity("rule_efficiency", 0.06, "rule_efficiency_proxy"),
            _identity("strategy_balance", 0.06, "strategy_balance_proxy"),
        ),
        max_drawdown_pct=8.5,
        source_reference="backtests/baselines/crypto/portfolio/round_3",
        source_round="round_3_with_round_2_scoring_contract",
        reference_hard_rejects=(
            "total_trades >= 42",
            "profit_factor >= 1.55",
            "expectancy_r >= 0.2",
            "exit_efficiency >= 0.3",
            "max_drawdown_pct <= 8.5",
        ),
    ),
    "trading.momentum.downturn": _profile(
        "trading.momentum.downturn",
        "trading_momentum_family",
        "strategy",
        (
            _component("net_return", 0.20, "net_return_pct", "ceiling", 100.0),
            _component("correction_pnl", 0.18, "total_r_per_month", "ceiling", 80.0),
            _identity("edge", 0.15, "edge_quality_proxy"),
            _identity("alpha_capture", 0.15, "capture_quality_proxy"),
            _component("frequency", 0.12, "trade_count", "ceiling", 160.0),
            _identity("coverage", 0.12, "coverage_ratio"),
            _component("risk", 0.08, "max_drawdown_pct", "inverse_ceiling", 25.0),
        ),
        max_drawdown_pct=25.0,
        base_profile_id="trading.momentum",
        overlay_id="downturn",
        source_reference="backtests/baselines/ibkr/momentum/downturn/round_4",
        source_round="round_4",
    ),
    "trading.momentum.nqdtc": _profile(
        "trading.momentum.nqdtc",
        "trading_momentum_family",
        "strategy",
        (
            _component("returns", 0.22, "net_return_pct", "ceiling", 120.0),
            _component("frequency", 0.18, "trade_count", "ceiling", 155.0),
            _identity("exit_capture", 0.16, "capture_quality_proxy"),
            _component("expectancy", 0.14, "expectancy_pct", "floor_ceiling", 0.55, 0.10),
            _component("pf", 0.12, "profit_factor_effective", "floor_ceiling", 2.10, 1.20),
            _component("risk", 0.10, "max_drawdown_pct", "inverse_ceiling", 35.0),
            _identity("stability", 0.08, "stability_proxy"),
        ),
        max_drawdown_pct=35.0,
        base_profile_id="trading.momentum",
        overlay_id="nqdtc",
        source_reference="backtests/baselines/ibkr/momentum/nqdtc/round_4",
        source_round="round_4",
    ),
    "trading.momentum.nq_regime": _profile(
        "trading.momentum.nq_regime",
        "trading_momentum_family",
        "strategy",
        (
            _component("alpha_return", 0.24, "total_r_per_month", "ceiling", 8.8),
            _component("trade_frequency", 0.18, "trades_per_month", "ceiling", 7.8),
            _identity("expectancy_quality", 0.18, "edge_quality_proxy"),
            _identity("component_synergy", 0.18, "strategy_balance_proxy"),
            _identity("module_guardrails", 0.10, "process_quality_proxy"),
            _component("drawdown_robustness", 0.06, "max_drawdown_pct", "inverse_ceiling", 6.0),
            _identity("execution", 0.06, "capture_quality_proxy"),
        ),
        max_drawdown_pct=6.0,
        base_profile_id="trading.momentum",
        overlay_id="nq_regime",
        source_reference="backtests/baselines/ibkr/momentum/nq_regime/round_5",
        source_round="round_5_penultimate_scored",
    ),
    "trading.momentum.vdubus": _profile(
        "trading.momentum.vdubus",
        "trading_momentum_family",
        "strategy",
        (
            _component("r_per_month", 0.28, "total_r_per_month", "ceiling", 3.0),
            _component("pf", 0.18, "profit_factor_effective", "floor_ceiling", 2.8, 1.2),
            _component("calmar", 0.14, "calmar", "ceiling", 6.0),
            _identity("capture", 0.12, "capture_quality_proxy"),
            _component("inv_dd", 0.10, "max_drawdown_pct", "inverse_ceiling", 30.0),
            _component("frequency", 0.10, "trades_per_month", "floor_ceiling", 8.5, 4.0),
            _component("sharpe", 0.08, "sharpe", "ceiling", 2.4),
        ),
        max_drawdown_pct=30.0,
        base_profile_id="trading.momentum",
        overlay_id="vdubus",
        source_reference="backtests/baselines/ibkr/momentum/vdubus/round_3",
        source_round="round_3",
    ),
    "trading.momentum.family": _profile(
        "trading.momentum.family",
        "trading_momentum_family",
        "family",
        (
            _component("alpha_return", 0.24, "net_return_pct", "ceiling", 100.0),
            _component("trade_frequency", 0.18, "trade_count", "ceiling", 120.0),
            _identity("edge_quality", 0.18, "edge_quality_proxy"),
            _identity("capture_quality", 0.14, "capture_quality_proxy"),
            _component("drawdown_robustness", 0.10, "max_drawdown_pct", "inverse_ceiling", 25.0),
            _identity("stability", 0.08, "stability_proxy"),
            _identity("execution", 0.08, "process_quality_proxy"),
        ),
        max_drawdown_pct=25.0,
        source_reference="backtests/baselines/ibkr/momentum",
        source_round="latest_strategy_overlay_or_family_default",
    ),
    "trading.momentum.portfolio": _profile(
        "trading.momentum.portfolio",
        "trading_momentum_family",
        "portfolio",
        (
            _component("expected_return", 0.24, "net_return_pct", "ceiling", 220.0),
            _component("trade_frequency", 0.18, "trades_per_month", "ceiling", 40.0),
            _component("drawdown_control", 0.18, "max_drawdown_pct", "inverse_ceiling", 20.0),
            _identity("profit_quality", 0.13, "edge_quality_proxy"),
            _component("risk_efficiency", 0.12, "calmar", "ceiling", 8.0),
            _identity("strategy_balance", 0.10, "strategy_balance_proxy"),
            _identity("live_rule_health", 0.05, "rule_efficiency_proxy"),
        ),
        max_drawdown_pct=20.0,
        source_reference="backtests/baselines/ibkr/momentum/portfolio_synergy/round_2",
        source_round="round_2",
    ),
    "trading.regime.crisis": _profile(
        "trading.regime.crisis",
        "trading_regime_family",
        "strategy",
        (
            _identity("fp_control", 0.20, "false_positive_control_proxy"),
            _identity("detection_speed", 0.17, "signal_quality_proxy"),
            _identity("early_action_quality", 0.17, "selection_quality_proxy"),
            _identity("coverage", 0.14, "coverage_ratio"),
            _identity("severity", 0.11, "edge_quality_proxy"),
            _identity("stability_calibration", 0.11, "stability_proxy"),
            _identity("recovery_preaction_quality", 0.10, "drawdown_resilience_proxy"),
        ),
        max_drawdown_pct=10.0,
        source_reference="backtests/baselines/ibkr/regime/crisis/round_9",
        source_round="round_9",
        reference_hard_rejects=(
            "warning false-positive rate <= 10%",
            "crisis false-positive rate <= 5%",
            "crises_detected >= 7",
        ),
    ),
    "trading.stock.alcb": _profile(
        "trading.stock.alcb",
        "trading_stock_family",
        "strategy",
        (
            _component("expected_total_r", 0.24, "net_return_pct", "floor_ceiling", 155.0, 115.0),
            _component("trades_per_month", 0.20, "trades_per_month", "floor_ceiling", 24.5, 19.0),
            _identity("edge_quality", 0.25, "edge_quality_proxy"),
            _component("net_profit", 0.10, "net_return_pct", "ceiling", 125.0),
            _identity("profit_protection", 0.09, "drawdown_resilience_proxy"),
            _identity("signal_quality", 0.07, "signal_quality_proxy"),
            _identity("timing_capture_quality", 0.05, "capture_quality_proxy"),
        ),
        max_drawdown_pct=12.0,
        base_profile_id="trading.stock",
        overlay_id="alcb",
        source_reference="backtests/baselines/ibkr/stock/alcb/round_2",
        source_round="round_2",
    ),
    "trading.stock.iaric": _profile(
        "trading.stock.iaric",
        "trading_stock_family",
        "strategy",
        (
            _component("expected_total_r", 0.30, "net_return_pct", "ceiling", 40.0),
            _component(
                "profit_factor",
                0.20,
                "profit_factor_effective",
                "floor_ceiling",
                2.0,
                1.15,
            ),
            _component("sharpe", 0.18, "sharpe", "ceiling", 1.0),
            _component("inv_dd", 0.14, "max_drawdown_pct", "inverse_ceiling", 8.0),
            _component("total_trades", 0.12, "trade_count", "ceiling", 500.0),
            _component("avg_r", 0.06, "avg_r_proxy", "ceiling", 0.20),
        ),
        max_drawdown_pct=8.0,
        base_profile_id="trading.stock",
        overlay_id="iaric",
        source_reference="backtests/baselines/ibkr/stock/iaric/round_1",
        source_round="round_1_live_aligned_ablation",
    ),
    "trading.stock.family": _profile(
        "trading.stock.family",
        "trading_stock_family",
        "family",
        (
            _component("official_return", 0.24, "net_return_pct", "ceiling", 60.0),
            _component("expected_total_r", 0.18, "total_r_per_month", "ceiling", 20.0),
            _component("trade_frequency", 0.16, "trades_per_month", "ceiling", 20.0),
            _identity("edge_quality", 0.16, "edge_quality_proxy"),
            _component("drawdown_resilience", 0.11, "max_drawdown_pct", "inverse_ceiling", 12.0),
            _identity("capture_quality", 0.08, "capture_quality_proxy"),
            _identity("signal_quality", 0.07, "signal_quality_proxy"),
        ),
        max_drawdown_pct=12.0,
        source_reference="backtests/baselines/ibkr/stock",
        source_round="latest_strategy_overlay_or_family_default",
    ),
    "trading.stock.portfolio": _profile(
        "trading.stock.portfolio",
        "trading_stock_family",
        "portfolio",
        (
            _component("alpha_return", 0.27, "net_return_pct", "ceiling", 250.0),
            _component("trade_frequency", 0.23, "trades_per_month", "ceiling", 60.0),
            _component("drawdown_control", 0.19, "max_drawdown_pct", "inverse_ceiling", 20.0),
            _identity("profit_factor_quality", 0.12, "edge_quality_proxy"),
            _identity("synergy_capture", 0.08, "capture_quality_proxy"),
            _identity("allocation_balance", 0.06, "strategy_balance_proxy"),
            _identity("robustness", 0.05, "stability_proxy"),
        ),
        max_drawdown_pct=20.0,
        source_reference="backtests/baselines/ibkr/stock/portfolio_synergy/round_2",
        source_round="round_2",
    ),
    "trading.swing.helix": _profile(
        "trading.swing.helix",
        "trading_swing_family",
        "strategy",
        (
            _component("net_profit", 0.43, "net_return_pct", "floor_ceiling", 150.0, 55.0),
            _component("win_rate", 0.18, "win_rate", "floor_ceiling", 0.55, 0.45),
            _component("winning_trades", 0.11, "winning_trades", "floor_ceiling", 230.0, 150.0),
            _component("pf", 0.10, "profit_factor_effective", "floor_ceiling", 3.5, 1.2),
            _identity("exit_quality", 0.08, "capture_quality_proxy"),
            _component("frequency", 0.06, "trade_count", "floor_ceiling", 420.0, 300.0),
            _component("inv_dd", 0.04, "max_drawdown_pct", "inverse_ceiling", 14.0),
        ),
        max_drawdown_pct=14.0,
        base_profile_id="trading.swing",
        overlay_id="helix",
        source_reference="backtests/baselines/ibkr/swing/helix/round_4",
        source_round="round_4_penultimate_scored",
    ),
    "trading.swing.tpc": _profile(
        "trading.swing.tpc",
        "trading_swing_family",
        "strategy",
        (
            _identity("alpha_quality", 0.28, "edge_quality_proxy"),
            _identity("false_positive_control", 0.22, "false_positive_control_proxy"),
            _identity("risk_quality", 0.17, "drawdown_resilience_proxy"),
            _component("frequency_floor", 0.14, "trades_per_month", "ceiling", 10.0),
            _identity("symbol_balance", 0.11, "strategy_balance_proxy"),
            _identity("stability", 0.08, "stability_proxy"),
        ),
        max_drawdown_pct=16.0,
        base_profile_id="trading.swing",
        overlay_id="tpc",
        source_reference="backtests/baselines/ibkr/swing/tpc/round_8",
        source_round="round_8",
    ),
    "trading.swing.atrss": _profile(
        "trading.swing.atrss",
        "trading_swing_family",
        "strategy",
        (
            _component("return_quality", 0.24, "net_return_pct", "ceiling", 120.0),
            _identity("edge_quality", 0.18, "edge_quality_proxy"),
            _component("win_rate_quality", 0.16, "win_rate", "ceiling", 0.60),
            _component("frequency_quality", 0.14, "trades_per_month", "ceiling", 8.0),
            _identity("drawdown_quality", 0.12, "drawdown_resilience_proxy"),
            _identity("capture_quality", 0.10, "capture_quality_proxy"),
            _identity("robustness_quality", 0.06, "stability_proxy"),
        ),
        max_drawdown_pct=16.0,
        base_profile_id="trading.swing",
        overlay_id="atrss",
        source_reference="backtests/baselines/ibkr/swing/atrss/round_3",
        source_round="round_3",
    ),
    "trading.swing.family": _profile(
        "trading.swing.family",
        "trading_swing_family",
        "family",
        (
            _component("alpha_quality", 0.26, "net_return_pct", "ceiling", 150.0),
            _component("frequency_quality", 0.18, "trades_per_month", "ceiling", 10.0),
            _identity("edge_quality", 0.16, "edge_quality_proxy"),
            _identity("capture_quality", 0.14, "capture_quality_proxy"),
            _identity("drawdown_quality", 0.12, "drawdown_resilience_proxy"),
            _identity("balance_quality", 0.08, "strategy_balance_proxy"),
            _identity("robustness_quality", 0.06, "stability_proxy"),
        ),
        max_drawdown_pct=16.0,
        source_reference="backtests/baselines/ibkr/swing",
        source_round="latest_strategy_overlay_or_family_default",
    ),
    "trading.swing.portfolio": _profile(
        "trading.swing.portfolio",
        "trading_swing_family",
        "portfolio",
        (
            _component("alpha_quality", 0.30, "net_return_pct", "ceiling", 520.0),
            _component("frequency_quality", 0.24, "trade_count", "ceiling", 620.0),
            _component("drawdown_quality", 0.16, "max_drawdown_pct", "inverse_ceiling", 16.0),
            _identity("balance_quality", 0.10, "strategy_balance_proxy"),
            _component("pf_quality", 0.09, "profit_factor_effective", "floor_ceiling", 3.8, 2.2),
            _identity("capture_quality", 0.07, "capture_quality_proxy"),
            _identity("robustness_quality", 0.04, "stability_proxy"),
        ),
        max_drawdown_pct=16.0,
        source_reference="backtests/baselines/ibkr/swing/portfolio_synergy/round_3",
        source_round="round_3",
    ),
    "k_stock.kalcb": _profile(
        "k_stock.kalcb",
        "k_stock_olr_kalcb",
        "strategy",
        (
            _component(
                "official_mtm_net_return_pct",
                0.24,
                "official_mtm_net_return_pct",
                "ceiling",
                1.0,
            ),
            _component("expected_total_r", 0.22, "total_r_per_month", "ceiling", 10.0),
            _component(
                "profit_factor",
                0.16,
                "profit_factor_effective",
                "floor_ceiling",
                2.25,
                1.05,
            ),
            _component("avg_r", 0.14, "avg_r_proxy", "floor_ceiling", 0.20, 0.02),
            _component("entry_count", 0.12, "trade_count", "floor_ceiling", 75.0, 20.0),
            _identity("mfe_capture", 0.06, "capture_quality_proxy"),
            _component("max_drawdown_pct", 0.06, "max_drawdown_pct", "inverse_ceiling", 1.2),
        ),
        max_drawdown_pct=1.2,
        base_profile_id="k_stock.single",
        overlay_id="kalcb",
        source_reference="bots/k_stock_trader/backtests/strategies/kalcb/phase_scoring.py",
        source_round="source_only_no_output_rounds_in_checkout",
    ),
    "k_stock.olr": _profile(
        "k_stock.olr",
        "k_stock_olr_kalcb",
        "strategy",
        (
            _component(
                "official_mtm_net_return_pct",
                0.24,
                "official_mtm_net_return_pct",
                "ceiling",
                1.5,
            ),
            _identity("olr_alpha_capture", 0.16, "capture_quality_proxy"),
            _component("expected_total_r", 0.15, "total_r_per_month", "ceiling", 65.0),
            _component("total_trades", 0.13, "trade_count", "ceiling", 320.0),
            _component("max_drawdown_pct", 0.12, "max_drawdown_pct", "inverse_ceiling", 18.0),
            _identity("olr_discrimination_quality", 0.12, "selection_quality_proxy"),
            _component("profit_factor", 0.08, "profit_factor_effective", "ceiling", 3.0),
        ),
        max_drawdown_pct=18.0,
        base_profile_id="k_stock.single",
        overlay_id="olr",
        source_reference="bots/k_stock_trader/backtests/strategies/olr/phase_scoring.py",
        source_round="source_only_no_output_rounds_in_checkout",
    ),
    "k_stock.olr_kalcb": _profile(
        "k_stock.olr_kalcb",
        "k_stock_olr_kalcb",
        "family",
        (
            _component("official_mtm_return", 0.24, "official_mtm_net_return_pct", "ceiling", 1.25),
            _component("expected_total_r", 0.18, "total_r_per_month", "ceiling", 35.0),
            _component("trade_frequency", 0.14, "trade_count", "ceiling", 180.0),
            _identity("edge_quality", 0.14, "edge_quality_proxy"),
            _component("drawdown_control", 0.12, "max_drawdown_pct", "inverse_ceiling", 10.0),
            _identity("alpha_capture", 0.10, "capture_quality_proxy"),
            _identity("selection_quality", 0.08, "selection_quality_proxy"),
        ),
        max_drawdown_pct=10.0,
        source_reference="bots/k_stock_trader/backtests/strategies/kalcb+olr",
        source_round="source_only_no_output_rounds_in_checkout",
    ),
    "k_stock.portfolio": _profile(
        "k_stock.portfolio",
        "k_stock_olr_kalcb",
        "portfolio",
        (
            _component(
                "capital_normalized_mtm_return",
                0.24,
                "official_mtm_net_return_pct",
                "ceiling",
                1.5,
            ),
            _component("trade_frequency", 0.17, "trades_per_month", "ceiling", 21.0),
            _component("risk_normalized_total_r", 0.16, "total_r_per_month", "ceiling", 35.0),
            _identity("block_selectivity", 0.15, "selection_quality_proxy"),
            _component("drawdown_control", 0.11, "max_drawdown_pct", "inverse_ceiling", 10.0),
            _identity("profit_quality", 0.10, "edge_quality_proxy"),
            _identity("robust_balance", 0.07, "strategy_balance_proxy"),
        ),
        max_drawdown_pct=10.0,
        source_reference="bots/k_stock_trader/backtests/strategies/portfolio_synergy/phase_scoring.py",
        source_round="source_only_no_output_rounds_in_checkout",
    ),
}

_PROFILE_ID_BY_PLUGIN = {
    "crypto_breakout_v1": "crypto.single.breakout",
    "crypto_momentum_v1": "crypto.single.momentum",
    "crypto_trend_v1": "crypto.single.trend",
}

_FAMILY_BY_PLUGIN = {
    "crypto_breakout_v1": "crypto_portfolio",
    "crypto_momentum_v1": "crypto_portfolio",
    "crypto_trend_v1": "crypto_portfolio",
    "k_stock_olr_kalcb": "k_stock_olr_kalcb",
    "trading_stock_family": "trading_stock_family",
    "trading_momentum_family": "trading_momentum_family",
    "trading_swing_family": "trading_swing_family",
}


def _coverage_rows(coverage: list[dict[str, Any]]) -> int:
    return sum(
        int(row.get("rows", 0) or 0)
        for row in coverage
        if isinstance(row, dict)
    )


def _effective_profit_factor(
    *,
    profit_factor: float,
    wins: list[float],
    losses: list[float],
) -> float:
    if losses:
        return _finite(sum(wins) / sum(losses) if sum(losses) else profit_factor)
    if wins:
        return 99.99
    return _finite(profit_factor)


def _balance_score(values: list[str]) -> float:
    clean = [value for value in values if value]
    if not clean:
        return 0.0
    counts = {value: clean.count(value) for value in set(clean)}
    if len(counts) == 1:
        return 1.0
    total = len(clean)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    max_entropy = math.log(len(counts)) or 1.0
    min_share = min(counts.values()) / total
    return _clip(0.75 * (entropy / max_entropy) + 0.25 * (min_share * len(counts)))


def _sharpe(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    deviation = pstdev(returns)
    if deviation <= 0.0:
        return 0.0
    return _finite((mean(returns) / deviation) * math.sqrt(len(returns)))


def _window_months(start: date, end: date) -> float:
    days = max(1, (end - start).days + 1)
    return max(days / 30.4375, 1.0 / 30.4375)


def _key(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))
