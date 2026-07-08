"""Cross-strategy portfolio risk rules for live trading.

Implements the rules from PortfolioConfig v7:
  1. NQDTC direction filter (affects Vdubus sizing)
  2. Directional cap (max same-direction risk; asymmetric long/short supported)
  2a. Family contract cap (MNQ-equivalent ceiling across momentum family)
  2b. Symbol collision guard (stock family: block/reduce when sibling holds same ticker)
  3. Drawdown tiers (size reduction as DD increases)

These rules query the shared `strategy_signals` and `positions` tables
to coordinate across independently-running strategy containers.
Used by momentum family and stock family.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Callable, Awaitable
from zoneinfo import ZoneInfo
from libs.oms.instrumentation.portfolio_rule_event import build_portfolio_rule_event
from libs.instrumentation.lineage import stable_hash

logger = logging.getLogger(__name__)
_RULE_EVENT_CONTEXT: ContextVar[dict | None] = ContextVar(
    "portfolio_rule_event_context",
    default=None,
)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PortfolioRulesConfig:
    """Live portfolio rules matching PortfolioConfig v7."""

    cooldown_session_only: bool = True
    nqdtc_strategy_id: str = "NQDTC_v2.1"

    # NQDTC direction filter (affects Vdubus)
    nqdtc_direction_filter_enabled: bool = True
    nqdtc_agree_size_mult: float = 1.50
    nqdtc_oppose_size_mult: float = 0.0  # 0 = block
    vdubus_strategy_id: str = "VdubusNQ_v4"

    # Directional cap (symmetric fallback)
    directional_cap_R: float = 3.5               # raised from 2.5 to match heat_cap
    # Asymmetric directional caps (0.0 = use symmetric directional_cap_R)
    directional_cap_long_R: float = 0.0
    directional_cap_short_R: float = 0.0

    # Drawdown tiers: (dd_pct_threshold, size_multiplier)
    # Applied in order: first tier where current_dd < threshold wins
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.08, 1.00),  # < 8%: full size
        (0.12, 0.50),  # 8-12%: half size
        (0.15, 0.25),  # 12-15%: quarter size
        (1.00, 0.00),  # > 15%: halt
    )
    initial_equity: float = 10_000.0

    # Family-scoped rules (stock family)
    family_strategy_ids: tuple[str, ...] = ()  # if set, scope directional cap to these IDs
    symbol_collision_action: str = "none"       # "none", "block", "half_size"

    # Per-pair symbol collision overrides: (holder_id, requester_id, action)
    # When holder has position on symbol, apply action to requester instead of default.
    # Checked before the generic symbol_collision_action fallback.
    symbol_collision_pairs: tuple[tuple[str, str, str], ...] = ()

    # Dynamic family contract cap (0 = disabled)
    max_family_contracts_mnq_eq: int = 0

    # Strategy priority for directional cap (lower number = higher priority)
    strategy_priorities: tuple[tuple[str, int], ...] = ()
    # When remaining directional cap <= this value, only strategies with
    # priority <= priority_reserve_threshold may enter
    priority_headroom_R: float = 0.0       # 0 = disabled (backward compatible)
    priority_reserve_threshold: int = 0

    # Dollar-based directional cap: when > 0, directional cap is checked in
    # dollars (cap_R * reference_urd) instead of R units.  Fixes mixed-R-unit
    # families where strategies have different URDs (e.g. momentum: $200 vs $50).
    reference_unit_risk_dollars: float = 0.0
    reference_unit_risk_pct: float = 0.0

    # Stock portfolio-synergy caps. Defaults are disabled so existing families
    # keep their previous behaviour unless they opt in.
    max_total_active_positions: int = 0
    max_symbol_heat_R: float = 0.0
    same_sector_heat_cap_R: float = 0.0
    symbol_sector_map: tuple[tuple[str, str], ...] = ()
    max_single_strategy_trade_share: float = 1.0
    strategy_trade_share_min_total: int = 50
    dynamic_allocation_enabled: bool = False
    dynamic_lookback_trades: int = 60
    dynamic_min_mult: float = 0.65
    dynamic_max_mult: float = 1.22
    dynamic_positive_expectancy_boost: float = 0.10
    dynamic_negative_expectancy_cut: float = 0.18

    # Momentum portfolio-synergy dynamic sizing. These mirror the optimized
    # replay policy but default to no-op so other families are unaffected.
    max_strategy_active_positions: tuple[tuple[str, int], ...] = ()
    max_strategy_heat_R: tuple[tuple[str, float], ...] = ()
    strategy_size_multipliers: tuple[tuple[str, float], ...] = ()
    existing_position_mult: float = 1.0
    portfolio_heat_cap_R: float = 0.0
    heat_pressure_threshold: float = 1.0
    heat_pressure_mult: float = 1.0
    same_direction_pressure_threshold: float = 1.0
    same_direction_pressure_mult: float = 1.0
    max_trade_risk_R: float = 0.0
    min_qty: int = 1
    fit_to_remaining_heat: bool = False
    fit_to_remaining_directional_cap: bool = False
    fit_to_remaining_family_cap: bool = False

    # Regime-driven sizing scalars (applied as multipliers in check_entry)
    regime_unit_risk_mult: float = 1.0
    regime_unit_risk_long_mult: float = 1.0
    regime_unit_risk_short_mult: float = 1.0
    # Strategies blocked from new entries by regime (checked first in check_entry)
    disabled_strategies: frozenset[str] = frozenset()

    _VALID_COLLISION_ACTIONS = frozenset({"none", "block", "half_size"})
    _VALID_PAIR_ACTIONS = frozenset({"block", "half_size"})

    def __post_init__(self):
        if self.symbol_collision_action not in self._VALID_COLLISION_ACTIONS:
            raise ValueError(
                f"Invalid symbol_collision_action {self.symbol_collision_action!r}, "
                f"must be one of {sorted(self._VALID_COLLISION_ACTIONS)}"
            )
        for holder, requester, action in self.symbol_collision_pairs:
            if action not in self._VALID_PAIR_ACTIONS:
                raise ValueError(
                    f"Invalid pair action {action!r} for ({holder}, {requester}), "
                    f"must be one of {sorted(self._VALID_PAIR_ACTIONS)}"
                )


# ── Result ────────────────────────────────────────────────────────────

@dataclass
class PortfolioRuleResult:
    """Result of portfolio rule checks."""
    approved: bool = True
    denial_reason: Optional[str] = None
    size_multiplier: float = 1.0  # Applied to position size
    requested_qty: Optional[int] = None
    approved_qty: Optional[int] = None
    requested_risk_R: Optional[float] = None
    approved_risk_R: Optional[float] = None
    requested_risk_dollars: Optional[float] = None
    approved_risk_dollars: Optional[float] = None
    rule_trace_id: str = ""
    applied_rules: tuple[str, ...] = ()
    lineage_gap: bool = False


# ── Checker ───────────────────────────────────────────────────────────

class PortfolioRuleChecker:
    """Checks cross-strategy portfolio rules using shared DB state.

    Each strategy container creates one of these at startup and calls
    check_entry() before submitting orders.
    """

    def __init__(
        self,
        config: PortfolioRulesConfig,
        get_strategy_signal: Callable[[str], Awaitable[Optional[dict]]],
        get_directional_risk_R: Callable[[str], Awaitable[float]],
        get_current_equity: Callable[[], float],
        on_rule_event: Optional[Callable[[dict], None]] = None,
        get_directional_risk_R_for_strategies: Optional[
            Callable[[str, list[str]], Awaitable[float]]
        ] = None,
        get_sibling_positions_for_symbol: Optional[
            Callable[[list[str], str], Awaitable[bool]]
        ] = None,
        get_family_aggregate_mnq_eq: Optional[
            Callable[[list[str]], Awaitable[int]]
        ] = None,
        get_directional_risk_dollars_for_strategies: Optional[
            Callable[[str, list[str]], Awaitable[float]]
        ] = None,
        get_open_position_count_for_strategies: Optional[
            Callable[[list[str]], Awaitable[int]]
        ] = None,
        get_symbol_open_risk_dollars_for_strategies: Optional[
            Callable[[list[str], str], Awaitable[float]]
        ] = None,
        get_symbols_open_risk_dollars_for_strategies: Optional[
            Callable[[list[str], list[str]], Awaitable[float]]
        ] = None,
        get_active_risk_dollars_for_strategies: Optional[
            Callable[[list[str]], Awaitable[float]]
        ] = None,
        get_completed_trade_counts_for_strategies: Optional[
            Callable[[list[str]], Awaitable[dict[str, int]]]
        ] = None,
        get_recent_strategy_r_multiples: Optional[
            Callable[[str, int], Awaitable[list[float]]]
        ] = None,
        now_provider: Optional[Callable[[], datetime]] = None,
        on_config_update: Optional[Callable[[PortfolioRulesConfig], None]] = None,
    ):
        self._cfg = config
        self._get_signal = get_strategy_signal
        self._get_equity = get_current_equity
        self._on_rule_event = on_rule_event
        self._get_sibling = get_sibling_positions_for_symbol
        self._get_family_mnq_eq = get_family_aggregate_mnq_eq
        self._get_open_position_count = get_open_position_count_for_strategies
        self._get_symbol_risk_dollars = get_symbol_open_risk_dollars_for_strategies
        self._get_symbols_risk_dollars = get_symbols_open_risk_dollars_for_strategies
        self._get_active_risk_dollars = get_active_risk_dollars_for_strategies
        self._get_trade_counts = get_completed_trade_counts_for_strategies
        self._get_recent_r = get_recent_strategy_r_multiples
        self._now_provider = now_provider
        self._on_config_update = on_config_update
        self._symbol_sector_map = dict(config.symbol_sector_map)
        self._strategy_size_multipliers = dict(config.strategy_size_multipliers)
        self._max_strategy_active_positions = dict(config.max_strategy_active_positions)
        self._max_strategy_heat_R = dict(config.max_strategy_heat_R)

        # Family-scoped directional risk: wrap callback if strategy IDs provided
        family_ids = config.family_strategy_ids
        if family_ids and get_directional_risk_R_for_strategies is not None:
            ids_list = list(family_ids)
            self._get_dir_risk = lambda d: get_directional_risk_R_for_strategies(d, ids_list)
            logger.info("Directional cap scoped to strategies: %s", family_ids)
        else:
            self._get_dir_risk = get_directional_risk_R

        # Dollar-based directional risk (avoids mixed-R-unit problems)
        if family_ids and get_directional_risk_dollars_for_strategies is not None:
            ids_list = list(family_ids)
            self._get_dir_risk_dollars = lambda d: get_directional_risk_dollars_for_strategies(d, ids_list)
        else:
            self._get_dir_risk_dollars = None

    def _now(self) -> datetime:
        now = self._now_provider() if self._now_provider is not None else datetime.now(timezone.utc)
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now

    def update_config(self, new_cfg: PortfolioRulesConfig) -> None:
        """Atomically replace config for regime updates. GIL-safe for single attr assign."""
        self._cfg = new_cfg
        self._symbol_sector_map = dict(new_cfg.symbol_sector_map)
        self._strategy_size_multipliers = dict(new_cfg.strategy_size_multipliers)
        self._max_strategy_active_positions = dict(new_cfg.max_strategy_active_positions)
        self._max_strategy_heat_R = dict(new_cfg.max_strategy_heat_R)
        if self._on_config_update is not None:
            try:
                self._on_config_update(new_cfg)
            except Exception:
                logger.debug("Portfolio rule config update callback failed", exc_info=True)

    def _emit(self, event: dict) -> None:
        if self._on_rule_event:
            try:
                context = _RULE_EVENT_CONTEXT.get() or {}
                check_sequence = int(context.get("check_sequence", 0) or 0) + 1
                context["check_sequence"] = check_sequence
                event = dict(event)
                event.setdefault("check_sequence", check_sequence)
                if event.get("rule"):
                    context.setdefault("applied_rules", []).append(str(event["rule"]))
                for key in ("trace_id", "rule_trace_id", "signal_id", "bar_id", "exchange_timestamp"):
                    if context.get(key):
                        event.setdefault(key, context[key])
                normalized = build_portfolio_rule_event(
                    event,
                    portfolio_rules_config=self._cfg,
                    request_context=context,
                    lineage=context.get("lineage_context"),
                )
                self._on_rule_event(normalized)
            except Exception:
                pass

    def _current_dd_pct(self) -> float:
        equity = self._get_equity()
        initial = self._cfg.initial_equity
        if initial <= 0 or equity >= initial:
            return 0.0
        return (initial - equity) / initial

    def _reference_unit_risk_dollars(self) -> float:
        if self._cfg.reference_unit_risk_pct > 0:
            try:
                equity = float(self._get_equity())
            except (TypeError, ValueError):
                equity = 0.0
            if equity > 0:
                return max(equity * self._cfg.reference_unit_risk_pct, 1.0)
        return self._cfg.reference_unit_risk_dollars

    async def check_entry(
        self,
        strategy_id: str,
        direction: str,  # "LONG" or "SHORT"
        new_risk_R: float = 1.0,
        symbol: Optional[str] = None,
        new_qty: int = 0,
        new_risk_dollars: float = 0.0,
        *,
        trace_id: str = "",
        signal_id: str = "",
        bar_id: str = "",
        exchange_timestamp: datetime | None = None,
        lineage_context: Any = None,
    ) -> PortfolioRuleResult:
        """Run all portfolio rules. Returns result with approval and size multiplier."""
        result = PortfolioRuleResult()
        try:
            current_equity = float(self._get_equity())
        except Exception:
            current_equity = 0.0
        rule_trace_id = stable_hash(
            "rule_trace_",
            {
                "trace_id": trace_id,
                "strategy_id": strategy_id,
                "direction": direction,
                "symbol": symbol or "",
                "requested_qty": new_qty,
                "requested_risk_R": new_risk_R,
                "requested_risk_dollars": new_risk_dollars,
            },
        )
        rule_context = {
            "strategy_id": strategy_id,
            "direction": direction,
            "symbol": symbol or "",
            "requested_sizing": {
                "risk_R": new_risk_R,
                "qty": new_qty,
                "risk_dollars": new_risk_dollars,
            },
            "state_before": {
                "current_equity": current_equity,
                "drawdown_pct": self._current_dd_pct(),
                "family_strategy_ids": list(self._cfg.family_strategy_ids),
                "portfolio_heat_cap_R": self._cfg.portfolio_heat_cap_R,
                "directional_cap_R": self._direction_cap_for(direction),
            },
            "current_size_multiplier": 1.0,
            "trace_id": trace_id,
            "rule_trace_id": rule_trace_id,
            "signal_id": signal_id,
            "bar_id": bar_id,
            "exchange_timestamp": (
                exchange_timestamp.isoformat()
                if hasattr(exchange_timestamp, "isoformat")
                else str(exchange_timestamp or "")
            ),
            "lineage_context": lineage_context,
            "check_sequence": 0,
            "applied_rules": [],
        }
        context_token = _RULE_EVENT_CONTEXT.set(rule_context)
        context_active = True

        def _result(
            *,
            approved: bool = True,
            denial_reason: Optional[str] = None,
            size_multiplier: Optional[float] = None,
        ) -> PortfolioRuleResult:
            nonlocal context_active
            multiplier = result.size_multiplier if size_multiplier is None else size_multiplier
            approved_qty = self._adjusted_qty(new_qty, multiplier) if approved else 0
            if new_qty > 0:
                approved_risk_R = new_risk_R * (approved_qty / new_qty)
                approved_risk_dollars = new_risk_dollars * (approved_qty / new_qty)
            else:
                approved_risk_R = new_risk_R * multiplier if approved else 0.0
                approved_risk_dollars = new_risk_dollars * multiplier if approved else 0.0
            rule_result = PortfolioRuleResult(
                approved=approved,
                denial_reason=denial_reason,
                size_multiplier=multiplier,
                requested_qty=new_qty,
                approved_qty=approved_qty,
                requested_risk_R=new_risk_R,
                approved_risk_R=approved_risk_R,
                requested_risk_dollars=new_risk_dollars,
                approved_risk_dollars=approved_risk_dollars,
                rule_trace_id=rule_trace_id,
                applied_rules=tuple(rule_context.get("applied_rules", ())),
                lineage_gap=lineage_context is None,
            )
            if context_active:
                _RULE_EVENT_CONTEXT.reset(context_token)
                context_active = False
            return rule_result

        # 0. Regime strategy disable
        if strategy_id in self._cfg.disabled_strategies:
            self._emit({"rule": "regime_disabled", "strategy_id": strategy_id, "approved": False})
            return _result(
                approved=False,
                denial_reason=f"regime_disabled: {strategy_id} blocked in current regime",
                size_multiplier=1.0,
            )

        # 1. NQDTC direction filter (Vdubus only)
        size_mult = await self._check_direction_filter(strategy_id, direction)
        if size_mult == 0.0:
            reason = f"nqdtc_direction_filter: NQDTC opposes {direction}"
            self._emit({"rule": "nqdtc_direction_filter", "strategy_id": strategy_id,
                         "direction": direction, "approved": False, "denial_reason": reason})
            return _result(approved=False, denial_reason=reason, size_multiplier=1.0)
        if size_mult != 1.0:
            self._emit({"rule": "nqdtc_direction_filter", "strategy_id": strategy_id,
                         "direction": direction, "approved": True, "size_multiplier": size_mult})
        result.size_multiplier *= size_mult
        rule_context["current_size_multiplier"] = result.size_multiplier

        # 2. Static optimized per-strategy multipliers.
        strategy_mult = self._strategy_size_multipliers.get(strategy_id, 1.0)
        if strategy_mult != 1.0:
            self._emit({"rule": "strategy_size_multiplier", "strategy_id": strategy_id,
                        "approved": True, "size_multiplier": strategy_mult})
        result.size_multiplier *= strategy_mult
        rule_context["current_size_multiplier"] = result.size_multiplier

        # 3. Drawdown tiers
        dd_mult = self._check_drawdown_tier()
        if dd_mult == 0.0:
            reason = "drawdown_halt: equity drawdown exceeds maximum tier"
            self._emit({"rule": "drawdown_tier", "strategy_id": strategy_id,
                         "approved": False, "denial_reason": reason,
                         "drawdown_pct": self._current_dd_pct(), "size_multiplier": 0.0})
            return _result(approved=False, denial_reason=reason, size_multiplier=1.0)
        if dd_mult < 1.0:
            self._emit({"rule": "drawdown_tier", "strategy_id": strategy_id,
                         "approved": True, "size_multiplier": dd_mult,
                         "drawdown_pct": self._current_dd_pct()})
        result.size_multiplier *= dd_mult
        rule_context["current_size_multiplier"] = result.size_multiplier

        # 4b. Regime unit-risk multipliers
        regime_mult = self._cfg.regime_unit_risk_mult
        if regime_mult != 1.0:
            self._emit({"rule": "regime_unit_risk", "strategy_id": strategy_id,
                        "approved": True, "size_multiplier": regime_mult})
            result.size_multiplier *= regime_mult
            rule_context["current_size_multiplier"] = result.size_multiplier

        direction_mult = self._directional_unit_risk_mult(direction)
        if direction_mult != 1.0:
            self._emit({"rule": "regime_directional_unit_risk", "strategy_id": strategy_id,
                        "direction": direction, "approved": True,
                        "size_multiplier": direction_mult})
            result.size_multiplier *= direction_mult
            rule_context["current_size_multiplier"] = result.size_multiplier

        dynamic_mult = await self._dynamic_allocation_mult(strategy_id)
        if dynamic_mult != 1.0:
            self._emit({"rule": "dynamic_allocation", "strategy_id": strategy_id,
                        "approved": True, "size_multiplier": dynamic_mult})
            result.size_multiplier *= dynamic_mult
            rule_context["current_size_multiplier"] = result.size_multiplier

        momentum_mult = await self._momentum_dynamic_sizing_mult(
            strategy_id=strategy_id,
            direction=direction,
        )
        if momentum_mult == 0.0:
            reason = "dynamic_capacity_floor"
            self._emit({"rule": "dynamic_capacity_floor", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": reason})
            return _result(approved=False, denial_reason=reason, size_multiplier=1.0)
        if momentum_mult != 1.0:
            self._emit({"rule": "momentum_dynamic_sizing", "strategy_id": strategy_id,
                        "direction": direction, "approved": True,
                        "size_multiplier": momentum_mult})
            result.size_multiplier *= momentum_mult
            rule_context["current_size_multiplier"] = result.size_multiplier

        capacity_mult = await self._capacity_fit_mult(
            strategy_id=strategy_id,
            direction=direction,
            symbol=symbol or "",
            new_qty=new_qty,
            new_risk_dollars=new_risk_dollars,
            current_size_multiplier=result.size_multiplier,
        )
        if capacity_mult == 0.0:
            reason = "dynamic_capacity_floor"
            self._emit({"rule": "dynamic_capacity_floor", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": reason})
            return _result(approved=False, denial_reason=reason, size_multiplier=1.0)
        if capacity_mult != 1.0:
            self._emit({"rule": "dynamic_capacity_fit", "strategy_id": strategy_id,
                        "approved": True, "size_multiplier": capacity_mult})
            result.size_multiplier *= capacity_mult
            rule_context["current_size_multiplier"] = result.size_multiplier

        # 5. Symbol collision (stock family -- block/reduce when sibling holds same ticker)
        collision_result = await self._check_symbol_collision(strategy_id, symbol)
        if collision_result is not None:
            if collision_result == 0.0:
                reason = f"symbol_collision: sibling strategy holds {symbol}"
                self._emit({"rule": "symbol_collision", "strategy_id": strategy_id,
                             "symbol": symbol, "approved": False, "denial_reason": reason})
                return _result(approved=False, denial_reason=reason, size_multiplier=1.0)
            self._emit({"rule": "symbol_collision", "strategy_id": strategy_id,
                         "symbol": symbol, "approved": True, "size_multiplier": collision_result})
            result.size_multiplier *= collision_result
            rule_context["current_size_multiplier"] = result.size_multiplier

        adjusted_qty = self._adjusted_qty(new_qty, result.size_multiplier)
        if new_qty > 0 and new_risk_dollars > 0:
            risk_per_contract = new_risk_dollars / new_qty
            adjusted_risk_dollars = adjusted_qty * risk_per_contract
            adjusted_risk_R = (
                new_risk_R * (adjusted_risk_dollars / new_risk_dollars)
                if new_risk_dollars > 0 else new_risk_R * result.size_multiplier
            )
        else:
            adjusted_risk_R = new_risk_R * result.size_multiplier
            adjusted_risk_dollars = new_risk_dollars * result.size_multiplier

        # 6. Portfolio-synergy hard caps
        denial = await self._check_total_active_positions()
        if denial:
            self._emit({"rule": "max_total_active_positions", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_strategy_active_positions(strategy_id)
        if denial:
            self._emit({"rule": "max_strategy_active_positions", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_portfolio_heat_cap(adjusted_risk_R, adjusted_risk_dollars)
        if denial:
            self._emit({"rule": "portfolio_heat_cap", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_strategy_heat_cap(
            strategy_id, adjusted_risk_R, adjusted_risk_dollars,
        )
        if denial:
            self._emit({"rule": "strategy_heat_cap", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_strategy_trade_share(strategy_id)
        if denial:
            self._emit({"rule": "strategy_trade_share_cap", "strategy_id": strategy_id,
                        "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_directional_cap(
            strategy_id, direction, adjusted_risk_R, adjusted_risk_dollars,
        )
        if denial:
            self._emit({"rule": "directional_cap", "strategy_id": strategy_id,
                         "direction": direction, "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_family_contract_cap(
            strategy_id, symbol or "", adjusted_qty,
        )
        if denial:
            self._emit({"rule": "family_contract_cap", "strategy_id": strategy_id,
                         "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_symbol_heat_cap(symbol, adjusted_risk_R, adjusted_risk_dollars)
        if denial:
            self._emit({"rule": "symbol_heat_cap", "strategy_id": strategy_id,
                        "symbol": symbol, "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        denial = await self._check_sector_heat_cap(symbol, adjusted_risk_R, adjusted_risk_dollars)
        if denial:
            self._emit({"rule": "sector_heat_cap", "strategy_id": strategy_id,
                        "symbol": symbol, "approved": False, "denial_reason": denial})
            return _result(approved=False, denial_reason=denial, size_multiplier=1.0)

        return _result(approved=True, size_multiplier=result.size_multiplier)

    @staticmethod
    def _adjusted_qty(new_qty: int, size_multiplier: float) -> int:
        if new_qty <= 0 or size_multiplier <= 0:
            return 0
        return max(1, int(new_qty * size_multiplier))

    def _directional_unit_risk_mult(self, direction: str) -> float:
        direction_u = (direction or "").upper()
        if direction_u == "LONG":
            return self._cfg.regime_unit_risk_long_mult
        if direction_u == "SHORT":
            return self._cfg.regime_unit_risk_short_mult
        return 1.0

    async def _dynamic_allocation_mult(self, strategy_id: str) -> float:
        if not self._cfg.dynamic_allocation_enabled or self._get_recent_r is None:
            return 1.0
        lookback = max(1, int(self._cfg.dynamic_lookback_trades))
        recent = await self._get_recent_r(strategy_id, lookback)
        if len(recent) < 20:
            return 1.0
        avg_r = sum(recent) / len(recent)
        win_rate = sum(1 for value in recent if value > 0) / len(recent)
        mult = 1.0
        if avg_r > 0.20 and win_rate > 0.58:
            mult += self._cfg.dynamic_positive_expectancy_boost
        elif avg_r < 0.0 or win_rate < 0.45:
            mult -= self._cfg.dynamic_negative_expectancy_cut
        return max(self._cfg.dynamic_min_mult, min(self._cfg.dynamic_max_mult, mult))

    async def _momentum_dynamic_sizing_mult(
        self,
        *,
        strategy_id: str,
        direction: str,
    ) -> float:
        mult = 1.0
        family_ids = list(self._cfg.family_strategy_ids)
        if family_ids and self._cfg.existing_position_mult != 1.0 and self._get_open_position_count is not None:
            if await self._get_open_position_count(family_ids) > 0:
                mult *= self._cfg.existing_position_mult

        ref_urd = self._reference_unit_risk_dollars()
        if ref_urd <= 0:
            return mult

        heat_cap = self._cfg.portfolio_heat_cap_R
        if (
            heat_cap > 0
            and self._cfg.heat_pressure_mult != 1.0
            and self._get_active_risk_dollars is not None
            and family_ids
        ):
            current_heat_R = await self._get_active_risk_dollars(family_ids) / ref_urd
            if current_heat_R / heat_cap >= self._cfg.heat_pressure_threshold:
                mult *= self._cfg.heat_pressure_mult

        direction_cap = self._direction_cap_for(direction)
        if (
            direction_cap > 0
            and self._cfg.same_direction_pressure_mult != 1.0
            and self._get_dir_risk_dollars is not None
        ):
            current_direction_R = await self._get_dir_risk_dollars(direction) / ref_urd
            if current_direction_R / direction_cap >= self._cfg.same_direction_pressure_threshold:
                mult *= self._cfg.same_direction_pressure_mult

        return mult

    async def _capacity_fit_mult(
        self,
        *,
        strategy_id: str,
        direction: str,
        symbol: str,
        new_qty: int,
        new_risk_dollars: float,
        current_size_multiplier: float,
    ) -> float:
        if new_qty <= 0 or new_risk_dollars <= 0:
            return 1.0
        ref_urd = self._reference_unit_risk_dollars()
        risk_per_contract = new_risk_dollars / new_qty
        if ref_urd <= 0 or risk_per_contract <= 0:
            return 1.0

        max_qty = self._adjusted_qty(new_qty, current_size_multiplier)
        cap_limited = False

        if self._cfg.max_trade_risk_R > 0:
            max_qty = min(
                max_qty,
                self._qty_for_risk_dollars(
                    self._cfg.max_trade_risk_R * ref_urd,
                    risk_per_contract,
                ),
            )

        family_ids = list(self._cfg.family_strategy_ids)
        if (
            self._cfg.fit_to_remaining_heat
            and self._cfg.portfolio_heat_cap_R > 0
            and self._get_active_risk_dollars is not None
            and family_ids
        ):
            current_dollars = await self._get_active_risk_dollars(family_ids)
            remaining_dollars = self._cfg.portfolio_heat_cap_R * ref_urd - current_dollars
            max_qty = min(max_qty, self._qty_for_risk_dollars(remaining_dollars, risk_per_contract))
            cap_limited = True

        direction_cap = self._direction_cap_for(direction)
        if self._cfg.fit_to_remaining_directional_cap and direction_cap > 0 and self._get_dir_risk_dollars is not None:
            current_dollars = await self._get_dir_risk_dollars(direction)
            remaining_dollars = direction_cap * ref_urd - current_dollars
            max_qty = min(max_qty, self._qty_for_risk_dollars(remaining_dollars, risk_per_contract))
            cap_limited = True

        if (
            self._cfg.fit_to_remaining_family_cap
            and self._cfg.max_family_contracts_mnq_eq > 0
            and self._get_family_mnq_eq is not None
            and family_ids
        ):
            current_mnq_eq = await self._get_family_mnq_eq(family_ids)
            symbol_mult = 10 if symbol == "NQ" else 1
            remaining_mnq_eq = self._cfg.max_family_contracts_mnq_eq - current_mnq_eq
            max_qty = min(max_qty, max(0, remaining_mnq_eq // symbol_mult))
            cap_limited = True

        min_qty = max(1, int(self._cfg.min_qty))
        if max_qty < min_qty:
            return 0.0 if cap_limited or max_qty <= 0 else 1.0
        target_mult = max_qty / new_qty
        if target_mult < current_size_multiplier:
            return max(0.0, target_mult / max(current_size_multiplier, 1e-12))
        return 1.0

    @staticmethod
    def _qty_for_risk_dollars(risk_dollars: float, risk_per_contract: float) -> int:
        if risk_dollars <= 0 or risk_per_contract <= 0:
            return 0
        return max(0, int(risk_dollars // risk_per_contract))

    def _direction_cap_for(self, direction: str) -> float:
        direction_u = (direction or "").upper()
        if direction_u == "LONG" and self._cfg.directional_cap_long_R > 0:
            return self._cfg.directional_cap_long_R
        if direction_u == "SHORT" and self._cfg.directional_cap_short_R > 0:
            return self._cfg.directional_cap_short_R
        return self._cfg.directional_cap_R

    async def _check_direction_filter(self, strategy_id: str, direction: str) -> float:
        """NQDTC direction filter — affects Vdubus sizing."""
        if not self._cfg.nqdtc_direction_filter_enabled:
            return 1.0

        if strategy_id != self._cfg.vdubus_strategy_id:
            return 1.0

        nqdtc_signal = await self._get_signal(self._cfg.nqdtc_strategy_id)
        if nqdtc_signal is None or nqdtc_signal["last_direction"] is None:
            return 1.0  # No NQDTC trade today — no filter

        # Only apply if NQDTC traded today
        today = self._now().astimezone(ZoneInfo("America/New_York")).date()
        if nqdtc_signal["signal_date"] != today:
            return 1.0

        if nqdtc_signal["last_direction"] == direction:
            return self._cfg.nqdtc_agree_size_mult
        else:
            return self._cfg.nqdtc_oppose_size_mult

    async def _check_directional_cap(
        self, strategy_id: str, direction: str, new_risk_R: float,
        new_risk_dollars: float = 0.0,
    ) -> Optional[str]:
        """Max same-direction risk, with optional priority-based reservation.

        When ``reference_unit_risk_dollars > 0`` and a dollar callback is wired,
        the cap is checked in dollar space to avoid mixing R units from
        strategies with different URDs.
        """
        # Resolve per-direction cap (asymmetric overrides symmetric fallback)
        cap_R = self._cfg.directional_cap_R
        if direction == "LONG" and self._cfg.directional_cap_long_R > 0:
            cap_R = self._cfg.directional_cap_long_R
        elif direction == "SHORT" and self._cfg.directional_cap_short_R > 0:
            cap_R = self._cfg.directional_cap_short_R
        if cap_R <= 0:
            return None

        ref_urd = self._reference_unit_risk_dollars()
        use_dollars = ref_urd > 0 and self._get_dir_risk_dollars is not None

        if use_dollars:
            # Dollar-based path: avoids mixed R units across strategies
            current_dollars = await self._get_dir_risk_dollars(direction)
            cap_dollars = cap_R * ref_urd
            total_dollars = current_dollars + new_risk_dollars

            if total_dollars > cap_dollars:
                return (
                    f"directional_cap: {direction} risk ${current_dollars:.0f} + "
                    f"new ${new_risk_dollars:.0f} = ${total_dollars:.0f} > "
                    f"cap {cap_R}R * ${ref_urd:.0f} = ${cap_dollars:.0f}"
                )

            # Soft reservation in dollar space
            headroom_R = self._cfg.priority_headroom_R
            if headroom_R > 0 and self._cfg.strategy_priorities:
                remaining_dollars = cap_dollars - current_dollars
                headroom_dollars = headroom_R * ref_urd
                priority_map = dict(self._cfg.strategy_priorities)
                my_priority = priority_map.get(strategy_id, 99)

                if remaining_dollars <= headroom_dollars and my_priority > self._cfg.priority_reserve_threshold:
                    return (
                        f"directional_cap_reserved: {direction} remaining "
                        f"${remaining_dollars:.0f} <= headroom ${headroom_dollars:.0f}, "
                        f"strategy {strategy_id} priority {my_priority} "
                        f"> threshold {self._cfg.priority_reserve_threshold}"
                    )

            return None

        # R-based path (original): safe when all strategies share the same URD
        current_dir_risk = await self._get_dir_risk(direction)
        total = current_dir_risk + new_risk_R

        # Hard cap: no strategy can exceed the absolute cap
        if total > cap_R:
            return (
                f"directional_cap: {direction} risk {current_dir_risk:.2f}R + "
                f"new {new_risk_R:.2f}R = {total:.2f}R > cap {cap_R}R"
            )

        # Soft reservation: when headroom is tight, reserve for higher-priority strategies
        headroom_R = self._cfg.priority_headroom_R
        if headroom_R <= 0 or not self._cfg.strategy_priorities:
            return None

        remaining = cap_R - current_dir_risk
        priority_map = dict(self._cfg.strategy_priorities)
        my_priority = priority_map.get(strategy_id, 99)

        if remaining <= headroom_R and my_priority > self._cfg.priority_reserve_threshold:
            return (
                f"directional_cap_reserved: {direction} remaining "
                f"{remaining:.2f}R <= headroom {headroom_R:.1f}R, "
                f"strategy {strategy_id} priority {my_priority} "
                f"> threshold {self._cfg.priority_reserve_threshold}"
            )

        return None

    async def _check_symbol_collision(
        self, strategy_id: str, symbol: Optional[str],
    ) -> Optional[float]:
        """Check if a sibling strategy already holds the same symbol.

        Returns None if check not applicable, 0.0 to block, or a multiplier to reduce size.
        Per-pair overrides in symbol_collision_pairs are checked first; if none match,
        falls through to the generic symbol_collision_action.
        """
        if not symbol:
            return None
        family_ids = self._cfg.family_strategy_ids
        if not family_ids or self._get_sibling is None:
            return None

        # --- Per-pair overrides (checked first) ---
        for holder_id, requester_id, pair_action in self._cfg.symbol_collision_pairs:
            if requester_id != strategy_id:
                continue
            holder_has = await self._get_sibling([holder_id], symbol)
            if not holder_has:
                continue
            logger.info(
                "Symbol collision pair: %s holds %s → %s for %s",
                holder_id, symbol, pair_action, strategy_id,
            )
            # Actions validated in __post_init__
            if pair_action == "block":
                return 0.0
            return 0.5  # half_size

        # --- Generic fallback ---
        action = self._cfg.symbol_collision_action
        if action == "none":
            return None

        sibling_ids = [sid for sid in family_ids if sid != strategy_id]
        if not sibling_ids:
            return None

        has_collision = await self._get_sibling(sibling_ids, symbol)
        if not has_collision:
            return None

        if action == "block":
            return 0.0
        if action == "half_size":
            logger.info(
                "Symbol collision: sibling holds %s → half size for %s",
                symbol, strategy_id,
            )
            return 0.5
        return None

    async def _check_total_active_positions(self) -> Optional[str]:
        cap = self._cfg.max_total_active_positions
        family_ids = list(self._cfg.family_strategy_ids)
        if cap <= 0 or not family_ids or self._get_open_position_count is None:
            return None
        current = await self._get_open_position_count(family_ids)
        if current >= cap:
            return f"max_total_active_positions: {current} open >= cap {cap}"
        return None

    async def _check_strategy_active_positions(self, strategy_id: str) -> Optional[str]:
        cap = self._max_strategy_active_positions.get(strategy_id, 0)
        if cap <= 0 or self._get_open_position_count is None:
            return None
        current = await self._get_open_position_count([strategy_id])
        if current >= cap:
            return f"max_strategy_active_positions: {strategy_id} {current} open >= cap {cap}"
        return None

    async def _check_portfolio_heat_cap(
        self,
        new_risk_R: float,
        new_risk_dollars: float,
    ) -> Optional[str]:
        cap = self._cfg.portfolio_heat_cap_R
        family_ids = list(self._cfg.family_strategy_ids)
        if cap <= 0 or not family_ids:
            return None

        ref_urd = self._reference_unit_risk_dollars()
        if ref_urd > 0 and self._get_active_risk_dollars is not None:
            current_dollars = await self._get_active_risk_dollars(family_ids)
            total_R = (current_dollars + new_risk_dollars) / ref_urd
            if total_R > cap:
                return (
                    f"portfolio_heat_cap: {current_dollars / ref_urd:.2f}R + "
                    f"new {new_risk_dollars / ref_urd:.2f}R > cap {cap:.2f}R"
                )
            return None

        if new_risk_R > cap:
            return f"portfolio_heat_cap: new {new_risk_R:.2f}R > cap {cap:.2f}R"
        return None

    async def _check_strategy_heat_cap(
        self,
        strategy_id: str,
        new_risk_R: float,
        new_risk_dollars: float,
    ) -> Optional[str]:
        cap = self._max_strategy_heat_R.get(strategy_id, 0.0)
        if cap <= 0:
            return None

        ref_urd = self._reference_unit_risk_dollars()
        if ref_urd > 0 and self._get_active_risk_dollars is not None:
            current_dollars = await self._get_active_risk_dollars([strategy_id])
            total_R = (current_dollars + new_risk_dollars) / ref_urd
            if total_R > cap:
                return (
                    f"strategy_heat_cap: {strategy_id} {current_dollars / ref_urd:.2f}R + "
                    f"new {new_risk_dollars / ref_urd:.2f}R > cap {cap:.2f}R"
                )
            return None

        if new_risk_R > cap:
            return f"strategy_heat_cap: {strategy_id} new {new_risk_R:.2f}R > cap {cap:.2f}R"
        return None

    async def _check_strategy_trade_share(self, strategy_id: str) -> Optional[str]:
        cap = self._cfg.max_single_strategy_trade_share
        family_ids = list(self._cfg.family_strategy_ids)
        if cap >= 1.0 or not family_ids or self._get_trade_counts is None:
            return None
        counts = await self._get_trade_counts(family_ids)
        future_total = sum(counts.values()) + 1
        if future_total <= self._cfg.strategy_trade_share_min_total:
            return None
        future_count = int(counts.get(strategy_id, 0)) + 1
        future_share = future_count / future_total
        if future_share > cap:
            return (
                f"strategy_trade_share_cap: {strategy_id} future share "
                f"{future_share:.2%} > cap {cap:.2%}"
            )
        return None

    async def _check_symbol_heat_cap(
        self,
        symbol: Optional[str],
        new_risk_R: float,
        new_risk_dollars: float,
    ) -> Optional[str]:
        cap = self._cfg.max_symbol_heat_R
        family_ids = list(self._cfg.family_strategy_ids)
        if cap <= 0 or not symbol or not family_ids:
            return None

        ref_urd = self._reference_unit_risk_dollars()
        if ref_urd > 0 and self._get_symbol_risk_dollars is not None:
            current_dollars = await self._get_symbol_risk_dollars(family_ids, symbol)
            total_R = (current_dollars + new_risk_dollars) / ref_urd
            if total_R > cap:
                return (
                    f"symbol_heat_cap: {symbol} "
                    f"{current_dollars / ref_urd:.2f}R + new "
                    f"{new_risk_dollars / ref_urd:.2f}R > cap {cap:.2f}R"
                )
            return None

        if new_risk_R > cap:
            return f"symbol_heat_cap: new {new_risk_R:.2f}R > cap {cap:.2f}R"
        return None

    async def _check_sector_heat_cap(
        self,
        symbol: Optional[str],
        new_risk_R: float,
        new_risk_dollars: float,
    ) -> Optional[str]:
        cap = self._cfg.same_sector_heat_cap_R
        family_ids = list(self._cfg.family_strategy_ids)
        sector = self._symbol_sector_map.get(symbol or "")
        if cap <= 0 or not sector or not family_ids:
            return None

        sector_symbols = [
            mapped_symbol
            for mapped_symbol, mapped_sector in self._symbol_sector_map.items()
            if mapped_sector == sector
        ]
        ref_urd = self._reference_unit_risk_dollars()
        if ref_urd > 0 and self._get_symbols_risk_dollars is not None:
            current_dollars = await self._get_symbols_risk_dollars(
                family_ids, sector_symbols,
            )
            total_R = (current_dollars + new_risk_dollars) / ref_urd
            if total_R > cap:
                return (
                    f"sector_heat_cap: {sector} "
                    f"{current_dollars / ref_urd:.2f}R + new "
                    f"{new_risk_dollars / ref_urd:.2f}R > cap {cap:.2f}R"
                )
            return None

        if new_risk_R > cap:
            return f"sector_heat_cap: new {new_risk_R:.2f}R > cap {cap:.2f}R"
        return None

    async def _check_family_contract_cap(
        self, strategy_id: str, symbol: str, new_qty: int,
    ) -> Optional[str]:
        """Limit total MNQ-equivalent contracts across the family."""
        cap = self._cfg.max_family_contracts_mnq_eq
        if cap <= 0 or self._get_family_mnq_eq is None:
            return None
        family_ids = list(self._cfg.family_strategy_ids)
        if not family_ids:
            return None
        current = await self._get_family_mnq_eq(family_ids)
        new_mnq_eq = abs(new_qty) * (10 if symbol == "NQ" else 1)
        total = current + new_mnq_eq
        if total > cap:
            return (
                f"family_contract_cap: {current} + {new_mnq_eq} = "
                f"{total} MNQ-eq > cap {cap}"
            )
        return None

    def _check_drawdown_tier(self) -> float:
        """Drawdown-based size multiplier."""
        equity = self._get_equity()
        initial = self._cfg.initial_equity
        if initial <= 0 or equity >= initial:
            return 1.0

        dd_pct = (initial - equity) / initial
        for threshold, mult in self._cfg.dd_tiers:
            if dd_pct < threshold:
                if mult < 1.0:
                    logger.info(
                        "Drawdown tier active: %.1f%% DD → %.0f%% size",
                        dd_pct * 100, mult * 100,
                    )
                return mult

        return 0.0  # Beyond all tiers
