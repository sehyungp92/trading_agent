"""Shared swing portfolio risk adapter.

The swing family uses a portfolio heat model that is different from the
momentum/stock directional-rule checker: strategy sleeves have their own R
units, portfolio heat is normalized to the highest-priority sleeve, and tight
remaining heat is reserved for higher-priority idle sleeves.  This module keeps
that policy in one pure evaluator and exposes thin live/replay adapters around
it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping


@dataclass(frozen=True)
class SwingStrategyRiskSlot:
    strategy_id: str
    priority: int
    unit_risk_dollars: float
    max_heat_R: float
    daily_stop_R: float


@dataclass(frozen=True)
class SwingPortfolioRiskSnapshot:
    heat_cap_R: float
    portfolio_daily_stop_R: float
    portfolio_unit_risk_dollars: float
    strategy_slots: Mapping[str, SwingStrategyRiskSlot]
    strategy_open_risk_dollars: Mapping[str, float] = field(default_factory=dict)
    strategy_daily_realized_dollars: Mapping[str, float] = field(default_factory=dict)
    portfolio_daily_realized_dollars: float | None = None
    portfolio_pending_entry_risk_dollars: float = 0.0
    portfolio_halted: bool = False
    simulate_live_r_normalization: bool = False
    reserve_idle_higher_priority: bool = True


@dataclass(frozen=True)
class SwingPortfolioRiskDecision:
    approved: bool
    reason: str = ""
    risk_context: dict[str, float] = field(default_factory=dict)


@dataclass
class SwingStrategyHeatState:
    """Replay-side mutable risk state for a single strategy sleeve."""

    strategy_id: str = ""
    priority: int = 0
    unit_risk_dollars: float = 0.0
    max_heat_R: float = 1.0
    daily_stop_R: float = 2.0
    open_risk_dollars: float = 0.0
    daily_realized_dollars: float = 0.0

    @property
    def daily_realized_R(self) -> float:
        if self.unit_risk_dollars <= 0:
            return 0.0
        return self.daily_realized_dollars / self.unit_risk_dollars

    @property
    def open_risk_R(self) -> float:
        if self.unit_risk_dollars <= 0:
            return 0.0
        return self.open_risk_dollars / self.unit_risk_dollars


def evaluate_swing_entry(
    snapshot: SwingPortfolioRiskSnapshot,
    strategy_id: str,
    risk_dollars: float,
) -> SwingPortfolioRiskDecision:
    """Evaluate a swing entry request against shared portfolio risk rules."""

    risk_dollars = float(risk_dollars or 0.0)
    strat = snapshot.strategy_slots.get(strategy_id)
    if strat is None:
        return SwingPortfolioRiskDecision(False, f"Unknown strategy {strategy_id}")

    context = build_swing_entry_risk_context(snapshot, strategy_id, risk_dollars)

    if snapshot.portfolio_halted:
        return SwingPortfolioRiskDecision(False, "Portfolio daily stop hit", context)

    portfolio_unit = float(snapshot.portfolio_unit_risk_dollars or 0.0)
    portfolio_daily_dollars = snapshot.portfolio_daily_realized_dollars
    if portfolio_daily_dollars is None:
        portfolio_daily_dollars = sum(
            float(v or 0.0) for v in snapshot.strategy_daily_realized_dollars.values()
        )
    if portfolio_unit > 0:
        portfolio_daily_R = float(portfolio_daily_dollars or 0.0) / portfolio_unit
        if portfolio_daily_R <= -float(snapshot.portfolio_daily_stop_R):
            return SwingPortfolioRiskDecision(False, "Portfolio daily stop hit", context)

    strategy_daily_dollars = float(
        snapshot.strategy_daily_realized_dollars.get(strategy_id, 0.0) or 0.0
    )
    if strat.unit_risk_dollars > 0:
        strategy_daily_R = strategy_daily_dollars / strat.unit_risk_dollars
        if strategy_daily_R <= -strat.daily_stop_R:
            return SwingPortfolioRiskDecision(
                False,
                f"{strategy_id} daily stop hit",
                context,
            )

    strategy_open_dollars = float(
        snapshot.strategy_open_risk_dollars.get(strategy_id, 0.0) or 0.0
    )
    if strat.unit_risk_dollars > 0:
        strategy_open_R = strategy_open_dollars / strat.unit_risk_dollars
        strategy_request_R = risk_dollars / strat.unit_risk_dollars
        if strategy_open_R + strategy_request_R > strat.max_heat_R:
            return SwingPortfolioRiskDecision(
                False,
                (
                    f"{strategy_id} heat ceiling "
                    f"({strategy_open_R:.2f}R + {strategy_request_R:.2f}R > {strat.max_heat_R}R)"
                ),
                context,
            )

    total_open_dollars = sum(float(v or 0.0) for v in snapshot.strategy_open_risk_dollars.values())
    pending_dollars = float(snapshot.portfolio_pending_entry_risk_dollars or 0.0)
    total_reserved_dollars = total_open_dollars + pending_dollars

    if snapshot.simulate_live_r_normalization:
        total_open_R = 0.0
        for sid, slot in snapshot.strategy_slots.items():
            unit = slot.unit_risk_dollars
            if unit > 0:
                total_open_R += float(snapshot.strategy_open_risk_dollars.get(sid, 0.0) or 0.0) / unit
        pending_R = pending_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        request_R = risk_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else float("inf")
    elif portfolio_unit > 0:
        total_open_R = total_open_dollars / portfolio_unit
        pending_R = pending_dollars / portfolio_unit
        request_R = risk_dollars / portfolio_unit
    else:
        total_open_R = 0.0
        pending_R = 0.0
        request_R = 0.0

    if total_open_R + pending_R + request_R > snapshot.heat_cap_R:
        if pending_R > 0:
            reason = (
                f"Portfolio heat cap ({total_open_R:.2f}R + pending {pending_R:.2f}R + "
                f"{request_R:.2f}R > {snapshot.heat_cap_R}R)"
            )
        else:
            reason = f"Portfolio heat cap ({total_open_R:.2f}R + {request_R:.2f}R > {snapshot.heat_cap_R}R)"
        return SwingPortfolioRiskDecision(False, reason, context)

    if snapshot.simulate_live_r_normalization:
        remaining_R = snapshot.heat_cap_R - (total_open_R + pending_R)
        remaining_dollars = remaining_R * strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
    else:
        remaining_dollars = (
            snapshot.heat_cap_R * portfolio_unit - total_reserved_dollars
            if portfolio_unit > 0
            else 0.0
        )
    if snapshot.reserve_idle_higher_priority and remaining_dollars < 2 * risk_dollars:
        for other in sorted(snapshot.strategy_slots.values(), key=lambda item: item.priority):
            if other.priority < strat.priority and float(snapshot.strategy_open_risk_dollars.get(other.strategy_id, 0.0) or 0.0) == 0.0:
                return SwingPortfolioRiskDecision(
                    False,
                    f"Heat reserved for {other.strategy_id}",
                    context,
                )

    return SwingPortfolioRiskDecision(True, "", context)


def build_swing_entry_risk_context(
    snapshot: SwingPortfolioRiskSnapshot,
    strategy_id: str,
    risk_dollars: float,
) -> dict[str, float]:
    """Return normalized context used by live rule events and replay diagnostics."""

    risk_dollars = float(risk_dollars or 0.0)
    portfolio_unit = float(snapshot.portfolio_unit_risk_dollars or 0.0)
    total_open_dollars = sum(float(v or 0.0) for v in snapshot.strategy_open_risk_dollars.values())
    pending_dollars = float(snapshot.portfolio_pending_entry_risk_dollars or 0.0)
    portfolio_open_r = total_open_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    portfolio_pending_r = pending_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    portfolio_request_r = risk_dollars / portfolio_unit if portfolio_unit > 0 else 0.0
    context = {
        "portfolio_open_risk_R": float(portfolio_open_r),
        "portfolio_pending_risk_R": float(portfolio_pending_r),
        "portfolio_request_risk_R": float(portfolio_request_r),
        "portfolio_after_request_R": float(portfolio_open_r + portfolio_pending_r + portfolio_request_r),
        "portfolio_heat_cap_R": float(snapshot.heat_cap_R),
    }
    strat = snapshot.strategy_slots.get(strategy_id)
    if strat is not None:
        strategy_open_dollars = float(snapshot.strategy_open_risk_dollars.get(strategy_id, 0.0) or 0.0)
        strategy_daily_dollars = float(snapshot.strategy_daily_realized_dollars.get(strategy_id, 0.0) or 0.0)
        strategy_open_r = strategy_open_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        strategy_request_r = risk_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        strategy_daily_r = strategy_daily_dollars / strat.unit_risk_dollars if strat.unit_risk_dollars > 0 else 0.0
        context.update(
            {
                "strategy_open_risk_R": float(strategy_open_r),
                "strategy_request_risk_R": float(strategy_request_r),
                "strategy_after_request_R": float(strategy_open_r + strategy_request_r),
                "strategy_heat_cap_R": float(strat.max_heat_R),
                "strategy_daily_realized_R": float(strategy_daily_r),
                "strategy_daily_stop_R": float(strat.daily_stop_R),
            }
        )
    return context


class SwingPortfolioHeatAdapter:
    """Stateful replay adapter over the shared swing portfolio risk evaluator."""

    def __init__(
        self,
        heat_cap_R: float,
        portfolio_daily_stop_R: float,
        strategy_slots: list[Any],
        equity: float,
        simulate_live_r_normalization: bool = False,
        reserve_idle_higher_priority: bool = True,
    ):
        self._heat_cap_R = float(heat_cap_R)
        self._portfolio_daily_stop_R = float(portfolio_daily_stop_R)
        self._strats: dict[str, SwingStrategyHeatState] = {}
        self._portfolio_halted = False
        self._simulate_live_r = bool(simulate_live_r_normalization)
        self._reserve_idle_higher_priority = bool(reserve_idle_higher_priority)
        self._pending_entry_risk_by_strategy: dict[str, float] = {}

        sorted_slots = sorted(strategy_slots, key=lambda s: int(getattr(s, "priority")))
        self._portfolio_unit_risk_pct = (
            float(getattr(sorted_slots[0], "unit_risk_pct"))
            if sorted_slots
            else 0.01
        )
        self._portfolio_unit_risk = float(equity) * self._portfolio_unit_risk_pct

        for slot in strategy_slots:
            urd = float(equity) * float(getattr(slot, "unit_risk_pct"))
            strategy_id = str(getattr(slot, "strategy_id"))
            self._strats[strategy_id] = SwingStrategyHeatState(
                strategy_id=strategy_id,
                priority=int(getattr(slot, "priority")),
                unit_risk_dollars=urd,
                max_heat_R=float(getattr(slot, "max_heat_R")),
                daily_stop_R=float(getattr(slot, "daily_stop_R")),
            )

    def update_unit_risk(self, equity: float, slots: list[Any]) -> None:
        """Recalibrate unit risk dollars from current equity."""

        self._portfolio_unit_risk = float(equity) * self._portfolio_unit_risk_pct
        for slot in slots:
            strategy_id = str(getattr(slot, "strategy_id"))
            if strategy_id in self._strats:
                self._strats[strategy_id].unit_risk_dollars = float(equity) * float(
                    getattr(slot, "unit_risk_pct")
                )

    def can_enter(self, strategy_id: str, risk_dollars: float) -> tuple[bool, str]:
        """Check whether a new replay entry is allowed."""

        decision = evaluate_swing_entry(self._snapshot(), strategy_id, risk_dollars)
        if decision.reason == "Portfolio daily stop hit":
            self._portfolio_halted = True
        return decision.approved, decision.reason

    def update_open_risk(self, risk_by_strategy: dict[str, float]) -> None:
        """Refresh open risk from current replay positions."""

        for sid, dollars in risk_by_strategy.items():
            if sid in self._strats:
                self._strats[sid].open_risk_dollars = float(dollars or 0.0)
        self._pending_entry_risk_by_strategy.clear()

    def reserve_entry(self, strategy_id: str, risk_dollars: float) -> None:
        """Reserve newly accepted replay risk until the next open-risk refresh."""

        if strategy_id in self._strats:
            self._pending_entry_risk_by_strategy[strategy_id] = (
                self._pending_entry_risk_by_strategy.get(strategy_id, 0.0)
                + float(risk_dollars or 0.0)
            )

    def record_trade_close(self, strategy_id: str, pnl_dollars: float) -> None:
        """Update daily realized PnL after a trade close."""

        if strategy_id in self._strats:
            self._strats[strategy_id].daily_realized_dollars += float(pnl_dollars or 0.0)

    def on_new_day(self) -> None:
        """Reset daily counters."""

        for state in self._strats.values():
            state.daily_realized_dollars = 0.0
        self._portfolio_halted = False
        self._pending_entry_risk_by_strategy.clear()

    def entry_risk_context(self, strategy_id: str, risk_dollars: float) -> dict[str, float]:
        return build_swing_entry_risk_context(self._snapshot(), strategy_id, risk_dollars)

    @property
    def total_open_risk_dollars(self) -> float:
        return sum(state.open_risk_dollars for state in self._strats.values())

    @property
    def portfolio_unit_risk_dollars(self) -> float:
        return self._portfolio_unit_risk

    def _snapshot(self) -> SwingPortfolioRiskSnapshot:
        return SwingPortfolioRiskSnapshot(
            heat_cap_R=self._heat_cap_R,
            portfolio_daily_stop_R=self._portfolio_daily_stop_R,
            portfolio_unit_risk_dollars=self._portfolio_unit_risk,
            strategy_slots={
                sid: SwingStrategyRiskSlot(
                    strategy_id=state.strategy_id,
                    priority=state.priority,
                    unit_risk_dollars=state.unit_risk_dollars,
                    max_heat_R=state.max_heat_R,
                    daily_stop_R=state.daily_stop_R,
                )
                for sid, state in self._strats.items()
            },
            strategy_open_risk_dollars={
                sid: state.open_risk_dollars + self._pending_entry_risk_by_strategy.get(sid, 0.0)
                for sid, state in self._strats.items()
            },
            strategy_daily_realized_dollars={sid: state.daily_realized_dollars for sid, state in self._strats.items()},
            portfolio_halted=self._portfolio_halted,
            simulate_live_r_normalization=self._simulate_live_r,
            reserve_idle_higher_priority=self._reserve_idle_higher_priority,
        )


class SwingLivePortfolioRiskAdapter:
    """Live OMS adapter over the shared swing portfolio risk evaluator."""

    def __init__(self, risk_config: Any):
        self._risk_config = risk_config

    async def check_entry(
        self,
        *,
        strategy_id: str,
        new_risk_dollars: float,
        strat_cfg: Any,
        strat_risk: Any,
        port_risk: Any,
        get_strategy_risk: Callable[[str], Awaitable[Any]],
    ) -> SwingPortfolioRiskDecision:
        slots: dict[str, SwingStrategyRiskSlot] = {}
        open_risk: dict[str, float] = {}
        daily_realized: dict[str, float] = {}

        for sid, cfg in self._risk_config.strategy_configs.items():
            state = strat_risk if sid == strategy_id else await get_strategy_risk(sid)
            slots[sid] = SwingStrategyRiskSlot(
                strategy_id=sid,
                priority=int(getattr(cfg, "priority", 99)),
                unit_risk_dollars=float(getattr(cfg, "unit_risk_dollars", 0.0) or 0.0),
                max_heat_R=float(getattr(cfg, "max_heat_R", 0.0) or 0.0),
                daily_stop_R=float(getattr(cfg, "daily_stop_R", 0.0) or 0.0),
            )
            slot_unit = slots[sid].unit_risk_dollars
            open_dollars = float(getattr(state, "open_risk_dollars", 0.0) or 0.0)
            if open_dollars == 0.0 and slot_unit > 0:
                open_dollars = float(getattr(state, "open_risk_R", 0.0) or 0.0) * slot_unit
            daily_dollars = float(getattr(state, "daily_realized_pnl", 0.0) or 0.0)
            if daily_dollars == 0.0 and slot_unit > 0:
                daily_dollars = float(getattr(state, "daily_realized_R", 0.0) or 0.0) * slot_unit
            open_risk[sid] = open_dollars
            daily_realized[sid] = daily_dollars

        portfolio_unit = float(getattr(self._risk_config, "portfolio_urd", 0.0) or 0.0)
        if portfolio_unit <= 0:
            portfolio_unit = float(getattr(strat_cfg, "unit_risk_dollars", 0.0) or 0.0)
        pending_r = float(getattr(port_risk, "pending_entry_risk_R", 0.0) or 0.0)
        portfolio_daily_dollars = float(getattr(port_risk, "daily_realized_pnl", 0.0) or 0.0)
        if portfolio_daily_dollars == 0.0 and portfolio_unit > 0:
            portfolio_daily_dollars = float(getattr(port_risk, "daily_realized_R", 0.0) or 0.0) * portfolio_unit
        snapshot = SwingPortfolioRiskSnapshot(
            heat_cap_R=float(getattr(self._risk_config, "heat_cap_R", 0.0) or 0.0),
            portfolio_daily_stop_R=float(getattr(self._risk_config, "portfolio_daily_stop_R", 0.0) or 0.0),
            portfolio_unit_risk_dollars=portfolio_unit,
            strategy_slots=slots,
            strategy_open_risk_dollars=open_risk,
            strategy_daily_realized_dollars=daily_realized,
            portfolio_daily_realized_dollars=portfolio_daily_dollars,
            portfolio_pending_entry_risk_dollars=pending_r * portfolio_unit,
            portfolio_halted=bool(getattr(port_risk, "halted", False)),
        )
        return evaluate_swing_entry(snapshot, strategy_id, new_risk_dollars)
