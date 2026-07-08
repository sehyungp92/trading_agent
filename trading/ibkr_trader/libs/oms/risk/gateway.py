"""Risk gateway for pre-trade checks.

Merged from swing_trader, momentum_trader, and stock_trader families.
Supports the union of all optional parameters:
  - market_calendar (swing): market holiday / half-day checks
  - portfolio_checker (momentum): cross-strategy portfolio rules
  - account_gate (all): cross-family account risk
  - portfolio_weekly_stop_R (momentum/stock): weekly stop loss via RiskConfig
"""
import dataclasses
import math
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Optional, TYPE_CHECKING

from ..models.order import OMSOrder, OrderRole
from ..models.risk_state import StrategyRiskState, PortfolioRiskState
from ..config.risk_config import RiskConfig
from .calendar import EventCalendar

if TYPE_CHECKING:
    from libs.config.market_calendar import MarketCalendar
    from .portfolio_rules import PortfolioRuleChecker
    from .swing_portfolio_adapter import SwingLivePortfolioRiskAdapter

logger = logging.getLogger(__name__)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class RiskGateway:
    """Pre-trade risk checks. Only gates ENTRY orders; exits always allowed.

    Union of all three strategy-family gateways.  Optional parameters
    default to ``None``; when ``None`` the corresponding check is skipped.
    """

    def __init__(
        self,
        config: RiskConfig,
        calendar: EventCalendar,
        get_strategy_risk: Callable[[str], Awaitable[StrategyRiskState]],
        get_portfolio_risk: Callable[[], Awaitable[PortfolioRiskState]],
        get_working_order_count: Callable[[str], Awaitable[int]] = None,
        market_calendar: Optional["MarketCalendar"] = None,
        portfolio_checker: Optional["PortfolioRuleChecker"] = None,
        portfolio_risk_adapter: Optional["SwingLivePortfolioRiskAdapter"] = None,
        account_gate: Optional[object] = None,
        family_id: str = "unknown",
    ):
        self._config = config
        self._calendar = calendar
        self._get_strat_risk = get_strategy_risk
        self._get_port_risk = get_portfolio_risk
        self._get_working_count = get_working_order_count
        self._market_cal = market_calendar
        self._portfolio_checker = portfolio_checker
        self._portfolio_risk_adapter = portfolio_risk_adapter
        self._account_gate = account_gate
        self._family_id = family_id
        self._last_decision_context: dict[str, dict[str, Any]] = {}

    # ── Validation (from momentum/stock families) ─────────────────────

    @staticmethod
    def _validate_entry_order(order: OMSOrder) -> Optional[str]:
        """Reject malformed entries before they can distort live risk state."""
        if order.qty <= 0:
            return f"ENTRY qty must be positive: {order.qty}"

        instrument = order.instrument
        if not instrument:
            return "Order missing instrument"
        if not math.isfinite(instrument.point_value) or instrument.point_value <= 0:
            return f"Instrument point_value must be positive: {instrument.point_value}"

        risk_ctx = order.risk_context
        if risk_ctx is None:
            return "ENTRY order missing risk_context"

        planned_entry = risk_ctx.planned_entry_price
        stop_for_risk = risk_ctx.stop_for_risk
        if not math.isfinite(planned_entry) or not math.isfinite(stop_for_risk):
            return "ENTRY risk prices must be finite"
        if planned_entry <= 0 or stop_for_risk <= 0:
            return (
                "ENTRY risk prices must be positive: "
                f"entry={planned_entry} stop={stop_for_risk}"
            )

        risk_per_contract = abs(planned_entry - stop_for_risk) * instrument.point_value
        if not math.isfinite(risk_per_contract) or risk_per_contract <= 0:
            return (
                "ENTRY risk distance must be positive: "
                f"entry={planned_entry} stop={stop_for_risk}"
            )

        return None

    # ── Main entry check ──────────────────────────────────────────────

    @staticmethod
    def _entry_risk_dollars(order: OMSOrder) -> float:
        risk_ctx = order.risk_context
        instrument = order.instrument
        if risk_ctx is None or instrument is None:
            return 0.0
        return (
            order.qty
            * abs(risk_ctx.planned_entry_price - risk_ctx.stop_for_risk)
            * instrument.point_value
        )

    @staticmethod
    def _portfolio_rule_context(order: OMSOrder) -> dict:
        risk_ctx = order.risk_context

        def _first(*values):
            for value in values:
                if value not in (None, "", [], {}):
                    return value
            return ""

        return {
            "trace_id": str(_first(
                getattr(risk_ctx, "trace_id", ""),
                getattr(order, "trace_id", ""),
                getattr(risk_ctx, "intent_id", ""),
                order.client_order_id,
                order.oms_order_id,
            )),
            "signal_id": str(_first(
                getattr(risk_ctx, "signal_id", ""),
                getattr(order, "signal_id", ""),
                order.client_order_id,
                getattr(risk_ctx, "intent_id", ""),
            )),
            "bar_id": str(_first(
                getattr(risk_ctx, "bar_id", ""),
                getattr(order, "bar_id", ""),
            )),
            "exchange_timestamp": _first(
                getattr(risk_ctx, "exchange_timestamp", None),
                getattr(order, "exchange_timestamp", None),
                order.submitted_at,
                order.created_at,
            ) or None,
            "lineage_context": _first(
                getattr(risk_ctx, "lineage_context", None),
                getattr(order, "lineage_context", None),
            ) or None,
        }

    def _record_gateway_decision(
        self,
        order: OMSOrder,
        *,
        decision: str,
        reason: str = "",
        gate: str = "",
        strat_cfg: Any = None,
        strat_risk: Any = None,
        port_risk: Any = None,
        requested_risk_dollars: float = 0.0,
        requested_risk_R: float = 0.0,
        requested_portfolio_risk_R: float = 0.0,
        approved_qty: int | None = None,
        approved_risk_dollars: float | None = None,
        approved_risk_R: float | None = None,
        approved_portfolio_risk_R: float | None = None,
        working_order_count: int | None = None,
        portfolio_rule_result: Any = None,
        account_gate_result: str = "",
        session_gate_result: str = "",
        market_gate_result: str = "",
    ) -> dict[str, Any]:
        risk_ctx = order.risk_context
        strategy_id = str(getattr(order, "strategy_id", "") or "")
        symbol = order.instrument.symbol if order.instrument else ""
        if approved_qty is None:
            approved_qty = int(order.qty) if decision in {"approve", "scale", "route"} else 0
        if approved_risk_dollars is None:
            approved_risk_dollars = requested_risk_dollars if approved_qty else 0.0
        if approved_risk_R is None:
            approved_risk_R = requested_risk_R if approved_qty else 0.0
        if approved_portfolio_risk_R is None:
            approved_portfolio_risk_R = requested_portfolio_risk_R if approved_qty else 0.0

        portfolio_ref = getattr(risk_ctx, "portfolio_decision_ref", "") if risk_ctx else ""
        portfolio_rule_payload = _plain(portfolio_rule_result) if portfolio_rule_result is not None else {}
        if portfolio_rule_payload and not portfolio_ref:
            portfolio_ref = str(portfolio_rule_payload.get("rule_trace_id", "") or "")

        payload = {
            "gateway_checked": True,
            "gateway_gate": gate,
            "decision": decision,
            "reason": reason,
            "intent_id": getattr(risk_ctx, "intent_id", "") if risk_ctx else "",
            "strategy_id": strategy_id,
            "family_id": self._family_id,
            "portfolio_id": getattr(self._config, "portfolio_id", ""),
            "symbol": symbol,
            "side": getattr(getattr(order, "side", None), "value", ""),
            "role": getattr(getattr(order, "role", None), "value", ""),
            "requested_qty": int(order.qty),
            "approved_qty": approved_qty,
            "requested_risk_dollars": float(requested_risk_dollars or 0.0),
            "approved_risk_dollars": float(approved_risk_dollars or 0.0),
            "requested_risk_R": float(requested_risk_R or 0.0),
            "approved_risk_R": float(approved_risk_R or 0.0),
            "requested_portfolio_risk_R": float(requested_portfolio_risk_R or 0.0),
            "approved_portfolio_risk_R": float(approved_portfolio_risk_R or 0.0),
            "portfolio_size_mult": getattr(risk_ctx, "portfolio_size_mult", 1.0) if risk_ctx else 1.0,
            "portfolio_decision_ref": portfolio_ref,
            "daily_stop_usage": {
                "strategy_realized_R": float(getattr(strat_risk, "daily_realized_R", 0.0) or 0.0),
                "strategy_stop_R": float(getattr(strat_cfg, "daily_stop_R", 0.0) or 0.0),
                "portfolio_realized_R": float(getattr(port_risk, "daily_realized_R", 0.0) or 0.0),
                "portfolio_stop_R": float(getattr(self._config, "portfolio_daily_stop_R", 0.0) or 0.0),
            },
            "weekly_stop_usage": {
                "strategy_realized_R": float(getattr(strat_risk, "weekly_realized_R", 0.0) or 0.0),
                "portfolio_realized_R": float(getattr(port_risk, "weekly_realized_R", 0.0) or 0.0),
                "portfolio_stop_R": float(getattr(self._config, "portfolio_weekly_stop_R", 0.0) or 0.0),
            },
            "strategy_heat": {
                "open_risk_R": float(getattr(strat_risk, "open_risk_R", 0.0) or 0.0),
                "new_risk_R": float(requested_risk_R or 0.0),
                "approved_new_risk_R": float(approved_risk_R or 0.0),
                "max_heat_R": float(getattr(strat_cfg, "max_heat_R", 0.0) or 0.0),
                "daily_stop_R": float(getattr(strat_cfg, "daily_stop_R", 0.0) or 0.0),
                "unit_risk_dollars": float(getattr(strat_cfg, "unit_risk_dollars", 0.0) or 0.0),
                "max_working_orders": int(getattr(strat_cfg, "max_working_orders", 0) or 0),
                "working_order_count": working_order_count,
            },
            "portfolio_heat": {
                "open_risk_R": float(getattr(port_risk, "open_risk_R", 0.0) or 0.0),
                "pending_entry_risk_R": float(getattr(port_risk, "pending_entry_risk_R", 0.0) or 0.0),
                "new_risk_R": float(requested_portfolio_risk_R or 0.0),
                "approved_new_risk_R": float(approved_portfolio_risk_R or 0.0),
                "heat_cap_R": float(getattr(self._config, "heat_cap_R", 0.0) or 0.0),
                "portfolio_urd": float(getattr(self._config, "portfolio_urd", 0.0) or 0.0),
            },
            "portfolio_rule": {
                "checked": bool(portfolio_rule_result),
                "rule_trace_id": portfolio_ref,
                "approved": getattr(portfolio_rule_result, "approved", None) if portfolio_rule_result is not None else None,
                "denial_reason": getattr(portfolio_rule_result, "denial_reason", "") if portfolio_rule_result is not None else "",
                "size_multiplier": getattr(portfolio_rule_result, "size_multiplier", None) if portfolio_rule_result is not None else None,
                "applied_rules": list(getattr(portfolio_rule_result, "applied_rules", ()) or ()),
            },
            "account_gate": {
                "checked": bool(account_gate_result),
                "result": "deny" if account_gate_result else "",
                "reason": account_gate_result,
            },
            "session_gate": {
                "checked": bool(session_gate_result),
                "result": "deny" if session_gate_result else "",
                "reason": session_gate_result,
            },
            "market_gate": {
                "checked": bool(market_gate_result),
                "result": "deny" if market_gate_result else "",
                "reason": market_gate_result,
            },
        }
        if risk_ctx is not None:
            risk_ctx.gateway_decision_context = payload
        self._last_decision_context[order.oms_order_id] = payload
        intent_id = getattr(risk_ctx, "intent_id", "") if risk_ctx else ""
        if intent_id:
            self._last_decision_context[intent_id] = payload
        return payload

    async def check_entry(
        self,
        order: OMSOrder,
        *,
        skip_account_gate: bool = False,
        reserved_entry_risk_R: float = 0.0,
    ) -> Optional[str]:
        """Returns denial reason string, or None if approved.

        Check order:
          1. global standdown
          2. event blackout
          3. market holiday (if calendar)
          4. session block
          5. strategy daily halt
          6. portfolio daily halt
          7. portfolio weekly halt (if > 0)
          8. max working orders
          9. heat cap
         10. order type
         11. portfolio rules (if checker)
         12. account gate (if gate)
        """
        strat_cfg = None
        strat_risk = None
        port_risk = None
        working_order_count: int | None = None
        requested_risk_dollars = 0.0
        requested_risk_R = 0.0
        requested_portfolio_risk_R = 0.0
        portfolio_rule_result = None

        def _deny(reason: str, gate: str, **extra) -> str:
            self._record_gateway_decision(
                order,
                decision="deny",
                reason=reason,
                gate=gate,
                strat_cfg=strat_cfg,
                strat_risk=strat_risk,
                port_risk=port_risk,
                requested_risk_dollars=requested_risk_dollars,
                requested_risk_R=requested_risk_R,
                requested_portfolio_risk_R=requested_portfolio_risk_R,
                working_order_count=working_order_count,
                portfolio_rule_result=portfolio_rule_result,
                **extra,
            )
            return reason

        def _approve(decision: str = "approve") -> None:
            self._record_gateway_decision(
                order,
                decision=decision,
                gate=decision,
                strat_cfg=strat_cfg,
                strat_risk=strat_risk,
                port_risk=port_risk,
                requested_risk_dollars=requested_risk_dollars,
                requested_risk_R=requested_risk_R,
                requested_portfolio_risk_R=requested_portfolio_risk_R,
                approved_qty=order.qty,
                approved_risk_dollars=new_risk_dollars,
                approved_risk_R=new_risk_R,
                approved_portfolio_risk_R=new_risk_portfolio_R,
                working_order_count=working_order_count,
                portfolio_rule_result=portfolio_rule_result,
            )

        # Exits/stops always allowed
        if order.role != OrderRole.ENTRY:
            return None

        # Must have risk context for entries
        if not order.risk_context:
            return _deny("ENTRY order missing risk_context", "validation")

        strat_cfg = self._config.strategy_configs.get(order.strategy_id)
        if not strat_cfg:
            return _deny(f"No risk config for strategy {order.strategy_id}", "config")

        validation_error = self._validate_entry_order(order)
        if validation_error:
            return _deny(validation_error, "validation")
        if not math.isfinite(strat_cfg.unit_risk_dollars) or strat_cfg.unit_risk_dollars <= 0:
            return _deny(
                "Strategy unit_risk_dollars must be positive: "
                f"{strat_cfg.unit_risk_dollars}",
                "config",
            )

        now_utc = datetime.now(timezone.utc)

        # 1. Global stand-down
        if self._config.global_standdown:
            return _deny("Global stand-down active", "global_standdown")

        # 2. Event blackout
        if self._calendar.is_blocked(now_utc):
            return _deny("Event blackout active", "event_blackout")

        # 3. Market holiday / half-day block (swing family)
        if self._market_cal:
            from libs.config.market_calendar import AssetClass
            instrument = order.instrument
            asset_class = (
                AssetClass.CME_FUTURES
                if instrument and instrument.venue in ("CME", "COMEX", "NYMEX")
                else AssetClass.EQUITY
            )
            holiday_block = self._market_cal.is_entry_blocked(now_utc, asset_class)
            if holiday_block:
                return _deny(holiday_block, "market_calendar", market_gate_result=holiday_block)

        # 4. Market session block (strategy-specific)
        session_block = strat_cfg.check_session_block(now_utc)
        if session_block:
            return _deny(session_block, "session", session_gate_result=session_block)

        # 5. Strategy daily halt
        strat_risk = await self._get_strat_risk(order.strategy_id)
        if strat_risk.halted:
            return _deny(f"Strategy halted: {strat_risk.halt_reason}", "strategy_halt")
        if self._portfolio_risk_adapter is None and strat_risk.daily_realized_R <= -strat_cfg.daily_stop_R:
            return _deny(
                f"Strategy daily stop: realized {strat_risk.daily_realized_R:.2f}R "
                f"<= -{strat_cfg.daily_stop_R}R",
                "strategy_daily_stop",
            )

        # 6. Portfolio daily halt
        port_risk = await self._get_port_risk()
        if reserved_entry_risk_R > 0:
            port_risk = dataclasses.replace(
                port_risk,
                pending_entry_risk_R=max(
                    0.0,
                    float(port_risk.pending_entry_risk_R or 0.0)
                    - reserved_entry_risk_R,
                ),
            )
        if port_risk.halted:
            return _deny(f"Portfolio halted: {port_risk.halt_reason}", "portfolio_halt")
        if self._portfolio_risk_adapter is None and port_risk.daily_realized_R <= -self._config.portfolio_daily_stop_R:
            return _deny(
                f"Portfolio daily stop: {port_risk.daily_realized_R:.2f}R",
                "portfolio_daily_stop",
            )

        # 7. Portfolio weekly halt (momentum/stock families)
        if (self._config.portfolio_weekly_stop_R > 0
                and port_risk.weekly_realized_R <= -self._config.portfolio_weekly_stop_R):
            return _deny(
                f"Portfolio weekly stop: {port_risk.weekly_realized_R:.2f}R "
                f"<= -{self._config.portfolio_weekly_stop_R}R",
                "portfolio_weekly_stop",
            )

        # 8. Max working orders
        if self._get_working_count and strat_cfg.max_working_orders > 0:
            working_order_count = await self._get_working_count(order.strategy_id)
            if working_order_count >= strat_cfg.max_working_orders:
                return _deny(
                    f"Max working orders ({strat_cfg.max_working_orders}) reached: {working_order_count} active",
                    "max_working_orders",
                )

        # 9. Heat cap check
        instrument = order.instrument
        risk_ctx = order.risk_context
        risk_per_contract = (
            abs(risk_ctx.planned_entry_price - risk_ctx.stop_for_risk)
            * instrument.point_value
        )
        new_risk_dollars = order.qty * risk_per_contract
        new_risk_R = (
            new_risk_dollars / strat_cfg.unit_risk_dollars
            if strat_cfg.unit_risk_dollars > 0
            else float("inf")
        )
        requested_risk_dollars = new_risk_dollars
        requested_risk_R = new_risk_R
        risk_ctx.risk_dollars = new_risk_dollars
        risk_ctx.unit_risk_dollars = strat_cfg.unit_risk_dollars

        # Normalize to portfolio_urd basis for portfolio-level heat cap (C1c fix:
        # prevents mixed R-unit bases when strategies have different URDs in shared OMS)
        portfolio_urd = self._config.portfolio_urd
        if portfolio_urd > 0 and strat_cfg.unit_risk_dollars > 0:
            new_risk_portfolio_R = new_risk_dollars / portfolio_urd
        else:
            new_risk_portfolio_R = new_risk_R
        requested_portfolio_risk_R = new_risk_portfolio_R

        portfolio_rules_checked = False
        if self._portfolio_checker and self._portfolio_risk_adapter is None:
            direction = "LONG" if order.side.value == "BUY" else "SHORT"
            rule_context = self._portfolio_rule_context(order)
            port_result = await self._portfolio_checker.check_entry(
                strategy_id=order.strategy_id,
                direction=direction,
                new_risk_R=new_risk_R,
                symbol=order.instrument.symbol if order.instrument else None,
                new_qty=order.qty,
                new_risk_dollars=new_risk_dollars,
                **rule_context,
            )
            portfolio_rule_result = port_result
            portfolio_rules_checked = True
            if getattr(port_result, "rule_trace_id", ""):
                risk_ctx.portfolio_decision_ref = port_result.rule_trace_id
            if not port_result.approved:
                return _deny(f"Portfolio rule: {port_result.denial_reason}", "portfolio_rule")
            if port_result.size_multiplier != 1.0:
                risk_ctx.portfolio_size_mult = port_result.size_multiplier
                adjusted_qty = max(1, int(order.qty * port_result.size_multiplier))
                new_risk_dollars = adjusted_qty * risk_per_contract
                new_risk_R = (
                    new_risk_dollars / strat_cfg.unit_risk_dollars
                    if strat_cfg.unit_risk_dollars > 0
                    else float("inf")
                )
                new_risk_portfolio_R = (
                    new_risk_dollars / portfolio_urd
                    if portfolio_urd > 0
                    else new_risk_R
                )
                logger.info(
                    "Portfolio size multiplier %.2fx applied to %s %s before heat checks",
                    port_result.size_multiplier, order.strategy_id, direction,
                )

        if self._portfolio_risk_adapter is not None:
            decision = await self._portfolio_risk_adapter.check_entry(
                strategy_id=order.strategy_id,
                new_risk_dollars=new_risk_dollars,
                strat_cfg=strat_cfg,
                strat_risk=strat_risk,
                port_risk=port_risk,
                get_strategy_risk=self._get_strat_risk,
            )
            if not decision.approved:
                return _deny(decision.reason, "portfolio_adapter")
        else:
            total_risk_R = port_risk.open_risk_R + port_risk.pending_entry_risk_R + new_risk_portfolio_R
            if total_risk_R > self._config.heat_cap_R:
                return _deny(
                    f"Heat cap breach: open {port_risk.open_risk_R:.2f}R + "
                    f"pending {port_risk.pending_entry_risk_R:.2f}R + "
                    f"new {new_risk_portfolio_R:.2f}R > cap {self._config.heat_cap_R}R",
                    "portfolio_heat_cap",
                )

            # 9b. Per-strategy heat ceiling (swing family): prevent one strategy
            # from monopolising the shared pool.
            if strat_cfg.max_heat_R > 0:
                strat_heat_R = strat_risk.open_risk_R + new_risk_R
                if strat_heat_R > strat_cfg.max_heat_R:
                    return _deny(
                        f"Strategy heat ceiling: {order.strategy_id} open "
                        f"{strat_risk.open_risk_R:.2f}R + new {new_risk_R:.2f}R "
                        f"> cap {strat_cfg.max_heat_R:.2f}R",
                        "strategy_heat_cap",
                    )

            # Priority-aware heat reservation (swing family): when remaining
            # heat is tight, reserve capacity for higher-priority strategies
            # that are IDLE (no open exposure).
            remaining_R = self._config.heat_cap_R - (port_risk.open_risk_R + port_risk.pending_entry_risk_R)
            if remaining_R < 2 * new_risk_portfolio_R:
                for other_cfg in self._config.strategy_configs.values():
                    if other_cfg.priority < strat_cfg.priority:
                        other_risk = await self._get_strat_risk(other_cfg.strategy_id)
                        if other_risk.open_risk_R == 0:
                            return _deny(
                                f"Heat cap reserved: {remaining_R:.2f}R remaining, "
                                f"priority strategy {other_cfg.strategy_id} may need it",
                                "priority_heat_reservation",
                            )

        # 10. Order type allowed
        if not strat_cfg.is_order_type_allowed(order.role, order.order_type):
            return _deny(f"Order type {order.order_type} not allowed for role {order.role}", "order_type")

        # 11. Cross-strategy portfolio rules (momentum / stock family)
        if self._portfolio_checker and not portfolio_rules_checked:
            direction = "LONG" if order.side.value == "BUY" else "SHORT"
            rule_context = self._portfolio_rule_context(order)
            port_result = await self._portfolio_checker.check_entry(
                strategy_id=order.strategy_id,
                direction=direction,
                new_risk_R=new_risk_R,
                symbol=order.instrument.symbol if order.instrument else None,
                new_qty=order.qty,
                new_risk_dollars=new_risk_dollars,
                **rule_context,
            )
            portfolio_rule_result = port_result
            if getattr(port_result, "rule_trace_id", ""):
                risk_ctx.portfolio_decision_ref = port_result.rule_trace_id
            if not port_result.approved:
                return _deny(f"Portfolio rule: {port_result.denial_reason}", "portfolio_rule")
            # Apply size multiplier to risk context for downstream sizing
            if port_result.size_multiplier != 1.0:
                risk_ctx.portfolio_size_mult = port_result.size_multiplier
                adjusted_qty = max(1, int(order.qty * port_result.size_multiplier))
                new_risk_dollars = adjusted_qty * risk_per_contract
                new_risk_R = (
                    new_risk_dollars / strat_cfg.unit_risk_dollars
                    if strat_cfg.unit_risk_dollars > 0
                    else float("inf")
                )
                new_risk_portfolio_R = (
                    new_risk_dollars / portfolio_urd
                    if portfolio_urd > 0
                    else new_risk_R
                )
                logger.info(
                    "Portfolio size multiplier %.2fx applied to %s %s",
                    port_result.size_multiplier, order.strategy_id, direction,
                )

        # Store computed risk for pending-entry tracking
        risk_ctx.risk_dollars = new_risk_dollars
        risk_ctx.unit_risk_dollars = strat_cfg.unit_risk_dollars

        # 12. Account-level cross-family risk gate. IntentHandler can skip this
        # and call check_account_gate() inside the RISK_APPROVED transaction.
        if not skip_account_gate:
            denial = await self.check_account_gate(
                order,
                reserved_entry_risk_dollars=(
                    reserved_entry_risk_R * float(self._config.portfolio_urd or 0.0)
                ),
            )
            if denial:
                return _deny(denial, "account_gate", account_gate_result=denial)

        _approve("scale" if getattr(risk_ctx, "portfolio_size_mult", 1.0) != 1.0 else "approve")
        return None  # Approved

    async def check_preapproved_entry(self, order: OMSOrder) -> Optional[str]:
        """Validate an entry whose family portfolio decision is already final.

        Replay/backtest family surfaces can be authoritative for accept/reduce/
        reject decisions, but the order should still travel through the OMS
        service and handler path. This check keeps non-family controls intact
        while deliberately skipping heat, directional, and family portfolio
        approval checks that have already been applied upstream.
        """

        strat_cfg = None
        strat_risk = None
        port_risk = None
        working_order_count: int | None = None
        requested_risk_dollars = 0.0
        requested_risk_R = 0.0
        requested_portfolio_risk_R = 0.0

        def _sync_requested_risk() -> None:
            nonlocal requested_risk_dollars, requested_risk_R, requested_portfolio_risk_R
            risk_ctx = order.risk_context
            if risk_ctx is None or strat_cfg is None:
                return
            risk_dollars = float(getattr(risk_ctx, "risk_dollars", 0.0) or 0.0)
            if not math.isfinite(risk_dollars) or risk_dollars <= 0:
                risk_dollars = self._entry_risk_dollars(order)
            requested_risk_dollars = float(risk_dollars or 0.0)
            if strat_cfg.unit_risk_dollars > 0:
                requested_risk_R = requested_risk_dollars / strat_cfg.unit_risk_dollars
            portfolio_urd = float(getattr(self._config, "portfolio_urd", 0.0) or strat_cfg.unit_risk_dollars or 0.0)
            if portfolio_urd > 0:
                requested_portfolio_risk_R = requested_risk_dollars / portfolio_urd
            else:
                requested_portfolio_risk_R = requested_risk_R

        def _deny(reason: str, gate: str, **extra) -> str:
            _sync_requested_risk()
            self._record_gateway_decision(
                order,
                decision="deny",
                reason=reason,
                gate=f"preapproved_{gate}",
                strat_cfg=strat_cfg,
                strat_risk=strat_risk,
                port_risk=port_risk,
                requested_risk_dollars=requested_risk_dollars,
                requested_risk_R=requested_risk_R,
                requested_portfolio_risk_R=requested_portfolio_risk_R,
                working_order_count=working_order_count,
                **extra,
            )
            return reason

        def _approve() -> None:
            _sync_requested_risk()
            self._record_gateway_decision(
                order,
                decision="approve",
                gate="preapproved",
                strat_cfg=strat_cfg,
                strat_risk=strat_risk,
                port_risk=port_risk,
                requested_risk_dollars=requested_risk_dollars,
                requested_risk_R=requested_risk_R,
                requested_portfolio_risk_R=requested_portfolio_risk_R,
                approved_qty=order.qty,
                approved_risk_dollars=requested_risk_dollars,
                approved_risk_R=requested_risk_R,
                approved_portfolio_risk_R=requested_portfolio_risk_R,
                working_order_count=working_order_count,
            )

        if order.role != OrderRole.ENTRY:
            return None
        if not order.risk_context:
            return _deny("ENTRY order missing risk_context", "validation")

        strat_cfg = self._config.strategy_configs.get(order.strategy_id)
        if not strat_cfg:
            return _deny(f"No risk config for strategy {order.strategy_id}", "config")

        validation_error = self._validate_entry_order(order)
        if validation_error:
            return _deny(validation_error, "validation")
        if not math.isfinite(strat_cfg.unit_risk_dollars) or strat_cfg.unit_risk_dollars <= 0:
            return _deny(
                "Strategy unit_risk_dollars must be positive: "
                f"{strat_cfg.unit_risk_dollars}",
                "config",
            )

        now_utc = datetime.now(timezone.utc)
        if self._config.global_standdown:
            return _deny("Global stand-down active", "global_standdown")
        if self._calendar.is_blocked(now_utc):
            return _deny("Event blackout active", "event_blackout")
        if self._market_cal:
            from libs.config.market_calendar import AssetClass

            instrument = order.instrument
            asset_class = (
                AssetClass.CME_FUTURES
                if instrument and instrument.venue in ("CME", "COMEX", "NYMEX")
                else AssetClass.EQUITY
            )
            holiday_block = self._market_cal.is_entry_blocked(now_utc, asset_class)
            if holiday_block:
                return _deny(holiday_block, "market_calendar", market_gate_result=holiday_block)

        session_block = strat_cfg.check_session_block(now_utc)
        if session_block:
            return _deny(session_block, "session", session_gate_result=session_block)

        strat_risk = await self._get_strat_risk(order.strategy_id)
        if strat_risk.halted:
            return _deny(f"Strategy halted: {strat_risk.halt_reason}", "strategy_halt")
        if self._portfolio_risk_adapter is None and strat_risk.daily_realized_R <= -strat_cfg.daily_stop_R:
            return _deny(
                f"Strategy daily stop: realized {strat_risk.daily_realized_R:.2f}R "
                f"<= -{strat_cfg.daily_stop_R}R",
                "strategy_daily_stop",
            )

        port_risk = await self._get_port_risk()
        if port_risk.halted:
            return _deny(f"Portfolio halted: {port_risk.halt_reason}", "portfolio_halt")
        if self._portfolio_risk_adapter is None and port_risk.daily_realized_R <= -self._config.portfolio_daily_stop_R:
            return _deny(
                f"Portfolio daily stop: {port_risk.daily_realized_R:.2f}R",
                "portfolio_daily_stop",
            )
        if (
            self._config.portfolio_weekly_stop_R > 0
            and port_risk.weekly_realized_R <= -self._config.portfolio_weekly_stop_R
        ):
            return _deny(
                f"Portfolio weekly stop: {port_risk.weekly_realized_R:.2f}R "
                f"<= -{self._config.portfolio_weekly_stop_R}R",
                "portfolio_weekly_stop",
            )

        if self._get_working_count and strat_cfg.max_working_orders > 0:
            working_order_count = await self._get_working_count(order.strategy_id)
            if working_order_count >= strat_cfg.max_working_orders:
                return _deny(
                    f"Max working orders ({strat_cfg.max_working_orders}) reached: "
                    f"{working_order_count} active",
                    "max_working_orders",
                )

        if not strat_cfg.is_order_type_allowed(order.role, order.order_type):
            return _deny(f"Order type {order.order_type} not allowed for role {order.role}", "order_type")

        risk_ctx = order.risk_context
        if not math.isfinite(risk_ctx.risk_dollars) or risk_ctx.risk_dollars <= 0:
            risk_ctx.risk_dollars = self._entry_risk_dollars(order)
        risk_ctx.unit_risk_dollars = strat_cfg.unit_risk_dollars
        _approve()
        return None

    async def check_account_gate(
        self,
        order: OMSOrder,
        conn=None,
        *,
        reserved_entry_risk_dollars: float = 0.0,
    ) -> Optional[str]:
        """Run only the account-global reservation gate for an approved ENTRY."""
        if order.role != OrderRole.ENTRY or self._account_gate is None:
            return None

        def _account_gate_denial(reason: str) -> str:
            risk_ctx = order.risk_context
            if risk_ctx is None:
                self._record_gateway_decision(
                    order,
                    decision="deny",
                    reason=reason,
                    gate="account_gate",
                    account_gate_result=reason,
                )
                return reason
            context = dict(getattr(risk_ctx, "gateway_decision_context", {}) or {})
            context.update(
                {
                    "gateway_checked": True,
                    "gateway_gate": "account_gate",
                    "decision": "deny",
                    "reason": reason,
                    "approved_qty": 0,
                    "approved_risk_dollars": 0.0,
                    "approved_risk_R": 0.0,
                    "account_gate": {
                        "checked": True,
                        "result": "deny",
                        "reason": reason,
                    },
                }
            )
            risk_ctx.gateway_decision_context = context
            self._last_decision_context[order.oms_order_id] = context
            if getattr(risk_ctx, "intent_id", ""):
                self._last_decision_context[risk_ctx.intent_id] = context
            return reason

        if not order.risk_context:
            return _account_gate_denial("ENTRY order missing risk_context")

        risk_dollars = order.risk_context.risk_dollars
        if not math.isfinite(risk_dollars) or risk_dollars <= 0:
            risk_dollars = self._entry_risk_dollars(order)
            order.risk_context.risk_dollars = risk_dollars
        if not math.isfinite(risk_dollars) or risk_dollars <= 0:
            return _account_gate_denial("ENTRY risk_dollars must be positive before account gate")

        decision = await self._account_gate.check_entry(
            self._family_id,
            risk_dollars,
            conn=conn,
            reserved_risk_dollars=reserved_entry_risk_dollars,
        )
        if not decision.approved:
            return _account_gate_denial(f"Account gate: {decision.reason}")
        context = dict(getattr(order.risk_context, "gateway_decision_context", {}) or {})
        if context:
            context["account_gate"] = {"checked": True, "result": "pass", "reason": ""}
            order.risk_context.gateway_decision_context = context
        return None
