"""Portfolio manager — synchronous rule checker for multi-strategy coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import structlog

from crypto_trader.core.models import Side
from crypto_trader.instrumentation.lineage import stable_hash
from crypto_trader.portfolio.config import PortfolioConfig
from crypto_trader.portfolio.state import OpenRisk, PortfolioState

log = structlog.get_logger()


@dataclass(frozen=True)
class PortfolioRuleResult:
    """Result of a portfolio entry check."""

    approved: bool
    denial_reason: str | None = None
    size_multiplier: float = 1.0
    rule_evaluations: list[dict] = field(default_factory=list)
    state_before: dict = field(default_factory=dict)
    state_after_preview: dict = field(default_factory=dict)
    allocation: dict | None = None
    blocking_rule: str = ""
    rule_event_id: str = ""
    risk_decision_id: str = ""
    request: dict = field(default_factory=dict)


class PortfolioManager:
    """Evaluates portfolio-level risk rules before allowing new entries.

    Rule sequence (9 checks, evaluated in order):
      1. Strategy enabled
      2. Max total positions
      3. Per-strategy max concurrent
      4. Heat cap (total open risk)
      5. Directional cap (with priority reservation)
      6. Symbol exposure cap / collision
      7. Portfolio daily stop
      8. Per-strategy daily stop
      9. Drawdown tiers (cascading size multiplier)

    Any rule can deny entry. Drawdown tiers may reduce size via multiplier.
    """

    def __init__(self, config: PortfolioConfig, state: PortfolioState) -> None:
        self.config = config
        self.state = state

    def check_entry(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        new_risk_R: float,
    ) -> PortfolioRuleResult:
        """Check whether a new entry is allowed by portfolio rules.

        Returns PortfolioRuleResult with approved=True and any size multiplier,
        or approved=False with denial_reason.
        """
        request = {
            "strategy_id": strategy_id,
            "symbol": symbol,
            "direction": direction.value,
            "requested_risk_R": new_risk_R,
        }
        state_before = self._state_snapshot()
        evaluations: list[dict] = []

        blocked_reason = getattr(self, "entries_blocked_reason", "")
        if blocked_reason:
            evaluations.append(self._evaluation(
                "reconciliation_block",
                False,
                reason=blocked_reason,
            ))
            return self._result(
                approved=False,
                denial_reason=blocked_reason,
                rule_evaluations=evaluations,
                state_before=state_before,
                request=request,
                blocking_rule="reconciliation_block",
            )

        cfg = self.config
        state = self.state

        # Rule 1: Strategy enabled
        alloc = cfg.get_strategy(strategy_id)
        if alloc is None:
            reason = f"strategy '{strategy_id}' not in portfolio config"
            evaluations.append(self._evaluation("strategy_enabled", False, reason=reason))
            return self._result(False, reason, evaluations, state_before, request, blocking_rule="strategy_enabled")
        alloc_dict = {
            "strategy_id": alloc.strategy_id,
            "enabled": alloc.enabled,
            "base_risk_pct": alloc.base_risk_pct,
            "max_concurrent": alloc.max_concurrent,
            "daily_stop_R": alloc.daily_stop_R,
            "priority": alloc.priority,
        }
        if not alloc.enabled:
            reason = f"strategy '{strategy_id}' disabled"
            evaluations.append(self._evaluation("strategy_enabled", False, reason=reason, allocation=alloc_dict))
            return self._result(False, reason, evaluations, state_before, request, allocation=alloc_dict, blocking_rule="strategy_enabled")
        evaluations.append(self._evaluation("strategy_enabled", True, allocation=alloc_dict))

        # Rule 2: Max total positions
        total_positions = state.total_positions()
        evaluations.append(self._evaluation(
            "max_total_positions",
            total_positions < cfg.max_total_positions,
            actual=total_positions,
            threshold=cfg.max_total_positions,
        ))
        if state.total_positions() >= cfg.max_total_positions:
            return self._result(False, "max_total_positions reached", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="max_total_positions")

        # Rule 3: Per-strategy max concurrent
        strategy_positions = state.strategy_position_count(strategy_id)
        evaluations.append(self._evaluation(
            "strategy_max_concurrent",
            strategy_positions < alloc.max_concurrent,
            actual=strategy_positions,
            threshold=alloc.max_concurrent,
        ))
        if strategy_positions >= alloc.max_concurrent:
            return self._result(False, f"strategy '{strategy_id}' max_concurrent reached", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="strategy_max_concurrent")

        # Rule 4: Heat cap (total open risk)
        projected_heat = state.total_heat_R() + new_risk_R
        evaluations.append(self._evaluation(
            "heat_cap_R",
            projected_heat <= cfg.heat_cap_R,
            actual=projected_heat,
            threshold=cfg.heat_cap_R,
        ))
        if projected_heat > cfg.heat_cap_R:
            return self._result(False, "heat_cap_R exceeded", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="heat_cap_R")

        # Rule 5: Directional cap (with priority reservation)
        dir_risk = state.directional_risk_R(direction) + new_risk_R
        effective_cap = cfg.directional_cap_R
        if cfg.priority_headroom_R > 0 and alloc.priority >= cfg.priority_reserve_threshold:
            remaining = cfg.directional_cap_R - state.directional_risk_R(direction)
            headroom_passed = remaining > cfg.priority_headroom_R
            evaluations.append(self._evaluation(
                "directional_priority_headroom",
                headroom_passed,
                actual=remaining,
                threshold=cfg.priority_headroom_R,
                allocation=alloc_dict,
            ))
            if remaining <= cfg.priority_headroom_R:
                reason = (
                    f"directional_cap_R headroom reserved (remaining={remaining:.2f}R, "
                    f"headroom={cfg.priority_headroom_R:.2f}R)"
                )
                return self._result(False, reason, evaluations, state_before, request, allocation=alloc_dict, blocking_rule="directional_priority_headroom")
        evaluations.append(self._evaluation(
            "directional_cap_R",
            dir_risk <= effective_cap,
            actual=dir_risk,
            threshold=effective_cap,
        ))
        if dir_risk > effective_cap:
            return self._result(False, "directional_cap_R exceeded", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="directional_cap_R")

        # Rule 6: Symbol exposure cap / collision
        deny = self._check_symbol_collision(strategy_id, symbol, direction, new_risk_R)
        symbol_eval = getattr(deny, "rule_evaluations", []) if deny is not None else []
        if symbol_eval:
            evaluations.extend(symbol_eval)
        else:
            evaluations.append(self._evaluation(
                "symbol_exposure",
                True,
                actual=state.symbol_risk_R(symbol, direction) + new_risk_R,
                threshold=cfg.symbol_exposure_cap_R,
                mode=cfg.symbol_collision,
            ))
        if deny is not None:
            return self._result(False, deny.denial_reason, evaluations, state_before, request, allocation=alloc_dict, blocking_rule=deny.blocking_rule or "symbol_exposure")

        # Rule 7: Portfolio daily stop
        evaluations.append(self._evaluation(
            "portfolio_daily_stop_R",
            state.portfolio_daily_pnl_R > -cfg.portfolio_daily_stop_R,
            actual=state.portfolio_daily_pnl_R,
            threshold=-cfg.portfolio_daily_stop_R,
        ))
        if state.portfolio_daily_pnl_R <= -cfg.portfolio_daily_stop_R:
            return self._result(False, "portfolio_daily_stop_R hit", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="portfolio_daily_stop_R")

        # Rule 8: Per-strategy daily stop
        strategy_daily = state.strategy_daily_pnl_R(strategy_id)
        evaluations.append(self._evaluation(
            "strategy_daily_stop_R",
            strategy_daily > -alloc.daily_stop_R,
            actual=strategy_daily,
            threshold=-alloc.daily_stop_R,
        ))
        if strategy_daily <= -alloc.daily_stop_R:
            return self._result(False, f"strategy '{strategy_id}' daily_stop_R hit", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="strategy_daily_stop_R")

        # Rule 9: Drawdown tiers (cascading multiplier)
        multiplier = self._dd_multiplier()
        evaluations.append(self._evaluation(
            "drawdown_tier",
            multiplier > 0.0,
            actual=state.dd_pct(),
            threshold=0.0,
            size_multiplier=multiplier,
        ))
        if multiplier <= 0.0:
            return self._result(False, "drawdown tier blocks all entries", evaluations, state_before, request, allocation=alloc_dict, blocking_rule="drawdown_tier")

        log.debug(
            "portfolio.entry_approved",
            strategy=strategy_id,
            symbol=symbol,
            direction=direction.value,
            risk_R=new_risk_R,
            multiplier=multiplier,
        )
        return self._result(
            True,
            None,
            evaluations,
            state_before,
            request,
            size_multiplier=multiplier,
            allocation=alloc_dict,
        )

    def register_entry(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        risk_R: float,
        entry_time=None,
        risk_id: str = "",
        position_instance_id: str = "",
        intent_id: str = "",
        client_order_id: str = "",
        order_id: str = "",
        exchange_order_id: str = "",
        order_qty: float = 0.0,
        fill_qty: float = 0.0,
        fill_id: str = "",
    ) -> None:
        """Record a new open risk after entry fill."""
        if risk_id:
            existing = self.state.find_risk(risk_id)
            if existing is not None:
                if fill_id and fill_id in existing.applied_fill_ids:
                    return
                existing.risk_R += risk_R
                existing.filled_qty += fill_qty
                existing.order_qty = max(existing.order_qty, order_qty)
                if fill_id:
                    existing.applied_fill_ids.append(fill_id)
                log.debug(
                    "portfolio.entry_risk_updated",
                    strategy=strategy_id,
                    symbol=symbol,
                    risk_id=risk_id,
                    risk_R=existing.risk_R,
                    filled_qty=existing.filled_qty,
                    total_heat=self.state.total_heat_R(),
                )
                return

        self.state.add_risk(OpenRisk(
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            risk_R=risk_R,
            entry_time=entry_time,
            risk_id=risk_id,
            position_instance_id=position_instance_id,
            intent_id=intent_id,
            client_order_id=client_order_id,
            order_id=order_id,
            exchange_order_id=exchange_order_id,
            order_qty=order_qty,
            filled_qty=fill_qty,
            applied_fill_ids=[fill_id] if fill_id else [],
        ))
        log.debug(
            "portfolio.entry_registered",
            strategy=strategy_id,
            symbol=symbol,
            direction=direction.value,
            risk_R=risk_R,
            total_heat=self.state.total_heat_R(),
        )

    def register_exit(
        self,
        strategy_id: str,
        symbol: str,
        pnl_R: float,
        *,
        risk_id: str = "",
        order_refs: set[str] | None = None,
    ) -> None:
        """Remove an open risk and record daily P&L."""
        removed = self.state.remove_risks(
            strategy_id,
            symbol,
            risk_id=risk_id,
            order_refs=order_refs,
            remove_all=not risk_id and not order_refs,
        )
        if not removed:
            log.warning(
                "portfolio.exit_no_matching_risk",
                strategy=strategy_id,
                symbol=symbol,
            )

        # Update daily P&L
        current = self.state.daily_pnl_R.get(strategy_id, 0.0)
        self.state.daily_pnl_R[strategy_id] = current + pnl_R
        self.state.portfolio_daily_pnl_R += pnl_R

        log.debug(
            "portfolio.exit_registered",
            strategy=strategy_id,
            symbol=symbol,
            pnl_R=pnl_R,
            total_heat=self.state.total_heat_R(),
        )

    def update_equity(self, equity: float) -> None:
        """Update portfolio equity (call periodically or after fills)."""
        self.state.update_equity(equity)

    def maybe_reset_daily(self, today: date) -> None:
        """Reset daily counters if the day has changed."""
        if self.state.current_day != today:
            self.state.reset_daily(today)

    def _check_symbol_collision(
        self,
        strategy_id: str,
        symbol: str,
        direction: Side,
        new_risk_R: float,
    ) -> PortfolioRuleResult | None:
        """Check symbol collision rules. Returns denial result or None if OK."""
        mode = self.config.symbol_collision

        if mode == "allow":
            return None

        # Check if another strategy has an open risk on this symbol
        other_risks = [
            r for r in self.state.open_risks
            if r.symbol == symbol and r.strategy_id != strategy_id
        ]

        if mode == "block" and other_risks:
            blockers = ", ".join(r.strategy_id for r in other_risks)
            return PortfolioRuleResult(
                False,
                f"symbol_collision=block: {symbol} already held by {blockers}",
                rule_evaluations=[self._evaluation(
                    "symbol_collision",
                    False,
                    mode=mode,
                    blockers=blockers,
                )],
                blocking_rule="symbol_collision",
            )

        if mode == "cap":
            current_sym_risk = self.state.symbol_risk_R(symbol, direction)
            if current_sym_risk + new_risk_R > self.config.symbol_exposure_cap_R:
                return PortfolioRuleResult(
                    False,
                    f"symbol_exposure_cap_R exceeded for {symbol} {direction.value} "
                    f"(current={current_sym_risk:.2f}R + new={new_risk_R:.2f}R "
                    f"> cap={self.config.symbol_exposure_cap_R:.2f}R)",
                    rule_evaluations=[self._evaluation(
                        "symbol_exposure",
                        False,
                        actual=current_sym_risk + new_risk_R,
                        threshold=self.config.symbol_exposure_cap_R,
                        mode=mode,
                    )],
                    blocking_rule="symbol_exposure",
                )

        return None

    def _state_snapshot(self) -> dict:
        return {
            "equity": self.state.equity,
            "peak_equity": self.state.peak_equity,
            "drawdown_pct": self.state.dd_pct(),
            "total_heat_R": self.state.total_heat_R(),
            "total_positions": self.state.total_positions(),
            "portfolio_daily_pnl_R": self.state.portfolio_daily_pnl_R,
            "daily_pnl_R": dict(self.state.daily_pnl_R),
            "open_risks": [
                {
                    "strategy_id": risk.strategy_id,
                    "symbol": risk.symbol,
                    "direction": risk.direction.value,
                    "risk_R": risk.risk_R,
                    "entry_time": str(risk.entry_time) if risk.entry_time else None,
                    "risk_id": risk.risk_id,
                    "position_instance_id": risk.position_instance_id,
                    "intent_id": risk.intent_id,
                    "client_order_id": risk.client_order_id,
                    "order_id": risk.order_id,
                    "exchange_order_id": risk.exchange_order_id,
                    "order_qty": risk.order_qty,
                    "filled_qty": risk.filled_qty,
                }
                for risk in self.state.open_risks
            ],
        }

    @staticmethod
    def _evaluation(rule: str, passed: bool, **details) -> dict:
        payload = {
            "rule": rule,
            "passed": passed,
        }
        payload.update({key: value for key, value in details.items() if value is not None})
        if "reason" not in payload and not passed:
            payload["reason"] = rule
        return payload

    def _result(
        self,
        approved: bool,
        denial_reason: str | None,
        rule_evaluations: list[dict],
        state_before: dict,
        request: dict,
        *,
        size_multiplier: float = 1.0,
        allocation: dict | None = None,
        blocking_rule: str = "",
    ) -> PortfolioRuleResult:
        effective_risk = float(request.get("requested_risk_R", 0.0) or 0.0) * size_multiplier
        state_after_preview = dict(state_before)
        if approved:
            state_after_preview["total_heat_R"] = state_before.get("total_heat_R", 0.0) + effective_risk
            state_after_preview["total_positions"] = state_before.get("total_positions", 0) + 1
        result_seed = {
            "request": request,
            "approved": approved,
            "denial_reason": denial_reason,
            "size_multiplier": size_multiplier,
            "state_before": state_before,
            "rule_evaluations": rule_evaluations,
        }
        rule_event_id = stable_hash({"portfolio_rule": result_seed})
        risk_decision_id = stable_hash({"risk_decision": result_seed})
        return PortfolioRuleResult(
            approved=approved,
            denial_reason=denial_reason,
            size_multiplier=size_multiplier,
            rule_evaluations=rule_evaluations,
            state_before=state_before,
            state_after_preview=state_after_preview,
            allocation=allocation,
            blocking_rule=blocking_rule,
            rule_event_id=rule_event_id,
            risk_decision_id=risk_decision_id,
            request=request,
        )

    def _dd_multiplier(self) -> float:
        """Compute sizing multiplier from drawdown tiers."""
        dd = self.state.dd_pct()
        multiplier = 1.0
        for threshold, mult in self.config.dd_tiers:
            if dd >= threshold:
                multiplier = mult
            else:
                break
        return multiplier
