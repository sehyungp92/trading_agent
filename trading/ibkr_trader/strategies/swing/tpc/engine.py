"""Live adapter for TPC.

STRAT-1 / Phase D: TPC ships a full live execution shell over the
ETFCoreLiveEngine base. The base class owns state + on_bar/on_fill core
calls but has no scheduler, no action dispatcher, and no OMS event loop.
This module supplies all three:

  start()              hydrate state, open OMS event stream, fetch initial
                       bars, spawn the 15-minute scheduler.
  _15m_scheduler       on each 15m boundary fetch the latest bars per
                       symbol, build ETFBarInput, run process_bar_input,
                       and dispatch the resulting actions to OMS.
  _dispatch_actions    convert SubmitEntry/SubmitProtectiveStop/etc into
                       OMSOrder + Intent submissions.
  _event_loop          consume OMS fills and order updates and route them
                       into process_fill / process_order_update.
  stop()               cancel tasks and snapshot state to disk.

Without this shell, TPC's process_bar_input would never be called from a
production path and `bars_processed` would stay 0 forever — the symptom
the audit identified as "TPC enabled but inert".
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext
from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitAddOnEntry,
    SubmitEntry,
    SubmitExit,
    SubmitMarketExit,
    SubmitPartialExit,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)
from strategies.swing._shared.etf_core import (
    BarData,
    BarWindow,
    ETFBarInput,
    ETFFill,
    ETFOrderUpdate,
)
from strategies.swing._shared.etf_live_engine import ETFCoreLiveEngine
from strategies.swing.tpc import STRATEGY_ID
from strategies.swing.tpc import instrumentation_adapter as _adapter
from strategies.swing.tpc.config import TPCSymbolConfig
from strategies.swing.tpc.core import logic, serializers

logger = logging.getLogger(__name__)


class TPCEngine(ETFCoreLiveEngine):
    def __init__(
        self,
        ib_session: Any,
        oms_service: Any,
        instruments: dict[str, Any],
        config: dict[str, TPCSymbolConfig],
        trade_recorder: Any | None = None,
        equity: float = 100_000.0,
        market_calendar: Any | None = None,
        kit: Any | None = None,
        equity_offset: float = 0.0,
        equity_alloc_pct: float = 1.0,
        coordinator: Any | None = None,
        state_dir: Path | str | None = None,
        bar_input_provider: Callable[[str, str], Any | Awaitable[Any]] | None = None,
        disable_scheduler: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            strategy_id=STRATEGY_ID,
            ib_session=ib_session,
            oms_service=oms_service,
            instruments=instruments,
            config=config,
            core_logic=logic,
            serializers=serializers,
            trade_recorder=trade_recorder,
            equity=equity,
            market_calendar=market_calendar,
            kit=kit,
            equity_offset=equity_offset,
            equity_alloc_pct=equity_alloc_pct,
            coordinator=coordinator,
        )
        self._setup_cache: dict[str, Any] = {}
        self._position_stop_history: dict[str, float] = {}
        # STRAT-1: live execution shell state.
        self._cycle_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._event_queue: Any = None
        self._state_dir = Path(state_dir) if state_dir else Path("data/tpc_state")
        self._bar_input_provider = bar_input_provider
        self._disable_scheduler = bool(disable_scheduler)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        # Map oms_order_id -> setup_id so OMS event callbacks know which
        # setup to update. Populated when SubmitEntry/Stop/TP receipts arrive.
        self._oms_order_to_setup: dict[str, str] = {}
        self._oms_order_role: dict[str, str] = {}

    def process_bar_input(self, bar_input: ETFBarInput) -> tuple[list[Any], list[Any]]:
        cfg = self._config.get(bar_input.symbol)
        pre_position = deepcopy(self._state.positions.get(bar_input.symbol))
        actions, events = super().process_bar_input(bar_input)
        if self._kit is None or cfg is None:
            return actions, events

        decision = "NO_SIGNAL"
        emitted_setup = None
        rejections: list[dict[str, Any]] = []
        for event in events:
            code = event.code
            if code == "ENTRY_REQUESTED":
                setup_id = event.details.get("setup_id")
                setup = self._state.setups.get(setup_id) if setup_id else None
                if setup is not None:
                    self._setup_cache[setup.setup_id] = setup
                    emitted_setup = setup
                decision = "ENTRY_REQUESTED"
            elif code == "SETUP_REJECTED":
                rejection = dict(event.details)
                rejections.append(rejection)
                bar_ts = bar_input.bar_15m.timestamp if bar_input.bar_15m else None
                _adapter.route_missed(self._kit, rejection, cfg, bar_ts=bar_ts)
            elif code in {"MANAGING_POSITION", "ENTRY_PENDING", "NO_SIGNAL"}:
                decision = code

        post_position = self._state.positions.get(bar_input.symbol)
        position = post_position or pre_position
        for action in actions:
            if isinstance(action, ReplaceProtectiveStop):
                setup_id = self._lookup_setup_id_for_symbol(action.symbol, position)
                if not setup_id:
                    continue
                if setup_id not in self._position_stop_history and pre_position is not None:
                    self._position_stop_history[setup_id] = float(pre_position.current_stop)
                old_stop = self._position_stop_history.get(setup_id)
                new_stop = float(action.stop_price)
                if old_stop is not None and abs(old_stop - new_stop) > 1e-9:
                    _adapter.route_stop_adjustment(
                        self._kit,
                        setup_id=setup_id,
                        symbol=action.symbol,
                        old_stop=float(old_stop),
                        new_stop=new_stop,
                        action_reason=str(getattr(action, "reason", "") or ""),
                        position=position,
                    )
                self._position_stop_history[setup_id] = new_stop

        if post_position is not None:
            self._position_stop_history.setdefault(post_position.setup_id, float(post_position.current_stop))

        _adapter.route_filter_decisions(
            self._kit, bar_input, cfg, rejections=rejections, entry_setup=emitted_setup,
        )
        _adapter.route_indicator_snapshot(self._kit, bar_input, self._state, cfg, decision, emitted_setup)
        return actions, events

    def process_fill(self, fill: ETFFill) -> tuple[list[Any], list[Any]]:
        role = (fill.order_role or "").lower()
        pre_position = None
        if role != "entry":
            pre_position = deepcopy(self._state.positions.get(fill.symbol)) if fill.symbol in self._state.positions else None
        actions, events = super().process_fill(fill)
        if self._kit is None:
            return actions, events
        cfg = self._config.get(fill.symbol)

        for event in events:
            if event.code == "ENTRY_FILLED":
                setup_id = event.details.get("setup_id")
                setup = self._setup_cache.pop(setup_id, None) if setup_id else None
                if setup is None:
                    continue
                if cfg is None:
                    continue
                _adapter.route_entry(self._kit, setup, fill, cfg, self._state)
                position = self._state.positions.get(fill.symbol)
                if position is not None:
                    self._position_stop_history[setup.setup_id] = float(position.current_stop)
            elif event.code in {"EXIT_FILLED", "STOP_FILLED"}:
                if pre_position is None:
                    continue
                _adapter.route_exit(
                    self._kit,
                    pre_position=pre_position,
                    fill=fill,
                    event_code=event.code,
                    event_reason=event.details.get("reason"),
                )
                self._position_stop_history.pop(pre_position.setup_id, None)
            # PARTIAL_EXIT_FILLED: cache update only — no log_exit. Pre/post state delta
            # is captured by subsequent stop-adjustment routing on the partial_resize.
        return actions, events

    def process_order_update(self, update: ETFOrderUpdate) -> tuple[list[Any], list[Any]]:
        actions, events = super().process_order_update(update)
        if self._kit is None:
            return actions, events
        for event in events:
            if event.code in {"ORDER_TERMINAL", "ADDON_ORDER_TERMINAL"}:
                _adapter.route_order_event(self._kit, update, event)
        return actions, events

    def _lookup_setup_id_for_symbol(self, symbol: str, position: Any | None) -> str:
        if position is not None and getattr(position, "setup_id", ""):
            return position.setup_id
        # fallback: scan state.positions in case override doesn't have it
        for sym, pos in self._state.positions.items():
            if sym == symbol:
                return pos.setup_id
        return ""

    # ------------------------------------------------------------------
    # STRAT-1 / Phase D: live execution shell
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the live shell: state hydration, OMS events, 15m scheduler."""
        await super().start()  # marks _running, logs base message
        self._restore_state()
        if self._oms is not None and hasattr(self._oms, "stream_events"):
            self._event_queue = self._oms.stream_events(STRATEGY_ID)
            self._event_task = asyncio.create_task(self._oms_event_loop())
        # Best-effort initial bar fetch + cycle
        try:
            await self._cycle_once(request_kind="startup")
        except Exception:
            logger.exception("TPC initial 15m cycle failed")
        if not self._disable_scheduler:
            self._cycle_task = asyncio.create_task(self._15m_scheduler())
        logger.info("TPC live shell active (symbols=%s)", list(self._config.keys()))

    async def stop(self) -> None:
        await super().stop()
        for task in (self._cycle_task, self._event_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._cycle_task = None
        self._event_task = None
        self._persist_state()

    # 15-minute boundary scheduler -------------------------------------

    async def _15m_scheduler(self) -> None:
        while self._running:
            now = self._clock()
            minute = now.minute
            next_15 = ((minute // 15) + 1) * 15
            if next_15 >= 60:
                next_bar = (now + timedelta(hours=1)).replace(
                    minute=next_15 - 60, second=10, microsecond=0,
                )
            else:
                next_bar = now.replace(minute=next_15, second=10, microsecond=0)
            wait = max(0.0, (next_bar - now).total_seconds())
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                await self._cycle_once(request_kind="recurring")
            except Exception:
                logger.exception("TPC 15m cycle error")

    async def _cycle_once(self, *, request_kind: str) -> None:
        """One pass over all configured symbols."""
        for symbol in self._config.keys():
            try:
                bar_input = await self._build_bar_input(symbol, request_kind)
                if bar_input is None or bar_input.bar_15m is None:
                    continue
                actions, _events = self.process_bar_input(bar_input)
                for action in actions:
                    try:
                        await self._dispatch_action(action, symbol)
                    except Exception:
                        logger.exception(
                            "TPC dispatch failed for %s action=%s",
                            symbol, type(action).__name__,
                        )
                self._persist_state()
            except Exception:
                logger.exception("TPC symbol cycle failed for %s", symbol)

    # Bar fetching -----------------------------------------------------

    def _get_contract(self, symbol: str) -> Any:
        """Build an IBKR Stock contract for an ETF symbol."""
        try:
            from ib_async import Stock
            return Stock(symbol=symbol, exchange="SMART", currency="USD")
        except Exception:
            logger.warning("TPC: cannot build Stock contract for %s", symbol)
            return None

    async def _req_bars(
        self,
        contract: Any,
        duration: str,
        bar_size: str,
        *,
        request_kind: str,
        use_rth: bool = True,
    ) -> list | None:
        try:
            return await self._ib.req_historical_data(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
                request_kind=request_kind,
                completed_only=True,
            )
        except Exception:
            logger.debug(
                "TPC bar fetch failed (%s %s)", duration, bar_size, exc_info=True,
            )
            return None

    @staticmethod
    def _bars_to_window(bars: list | None) -> BarWindow | None:
        if not bars:
            return None
        opens = np.array([float(b.open) for b in bars], dtype=float)
        highs = np.array([float(b.high) for b in bars], dtype=float)
        lows = np.array([float(b.low) for b in bars], dtype=float)
        closes = np.array([float(b.close) for b in bars], dtype=float)
        volumes = np.array(
            [float(getattr(b, "volume", 0.0) or 0.0) for b in bars], dtype=float,
        )

        def _ts(b: Any) -> datetime:
            t = getattr(b, "date", None) or getattr(b, "time", None) or getattr(b, "timestamp", None)
            if isinstance(t, str):
                t = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if t is None:
                return datetime.now(timezone.utc)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            return t

        times = tuple(_ts(b) for b in bars)
        return BarWindow(
            opens=opens, highs=highs, lows=lows,
            closes=closes, volumes=volumes, times=times,
        )

    async def _build_bar_input(
        self, symbol: str, request_kind: str,
    ) -> ETFBarInput | None:
        if self._bar_input_provider is not None:
            value = self._bar_input_provider(symbol, request_kind)
            if asyncio.iscoroutine(value):
                value = await value
            return value
        if self._ib is None or not getattr(self._ib, "ib", None):
            return None
        if not self._ib.ib.isConnected():
            return None
        contract = self._get_contract(symbol)
        if contract is None:
            return None
        try:
            qualified = await self._ib.ib.qualifyContractsAsync(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            logger.debug("TPC contract qualification failed for %s", symbol, exc_info=True)

        # Fetch the timeframes the TPC core logic consumes. Daily/4h/1h
        # windows are hydrated on best-effort; missing windows are tolerated
        # by the core (None checks throughout logic.py).
        bars_15 = await self._req_bars(contract, "10 D", "15 mins", request_kind=request_kind)
        bars_30 = await self._req_bars(contract, "10 D", "30 mins", request_kind=request_kind)
        bars_1h = await self._req_bars(contract, "30 D", "1 hour", request_kind=request_kind)
        bars_4h = await self._req_bars(contract, "60 D", "4 hours", request_kind=request_kind)
        bars_d = await self._req_bars(
            contract, "200 D", "1 day", request_kind=request_kind, use_rth=True,
        )

        win15 = self._bars_to_window(bars_15)
        win30 = self._bars_to_window(bars_30)
        win1h = self._bars_to_window(bars_1h)
        win4h = self._bars_to_window(bars_4h)
        wind = self._bars_to_window(bars_d)

        last_15: BarData | None = win15.last if win15 else None
        if last_15 is None:
            return None

        equity_for_input = max(
            0.0, (self._equity + self._equity_offset) * self._equity_alloc_pct,
        )
        return ETFBarInput(
            symbol=symbol,
            bar_15m=last_15,
            bars_15m=win15,
            bars_30m=win30,
            bars_1h=win1h,
            bars_4h=win4h,
            bars_daily=wind,
            indicators={},
            equity=equity_for_input,
            timestamp=last_15.timestamp,
        )

    # Action dispatch --------------------------------------------------

    async def _dispatch_action(self, action: Any, symbol: str) -> None:
        """Convert an ETFCore action into an OMS Intent.

        Action fields (see strategies/core/actions.py):
          - SubmitEntry/SubmitExit variants and protective orders carry
            `side: ActionSide` ("BUY"/"SELL"), converted via OrderSide(...).
          - Entry variants carry `risk_context: dict` with stop_for_risk and
            planned_entry_price; ENTRY OMSOrders MUST set order.risk_context
            or RiskGateway.check_entry will deny them on the heat-cap path.
          - ReplaceProtectiveStop/CancelAction carry `target_order_id`
            (the previously-issued OMS order id), not `oms_order_id`.
          - Intent.flatten uses `instrument_symbol`, not `flatten_symbol`.
        """
        if self._oms is None:
            return
        inst = self._instruments.get(symbol)
        if inst is None:
            logger.warning("TPC dispatch: no instrument for %s", symbol)
            return

        if isinstance(action, (SubmitEntry, SubmitAddOnEntry)):
            order = OMSOrder(
                client_order_id=action.client_order_id,
                strategy_id=STRATEGY_ID, instrument=inst,
                side=OrderSide(action.side),
                qty=int(action.qty),
                order_type=OrderType(action.order_type),
                limit_price=action.limit_price,
                stop_price=action.stop_price,
                tif=action.tif, role=OrderRole.ENTRY,
                oca_group=action.oca_group,
            )
            order.risk_context = self._build_risk_context(action, inst)
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID, order=order,
            ))
            if receipt and receipt.oms_order_id:
                setup_id = (
                    action.metadata.get("setup_id", "")
                    if isinstance(action.metadata, dict) else ""
                ) or getattr(action, "parent_order_id", "") or ""
                if setup_id:
                    self._oms_order_to_setup[receipt.oms_order_id] = setup_id
                self._oms_order_role[receipt.oms_order_id] = (
                    "add_on_entry" if isinstance(action, SubmitAddOnEntry) else "entry"
                )

        elif isinstance(action, (SubmitExit, SubmitPartialExit, SubmitMarketExit)):
            order_type = OrderType.MARKET
            limit_price = None
            stop_price = None
            if hasattr(action, "order_type"):
                order_type = OrderType(action.order_type)
                limit_price = getattr(action, "limit_price", None)
                stop_price = getattr(action, "stop_price", None)
            order = OMSOrder(
                client_order_id=action.client_order_id,
                strategy_id=STRATEGY_ID, instrument=inst,
                side=OrderSide(action.side),
                qty=int(action.qty),
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                tif=action.tif, role=OrderRole.EXIT,
                oca_group=action.oca_group,
            )
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID, order=order,
            ))
            if receipt and receipt.oms_order_id:
                if isinstance(action, SubmitPartialExit):
                    role = "partial_exit"
                elif isinstance(action, SubmitMarketExit):
                    role = "market_exit"
                else:
                    role = "exit"
                self._oms_order_role[receipt.oms_order_id] = role

        elif isinstance(action, SubmitProtectiveStop):
            order = OMSOrder(
                client_order_id=action.client_order_id,
                strategy_id=STRATEGY_ID, instrument=inst,
                side=OrderSide(action.side),
                qty=int(action.qty),
                order_type=OrderType.STOP,
                stop_price=float(action.stop_price),
                tif=action.tif, role=OrderRole.STOP,
                oca_group=action.oca_group,
            )
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID, order=order,
            ))
            if receipt and receipt.oms_order_id:
                self._oms_order_role[receipt.oms_order_id] = "stop"

        elif isinstance(action, SubmitProfitTarget):
            order = OMSOrder(
                client_order_id=action.client_order_id,
                strategy_id=STRATEGY_ID, instrument=inst,
                side=OrderSide(action.side),
                qty=int(action.qty),
                order_type=OrderType.LIMIT,
                limit_price=float(action.limit_price),
                tif=action.tif, role=OrderRole.TP,
                oca_group=action.oca_group,
            )
            receipt = await self._oms.submit_intent(Intent(
                intent_type=IntentType.NEW_ORDER,
                strategy_id=STRATEGY_ID, order=order,
            ))
            if receipt and receipt.oms_order_id:
                self._oms_order_role[receipt.oms_order_id] = "tp"

        elif isinstance(action, ReplaceProtectiveStop):
            target = action.target_order_id
            if not target:
                logger.debug("TPC ReplaceProtectiveStop: no target_order_id for %s", symbol)
                return
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.REPLACE_ORDER,
                strategy_id=STRATEGY_ID,
                target_oms_order_id=target,
                new_qty=int(action.qty) if getattr(action, "qty", 0) else None,
                new_stop_price=float(action.stop_price),
            ))

        elif isinstance(action, CancelAction):
            target = action.target_order_id
            if target:
                await self._oms.submit_intent(Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=STRATEGY_ID,
                    target_oms_order_id=target,
                ))

        elif isinstance(action, FlattenPosition):
            await self._oms.submit_intent(Intent(
                intent_type=IntentType.FLATTEN,
                strategy_id=STRATEGY_ID,
                instrument_symbol=action.symbol or symbol,
            ))

        else:
            logger.debug("TPC: unhandled action type %s", type(action).__name__)

    def _build_risk_context(self, action: SubmitEntry, inst: Any) -> RiskContext:
        """Compose the per-entry RiskContext expected by RiskGateway.

        Pulls stop_for_risk and planned_entry_price from action.risk_context
        (a plain dict on the action), falling back to action.limit_price /
        action.stop_price if the core didn't populate the dict. risk_dollars
        is the same point_value-aware notional that RiskGateway later
        normalises into R via the strategy's unit_risk_dollars.
        """
        rc = action.risk_context if isinstance(action.risk_context, dict) else {}
        stop_for_risk = float(
            rc.get("stop_for_risk", action.stop_price or action.price or 0.0) or 0.0
        )
        planned_entry = float(
            rc.get("planned_entry_price", action.limit_price or action.price or 0.0) or 0.0
        )
        point_value = float(getattr(inst, "point_value", 1.0) or 1.0)
        risk_dollars = abs(planned_entry - stop_for_risk) * int(action.qty) * point_value
        budget_tag = (
            action.metadata.get("risk_budget_tag")
            if isinstance(action.metadata, dict) else None
        ) or STRATEGY_ID
        metadata = action.metadata if isinstance(action.metadata, dict) else {}
        signal_id = str(
            action.risk_context.get("signal_id")
            or metadata.get("signal_id")
            or metadata.get("setup_id")
            or action.client_order_id
            or ""
        )
        bar_id = str(action.risk_context.get("bar_id") or metadata.get("bar_id") or "")
        exchange_timestamp = action.risk_context.get("exchange_timestamp") or metadata.get("exchange_timestamp")
        if isinstance(exchange_timestamp, str) and exchange_timestamp:
            try:
                exchange_timestamp = datetime.fromisoformat(exchange_timestamp)
            except ValueError:
                exchange_timestamp = None
        return RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=planned_entry,
            risk_budget_tag=str(budget_tag),
            risk_dollars=risk_dollars,
            signal_id=signal_id,
            bar_id=bar_id,
            exchange_timestamp=exchange_timestamp,
        )

    # OMS event loop ---------------------------------------------------

    async def _oms_event_loop(self) -> None:
        if self._event_queue is None:
            return
        from libs.oms.models.events import OMSEventType
        while self._running:
            try:
                event = await self._event_queue.get()
            except asyncio.CancelledError:
                return
            try:
                etype = getattr(event, "event_type", None)
                payload = getattr(event, "payload", {}) or {}
                oms_order_id = getattr(event, "oms_order_id", "") or payload.get("oms_order_id", "")
                symbol = payload.get("symbol", "")
                role = self._oms_order_role.get(oms_order_id, "")

                if etype == OMSEventType.FILL:
                    fill = ETFFill(
                        oms_order_id=oms_order_id,
                        fill_price=float(payload.get("price", 0.0)),
                        fill_qty=int(payload.get("qty", 0)),
                        symbol=symbol,
                        fill_time=self._clock(),
                        commission=float(payload.get("commission", 0.0)),
                        order_role=role or payload.get("role", "").lower() or "entry",
                        fill_id=str(payload.get("fill_id") or payload.get("exec_id") or ""),
                        intent_id=str(payload.get("intent_id") or ""),
                        risk_decision_ref=str(payload.get("risk_decision_ref") or ""),
                        portfolio_decision_ref=str(payload.get("portfolio_decision_ref") or ""),
                        runtime_payload={**payload, "oms_order_id": oms_order_id},
                    )
                    self.process_fill(fill)
                    self._persist_state()
                elif etype in (
                    OMSEventType.ORDER_ACKED,
                    OMSEventType.ORDER_CANCELLED,
                    OMSEventType.ORDER_REJECTED,
                    OMSEventType.ORDER_EXPIRED,
                    OMSEventType.ORDER_FILLED,
                ):
                    update = ETFOrderUpdate(
                        oms_order_id=oms_order_id,
                        status=str(etype.value if hasattr(etype, "value") else etype),
                        symbol=symbol,
                        timestamp=self._clock(),
                        order_role=role,
                    )
                    self.process_order_update(update)
                    self._persist_state()
            except Exception:
                logger.exception("TPC OMS event handling failed")

    # State persistence ------------------------------------------------

    def _restore_state(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self._state_dir / f"{STRATEGY_ID}.json"
            if not snap_path.exists():
                return
            with snap_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not payload:
                return
            # ETFCoreLiveEngine.hydrate is async-safe (it just calls
            # serializers.restore_state). Use the sync path to avoid
            # awaiting inside startup before super().start() completes.
            self._state = self._serializers.restore_state(payload)
            logger.info("TPC state restored from %s", snap_path)
        except Exception:
            logger.warning("TPC state restore failed", exc_info=True)

    def _persist_state(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self._state_dir / f"{STRATEGY_ID}.json"
            tmp_path = snap_path.with_suffix(".json.tmp")
            payload = self._serializers.snapshot_state(self._state)
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            tmp_path.replace(snap_path)
        except Exception:
            logger.warning("TPC state persist failed", exc_info=True)
