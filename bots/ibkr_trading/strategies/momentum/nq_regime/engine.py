from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from libs.market_data.futures_roll import roll_blackout_reason, roll_force_flatten_reason
from libs.market_data.live_futures import req_panama_adjusted_historical_data
from libs.oms.models.events import OMSEventType
from libs.oms.models.intent import Intent, IntentType
from libs.oms.models.order import OMSOrder, OrderRole, OrderSide, OrderType, RiskContext
from libs.oms.instrumentation.runtime_refs import fill_runtime_refs
from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitProfitTarget,
    SubmitProtectiveStop,
)

from . import config
from .config import TradeSide
from .core.data_policy import CompletedBarPolicy
from .core.levels import KeyLevels
from .core.logic import on_bar as core_on_bar
from .core.logic import on_fill as core_on_fill
from .core.logic import on_order_update as core_on_order_update
from .core.serializers import hydrate_state, snapshot_state
from .core.state import BarData, FillEvent, OrderUpdateEvent, RegimeCoreState
from .modules.base import NewsEvent, SetupCandidate
from strategies.scalp._shared.time_utils import session_date

logger = logging.getLogger(__name__)


def _stop_adj_type(reason: str) -> str:
    if "be" in reason.lower() or "breakeven" in reason.lower():
        return "breakeven"
    if "profit" in reason.lower() or "floor" in reason.lower():
        return "profit_protection"
    return "trailing"


class NQRegimeEngine:
    """Live shell for the NQ regime core.

    Market-data adapters can feed completed 5m bars through ``on_bar`` and pass
    live-only refinements such as imbalance, book pressure, spread quality, or
    news state in ``live_context``. The shared core remains the source of truth.
    """

    strategy_id = config.STRATEGY_ID

    def __init__(
        self,
        *,
        ib_session: Any = None,
        oms_service: Any = None,
        instruments: dict[str, Any] | None = None,
        trade_recorder: Any = None,
        equity: float = 100_000.0,
        instrumentation: Any = None,
        analysis_symbol: str = config.ANALYSIS_SYMBOL,
        trade_symbol: str = config.TRADE_SYMBOL,
        state_dir: Path | str | None = None,
        equity_alloc_pct: float = 1.0,
        clock: Callable[[], datetime] | None = None,
        disable_scheduler: bool = False,
        **_: Any,
    ) -> None:
        self._ib = ib_session
        self._oms = oms_service
        self._instruments = dict(instruments or {})
        self._trade_recorder = trade_recorder
        self._equity = float(equity)
        self._equity_alloc_pct = float(equity_alloc_pct)
        self._instrumentation = instrumentation
        self._settings = config.StrategyRuntimeSettings(
            analysis_symbol=analysis_symbol.upper(),
            trade_symbol=trade_symbol.upper(),
            initial_equity=self._equity,
        )
        self._bar_policy = CompletedBarPolicy()
        self._state = RegimeCoreState()
        self._scheduled_news: list[NewsEvent] = []
        self._event_queue: Any = None
        self._event_task: asyncio.Task | None = None
        # STRAT-2: live wiring needs a 5m bar scheduler task and a place to
        # persist core state across restarts. Mirrors NQDTC's _cycle_task +
        # _restore_state pattern. state_dir defaults align with the
        # NQDTC_STATE_DIR convention.
        self._cycle_task: asyncio.Task | None = None
        self._state_dir = Path(state_dir) if state_dir else Path("data/nq_regime_state")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._disable_scheduler = bool(disable_scheduler)
        self._running = False

        # Instrumentation kit
        from strategies.momentum.instrumentation.src.facade import InstrumentationKit

        self._kit = InstrumentationKit(self._instrumentation, strategy_type="nq_regime")

        # Liveness tracking
        self._bars_processed: int = 0
        self._symbol_last_bar_ts: dict[str, datetime] = {}
        self._last_decision_code: str = self._state.last_decision_code
        self._last_decision_details: dict[str, Any] = dict(self._state.last_decision_details)
        self._last_bar_ts: datetime | None = self._state.last_bar_ts

        # Instrumentation tracking state
        self._entry_candidate: SetupCandidate | None = None
        self._mfe_r: float = 0.0
        self._mae_r: float = 0.0
        self._prev_stop_price: float = 0.0
        self._prev_regime: Any = None
        self._daily_levels: KeyLevels | None = None
        self._daily_levels_session_date: str = ""
        self._roll_flatten_pending: bool = False
        self._roll_flatten_oms_id: str = ""

    async def start(self) -> None:
        """STRAT-2 / Phase D live wiring.

        Mirrors NQDTC's start() pattern:
          1. Restore persisted core state (signal evolution, last_decision_*).
          2. Open OMS event stream for fill/order updates.
          3. Fetch initial NQ analysis bars so the first 5m boundary has
             enough history for the regime classifier.
          4. Spawn the 5m boundary scheduler that calls on_bar(...), the
             code path that turns raw bars into actions and dispatches them
             through the existing _dispatch_action.

        Without this, NQ_REGIME would appear running while processing zero
        bars: heartbeats green, no trades, no missed-opportunity evidence.
        """
        logger.info("NQ_REGIME engine starting")
        self._running = True
        self._restore_state()

        if self._oms is not None and hasattr(self._oms, "stream_events"):
            self._event_queue = self._oms.stream_events(config.STRATEGY_ID)
            self._event_task = asyncio.create_task(self._event_loop())

        if not self._disable_scheduler:
            try:
                await self._fetch_and_emit_bar(request_kind="startup")
            except Exception:
                logger.exception("NQ_REGIME initial bar fetch failed")
            self._cycle_task = asyncio.create_task(self._5m_scheduler())
        logger.info(
            "NQ_REGIME engine started (analysis=%s, trade=%s)",
            self._settings.analysis_symbol, self._settings.trade_symbol,
        )

    async def stop(self) -> None:
        logger.info("NQ_REGIME engine stopping")
        self._running = False
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

    async def hydrate(self, snapshot: dict[str, Any]) -> None:
        if snapshot:
            self._state = hydrate_state(snapshot.get("core", snapshot))
            self._sync_health_fields()
            # Sync instrumentation tracking from restored state
            self._prev_stop_price = self._state.stop_price
            self._prev_regime = self._state.regime

    def snapshot_state(self) -> dict[str, Any]:
        return {"strategy_id": self.strategy_id, "core": snapshot_state(self._state)}

    def health_status(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "running": self._running,
            "analysis_symbol": self._settings.analysis_symbol,
            "trade_symbol": self._settings.trade_symbol,
            "phase": self._state.phase.value,
            "regime": self._state.regime.name if self._state.regime is not None else None,
            "active_module": self._state.active_module.value,
            "last_decision_code": self._state.last_decision_code,
            "last_decision_details": self._state.last_decision_details,
            "last_bar_ts": self._state.last_bar_ts,
            "last_seen_bar_ts": self._state.last_bar_ts,
        }

    async def on_bar(
        self,
        bar: Any,
        *,
        daily_context: KeyLevels | None = None,
        live_context: dict[str, Any] | None = None,
        scheduled_news: list[NewsEvent] | None = None,
    ) -> None:
        bar_data = _bar_data_from_any(bar)
        event = self._bar_policy.build_event(
            bar_5m=bar_data,
            recent_5m=[*self._state.bars_5m, bar_data],
            daily_context=daily_context,
            live_context=live_context,
        )
        news = scheduled_news if scheduled_news is not None else self._scheduled_news
        self._state, actions, events = core_on_bar(
            self._state,
            event,
            scheduled_news=news,
            settings=self._settings,
        )
        self._record_events(events)
        self._bars_processed += 1
        if self._state.last_bar_ts is not None:
            self._symbol_last_bar_ts[self._settings.analysis_symbol] = self._state.last_bar_ts

        # Regime transition tracking for coordination events
        new_regime = self._state.regime
        if self._prev_regime is not None and new_regime != self._prev_regime:
            self._emit_regime_coordination(new_regime)
        self._prev_regime = new_regime

        if await self._force_flatten_for_roll(bar_data.ts):
            self._sync_health_fields()
            self._persist_state()
            return

        for action in actions:
            await self._dispatch_action(action)
        self._sync_health_fields()
        self._persist_state()

    def set_scheduled_news(self, events: list[NewsEvent]) -> None:
        self._scheduled_news = list(events)

    # ------------------------------------------------------------------
    # STRAT-2 / Phase D: 5m bar scheduler + state persistence
    # ------------------------------------------------------------------

    async def _5m_scheduler(self) -> None:
        """Sleep until the next 5m boundary + 10s offset, then fetch & emit.

        Pattern mirrors NQDTC's _5m_scheduler. The 10s offset gives IBKR
        time to mark the just-closed bar as 'completed'; without it the
        request can return the still-open bar.
        """
        while self._running:
            now = self._clock()
            minute = now.minute
            next_5 = ((minute // 5) + 1) * 5
            if next_5 >= 60:
                next_bar = (now + timedelta(hours=1)).replace(
                    minute=next_5 - 60, second=10, microsecond=0,
                )
            else:
                next_bar = now.replace(minute=next_5, second=10, microsecond=0)
            wait = max(0.0, (next_bar - now).total_seconds())
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                await self._fetch_and_emit_bar(request_kind="recurring")
            except Exception:
                logger.exception("NQ_REGIME 5m cycle error")

    def _get_analysis_contract(self) -> Any:
        """Build the continuous future contract used for analysis bars.

        analysis_symbol is `NQ` (full-size), `trade_symbol` is `MNQ`. We
        fetch bars on the analysis symbol because it has tighter spreads
        and is the canonical price reference for regime classification;
        order routing in _dispatch_action uses the trade_symbol's
        instrument from self._instruments.
        """
        try:
            from ib_async import ContFuture
            return ContFuture(
                symbol=self._settings.analysis_symbol,
                exchange="CME",
                currency="USD",
            )
        except Exception:
            logger.warning(
                "NQ_REGIME: failed to build ContFuture for %s",
                self._settings.analysis_symbol,
            )
            return None

    async def _fetch_and_emit_bar(self, request_kind: str = "recurring") -> None:
        """Fetch the most recent completed 5m bar for NQ and run on_bar(...)."""
        if self._ib is None:
            return
        ib_obj = getattr(self._ib, "ib", None)
        if ib_obj is None or not ib_obj.isConnected():
            logger.debug("NQ_REGIME: IB not connected, skipping fetch")
            return
        contract = self._get_analysis_contract()
        if contract is None:
            return
        try:
            qualified = await ib_obj.qualifyContractsAsync(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            logger.debug("NQ_REGIME: contract qualification failed", exc_info=True)

        try:
            bars = await req_panama_adjusted_historical_data(
                self._ib,
                contract,
                symbol=self._settings.analysis_symbol,
                endDateTime="",
                durationStr="2 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                request_kind=request_kind,
                completed_only=True,
            )
        except Exception:
            logger.exception("NQ_REGIME 5m fetch failed")
            return
        if not bars:
            return
        # Pass the most recent completed bar; CompletedBarPolicy.build_event
        # internally decides whether to emit using the recent bar history.
        latest = _bar_data_from_any(bars[-1])
        target_session = str(session_date(latest.ts))
        if self._daily_levels is None or target_session != self._daily_levels_session_date:
            await self._refresh_daily_context(force=True, for_ts=latest.ts)
        if self._daily_levels is None:
            logger.error(
                "NQ_REGIME skipping scheduled bar without daily key levels (session=%s)",
                target_session,
            )
            self._record_decision(
                "MISSING_DAILY_KEY_LEVELS",
                {"session": target_session, "request_kind": request_kind},
                latest.ts,
            )
            return
        await self.on_bar(latest, daily_context=self._daily_levels)

    async def _refresh_daily_context(
        self,
        *,
        force: bool = False,
        for_ts: datetime | None = None,
    ) -> None:
        """Refresh previous-day and weekly levels from completed daily bars."""
        if self._ib is None:
            return
        ib_obj = getattr(self._ib, "ib", None)
        if ib_obj is None or not ib_obj.isConnected():
            return
        target_session = str(session_date(for_ts or datetime.now(timezone.utc)))
        if not force and self._daily_levels_session_date == target_session:
            return
        contract = self._get_analysis_contract()
        if contract is None:
            return
        try:
            qualified = await ib_obj.qualifyContractsAsync(contract)
            if qualified:
                contract = qualified[0]
        except Exception:
            logger.debug("NQ_REGIME: daily contract qualification failed", exc_info=True)
        try:
            bars = await req_panama_adjusted_historical_data(
                self._ib,
                contract,
                symbol=self._settings.analysis_symbol,
                endDateTime="",
                durationStr="10 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                request_kind="daily_levels",
                completed_only=True,
            )
        except Exception:
            logger.exception("NQ_REGIME daily level fetch failed")
            return
        daily = [_bar_data_from_any(bar) for bar in (bars or [])]
        if not daily:
            return
        prev = daily[-1]
        lookback = daily[-5:]
        self._daily_levels = KeyLevels(
            pdh=prev.high,
            pdl=prev.low,
            pdm=(prev.high + prev.low) / 2.0,
            weekly_high=max(bar.high for bar in lookback),
            weekly_low=min(bar.low for bar in lookback),
        )
        self._daily_levels_session_date = target_session
        logger.info(
            "NQ_REGIME daily levels refreshed: pdh=%.2f pdl=%.2f weekly_high=%.2f weekly_low=%.2f",
            self._daily_levels.pdh,
            self._daily_levels.pdl,
            self._daily_levels.weekly_high,
            self._daily_levels.weekly_low,
        )

    def _restore_state(self) -> None:
        """Load core state from disk if present. Best-effort, never raises."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self._state_dir / f"{config.STRATEGY_ID}.json"
            if not snap_path.exists():
                return
            with snap_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            core = payload.get("core") if isinstance(payload, dict) else None
            if core is None:
                return
            self._state = hydrate_state(core)
            self._sync_health_fields()
            self._prev_stop_price = self._state.stop_price
            self._prev_regime = self._state.regime
            logger.info(
                "NQ_REGIME state restored from %s (last_bar=%s)",
                snap_path, self._state.last_bar_ts,
            )
        except Exception:
            logger.warning("NQ_REGIME state restore failed", exc_info=True)

    def _persist_state(self) -> None:
        """Write current core state to disk. Best-effort, never raises."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self._state_dir / f"{config.STRATEGY_ID}.json"
            tmp_path = snap_path.with_suffix(".json.tmp")
            payload = self.snapshot_state()
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            tmp_path.replace(snap_path)
        except Exception:
            logger.warning("NQ_REGIME state persist failed", exc_info=True)

    async def _event_loop(self) -> None:
        while True:
            event = await self._event_queue.get()
            try:
                await self._handle_oms_event(event)
            except Exception:
                logger.exception("NQ_REGIME OMS event handling failed")

    async def _handle_oms_event(self, event: Any) -> None:
        etype = getattr(event, "event_type", None)
        if etype == OMSEventType.FILL:
            payload = getattr(event, "payload", {}) or {}

            # Capture pre-fill state for instrumentation
            pre_fill_position_side = self._state.position_side
            pre_fill_qty_open = self._state.qty_open
            pre_fill_trade_id = self._state.active_trade_id
            pre_fill_stop_at_be = self._state.stop_at_be

            fill = FillEvent(
                oms_order_id=str(getattr(event, "oms_order_id", "") or payload.get("oms_order_id", "")),
                fill_price=float(payload.get("price", payload.get("fill_price", 0.0)) or 0.0),
                fill_qty=int(payload.get("qty", payload.get("fill_qty", 0)) or 0),
                fill_time=_event_ts(event),
                symbol=self._settings.trade_symbol,
                commission=float(payload.get("commission", 0.0) or 0.0),
                exit_type=str(payload.get("exit_type", "")),
                fill_id=str(payload.get("fill_id") or payload.get("exec_id") or ""),
                intent_id=str(payload.get("intent_id") or ""),
                risk_decision_ref=str(payload.get("risk_decision_ref") or ""),
                portfolio_decision_ref=str(payload.get("portfolio_decision_ref") or ""),
            )
            self._state, actions, events = core_on_fill(self._state, fill, settings=self._settings)
            if self._state.position_side is TradeSide.FLAT or self._state.qty_open <= 0:
                self._roll_flatten_pending = False
                self._roll_flatten_oms_id = ""
            self._record_events(events)

            # Detect entry fill: was FLAT, now has position
            if pre_fill_position_side is TradeSide.FLAT and self._state.position_side is not TradeSide.FLAT:
                self._log_entry_fill(fill)

            # Detect exit fill: had position, now flat (or partial exit)
            elif pre_fill_qty_open > 0 and self._state.qty_open < pre_fill_qty_open:
                is_full_exit = self._state.qty_open <= 0
                self._log_exit_fill(fill, events, pre_fill_trade_id, is_full_exit, pre_fill_stop_at_be)

            for action in actions:
                await self._dispatch_action(action)
            self._sync_health_fields()
            self._persist_state()
            return

        if etype in (OMSEventType.ORDER_CANCELLED, OMSEventType.ORDER_REJECTED, OMSEventType.ORDER_EXPIRED):
            oms_order_id = str(getattr(event, "oms_order_id", ""))
            is_roll_flatten_terminal = bool(
                self._roll_flatten_oms_id and oms_order_id == self._roll_flatten_oms_id
            )
            update = OrderUpdateEvent(
                oms_order_id=oms_order_id,
                status=etype.value if hasattr(etype, "value") else str(etype),
                timestamp=_event_ts(event),
                symbol=self._settings.trade_symbol,
            )
            self._state, actions, events = core_on_order_update(self._state, update)
            self._record_events(events)
            for action in actions:
                await self._dispatch_action(action)
            if is_roll_flatten_terminal:
                self._roll_flatten_pending = False
                self._roll_flatten_oms_id = ""
                if self._state.position_side is not TradeSide.FLAT and self._state.qty_open > 0:
                    self._record_decision(
                        "ROLL_FORCE_FLATTEN_RETRY",
                        {"oms_order_id": oms_order_id, "status": update.status},
                        update.timestamp,
                    )
                    await self._force_flatten_for_roll(update.timestamp)
            self._sync_health_fields()
            self._persist_state()

    async def _force_flatten_for_roll(self, now: datetime) -> bool:
        if self._oms is None:
            return False
        if self._state.position_side is TradeSide.FLAT or self._state.qty_open <= 0:
            self._roll_flatten_pending = False
            self._roll_flatten_oms_id = ""
            return False
        instrument = self._instrument(self._settings.trade_symbol)
        reason = roll_force_flatten_reason(instrument, as_of=now)
        if not reason:
            return False
        if self._roll_flatten_pending:
            self._record_decision(
                "ROLL_FORCE_FLATTEN_PENDING",
                {"reason": reason, "oms_order_id": self._roll_flatten_oms_id},
            )
            return True
        for order_id in [
            self._state.working_entry_order_id,
            self._state.working_stop_order_id,
            *self._state.working_target_order_ids,
        ]:
            if order_id:
                try:
                    await self._oms.submit_intent(
                        Intent(
                            intent_type=IntentType.CANCEL_ORDER,
                            strategy_id=config.STRATEGY_ID,
                            target_oms_order_id=order_id,
                        )
                    )
                except Exception:
                    logger.warning("NQ_REGIME roll-safety cancel failed for %s", order_id, exc_info=True)
        try:
            receipt = await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.FLATTEN,
                    strategy_id=config.STRATEGY_ID,
                    instrument_symbol=self._settings.trade_symbol,
                )
            )
        except Exception:
            self._roll_flatten_pending = False
            self._roll_flatten_oms_id = ""
            self._record_decision("ROLL_FORCE_FLATTEN_FAILED", {"reason": reason}, now)
            logger.exception("NQ_REGIME roll-safety flatten submission failed")
            return True
        self._roll_flatten_oms_id = str(getattr(receipt, "oms_order_id", "") or "")
        self._roll_flatten_pending = bool(self._roll_flatten_oms_id)
        self._record_decision(
            "ROLL_FORCE_FLATTEN",
            {"reason": reason, "oms_order_id": self._roll_flatten_oms_id},
            now,
        )
        logger.critical("NQ_REGIME forcing flatten for roll safety: %s", reason)
        return True

    async def _dispatch_action(self, action: Any) -> None:
        if self._oms is None:
            return
        if isinstance(action, SubmitEntry):
            now = datetime.now(timezone.utc)
            instrument = self._instrument(action.symbol)
            denial = roll_blackout_reason(instrument, as_of=now)
            if denial:
                self._state, _, events = core_on_order_update(
                    self._state,
                    OrderUpdateEvent(
                        oms_order_id=action.client_order_id,
                        status="rejected",
                        timestamp=now,
                        symbol=action.symbol,
                        order_role="entry",
                        reason=denial,
                    ),
                )
                self._record_events(events)
                self._record_decision(
                    "ENTRY_BLOCKED_BY_ROLL_BLACKOUT",
                    {"reason": denial, "symbol": action.symbol},
                    now,
                )
                self._sync_health_fields()
                self._persist_state()
                logger.warning("NQ_REGIME entry blocked by roll blackout: %s", denial)
                return
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=config.STRATEGY_ID, order=self._order_from_entry(action))
            )
            self._promote_order_id(action.client_order_id, getattr(receipt, "oms_order_id", None))
        elif isinstance(action, SubmitProtectiveStop):
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=config.STRATEGY_ID, order=self._order_from_stop(action))
            )
            self._promote_order_id(action.client_order_id, getattr(receipt, "oms_order_id", None))
        elif isinstance(action, SubmitProfitTarget):
            receipt = await self._oms.submit_intent(
                Intent(intent_type=IntentType.NEW_ORDER, strategy_id=config.STRATEGY_ID, order=self._order_from_target(action))
            )
            self._promote_order_id(action.client_order_id, getattr(receipt, "oms_order_id", None))
        elif isinstance(action, ReplaceProtectiveStop):
            # Instrumentation: log stop adjustment
            old_stop = self._prev_stop_price
            new_stop = action.stop_price
            if self._kit.active and old_stop > 0 and old_stop != new_stop:
                try:
                    self._kit.log_stop_adjustment(
                        trade_id=self._state.active_trade_id or "",
                        symbol=self._settings.trade_symbol,
                        old_stop=old_stop,
                        new_stop=new_stop,
                        adjustment_type=_stop_adj_type(getattr(action, "reason", "") or ""),
                        trigger=getattr(action, "reason", "") or "unknown",
                        metadata={
                            "module": self._state.entry_module.value,
                            "mfe_r": self._mfe_r,
                            "partial_taken": self._state.partial_taken,
                        },
                    )
                except Exception:
                    pass
            self._prev_stop_price = new_stop

            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.REPLACE_ORDER,
                    strategy_id=config.STRATEGY_ID,
                    target_oms_order_id=action.target_order_id,
                    new_qty=action.qty,
                    new_stop_price=action.stop_price,
                )
            )
        elif isinstance(action, CancelAction):
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.CANCEL_ORDER,
                    strategy_id=config.STRATEGY_ID,
                    target_oms_order_id=action.target_order_id,
                )
            )
        elif isinstance(action, FlattenPosition):
            await self._oms.submit_intent(
                Intent(
                    intent_type=IntentType.FLATTEN,
                    strategy_id=config.STRATEGY_ID,
                    instrument_symbol=action.symbol,
                )
            )

    # ----------------------------------------------------------------
    # Instrumentation helpers
    # ----------------------------------------------------------------

    def _record_events(self, events: list[Any]) -> None:
        for event in events:
            code = getattr(event, "code", "")
            details = getattr(event, "details", {})
            try:
                if code == "ROUTING_DECISION":
                    self._on_routing_decision(event)
                elif code == "DAILY_LOCKOUT":
                    self._on_daily_lockout(event)
                elif code == "NEWS_VETO":
                    self._on_news_veto(event)
                elif code == "ENTRY_BLOCKED_BY_SESSION":
                    self._on_entry_blocked(event, "session_phase")
                elif code == "ENTRY_BLOCKED_BY_SIZE":
                    self._on_entry_blocked(event, "sizing_zero")
                elif code == "ENTRY_REQUESTED":
                    cid = details.get("candidate", "")
                    self._entry_candidate = self._state.pending_candidates.get(cid)
                elif code == "MANAGE_OPEN_POSITION":
                    mfe = details.get("mfe_r", 0)
                    self._mfe_r = max(self._mfe_r, mfe)
                    self._update_mae()
            except Exception:
                logger.debug("nq_regime instrumentation event failed", exc_info=True)

    def _record_decision(
        self,
        code: str,
        details: dict[str, Any] | None = None,
        last_bar_ts: datetime | None = None,
    ) -> None:
        self._state.last_decision_code = code
        self._state.last_decision_details = dict(details or {})
        if last_bar_ts is not None:
            self._state.last_bar_ts = last_bar_ts
        self._sync_health_fields()

    def _log_entry_fill(self, fill: FillEvent) -> None:
        if not self._kit.active:
            return
        try:
            state = self._state
            candidate = self._entry_candidate
            pv = self._settings.trade_spec.point_value

            self._kit.log_entry(
                trade_id=state.active_trade_id or "",
                pair=self._settings.trade_symbol,
                side="LONG" if state.position_side is TradeSide.LONG else "SHORT",
                entry_price=state.entry_price,
                position_size=float(state.qty_open),
                position_size_quote=state.qty_open * state.entry_price * pv,
                entry_signal=f"{state.entry_module.value}_{candidate.setup_type}" if candidate else state.entry_module.value,
                entry_signal_id=state.active_trade_id or "",
                entry_signal_strength=state.setup_score / 12.0,
                expected_entry_price=candidate.entry_price if candidate else None,
                strategy_params=self._build_strategy_params(candidate),
                signal_factors=self._build_signal_factors(candidate),
                sizing_inputs={
                    "grade": state.setup_grade.value,
                    "risk_pct": candidate.details.get("risk_pct", 0) if candidate else 0,
                    "stop_distance_pts": state.initial_risk_points,
                    "contracts": state.qty_open,
                    "after_loss": state.daily_losses > 0,
                    "equity": self._equity,
                },
                session_type=state.phase.value,
                concurrent_positions=1,
                exchange_timestamp=fill.fill_time,
                **fill_runtime_refs(
                    getattr(fill, "oms_order_id", ""),
                    {
                        "fill_id": getattr(fill, "fill_id", ""),
                        "qty": getattr(fill, "fill_qty", 0),
                        "intent_id": getattr(fill, "intent_id", ""),
                        "risk_decision_ref": getattr(fill, "risk_decision_ref", ""),
                        "portfolio_decision_ref": getattr(fill, "portfolio_decision_ref", ""),
                    },
                    fill_qty=getattr(fill, "fill_qty", 0),
                ),
            )
            # Reset tracking
            self._mfe_r = 0.0
            self._mae_r = 0.0
            self._prev_stop_price = state.stop_price
        except Exception:
            logger.debug("nq_regime entry instrumentation failed", exc_info=True)

    def _log_exit_fill(self, fill: FillEvent, events: list, pre_trade_id: str | None, is_full: bool, pre_stop_at_be: bool = False) -> None:
        if not self._kit.active:
            return
        try:
            exit_reason = self._determine_exit_reason(events, pre_stop_at_be)
            self._kit.log_exit(
                trade_id=pre_trade_id or "",
                exit_price=fill.fill_price,
                exit_reason=exit_reason,
                fees_paid=fill.commission,
                exchange_timestamp=fill.fill_time,
                mfe_r=self._mfe_r,
                mae_r=self._mae_r,
                **fill_runtime_refs(
                    getattr(fill, "oms_order_id", ""),
                    {
                        "fill_id": getattr(fill, "fill_id", ""),
                        "qty": getattr(fill, "fill_qty", 0),
                        "intent_id": getattr(fill, "intent_id", ""),
                        "risk_decision_ref": getattr(fill, "risk_decision_ref", ""),
                        "portfolio_decision_ref": getattr(fill, "portfolio_decision_ref", ""),
                    },
                    fill_qty=getattr(fill, "fill_qty", 0),
                    is_exit=True,
                ),
            )
            if is_full:
                self._entry_candidate = None
                self._mfe_r = 0.0
                self._mae_r = 0.0
        except Exception:
            logger.debug("nq_regime exit instrumentation failed", exc_info=True)

    def _on_routing_decision(self, event: Any) -> None:
        if not self._kit.active:
            return
        details = getattr(event, "details", {})
        # Log blocked candidates
        blocked_list = details.get("blocked_candidates", [])
        for bc in blocked_list:
            self._kit.log_missed(
                pair=self._settings.trade_symbol,
                side=bc.get("side", ""),
                signal=f"{bc.get('module', '')}_{bc.get('setup_type', '')}",
                signal_id=bc.get("candidate_id", ""),
                signal_strength=bc.get("score", 0) / 12.0,
                blocked_by="regime_routing",
                block_reason=bc.get("block_reason", details.get("reason", "")),
                strategy_params={
                    "regime": details.get("regime", ""),
                    "confidence": details.get("confidence", 0),
                    "margin": details.get("margin", 0),
                    "phase": self._state.phase.value,
                    "ib_type": self._state.ib_type.value,
                },
                session_type=self._state.phase.value,
                exchange_timestamp=getattr(event, "ts", None),
            )
        # Emit indicator snapshot when routing produces any candidates
        if details.get("candidate_count", 0) > 0:
            self._emit_indicator_snapshot(event)

    def _on_daily_lockout(self, event: Any) -> None:
        if not self._kit.active:
            return
        details = getattr(event, "details", {})
        self._kit.log_missed(
            pair=self._settings.trade_symbol,
            side="",
            signal="all_candidates_vetoed",
            signal_id="",
            signal_strength=0.0,
            blocked_by="daily_lockout",
            block_reason=f"daily_realized_r={details.get('daily_realized_r', 0):.2f}",
            strategy_params={
                "daily_realized_r": self._state.daily_realized_r,
                "daily_losses": self._state.daily_losses,
            },
            session_type=self._state.phase.value,
            exchange_timestamp=getattr(event, "ts", None),
        )

    def _on_news_veto(self, event: Any) -> None:
        if not self._kit.active:
            return
        details = getattr(event, "details", {})
        self._kit.log_missed(
            pair=self._settings.trade_symbol,
            side="",
            signal="all_candidates_vetoed",
            signal_id="",
            signal_strength=0.0,
            blocked_by="news_veto",
            block_reason=details.get("news", ""),
            strategy_params={
                "regime": self._state.regime.name if self._state.regime else "",
            },
            session_type=self._state.phase.value,
            exchange_timestamp=getattr(event, "ts", None),
        )

    def _on_entry_blocked(self, event: Any, reason: str) -> None:
        if not self._kit.active:
            return
        details = getattr(event, "details", {})
        candidate = self._entry_candidate
        self._kit.log_missed(
            pair=self._settings.trade_symbol,
            side=candidate.side.value if candidate else "",
            signal=f"{details.get('module', '')}_{candidate.setup_type}" if candidate else reason,
            signal_id=candidate.candidate_id if candidate else details.get("candidate", ""),
            signal_strength=candidate.score / 12.0 if candidate else 0.0,
            blocked_by=reason,
            block_reason=f"phase={details.get('phase', '')}" if reason == "session_phase" else reason,
            strategy_params={
                "regime": self._state.regime.name if self._state.regime else "",
                "phase": self._state.phase.value,
                "module": details.get("module", ""),
                "grade": candidate.grade.value if candidate else "",
            },
            session_type=self._state.phase.value,
            exchange_timestamp=getattr(event, "ts", None),
        )
        self._entry_candidate = None  # Clear stale reference

    def _emit_indicator_snapshot(self, routing_event: Any) -> None:
        if not self._kit.active:
            return
        try:
            ind = self._state.indicators
            if ind is None:
                return
            details = getattr(routing_event, "details", {})
            self._kit.on_indicator_snapshot(
                pair=self._settings.trade_symbol,
                indicators={
                    "vwap": getattr(ind, "vwap", 0.0),
                    "vwap_sd": getattr(ind, "vwap_sd", 0.0),
                    "vwap_slope": getattr(ind, "vwap_slope", 0.0),
                    "atr_15m": getattr(ind, "atr_15m", 0.0),
                    "atr_5m": getattr(ind, "atr_5m", 0.0),
                    "ema9_15m": getattr(ind, "ema9_15m", 0.0),
                    "ema20_15m": getattr(ind, "ema20_15m", 0.0),
                    "ema50_15m": getattr(ind, "ema50_15m", 0.0),
                    "bb_upper": getattr(ind, "bb_upper", 0.0),
                    "bb_lower": getattr(ind, "bb_lower", 0.0),
                    "kc_upper": getattr(ind, "kc_upper", 0.0),
                    "kc_lower": getattr(ind, "kc_lower", 0.0),
                    "squeeze_on": getattr(ind, "squeeze_on", False),
                    "squeeze_duration": getattr(ind, "squeeze_duration", 0),
                    "rsi14_15m": getattr(ind, "rsi14_15m", 0.0),
                    "macd_15m": getattr(ind, "macd_15m", 0.0),
                    "macd_signal_15m": getattr(ind, "macd_signal_15m", 0.0),
                    "volume_multiple_15m": getattr(ind, "volume_multiple_15m", 0.0),
                    "volume_multiple_5m": getattr(ind, "volume_multiple_5m", 0.0),
                },
                signal_name=f"nq_regime_{details.get('selected_module', 'none')}",
                signal_strength=details.get("selected_score", 0) / 12.0,
                decision="enter" if details.get("selected") else "skip",
                strategy_type="nq_regime",
                exchange_timestamp=getattr(routing_event, "ts", None),
                context={
                    "regime": details.get("regime", ""),
                    "phase": self._state.phase.value,
                    "ib_type": self._state.ib_type.value,
                    "confidence": details.get("confidence", 0),
                },
            )
        except Exception:
            logger.debug("nq_regime indicator snapshot failed", exc_info=True)

    def _update_mae(self) -> None:
        """Update MAE from latest bar. Called on MANAGE_OPEN_POSITION events."""
        if self._state.position_side is TradeSide.FLAT or self._state.initial_risk_points <= 0:
            return
        bars = self._state.bars_5m
        if not bars:
            return
        bar = bars[-1]
        if self._state.position_side is TradeSide.LONG:
            mae = (self._state.entry_price - bar.low) / self._state.initial_risk_points
        else:
            mae = (bar.high - self._state.entry_price) / self._state.initial_risk_points
        self._mae_r = max(self._mae_r, max(0.0, mae))

    def _emit_regime_coordination(self, new_regime: Any) -> None:
        """Write enriched regime classification change for sidecar pickup."""
        try:
            data_dir = getattr(self._instrumentation, "_config", {}).get("data_dir") if self._instrumentation else None
            if not data_dir:
                return
            from libs.instrumentation.event_contract import append_jsonl_event, enrich_payload

            now = datetime.now(timezone.utc)
            record = {
                "timestamp": now.isoformat(),
                "action_type": "nq_regime_classification_change",
                "strategy_id": config.STRATEGY_ID,
                "prev_regime": self._prev_regime.name if self._prev_regime else None,
                "new_regime": new_regime.name if new_regime else None,
                "phase": self._state.phase.value,
                "ib_type": self._state.ib_type.value,
            }
            event = enrich_payload(
                record,
                lineage=getattr(self._instrumentation, "lineage", None),
                event_type="coordinator_action",
                scope="family",
            )
            append_jsonl_event(data_dir, "coordination_events", "coordination_events", event)
        except Exception:
            logger.debug("Failed to emit NQ regime coordination event", exc_info=True)

    def _build_strategy_params(self, candidate: SetupCandidate | None) -> dict:
        state = self._state
        params: dict[str, Any] = {
            "regime": state.regime.name if state.regime else "UNKNOWN",
            "module": state.entry_module.value,
            "grade": state.setup_grade.value,
            "score": state.setup_score,
            "ib_type": state.ib_type.value,
            "ib_locked": state.ib_locked,
            "session_phase": state.phase.value,
            "daily_trades": state.daily_trades,
            "daily_losses": state.daily_losses,
            "daily_realized_r": round(state.daily_realized_r, 3),
        }
        if candidate:
            params.update({
                "regime_confidence": candidate.details.get("regime_confidence", 0),
                "regime_margin": candidate.details.get("regime_margin", 0),
                "entry_model": candidate.entry_model,
                "setup_type": candidate.setup_type,
                "target_room_r": candidate.target_room_r,
                "invalidation_price": candidate.invalidation_price,
            })
        return params

    def _build_signal_factors(self, candidate: SetupCandidate | None) -> list[dict]:
        if not candidate:
            return []
        details = candidate.details or {}
        factors = [
            {"factor_name": "module_score", "factor_value": candidate.score, "threshold": 8, "contribution": candidate.score / 12.0},
            {"factor_name": "grade", "factor_value": candidate.grade.value, "threshold": "B", "contribution": 1.0 if candidate.grade.value in ("A+", "A") else 0.5},
            {"factor_name": "target_room_r", "factor_value": candidate.target_room_r, "threshold": 0.5, "contribution": min(1.0, candidate.target_room_r / 2.0)},
        ]
        for key in ("body_pct", "squeeze_duration", "volume_multiple", "vwap_distance", "penetration_depth", "reclaim_pct"):
            if key in details:
                factors.append({"factor_name": key, "factor_value": details[key], "threshold": 0, "contribution": 0})
        return factors

    def _determine_exit_reason(self, events: list, pre_stop_at_be: bool = False) -> str:
        for evt in events:
            code = getattr(evt, "code", "")
            details = getattr(evt, "details", {})
            if code == "EXIT_FILLED":
                role = details.get("role", "")
                if role == "stop":
                    return "TRAILING_STOP" if pre_stop_at_be else "INITIAL_STOP"
                if role == "target_1":
                    return "TARGET_1"
                if role == "target_2":
                    return "TARGET_2"
                if "flatten" in role or "time_stop" in role:
                    return role.upper()
                return role.upper() or "UNKNOWN"
            if code == "PARTIAL_EXIT_FILLED":
                return "PARTIAL_TARGET"
        return "UNKNOWN"

    def emit_heartbeat(self, uptime_s: float, error_count_1h: int = 0) -> None:
        if not self._kit.active:
            return
        positions = []
        if self._state.position_side is not TradeSide.FLAT:
            positions.append({
                "trade_id": self._state.active_trade_id,
                "side": self._state.position_side.value,
                "module": self._state.entry_module.value,
                "entry_price": self._state.entry_price,
                "qty_open": self._state.qty_open,
                "stop": self._state.stop_price,
                "mfe_r": self._mfe_r,
            })
        self._kit.emit_heartbeat(
            active_positions=1 if self._state.position_side is not TradeSide.FLAT else 0,
            open_orders=len([x for x in [self._state.working_entry_order_id, self._state.working_stop_order_id, *self._state.working_target_order_ids] if x]),
            uptime_s=uptime_s,
            error_count_1h=error_count_1h,
            positions=positions,
        )

    # ----------------------------------------------------------------
    # Order construction helpers
    # ----------------------------------------------------------------

    def _order_from_entry(self, action: SubmitEntry) -> OMSOrder:
        inst = self._instrument(action.symbol)
        order = OMSOrder(
            client_order_id=action.client_order_id,
            strategy_id=config.STRATEGY_ID,
            instrument=inst,
            side=OrderSide(action.side),
            qty=action.qty,
            order_type=OrderType(action.order_type),
            limit_price=action.limit_price,
            stop_price=action.stop_price,
            tif=action.tif,
            role=OrderRole.ENTRY,
        )
        stop_for_risk = float(action.risk_context.get("stop_for_risk", 0.0) or 0.0)
        planned_entry = float(action.risk_context.get("planned_entry_price", action.limit_price or action.price or 0.0) or 0.0)
        signal_id = str(action.metadata.get("candidate_id") or action.client_order_id or "")
        signal_ts_raw = action.metadata.get("signal_ts") or ""
        signal_ts = None
        if signal_ts_raw:
            try:
                signal_ts = datetime.fromisoformat(str(signal_ts_raw))
            except ValueError:
                signal_ts = None
        order.risk_context = RiskContext(
            stop_for_risk=stop_for_risk,
            planned_entry_price=planned_entry,
            risk_budget_tag=action.metadata.get("module", config.STRATEGY_ID),
            risk_dollars=abs(planned_entry - stop_for_risk) * action.qty * self._settings.trade_spec.point_value,
            signal_id=signal_id,
            bar_id=f"{action.symbol}:{signal_ts_raw}" if signal_ts_raw else signal_id,
            exchange_timestamp=signal_ts,
        )
        return order

    def _order_from_stop(self, action: SubmitProtectiveStop) -> OMSOrder:
        return OMSOrder(
            client_order_id=action.client_order_id,
            strategy_id=config.STRATEGY_ID,
            instrument=self._instrument(action.symbol),
            side=OrderSide(action.side),
            qty=action.qty,
            order_type=OrderType.STOP,
            stop_price=action.stop_price,
            tif=action.tif,
            role=OrderRole.STOP,
            oca_group=action.oca_group,
        )

    def _order_from_target(self, action: SubmitProfitTarget) -> OMSOrder:
        return OMSOrder(
            client_order_id=action.client_order_id,
            strategy_id=config.STRATEGY_ID,
            instrument=self._instrument(action.symbol),
            side=OrderSide(action.side),
            qty=action.qty,
            order_type=OrderType.LIMIT,
            limit_price=action.limit_price,
            tif=action.tif,
            role=OrderRole.TP,
            oca_group=action.oca_group,
        )

    def _instrument(self, symbol: str) -> Any:
        inst = self._instruments.get(symbol) or self._instruments.get(symbol.upper())
        if inst is None:
            built = config.build_instruments()
            inst = built.get(symbol.upper())
            self._instruments.update(built)
        return inst

    def _promote_order_id(self, client_order_id: str, oms_order_id: str | None) -> None:
        if not oms_order_id or oms_order_id == client_order_id:
            return
        role = self._state.order_to_role.pop(client_order_id, "")
        candidate = self._state.order_to_candidate.pop(client_order_id, "")
        if role:
            self._state.order_to_role[oms_order_id] = role
        if candidate:
            self._state.order_to_candidate[oms_order_id] = candidate
        if self._state.working_entry_order_id == client_order_id:
            self._state.working_entry_order_id = oms_order_id
        if self._state.working_stop_order_id == client_order_id:
            self._state.working_stop_order_id = oms_order_id
        targets = tuple(oms_order_id if item == client_order_id else item for item in self._state.working_target_order_ids)
        self._state.working_target_order_ids = targets

    def _sync_health_fields(self) -> None:
        self._last_decision_code = self._state.last_decision_code
        self._last_decision_details = dict(self._state.last_decision_details)
        self._last_bar_ts = self._state.last_bar_ts

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._bars_processed,
            "symbol_freshness": {
                sym: ts.isoformat() for sym, ts in self._symbol_last_bar_ts.items()
            },
        }


def _bar_data_from_any(bar: Any) -> BarData:
    if isinstance(bar, BarData):
        return bar
    ts = getattr(bar, "ts", None) or getattr(bar, "date", None) or getattr(bar, "time", None) or getattr(bar, "timestamp", None)
    if ts is None and isinstance(bar, dict):
        ts = bar.get("ts") or bar.get("date") or bar.get("time") or bar.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts is None:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    getter = bar.get if isinstance(bar, dict) else lambda key, default=None: getattr(bar, key, default)
    return BarData(
        ts=ts,
        open=float(getter("open", 0.0)),
        high=float(getter("high", 0.0)),
        low=float(getter("low", 0.0)),
        close=float(getter("close", 0.0)),
        volume=float(getter("volume", 0.0) or 0.0),
        vwap=float(getter("vwap", 0.0)) if getter("vwap", None) is not None else None,
    )


def _event_ts(event: Any) -> datetime:
    ts = getattr(event, "timestamp", None) or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts
