"""EMA crossover overlay engine — deploys idle capital into ETFs.

Daily-rebalancing capital allocator that places market orders directly via the
IB API, bypassing OMS entirely. Runs on a 16:15 ET daily schedule and persists
state to JSON for crash recovery.

Ported from backtest/engine/unified_portfolio_engine.py (legacy "ema" mode).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import OverlayConfig
from .shared import allocate_weighted_targets, compute_ema

logger = logging.getLogger(__name__)

from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# EMA computation (ported from backtest)
# ---------------------------------------------------------------------------

def _compute_ema(series: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed shared with the backtest mirror."""
    return compute_ema(series, period)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OverlayEngine:
    """Async live engine for idle-capital EMA crossover overlay."""

    def __init__(
        self,
        ib_session: Any,
        equity: float,
        config: OverlayConfig,
        market_calendar: Any | None = None,
        instrumentation: Any | None = None,
        equity_offset: float = 0.0,
        db_pool: Any | None = None,
        get_deployed_capital: Any | None = None,
        equity_alloc_pct: float = 1.0,
        disable_scheduler: bool = False,
    ) -> None:
        self._ib = ib_session
        self._equity = equity
        self._config = config
        self._market_cal = market_calendar
        self._instr = getattr(instrumentation, 'ctx', instrumentation) if instrumentation else None
        self._equity_offset = equity_offset  # paper capital offset applied on refresh
        self._db_pool = db_pool
        self._get_deployed_capital = get_deployed_capital  # callback: () -> float (swing OMS notional)
        self._equity_alloc_pct = equity_alloc_pct
        self._disable_scheduler = bool(disable_scheduler)

        # Resolved IB contracts: symbol -> Contract
        self._contracts: dict[str, Any] = {}

        # Current overlay shares (loaded from / saved to state file)
        self._shares: dict[str, int] = {sym: 0 for sym in config.symbols}
        self._last_rebalance_date: str = ""
        # Trade IDs for open overlay positions (persisted for exit instrumentation)
        self._entry_trade_ids: dict[str, str] = {}
        # Last EMA crossover signals (persisted for transition detection)
        self._last_signals: dict[str, bool] = {}

        # Async state
        self._daily_task: asyncio.Task | None = None
        self._running = False

        # Diagnostic pulse state
        self._last_decision_code: str = "IDLE"
        self._last_decision_details: dict = {}
        self._last_bar_ts: datetime | None = None
        self._rebalances_completed: int = 0

    def _record_decision(self, code: str, details: dict | None = None) -> None:
        self._last_decision_code = code
        self._last_decision_details = details or {}

    def health_status(self) -> dict:
        return {
            "strategy_id": "OVERLAY",
            "running": self._running,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": self._last_decision_details,
            "last_bar_ts": self._last_bar_ts.isoformat() if self._last_bar_ts else None,
        }

    def liveness_payload(self) -> dict:
        return {
            "bars_processed": self._rebalances_completed,
            "last_rebalance_date": self._last_rebalance_date or None,
            "symbol_freshness": {},
        }

    def build_rebalance_plan_from_bars(
        self,
        daily_bars: dict[str, Any],
        *,
        equity: float | None = None,
        deployed_capital: float | None = None,
        min_bars: int = 50,
    ) -> dict[str, Any]:
        """Plan an overlay rebalance from daily bars without placing orders."""

        base_equity = float(equity if equity is not None else self._equity)
        deployed = (
            float(deployed_capital)
            if deployed_capital is not None
            else self._resolve_deployed_capital()
        )
        net_equity = max(base_equity - deployed, 0.0)
        available = max(net_equity * self._config.max_equity_pct, 0.0)
        signals: dict[str, bool] = {}
        prices: dict[str, float] = {}
        ema_cache: dict[str, tuple[float, float, int, int]] = {}

        for sym in self._config.symbols:
            rows = daily_bars.get(sym) or daily_bars.get(str(sym).upper()) or []
            if not rows or len(rows) < min_bars:
                logger.warning("Overlay: insufficient bars for %s (%d)", sym, len(rows) if rows else 0)
                signals[sym] = False
                continue

            closes = np.array([self._bar_close(row) for row in rows], dtype=float)
            prices[sym] = float(closes[-1])
            fast, slow = self._config.ema_overrides.get(
                sym,
                (self._config.ema_fast, self._config.ema_slow),
            )
            ema_fast = _compute_ema(closes, fast)
            ema_slow = _compute_ema(closes, slow)
            if np.isnan(ema_fast[-1]) or np.isnan(ema_slow[-1]):
                signals[sym] = False
            else:
                signals[sym] = bool(ema_fast[-1] > ema_slow[-1])
                ema_cache[sym] = (float(ema_fast[-1]), float(ema_slow[-1]), fast, slow)
            logger.info(
                "Overlay: %s EMA(%d)=%.2f EMA(%d)=%.2f -> %s",
                sym, fast, ema_fast[-1], slow, ema_slow[-1],
                "BULLISH" if signals[sym] else "BEARISH",
            )

        target_shares = allocate_weighted_targets(
            self._config.symbols,
            signals=signals,
            prices=prices,
            portfolio_equity=net_equity,
            max_equity_pct=self._config.max_equity_pct,
            weights=self._config.weights,
        )
        bullish_w = {s: 1.0 for s in self._config.symbols if signals.get(s)}
        if self._config.weights is not None:
            bullish_w = {
                s: self._config.weights.get(s, 1.0)
                for s in self._config.symbols if signals.get(s)
            }
        return {
            "signals": signals,
            "prices": prices,
            "ema_cache": ema_cache,
            "target_shares": target_shares,
            "bullish_weights": bullish_w,
            "total_weight": sum(bullish_w.values()),
            "equity": base_equity,
            "deployed_capital": deployed,
            "net_equity": net_equity,
            "available_capital": available,
        }

    def apply_rebalance_plan_dry_run(
        self,
        plan: dict[str, Any],
        *,
        timestamp: str | datetime | None = None,
        reason: str = "fixture",
    ) -> dict[str, Any]:
        """Apply a planned rebalance to state without broker orders."""

        target_shares = plan.get("target_shares", {}) or {}
        for symbol in self._config.symbols:
            self._shares[symbol] = int(target_shares.get(symbol, 0))
        self._last_signals = dict(plan.get("signals", {}) or {})
        self._last_rebalance_date = str(timestamp or datetime.now(timezone.utc))[:10]
        self._rebalances_completed += 1
        active_symbols = [symbol for symbol in self._config.symbols if self._shares.get(symbol, 0) > 0]
        self._record_decision(
            "MANAGING_POSITION" if active_symbols else "NO_SIGNAL",
            {
                "reason": reason,
                "target_shares": {
                    symbol: int(target_shares.get(symbol, 0))
                    for symbol in self._config.symbols
                },
                "prices": {
                    symbol: (plan.get("prices", {}) or {}).get(symbol)
                    for symbol in self._config.symbols
                    if symbol in (plan.get("prices", {}) or {})
                },
                "equity": plan.get("equity"),
                "deployed_capital": plan.get("deployed_capital"),
                "net_equity": plan.get("net_equity"),
                "available_capital": plan.get("available_capital"),
            },
        )
        return {
            "positions": self.get_positions(),
            "signals": self.get_signals(),
            "last_rebalance_date": self._last_rebalance_date,
            "last_decision_code": self._last_decision_code,
            "last_decision_details": dict(self._last_decision_details),
            "rebalances_completed": self._rebalances_completed,
        }

    def _resolve_deployed_capital(self) -> float:
        if self._get_deployed_capital is None:
            return 0.0
        try:
            return float(self._get_deployed_capital())
        except Exception:
            logger.warning("Overlay: could not query deployed capital, assuming $0")
            return 0.0

    @staticmethod
    def _bar_close(row: Any) -> float:
        if isinstance(row, dict):
            return float(row["close"])
        return float(row.close)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Qualify ETF contracts, load state, launch daily scheduler."""
        logger.info("Overlay engine starting …")
        self._running = True

        if not self._disable_scheduler:
            # Resolve ETF contracts through the shared contract factory when available.
            cf = getattr(self._ib, "_contract_factory", None)
            for sym in self._config.symbols:
                try:
                    if cf is not None:
                        contract, _ = await cf.resolve(sym)
                        self._contracts[sym] = contract
                        continue
                    from ib_async import Stock
                    contract = Stock(sym, "SMART", "USD")
                    qualified = await self._ib.ib.qualifyContractsAsync(contract)
                    if qualified:
                        self._contracts[sym] = qualified[0]
                except Exception as e:
                    logger.warning("Overlay: could not resolve contract for %s: %s", sym, e)

        # Load persisted state
        self._load_state()

        # Launch daily scheduler
        if not self._disable_scheduler:
            self._daily_task = asyncio.create_task(self._daily_scheduler())

        logger.info(
            "Overlay engine started (symbols: %s, shares: %s)",
            list(self._contracts.keys()), self._shares,
        )

    async def stop(self) -> None:
        """Cancel scheduler, save state."""
        logger.info("Overlay engine stopping …")
        self._running = False

        if self._daily_task:
            self._daily_task.cancel()
            try:
                await self._daily_task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("Overlay engine stopped")

    # ------------------------------------------------------------------
    # Daily scheduler
    # ------------------------------------------------------------------

    async def _daily_scheduler(self) -> None:
        """Sleep until 16:15 ET each trading day, then rebalance."""
        while self._running:
            now = datetime.now(timezone.utc)
            try:
                now_et = now.astimezone(ET)
            except Exception:
                now_et = now
            target = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
            if target <= now_et:
                target += timedelta(days=1)
            # Skip weekends and holidays
            while target.weekday() >= 5 or (
                self._market_cal
                and not self._market_cal.is_trading_day(target.date())
            ):
                target += timedelta(days=1)
            wait = (target - now_et).total_seconds()
            await asyncio.sleep(max(1, wait))
            if not self._running:
                break
            try:
                await self._daily_rebalance()
            except Exception:
                logger.exception("Overlay: error in daily rebalance")

    # ------------------------------------------------------------------
    # Rebalance logic (ported from backtest legacy "ema" mode)
    # ------------------------------------------------------------------

    async def _daily_rebalance(self) -> None:
        """Fetch bars, compute EMAs, rebalance overlay positions."""
        logger.info("Overlay: === Daily rebalance ===")
        self._last_bar_ts = datetime.now(timezone.utc)
        self._rebalances_completed += 1

        # 1. Refresh equity from IB
        await self._refresh_equity()

        deployed = self._resolve_deployed_capital()
        net_equity = max(self._equity - deployed, 0.0)
        available = max(net_equity * self._config.max_equity_pct, 0.0)
        if deployed > 0:
            logger.info("Overlay: equity=$%.2f deployed=$%.2f net=$%.2f available=$%.2f",
                        self._equity, deployed, net_equity, available)

        # 2-3. Fetch bars; the shared planner computes EMAs and targets.
        daily_bars: dict[str, Any] = {}

        for sym in self._config.symbols:
            contract = self._contracts.get(sym)
            if not contract:
                daily_bars[sym] = []
                continue

            try:
                bars = await self._ib.req_historical_data(
                    contract, endDateTime="", durationStr="200 D",
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, formatDate=1, request_kind="recurring",
                )
            except Exception:
                logger.warning("Overlay: failed to fetch bars for %s", sym)
                daily_bars[sym] = []
                continue

            daily_bars[sym] = list(bars or [])

        plan = self.build_rebalance_plan_from_bars(
            daily_bars,
            equity=self._equity,
            deployed_capital=deployed,
            min_bars=50,
        )
        signals = plan["signals"]
        prices = plan["prices"]
        ema_cache = plan["ema_cache"]
        target_shares = plan["target_shares"]
        bullish_w = plan["bullish_weights"]
        total_w = plan["total_weight"]

        # 4b. Log signal transitions via coordination logger
        if self._last_signals:
            for sym in self._config.symbols:
                old = self._last_signals.get(sym)
                new = signals.get(sym, False)
                if old is not None and old != new:
                    try:
                        if self._instr and getattr(self._instr, 'coordination_logger', None):
                            self._instr.coordination_logger.log_action(
                                action="overlay_signal_change",
                                trigger_strategy="OVERLAY",
                                target_strategy="ALL",
                                symbol=sym,
                                rule="ema_crossover",
                                details={
                                    "old_bullish": old,
                                    "new_bullish": new,
                                    "direction": "BULLISH" if new else "BEARISH",
                                },
                                outcome="emitted",
                            )
                    except Exception:
                        pass
        self._last_signals = dict(signals)

        # 6-8. Compute deltas and place orders
        for sym in self._config.symbols:
            target = target_shares.get(sym, 0)
            current = self._shares.get(sym, 0)
            delta = target - current

            if delta == 0:
                logger.info("Overlay: %s no change (target=%d, current=%d)", sym, target, current)
                continue

            contract = self._contracts.get(sym)
            if not contract:
                continue

            # Detect entry (0 → >0) and exit (>0 → 0) transitions
            entering = current == 0 and target > 0
            exiting = current > 0 and target == 0
            if entering:
                self._record_decision("ENTRY_SUBMITTED", {
                    "symbol": sym, "target_shares": target,
                })

            action = "BUY" if delta > 0 else "SELL"
            qty = abs(delta)

            try:
                from ib_async import MarketOrder
                order = MarketOrder(action, qty)
                trade = self._ib.ib.placeOrder(contract, order)
                logger.info(
                    "Overlay: %s %s %d shares (current=%d → target=%d)",
                    sym, action, qty, current, target,
                )

                # Wait for fill (with timeout)
                fill_price = prices.get(sym, 0.0)
                try:
                    await asyncio.wait_for(trade.filledEvent, timeout=60.0)
                    fill_price = trade.orderStatus.avgFillPrice or fill_price
                    logger.info(
                        "Overlay: %s fill confirmed — %d @ %.2f",
                        sym, qty, fill_price,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Overlay: %s fill timeout — order may fill at next RTH open",
                        sym,
                    )

                # Update tracked shares
                self._shares[sym] = target

                # Hook 4: entry instrumentation (0 → >0)
                if self._instr and entering:
                    try:
                        from strategies.swing.instrumentation.src.hooks import safe_instrument
                        regime = self._instr.regime_classifier.current_regime(sym)
                        tid = f"overlay_{sym}_{datetime.now(timezone.utc).isoformat()}"
                        self._entry_trade_ids[sym] = tid
                        ema_vals = ema_cache.get(sym, (0.0, 0.0, self._config.ema_fast, self._config.ema_slow))
                        safe_instrument(
                            self._instr.trade_logger.log_entry,
                            trade_id=tid,
                            pair=sym,
                            side="LONG",
                            entry_price=fill_price,
                            position_size=float(target),
                            position_size_quote=float(target) * fill_price,
                            entry_signal="ema_crossover_overlay",
                            entry_signal_id=f"overlay_{sym}",
                            entry_signal_strength=0.5,
                            signal_factors=[
                                {"factor_name": "ema_fast", "factor_value": ema_vals[0],
                                 "threshold": ema_vals[1], "contribution": 0.5},
                                {"factor_name": "ema_slow", "factor_value": ema_vals[1],
                                 "threshold": 0, "contribution": 0.5},
                                {"factor_name": "crossover_direction", "factor_value": "BULLISH",
                                 "threshold": "BULLISH", "contribution": 1.0},
                            ],
                            sizing_inputs={
                                "equity": self._equity,
                                "deployed_capital": deployed,
                                "net_equity": net_equity,
                                "available_capital": available,
                                "allocation_pct": self._config.max_equity_pct,
                                "weight": bullish_w.get(sym, 0.0),
                                "total_weight": total_w,
                                "target_shares": target,
                                "price": prices.get(sym, 0),
                            },
                            portfolio_state={
                                "equity": self._equity,
                                "deployed_capital": deployed,
                                "overlay_positions": {s: sh for s, sh in self._shares.items() if sh > 0},
                                "bullish_count": sum(1 for v in signals.values() if v),
                                "total_symbols": len(self._config.symbols),
                            },
                            filter_decisions=[],
                            active_filters=[],
                            passed_filters=[],
                            strategy_params={
                                "ema_fast_period": ema_vals[2],
                                "ema_slow_period": ema_vals[3],
                                "ema_overrides": str(self._config.ema_overrides.get(sym)),
                            },
                            strategy_id="OVERLAY",
                            expected_entry_price=prices.get(sym, 0),
                            market_regime=regime,
                        )
                    except Exception:
                        pass

                # Hook 5: exit instrumentation (>0 → 0) + process scoring
                if self._instr and exiting:
                    try:
                        from strategies.swing.instrumentation.src.hooks import safe_instrument
                        tid = self._entry_trade_ids.pop(sym, f"overlay_{sym}")
                        entry_price_est = prices.get(sym, fill_price)
                        pnl_pct = ((fill_price - entry_price_est) / entry_price_est * 100) if entry_price_est > 0 else 0.0
                        trade_event = safe_instrument(
                            self._instr.trade_logger.log_exit,
                            trade_id=tid,
                            exit_price=fill_price,
                            exit_reason="EMA_BEARISH",
                            pnl_pct=round(pnl_pct, 4),
                            position_size=float(current),
                            position_size_quote=float(current) * fill_price,
                            expected_exit_price=prices.get(sym, 0),
                        )
                        if trade_event:
                            safe_instrument(
                                self._instr.process_scorer.score_and_write,
                                trade_event.to_dict(),
                                "OVERLAY",
                                self._instr.data_dir,
                            )
                    except Exception:
                        pass

            except Exception:
                logger.exception("Overlay: failed to place order for %s", sym)
                continue

        # Hook 1: regime classification + market snapshot (post-rebalance)
        if self._instr:
            try:
                from strategies.swing.instrumentation.src.hooks import safe_instrument, async_safe_instrument
                for sym in self._config.symbols:
                    await async_safe_instrument(self._instr.regime_classifier.classify, sym)
                    safe_instrument(self._instr.snapshot_service.capture_now, sym)
            except Exception:
                pass

        self._last_rebalance_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_state()

        # Persist overlay positions to DB for dashboard visibility
        await self._persist_positions_to_db(prices)

        # Record final decision
        active_syms = [s for s, sh in self._shares.items() if sh > 0]
        if active_syms:
            self._record_decision("MANAGING_POSITION", {
                "active_symbols": active_syms,
                "shares": {s: sh for s, sh in self._shares.items() if sh > 0},
            })
        else:
            self._record_decision("NO_SIGNAL", {
                "signals": {s: v for s, v in signals.items()},
            })

        logger.info("Overlay: rebalance complete — shares: %s", self._shares)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load overlay state from JSON file."""
        path = Path(self._config.state_file)
        if not path.exists():
            logger.info("Overlay: no state file found, starting fresh")
            return
        try:
            data = json.loads(path.read_text())
            self._shares = {sym: data.get("shares", {}).get(sym, 0) for sym in self._config.symbols}
            self._last_rebalance_date = data.get("last_rebalance_date", "")
            self._entry_trade_ids = data.get("entry_trade_ids", {})
            self._last_signals = data.get("last_signals", {})
            logger.info(
                "Overlay: loaded state — shares=%s, last_rebalance=%s",
                self._shares, self._last_rebalance_date,
            )
        except Exception:
            logger.warning("Overlay: failed to load state file, starting fresh")

    def _save_state(self) -> None:
        """Save overlay state to JSON file."""
        path = Path(self._config.state_file)
        data = {
            "shares": self._shares,
            "last_rebalance_date": self._last_rebalance_date,
            "entry_trade_ids": self._entry_trade_ids,
            "last_signals": self._last_signals,
        }
        try:
            path.write_text(json.dumps(data, indent=2))
            logger.debug("Overlay: state saved to %s", path)
        except Exception:
            logger.warning("Overlay: failed to save state file")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_signals(self) -> dict[str, bool]:
        """Return last computed EMA crossover signals per symbol."""
        return dict(self._last_signals)

    def get_positions(self) -> dict[str, int]:
        """Return current overlay share counts."""
        return dict(self._shares)

    def get_state_summary(self) -> str:
        """Return human-readable state summary for logging."""
        parts = [f"{sym}={qty}" for sym, qty in self._shares.items() if qty > 0]
        if not parts:
            return "Overlay: no positions"
        return f"Overlay: {', '.join(parts)} (last rebalance: {self._last_rebalance_date})"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _persist_positions_to_db(self, prices: dict[str, float]) -> None:
        """Write current overlay positions to PostgreSQL for dashboard visibility."""
        if not self._db_pool:
            return
        try:
            from libs.oms.persistence.postgres import PgStore
            store = PgStore(self._db_pool)
            now_utc = datetime.now(timezone.utc)
            rows = []
            for sym in self._config.symbols:
                shares = self._shares.get(sym, 0)
                price = prices.get(sym, 0.0)
                notional = shares * price
                pct = notional / self._equity if self._equity > 0 else 0.0
                rows.append({
                    "symbol": sym,
                    "shares": shares,
                    "notional": notional,
                    "pct_of_nav": pct,
                    "rebalance_ts": now_utc,
                })
            await store.upsert_overlay_positions(rows)
            logger.info("Overlay: positions persisted to DB (%d symbols)", len(rows))
        except Exception:
            logger.warning("Overlay: failed to persist positions to DB", exc_info=True)

    async def _refresh_equity(self) -> None:
        """Fetch current account equity from IB (applies paper capital offset)."""
        try:
            accounts = self._ib.ib.managedAccounts()
            if accounts:
                for item in self._ib.ib.accountValues():
                    if item.tag == "NetLiquidation" and item.currency == "USD" and item.account == accounts[0]:
                        raw = float(item.value)
                        self._equity = raw * self._equity_alloc_pct + self._equity_offset
                        logger.info("Overlay: equity refreshed — $%.2f", self._equity)
                        return
        except Exception:
            logger.warning("Overlay: could not refresh equity, using $%.2f", self._equity)
