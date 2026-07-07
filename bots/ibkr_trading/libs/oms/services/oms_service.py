"""Main OMS service."""
import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

if TYPE_CHECKING:
    from ..intent.handler import IntentHandler
    from ..events.bus import EventBus
    from ..execution.router import ExecutionRouter
    from ..reconciliation.orchestrator import ReconciliationOrchestrator
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    from ..models.intent import Intent, IntentReceipt
    from ..models.risk_state import PortfolioRiskState, StrategyRiskState

logger = logging.getLogger(__name__)


class OMSService:
    """Main OMS runtime. Exposes intent API, manages lifecycle."""

    def __init__(
        self,
        intent_handler: "IntentHandler",
        bus: "EventBus",
        reconciler: "ReconciliationOrchestrator",
        router: "ExecutionRouter" = None,
        recon_interval_s: float = 120.0,
        timeout_monitor: "OrderTimeoutMonitor | None" = None,
        get_portfolio_risk: "Callable[[], Awaitable[PortfolioRiskState]] | None" = None,
        get_strategy_risk: "Callable[[str], Awaitable[StrategyRiskState]] | None" = None,
        on_intent_denied: Optional[Callable[[dict], None]] = None,
    ):
        self._handler = intent_handler
        self._bus = bus
        self._reconciler = reconciler
        self._router = router
        self._recon_interval = recon_interval_s
        self._timeout_monitor = timeout_monitor
        self._get_portfolio_risk = get_portfolio_risk
        self._get_strategy_risk = get_strategy_risk
        # Forensic per-denial event sink (parallels PortfolioRuleChecker.on_rule_event).
        # Counters above tell the watchdog "many denials happening";
        # this callback lets operators audit each one with rule + reason.
        self._on_intent_denied = on_intent_denied
        self._ready = asyncio.Event()
        self._recon_task: asyncio.Task | None = None

        # Execution health counters (read by coordinator heartbeat)
        self._intents_submitted: int = 0
        self._intents_accepted: int = 0
        self._intents_denied: int = 0
        self._consecutive_denials: int = 0
        self._last_accepted_ts: datetime | None = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def event_bus(self) -> "EventBus":
        return self._bus

    async def start(self) -> None:
        """Start OMS: reconcile, start router, start timeout monitor, then accept intents."""
        authoritative = bool(getattr(self._reconciler, "is_authoritative", True))
        if authoritative:
            await self._reconciler.startup_reconciliation()
        else:
            logger.info("OMS service skipping startup reconciliation (non-authoritative)")
        if self._router:
            await self._router.start()
        # C4 fix: start order timeout monitor
        if self._timeout_monitor:
            await self._timeout_monitor.start()
        if authoritative:
            self._recon_task = asyncio.create_task(self._periodic_recon_loop())
        self._ready.set()
        logger.info("OMS service ready")

    async def stop(self) -> None:
        self._ready.clear()
        if self._timeout_monitor:
            await self._timeout_monitor.stop()
        if self._router:
            await self._router.stop()
        if self._recon_task:
            self._recon_task.cancel()
            try:
                await self._recon_task
            except asyncio.CancelledError:
                pass

    async def submit_intent(self, intent: "Intent") -> "IntentReceipt":
        await self._ready.wait()
        self._intents_submitted += 1
        receipt = await self._handler.submit(intent)
        from ..models.intent import IntentResult
        if receipt.result == IntentResult.ACCEPTED:
            self._intents_accepted += 1
            self._consecutive_denials = 0
            self._last_accepted_ts = datetime.now(timezone.utc)
        elif receipt.result == IntentResult.DENIED:
            self._intents_denied += 1
            self._consecutive_denials += 1
            if self._on_intent_denied is not None:
                order = intent.order
                event = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "strategy_id": getattr(order, "strategy_id", "") if order else "",
                    "symbol": (
                        order.instrument.symbol
                        if order and getattr(order, "instrument", None)
                        else ""
                    ),
                    "side": getattr(order.side, "value", str(order.side)) if order and order.side is not None else "",
                    "qty": getattr(order, "qty", 0) if order else 0,
                    "role": getattr(order.role, "value", str(order.role)) if order and order.role is not None else "",
                    "intent_id": receipt.intent_id,
                    "denial_reason": receipt.denial_reason or "",
                    "consecutive_denials": self._consecutive_denials,
                }
                try:
                    self._on_intent_denied(event)
                except Exception:
                    logger.debug("on_intent_denied callback failed", exc_info=True)
        return receipt

    async def submit_preapproved_family_intent(
        self,
        *,
        strategy_id: str,
        order: "OMSOrder",
        decision: "PreapprovedFamilyDecision",
    ) -> "IntentReceipt":
        from ..models.intent import Intent, IntentType

        return await self.submit_intent(
            Intent(
                intent_type=IntentType.PREAPPROVED_ORDER,
                strategy_id=strategy_id,
                order=order,
                preapproved_family_decision=decision,
            )
        )

    def stream_events(self, strategy_id: str) -> "asyncio.Queue":
        """Returns an asyncio.Queue that receives OMSEvent objects for this strategy."""
        return self._bus.subscribe(strategy_id)

    def stream_all_events(self) -> "asyncio.Queue":
        """For dashboard/logging — receives all events."""
        return self._bus.subscribe_all()

    async def request_reconciliation(self) -> None:
        """Trigger an immediate OMS-vs-broker reconciliation (e.g. after reconnect)."""
        await self._reconciler.on_reconnect_reconciliation()

    async def get_portfolio_risk(self) -> "PortfolioRiskState":
        if self._get_portfolio_risk is None:
            raise RuntimeError("Portfolio risk provider is not configured")
        return await self._get_portfolio_risk()

    async def get_strategy_risk(self, strategy_id: str) -> "StrategyRiskState":
        if self._get_strategy_risk is None:
            raise RuntimeError("Strategy risk provider is not configured")
        return await self._get_strategy_risk(strategy_id)

    async def _periodic_recon_loop(self) -> None:
        while True:
            await asyncio.sleep(self._recon_interval)
            try:
                await self._reconciler.periodic_reconciliation()
            except Exception as e:
                logger.error(f"Periodic recon failed: {e}")
