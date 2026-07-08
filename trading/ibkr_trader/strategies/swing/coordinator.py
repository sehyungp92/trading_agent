"""Swing family coordinator — orchestrates swing strategies sharing one OMS.

Strategies by priority: ATRSS(0), TPC(1), AKC_HELIX(2). Shares a single IBKR
adapter, multi-strategy OMS, StrategyCoordinator, and OverlayEngine across
all engines.

Cross-strategy coordination:
  - ATRSS entry fill on symbol X -> tighten Helix stop to breakeven on X
  - has_atrss_position() -> Helix 1.25x size boost when ATRSS confirms direction

Extracted from swing_trader/main_multi.py (826 lines) into a clean coordinator
that receives its dependencies via RuntimeContext.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
from contextlib import suppress
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from libs.oms.persistence.db_config import get_environment
from libs.runtime.active_config import (
    ActiveRuntimeConfigRecord,
    active_config_expiry,
    build_family_runtime_config,
    build_strategy_runtime_config,
    upsert_active_runtime_config,
)
from libs.services.heartbeat import emit_family_heartbeats
from strategies.contracts import RuntimeContext
from strategies.core.capital import build_family_allocation_targets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-strategy risk parameters (unit_risk_pct passed to RiskCalculator)
# ---------------------------------------------------------------------------
_RISK_PARAMS: dict[str, dict[str, Any]] = {
    "ATRSS": {
        "unit_risk_pct": 0.0165,
        "daily_stop_R": 2.25,
        "priority": 0,            # highest expectancy
        "max_heat_R": 2.15,
        "max_working_orders": 4,
    },
    "AKC_HELIX": {
        "unit_risk_pct": 0.013,
        "daily_stop_R": 2.5,
        "priority": 2,
        "max_heat_R": 2.10,
        "max_working_orders": 4,
    },
    "TPC": {
        "unit_risk_pct": 0.005,
        "daily_stop_R": 2.0,
        "priority": 1,
        "max_heat_R": 4.00,
        "max_working_orders": 3,
    },
}

# Swing-family-level risk caps from swing portfolio synergy round 3.
# Two-tier model: portfolio.yaml.risk.* are ACCOUNT-level caps that the
# AccountRiskGate enforces across all families (default 2.5R / 3.0R / 5.0R).
# These constants are intra-family: they bound the combined exposure of
# ATRSS+Helix+TPC inside the swing OMS, sized for swing's higher win-rate
# distribution. Both layers must approve before an entry is admitted.
_SWING_FAMILY_HEAT_CAP_R = 5.5
_SWING_FAMILY_DAILY_STOP_R = 3.75
_SWING_FAMILY_WEEKLY_STOP_R = 9.0
_DD_TIERS = (
    (0.04, 0.90),
    (0.07, 0.70),
    (0.10, 0.50),
    (0.14, 0.25),
    (0.18, 0.00),
)


def _swing_family_nav(
    base_equity: float,
    allocs: dict[str, Any],
    strategy_ids: tuple[str, ...],
    ctx: RuntimeContext,
) -> float:
    """Return the swing-family NAV used by optimized portfolio sizing."""
    for strategy_id in strategy_ids:
        alloc = allocs.get(strategy_id)
        family_fraction = getattr(alloc, "family_fraction", None)
        if _positive_finite(family_fraction):
            return float(base_equity) * float(family_fraction)

    capital = getattr(getattr(ctx, "portfolio", None), "capital", None)
    family_allocations = getattr(capital, "family_allocations", {}) or {}
    family_fraction = family_allocations.get("swing")
    if _positive_finite(family_fraction):
        return float(base_equity) * float(family_fraction)

    allocated = [
        float(getattr(allocs.get(strategy_id), "allocated_nav", 0.0) or 0.0)
        for strategy_id in strategy_ids
    ]
    total_allocated = sum(value for value in allocated if _positive_finite(value))
    if total_allocated > 0:
        return total_allocated
    return float(base_equity)


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _call_override_factory(factory, **kwargs):
    try:
        return factory(**kwargs)
    except TypeError:
        return factory()


class SwingFamilyCoordinator:
    """Orchestrates swing strategies sharing one OMS instance.

    Lifecycle:
        coordinator = SwingFamilyCoordinator(ctx)
        await coordinator.start()
        ...
        await coordinator.stop()
    """

    family_id = "swing"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list[tuple[str, Any]] = []  # (strategy_id, engine)
        self._oms: Any = None
        self._coordinator: Any = None  # StrategyCoordinator
        self._overlay_engine: Any = None
        self._instrumentation_ctx: Any = None
        self._kits: dict[str, Any] = {}
        self._portfolio_checker: Any = None
        self._base_portfolio_rules: Any = None
        self._regime_ctx: Any = None
        self._regime_adjusted_rules: Any = None  # stored after apply_regime for crisis overlay
        self._crisis_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None
        self._base_overlay_max_equity_pct: float | None = None

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Build shared OMS, create engines, wire overlay, and start."""
        # -- Lazy imports (keep module-level lightweight) ------------------
        from libs.oms.services.factory import build_multi_strategy_oms as default_build_multi_strategy_oms
        from libs.oms.risk.calculator import RiskCalculator
        from libs.config.capital_bootstrap import bootstrap_capital

        from strategies.swing.atrss.config import (
            STRATEGY_ID as ATRSS_ID,
            SYMBOL_CONFIGS as ATRSS_CONFIGS,
            build_instruments as atrss_build_instruments,
        )
        from strategies.swing.atrss.engine import ATRSSEngine

        from strategies.swing.akc_helix.config import (
            STRATEGY_ID as HELIX_ID,
            SYMBOL_CONFIGS as HELIX_CONFIGS,
            build_instruments as helix_build_instruments,
        )
        from strategies.swing.akc_helix.engine import HelixEngine

        from strategies.swing.tpc.config import (
            STRATEGY_ID as TPC_ID,
            SYMBOL_CONFIGS as TPC_CONFIGS,
            build_instruments as tpc_build_instruments,
        )
        from strategies.swing.tpc.engine import TPCEngine

        ctx = self._ctx
        require_instrumentation = bool(getattr(ctx, "require_instrumentation", False))
        overrides = getattr(ctx, "runtime_overrides", None)
        build_multi_strategy_oms = (
            getattr(overrides, "build_multi_strategy_oms", None)
            or default_build_multi_strategy_oms
        )
        session = ctx.session
        db_pool = getattr(ctx, "db_pool", None)
        override_strategy_ids = None
        if overrides is not None:
            provider = getattr(overrides, "strategy_ids_provider", None)
            override_strategy_ids = provider() if provider is not None else getattr(overrides, "strategy_ids", None)
        strategy_ids = (
            list(dict.fromkeys(str(sid) for sid in override_strategy_ids))
            if override_strategy_ids is not None
            else [ATRSS_ID, HELIX_ID, TPC_ID]
        )

        # -- Config dir (needed for adapter + capital bootstrap) -----------
        config_dir = Path(
            os.environ.get(
                "CONFIG_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "config"),
            )
        )

        # -- Build execution adapter (same pattern as momentum/stock) ------
        adapter = None
        contract_factory = None
        adapter_factory = getattr(overrides, "adapter_factory", None)
        if adapter_factory is not None:
            adapter = _call_override_factory(adapter_factory, session=session)
            ibkr_config = None
        elif session is not None:
            from libs.broker_ibkr.config.loader import IBKRConfig
            from libs.broker_ibkr.mapping.contract_factory import ContractFactory
            from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

            # SWING-2: hard-fail on IBKRConfig load. The previous soft-degrade
            # let the runtime "start" with adapter=None, log healthy
            # heartbeats for hours, and never place an order. Compare with
            # momentum/coordinator.py:89-94 which already raises on the same
            # path. Keep the soft path only when session is None (dev/shadow).
            try:
                ibkr_config = IBKRConfig(config_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Swing IBKRConfig load failed — ensure "
                    f"config/ibkr_profiles.yaml exists and IB_ACCOUNT_ID is "
                    f"set: {exc}"
                ) from exc

            contract_factory = ContractFactory(
                ib=session.ib,
                templates=ibkr_config.contracts,
                routes=ibkr_config.routes,
            )
            setattr(session, "_contract_factory", contract_factory)
            adapter = IBKRExecutionAdapter(
                session=session,
                contract_factory=contract_factory,
                account=ibkr_config.profile.account_id,
            )
            logger.info("Swing execution adapter created (account=%s)", ibkr_config.profile.account_id)
        else:
            logger.warning("No IB session -- swing adapter is None (shadow/test mode)")
            ibkr_config = None
        active_account_id = str(
            getattr(getattr(ibkr_config, "profile", None), "account_id", "")
            or getattr(adapter, "_account", "")
            or (getattr(ctx, "contracts", {}) or {}).get("account_id", "")
        )

        # -- Market calendar -----------------------------------------------
        from libs.config.market_calendar import MarketCalendar
        calendar_factory = getattr(overrides, "calendar_factory", None)
        market_cal = calendar_factory() if calendar_factory is not None else MarketCalendar()

        # -- Equity & paper capital ----------------------------------------
        from libs.oms.persistence.db_config import get_environment
        paper_mode = get_environment() == "paper"

        _env = os.getenv("PAPER_INITIAL_EQUITY")
        _paper_account_seed = float(_env) if _env else ctx.portfolio.capital.paper_initial_equity

        if getattr(overrides, "equity_provider", None) is not None:
            account_equity = float(overrides.equity_provider())
        elif paper_mode:
            account_equity = _paper_account_seed
        else:
            # SWING-1: live equity from NetLiquidation. The previous literal
            # 100_000.0 mis-sized every swing trade by the ratio of real
            # account NAV to $100k. Using the shared helper makes this
            # consistent with momentum and stock (after EQUITY-1).
            from libs.services.equity import resolve_live_nlv
            account_id = ibkr_config.profile.account_id if ibkr_config else None
            account_equity = await resolve_live_nlv(session, account_id=account_id)

        account_allocs: dict[str, Any] | None = None
        if getattr(overrides, "equity_provider", None) is None:
            try:
                account_allocs = bootstrap_capital(
                    account_equity,
                    config_dir,
                    live=get_environment() == "live",
                )
            except Exception as exc:
                logger.warning(
                    "bootstrap_capital failed (%s), using configured swing family allocation fallback",
                    exc,
                )

        _seed_equity = _swing_family_nav(
            account_equity,
            account_allocs or {},
            tuple(strategy_ids),
            ctx,
        )
        equity = _seed_equity
        if (
            paper_mode
            and db_pool is not None
            and getattr(overrides, "equity_provider", None) is None
        ):
            from libs.persistence.paper_equity import PaperEquityManager
            _pem = PaperEquityManager(db_pool, account_scope=self.family_id, initial_equity=_seed_equity)
            equity = await _pem.load()
            logger.info("Paper mode equity for swing family: $%.2f", equity)
        paper_equity_offset: float = getattr(ctx, "paper_equity_offset", 0.0)

        # -- Capital allocation per strategy -------------------------------
        family_initial_nav = _seed_equity

        swing_nav = {
            sid: equity
            for sid in strategy_ids
        }
        allocation_targets = build_family_allocation_targets(
            self.family_id,
            strategy_ids,
            allocations=account_allocs or {},
            portfolio=getattr(ctx, "portfolio", None),
        )
        logger.info(
            "Capital allocation (swing family): %s",
            {k: f"${v:,.2f}" for k, v in swing_nav.items()},
        )

        # -- Compute unit risk dollars per strategy ------------------------
        urds: dict[str, float] = {}
        for sid in strategy_ids:
            params = _RISK_PARAMS.get(
                sid,
                {
                    "unit_risk_pct": 0.005,
                    "daily_stop_R": 2.0,
                    "priority": 99,
                    "max_heat_R": 4.0,
                    "max_working_orders": 3,
                },
            )
            urds[sid] = RiskCalculator.compute_unit_risk_dollars(
                nav=equity,
                unit_risk_pct=params["unit_risk_pct"],
            )
        capital_cfg = getattr(getattr(ctx, "portfolio", None), "capital", None)
        family_allocation_pct = float(
            (getattr(capital_cfg, "family_allocations", {}) or {}).get(self.family_id, 0.0)
        )
        await upsert_active_runtime_config(
            db_pool,
            ActiveRuntimeConfigRecord(
                account_id=active_account_id,
                config_scope="family",
                scope_id=self.family_id,
                runtime_env=get_environment(),
                payload=build_family_runtime_config(
                    account_id=active_account_id,
                    family_id=self.family_id,
                    family_allocation_pct=family_allocation_pct,
                    family_nav=equity,
                    family_heat_cap_R=_SWING_FAMILY_HEAT_CAP_R,
                    family_daily_stop_R=_SWING_FAMILY_DAILY_STOP_R,
                    family_weekly_stop_R=_SWING_FAMILY_WEEKLY_STOP_R,
                    active_strategy_ids=list(strategy_ids),
                ),
                expires_at=active_config_expiry(),
            ),
        )
        for sid in strategy_ids:
            params = _RISK_PARAMS.get(sid, {"unit_risk_pct": 0.005, "daily_stop_R": 2.0})
            await upsert_active_runtime_config(
                db_pool,
                ActiveRuntimeConfigRecord(
                    account_id=active_account_id,
                    config_scope="strategy",
                    scope_id=sid,
                    runtime_env=get_environment(),
                    payload=build_strategy_runtime_config(
                        account_id=active_account_id,
                        strategy_id=sid,
                        family_id=self.family_id,
                        enabled=True,
                        live=get_environment() == "live",
                        allocated_nav=swing_nav.get(sid, equity),
                        unit_risk_dollars=urds[sid],
                        max_heat_R=float(params.get("max_heat_R", 0.0)),
                        max_daily_loss_R=float(params.get("daily_stop_R", 2.0)),
                        max_weekly_loss_R=_SWING_FAMILY_WEEKLY_STOP_R,
                        risk_per_trade=float(params.get("unit_risk_pct", 0.005)),
                    ),
                    expires_at=active_config_expiry(),
                ),
            )

        # -- Build shared multi-strategy OMS -------------------------------
        # Use the runtime-provided account gate (shared across all families)
        # instead of creating a local one with min(URDs) which was ~12x too
        # restrictive for strategies with small URDs.
        account_gate = self._ctx.account_gate

        # Portfolio rules: directional cap + symbol collision for swing family
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        override_rules_provider = getattr(overrides, "portfolio_rules_provider", None)
        override_portfolio_rules = override_rules_provider() if override_rules_provider is not None else None
        portfolio_rules = override_portfolio_rules or PortfolioRulesConfig(
            directional_cap_R=_SWING_FAMILY_HEAT_CAP_R,
            directional_cap_long_R=4.0,
            directional_cap_short_R=4.0,
            initial_equity=family_initial_nav,
            family_strategy_ids=tuple(strategy_ids),
            symbol_collision_action="half_size",
            strategy_priorities=tuple((sid, _RISK_PARAMS.get(sid, {"priority": 99})["priority"]) for sid in strategy_ids),
            priority_headroom_R=0.75,
            priority_reserve_threshold=1,
            reference_unit_risk_dollars=urds[ATRSS_ID],
            dd_tiers=_DD_TIERS,
            nqdtc_direction_filter_enabled=False,
        )

        self._live_equity = [equity]
        strategy_lineage_manifests = {
            sid: {
                "family": self.family_id,
                "artifact_config": {"version": f"{sid}.unversioned"},
            }
            for sid in strategy_ids
        }
        strategy_parameter_sets = {
            sid: {
                "risk_params": dict(_RISK_PARAMS.get(sid, {})),
                "unit_risk_dollars": urds[sid],
                "allocation_target": (allocation_targets or {}).get("strategies", {}).get(sid),
            }
            for sid in strategy_ids
        }
        self._oms, self._coordinator = await build_multi_strategy_oms(
            adapter=adapter,
            strategies=[
                {
                    "id": sid,
                    "unit_risk_dollars": urds[sid],
                    "daily_stop_R": _RISK_PARAMS.get(sid, {"daily_stop_R": 2.0})["daily_stop_R"],
                    "priority": _RISK_PARAMS.get(sid, {"priority": 99})["priority"],
                    "max_heat_R": _RISK_PARAMS.get(sid, {"max_heat_R": 4.0})["max_heat_R"],
                    "max_working_orders": _RISK_PARAMS.get(sid, {"max_working_orders": 3})["max_working_orders"],
                }
                for sid in strategy_ids
            ],
            heat_cap_R=_SWING_FAMILY_HEAT_CAP_R,
            portfolio_daily_stop_R=_SWING_FAMILY_DAILY_STOP_R,
            portfolio_weekly_stop_R=_SWING_FAMILY_WEEKLY_STOP_R,
            db_pool=db_pool,
            market_calendar=market_cal,
            family_id=self.family_id,
            account_gate=account_gate,
            portfolio_rules_config=portfolio_rules,
            get_current_equity=lambda: self._live_equity[0],
            live_equity=self._live_equity,
            paper_equity_pool=db_pool if paper_mode else None,
            paper_equity_scope=self.family_id,
            paper_initial_equity=_seed_equity,
            allocation_targets=allocation_targets,
            strategy_manifests=strategy_lineage_manifests,
            parameter_sets=strategy_parameter_sets,
        )
        self._portfolio_checker = getattr(self._oms, '_portfolio_checker', None)
        self._base_portfolio_rules = portfolio_rules

        # SWING-3: action logger wiring is deferred until AFTER
        # _bootstrap_instrumentation_kits populates self._instrumentation_ctx.
        # The runtime constructs RuntimeContext with instrumentation=None, so
        # reading ctx.instrumentation here returned None and the wiring was
        # silently skipped — dropping ATRSS/Helix/TPC coordination evidence.
        instrumentation_ctx = getattr(ctx, "instrumentation", None)
        # Keep the placeholder write so any code path that reads
        # self._instrumentation_ctx before bootstrap doesn't AttributeError.
        self._instrumentation_ctx = instrumentation_ctx

        # -- Start OMS -----------------------------------------------------
        await self._oms.start()
        logger.info("Multi-strategy OMS started")

        # Wire post-reconnect reconciliation. Use add_reconnect_callback so we
        # don't clobber stock/momentum callbacks (CONN-1).
        if session is not None and hasattr(session, "add_reconnect_callback") and hasattr(self._oms, "_reconciler"):
            session.add_reconnect_callback(self._oms._reconciler.on_reconnect_reconciliation)
            logger.info("Swing post-reconnect reconciliation callback wired")

        # -- Bootstrap instrumentation kits --------------------------------
        _data_provider = None
        try:
            if getattr(overrides, "disable_instrumentation", False):
                raise RuntimeError("instrumentation disabled by runtime overrides")
            import asyncio as _asyncio
            from .instrumentation.src.ibkr_provider import IBKRHistoricalProvider
            _ib = getattr(session, "ib", None)
            _loop = _asyncio.get_running_loop()
            if _ib is not None:
                _data_provider = IBKRHistoricalProvider(
                    ib=_ib,
                    contract_factory=contract_factory,
                    loop=_loop,
                    historical_requester=getattr(session, "req_historical_data", None),
                )
                logger.info("IBKRHistoricalProvider created for post-exit backfill")
        except Exception:
            logger.debug("IBKRHistoricalProvider creation skipped", exc_info=True)

        if getattr(overrides, "disable_instrumentation", False):
            self._kits = {}
        else:
            self._kits = self._bootstrap_instrumentation_kits(
                strategy_ids,
                {
                    ATRSS_ID: ATRSS_CONFIGS,
                    HELIX_ID: HELIX_CONFIGS,
                    TPC_ID: TPC_CONFIGS,
                },
                data_provider=_data_provider,
            )
        if require_instrumentation:
            if not self._kits or self._instrumentation_ctx is None:
                raise RuntimeError("Required swing instrumentation failed to bootstrap")
            if not getattr(self._instrumentation_ctx, "_sidecar_forwarding_enabled", True):
                raise RuntimeError("Required swing instrumentation sidecar forwarding disabled")

        # SWING-3: now that _bootstrap_instrumentation_kits has populated
        # self._instrumentation_ctx, wire the coordinator action logger.
        # CoordinationLogger writes its events to the data_dir/coordination/
        # subdir; the swing sidecar maps that directory (see DIR-MAP fix in
        # this same phase) so they reach the relay as `coordinator_action`.
        if self._instrumentation_ctx is not None and getattr(
            self._instrumentation_ctx, "coordination_logger", None
        ):
            self._coordinator.set_action_logger(
                self._instrumentation_ctx.coordination_logger.log_action
            )
            logger.info("Swing coordinator action logger wired (post-bootstrap)")
        else:
            logger.warning(
                "Swing coordinator action logger NOT wired — coordination "
                "evidence will not flow. Check instrumentation bootstrap."
            )

        # -- Build instruments per strategy --------------------------------
        atrss_instruments = atrss_build_instruments()
        helix_instruments = helix_build_instruments()
        tpc_instruments = tpc_build_instruments()

        # -- Trade recorder (from bootstrap context) -----------------------
        trade_recorder = getattr(ctx, "trade_recorder", None)
        if trade_recorder is None and instrumentation_ctx is not None:
            trade_recorder = getattr(instrumentation_ctx, "trade_recorder", None)

        # -- Create strategy engines (all sharing single OMS) --------------
        oms = self._oms
        coordinator = self._coordinator

        atrss_engine = ATRSSEngine(
            ib_session=session,
            oms_service=oms,
            instruments=atrss_instruments,
            config=dict(ATRSS_CONFIGS),
            trade_recorder=trade_recorder,
            equity=equity,
            market_calendar=market_cal,
            kit=self._kits.get(ATRSS_ID),
            equity_offset=paper_equity_offset,
            equity_alloc_pct=1.0,
            disable_background_tasks=getattr(overrides, "disable_background_tasks", False),
        )

        helix_engine = HelixEngine(
            ib_session=session,
            oms_service=oms,
            instruments=helix_instruments,
            config=dict(HELIX_CONFIGS),
            trade_recorder=trade_recorder,
            equity=equity,
            coordinator=coordinator,  # enables ATRSS->Helix cross-strategy rules
            market_calendar=market_cal,
            instrumentation_kit=self._kits.get(HELIX_ID),
            equity_offset=paper_equity_offset,
            equity_alloc_pct=1.0,
            disable_background_tasks=getattr(overrides, "disable_background_tasks", False),
        )

        # STRAT-1: TPCEngine now has a live execution shell (15m scheduler +
        # action dispatcher + OMS event loop + state hydration). state_dir
        # defaults to data/tpc_state but can be overridden via TPC_STATE_DIR.
        tpc_state_dir = Path(
            (getattr(overrides, "state_dir_overrides", {}) or {}).get(TPC_ID)
            or os.environ.get("TPC_STATE_DIR")
            or (
                ctx.registry.strategies.get(TPC_ID).engine_config.get("state_dir")
                if ctx.registry and TPC_ID in ctx.registry.strategies
                and ctx.registry.strategies[TPC_ID].engine_config
                else None
            )
            or "data/tpc_state"
        )
        tpc_engine = TPCEngine(
            ib_session=session,
            oms_service=oms,
            instruments=tpc_instruments,
            config=dict(TPC_CONFIGS),
            trade_recorder=trade_recorder,
            equity=equity,
            market_calendar=market_cal,
            kit=self._kits.get(TPC_ID),
            equity_offset=paper_equity_offset,
            equity_alloc_pct=1.0,
            coordinator=coordinator,
            state_dir=tpc_state_dir,
            disable_scheduler=getattr(overrides, "disable_background_tasks", False),
        )

        # Store engines in priority order.
        self._engines = [
            (ATRSS_ID, atrss_engine),
            (HELIX_ID, helix_engine),
            (TPC_ID, tpc_engine),
        ]

        # -- Create OverlayEngine (idle-capital EMA crossover) -------------
        # Overlay operates on the swing family's total allocated NAV, not full account
        swing_family_nav = equity
        self._overlay_engine = self._create_overlay_engine(
            session=session,
            equity=swing_family_nav,
            market_cal=market_cal,
            paper_equity_offset=paper_equity_offset,
            equity_alloc_pct=1.0,
            disable_scheduler=getattr(overrides, "disable_background_tasks", False),
        )

        # -- Wire overlay state provider to all instrumentation kits -------
        if self._overlay_engine is not None:
            overlay_state_fn = self._overlay_engine.get_signals
            if instrumentation_ctx is not None:
                instrumentation_ctx.overlay_state_provider = overlay_state_fn
            for kit in self._kits.values():
                if kit is not None and hasattr(kit, "ctx"):
                    kit.ctx.overlay_state_provider = overlay_state_fn

        # -- Start engines in priority order --------------------------------
        for sid, engine in self._engines:
            await engine.start()
            logger.info("%s engine started (priority %d)", sid, _RISK_PARAMS[sid]["priority"])

        # -- Start overlay engine ------------------------------------------
        if self._overlay_engine is not None:
            await self._overlay_engine.start()
            logger.info("Overlay engine started")

        # -- Heartbeat background task --------------------------------------
        if not getattr(overrides, "disable_background_tasks", False):
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "Swing family coordinator active -- %d engines + overlay",
            len(self._engines),
        )

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """Graceful shutdown: overlay -> engines -> OMS."""
        # 0. Stop heartbeat loop
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        # 1. Stop overlay engine first (independent of OMS)
        if self._overlay_engine is not None:
            try:
                await self._overlay_engine.stop()
                logger.info("Overlay engine stopped")
            except Exception:
                logger.debug("Overlay engine stop failed", exc_info=True)

        # 2. Stop strategy engines in reverse priority.
        for sid, engine in reversed(self._engines):
            try:
                await engine.stop()
                logger.info("%s engine stopped", sid)
            except Exception:
                logger.warning("Failed to stop %s engine", sid, exc_info=True)
        logger.info("All strategy engines stopped")

        # 3. Stop instrumentation sidecar
        if self._instrumentation_ctx is not None:
            try:
                stop_async = getattr(self._instrumentation_ctx, "stop_async", None)
                if callable(stop_async):
                    await stop_async()
                else:
                    result = self._instrumentation_ctx.stop()
                    if result is not None and hasattr(result, "__await__"):
                        await result
                logger.info("Instrumentation stopped")
            except Exception:
                logger.debug("Instrumentation stop failed", exc_info=True)

        # 4. Stop OMS last (drain execution queue, flush pending state)
        if self._oms is not None:
            try:
                await self._oms.stop()
                logger.info("OMS stopped")
            except Exception:
                logger.warning("OMS stop failed", exc_info=True)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_status(self) -> dict[str, Any]:
        """Return health of all engines + OMS + overlay."""
        status: dict[str, Any] = {
            "family_id": self.family_id,
            "oms_running": getattr(self._oms, "_running", False) if self._oms else False,
            "overlay_running": (
                getattr(self._overlay_engine, "_running", False)
                if self._overlay_engine
                else False
            ),
            "engines": {},
        }
        for sid, engine in self._engines:
            if hasattr(engine, "health_status"):
                status["engines"][sid] = engine.health_status()
            else:
                status["engines"][sid] = {
                    "strategy_id": sid,
                    "running": getattr(engine, "_running", False),
                }
        return status

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to swing portfolio rules and overlay weights."""
        from regime.integration import build_swing_rules, OVERLAY_WEIGHTS

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx

        # Tier 1: portfolio rules
        new_rules = build_swing_rules(ctx, self._base_portfolio_rules)
        self._regime_adjusted_rules = new_rules  # Store for crisis overlay

        # Apply crisis overlay if active, including pre-action stress formation.
        if self._crisis_ctx is not None:
            from regime.crisis.integration import apply_crisis_overlay
            new_rules = apply_crisis_overlay(
                new_rules,
                self._crisis_ctx,
                self.family_id,
                regime=ctx.regime,
            )

        if self._portfolio_checker is not None:
            self._portfolio_checker.update_config(new_rules)

        # Overlay QQQ/GLD capital split from regime
        from regime.integration import _validated_regime
        regime = _validated_regime(ctx.regime)
        if self._overlay_engine is not None:
            self._overlay_engine._config.weights = dict(OVERLAY_WEIGHTS[regime])
            overlay_mult = 1.0
            if self._crisis_ctx is not None:
                from regime.crisis.actions import resolve_crisis_action

                action = resolve_crisis_action(
                    self._crisis_ctx,
                    self.family_id,
                    regime=ctx.regime,
                )
                overlay_mult = self._crisis_overlay_multiplier(action)
            self._apply_overlay_crisis_multiplier(overlay_mult)

        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Swing regime applied: %s%s (cap=%.1fR, risk=%.2fx, overlay=%s, overlay_max=%.3f)",
                    ctx.regime, changed,
                    self._portfolio_checker._cfg.directional_cap_R if self._portfolio_checker else 0,
                    self._portfolio_checker._cfg.regime_unit_risk_mult if self._portfolio_checker else 1,
                    OVERLAY_WEIGHTS[regime],
                    self._overlay_engine._config.max_equity_pct if self._overlay_engine else 0.0)

        self._emit_regime_event({
            "family": "swing",
            "regime": str(ctx.regime),
            "prev_regime": str(prev_regime) if prev_regime else None,
            "rules_applied": {
                "directional_cap_R": self._portfolio_checker._cfg.directional_cap_R if self._portfolio_checker else None,
                "regime_unit_risk_mult": self._portfolio_checker._cfg.regime_unit_risk_mult if self._portfolio_checker else None,
                "overlay_weights": dict(OVERLAY_WEIGHTS[regime]),
                "overlay_max_equity_pct": self._overlay_engine._config.max_equity_pct if self._overlay_engine else None,
            },
        })

    def apply_crisis(self, ctx) -> None:
        """Apply crisis context overlay on top of regime-adjusted rules.

        Always starts from _regime_adjusted_rules (not current checker config)
        to prevent compounding across calls.
        """
        from regime.crisis.actions import resolve_crisis_action
        from regime.crisis.integration import apply_crisis_overlay

        prev_level = getattr(self._crisis_ctx, "alert_level", "NORMAL") if self._crisis_ctx else "NORMAL"
        self._crisis_ctx = ctx

        if self._regime_adjusted_rules is None:
            logger.warning("apply_crisis called before apply_regime — skipping")
            return

        regime = getattr(self._regime_ctx, "regime", None)
        action = resolve_crisis_action(ctx, self.family_id, regime=regime)
        if action.is_no_action():
            # NORMAL or WATCH: revert to regime-only rules
            if self._portfolio_checker is not None:
                self._portfolio_checker.update_config(self._regime_adjusted_rules)
            self._apply_overlay_crisis_multiplier(1.0)
            self._refresh_instrumentation_lineage()
            if prev_level not in ("NORMAL", "WATCH"):
                logger.info("Swing crisis overlay removed (level=%s)", ctx.alert_level)
                current_rules = self._current_portfolio_rules_config()
                overlay_max = (
                    self._overlay_engine._config.max_equity_pct
                    if self._overlay_engine is not None else None
                )
                self._emit_crisis_event({
                    "family": "swing",
                    "alert_level": ctx.alert_level,
                    "prev_level": prev_level,
                    "crisis_action": "removed",
                    "overlay_exposure_multiplier": 1.0,
                    "overlay_max_equity_pct": overlay_max,
                    "rules_applied": {
                        "directional_cap_R": getattr(current_rules, "directional_cap_R", None),
                        "regime_unit_risk_mult": getattr(current_rules, "regime_unit_risk_mult", None),
                        "disabled_strategies": list(getattr(current_rules, "disabled_strategies", ()) or ()),
                    },
                })
            return

        tightened = apply_crisis_overlay(
            self._regime_adjusted_rules,
            ctx,
            self.family_id,
            regime=regime,
        )
        if self._portfolio_checker is not None:
            self._portfolio_checker.update_config(tightened)
        overlay_max = self._apply_overlay_crisis_multiplier(
            self._crisis_overlay_multiplier(action),
        )

        changed = f" (was {prev_level})" if prev_level != ctx.alert_level else ""
        logger.info(
            "Swing crisis applied: %s%s (risk_mult=%.2f, dd_mult=%.2f, "
            "overlay_mult=%.2f, overlay_max=%s, provenance=%s, dominant=%s)",
            ctx.alert_level, changed, action.risk_multiplier,
            action.dd_tier_multiplier, self._crisis_overlay_multiplier(action),
            f"{overlay_max:.3f}" if overlay_max is not None else "n/a",
            action.action_provenance, ctx.dominant_channel,
        )

        self._emit_crisis_event({
            "family": "swing",
            "alert_level": ctx.alert_level,
            "prev_level": prev_level,
            "risk_multiplier": ctx.risk_multiplier,
            "dd_tier_multiplier": ctx.dd_tier_multiplier,
            "overlay_exposure_multiplier": self._crisis_overlay_multiplier(action),
            "overlay_max_equity_pct": overlay_max,
            "dominant_channel": ctx.dominant_channel,
            "vix_level": ctx.vix_level,
            "credit_spread_bps": ctx.credit_spread_bps,
            "action_policy": action.to_dict(),
        })

    def _current_portfolio_rules_config(self):
        if self._portfolio_checker is not None:
            return getattr(self._portfolio_checker, "_cfg", None)
        return self._regime_adjusted_rules or self._base_portfolio_rules

    def _refresh_instrumentation_lineage(self, rules_config=None) -> None:
        rules_config = rules_config or self._current_portfolio_rules_config()
        contexts: list[Any] = []
        if self._instrumentation_ctx is not None:
            contexts.append(self._instrumentation_ctx)
        for kit in self._kits.values():
            ctx = getattr(kit, "ctx", None) or getattr(kit, "shared_ctx", None)
            if ctx is not None and all(id(ctx) != id(existing) for existing in contexts):
                contexts.append(ctx)
        for ctx in contexts:
            try:
                refresh = getattr(ctx, "refresh_lineage", None)
                if callable(refresh):
                    refresh(rules_config)
            except Exception:
                logger.debug("Failed to refresh swing instrumentation lineage", exc_info=True)

    def _write_coordination_event(self, action_type: str, payload: dict) -> None:
        from libs.instrumentation.event_contract import append_jsonl_event, enrich_payload
        from libs.instrumentation.lineage import compute_risk_config_version, redact_config

        ctx = getattr(self, "_instrumentation_ctx", None)
        if ctx is None:
            return
        data_dir = getattr(ctx, "data_dir", None)
        if not data_dir:
            return
        now = datetime.now(timezone.utc)
        rules_config = self._current_portfolio_rules_config()
        before_rules = getattr(self, "_last_coordination_rules_config", None) or self._base_portfolio_rules
        before_version = (
            compute_risk_config_version({}, before_rules, {}) if before_rules is not None else ""
        )
        after_version = (
            compute_risk_config_version({}, rules_config, {}) if rules_config is not None else ""
        )
        self._refresh_instrumentation_lineage(rules_config)
        record = {
            "timestamp": now.isoformat(),
            "action_type": action_type,
            "portfolio_rule_config_version_before": before_version,
            "portfolio_rule_config_version_after": after_version,
            "risk_config_version_before": before_version,
            "risk_config_version_after": after_version,
            "effective_config_evidence": {
                "portfolio_rules_config_before": redact_config(before_rules) if before_rules is not None else {},
                "portfolio_rules_config_after": redact_config(rules_config) if rules_config is not None else {},
            },
            **payload,
        }
        try:
            event = enrich_payload(
                record,
                lineage=getattr(ctx, "lineage", None),
                event_type="coordinator_action",
                scope="family",
            )
            append_jsonl_event(data_dir, "coordination_events", "coordination_events", event)
        except Exception:
            logger.debug("Failed to emit swing coordination event", exc_info=True)
        self._last_coordination_rules_config = rules_config

    def _emit_crisis_event(self, payload: dict) -> None:
        """Write an enriched crisis event to the shared data_dir."""
        self._write_coordination_event("crisis_alert_change", payload)

    def _emit_regime_event(self, payload: dict) -> None:
        """Write an enriched regime/rules event to the shared data_dir."""
        self._write_coordination_event("regime_rules_change", payload)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        heartbeat = getattr(self._ctx, "heartbeat", None)
        if heartbeat is None:
            return
        session = self._ctx.session
        # HB-1: self-heal — a transient exception in payload construction or
        # emission must not kill the loop. The previous shape only caught
        # CancelledError around the sleep, so any error in the body
        # silently terminated the heartbeat task and the watchdog only
        # detected it after stale_threshold_sec elapsed.
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                return
            try:
                await self._heartbeat_iteration(heartbeat, session)
            except Exception:
                logger.exception("Swing heartbeat iteration failed; continuing")

    async def _heartbeat_iteration(self, heartbeat, session) -> None:
        rs = getattr(self._oms, "_portfolio_risk_state", None)
        srs = getattr(self._oms, "_strategy_risk_states", {})
        payloads = []
        for sid, _engine in self._engines:
            sr = srs.get(sid)
            payload = {
                "strategy_id": sid,
                "heat_r": Decimal(str(sr.open_risk_R)) if sr else Decimal("0"),
                "daily_pnl_r": Decimal(str(sr.daily_realized_R)) if sr else Decimal("0"),
                "mode": "HALTED" if (rs and rs.halted) else "RUNNING",
            }
            # Diagnostic pulse fields
            if hasattr(_engine, "_last_decision_code"):
                payload["last_decision_code"] = _engine._last_decision_code
                payload["last_decision_details"] = getattr(_engine, "_last_decision_details", None)
                payload["last_seen_bar_ts"] = getattr(_engine, "_last_bar_ts", None)
            if hasattr(_engine, "liveness_payload") or hasattr(self._oms, "_intents_submitted"):
                details = payload.get("last_decision_details") or {}
                if hasattr(_engine, "liveness_payload"):
                    details["liveness"] = _engine.liveness_payload()
                if hasattr(self._oms, "_intents_submitted"):
                    details["oms_health"] = {
                        "submitted": self._oms._intents_submitted,
                        "accepted": self._oms._intents_accepted,
                        "denied": self._oms._intents_denied,
                        "consecutive_denials": self._oms._consecutive_denials,
                    }
                payload["last_decision_details"] = details
            payloads.append(payload)
        connected = session.ib.isConnected() if session else False
        # Enrich with IB farm status for diagnostic context
        if session:
            farm_statuses = {}
            for group in session.groups.values():
                if group.farm_monitor:
                    farm_statuses.update(group.farm_monitor.all_statuses())
            if farm_statuses:
                for p in payloads:
                    details = p.get("last_decision_details") or {}
                    if isinstance(details, dict):
                        details["ib_farm_status"] = farm_statuses
                        p["last_decision_details"] = details
        await emit_family_heartbeats(
            heartbeat, self.family_id, payloads, adapter_connected=connected,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_instrumentation_kits(
        self,
        strategy_ids: list[str],
        config_maps: dict[str, dict],
        data_provider=None,
    ) -> dict[str, Any]:
        """Create per-strategy InstrumentationKits (graceful degradation)."""
        kits: dict[str, Any] = {}
        try:
            from .instrumentation.src.bootstrap import (
                bootstrap_instrumentation,
                bootstrap_kit,
            )

            all_symbols = sorted({
                sym for configs in config_maps.values() for sym in configs
            })
            self._instrumentation_ctx = bootstrap_instrumentation(
                symbols=all_symbols, data_provider=data_provider,
                get_regime_ctx=lambda: self._regime_ctx,
                get_applied_config=lambda: self._portfolio_checker._cfg if self._portfolio_checker else None,
                pg_store=getattr(self._ctx, "pg_store", None),
            )
            self._instrumentation_ctx.oms = self._oms
            if self._instrumentation_ctx.pg_store is None and getattr(self._ctx, "db_pool", None) is not None:
                from libs.oms.persistence.postgres import PgStore

                self._instrumentation_ctx.pg_store = PgStore(self._ctx.db_pool)
            logger.info("Instrumentation bootstrapped for %s", all_symbols)

            for sid in strategy_ids:
                kits[sid] = bootstrap_kit(
                    strategy_id=sid,
                    shared_ctx=self._instrumentation_ctx,
                )
                logger.info("%s InstrumentationKit bootstrapped", sid)

            # Register OVERLAY kit so overlay engine gets instrumentation
            kits["OVERLAY"] = bootstrap_kit(
                strategy_id="OVERLAY",
                shared_ctx=self._instrumentation_ctx,
            )
            logger.info("OVERLAY InstrumentationKit bootstrapped")

            # Start instrumentation sidecar (background thread)
            try:
                self._instrumentation_ctx.start()
            except Exception:
                logger.debug("Instrumentation sidecar start failed", exc_info=True)

        except ImportError:
            logger.info("Instrumentation not available — running without kits")
        except Exception:
            logger.warning("Instrumentation bootstrap failed", exc_info=True)

        return kits

    def _get_swing_deployed_capital(self) -> float:
        """Return estimated notional capital deployed by swing strategies.

        Uses per-strategy risk states and their actual unit_risk_pct to
        convert open_risk_dollars into notional:
            notional ≈ risk_dollars / unit_risk_pct
        This is far more accurate than a fixed 3% divisor because strategies
        vary by strategy.
        """
        if not hasattr(self, "_oms") or self._oms is None:
            return 0.0
        try:
            srs = getattr(self._oms, "_strategy_risk_states", {})
            if not srs:
                return 0.0
            total_notional = 0.0
            for sid, sr in srs.items():
                if sr is None:
                    continue
                risk_dollars = getattr(sr, "open_risk_dollars", 0.0)
                if risk_dollars <= 0:
                    continue
                risk_pct = _RISK_PARAMS.get(sid, {}).get("unit_risk_pct", 0.02)
                total_notional += risk_dollars / risk_pct
            return total_notional
        except Exception:
            return 0.0

    def _create_overlay_engine(
        self,
        session: Any,
        equity: float,
        market_cal: Any,
        paper_equity_offset: float,
        equity_alloc_pct: float = 1.0,
        disable_scheduler: bool = False,
    ) -> Any | None:
        """Create OverlayEngine for idle-capital EMA crossover (QQQ, GLD)."""
        try:
            from strategies.swing.overlay.config import OverlayConfig
            from strategies.swing.overlay.engine import OverlayEngine

            overlay_config = OverlayConfig()
            overrides = getattr(self._ctx, "runtime_overrides", None)
            rebalance_provider = getattr(overrides, "overlay_rebalance_provider", None)
            if rebalance_provider is not None:
                payload = dict(rebalance_provider() or {})
                target_weights = dict(payload.get("target_weights", {}) or {})
                symbols = list(payload.get("symbols", []) or target_weights.keys())
                if symbols:
                    overlay_config.symbols = [str(symbol) for symbol in symbols]
                if target_weights:
                    overlay_config.weights = {str(symbol): float(weight) for symbol, weight in target_weights.items()}
                ema_overrides = payload.get("ema_overrides", {}) or {}
                if ema_overrides:
                    overlay_config.ema_overrides = {
                        str(symbol): (int(periods[0]), int(periods[1]))
                        for symbol, periods in ema_overrides.items()
                        if len(periods) >= 2
                    }
                overlay_config.max_equity_pct = float(payload.get("max_equity_pct", overlay_config.max_equity_pct))
                state_dir = (getattr(overrides, "state_dir_overrides", {}) or {}).get("OVERLAY")
                if state_dir is not None:
                    Path(state_dir).mkdir(parents=True, exist_ok=True)
                    overlay_config.state_file = str(Path(state_dir) / "overlay_state.json")
                overlay_config.enabled = True
            self._base_overlay_max_equity_pct = overlay_config.max_equity_pct

            if not overlay_config.enabled:
                return None

            engine = OverlayEngine(
                ib_session=session,
                equity=equity,
                config=overlay_config,
                market_calendar=market_cal,
                equity_offset=paper_equity_offset,
                get_deployed_capital=self._get_swing_deployed_capital,
                instrumentation=self._kits.get("OVERLAY"),
                equity_alloc_pct=equity_alloc_pct,
                disable_scheduler=disable_scheduler,
            )
            return engine

        except ImportError:
            logger.info("OverlayEngine not available — overlay disabled")
            return None
        except Exception:
            logger.warning("OverlayEngine creation failed", exc_info=True)
            return None

    async def run_overlay_rebalance_once(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Drive one deterministic overlay rebalance from a fixture payload.

        Production overlay scheduling and IB order placement remain unchanged.
        Parity overrides use this hook to exercise coordinator-owned overlay
        state without live market data or broker access.
        """

        overrides = getattr(self._ctx, "runtime_overrides", None)
        if payload is None:
            provider = getattr(overrides, "overlay_rebalance_provider", None)
            payload = dict(provider() or {}) if provider is not None else {}
        if self._overlay_engine is None or not payload:
            return {}

        engine = self._overlay_engine
        symbols = [str(symbol) for symbol in payload.get("symbols", []) or engine._config.symbols]
        target_weights = {
            str(symbol): float(weight)
            for symbol, weight in (payload.get("target_weights", {}) or {}).items()
        }
        starting = {
            str(symbol): int(qty)
            for symbol, qty in (payload.get("starting_holdings", {}) or {}).items()
        }
        if starting:
            engine._shares.update(starting)

        engine._config.symbols = symbols
        engine._config.weights = target_weights or engine._config.weights
        engine._config.max_equity_pct = float(payload.get("max_equity_pct", engine._config.max_equity_pct))
        equity = float(payload.get("equity", getattr(engine, "_equity", 0.0)) or 0.0)
        engine._equity = equity
        plan = engine.build_rebalance_plan_from_bars(
            payload.get("daily_bars", {}) or {},
            equity=equity,
            min_bars=0,
        )
        return engine.apply_rebalance_plan_dry_run(
            plan,
            timestamp=payload.get("timestamp", ""),
            reason=str(payload.get("rebalance_reason", "fixture")),
        )

    def _apply_overlay_crisis_multiplier(self, multiplier: float) -> float | None:
        """Apply crisis as a total overlay exposure throttle.

        HMM regime still chooses the QQQ/GLD mix. The crisis layer only scales
        the total idle-capital overlay so it does not fight the regime rotation.
        """
        if self._overlay_engine is None or self._base_overlay_max_equity_pct is None:
            return None
        mult = max(0.0, min(float(multiplier), 1.0))
        max_pct = round(self._base_overlay_max_equity_pct * mult, 6)
        self._overlay_engine._config.max_equity_pct = max_pct
        return max_pct

    @staticmethod
    def _crisis_overlay_multiplier(action: Any) -> float:
        overlay_mult = getattr(action, "overlay_exposure_multiplier", 1.0)
        risk_mult = getattr(action, "risk_multiplier", 1.0)
        return min(float(overlay_mult), float(risk_mult))
