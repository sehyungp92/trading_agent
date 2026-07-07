"""Stock family coordinator — each enabled stock strategy gets its own OMS instance.

Like momentum (per-strategy OMS), unlike swing (shared OMS).
Supported stock strategies:
  - IARIC_v1   (IARICEngine)   — WatchlistArtifact
  - ALCB_v1    (ALCBT2Engine)  — CandidateArtifact

Stock-specific differences from momentum:
  - portfolio_rules with family-scoped directional cap + symbol collision guard
  - Paper equity NAV tracking via resolve_paper_nav / capital_bootstrap
  - Artifacts or cache dicts instead of instrument dicts
  - IBMarketDataSource per engine, wired via on_bar/on_quote callbacks
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
from typing import TYPE_CHECKING, Any

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
from strategies.stock.readiness import validate_stock_readiness

if TYPE_CHECKING:
    from regime.context import RegimeContext

logger = logging.getLogger(__name__)

_STOCK_SYMBOL_COLLISION_PAIRS: tuple[tuple[str, str, str], ...] = ()
_STOCK_STRATEGY_PRIORITIES: tuple[tuple[str, int], ...] = (
    ("IARIC_v1", 0),
    ("ALCB_v1", 1),
)
_STOCK_DIRECTIONAL_CAP_R = 6.5
_STOCK_DIRECTIONAL_LONG_CAP_R = 6.25
_STOCK_PRIORITY_HEADROOM_R = 1.15
_STOCK_REFERENCE_RISK_PCT = 0.00648
_STOCK_MAX_TOTAL_ACTIVE_POSITIONS = 12
_STOCK_MAX_SYMBOL_HEAT_R = 2.2
_STOCK_SAME_SECTOR_HEAT_CAP_R = 3.8
_STOCK_MAX_SINGLE_STRATEGY_TRADE_SHARE = 0.85
_STOCK_DYNAMIC_MIN_MULT = 0.65
_STOCK_DYNAMIC_MAX_MULT = 1.22
_STOCK_DYNAMIC_POSITIVE_BOOST = 0.10
_STOCK_DYNAMIC_NEGATIVE_CUT = 0.18
_STOCK_DYNAMIC_LOOKBACK_TRADES = 60
_STOCK_PORTFOLIO_WEEKLY_STOP_R = 8.0
_STOCK_DD_TIERS = (
    (0.04, 1.00),
    (0.07, 0.75),
    (0.10, 0.40),
    (0.13, 0.00),
)


def _stock_family_nav(
    base_equity: float,
    allocs: dict[str, Any],
    strategy_ids: tuple[str, ...],
    ctx: RuntimeContext,
) -> float:
    """Return the stock-family NAV used by optimized stock portfolio sizing."""
    for strategy_id in strategy_ids:
        alloc = allocs.get(strategy_id)
        family_fraction = getattr(alloc, "family_fraction", None)
        if _positive_finite(family_fraction):
            return float(base_equity) * float(family_fraction)

    capital = getattr(getattr(ctx, "portfolio", None), "capital", None)
    family_allocations = getattr(capital, "family_allocations", {}) or {}
    family_fraction = family_allocations.get("stock")
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


def _override_portfolio_rules(overrides: Any) -> Any | None:
    provider = getattr(overrides, "portfolio_rules_provider", None)
    if provider is None:
        return None
    return provider()


class StockFamilyCoordinator:
    """Lifecycle manager for the configured stock strategies.

    Each strategy receives its own OMS built via ``build_oms_service``.
    Shared across the stock family: *db_pool*, *AccountRiskGate*, *family_id*.
    """

    family_id = "stock"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list = []
        self._oms_services: list = []
        self._instrumentations: list = []
        self._strategy_ids: list[str] = []
        self._market_data_sources: list = []
        self._market_data_task: asyncio.Task | None = None
        self._contract_factory = None
        self._portfolio_checkers: list = []
        self._engine_map: dict[str, Any] = {}
        self._base_portfolio_rules: Any = None
        self._regime_ctx: Any = None
        self._regime_adjusted_rules: Any = None  # stored after apply_regime for crisis overlay
        self._regime_stock_profile: dict | None = None  # stored Tier 2 profile
        self._crisis_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None
        self._shared_sidecar: Any = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Import, build OMS, and start all enabled stock engines."""
        from libs.oms.services.factory import build_oms_service as default_build_oms_service
        from libs.oms.risk.calculator import RiskCalculator
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        from libs.config.capital_bootstrap import bootstrap_capital
        from .live_universe import LIVE_STOCK_UNIVERSE

        overrides = getattr(self._ctx, "runtime_overrides", None)
        build_oms_service = (
            getattr(overrides, "build_oms_service", None) or default_build_oms_service
        )
        active_strategy_ids = self._enabled_stock_strategy_ids()
        if not active_strategy_ids:
            logger.warning("No enabled stock strategies remain after registry filtering")
            return

        artifact_provider = getattr(overrides, "stock_artifact_provider", None)
        if getattr(overrides, "disable_market_data", False) and artifact_provider is not None:
            artifacts = dict(artifact_provider() or {})
            readiness_failures = []
        else:
            artifacts, readiness_failures = validate_stock_readiness(
                self._ctx.registry,
                live=get_environment() == "live",
                strategy_ids=active_strategy_ids,
            )
        if readiness_failures:
            detail = "; ".join(
                f"{failure.check_name}={failure.detail}" for failure in readiness_failures
            )
            raise RuntimeError(f"Stock family readiness failed: {detail}")

        ctx = self._ctx
        session = ctx.session
        db_pool = ctx.db_pool
        account_gate = ctx.account_gate
        require_instrumentation = bool(getattr(ctx, "require_instrumentation", False))

        config_dir = Path(
            os.environ.get(
                "CONFIG_DIR",
                str(Path(__file__).resolve().parent.parent.parent / "config"),
            )
        )

        # Resolve paper-mode equity
        runtime_env = get_environment()
        paper_mode = runtime_env == "paper"
        strict_market_data = runtime_env in {"paper", "live"}
        _env = os.getenv("PAPER_INITIAL_EQUITY")
        _paper_seed = (
            float(_env) if _env else ctx.portfolio.capital.paper_initial_equity
        )

        if getattr(overrides, "equity_provider", None) is not None:
            base_equity = float(overrides.equity_provider())
        elif paper_mode:
            base_equity = _paper_seed
        else:
            # EQUITY-1: hard-fail in live mode if NetLiquidation can't be
            # resolved, using the configured IBKR profile account. The helper polls
            # accountValues() up to its timeout to handle the async populate
            # race between ib.connectAsync and the first accountValues read.
            from libs.services.equity import resolve_live_nlv
            from libs.broker_ibkr.config.loader import IBKRConfig

            ibkr_config = IBKRConfig(config_dir)
            if not ibkr_config.profile.account_id:
                raise RuntimeError("Stock live equity requires configured IBKR account_id")
            base_equity = await resolve_live_nlv(
                session,
                account_id=ibkr_config.profile.account_id,
            )

        # Capital allocation per strategy. Runtime parity/diagnostic overrides
        # provide an explicit equity source and family allocation in the context;
        # avoid loading the user's configured bootstrap file in that offline path.
        if getattr(overrides, "equity_provider", None) is not None:
            allocs = {}
        else:
            try:
                allocs = bootstrap_capital(
                    base_equity,
                    config_dir,
                    live=get_environment() == "live",
                )
            except Exception as exc:
                logger.warning("bootstrap_capital failed (%s), using equal split", exc)
                allocs = {}

        # ── Strategy descriptors ─────────────────────────────────────
        _strategies = self._build_strategy_descriptors(artifacts, active_strategy_ids)
        if not _strategies:
            logger.warning("No enabled stock strategies remain after registry filtering")
            return
        active_account_id = str(
            next(
                (
                    desc.get("account_id", "")
                    for desc in _strategies
                    if desc.get("account_id")
                ),
                "",
            )
        )

        family_initial_nav = _stock_family_nav(base_equity, allocs, active_strategy_ids, ctx)
        family_current_nav = family_initial_nav
        if paper_mode and getattr(overrides, "equity_provider", None) is None:
            from libs.persistence.paper_equity import PaperEquityManager

            pem = PaperEquityManager(
                db_pool,
                account_scope=self.family_id,
                initial_equity=family_initial_nav,
            )
            family_current_nav = await pem.load()
            logger.info(
                "Paper mode equity for stock portfolio: scope=%s current=$%.2f initial=$%.2f",
                self.family_id,
                family_current_nav,
                family_initial_nav,
            )

        stock_equity_ref = [family_current_nav]
        reference_unit_risk = RiskCalculator.compute_unit_risk_dollars(
            nav=family_current_nav,
            unit_risk_pct=_STOCK_REFERENCE_RISK_PCT,
        )
        logger.info(
            "Stock portfolio sizing base: current_nav=$%.2f initial_nav=$%.2f "
            "account_nav=$%.2f reference_1R=$%.2f",
            family_current_nav,
            family_initial_nav,
            base_equity,
            reference_unit_risk,
        )

        # Portfolio rules: drawdown tiers + family-scoped directional cap + symbol collision
        rule_inputs = self._portfolio_rule_inputs(
            tuple(d["strategy_id"] for d in _strategies)
        )
        override_portfolio_rules = _override_portfolio_rules(overrides)
        all_strategy_ids = rule_inputs["family_strategy_ids"]
        allocation_targets = build_family_allocation_targets(
            self.family_id,
            all_strategy_ids,
            allocations=allocs,
            portfolio=getattr(ctx, "portfolio", None),
        )
        capital_cfg = getattr(getattr(ctx, "portfolio", None), "capital", None)
        family_allocation_pct = float(
            (getattr(capital_cfg, "family_allocations", {}) or {}).get(self.family_id, 0.0)
        )
        family_daily_stop_R = min(
            (
                float(desc["portfolio_daily_stop_R"])
                for desc in _strategies
                if desc.get("portfolio_daily_stop_R") is not None
            ),
            default=_STOCK_DIRECTIONAL_CAP_R,
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
                    family_nav=family_current_nav,
                    family_heat_cap_R=_STOCK_DIRECTIONAL_CAP_R,
                    family_daily_stop_R=family_daily_stop_R,
                    family_weekly_stop_R=_STOCK_PORTFOLIO_WEEKLY_STOP_R,
                    active_strategy_ids=list(all_strategy_ids),
                ),
                expires_at=active_config_expiry(),
            ),
        )
        collision_pairs = rule_inputs["symbol_collision_pairs"]
        strategy_priorities = rule_inputs["strategy_priorities"]
        symbol_sector_map = tuple(
            (symbol, sector) for symbol, sector, _primary in LIVE_STOCK_UNIVERSE
        )
        logger.info(
            "Stock portfolio rules: heat=%.1fR, long=%.1fR, max_active=%d, "
            "symbol_heat=%.1fR, sector_heat=%.1fR, trade_share=%.0f%%, "
            "dynamic=[%.2f, %.2f], collision=%s, "
            "collision_pairs=%s, headroom=%.1fR, strategies=%s",
            _STOCK_DIRECTIONAL_CAP_R,
            _STOCK_DIRECTIONAL_LONG_CAP_R,
            _STOCK_MAX_TOTAL_ACTIVE_POSITIONS,
            _STOCK_MAX_SYMBOL_HEAT_R,
            _STOCK_SAME_SECTOR_HEAT_CAP_R,
            _STOCK_MAX_SINGLE_STRATEGY_TRADE_SHARE * 100,
            _STOCK_DYNAMIC_MIN_MULT,
            _STOCK_DYNAMIC_MAX_MULT,
            "half_size",
            [f"{h}->{r}:{a}" for h, r, a in collision_pairs],
            _STOCK_PRIORITY_HEADROOM_R, all_strategy_ids,
        )

        # Log dollar-equivalent directional cap per strategy for monitoring
        for desc in _strategies:
            _unit = RiskCalculator.compute_unit_risk_dollars(
                nav=family_current_nav,
                unit_risk_pct=desc["base_risk_pct"],
            )
            logger.info(
                "Directional cap dollar-equiv for %s: 1R=$%.0f, cap=%.2fR=$%.0f, heat=%sR=$%.0f",
                desc["strategy_id"], _unit, _STOCK_DIRECTIONAL_CAP_R, _STOCK_DIRECTIONAL_CAP_R * _unit,
                desc["heat_cap_R"], desc["heat_cap_R"] * _unit,
            )

        self._shared_sidecar = None
        authoritative_strategy_id = all_strategy_ids[0] if all_strategy_ids else ""
        for desc in _strategies:
            sid = desc["strategy_id"]
            reconciliation_authoritative = sid == authoritative_strategy_id

            if desc.get("data_key") == "artifact" and desc.get("data_value") is None:
                raise RuntimeError(
                    f"Stock artifact missing after readiness validation for {sid}"
                )

            self._strategy_ids.append(sid)

            # Optimized stock portfolio sizing is based on the stock-family NAV.
            alloc = allocs.get(sid)
            initial_nav = family_initial_nav
            allocated_nav = stock_equity_ref[0]

            # Per-strategy OMS instances share the same stock portfolio equity
            # basis so optimized unit_risk_pct values are not multiplied by a
            # legacy per-strategy capital split.
            portfolio_rules = override_portfolio_rules or PortfolioRulesConfig(
                    directional_cap_R=_STOCK_DIRECTIONAL_CAP_R,
                    directional_cap_long_R=_STOCK_DIRECTIONAL_LONG_CAP_R,
                    max_total_active_positions=_STOCK_MAX_TOTAL_ACTIVE_POSITIONS,
                    max_symbol_heat_R=_STOCK_MAX_SYMBOL_HEAT_R,
                    same_sector_heat_cap_R=_STOCK_SAME_SECTOR_HEAT_CAP_R,
                    symbol_sector_map=symbol_sector_map,
                    max_single_strategy_trade_share=_STOCK_MAX_SINGLE_STRATEGY_TRADE_SHARE,
                    dynamic_allocation_enabled=True,
                    dynamic_lookback_trades=_STOCK_DYNAMIC_LOOKBACK_TRADES,
                    dynamic_min_mult=_STOCK_DYNAMIC_MIN_MULT,
                    dynamic_max_mult=_STOCK_DYNAMIC_MAX_MULT,
                    dynamic_positive_expectancy_boost=_STOCK_DYNAMIC_POSITIVE_BOOST,
                    dynamic_negative_expectancy_cut=_STOCK_DYNAMIC_NEGATIVE_CUT,
                    initial_equity=initial_nav,
                    family_strategy_ids=all_strategy_ids,
                    symbol_collision_action="half_size",
                    symbol_collision_pairs=collision_pairs,
                    strategy_priorities=strategy_priorities,
                    priority_headroom_R=_STOCK_PRIORITY_HEADROOM_R,
                    priority_reserve_threshold=1,  # priority 0-1 can use reserved headroom
                    reference_unit_risk_dollars=reference_unit_risk,
                    reference_unit_risk_pct=_STOCK_REFERENCE_RISK_PCT,
                    dd_tiers=_STOCK_DD_TIERS,
                    portfolio_heat_cap_R=_STOCK_DIRECTIONAL_CAP_R,
                    max_strategy_active_positions=tuple(
                        (
                            item["strategy_id"],
                            int(item.get("max_concurrent", 0) or 0),
                        )
                        for item in _strategies
                        if item["strategy_id"] in all_strategy_ids
                    ),
                    max_strategy_heat_R=tuple(
                        (
                            item["strategy_id"],
                            float(item.get("heat_cap_R", 0.0) or 0.0),
                        )
                        for item in _strategies
                        if item["strategy_id"] in all_strategy_ids
                    ),
                )
            # Save first portfolio_rules as base template for regime updates
            if self._base_portfolio_rules is None:
                self._base_portfolio_rules = portfolio_rules

            if alloc:
                logger.info(
                    "Capital allocation: %s -> stock portfolio NAV $%.2f "
                    "(config allocation %.1f%% kept for non-stock allocators only)",
                    sid, allocated_nav, getattr(alloc, "capital_pct", 0.0),
                )
            else:
                logger.warning(
                    "Strategy %s not in unified config, using stock portfolio NAV %.2f",
                    sid, allocated_nav,
                )

            # Risk parameters
            unit_risk = RiskCalculator.compute_unit_risk_dollars(
                nav=allocated_nav, unit_risk_pct=desc["base_risk_pct"],
            )
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
                        allocated_nav=allocated_nav,
                        unit_risk_dollars=unit_risk,
                        max_heat_R=float(desc["heat_cap_R"]),
                        max_daily_loss_R=float(desc["daily_stop_R"]),
                        max_weekly_loss_R=_STOCK_PORTFOLIO_WEEKLY_STOP_R,
                        risk_per_trade=float(desc["base_risk_pct"]),
                    ),
                    expires_at=active_config_expiry(),
                ),
            )

            # Build per-strategy OMS with portfolio rules
            oms = await build_oms_service(
                adapter=desc["adapter"](session),
                strategy_id=sid,
                unit_risk_dollars=unit_risk,
                portfolio_unit_risk_dollars=reference_unit_risk,
                daily_stop_R=desc["daily_stop_R"],
                strategy_heat_cap_R=desc["heat_cap_R"],
                heat_cap_R=_STOCK_DIRECTIONAL_CAP_R,
                portfolio_daily_stop_R=desc["portfolio_daily_stop_R"],
                portfolio_weekly_stop_R=_STOCK_PORTFOLIO_WEEKLY_STOP_R,
                db_pool=db_pool,
                portfolio_rules_config=portfolio_rules,
                get_current_equity=lambda eq=stock_equity_ref: eq[0],
                paper_equity_pool=db_pool if paper_mode else None,
                paper_equity_scope=self.family_id,
                paper_initial_equity=initial_nav,
                paper_equity_ref=stock_equity_ref if paper_mode else None,
                live_equity=stock_equity_ref if not paper_mode else None,
                family_id=self.family_id,
                account_gate=account_gate,
                family_strategy_ids=list(all_strategy_ids),
                allocation_targets=allocation_targets,
                reconciliation_authoritative=reconciliation_authoritative,
                reconciliation_owner_id=f"{self.family_id}:{sid}",
            )
            await oms.start()
            logger.info(
                "OMS started for %s (reconciliation_authoritative=%s)",
                sid,
                reconciliation_authoritative,
            )
            self._oms_services.append(oms)
            self._portfolio_checkers.append(getattr(oms, '_portfolio_checker', None))

            # Instrumentation (non-fatal) — share ONE sidecar across all strategies
            instr = None
            try:
                if getattr(overrides, "disable_instrumentation", False):
                    raise RuntimeError("instrumentation disabled by runtime overrides")
                from libs.oms.persistence.postgres import PgStore
                from .instrumentation.src.bootstrap import InstrumentationManager
                _pg_store = PgStore(db_pool) if db_pool is not None else None
                instr = InstrumentationManager(
                    oms, sid, strategy_type=desc["instr_type"],
                    pg_store=_pg_store,
                    family_strategy_ids=list(all_strategy_ids),
                    get_regime_ctx=lambda: self._regime_ctx,
                    get_applied_config=lambda: self._portfolio_checkers[0]._cfg if self._portfolio_checkers and self._portfolio_checkers[0] else None,
                    write_daily_closeout_on_stop=False,
                    stop_sidecar_on_stop=False,
                )
                if self._shared_sidecar is None:
                    self._shared_sidecar = instr.sidecar
                else:
                    instr.sidecar = self._shared_sidecar
                await instr.start()
                if require_instrumentation and not getattr(instr, "_sidecar_forwarding_enabled", True):
                    raise RuntimeError("sidecar forwarding disabled")
            except Exception as exc:
                if require_instrumentation:
                    raise RuntimeError(
                        f"Required instrumentation init failed for {sid}: {exc}"
                    ) from exc
                logger.warning(
                    "Instrumentation init failed for %s (non-fatal): %s", sid, exc,
                )
            self._instrumentations.append(instr)

            # Trade recorder
            trade_recorder = desc["trade_recorder"]
            try:
                from .instrumentation.src.facade import InstrumentationKit
                from .instrumentation.src.pg_bridge import InstrumentedTradeRecorder
                if instr is not None:
                    kit = InstrumentationKit(instr, strategy_type=desc["instr_type"])
                    trade_recorder = InstrumentedTradeRecorder(
                            trade_recorder,
                            kit,
                            strategy_id=sid,
                            strategy_type=desc["instr_type"],
                        )
            except Exception as exc:
                logger.warning(
                    "InstrumentedTradeRecorder setup failed for %s (non-fatal): %s",
                    sid, exc,
                )

            # Diagnostics
            diagnostics = desc["diagnostics_factory"]()

            # Build engine
            engine_cls = desc["engine_cls"]()
            engine_kwargs = dict(
                oms_service=oms,
                account_id=desc["account_id"],
                nav=allocated_nav,
                settings=desc["settings"],
                trade_recorder=trade_recorder,
                diagnostics=diagnostics,
                instrumentation=instr,
            )
            # Strategy-specific data source (artifact or cache)
            engine_kwargs[desc["data_key"]] = desc["data_value"]
            if getattr(overrides, "disable_background_tasks", False) and sid == "IARIC_v1":
                engine_kwargs["disable_background_tasks"] = True

            engine = engine_cls(**engine_kwargs)
            await engine.start()
            logger.info("Engine started for %s", sid)
            self._engines.append(engine)
            self._engine_map[sid] = engine

            # ── Market data source per engine ──────────────────────────
            md_source = None
            if not getattr(overrides, "disable_market_data", False) and self._contract_factory is not None:
                try:
                    MarketDataCls = desc["market_data_cls"]()
                    md_source = MarketDataCls(
                        ib=session.ib,
                        contract_factory=self._contract_factory,
                        on_quote=engine.on_quote,
                        on_bar=engine.on_bar,
                        historical_requester=getattr(session, "req_historical_data", None),
                    )
                    await md_source.start()
                    # Initial subscription setup
                    if hasattr(engine, "subscription_instruments"):
                        await md_source.ensure_hot_symbols(engine.subscription_instruments())
                    if hasattr(engine, "polling_instruments"):
                        await md_source.poll_due_bars(engine.polling_instruments())
                    logger.info("Market data started for %s", sid)
                except Exception as exc:
                    logger.error("Market data init failed for %s: %s", sid, exc, exc_info=exc)
                    if strict_market_data:
                        await self.stop()
                        raise RuntimeError(
                            f"Stock market data init failed for {sid} in {runtime_env} mode"
                        ) from exc
            else:
                logger.warning("No contract factory — market data NOT wired for %s", sid)
            if (
                self._contract_factory is None
                and strict_market_data
                and not getattr(overrides, "disable_market_data", False)
            ):
                await self.stop()
                raise RuntimeError(
                    f"Stock market data not wired for {sid} in {runtime_env} mode"
                )
            self._market_data_sources.append(md_source)

        # ── Reconnect callback ─────────────────────────────────────
        # CONN-1: also drive OMS reconciliation on each per-strategy OMS.
        # The previous version only did per-engine reconcile + subscription
        # invalidation, leaving OMS<->broker drift undetected.
        async def _on_reconnect() -> None:
            for i, eng in enumerate(self._engines):
                if hasattr(eng, "_reconcile_after_reconnect"):
                    try:
                        await eng._reconcile_after_reconnect()
                    except Exception as exc:
                        logger.error("Reconnect reconciliation failed for %s: %s",
                                     self._strategy_ids[i], exc)
                md = self._market_data_sources[i] if i < len(self._market_data_sources) else None
                if md is not None and hasattr(md, "invalidate_subscriptions"):
                    md.invalidate_subscriptions()
            for i, oms in enumerate(self._oms_services):
                reconciler = getattr(oms, "_reconciler", None)
                if reconciler is not None and getattr(reconciler, "is_authoritative", True):
                    try:
                        await reconciler.on_reconnect_reconciliation()
                    except Exception as exc:
                        logger.error(
                            "Stock OMS reconnect reconciliation failed for %s: %s",
                            self._strategy_ids[i], exc,
                        )
            logger.info(
                "Post-reconnect: per-engine + authoritative OMS reconciliation + "
                "subscription invalidation complete"
            )

        if session is not None and hasattr(session, "add_reconnect_callback"):
            session.add_reconnect_callback(_on_reconnect)

        # ── Background market data refresh loop ────────────────────
        if not getattr(overrides, "disable_market_data", False):
            self._market_data_task = asyncio.create_task(self._market_data_loop())

        # ── Heartbeat background task ──────────────────────────────────
        if not getattr(overrides, "disable_background_tasks", False):
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "StockFamilyCoordinator started %d strategies with market data", len(self._engines),
        )

    async def stop(self) -> None:
        """Stop market data, engines, and OMS instances in reverse order."""
        # Stop heartbeat loop
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        # Stop background market data loop
        if self._market_data_task is not None:
            self._market_data_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._market_data_task
            self._market_data_task = None

        for i in reversed(range(len(self._engines))):
            sid = self._strategy_ids[i]

            # Stop market data source
            if i < len(self._market_data_sources):
                md = self._market_data_sources[i]
                if md is not None:
                    try:
                        await md.stop()
                        logger.info("Market data stopped for %s", sid)
                    except Exception as exc:
                        logger.warning("Error stopping market data %s: %s", sid, exc)

            # Stop engine
            try:
                await self._engines[i].stop()
                logger.info("Engine stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping engine %s: %s", sid, exc)

            # Stop instrumentation
            try:
                instr = self._instrumentations[i]
                if instr is not None:
                    await instr.stop()
            except Exception as exc:
                logger.warning("Error stopping instrumentation %s: %s", sid, exc)

        await self._write_family_daily_closeout()
        self._flush_and_stop_shared_sidecar()

        for i in reversed(range(len(self._oms_services))):
            sid = self._strategy_ids[i] if i < len(self._strategy_ids) else "unknown"

            # Stop OMS
            try:
                await self._oms_services[i].stop()
                logger.info("OMS stopped for %s", sid)
            except Exception as exc:
                logger.error("Error stopping OMS %s: %s", sid, exc)

        self._engines.clear()
        self._oms_services.clear()
        self._instrumentations.clear()
        self._strategy_ids.clear()
        self._market_data_sources.clear()
        self._shared_sidecar = None
        logger.info("StockFamilyCoordinator stopped")

    async def _write_family_daily_closeout(self) -> None:
        instr = next((item for item in self._instrumentations if item is not None), None)
        if instr is None:
            return
        try:
            await instr.write_daily_closeout(
                oms_services=list(self._oms_services),
                strategy_ids=list(self._strategy_ids),
                family_id=self.family_id,
            )
        except Exception as exc:
            logger.warning("Stock family daily closeout failed: %s", exc)

    def _flush_and_stop_shared_sidecar(self) -> None:
        sidecar = self._shared_sidecar
        if sidecar is None:
            instr = next((item for item in self._instrumentations if item is not None), None)
            sidecar = getattr(instr, "sidecar", None) if instr is not None else None
        if sidecar is None:
            return
        try:
            sidecar.run_once()
            sidecar.stop()
        except Exception as exc:
            logger.warning("Stock shared sidecar stop error: %s", exc)

    def health_status(self) -> dict[str, Any]:
        """Return health of all three stock engines."""
        result: dict[str, Any] = {"family": self.family_id, "strategies": {}}
        for i, engine in enumerate(self._engines):
            sid = self._strategy_ids[i]
            try:
                result["strategies"][sid] = engine.health_status()
            except Exception as exc:
                result["strategies"][sid] = {"error": str(exc)}
        return result

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to all stock portfolio rules and engine configs."""
        import dataclasses
        from regime.integration import build_stock_rules, STOCK_PROFILES

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx
        new_rules = build_stock_rules(ctx, self._base_portfolio_rules)
        self._regime_adjusted_rules = new_rules  # Store for crisis overlay

        # Tier 2: engine position limit updates
        from regime.integration import _validated_regime
        regime = _validated_regime(ctx.regime)
        profile = STOCK_PROFILES[regime]
        self._regime_stock_profile = dict(profile)  # Store for crisis overlay
        for sid, engine in self._engine_map.items():
            settings = getattr(engine, '_settings', None)
            if settings is None:
                continue
            if sid == "ALCB_v1" and hasattr(settings, 'max_positions'):
                object.__setattr__(settings, 'max_positions', profile["alcb_max_positions"])
            elif sid == "IARIC_v1" and hasattr(settings, 'pb_max_positions'):
                object.__setattr__(settings, 'pb_max_positions', profile["iaric_pb_max_positions"])

        # Apply crisis overlay if active, including pre-action stress formation.
        if self._crisis_ctx is not None:
            from regime.crisis.integration import apply_crisis_overlay
            new_rules = apply_crisis_overlay(
                new_rules,
                self._crisis_ctx,
                self.family_id,
                regime=ctx.regime,
            )

        # Tier 1: update PortfolioRulesConfig on each checker
        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    new_rules, initial_equity=checker._cfg.initial_equity,
                ))

        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Stock regime applied: %s%s (cap=%.1fR, risk=%.2fx, disabled=%s)",
                    ctx.regime, changed, new_rules.directional_cap_R,
                    new_rules.regime_unit_risk_mult, new_rules.disabled_strategies or "none")

        # Emit structured regime→rules event for TA pipeline
        self._emit_regime_event({
            "family": "stock",
            "regime": str(ctx.regime),
            "prev_regime": str(prev_regime) if prev_regime else None,
            "rules_applied": {
                "directional_cap_R": new_rules.directional_cap_R,
                "regime_unit_risk_mult": new_rules.regime_unit_risk_mult,
                "disabled_strategies": new_rules.disabled_strategies or [],
                "alcb_max_positions": profile.get("alcb_max_positions"),
                "iaric_pb_max_positions": profile.get("iaric_pb_max_positions"),
            },
        })

    def apply_crisis(self, ctx) -> None:
        """Apply crisis context overlay on top of regime-adjusted rules.

        Handles both Tier 1 (PortfolioRulesConfig) and Tier 2 (engine settings).
        Always starts from _regime_adjusted_rules to prevent compounding.
        """
        import dataclasses
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
            for checker in self._portfolio_checkers:
                if checker is not None:
                    checker.update_config(dataclasses.replace(
                        self._regime_adjusted_rules,
                        initial_equity=checker._cfg.initial_equity,
                    ))
            # Restore regime Tier 2 settings
            if self._regime_stock_profile is not None:
                self._apply_tier2_settings(self._regime_stock_profile)
            self._refresh_instrumentation_lineage()
            if prev_level not in ("NORMAL", "WATCH"):
                logger.info("Stock crisis overlay removed (level=%s)", ctx.alert_level)
                current_rules = self._current_portfolio_rules_config()
                self._emit_crisis_event({
                    "family": "stock",
                    "alert_level": ctx.alert_level,
                    "prev_level": prev_level,
                    "crisis_action": "removed",
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
        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    tightened, initial_equity=checker._cfg.initial_equity,
                ))

        # Tier 2 crisis tightening: reduce engine position limits further
        if self._regime_stock_profile is not None and action.alert_level_int >= 2:
            crisis_profile = {
                "alcb_max_positions": max(
                    1, int(self._regime_stock_profile.get("alcb_max_positions", 6) * action.position_limit_multiplier)
                ),
                "iaric_pb_max_positions": max(
                    1, int(self._regime_stock_profile.get("iaric_pb_max_positions", 10) * action.position_limit_multiplier)
                ),
            }
            self._apply_tier2_settings(crisis_profile)

        changed = f" (was {prev_level})" if prev_level != ctx.alert_level else ""
        logger.info(
            "Stock crisis applied: %s%s (risk_mult=%.2f, dd_mult=%.2f, "
            "provenance=%s, dominant=%s)",
            ctx.alert_level, changed, action.risk_multiplier,
            action.dd_tier_multiplier, action.action_provenance, ctx.dominant_channel,
        )

        self._emit_crisis_event({
            "family": "stock",
            "alert_level": ctx.alert_level,
            "prev_level": prev_level,
            "risk_multiplier": ctx.risk_multiplier,
            "dd_tier_multiplier": ctx.dd_tier_multiplier,
            "dominant_channel": ctx.dominant_channel,
            "action_policy": action.to_dict(),
        })

    def _apply_tier2_settings(self, profile: dict) -> None:
        """Apply Tier 2 engine position limit settings."""
        for sid, engine in self._engine_map.items():
            settings = getattr(engine, '_settings', None)
            if settings is None:
                continue
            if sid == "ALCB_v1" and hasattr(settings, 'max_positions'):
                object.__setattr__(settings, 'max_positions', profile["alcb_max_positions"])
            elif sid == "IARIC_v1" and hasattr(settings, 'pb_max_positions'):
                object.__setattr__(settings, 'pb_max_positions', profile["iaric_pb_max_positions"])

    def _current_portfolio_rules_config(self):
        for checker in self._portfolio_checkers:
            if checker is not None:
                return getattr(checker, "_cfg", None)
        return self._regime_adjusted_rules or self._base_portfolio_rules

    def _refresh_instrumentation_lineage(self, rules_config=None) -> None:
        rules_config = rules_config or self._current_portfolio_rules_config()
        for instr in self._instrumentations:
            try:
                refresh = getattr(instr, "refresh_lineage", None)
                if callable(refresh):
                    refresh(rules_config)
            except Exception:
                logger.debug("Failed to refresh stock instrumentation lineage", exc_info=True)

    def _write_coordination_event(self, action_type: str, payload: dict) -> None:
        from libs.instrumentation.event_contract import append_jsonl_event, enrich_payload
        from libs.instrumentation.lineage import compute_risk_config_version, redact_config

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
        for instr in self._instrumentations:
            try:
                data_dir = getattr(instr, "_config", {}).get("data_dir")
                if not data_dir:
                    continue
                event = enrich_payload(
                    record,
                    lineage=getattr(instr, "lineage", None),
                    event_type="coordinator_action",
                    scope="family",
                )
                append_jsonl_event(data_dir, "coordination_events", "coordination_events", event)
            except Exception:
                logger.debug("Failed to emit stock coordination event", exc_info=True)
        self._last_coordination_rules_config = rules_config

    def _emit_crisis_event(self, payload: dict) -> None:
        """Write an enriched crisis event to each strategy's data_dir."""
        self._write_coordination_event("crisis_alert_change", payload)

    def _emit_regime_event(self, payload: dict) -> None:
        """Write a regime→rules event to each strategy's data_dir for TA pipeline."""
        self._write_coordination_event("regime_rules_change", payload)

    async def _heartbeat_loop(self) -> None:
        heartbeat = getattr(self._ctx, "heartbeat", None)
        if heartbeat is None:
            return
        session = self._ctx.session
        # HB-1: any exception in payload construction or emission must not
        # kill the loop. A dead heartbeat task wouldn't be observed until the
        # watchdog stale_threshold elapsed, masking a live runtime as silent.
        while True:
            try:
                await asyncio.sleep(15)
            except asyncio.CancelledError:
                return
            try:
                await self._heartbeat_iteration(heartbeat, session)
            except Exception:
                logger.exception("Stock heartbeat iteration failed; continuing")

    async def _heartbeat_iteration(self, heartbeat, session) -> None:
        payloads = []
        for i, sid in enumerate(self._strategy_ids):
            # Use per-strategy state (not portfolio state) for correct per-strategy metrics
            srs = getattr(self._oms_services[i], "_strategy_risk_states", {})
            sr = srs.get(sid)
            # Fall back to portfolio state for halted check
            prs = getattr(self._oms_services[i], "_portfolio_risk_state", None)
            payload = {
                "strategy_id": sid,
                "heat_r": Decimal(str(sr.open_risk_R)) if sr else Decimal("0"),
                "daily_pnl_r": Decimal(str(sr.daily_realized_R)) if sr else Decimal("0"),
                "mode": "HALTED" if (prs and prs.halted) else "RUNNING",
            }
            # Diagnostic pulse fields
            engine = self._engines[i]
            if hasattr(engine, "_last_decision_code"):
                payload["last_decision_code"] = engine._last_decision_code
                payload["last_decision_details"] = getattr(engine, "_last_decision_details", None)
                payload["last_seen_bar_ts"] = getattr(engine, "_last_bar_ts", None)
            oms = self._oms_services[i]
            if hasattr(engine, "liveness_payload") or hasattr(oms, "_intents_submitted"):
                details = payload.get("last_decision_details") or {}
                if hasattr(engine, "liveness_payload"):
                    details["liveness"] = engine.liveness_payload()
                if hasattr(oms, "_intents_submitted"):
                    details["oms_health"] = {
                        "submitted": oms._intents_submitted,
                        "accepted": oms._intents_accepted,
                        "denied": oms._intents_denied,
                        "consecutive_denials": oms._consecutive_denials,
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

    async def _market_data_loop(self) -> None:
        """Periodically refresh market data subscriptions for all engines."""
        while True:
            try:
                for i, engine in enumerate(self._engines):
                    md = self._market_data_sources[i] if i < len(self._market_data_sources) else None
                    if md is None:
                        continue
                    try:
                        if hasattr(engine, "subscription_instruments"):
                            await md.ensure_hot_symbols(engine.subscription_instruments())
                        if hasattr(engine, "polling_instruments"):
                            polling = engine.polling_instruments()
                            if polling:
                                await md.poll_due_bars(
                                    polling, now=datetime.now(timezone.utc),
                                )
                    except Exception as exc:
                        logger.warning(
                            "Market data refresh failed for %s: %s",
                            self._strategy_ids[i], exc,
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Market data loop error: %s", exc, exc_info=exc)
            await asyncio.sleep(5.0)

    # ── internal ─────────────────────────────────────────────────────

    def _enabled_stock_strategy_ids(self) -> tuple[str, ...]:
        """Return enabled stock strategy IDs for the current runtime environment."""
        overrides = getattr(self._ctx, "runtime_overrides", None)
        if overrides is not None:
            provider = getattr(overrides, "strategy_ids_provider", None)
            override_strategy_ids = provider() if provider is not None else getattr(overrides, "strategy_ids", None)
            if override_strategy_ids is not None:
                return tuple(dict.fromkeys(str(strategy_id) for strategy_id in override_strategy_ids))
        registry = getattr(self._ctx, "registry", None)
        if registry is None:
            return ()
        live = get_environment() == "live"
        return tuple(
            manifest.strategy_id
            for manifest in registry.enabled_strategies(live=live)
            if manifest.family == self.family_id
        )

    def _portfolio_rule_inputs(self, strategy_ids: tuple[str, ...]) -> dict[str, Any]:
        """Build stock-family rule inputs for the active strategy set."""
        active_ids = set(strategy_ids)
        return {
            "family_strategy_ids": strategy_ids,
            "symbol_collision_pairs": tuple(
                (holder_id, requester_id, action)
                for holder_id, requester_id, action in _STOCK_SYMBOL_COLLISION_PAIRS
                if holder_id in active_ids and requester_id in active_ids
            ),
            "strategy_priorities": tuple(
                (strategy_id, priority)
                for strategy_id, priority in _STOCK_STRATEGY_PRIORITIES
                if strategy_id in active_ids
            ),
        }

    def _build_strategy_descriptors(
        self,
        artifacts: dict[str, Any],
        strategy_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Return per-strategy wiring descriptors.

        Strategy modules are imported only for enabled stock strategies so a
        removed disabled package cannot break coordinator startup.
        """
        ctx = self._ctx
        overrides = getattr(ctx, "runtime_overrides", None)
        adapter_factory = getattr(overrides, "adapter_factory", None)
        offline_overrides = bool(
            adapter_factory is not None
            or getattr(overrides, "disable_market_data", False)
        )

        ibkr_config = None
        account_id = str((getattr(ctx, "contracts", {}) or {}).get("account_id", ""))
        ContractFactory = None
        IBKRExecutionAdapter = None
        if not offline_overrides:
            from libs.broker_ibkr.config.loader import IBKRConfig
            from libs.broker_ibkr.mapping.contract_factory import ContractFactory
            from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter

            config_dir = Path(os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
            try:
                ibkr_config = IBKRConfig(config_dir)
                account_id = ibkr_config.profile.account_id
            except Exception:
                ibkr_config = None
                account_id = ""
        elif not account_id:
            account_id = "ACCT-PARITY"

        # Build ContractFactory once, reuse for adapters and market data sources
        session = ctx.session
        if ibkr_config is not None and session is not None and ContractFactory is not None:
            self._contract_factory = ContractFactory(
                ib=session.ib,
                templates=ibkr_config.contracts,
                routes=ibkr_config.routes,
            )

        def _make_adapter(session: Any) -> Any:
            """Build an IBKRExecutionAdapter from the IB session."""
            if adapter_factory is not None:
                return _call_override_factory(adapter_factory, session=session)
            if self._contract_factory is None:
                raise RuntimeError("IBKRConfig not available")
            if IBKRExecutionAdapter is None:
                raise RuntimeError("IBKRExecutionAdapter not available")
            return IBKRExecutionAdapter(
                session=session,
                contract_factory=self._contract_factory,
                account=account_id,
            )

        # Shared trade recorder from bootstrap
        trade_recorder = getattr(ctx, "trade_recorder", None)

        def _build_iaric_descriptor() -> dict[str, Any]:
            from strategies.stock.iaric.config import (
                STRATEGY_ID as IARIC_ID,
                StrategySettings as IARICSettings,
            )

            iaric_settings = IARICSettings()
            return {
                "strategy_id": IARIC_ID,
                "base_risk_pct": iaric_settings.base_risk_fraction,
                "daily_stop_R": iaric_settings.daily_stop_r,
                "heat_cap_R": iaric_settings.heat_cap_r,
                "max_concurrent": iaric_settings.pb_max_positions,
                "portfolio_daily_stop_R": iaric_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": _import_iaric_engine,
                "market_data_cls": _import_iaric_market_data,
                "instr_type": "strategy_iaric",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": iaric_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(IARIC_ID),
                "diagnostics_factory": lambda s=iaric_settings: _make_diagnostics(
                    "strategies.stock.iaric.diagnostics", s.diagnostics_dir,
                ),
            }

        def _build_alcb_descriptor() -> dict[str, Any]:
            from strategies.stock.alcb.config import (
                STRATEGY_ID as ALCB_ID,
                StrategySettings as ALCBSettings,
            )

            alcb_settings = ALCBSettings()
            return {
                "strategy_id": ALCB_ID,
                "base_risk_pct": alcb_settings.base_risk_fraction,
                "daily_stop_R": alcb_settings.daily_stop_r,
                "heat_cap_R": alcb_settings.heat_cap_r,
                "max_concurrent": alcb_settings.max_positions,
                "portfolio_daily_stop_R": alcb_settings.portfolio_daily_stop_r,
                "adapter": _make_adapter,
                "engine_cls": _import_alcb_engine,
                "market_data_cls": _import_alcb_market_data,
                "instr_type": "strategy_alcb",
                "trade_recorder": trade_recorder,
                "account_id": account_id,
                "settings": alcb_settings,
                "data_key": "artifact",
                "data_value": artifacts.get(ALCB_ID),
                "diagnostics_factory": lambda s=alcb_settings: _make_diagnostics(
                    "strategies.stock.alcb.diagnostics", s.diagnostics_dir,
                ),
            }

        descriptor_builders = {
            "IARIC_v1": _build_iaric_descriptor,
            "ALCB_v1": _build_alcb_descriptor,
        }
        descriptors: list[dict[str, Any]] = []
        unsupported: list[str] = []
        for strategy_id in strategy_ids:
            builder = descriptor_builders.get(strategy_id)
            if builder is None:
                unsupported.append(strategy_id)
                continue
            descriptors.append(builder())
        if unsupported:
            logger.warning(
                "Skipping unsupported stock strategies with no wiring descriptor: %s",
                unsupported,
            )
        return descriptors


# ── Deferred engine imports ──────────────────────────────────────────

def _import_iaric_engine():
    from strategies.stock.iaric.engine import IARICEngine
    return IARICEngine


def _import_alcb_engine():
    from strategies.stock.alcb.engine import ALCBT2Engine
    return ALCBT2Engine


def _import_iaric_market_data():
    from strategies.stock.iaric.data import IBMarketDataSource
    return IBMarketDataSource


def _import_alcb_market_data():
    from strategies.stock.alcb.data import IBMarketDataSource
    return IBMarketDataSource


def _make_diagnostics(module_path: str, diagnostics_dir: Path) -> Any:
    """Instantiate JsonlDiagnostics from the given strategy module."""
    import importlib
    try:
        mod = importlib.import_module(module_path)
        return mod.JsonlDiagnostics(root=diagnostics_dir)
    except Exception as exc:
        logger.warning("Diagnostics init failed for %s: %s", module_path, exc)
        return None
