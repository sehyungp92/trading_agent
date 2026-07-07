"""Momentum family coordinator — each strategy gets its own OMS instance.

Unlike swing (shared OMS), momentum strategies are independent:
  - NQDTC_v2.1           (NQDTCEngine)      — MNQ day-trade continuation
  - NQ_REGIME            (NQRegimeEngine)   — NQ/MNQ intraday regime routing
  - VdubusNQ_v4          (VdubNQv4Engine)   — MNQ Vdubus signals
  - DownturnDominator_v1 (DownturnEngine)   — MNQ short-only regime scalping

Cross-strategy coordination is config-driven via PortfolioRulesConfig
(cooldown pairs, direction filter), NOT via in-process signaling.
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

_PORTFOLIO_WEEKLY_STOP_R = 9.0
_PORTFOLIO_DAILY_STOP_R = 2.75
_PORTFOLIO_HEAT_CAP_R = 10.0
_OPTIMIZED_INITIAL_EQUITY = 50_000.0
_OPTIMIZED_REFERENCE_UNIT_RISK_DOLLARS = 250.0
_REFERENCE_UNIT_RISK_PCT = _OPTIMIZED_REFERENCE_UNIT_RISK_DOLLARS / _OPTIMIZED_INITIAL_EQUITY
_MAX_TOTAL_POSITIONS = 8
_MAX_FAMILY_CONTRACTS_MNQ_EQ = 40
_DIRECTIONAL_CAP_R = 4.25
_DIRECTIONAL_CAP_LONG_R = 10.0
_DIRECTIONAL_CAP_SHORT_R = 10.5
_NQDTC_DIRECTION_FILTER_ENABLED = False
_NQDTC_AGREE_SIZE_MULT = 1.25
_NQDTC_OPPOSE_SIZE_MULT = 0.50
_DYNAMIC_EXISTING_POSITION_MULT = 0.85
_DYNAMIC_HEAT_PRESSURE_THRESHOLD = 0.65
_DYNAMIC_HEAT_PRESSURE_MULT = 0.65
_DYNAMIC_SAME_DIRECTION_PRESSURE_THRESHOLD = 0.65
_DYNAMIC_SAME_DIRECTION_PRESSURE_MULT = 0.70
_DYNAMIC_MAX_TRADE_RISK_R = 2.0
_DYNAMIC_MIN_QTY = 1
_DYNAMIC_FIT_TO_REMAINING_HEAT = True
_DYNAMIC_FIT_TO_REMAINING_DIRECTIONAL_CAP = True
_DYNAMIC_FIT_TO_REMAINING_FAMILY_CAP = True
_DD_TIERS = (
    (0.10, 1.00),
    (0.15, 0.60),
    (0.20, 0.30),
    (1.00, 0.00),
)
_STRATEGY_PRIORITIES = (
    ("VdubusNQ_v4", 0),
    ("NQ_REGIME", 0),
    ("NQDTC_v2.1", 1),
    ("DownturnDominator_v1", 1),
)
_MAX_STRATEGY_ACTIVE_POSITIONS = (
    ("NQ_REGIME", 3),
    ("VdubusNQ_v4", 2),
    ("NQDTC_v2.1", 2),
    ("DownturnDominator_v1", 2),
)
_STRATEGY_DAILY_STOPS_R = (
    ("NQ_REGIME", 3.0),
    ("VdubusNQ_v4", 2.5),
    ("NQDTC_v2.1", 2.5),
    ("DownturnDominator_v1", 2.0),
)
_STRATEGY_SIZE_MULTIPLIERS = (
    ("NQ_REGIME", 0.75),
    ("VdubusNQ_v4", 0.95),
    ("NQDTC_v2.1", 1.0),
    ("DownturnDominator_v1", 1.0),
)


def _momentum_family_nav(
    base_equity: float,
    allocs: dict[str, Any],
    strategy_ids: tuple[str, ...],
    ctx: RuntimeContext,
) -> float:
    """Return the momentum-family NAV used by optimized portfolio sizing."""
    for strategy_id in strategy_ids:
        alloc = allocs.get(strategy_id)
        family_fraction = getattr(alloc, "family_fraction", None)
        if _positive_finite(family_fraction):
            return float(base_equity) * float(family_fraction)

    capital = getattr(getattr(ctx, "portfolio", None), "capital", None)
    family_allocations = getattr(capital, "family_allocations", {}) or {}
    family_fraction = family_allocations.get("momentum")
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


class MomentumFamilyCoordinator:
    """Lifecycle manager for four momentum strategies.

    Each strategy receives its own OMS built via ``build_oms_service``.
    Shared across all four: *db_pool*, *AccountRiskGate*, *family_id*.
    """

    family_id = "momentum"

    def __init__(self, ctx: RuntimeContext) -> None:
        self._ctx = ctx
        self._engines: list = []
        self._oms_services: list = []
        self._instrumentations: list = []
        self._strategy_ids: list[str] = []
        self._portfolio_checkers: list = []
        self._base_portfolio_rules: Any = None
        self._base_max_family_contracts: int = 0
        self._regime_ctx: Any = None
        self._regime_adjusted_rules: Any = None  # stored after apply_regime for crisis overlay
        self._crisis_ctx: Any = None
        self._heartbeat_task: asyncio.Task | None = None
        self._shared_sidecar: Any = None

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:  # noqa: C901 — wiring is inherently sequential
        """Import, build OMS, and start all four momentum engines."""
        from libs.broker_ibkr.config.loader import IBKRConfig
        from libs.broker_ibkr.mapping.contract_factory import ContractFactory
        from libs.broker_ibkr.adapters.execution_adapter import IBKRExecutionAdapter
        from libs.oms.services.factory import build_oms_service as default_build_oms_service
        from libs.oms.risk.calculator import RiskCalculator
        from libs.oms.risk.portfolio_rules import PortfolioRulesConfig
        from libs.config.capital_bootstrap import bootstrap_capital

        ctx = self._ctx
        overrides = getattr(ctx, "runtime_overrides", None)
        build_oms_service = (
            getattr(overrides, "build_oms_service", None) or default_build_oms_service
        )
        adapter_factory = getattr(overrides, "adapter_factory", None)
        session = ctx.session
        db_pool = ctx.db_pool
        account_gate = ctx.account_gate
        require_instrumentation = bool(getattr(ctx, "require_instrumentation", False))

        if session is None and adapter_factory is None:
            raise RuntimeError(
                "Momentum family requires an IB session (connect_ib=True)"
            )

        # ── Execution adapter (shared IB session, one adapter instance) ──
        config_dir = Path(os.environ.get("CONFIG_DIR", str(Path(__file__).resolve().parent.parent.parent / "config")))
        ibkr_config = None
        contract_factory = None
        if adapter_factory is None:
            try:
                ibkr_config = IBKRConfig(config_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"IBKRConfig load failed: ensure config/ibkr_profiles.yaml exists: {exc}"
                ) from exc
            contract_factory = ContractFactory(
                ib=session.ib,
                templates=ibkr_config.contracts,
                routes=ibkr_config.routes,
            )
        active_account_id = str(
            getattr(getattr(ibkr_config, "profile", None), "account_id", "")
            or (getattr(ctx, "contracts", {}) or {}).get("account_id", "")
        )

        # ── Resolve equity ───────────────────────────────────────────
        paper_mode = get_environment() == "paper"
        paper_equity_pool: Any = None
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
            # resolved. The previous warn-and-default to $100k was a silent
            # mis-sizing risk when accountValues() raced startup.
            from libs.services.equity import resolve_live_nlv
            base_equity = await resolve_live_nlv(
                session, account_id=ibkr_config.profile.account_id,
            )

        # ── Capital allocation ───────────────────────────────────────
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
                logger.warning(
                    "bootstrap_capital failed (%s), using configured momentum family allocation fallback",
                    exc,
                )
                allocs = {}

        descriptors = self._build_strategy_descriptors()
        all_strategy_ids = tuple(d["strategy_id"] for d in descriptors)
        allocation_targets = build_family_allocation_targets(
            self.family_id,
            all_strategy_ids,
            allocations=allocs,
            portfolio=getattr(ctx, "portfolio", None),
        )
        override_portfolio_rules = _override_portfolio_rules(overrides)
        family_initial_nav = _momentum_family_nav(base_equity, allocs, all_strategy_ids, ctx)
        family_current_nav = family_initial_nav
        if paper_mode and getattr(overrides, "equity_provider", None) is None:
            from libs.persistence.paper_equity import PaperEquityManager

            pem = PaperEquityManager(
                db_pool,
                account_scope=self.family_id,
                initial_equity=family_initial_nav,
            )
            family_current_nav = await pem.load()
            paper_equity_pool = db_pool
            logger.info(
                "Paper mode equity for momentum portfolio: scope=%s current=$%.2f initial=$%.2f",
                self.family_id,
                family_current_nav,
                family_initial_nav,
            )
        momentum_equity_ref = [family_current_nav]
        reference_unit_risk = RiskCalculator.compute_unit_risk_dollars(
            nav=family_current_nav,
            unit_risk_pct=_REFERENCE_UNIT_RISK_PCT,
        )
        logger.info(
            "Momentum portfolio sizing base: current_nav=$%.2f initial_nav=$%.2f "
            "account_nav=$%.2f reference_1R=$%.2f",
            family_current_nav,
            family_initial_nav,
            base_equity,
            reference_unit_risk,
        )
        family_allocation_pct = float(
            getattr(getattr(ctx, "portfolio", None), "capital", None)
            .family_allocations.get(self.family_id, 0.0)
            if getattr(getattr(ctx, "portfolio", None), "capital", None) is not None
            else 0.0
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
                    family_heat_cap_R=_PORTFOLIO_HEAT_CAP_R,
                    family_daily_stop_R=_PORTFOLIO_DAILY_STOP_R,
                    family_weekly_stop_R=_PORTFOLIO_WEEKLY_STOP_R,
                    active_strategy_ids=list(all_strategy_ids),
                ),
                expires_at=active_config_expiry(),
            ),
        )

        # ── Dynamic MNQ contract cap ──────────────────────────────────
        try:
            if getattr(overrides, "disable_market_data", False) or session is None:
                raise RuntimeError("market data disabled by runtime overrides")
            from ib_async import ContFuture
            mnq_cont = ContFuture("MNQ", "CME")
            await session.ib.qualifyContractsAsync(mnq_cont)
            bars = await session.req_historical_data(
                mnq_cont, "", "1 D", "1 day", "TRADES", False,
                request_kind="quick",
            )
            mnq_price = bars[-1].close if bars else 21000.0
        except Exception:
            mnq_price = 21000.0
            logger.warning("MNQ price fetch failed, using default %.0f", mnq_price)

        max_family_contracts = _MAX_FAMILY_CONTRACTS_MNQ_EQ
        self._base_max_family_contracts = max_family_contracts
        logger.info(
            "Momentum MNQ family cap: %d contracts (equity=$%.0f, phase-auto cap, MNQ=%.0f)",
            max_family_contracts, family_current_nav, mnq_price,
        )

        # ── Strategy descriptors ─────────────────────────────────────
        self._shared_sidecar = None
        authoritative_strategy_id = all_strategy_ids[0] if all_strategy_ids else ""
        for desc in descriptors:
            sid = desc["strategy_id"]
            reconciliation_authoritative = sid == authoritative_strategy_id
            self._strategy_ids.append(sid)

            # Per-strategy adapter (same session, own adapter instance)
            if adapter_factory is not None:
                adapter = _call_override_factory(adapter_factory, strategy_id=sid, session=session)
            else:
                adapter = IBKRExecutionAdapter(
                    session=session,
                    contract_factory=contract_factory,
                    account=ibkr_config.profile.account_id,
                )

            # Resolve optimized family NAV.
            alloc = allocs.get(sid)
            initial_nav = family_initial_nav
            allocated_nav = momentum_equity_ref[0]
            if alloc:
                logger.info(
                    "Capital allocation: %s -> momentum portfolio NAV $%.2f "
                    "(config allocation %.1f%% kept for non-momentum allocators only)",
                    sid, allocated_nav, getattr(alloc, "capital_pct", 0.0),
                )
            else:
                logger.warning(
                    "Strategy %s not in unified config, using momentum portfolio NAV %.2f",
                    sid, allocated_nav,
                )

            # Risk
            unit_risk = RiskCalculator.compute_unit_risk_dollars(
                nav=allocated_nav, unit_risk_pct=desc["base_risk_pct"],
            )
            # Backtest strategy daily stops are expressed on portfolio reference-R.
            strategy_daily_stop_R = desc["daily_stop_R"]
            if unit_risk > 0 and reference_unit_risk > 0:
                strategy_daily_stop_R = (
                    desc["daily_stop_R"] * reference_unit_risk / unit_risk
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
                        max_heat_R=0.0,
                        max_daily_loss_R=strategy_daily_stop_R,
                        max_weekly_loss_R=_PORTFOLIO_WEEKLY_STOP_R,
                        risk_per_trade=float(desc["base_risk_pct"]),
                    ),
                    expires_at=active_config_expiry(),
                ),
            )

            portfolio_rules = override_portfolio_rules or PortfolioRulesConfig(
                    initial_equity=initial_nav,
                    directional_cap_R=_DIRECTIONAL_CAP_R,
                    directional_cap_long_R=_DIRECTIONAL_CAP_LONG_R,
                    directional_cap_short_R=_DIRECTIONAL_CAP_SHORT_R,
                    max_total_active_positions=_MAX_TOTAL_POSITIONS,
                    max_strategy_active_positions=tuple(
                        (strategy_id, cap)
                        for strategy_id, cap in _MAX_STRATEGY_ACTIVE_POSITIONS
                        if strategy_id in all_strategy_ids
                    ),
                    max_family_contracts_mnq_eq=max_family_contracts,
                    family_strategy_ids=all_strategy_ids,
                    symbol_collision_action="none",
                    cooldown_session_only=True,
                    nqdtc_direction_filter_enabled=_NQDTC_DIRECTION_FILTER_ENABLED,
                    nqdtc_agree_size_mult=_NQDTC_AGREE_SIZE_MULT,
                    nqdtc_oppose_size_mult=_NQDTC_OPPOSE_SIZE_MULT,
                    strategy_priorities=tuple(
                        (strategy_id, priority)
                        for strategy_id, priority in _STRATEGY_PRIORITIES
                        if strategy_id in all_strategy_ids
                    ),
                    strategy_size_multipliers=tuple(
                        (strategy_id, multiplier)
                        for strategy_id, multiplier in _STRATEGY_SIZE_MULTIPLIERS
                        if strategy_id in all_strategy_ids
                    ),
                    priority_headroom_R=1.0,
                    priority_reserve_threshold=1,
                    reference_unit_risk_dollars=reference_unit_risk,
                    portfolio_heat_cap_R=_PORTFOLIO_HEAT_CAP_R,
                    existing_position_mult=_DYNAMIC_EXISTING_POSITION_MULT,
                    heat_pressure_threshold=_DYNAMIC_HEAT_PRESSURE_THRESHOLD,
                    heat_pressure_mult=_DYNAMIC_HEAT_PRESSURE_MULT,
                    same_direction_pressure_threshold=_DYNAMIC_SAME_DIRECTION_PRESSURE_THRESHOLD,
                    same_direction_pressure_mult=_DYNAMIC_SAME_DIRECTION_PRESSURE_MULT,
                    max_trade_risk_R=_DYNAMIC_MAX_TRADE_RISK_R,
                    min_qty=_DYNAMIC_MIN_QTY,
                    fit_to_remaining_heat=_DYNAMIC_FIT_TO_REMAINING_HEAT,
                    fit_to_remaining_directional_cap=_DYNAMIC_FIT_TO_REMAINING_DIRECTIONAL_CAP,
                    fit_to_remaining_family_cap=_DYNAMIC_FIT_TO_REMAINING_FAMILY_CAP,
                    dd_tiers=_DD_TIERS,
                )

            if self._base_portfolio_rules is None:
                self._base_portfolio_rules = portfolio_rules

            # Build per-strategy OMS
            oms = await build_oms_service(
                adapter=adapter,
                strategy_id=sid,
                unit_risk_dollars=unit_risk,
                portfolio_unit_risk_dollars=reference_unit_risk,
                daily_stop_R=strategy_daily_stop_R,
                heat_cap_R=_PORTFOLIO_HEAT_CAP_R,
                portfolio_daily_stop_R=_PORTFOLIO_DAILY_STOP_R,
                portfolio_weekly_stop_R=_PORTFOLIO_WEEKLY_STOP_R,
                db_pool=db_pool,
                portfolio_rules_config=portfolio_rules,
                get_current_equity=lambda eq=momentum_equity_ref: eq[0],
                paper_equity_pool=paper_equity_pool,
                paper_equity_scope=self.family_id,
                paper_initial_equity=initial_nav,
                paper_equity_ref=momentum_equity_ref if paper_mode else None,
                live_equity=momentum_equity_ref if not paper_equity_pool else None,
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

            # Build engine
            engine_cls = desc["engine_cls"]
            engine_kwargs: dict[str, Any] = dict(
                ib_session=session,
                oms_service=oms,
                instruments=desc["build_instruments"](),
                trade_recorder=desc.get("trade_recorder"),
                equity=allocated_nav,
                instrumentation=instr,
                equity_alloc_pct=1.0,
            )
            engine_kwargs.update(desc.get("engine_extra_kwargs", {}))
            if getattr(overrides, "disable_background_tasks", False):
                if sid == "NQ_REGIME":
                    engine_kwargs["disable_scheduler"] = True
                if sid == "NQDTC_v2.1":
                    engine_kwargs["disable_background_tasks"] = True
                if sid in {"VdubusNQ_v4", "DownturnDominator_v1"}:
                    engine_kwargs["disable_background_tasks"] = True

            engine = engine_cls(**engine_kwargs)
            await engine.start()
            logger.info("Engine started for %s", sid)
            self._engines.append(engine)

        # ── Reconnect callback ─────────────────────────────────────
        # CONN-1: momentum previously installed no reconnect callback. With
        # 4 per-strategy OMS instances, that meant zero post-reconnect OMS
        # reconciliation across the whole family.
        async def _on_reconnect() -> None:
            for i, oms in enumerate(self._oms_services):
                reconciler = getattr(oms, "_reconciler", None)
                if reconciler is not None and getattr(reconciler, "is_authoritative", True):
                    try:
                        await reconciler.on_reconnect_reconciliation()
                    except Exception as exc:
                        sid = self._strategy_ids[i] if i < len(self._strategy_ids) else "?"
                        logger.error(
                            "Momentum OMS reconnect reconciliation failed for %s: %s",
                            sid, exc,
                        )
            logger.info(
                "Momentum post-reconnect: authoritative OMS reconciler fired",
            )

        if session is not None and hasattr(session, "add_reconnect_callback"):
            session.add_reconnect_callback(_on_reconnect)

        # ── Heartbeat background task ──────────────────────────────────
        if not getattr(overrides, "disable_background_tasks", False):
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(
            "MomentumFamilyCoordinator started %d strategies", len(self._engines),
        )

    async def stop(self) -> None:
        """Stop engines, instrumentation, and OMS instances in reverse order."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task
            self._heartbeat_task = None

        for i in reversed(range(len(self._engines))):
            sid = self._strategy_ids[i]

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
        self._shared_sidecar = None
        logger.info("MomentumFamilyCoordinator stopped")

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
            logger.warning("Momentum family daily closeout failed: %s", exc)

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
            logger.warning("Momentum shared sidecar stop error: %s", exc)

    def health_status(self) -> dict[str, Any]:
        """Return health of all four momentum engines."""
        result: dict[str, Any] = {"family": self.family_id, "strategies": {}}
        for i, engine in enumerate(self._engines):
            sid = self._strategy_ids[i]
            try:
                result["strategies"][sid] = engine.health_status()
            except Exception as exc:
                result["strategies"][sid] = {"error": str(exc)}
        return result

    def apply_regime(self, ctx: "RegimeContext") -> None:
        """Apply regime context to all momentum portfolio rules."""
        import dataclasses
        from regime.integration import build_momentum_rules

        if self._base_portfolio_rules is None:
            logger.warning("apply_regime called before start() — skipping")
            return

        prev_regime = getattr(self._regime_ctx, "regime", None)
        self._regime_ctx = ctx
        new_rules = build_momentum_rules(ctx, self._base_portfolio_rules, self._base_max_family_contracts)
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

        for checker in self._portfolio_checkers:
            if checker is not None:
                checker.update_config(dataclasses.replace(
                    new_rules, initial_equity=checker._cfg.initial_equity,
                ))

        r = new_rules
        changed = f" (was {prev_regime})" if prev_regime and prev_regime != ctx.regime else ""
        logger.info("Momentum regime applied: %s%s (cap=%.1fR, long=%.1fR, short=%.1fR, "
                    "risk=%.2fx, contracts=%d, oppose=%.1f, disabled=%s)",
                    ctx.regime, changed, r.directional_cap_R, r.directional_cap_long_R,
                    r.directional_cap_short_R, r.regime_unit_risk_mult,
                    r.max_family_contracts_mnq_eq, r.nqdtc_oppose_size_mult,
                    r.disabled_strategies or "none")

        # Emit structured regime→rules event for TA pipeline
        self._emit_regime_event({
            "family": "momentum",
            "regime": str(ctx.regime),
            "prev_regime": str(prev_regime) if prev_regime else None,
            "rules_applied": {
                "directional_cap_R": r.directional_cap_R,
                "directional_cap_long_R": r.directional_cap_long_R,
                "directional_cap_short_R": r.directional_cap_short_R,
                "regime_unit_risk_mult": r.regime_unit_risk_mult,
                "max_family_contracts_mnq_eq": r.max_family_contracts_mnq_eq,
                "nqdtc_oppose_size_mult": getattr(r, "nqdtc_oppose_size_mult", 1.0),
                "disabled_strategies": r.disabled_strategies or [],
            },
        })

    def apply_crisis(self, ctx) -> None:
        """Apply crisis context overlay on top of regime-adjusted rules.

        Always starts from _regime_adjusted_rules (not current checker config)
        to prevent compounding. Preserves per-checker initial_equity.
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
            self._refresh_instrumentation_lineage()
            if prev_level not in ("NORMAL", "WATCH"):
                logger.info("Momentum crisis overlay removed (level=%s)", ctx.alert_level)
                current_rules = self._current_portfolio_rules_config()
                self._emit_crisis_event({
                    "family": "momentum",
                    "alert_level": ctx.alert_level,
                    "prev_level": prev_level,
                    "crisis_action": "removed",
                    "rules_applied": {
                        "directional_cap_R": getattr(current_rules, "directional_cap_R", None),
                        "directional_cap_long_R": getattr(current_rules, "directional_cap_long_R", None),
                        "directional_cap_short_R": getattr(current_rules, "directional_cap_short_R", None),
                        "regime_unit_risk_mult": getattr(current_rules, "regime_unit_risk_mult", None),
                        "max_family_contracts_mnq_eq": getattr(current_rules, "max_family_contracts_mnq_eq", None),
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

        changed = f" (was {prev_level})" if prev_level != ctx.alert_level else ""
        logger.info(
            "Momentum crisis applied: %s%s (risk_mult=%.2f, dd_mult=%.2f, "
            "provenance=%s, dominant=%s)",
            ctx.alert_level, changed, action.risk_multiplier,
            action.dd_tier_multiplier, action.action_provenance, ctx.dominant_channel,
        )

        self._emit_crisis_event({
            "family": "momentum",
            "alert_level": ctx.alert_level,
            "prev_level": prev_level,
            "risk_multiplier": ctx.risk_multiplier,
            "dd_tier_multiplier": ctx.dd_tier_multiplier,
            "dominant_channel": ctx.dominant_channel,
            "action_policy": action.to_dict(),
        })

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
                logger.debug("Failed to refresh momentum instrumentation lineage", exc_info=True)

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
                logger.debug("Failed to emit momentum coordination event", exc_info=True)
        self._last_coordination_rules_config = rules_config

    def _emit_crisis_event(self, payload: dict) -> None:
        """Write an enriched crisis event to each strategy's data_dir."""
        self._write_coordination_event("crisis_alert_change", payload)

    def _emit_regime_event(self, payload: dict) -> None:
        """Write a regime→rules event to each strategy's data_dir for TA pipeline."""
        self._write_coordination_event("regime_rules_change", payload)

    # ── heartbeat ──────────────────────────────────────────────────

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
                logger.exception("Momentum heartbeat iteration failed; continuing")

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

    # ── internal ─────────────────────────────────────────────────────

    def _build_strategy_descriptors(self) -> list[dict[str, Any]]:
        """Return per-strategy wiring descriptors.

        NQDTC and VdubusNQ hardcode their OMS risk params (matching main.py).
        """
        ctx = self._ctx

        # CFG-5: prefer manifest.engine_config["state_dir"] from
        # config/strategies.yaml over env-var-only. This makes YAML edits
        # actually take effect (previously the YAML key was decorative).
        def _resolve_state_dir(strategy_id: str, env_var: str, default: str) -> Path:
            overrides = getattr(ctx, "runtime_overrides", None)
            override_dirs = getattr(overrides, "state_dir_overrides", {}) or {}
            if strategy_id in override_dirs:
                return Path(override_dirs[strategy_id])
            manifest = None
            if ctx.registry is not None and strategy_id in ctx.registry.strategies:
                manifest = ctx.registry.strategies[strategy_id]
            yaml_dir = (manifest.engine_config or {}).get("state_dir") if manifest else None
            return Path(yaml_dir or os.environ.get(env_var) or default)

        # Trade recorder from context
        trade_recorder = getattr(ctx, "trade_recorder", None)
        if trade_recorder is None:
            trade_recorder = getattr(ctx.instrumentation, "trade_recorder", None)

        # ── NQDTC v2.1 ──────────────────────────────────────────────
        from strategies.momentum.nqdtc.config import (
            STRATEGY_ID as NQDTC_ID,
            RISK_PCT as NQDTC_RISK_PCT,
            build_instruments as nqdtc_build_instruments,
        )
        from strategies.momentum.nqdtc.engine import NQDTCEngine

        # NQ Regime Engine
        from strategies.momentum.nq_regime.config import (
            STRATEGY_ID as NQ_REGIME_ID,
            BASE_RISK_PCT as NQ_REGIME_RISK_PCT,
            build_instruments as nq_regime_build_instruments,
        )
        from strategies.momentum.nq_regime.engine import NQRegimeEngine

        # ── VdubusNQ v4 ─────────────────────────────────────────────
        from strategies.momentum.vdub.config import (
            STRATEGY_ID as VDUB_ID,
            BASE_RISK_PCT as VDUB_RISK_PCT,
            build_instruments as vdub_build_instruments,
        )
        from strategies.momentum.vdub.engine import VdubNQv4Engine

        # ── Downturn Dominator v1 ──────────────────────────────────
        from strategies.momentum.downturn.config import (
            STRATEGY_ID as DOWNTURN_ID,
            BASE_RISK_PCT as DOWNTURN_RISK_PCT,
            build_instruments as downturn_build_instruments,
        )
        from strategies.momentum.downturn.engine import DownturnEngine

        daily_stops = dict(_STRATEGY_DAILY_STOPS_R)

        descriptors = [
            # ── NQDTC_v2.1 ─────────────────────────────────────────
            {
                "strategy_id": NQDTC_ID,
                "base_risk_pct": NQDTC_RISK_PCT,
                "daily_stop_R": daily_stops[NQDTC_ID],
                "build_instruments": nqdtc_build_instruments,
                "engine_cls": NQDTCEngine,
                "instr_type": "nqdtc",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {
                    "state_dir": _resolve_state_dir(
                        NQDTC_ID, "NQDTC_STATE_DIR", "data/nqdtc_state",
                    ),
                },
            },
            # NQ_REGIME — STRAT-2: state_dir wired so on-disk core snapshot
            # survives restarts. The engine itself has the 5m scheduler and
            # _restore_state/_persist_state hooks that consume this path.
            {
                "strategy_id": NQ_REGIME_ID,
                "base_risk_pct": NQ_REGIME_RISK_PCT,
                "daily_stop_R": daily_stops[NQ_REGIME_ID],
                "build_instruments": nq_regime_build_instruments,
                "engine_cls": NQRegimeEngine,
                "instr_type": "nq_regime",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {
                    "analysis_symbol": "NQ",
                    "trade_symbol": "MNQ",
                    "state_dir": _resolve_state_dir(
                        NQ_REGIME_ID, "NQ_REGIME_STATE_DIR", "data/nq_regime_state",
                    ),
                },
            },
            # ── VdubusNQ_v4 ────────────────────────────────────────
            {
                "strategy_id": VDUB_ID,
                "base_risk_pct": VDUB_RISK_PCT,
                "daily_stop_R": daily_stops[VDUB_ID],
                "build_instruments": vdub_build_instruments,
                "engine_cls": VdubNQv4Engine,
                "instr_type": "vdubus",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {},
            },
            # ── DownturnDominator_v1 ──────────────────────────────
            {
                "strategy_id": DOWNTURN_ID,
                "base_risk_pct": DOWNTURN_RISK_PCT,
                "daily_stop_R": daily_stops[DOWNTURN_ID],
                "build_instruments": downturn_build_instruments,
                "engine_cls": DownturnEngine,
                "instr_type": "downturn",
                "trade_recorder": trade_recorder,
                "engine_extra_kwargs": {
                    "state_dir": _resolve_state_dir(
                        DOWNTURN_ID, "DOWNTURN_STATE_DIR", "data/downturn_state",
                    ),
                },
            },
        ]
        overrides = getattr(ctx, "runtime_overrides", None)
        override_strategy_ids = None
        if overrides is not None:
            provider = getattr(overrides, "strategy_ids_provider", None)
            override_strategy_ids = provider() if provider is not None else getattr(overrides, "strategy_ids", None)
        if override_strategy_ids is not None:
            active = {str(strategy_id) for strategy_id in override_strategy_ids}
            descriptors = [desc for desc in descriptors if desc["strategy_id"] in active]
        return descriptors
