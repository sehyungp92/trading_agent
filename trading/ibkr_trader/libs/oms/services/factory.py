"""OMS factory for proper initialization with all dependencies.

Merged from swing_trader, momentum_trader, and stock_trader families.
- build_oms_service(): unified single-strategy factory (union of all params)
- build_multi_strategy_oms(): swing's coordinator pattern for cross-strategy OMS
"""
import os
import json
import logging
import math
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Awaitable, Callable, Optional, TYPE_CHECKING
from zoneinfo import ZoneInfo

import asyncpg

from ..config.risk_config import RiskConfig, StrategyRiskConfig
from ..coordination.coordinator import StrategyCoordinator
from ..engine.fill_processor import FillProcessor
from ..engine.state_machine import transition
from ..events.bus import EventBus
from ..execution.router import ExecutionRouter
from ..intent.handler import IntentHandler
from ..models.instrument_registry import InstrumentRegistry
from ..models.order import OrderRole, OrderSide, OrderStatus
from ..models.position import Position
from ..models.risk_state import StrategyRiskState, PortfolioRiskState
from ..persistence.in_memory import InMemoryRepository
from ..persistence.postgres import PgStore
from ..persistence.repository import OMSRepository
from ..persistence.schema import RiskDailyPortfolioRow, RiskDailyStrategyRow
from ..reconciliation.authority import ReconciliationAuthority
from ..reconciliation.orchestrator import ReconciliationOrchestrator
from ..risk.calendar import EventCalendar
from ..risk.gateway import RiskGateway
from .oms_service import OMSService
from libs.instrumentation.event_contract import COMMON_LINEAGE_FIELDS, enrich_payload
from libs.instrumentation.lineage import LineageContext, lineage_from_runtime, lineage_to_payload
from libs.oms.persistence.db_config import get_environment
from libs.runtime.active_config import (
    ActiveRuntimeConfigRecord,
    active_config_expiry,
    upsert_active_runtime_config,
)

if TYPE_CHECKING:
    from libs.config.market_calendar import MarketCalendar

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_reconciliation_authority(
    *,
    db_pool,
    provided_authority: object | None,
    enable_lease: bool | None,
) -> object | None:
    if provided_authority is not None:
        return provided_authority
    leases_enabled = (
        _env_flag("OMS_RECONCILIATION_AUTHORITY_LEASES")
        if enable_lease is None
        else enable_lease
    )
    if not leases_enabled or db_pool is None:
        return None
    return ReconciliationAuthority(db_pool)


def _make_oms_lineage(
    *,
    bot_id: str,
    strategy_id: str = "",
    family_id: str = "",
    portfolio_id: str = "",
    account_alias: str = "",
    portfolio_config=None,
    portfolio_rules_config=None,
    strategy_manifest=None,
    effective_strategy_config=None,
    parameter_set=None,
) -> LineageContext:
    try:
        return lineage_from_runtime(
            bot_id=bot_id,
            strategy_id=strategy_id,
            family_id=family_id,
            portfolio_id=portfolio_id,
            account_alias=account_alias,
            portfolio_config=portfolio_config,
            portfolio_rules_config=portfolio_rules_config,
            strategy_manifest=strategy_manifest,
            effective_strategy_config=effective_strategy_config,
            parameter_set=parameter_set,
        )
    except Exception as exc:
        logger.warning("OMS lineage computation failed; continuing with partial lineage: %s", exc)
        return LineageContext(
            bot_id=bot_id,
            strategy_id=strategy_id,
            family_id=family_id,
            portfolio_id=portfolio_id,
            account_alias=account_alias,
        )


def _resolve_lineage(lineage):
    if callable(lineage):
        try:
            return lineage()
        except Exception:
            return None
    return lineage


def _overlay_lineage_fields(payload: dict, lineage) -> dict:
    lineage_payload = lineage_to_payload(lineage)
    for field in COMMON_LINEAGE_FIELDS:
        if field in {"scope", "schema_version"}:
            continue
        value = lineage_payload.get(field)
        if value not in (None, "", [], {}):
            payload[field] = value
    if lineage_payload.get("bot_id"):
        payload["bot_id"] = lineage_payload["bot_id"]
    return payload


# OMS-5: Hydrate the per-strategy in-memory `open_positions` cache from the
# `positions` table on startup. The dict is consulted by exit-fill processing to
# release risk, compute realized P&L, and update equity. If we don't reseed it
# after a restart, an exit fill that arrives before any new entry has been
# processed in this lifetime cannot compute P&L or release risk. DB-backed
# paper/live startup therefore treats hydration failure as fatal, and the fill
# callback halts if it ever sees an unknown exit position. The schema doesn't
# store side/point_value/risk_per_contract directly, so we derive them.
async def _hydrate_open_positions_from_repo(
    *,
    open_positions: dict[tuple[str, str], dict],
    repo: OMSRepository,
    strategy_ids: list[str],
    unit_risk_dollars_for: Callable[[str], float],
    portfolio_unit_risk_dollars: Optional[float] = None,
    strict: bool = False,
) -> int:
    """Seed open_positions for any non-zero broker position on disk.

    Args:
        open_positions: dict to populate in-place.
        repo: OMS repository.
        strategy_ids: strategies whose positions we own.
        unit_risk_dollars_for: maps strategy_id -> URD for that strategy.
        portfolio_unit_risk_dollars: cross-family URD used by multi-strategy
            OMS to compute risk_per_contract_portfolio_R. None for single-OMS.

    Returns the number of positions hydrated.
    """
    if not strategy_ids:
        return 0
    try:
        positions = await repo.get_positions_for_strategies(strategy_ids)
    except Exception as exc:
        if strict:
            raise RuntimeError("open_positions hydration failed") from exc
        logger.warning(
            "open_positions hydration query failed (non-fatal): %s", exc,
        )
        return 0

    hydrated = 0
    for pos in positions:
        if not pos.net_qty:
            continue
        sid = pos.strategy_id
        sym = pos.instrument_symbol
        if not sym or sid not in strategy_ids:
            continue
        urd = unit_risk_dollars_for(sid) or 0.0
        # risk_per_contract_R is what the EXIT branch multiplies by qty to
        # release strategy-R; derive from total open_risk_R / |net_qty|.
        abs_qty = abs(float(pos.net_qty))
        risk_per_contract_R = (
            float(pos.open_risk_R) / abs_qty if abs_qty > 0 else 0.0
        )
        if portfolio_unit_risk_dollars and portfolio_unit_risk_dollars > 0 and urd > 0:
            risk_per_contract_portfolio_R = (
                risk_per_contract_R * urd / portfolio_unit_risk_dollars
            )
        else:
            risk_per_contract_portfolio_R = risk_per_contract_R

        instr = InstrumentRegistry.get(sym)
        point_value = float(getattr(instr, "point_value", 1.0)) if instr else 1.0
        side = OrderSide.BUY if pos.net_qty > 0 else OrderSide.SELL

        entry: dict = {
            "entry_price": float(pos.avg_price),
            "risk_per_contract_R": risk_per_contract_R,
            "point_value": point_value,
            "side": side,
            "open_qty": abs_qty,
        }
        if portfolio_unit_risk_dollars is not None:
            entry["risk_per_contract_portfolio_R"] = risk_per_contract_portfolio_R
        open_positions[(sid, sym)] = entry
        hydrated += 1

    if hydrated:
        logger.info(
            "open_positions hydrated: %d position(s) restored from DB across %s",
            hydrated, ", ".join(strategy_ids),
        )
    return hydrated


_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_decimal(value: float) -> Decimal:
    """Convert float state values to stable decimal strings for persistence."""
    return Decimal(str(value))


def _trade_date_for(timestamp: datetime | None = None):
    """Return the current US trading date in America/New_York."""
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_ET).date()


def _week_start(trade_day: date) -> date:
    """Mirror the existing Monday-based OMS weekly reset semantics."""
    return trade_day - timedelta(days=trade_day.weekday())


def _aggregate_strategy_daily_rows(
    rows: list[RiskDailyStrategyRow],
) -> tuple[float, float, dict[str, float]]:
    daily_realized_r = 0.0
    daily_realized_usd = 0.0
    strategy_daily_pnl: dict[str, float] = {}

    for row in rows:
        realized_r = float(row.daily_realized_r)
        realized_usd = float(row.daily_realized_usd or 0)
        daily_realized_r += realized_r
        daily_realized_usd += realized_usd
        strategy_daily_pnl[row.strategy_id] = realized_usd

    return daily_realized_r, daily_realized_usd, strategy_daily_pnl


def _build_strategy_daily_row(
    state: StrategyRiskState,
    existing_row: RiskDailyStrategyRow | None,
    as_of: datetime,
    family_id: str = "unknown",
) -> RiskDailyStrategyRow:
    return RiskDailyStrategyRow(
        trade_date=state.trade_date,
        strategy_id=state.strategy_id,
        family_id=family_id,
        daily_realized_r=_to_decimal(state.daily_realized_R),
        daily_realized_usd=_to_decimal(state.daily_realized_pnl),
        open_risk_r=_to_decimal(state.open_risk_R),
        filled_entries=existing_row.filled_entries if existing_row else 0,
        halted=state.halted or (existing_row.halted if existing_row else False),
        halt_reason=state.halt_reason or (existing_row.halt_reason if existing_row else None),
        last_update_at=as_of,
    )


def _build_portfolio_daily_row(
    trade_date: date,
    daily_rows: list[RiskDailyStrategyRow],
    open_risk_r: float,
    existing_row: RiskDailyPortfolioRow | None,
    halted: bool,
    halt_reason: str,
    as_of: datetime,
    family_id: str = "unknown",
    portfolio_urd: float = 0.0,
) -> RiskDailyPortfolioRow:
    if daily_rows:
        daily_realized_r, daily_realized_usd, _ = _aggregate_strategy_daily_rows(daily_rows)
        if portfolio_urd > 0 and daily_realized_usd != 0.0:
            daily_realized_r = daily_realized_usd / portfolio_urd
    elif existing_row is not None:
        daily_realized_r = float(existing_row.daily_realized_r)
        daily_realized_usd = float(existing_row.daily_realized_usd or 0)
    else:
        daily_realized_r = 0.0
        daily_realized_usd = 0.0

    return RiskDailyPortfolioRow(
        trade_date=trade_date,
        family_id=family_id,
        daily_realized_r=_to_decimal(daily_realized_r),
        daily_realized_usd=_to_decimal(daily_realized_usd),
        portfolio_open_risk_r=_to_decimal(open_risk_r),
        halted=halted or (existing_row.halted if existing_row else False),
        halt_reason=halt_reason or (existing_row.halt_reason if existing_row else None),
        last_update_at=as_of,
    )


def _make_portfolio_rule_logger(data_dir: str = "", family_id: str = "", lineage=None) -> Callable:
    """Create a JSONL-writing callback for portfolio rule events.

    Args:
        data_dir: Explicit data dir path. If empty, falls back to family-based path.
        family_id: Family identifier used to construct the sidecar-watched data dir.
    """
    if data_dir:
        rule_dir = Path(data_dir) / "portfolio_rules"
    elif family_id:
        rule_dir = Path(f"strategies/{family_id}/instrumentation/data/portfolio_rules")
    else:
        rule_dir = Path("instrumentation/data/portfolio_rules")

    def _log_rule(event: dict) -> None:
        try:
            rule_dir.mkdir(parents=True, exist_ok=True)
            payload = dict(event)
            payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            if family_id:
                payload.setdefault("family_id", family_id)
            current_lineage = _resolve_lineage(lineage)
            payload = _overlay_lineage_fields(payload, current_lineage)
            event = enrich_payload(
                payload,
                lineage=current_lineage,
                event_type="portfolio_rule_check",
                scope="portfolio",
            )
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = rule_dir / f"rules_{today}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    return _log_rule


def _make_intent_denial_logger(data_dir: str = "", family_id: str = "", lineage=None) -> Callable:
    """Create a JSONL-writing callback for intent-denial forensic events.

    Counterpart to :func:`_make_portfolio_rule_logger`. The OMS service
    fires this on every DENIED intent so operators can audit gateway-level
    rejections (heat cap, daily stop, session block, account gate, etc.)
    that the existing ``portfolio_rules/`` sidecar stream does not cover.
    """
    if data_dir:
        denial_dir = Path(data_dir) / "risk_denials"
    elif family_id:
        denial_dir = Path(f"strategies/{family_id}/instrumentation/data/risk_denials")
    else:
        denial_dir = Path("instrumentation/data/risk_denials")

    def _log_denial(event: dict) -> None:
        try:
            denial_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **dict(event),
            }
            if family_id:
                payload.setdefault("family_id", family_id)
            current_lineage = _resolve_lineage(lineage)
            payload = _overlay_lineage_fields(payload, current_lineage)
            event = enrich_payload(
                payload,
                lineage=current_lineage,
                event_type="risk_denial",
                scope="oms",
            )
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            path = denial_dir / f"denials_{today}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception:
            pass

    return _log_denial


# ---------------------------------------------------------------------------
# build_oms_service  — unified single-strategy factory
# ---------------------------------------------------------------------------

def _default_instrumentation_data_dir(data_dir: str = "", family_id: str = "") -> str:
    if data_dir:
        return data_dir
    if family_id and family_id != "unknown":
        repo_root = Path(__file__).resolve().parents[3]
        return str(repo_root / "strategies" / family_id / "instrumentation" / "data")
    return ""


def _risk_state_payload(state) -> dict:
    from libs.oms.instrumentation._shared import plain

    return plain(state) if state is not None else {}


def _order_payload(order, *, family_id: str = "") -> dict:
    from libs.oms.instrumentation._shared import account_alias_for, plain

    instrument = getattr(order, "instrument", None)
    risk_context = getattr(order, "risk_context", None)
    account_alias = account_alias_for(getattr(order, "account_id", ""))
    return {
        "oms_order_id": getattr(order, "oms_order_id", ""),
        "client_order_id": getattr(order, "client_order_id", ""),
        "broker_order_id": getattr(order, "broker_order_id", ""),
        "perm_id": getattr(order, "perm_id", ""),
        "strategy_id": getattr(order, "strategy_id", ""),
        "family_id": family_id,
        "account_alias": account_alias,
        "symbol": getattr(instrument, "symbol", "") if instrument else "",
        "root": getattr(instrument, "root", "") if instrument else "",
        "asset_class": getattr(instrument, "sec_type", "") if instrument else "",
        "side": getattr(getattr(order, "side", None), "value", getattr(order, "side", "")),
        "order_type": getattr(getattr(order, "order_type", None), "value", getattr(order, "order_type", "")),
        "role": getattr(getattr(order, "role", None), "value", getattr(order, "role", "")),
        "status": getattr(getattr(order, "status", None), "value", getattr(order, "status", "")),
        "requested_qty": getattr(order, "qty", 0),
        "filled_qty": getattr(order, "filled_qty", 0.0),
        "remaining_qty": getattr(order, "remaining_qty", 0.0),
        "avg_fill_price": getattr(order, "avg_fill_price", 0.0),
        "limit_price": getattr(order, "limit_price", None),
        "stop_price": getattr(order, "stop_price", None),
        "risk_context": plain(risk_context) if risk_context is not None else {},
    }


def _allocation_targets_payload(
    allocation_targets: Optional[dict],
    *,
    family_id: str = "",
    strategy_ids: Optional[list[str]] = None,
) -> dict:
    from libs.oms.instrumentation._shared import plain

    ids = [str(strategy_id) for strategy_id in list(strategy_ids or []) if str(strategy_id)]
    targets = plain(dict(allocation_targets or {}))
    families = dict(targets.get("families", targets.get("family_target_weights", {})) or {})
    strategies = dict(targets.get("strategies", targets.get("strategy_target_weights", {})) or {})
    if family_id and not families:
        families = {family_id: 1.0}
    if ids and not strategies:
        each = 1.0 / len(ids)
        strategies = {strategy_id: each for strategy_id in ids}
        targets.setdefault("source", "equal_family_fallback")
    return {
        **targets,
        "families": families,
        "strategies": strategies,
        "target_scope": targets.get("target_scope") or "family",
        "active_strategy_ids": targets.get("active_strategy_ids") or ids,
    }


def _finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _ref_value(ref) -> float:
    if ref is None:
        return 0.0
    try:
        if isinstance(ref, (list, tuple)):
            return _finite_float(ref[0] if ref else 0.0)
    except Exception:
        return 0.0
    return _finite_float(ref)


def _make_account_state_provider(
    *,
    get_current_equity: Optional[Callable[[], float]] = None,
    live_equity=None,
    paper_equity=None,
    fallback_equity: float = 0.0,
    family_id: str = "",
    account_id: str = "",
    paper_equity_scope: str = "",
) -> Callable[[], dict]:
    """Return the authoritative NAV/account state used by lifecycle snapshots."""
    from libs.oms.instrumentation._shared import account_alias_for

    account_alias = account_alias_for(account_id)

    def _provider() -> dict:
        source = ""
        equity = _ref_value(live_equity)
        if equity > 0:
            source = "live_equity_ref"
        if equity <= 0:
            equity = _ref_value(paper_equity)
            if equity > 0:
                source = "paper_equity_ref"
        if equity <= 0 and get_current_equity is not None:
            try:
                equity = _finite_float(get_current_equity())
            except Exception:
                equity = 0.0
            if equity > 0:
                source = "current_equity_provider"
        if equity <= 0:
            equity = _finite_float(fallback_equity)
            if equity > 0:
                source = "fallback_initial_equity"

        state = {
            "equity": equity,
            "net_liquidation": equity,
            "raw_nav": equity,
            "allocated_nav": equity,
            "currency": "USD",
            "source": source or "unknown",
            "family_id": family_id,
            "account_alias": account_alias,
        }
        if paper_equity_scope:
            state["paper_equity_scope"] = paper_equity_scope
        return state

    return _provider


def _account_state_snapshot(provider: Optional[Callable[[], dict]]) -> dict:
    if provider is None:
        return {}
    try:
        state = provider()
    except Exception as exc:
        logger.warning("Account state provider failed: %s", exc)
        return {}
    return dict(state or {}) if isinstance(state, dict) else {}


def _nav_from_account_state(account_state: dict, key: str = "raw_nav") -> float:
    value = _finite_float(account_state.get(key))
    if value > 0:
        return value
    for fallback_key in ("equity", "net_liquidation", "allocated_nav", "raw_nav"):
        value = _finite_float(account_state.get(fallback_key))
        if value > 0:
            return value
    return 0.0


def _position_payload(
    position: Position,
    *,
    family_id: str = "",
    portfolio_id: str = "",
    mark_price: float = 0.0,
) -> dict:
    from libs.oms.instrumentation._shared import account_alias_for

    return {
        "account_alias": account_alias_for(position.account_id),
        "portfolio_id": portfolio_id,
        "family_id": family_id,
        "strategy_id": position.strategy_id,
        "symbol": position.instrument_symbol,
        "instrument_symbol": position.instrument_symbol,
        "qty": position.net_qty,
        "net_qty": position.net_qty,
        "avg_price": position.avg_price,
        "mark_price": mark_price or position.avg_price,
        "realized_pnl": position.realized_pnl,
        "unrealized_pnl": position.unrealized_pnl,
        "open_risk_dollars": position.open_risk_dollars,
        "open_risk_R": position.open_risk_R,
        "last_update_at": position.last_update_at.isoformat() if position.last_update_at else "",
    }


def _resolve_lifecycle_lineage(lineage):
    if callable(lineage):
        try:
            return lineage()
        except Exception:
            logger.debug("Failed to resolve lifecycle lineage", exc_info=True)
            return None
    return lineage


def _make_fill_lifecycle_writer(data_dir: str = "", lineage=None) -> Callable:
    def _write(payload: dict) -> None:
        if not data_dir:
            return
        try:
            from libs.oms.instrumentation.lifecycle import write_fill_lifecycle_snapshots

            write_fill_lifecycle_snapshots(data_dir, _resolve_lifecycle_lineage(lineage), payload)
        except Exception as exc:
            logger.warning("Fill lifecycle instrumentation failed: %s", exc)

    return _write


def _make_reconciliation_lifecycle_writer(data_dir: str = "", lineage=None) -> Callable:
    def _write(payload: dict) -> None:
        if not data_dir:
            return
        try:
            from libs.oms.instrumentation.lifecycle import write_reconciliation_lifecycle_event

            write_reconciliation_lifecycle_event(data_dir, _resolve_lifecycle_lineage(lineage), payload)
        except Exception as exc:
            logger.warning("Reconciliation lifecycle instrumentation failed: %s", exc)

    return _write


async def build_oms_service(
    adapter,  # IBKRExecutionAdapter
    strategy_id: str,
    unit_risk_dollars: float,
    daily_stop_R: float = 2.0,
    heat_cap_R: float = 1.25,
    portfolio_daily_stop_R: float = 3.0,
    portfolio_weekly_stop_R: float = 12.0,
    calendar: Optional[EventCalendar] = None,
    db_pool: Optional[asyncpg.Pool] = None,
    market_calendar: Optional["MarketCalendar"] = None,
    family_id: str = "unknown",
    account_gate: Optional[object] = None,
    portfolio_rules_config=None,  # Optional[PortfolioRulesConfig]
    get_current_equity: Optional[callable] = None,
    paper_equity_pool: Optional[asyncpg.Pool] = None,
    paper_equity_scope: str = "paper",
    paper_initial_equity: float = 10_000.0,
    paper_equity_ref: Optional[list] = None,
    portfolio_unit_risk_dollars: Optional[float] = None,
    strategy_heat_cap_R: float = 0.0,
    halt_trading=None,  # Optional async callback
    recon_interval_s: float = 120.0,
    live_equity: Optional[list] = None,  # mutable [float] ref for live equity updates
    family_strategy_ids: Optional[list[str]] = None,
    instrumentation_data_dir: str = "",
    event_clock: Optional[Callable[[], datetime]] = None,
    repository: Optional[object] = None,
    allocation_targets: Optional[dict] = None,
    portfolio_id: str = "",
    account_alias: str = "",
    portfolio_config=None,
    strategy_manifest=None,
    parameter_set=None,
    reconciliation_authoritative: bool = True,
    reconciliation_owner_id: str = "",
    reconciliation_authority: Optional[object] = None,
    enable_reconciliation_authority_lease: Optional[bool] = None,
) -> OMSService:
    """Build a fully wired OMS service.

    Union of all parameters from swing, momentum, and stock families.
    When optional params are None, their features are simply not wired.

    Args:
        adapter: IBKRExecutionAdapter instance
        strategy_id: Strategy identifier
        unit_risk_dollars: Dollar risk per 1R unit for position sizing
        daily_stop_R: Strategy daily stop in R units
        heat_cap_R: Portfolio heat cap in R units
        portfolio_daily_stop_R: Portfolio daily stop in R units
        portfolio_weekly_stop_R: Portfolio weekly stop in R units
        calendar: Optional event calendar for blackouts
        db_pool: Optional asyncpg pool for PostgreSQL persistence.
                 If provided, uses OMSRepository; otherwise InMemoryRepository.
        market_calendar: Optional MarketCalendar for session-aware gating.
        family_id: Family identifier for per-family portfolio risk tracking.
        account_gate: Optional cross-family AccountRiskGate instance.
                      Threaded through to RiskGateway.__init__() (2B.13 fix).
        portfolio_rules_config: Optional PortfolioRulesConfig for cross-strategy rules.
        get_current_equity: Callback returning current equity (for portfolio rules).
        paper_equity_pool: Optional asyncpg pool for paper-trading equity tracking.
        paper_equity_scope: Scope key for paper-equity persistence.
        paper_initial_equity: Seed equity for paper-equity persistence.
        paper_equity_ref: Optional shared mutable [equity] for multi-OMS family scopes.
        portfolio_unit_risk_dollars: Optional shared portfolio 1R dollar base.
        strategy_heat_cap_R: Optional per-strategy heat ceiling in strategy R.
        halt_trading: Optional async callback to halt trading on critical errors.
        recon_interval_s: Reconciliation interval in seconds.

    Returns:
        Fully initialized OMSService ready for start()
    """
    instrumentation_data_dir = _default_instrumentation_data_dir(
        instrumentation_data_dir, family_id
    )

    # Event bus
    bus = EventBus(clock=event_clock)

    # Repository: use PostgreSQL if pool provided, otherwise in-memory
    if repository is not None:
        repo = repository
        logger.info(f"Using provided repository for strategy {strategy_id}")
    elif db_pool is not None:
        repo = OMSRepository(db_pool)
        logger.info(f"Using PostgreSQL repository for strategy {strategy_id}")
    else:
        repo = InMemoryRepository()
        logger.info(f"Using in-memory repository for strategy {strategy_id}")

    # Risk configuration
    strat_cfg = StrategyRiskConfig(
        strategy_id=strategy_id,
        daily_stop_R=daily_stop_R,
        unit_risk_dollars=unit_risk_dollars,
        max_heat_R=strategy_heat_cap_R,
    )
    try:
        portfolio_urd = float(portfolio_unit_risk_dollars)
        if portfolio_urd <= 0:
            portfolio_urd = unit_risk_dollars
    except (TypeError, ValueError):
        portfolio_urd = unit_risk_dollars
    sizing_equity = None
    if live_equity is not None and live_equity and live_equity[0] is not None:
        sizing_equity = live_equity[0]
    elif get_current_equity is not None:
        try:
            sizing_equity = get_current_equity()
        except Exception:
            sizing_equity = None
    try:
        sizing_equity = float(sizing_equity) if sizing_equity is not None else None
    except (TypeError, ValueError):
        sizing_equity = None
    if sizing_equity and sizing_equity > 0:
        strat_cfg.unit_risk_pct = unit_risk_dollars / sizing_equity
    risk_config = RiskConfig(
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=portfolio_daily_stop_R,
        portfolio_weekly_stop_R=portfolio_weekly_stop_R,
        strategy_configs={strategy_id: strat_cfg},
        portfolio_urd=portfolio_urd,
    )

    def _current_portfolio_urd() -> float:
        try:
            current = float(getattr(risk_config, "portfolio_urd", 0.0) or 0.0)
        except (TypeError, ValueError):
            current = 0.0
        return current if current > 0 else portfolio_urd

    # Paper equity tracker (paper mode only)
    if paper_equity_ref is not None:
        if not paper_equity_ref:
            paper_equity_ref.append(None)
        _paper_equity = paper_equity_ref
    else:
        _paper_equity: list[Optional[float]] = [None]
    if paper_equity_pool is not None:
        from libs.persistence.paper_equity import load_paper_equity
        _paper_equity[0] = await load_paper_equity(
            paper_equity_pool,
            account_scope=paper_equity_scope,
            initial_equity=paper_initial_equity,
        )
        get_current_equity = lambda: _paper_equity[0]
        # Derive actual risk pct from caller's unit_risk_dollars / current equity
        if _paper_equity[0] > 0:
            strat_cfg.unit_risk_pct = unit_risk_dollars / _paper_equity[0]
        logger.info("Paper equity mode enabled: $%.2f (risk_pct=%.4f)", _paper_equity[0], strat_cfg.unit_risk_pct)
    try:
        equity_for_portfolio_pct = float(
            _paper_equity[0] if _paper_equity[0] is not None else sizing_equity
        )
    except (TypeError, ValueError):
        equity_for_portfolio_pct = 0.0
    portfolio_unit_risk_pct = (
        portfolio_urd / equity_for_portfolio_pct
        if equity_for_portfolio_pct > 0 and portfolio_urd > 0 else 0.0
    )

    # Event calendar (empty if not provided)
    if calendar is None:
        calendar = EventCalendar()

    # Risk state providers
    strategy_risk_states: dict[str, StrategyRiskState] = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=_trade_date_for())
    pg_store = PgStore(db_pool) if db_pool is not None else None
    # Track open positions per (strategy, symbol) for exit P&L computation.
    open_positions: dict[tuple[str, str], dict] = {}

    async def _sync_strategy_risk_from_repo(state: StrategyRiskState) -> None:
        positions = await repo.get_positions(state.strategy_id)
        if not positions:
            state.daily_realized_pnl = 0.0
            state.daily_realized_R = 0.0
            state.open_risk_dollars = 0.0
            state.open_risk_R = 0.0
            return

        latest_ts = datetime.min.replace(tzinfo=timezone.utc)
        latest_pos = None
        for pos in positions:
            pos_ts = pos.last_update_at or latest_ts
            if pos_ts >= latest_ts:
                latest_ts = pos_ts
                latest_pos = pos

        if latest_pos and latest_pos.last_update_at and latest_pos.last_update_at.date() == state.trade_date:
            state.daily_realized_pnl = latest_pos.realized_pnl
            state.daily_realized_R = (
                latest_pos.realized_pnl / unit_risk_dollars if unit_risk_dollars > 0 else 0.0
            )
        else:
            state.daily_realized_pnl = 0.0
            state.daily_realized_R = 0.0

        open_positions_for_strategy = [p for p in positions if p.net_qty != 0]
        state.open_risk_dollars = sum(p.open_risk_dollars for p in open_positions_for_strategy)
        state.open_risk_R = sum(p.open_risk_R for p in open_positions_for_strategy)

    async def _load_strategy_risk_from_store(state: StrategyRiskState) -> RiskDailyStrategyRow | None:
        if pg_store is None:
            return None
        row = await pg_store.get_risk_daily_strategy(state.strategy_id, state.trade_date)
        if row is None:
            return None
        state.daily_realized_R = float(row.daily_realized_r)
        if row.daily_realized_usd is not None:
            state.daily_realized_pnl = float(row.daily_realized_usd)
        state.halted = row.halted
        state.halt_reason = row.halt_reason or ""
        return row

    async def _persist_strategy_risk_state(
        state: StrategyRiskState,
        as_of: datetime,
    ) -> None:
        if pg_store is None:
            return
        existing_row = await pg_store.get_risk_daily_strategy(state.strategy_id, state.trade_date)
        await pg_store.upsert_risk_daily_strategy(
            _build_strategy_daily_row(state, existing_row, as_of, family_id=family_id)
        )

    async def _sync_portfolio_open_risk_from_repo() -> None:
        positions = await repo.get_positions_for_strategies(_family_sids)
        open_pos = [pos for pos in positions if pos.net_qty != 0]
        portfolio_risk_state.open_risk_dollars = sum(
            pos.open_risk_dollars for pos in open_pos
        )
        portfolio_risk_state.open_risk_R = (
            portfolio_risk_state.open_risk_dollars / _current_portfolio_urd()
            if _current_portfolio_urd() > 0 else 0.0
        )

    _family_sids = list(family_strategy_ids) if family_strategy_ids else [strategy_id]
    _allocation_targets = _allocation_targets_payload(
        allocation_targets,
        family_id=family_id,
        strategy_ids=_family_sids,
    )

    async def _load_portfolio_risk_from_store(trade_day: date) -> None:
        if pg_store is None:
            return

        today_rows = await pg_store.get_risk_daily_strategies_for_date(trade_day, strategy_ids=_family_sids)
        today_portfolio_row = await pg_store.get_risk_daily_portfolio(trade_day, family_id=family_id)

        if today_rows:
            daily_realized_r, daily_realized_usd, strategy_daily_pnl = _aggregate_strategy_daily_rows(today_rows)
            current_portfolio_urd = _current_portfolio_urd()
            portfolio_risk_state.daily_realized_R = (
                daily_realized_usd / current_portfolio_urd
                if current_portfolio_urd > 0 and daily_realized_usd != 0.0
                else daily_realized_r
            )
            portfolio_risk_state.daily_realized_pnl = daily_realized_usd
            portfolio_risk_state.strategy_daily_pnl = strategy_daily_pnl
        elif today_portfolio_row is not None:
            portfolio_risk_state.daily_realized_R = float(today_portfolio_row.daily_realized_r)
            portfolio_risk_state.daily_realized_pnl = float(today_portfolio_row.daily_realized_usd or 0)
            portfolio_risk_state.strategy_daily_pnl = {}
        else:
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.strategy_daily_pnl = {}

        totals = await pg_store.get_risk_daily_strategy_totals(_week_start(trade_day), trade_day, strategy_ids=_family_sids)
        portfolio_risk_state.weekly_realized_pnl = float(totals["total_usd"])
        current_portfolio_urd = _current_portfolio_urd()
        portfolio_risk_state.weekly_realized_R = (
            portfolio_risk_state.weekly_realized_pnl / current_portfolio_urd
            if current_portfolio_urd > 0 and portfolio_risk_state.weekly_realized_pnl != 0.0
            else float(totals["total_r"])
        )
        if (
            today_portfolio_row is not None
            and not today_rows
            and portfolio_risk_state.weekly_realized_R == 0.0
        ):
            portfolio_risk_state.weekly_realized_R = float(today_portfolio_row.daily_realized_r)
            portfolio_risk_state.weekly_realized_pnl = float(today_portfolio_row.daily_realized_usd or 0)

        if today_portfolio_row is not None:
            portfolio_risk_state.halted = today_portfolio_row.halted
            portfolio_risk_state.halt_reason = today_portfolio_row.halt_reason or ""

    async def _persist_portfolio_risk_state(as_of: datetime) -> None:
        if pg_store is None:
            return
        await _sync_portfolio_open_risk_from_repo()
        trade_day = portfolio_risk_state.trade_date
        daily_rows = await pg_store.get_risk_daily_strategies_for_date(trade_day, strategy_ids=_family_sids)
        existing_row = await pg_store.get_risk_daily_portfolio(trade_day, family_id=family_id)
        await pg_store.upsert_risk_daily_portfolio(
            _build_portfolio_daily_row(
                trade_date=trade_day,
                daily_rows=daily_rows,
                open_risk_r=portfolio_risk_state.open_risk_R,
                existing_row=existing_row,
                halted=portfolio_risk_state.halted,
                halt_reason=portfolio_risk_state.halt_reason,
                as_of=as_of,
                family_id=family_id,
                portfolio_urd=_current_portfolio_urd(),
            )
        )

    async def _halt_trading(reason: str) -> None:
        halt_ts = datetime.now(timezone.utc)
        trade_day = _trade_date_for()
        portfolio_risk_state.trade_date = trade_day
        portfolio_risk_state.halted = True
        portfolio_risk_state.halt_reason = reason
        strat_state = strategy_risk_states.setdefault(
            strategy_id,
            StrategyRiskState(strategy_id=strategy_id, trade_date=trade_day),
        )
        strat_state.trade_date = trade_day
        strat_state.halted = True
        strat_state.halt_reason = reason
        if pg_store is not None:
            await pg_store.halt_strategy(strategy_id, reason, trade_day)
            await pg_store.halt_portfolio(reason, trade_day, family_id=family_id)
        await _persist_strategy_risk_state(strat_state, halt_ts)
        await _persist_portfolio_risk_state(halt_ts)

    async def get_strategy_risk(sid: str) -> StrategyRiskState:
        # L1 fix: reset risk state at date boundary
        today = _trade_date_for()
        if sid in strategy_risk_states:
            existing = strategy_risk_states[sid]
            if existing.trade_date != today:
                logger.info(f"Date boundary detected for {sid}: resetting daily risk state")
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=today,
                    open_risk_dollars=existing.open_risk_dollars,
                    open_risk_R=existing.open_risk_R,
                )
        if sid not in strategy_risk_states:
            strategy_risk_states[sid] = StrategyRiskState(strategy_id=sid, trade_date=today)
        state = strategy_risk_states[sid]
        await _sync_strategy_risk_from_repo(state)
        existing_row = await _load_strategy_risk_from_store(state)
        if pg_store is not None and existing_row is None:
            await _persist_strategy_risk_state(state, datetime.now(timezone.utc))
        return state

    async def get_portfolio_risk() -> PortfolioRiskState:
        # L1 fix: reset portfolio risk state at date boundary
        today = _trade_date_for()
        if portfolio_risk_state.trade_date != today:
            logger.info("Date boundary detected: resetting portfolio daily risk state")
            # Weekly reset on Monday (weekday 0)
            if today.weekday() == 0:
                logger.info("Monday weekly reset: weekly_R %.2f → 0.0",
                            portfolio_risk_state.weekly_realized_R)
                portfolio_risk_state.weekly_realized_pnl = 0.0
                portfolio_risk_state.weekly_realized_R = 0.0
            portfolio_risk_state.trade_date = today
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.strategy_daily_pnl = {}
            portfolio_risk_state.halted = False
            portfolio_risk_state.halt_reason = ""
        await _sync_portfolio_open_risk_from_repo()
        await _load_portfolio_risk_from_store(today)
        portfolio_risk_state.pending_entry_risk_R = await repo.get_pending_entry_risk_R_for_strategies(
            _family_sids, _current_portfolio_urd()
        )
        return portfolio_risk_state

    async def get_working_order_count(sid: str) -> int:
        return await repo.count_working_orders(sid)

    # Fill processor for OMS order state updates
    fill_proc = FillProcessor(repo)
    _adapter_account = getattr(adapter, "_account", "") or ""
    try:
        from libs.oms.instrumentation._shared import account_alias_for

        lineage_account_alias = account_alias or account_alias_for(_adapter_account) or str(paper_equity_scope or "")
    except Exception:
        lineage_account_alias = account_alias or str(paper_equity_scope or "")
    lineage_portfolio_id = portfolio_id or "paper_default"
    oms_effective_config = {
        "strategy_id": strategy_id,
        "unit_risk_dollars": unit_risk_dollars,
        "daily_stop_R": daily_stop_R,
        "heat_cap_R": heat_cap_R,
        "portfolio_daily_stop_R": portfolio_daily_stop_R,
        "portfolio_weekly_stop_R": portfolio_weekly_stop_R,
        "family_id": family_id,
        "portfolio_id": lineage_portfolio_id,
        "account_alias": lineage_account_alias,
    }
    oms_parameter_set = parameter_set or {
        "effective_strategy_config": oms_effective_config,
        "allocation_targets": _allocation_targets,
    }
    oms_lineage = _make_oms_lineage(
        bot_id=f"{family_id}_oms" if family_id else "oms",
        strategy_id=strategy_id,
        family_id=family_id,
        portfolio_id=lineage_portfolio_id,
        account_alias=lineage_account_alias,
        portfolio_config=portfolio_config,
        portfolio_rules_config=portfolio_rules_config,
        strategy_manifest=strategy_manifest,
        effective_strategy_config=oms_effective_config,
        parameter_set=oms_parameter_set,
    )
    oms_lineage_state = {"value": oms_lineage}

    def _current_oms_lineage(strategy_id: str = ""):
        return oms_lineage_state["value"]

    bus._current_oms_lineage = _current_oms_lineage

    def _refresh_oms_lineage(rules_config) -> None:
        current = oms_lineage_state["value"]
        refreshed = _make_oms_lineage(
            bot_id=f"{family_id}_oms" if family_id else "oms",
            strategy_id=strategy_id,
            family_id=family_id,
            portfolio_id=lineage_portfolio_id,
            account_alias=lineage_account_alias,
            portfolio_config=portfolio_config,
            portfolio_rules_config=rules_config,
            strategy_manifest=strategy_manifest,
            effective_strategy_config=oms_effective_config,
            parameter_set=oms_parameter_set,
        )
        oms_lineage_state["value"] = replace(
            refreshed,
            deployment_id=current.deployment_id or refreshed.deployment_id,
            trace_id=current.trace_id or refreshed.trace_id,
        )

    # Portfolio rules checker (cross-strategy coordination via shared store)
    portfolio_checker = None
    if portfolio_rules_config is not None:
        from ..risk.portfolio_rules import PortfolioRuleChecker

        rules_store = pg_store if pg_store is not None else repo
        portfolio_checker = PortfolioRuleChecker(
            config=portfolio_rules_config,
            get_strategy_signal=rules_store.get_strategy_signal,
            get_directional_risk_R=rules_store.get_directional_risk_R,
            get_current_equity=get_current_equity or (lambda: 10_000.0),
            on_rule_event=_make_portfolio_rule_logger(
                data_dir=instrumentation_data_dir,
                family_id=family_id,
                lineage=_current_oms_lineage,
            ),
            get_directional_risk_R_for_strategies=rules_store.get_directional_risk_R_for_strategies,
            get_sibling_positions_for_symbol=rules_store.get_sibling_positions_for_symbol,
            get_family_aggregate_mnq_eq=rules_store.get_family_aggregate_mnq_eq,
            get_directional_risk_dollars_for_strategies=rules_store.get_directional_risk_dollars_for_strategies,
            get_open_position_count_for_strategies=rules_store.get_open_position_count_for_strategies,
            get_symbol_open_risk_dollars_for_strategies=rules_store.get_symbol_open_risk_dollars_for_strategies,
            get_symbols_open_risk_dollars_for_strategies=rules_store.get_symbols_open_risk_dollars_for_strategies,
            get_active_risk_dollars_for_strategies=rules_store.get_active_risk_dollars_for_strategies,
            get_completed_trade_counts_for_strategies=rules_store.get_completed_trade_counts_for_strategies,
            get_recent_strategy_r_multiples=rules_store.get_recent_strategy_r_multiples,
            on_config_update=_refresh_oms_lineage,
        )
        logger.info("Portfolio rules enabled for %s", strategy_id)

    # Risk gateway — account_gate threaded through (2B.13 fix)
    risk_gateway = RiskGateway(
        config=risk_config,
        calendar=calendar,
        get_strategy_risk=get_strategy_risk,
        get_portfolio_risk=get_portfolio_risk,
        get_working_order_count=get_working_order_count,
        portfolio_checker=portfolio_checker,
        market_calendar=market_calendar,
        account_gate=account_gate,
        family_id=family_id,
    )
    risk_gateway._current_oms_lineage = _current_oms_lineage
    await upsert_active_runtime_config(
        db_pool,
        ActiveRuntimeConfigRecord(
            account_id=_adapter_account,
            config_scope="oms",
            scope_id=f"{family_id}:{strategy_id}",
            runtime_env=get_environment(),
            payload={
                "account_id": _adapter_account,
                "family_id": family_id,
                "strategy_id": strategy_id,
                "heat_cap_R": float(heat_cap_R),
                "portfolio_daily_stop_R": float(portfolio_daily_stop_R),
                "portfolio_weekly_stop_R": float(portfolio_weekly_stop_R),
                "portfolio_urd": float(portfolio_urd),
                "unit_risk_dollars": float(unit_risk_dollars),
                "strategy_daily_stop_R": float(daily_stop_R),
                "strategy_heat_cap_R": float(strategy_heat_cap_R),
                "account_gate_enabled": account_gate is not None,
                "source": "oms factory effective risk gateway config",
            },
            expires_at=active_config_expiry(),
        ),
    )

    async def _pre_submit_recheck(order, conn=None) -> str | None:
        reserved_risk_dollars = float(
            getattr(getattr(order, "risk_context", None), "risk_dollars", 0.0)
            or 0.0
        )
        reserved_entry_risk_R = (
            reserved_risk_dollars / portfolio_urd if portfolio_urd > 0 else 0.0
        )
        denial = await risk_gateway.check_entry(
            order,
            skip_account_gate=True,
            reserved_entry_risk_R=reserved_entry_risk_R,
        )
        if denial:
            return denial
        return await risk_gateway.check_account_gate(
            order,
            conn=conn,
            reserved_entry_risk_dollars=reserved_risk_dollars,
        )

    # Execution router
    router = ExecutionRouter(
        adapter,
        repo,
        bus=bus,
        pre_submit_recheck=_pre_submit_recheck,
        claimant_id=f"{family_id}:{strategy_id}",
    )

    # Intent handler
    # OMS-7: pass adapter's configured IB account so the handler can stamp
    # account_id on swing/momentum orders (which leave it blank by default).
    handler = IntentHandler(
        risk_gateway, router, repo, bus,
        default_account_id=_adapter_account,
    )
    account_state_provider = _make_account_state_provider(
        get_current_equity=get_current_equity,
        live_equity=live_equity,
        paper_equity=_paper_equity,
        fallback_equity=paper_initial_equity,
        family_id=family_id,
        account_id=_adapter_account,
        paper_equity_scope=paper_equity_scope,
    )

    # Reconciler — wire halt_trading if provided, else use internal _halt_trading
    _halt_cb = halt_trading if halt_trading is not None else _halt_trading
    _reconciliation_authority = _resolve_reconciliation_authority(
        db_pool=db_pool,
        provided_authority=reconciliation_authority,
        enable_lease=enable_reconciliation_authority_lease,
    )
    reconciler = ReconciliationOrchestrator(
        adapter,
        repo,
        bus,
        halt_trading=_halt_cb,
        fill_processor=fill_proc,
        offline_fill_importer=lambda oms_id, exec_report: _import_fill_through_adapter_callback(
            adapter, repo, oms_id, exec_report,
        ),
        lifecycle_event_writer=_make_reconciliation_lifecycle_writer(
            instrumentation_data_dir, _current_oms_lineage,
        ),
        family_id=family_id,
        owner_id=reconciliation_owner_id or strategy_id,
        reconciliation_authoritative=reconciliation_authoritative,
        authority=_reconciliation_authority,
    )

    # C4 fix: Order timeout monitor for stuck ROUTED / CANCEL_REQUESTED states
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    timeout_monitor = OrderTimeoutMonitor(repo, bus, router)

    # OMS-5: Hydrate the in-memory open_positions cache from DB before any
    # fill callback can fire. Without this, an exit fill arriving after a
    # restart would land in the `pos is None` branch and silently lose the
    # realized P&L update.
    if db_pool is not None or repository is not None:
        await _hydrate_open_positions_from_repo(
            open_positions=open_positions,
            repo=repo,
            strategy_ids=[strategy_id],
            unit_risk_dollars_for=lambda _sid: unit_risk_dollars,
            portfolio_unit_risk_dollars=portfolio_urd,
            strict=db_pool is not None,
        )

    # Wire adapter callbacks to bus (with risk state updates)
    _wire_adapter_callbacks(
        adapter, bus, repo, fill_proc, router,
        strategy_risk_states, portfolio_risk_state, unit_risk_dollars, open_positions,
        persist_strategy_risk_state=_persist_strategy_risk_state,
        persist_portfolio_risk_state=_persist_portfolio_risk_state,
        paper_equity=_paper_equity,
        paper_equity_pool=paper_equity_pool,
        strat_cfg=strat_cfg,
        risk_config=risk_config,
        portfolio_urd=portfolio_urd,
        portfolio_unit_risk_pct=portfolio_unit_risk_pct,
        db_pool=db_pool,
        live_equity=live_equity,
        paper_equity_scope=paper_equity_scope,
        paper_initial_equity=paper_initial_equity,
        halt_trading=_halt_cb,
        family_id=family_id,
        family_strategy_ids=_family_sids,
        portfolio_id=getattr(oms_lineage, "portfolio_id", ""),
        allocation_targets=_allocation_targets,
        account_state_provider=account_state_provider,
        on_position_update=_make_fill_lifecycle_writer(
            instrumentation_data_dir, _current_oms_lineage,
        ),
    )

    # Build OMS service
    oms = OMSService(
        intent_handler=handler,
        bus=bus,
        reconciler=reconciler,
        router=router,
        recon_interval_s=recon_interval_s,
        timeout_monitor=timeout_monitor,
        get_portfolio_risk=get_portfolio_risk,
        get_strategy_risk=get_strategy_risk,
        on_intent_denied=_make_intent_denial_logger(
            data_dir=instrumentation_data_dir,
            family_id=family_id,
            lineage=_current_oms_lineage,
        ),
    )
    oms._portfolio_checker = portfolio_checker  # for coordinator regime updates
    oms._oms_lineage = _current_oms_lineage
    oms._refresh_oms_lineage = _refresh_oms_lineage
    oms._portfolio_risk_state = portfolio_risk_state  # for coordinator heartbeat queries
    oms._strategy_risk_states = strategy_risk_states  # for per-strategy heartbeat metrics
    oms._oms_repo = repo  # instrumentation daily reconciliation hook
    oms._family_strategy_ids = list(_family_sids)
    oms._family_id = family_id
    oms._allocation_targets = _allocation_targets
    oms._account_state_provider = account_state_provider

    if pg_store is not None or repository is not None:
        seed_state = strategy_risk_states.setdefault(
            strategy_id,
            StrategyRiskState(strategy_id=strategy_id, trade_date=_trade_date_for()),
        )
        await _sync_strategy_risk_from_repo(seed_state)
        seed_ts = datetime.now(timezone.utc)
        await _sync_portfolio_open_risk_from_repo()
        if pg_store is not None:
            await _load_strategy_risk_from_store(seed_state)
            await _persist_strategy_risk_state(seed_state, seed_ts)
            await _load_portfolio_risk_from_store(seed_state.trade_date)
            await _persist_portfolio_risk_state(seed_ts)

    logger.info(f"OMS factory built for strategy {strategy_id}")
    return oms


# ---------------------------------------------------------------------------
# build_multi_strategy_oms  — swing's coordinator pattern
# ---------------------------------------------------------------------------

async def build_multi_strategy_oms(
    adapter,
    strategies: list[dict],
    heat_cap_R: float = 1.5,
    portfolio_daily_stop_R: float = 3.0,
    portfolio_weekly_stop_R: float = 0.0,
    calendar: Optional[EventCalendar] = None,
    recon_interval_s: float = 120.0,
    db_pool: Optional[asyncpg.Pool] = None,
    market_calendar: Optional["MarketCalendar"] = None,
    family_id: str = "unknown",
    account_gate: Optional[object] = None,
    halt_trading: Optional[Callable] = None,
    portfolio_rules_config=None,
    portfolio_unit_risk_dollars: Optional[float] = None,
    get_current_equity: Optional[Callable[[], float]] = None,
    live_equity: Optional[list] = None,  # mutable [float] ref for live equity updates
    paper_equity_pool: Optional[asyncpg.Pool] = None,
    paper_equity_scope: str = "paper",
    paper_initial_equity: float = 10_000.0,
    instrumentation_data_dir: str = "",
    event_clock: Optional[Callable[[], datetime]] = None,
    repository: Optional[object] = None,
    allocation_targets: Optional[dict] = None,
    portfolio_id: str = "",
    account_alias: str = "",
    portfolio_config=None,
    strategy_manifest=None,
    strategy_manifests=None,
    parameter_set=None,
    parameter_sets=None,
    reconciliation_authoritative: bool = True,
    reconciliation_owner_id: str = "",
    reconciliation_authority: Optional[object] = None,
    enable_reconciliation_authority_lease: Optional[bool] = None,
) -> tuple["OMSService", "StrategyCoordinator"]:
    """Build a shared OMS service for multiple strategies.

    Args:
        adapter: IBKRExecutionAdapter instance (shared)
        strategies: List of strategy config dicts, each with:
            - id: Strategy identifier (e.g. "ATRSS")
            - unit_risk_dollars: Dollar risk per 1R
            - daily_stop_R: Strategy daily stop in R units
            - priority: int (lower = higher priority)
            - max_working_orders: int (optional, default 4)
        heat_cap_R: Portfolio heat cap in R units (shared)
        portfolio_daily_stop_R: Portfolio daily stop in R units
        calendar: Optional event calendar for blackouts
        recon_interval_s: Reconciliation interval in seconds
        db_pool: Optional asyncpg pool for PostgreSQL persistence
        portfolio_unit_risk_dollars: Optional shared portfolio 1R base.

    Returns:
        Tuple of (OMSService, StrategyCoordinator) ready for start()
    """
    instrumentation_data_dir = _default_instrumentation_data_dir(
        instrumentation_data_dir, family_id
    )

    # Event bus (shared)
    bus = EventBus(clock=event_clock)

    # Repository (shared)
    if repository is not None:
        repo = repository
        logger.info("Using provided repository for multi-strategy OMS")
    elif db_pool is not None:
        repo = OMSRepository(db_pool)
        logger.info("Using PostgreSQL repository for multi-strategy OMS")
    else:
        repo = InMemoryRepository()
        logger.info("Using in-memory repository for multi-strategy OMS")

    # PgStore for risk state persistence
    pg_store = PgStore(db_pool) if db_pool is not None else None

    # Build per-strategy risk configs
    strategy_configs: dict[str, StrategyRiskConfig] = {}
    unit_risk_map: dict[str, float] = {}  # strategy_id → unit_risk_dollars
    for s in strategies:
        sid = s["id"]
        urd = s["unit_risk_dollars"]
        cfg = StrategyRiskConfig(
            strategy_id=sid,
            daily_stop_R=s.get("daily_stop_R", 2.0),
            unit_risk_dollars=urd,
            priority=s.get("priority", 99),
            max_heat_R=s.get("max_heat_R", 0.0),
            max_working_orders=s.get("max_working_orders", 4),
        )
        strategy_configs[sid] = cfg
        unit_risk_map[sid] = urd

    # Use the explicit portfolio URD when supplied. Otherwise preserve the
    # historic multi-strategy default of using the highest-priority strategy.
    _sorted_cfgs = sorted(strategy_configs.values(), key=lambda c: c.priority)
    if portfolio_unit_risk_dollars is not None and portfolio_unit_risk_dollars > 0:
        portfolio_urd = float(portfolio_unit_risk_dollars)
    else:
        portfolio_urd = _sorted_cfgs[0].unit_risk_dollars if _sorted_cfgs else 1.0

    risk_config = RiskConfig(
        heat_cap_R=heat_cap_R,
        portfolio_daily_stop_R=portfolio_daily_stop_R,
        portfolio_weekly_stop_R=portfolio_weekly_stop_R,
        strategy_configs=strategy_configs,
        portfolio_urd=portfolio_urd,
    )

    if calendar is None:
        calendar = EventCalendar()

    # Risk state providers (shared across strategies)
    strategy_risk_states: dict[str, StrategyRiskState] = {}
    portfolio_risk_state = PortfolioRiskState(trade_date=_trade_date_for())
    # Track positions by (strategy_id, symbol) for correct multi-strategy P&L
    open_positions: dict[tuple[str, str], dict] = {}

    # -- DB-backed risk state helpers (mirroring build_oms_service) ----------

    async def _sync_strategy_risk_from_repo(state: StrategyRiskState) -> None:
        urd = unit_risk_map.get(state.strategy_id, 1.0)
        positions = await repo.get_positions(state.strategy_id)
        if not positions:
            state.open_risk_dollars = 0.0
            state.open_risk_R = 0.0
            return
        open_pos = [p for p in positions if p.net_qty != 0]
        state.open_risk_dollars = sum(p.open_risk_dollars for p in open_pos)
        state.open_risk_R = sum(p.open_risk_R for p in open_pos)

    async def _load_strategy_risk_from_store(state: StrategyRiskState) -> RiskDailyStrategyRow | None:
        if pg_store is None:
            return None
        row = await pg_store.get_risk_daily_strategy(state.strategy_id, state.trade_date)
        if row is None:
            return None
        state.daily_realized_R = float(row.daily_realized_r)
        if row.daily_realized_usd is not None:
            state.daily_realized_pnl = float(row.daily_realized_usd)
        state.halted = row.halted
        state.halt_reason = row.halt_reason or ""
        return row

    async def _persist_strategy_risk_state(
        state: StrategyRiskState, as_of: datetime,
    ) -> None:
        if pg_store is None:
            return
        existing_row = await pg_store.get_risk_daily_strategy(state.strategy_id, state.trade_date)
        await pg_store.upsert_risk_daily_strategy(
            _build_strategy_daily_row(state, existing_row, as_of, family_id=family_id)
        )

    async def _sync_portfolio_open_risk_from_repo() -> None:
        positions = await repo.get_positions_for_strategies(_family_sids)
        open_pos = [pos for pos in positions if pos.net_qty != 0]
        portfolio_risk_state.open_risk_dollars = sum(
            pos.open_risk_dollars for pos in open_pos
        )
        portfolio_risk_state.open_risk_R = (
            portfolio_risk_state.open_risk_dollars / portfolio_urd
            if portfolio_urd > 0 else 0.0
        )

    _family_sids = [s["id"] for s in strategies]
    _allocation_targets = _allocation_targets_payload(
        allocation_targets,
        family_id=family_id,
        strategy_ids=_family_sids,
    )

    async def _load_portfolio_risk_from_store(trade_day: date) -> None:
        if pg_store is None:
            return
        today_rows = await pg_store.get_risk_daily_strategies_for_date(trade_day, strategy_ids=_family_sids)
        today_portfolio_row = await pg_store.get_risk_daily_portfolio(trade_day, family_id=family_id)

        if today_rows:
            daily_r, daily_usd, strat_pnl = _aggregate_strategy_daily_rows(today_rows)
            portfolio_risk_state.daily_realized_R = (
                daily_usd / portfolio_urd
                if portfolio_urd > 0 and daily_usd != 0.0 else daily_r
            )
            portfolio_risk_state.daily_realized_pnl = daily_usd
            portfolio_risk_state.strategy_daily_pnl = strat_pnl
        elif today_portfolio_row is not None:
            portfolio_risk_state.daily_realized_R = float(today_portfolio_row.daily_realized_r)
            portfolio_risk_state.daily_realized_pnl = float(today_portfolio_row.daily_realized_usd or 0)
            portfolio_risk_state.strategy_daily_pnl = {}
        else:
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.strategy_daily_pnl = {}

        totals = await pg_store.get_risk_daily_strategy_totals(_week_start(trade_day), trade_day, strategy_ids=_family_sids)
        portfolio_risk_state.weekly_realized_pnl = float(totals["total_usd"])
        portfolio_risk_state.weekly_realized_R = (
            portfolio_risk_state.weekly_realized_pnl / portfolio_urd
            if portfolio_urd > 0 and portfolio_risk_state.weekly_realized_pnl != 0.0
            else float(totals["total_r"])
        )
        if today_portfolio_row is not None:
            portfolio_risk_state.halted = today_portfolio_row.halted
            portfolio_risk_state.halt_reason = today_portfolio_row.halt_reason or ""

    async def _persist_portfolio_risk_state(as_of: datetime) -> None:
        if pg_store is None:
            return
        await _sync_portfolio_open_risk_from_repo()
        trade_day = portfolio_risk_state.trade_date
        daily_rows = await pg_store.get_risk_daily_strategies_for_date(trade_day, strategy_ids=_family_sids)
        existing_row = await pg_store.get_risk_daily_portfolio(trade_day, family_id=family_id)
        await pg_store.upsert_risk_daily_portfolio(
            _build_portfolio_daily_row(
                trade_date=trade_day, daily_rows=daily_rows,
                open_risk_r=portfolio_risk_state.open_risk_R,
                existing_row=existing_row, halted=portfolio_risk_state.halted,
                halt_reason=portfolio_risk_state.halt_reason,
                as_of=as_of, family_id=family_id,
                portfolio_urd=portfolio_urd,
            )
        )

    async def _internal_halt_trading(reason: str) -> None:
        halt_ts = datetime.now(timezone.utc)
        trade_day = _trade_date_for()
        portfolio_risk_state.trade_date = trade_day
        portfolio_risk_state.halted = True
        portfolio_risk_state.halt_reason = reason
        for sid, state in strategy_risk_states.items():
            state.halted = True
            state.halt_reason = reason
            if pg_store is not None:
                await pg_store.halt_strategy(sid, reason, trade_day)
            await _persist_strategy_risk_state(state, halt_ts)
        if pg_store is not None:
            await pg_store.halt_portfolio(reason, trade_day, family_id=family_id)
        await _persist_portfolio_risk_state(halt_ts)

    # -- Risk state accessors (DB-backed) -----------------------------------

    async def get_strategy_risk(sid: str) -> StrategyRiskState:
        today = _trade_date_for()
        if sid in strategy_risk_states:
            existing = strategy_risk_states[sid]
            if existing.trade_date != today:
                logger.info(f"Date boundary detected for {sid}: resetting daily risk state")
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=today,
                    open_risk_dollars=existing.open_risk_dollars,
                    open_risk_R=existing.open_risk_R,
                )
        if sid not in strategy_risk_states:
            strategy_risk_states[sid] = StrategyRiskState(strategy_id=sid, trade_date=today)
        state = strategy_risk_states[sid]
        await _sync_strategy_risk_from_repo(state)
        existing_row = await _load_strategy_risk_from_store(state)
        if pg_store is not None and existing_row is None:
            await _persist_strategy_risk_state(state, datetime.now(timezone.utc))
        return state

    async def get_portfolio_risk() -> PortfolioRiskState:
        today = _trade_date_for()
        if portfolio_risk_state.trade_date != today:
            logger.info("Date boundary detected: resetting portfolio daily risk state")
            # Monday weekly reset
            if today.weekday() == 0:
                logger.info("Monday weekly reset: weekly_R %.2f → 0.0",
                            portfolio_risk_state.weekly_realized_R)
                portfolio_risk_state.weekly_realized_pnl = 0.0
                portfolio_risk_state.weekly_realized_R = 0.0
            portfolio_risk_state.trade_date = today
            portfolio_risk_state.daily_realized_pnl = 0.0
            portfolio_risk_state.daily_realized_R = 0.0
            portfolio_risk_state.strategy_daily_pnl = {}
            portfolio_risk_state.halted = False
            portfolio_risk_state.halt_reason = ""
        await _sync_portfolio_open_risk_from_repo()
        await _load_portfolio_risk_from_store(today)
        portfolio_risk_state.pending_entry_risk_R = await repo.get_pending_entry_risk_R_for_strategies(
            _family_sids, portfolio_urd
        )
        return portfolio_risk_state

    async def get_working_order_count(sid: str) -> int:
        return await repo.count_working_orders(sid)

    # Fill processor (shared)
    fill_proc = FillProcessor(repo)
    _adapter_account = getattr(adapter, "_account", "") or ""
    try:
        from libs.oms.instrumentation._shared import account_alias_for

        lineage_account_alias = account_alias or account_alias_for(_adapter_account) or str(paper_equity_scope or "")
    except Exception:
        lineage_account_alias = account_alias or str(paper_equity_scope or "")
    lineage_portfolio_id = portfolio_id or "paper_default"
    oms_effective_config = {
        "strategies": strategies,
        "unit_risk_map": unit_risk_map,
        "family_id": family_id,
        "portfolio_daily_stop_R": portfolio_daily_stop_R,
        "portfolio_weekly_stop_R": portfolio_weekly_stop_R,
        "portfolio_id": lineage_portfolio_id,
        "account_alias": lineage_account_alias,
    }
    oms_parameter_set = parameter_set or {
        "effective_strategy_config": oms_effective_config,
        "allocation_targets": _allocation_targets,
    }
    strategy_by_id = {str(item.get("id", "")): dict(item) for item in strategies if item.get("id")}

    def _strategy_mapping_value(source, sid: str):
        if not isinstance(source, dict):
            return None
        if sid in source:
            return source[sid]
        nested = source.get("strategies")
        if isinstance(nested, dict) and sid in nested:
            return nested[sid]
        return None

    def _strategy_manifest_for(sid: str):
        return (
            _strategy_mapping_value(strategy_manifests, sid)
            or _strategy_mapping_value(strategy_manifest, sid)
            or strategy_manifest
        )

    def _strategy_effective_config_for(sid: str) -> dict:
        return {
            "strategy": strategy_by_id.get(sid, {"id": sid}),
            "family_effective_config": oms_effective_config,
        }

    def _strategy_parameter_set_for(sid: str):
        return (
            _strategy_mapping_value(parameter_sets, sid)
            or _strategy_mapping_value(parameter_set, sid)
            or {
                "effective_strategy_config": _strategy_effective_config_for(sid),
                "allocation_targets": _allocation_targets,
            }
        )

    oms_lineage = _make_oms_lineage(
        bot_id=f"{family_id}_oms" if family_id else "oms",
        family_id=family_id,
        portfolio_id=lineage_portfolio_id,
        account_alias=lineage_account_alias,
        portfolio_config=portfolio_config,
        portfolio_rules_config=portfolio_rules_config,
        strategy_manifest=strategy_manifest,
        effective_strategy_config=oms_effective_config,
        parameter_set=oms_parameter_set,
    )
    def _make_strategy_oms_lineage(sid: str, rules_config) -> LineageContext:
        return _make_oms_lineage(
            bot_id=f"{family_id}_oms" if family_id else "oms",
            strategy_id=sid,
            family_id=family_id,
            portfolio_id=lineage_portfolio_id,
            account_alias=lineage_account_alias,
            portfolio_config=portfolio_config,
            portfolio_rules_config=rules_config,
            strategy_manifest=_strategy_manifest_for(sid),
            effective_strategy_config=_strategy_effective_config_for(sid),
            parameter_set=_strategy_parameter_set_for(sid),
        )

    def _make_strategy_oms_lineages(rules_config, previous: dict[str, LineageContext] | None = None) -> dict[str, LineageContext]:
        result: dict[str, LineageContext] = {}
        for sid in strategy_by_id:
            lineage = _make_strategy_oms_lineage(sid, rules_config)
            current = (previous or {}).get(sid)
            if current is not None:
                lineage = replace(
                    lineage,
                    deployment_id=current.deployment_id or lineage.deployment_id,
                    trace_id=current.trace_id or lineage.trace_id,
                )
            result[sid] = lineage
        return result

    oms_lineage_state = {
        "value": oms_lineage,
        "strategies": _make_strategy_oms_lineages(portfolio_rules_config),
    }

    def _current_oms_lineage(strategy_id: str = ""):
        sid = str(strategy_id or "")
        if sid:
            return oms_lineage_state["strategies"].get(sid, oms_lineage_state["value"])
        return oms_lineage_state["value"]

    bus._current_oms_lineage = _current_oms_lineage

    def _refresh_oms_lineage(rules_config) -> None:
        current = oms_lineage_state["value"]
        refreshed = _make_oms_lineage(
            bot_id=f"{family_id}_oms" if family_id else "oms",
            family_id=family_id,
            portfolio_id=lineage_portfolio_id,
            account_alias=lineage_account_alias,
            portfolio_config=portfolio_config,
            portfolio_rules_config=rules_config,
            strategy_manifest=strategy_manifest,
            effective_strategy_config=oms_effective_config,
            parameter_set=oms_parameter_set,
        )
        oms_lineage_state["value"] = replace(
            refreshed,
            deployment_id=current.deployment_id or refreshed.deployment_id,
            trace_id=current.trace_id or refreshed.trace_id,
        )
        oms_lineage_state["strategies"] = _make_strategy_oms_lineages(
            rules_config,
            previous=oms_lineage_state.get("strategies", {}),
        )

    # Portfolio rules checker (cross-strategy coordination via shared store)
    portfolio_checker = None
    if portfolio_rules_config is not None:
        from ..risk.portfolio_rules import PortfolioRuleChecker

        rules_store = pg_store if pg_store is not None else repo
        portfolio_checker = PortfolioRuleChecker(
            config=portfolio_rules_config,
            get_strategy_signal=rules_store.get_strategy_signal,
            get_directional_risk_R=rules_store.get_directional_risk_R,
            get_current_equity=get_current_equity or (lambda: 10_000.0),
            on_rule_event=_make_portfolio_rule_logger(
                data_dir=instrumentation_data_dir,
                family_id=family_id,
                lineage=_current_oms_lineage,
            ),
            get_directional_risk_R_for_strategies=rules_store.get_directional_risk_R_for_strategies,
            get_sibling_positions_for_symbol=rules_store.get_sibling_positions_for_symbol,
            get_family_aggregate_mnq_eq=rules_store.get_family_aggregate_mnq_eq,
            get_directional_risk_dollars_for_strategies=rules_store.get_directional_risk_dollars_for_strategies,
            get_open_position_count_for_strategies=rules_store.get_open_position_count_for_strategies,
            get_symbol_open_risk_dollars_for_strategies=rules_store.get_symbol_open_risk_dollars_for_strategies,
            get_symbols_open_risk_dollars_for_strategies=rules_store.get_symbols_open_risk_dollars_for_strategies,
            get_active_risk_dollars_for_strategies=rules_store.get_active_risk_dollars_for_strategies,
            get_completed_trade_counts_for_strategies=rules_store.get_completed_trade_counts_for_strategies,
            get_recent_strategy_r_multiples=rules_store.get_recent_strategy_r_multiples,
            on_config_update=_refresh_oms_lineage,
        )
        logger.info("Portfolio rules enabled for multi-strategy OMS (%s)", family_id)

    # Risk gateway (shared, now with multiple strategy configs + priorities)
    portfolio_risk_adapter = None
    if family_id == "swing":
        from ..risk.swing_portfolio_adapter import SwingLivePortfolioRiskAdapter

        portfolio_risk_adapter = SwingLivePortfolioRiskAdapter(risk_config)

    risk_gateway = RiskGateway(
        config=risk_config,
        calendar=calendar,
        get_strategy_risk=get_strategy_risk,
        get_portfolio_risk=get_portfolio_risk,
        get_working_order_count=get_working_order_count,
        market_calendar=market_calendar,
        portfolio_checker=portfolio_checker,
        portfolio_risk_adapter=portfolio_risk_adapter,
        account_gate=account_gate,
        family_id=family_id,
    )
    risk_gateway._current_oms_lineage = _current_oms_lineage
    await upsert_active_runtime_config(
        db_pool,
        ActiveRuntimeConfigRecord(
            account_id=_adapter_account,
            config_scope="oms",
            scope_id=f"{family_id}:multi",
            runtime_env=get_environment(),
            payload={
                "account_id": _adapter_account,
                "family_id": family_id,
                "strategy_ids": list(unit_risk_map.keys()),
                "heat_cap_R": float(heat_cap_R),
                "portfolio_daily_stop_R": float(portfolio_daily_stop_R),
                "portfolio_weekly_stop_R": float(portfolio_weekly_stop_R),
                "portfolio_urd": float(portfolio_urd),
                "unit_risk_dollars": {
                    sid: float(value) for sid, value in unit_risk_map.items()
                },
                "account_gate_enabled": account_gate is not None,
                "source": "multi-strategy oms factory effective risk gateway config",
            },
            expires_at=active_config_expiry(),
        ),
    )

    # Execution router (shared)
    async def _pre_submit_recheck(order, conn=None) -> str | None:
        reserved_risk_dollars = float(
            getattr(getattr(order, "risk_context", None), "risk_dollars", 0.0)
            or 0.0
        )
        reserved_entry_risk_R = (
            reserved_risk_dollars / portfolio_urd if portfolio_urd > 0 else 0.0
        )
        denial = await risk_gateway.check_entry(
            order,
            skip_account_gate=True,
            reserved_entry_risk_R=reserved_entry_risk_R,
        )
        if denial:
            return denial
        return await risk_gateway.check_account_gate(
            order,
            conn=conn,
            reserved_entry_risk_dollars=reserved_risk_dollars,
        )

    router = ExecutionRouter(
        adapter,
        repo,
        bus=bus,
        pre_submit_recheck=_pre_submit_recheck,
        claimant_id=f"{family_id}:multi",
    )

    # Intent handler (shared)
    # OMS-7: pass adapter's configured IB account so the handler can stamp
    # account_id on swing/momentum orders (which leave it blank by default).
    handler = IntentHandler(
        risk_gateway, router, repo, bus,
        default_account_id=_adapter_account,
    )

    # Account/NAV provider for fill and closeout lifecycle snapshots.
    if paper_equity_pool is not None and not live_equity:
        from libs.persistence.paper_equity import load_paper_equity

        live_equity = [
            await load_paper_equity(
                paper_equity_pool,
                account_scope=paper_equity_scope,
                initial_equity=paper_initial_equity,
            )
        ]

    account_state_provider = _make_account_state_provider(
        get_current_equity=get_current_equity,
        live_equity=live_equity,
        fallback_equity=paper_initial_equity,
        family_id=family_id,
        account_id=_adapter_account,
        paper_equity_scope=paper_equity_scope,
    )

    # Reconciler (shared) with halt callback.
    _halt_cb = halt_trading if halt_trading is not None else _internal_halt_trading
    _reconciliation_authority = _resolve_reconciliation_authority(
        db_pool=db_pool,
        provided_authority=reconciliation_authority,
        enable_lease=enable_reconciliation_authority_lease,
    )
    reconciler = ReconciliationOrchestrator(
        adapter,
        repo,
        bus,
        halt_trading=_halt_cb,
        fill_processor=fill_proc,
        offline_fill_importer=lambda oms_id, exec_report: _import_fill_through_adapter_callback(
            adapter, repo, oms_id, exec_report,
        ),
        lifecycle_event_writer=_make_reconciliation_lifecycle_writer(
            instrumentation_data_dir, _current_oms_lineage,
        ),
        family_id=family_id,
        owner_id=reconciliation_owner_id or f"{family_id}:multi",
        reconciliation_authoritative=reconciliation_authoritative,
        authority=_reconciliation_authority,
    )

    # Timeout monitor
    from ..engine.timeout_monitor import OrderTimeoutMonitor
    timeout_monitor = OrderTimeoutMonitor(repo, bus, router)

    # Coordinator (cross-strategy signals)
    coordinator = StrategyCoordinator(bus, repo)

    # OMS-5: Hydrate the multi-strategy open_positions cache from DB before
    # any fill callback can fire. Same intent as the single-OMS path; the
    # multi version also seeds risk_per_contract_portfolio_R because the
    # multi callback's exit branch uses portfolio R for cross-family caps.
    if db_pool is not None or repository is not None:
        await _hydrate_open_positions_from_repo(
            open_positions=open_positions,
            repo=repo,
            strategy_ids=list(unit_risk_map.keys()),
            unit_risk_dollars_for=lambda sid: unit_risk_map.get(sid, 0.0),
            portfolio_unit_risk_dollars=portfolio_urd,
            strict=db_pool is not None,
        )

    # Wire adapter callbacks (multi-strategy aware, with DB persistence)
    _wire_adapter_callbacks_multi(
        adapter, bus, repo, fill_proc, router,
        strategy_risk_states, portfolio_risk_state,
        unit_risk_map, open_positions, coordinator,
        persist_strategy_risk_state=_persist_strategy_risk_state,
        persist_portfolio_risk_state=_persist_portfolio_risk_state,
        portfolio_urd=portfolio_urd,
        live_equity=live_equity,
        paper_equity_pool=paper_equity_pool,
        paper_equity_scope=paper_equity_scope,
        paper_initial_equity=paper_initial_equity,
        halt_trading=_halt_cb,
        family_id=family_id,
        family_strategy_ids=_family_sids,
        portfolio_id=getattr(oms_lineage, "portfolio_id", ""),
        allocation_targets=_allocation_targets,
        account_state_provider=account_state_provider,
        on_position_update=_make_fill_lifecycle_writer(
            instrumentation_data_dir, _current_oms_lineage,
        ),
    )

    # Build OMS service
    oms = OMSService(
        intent_handler=handler,
        bus=bus,
        reconciler=reconciler,
        router=router,
        recon_interval_s=recon_interval_s,
        timeout_monitor=timeout_monitor,
        get_portfolio_risk=get_portfolio_risk,
        get_strategy_risk=get_strategy_risk,
        on_intent_denied=_make_intent_denial_logger(
            data_dir=instrumentation_data_dir,
            family_id=family_id,
            lineage=_current_oms_lineage,
        ),
    )
    oms._portfolio_checker = portfolio_checker  # for coordinator regime updates
    oms._oms_lineage = _current_oms_lineage
    oms._refresh_oms_lineage = _refresh_oms_lineage
    oms._swing_portfolio_risk_adapter = portfolio_risk_adapter
    oms._portfolio_risk_state = portfolio_risk_state  # for coordinator deployed-capital queries
    oms._strategy_risk_states = strategy_risk_states  # for per-strategy heartbeat metrics
    oms._oms_repo = repo  # instrumentation daily reconciliation hook
    oms._family_strategy_ids = list(_family_sids)
    oms._family_id = family_id
    oms._allocation_targets = _allocation_targets
    oms._account_state_provider = account_state_provider

    # Seed risk state from the configured repository on startup. Persistence
    # remains Postgres-only; offline parity can still hydrate behavior from an
    # in-memory repository before the first decision is evaluated.
    if pg_store is not None or repository is not None:
        seed_ts = datetime.now(timezone.utc)
        for s in strategies:
            sid = s["id"]
            seed_state = strategy_risk_states.setdefault(
                sid, StrategyRiskState(strategy_id=sid, trade_date=_trade_date_for()),
            )
            await _sync_strategy_risk_from_repo(seed_state)
            if pg_store is None:
                continue
            await _load_strategy_risk_from_store(seed_state)
            await _persist_strategy_risk_state(seed_state, seed_ts)
        await _sync_portfolio_open_risk_from_repo()
        if pg_store is not None:
            await _load_portfolio_risk_from_store(_trade_date_for())
            await _persist_portfolio_risk_state(seed_ts)

    strategy_ids = [s["id"] for s in strategies]
    logger.info(f"Multi-strategy OMS factory built for: {strategy_ids}")
    return oms, coordinator


# ---------------------------------------------------------------------------
# Exception handler and adapter callback wiring
# ---------------------------------------------------------------------------

def _task_exception_handler(task: "asyncio.Task", bus: EventBus = None) -> None:
    """Log exceptions from fire-and-forget callback tasks and escalate.

    H4: On callback exception, emit RISK_HALT to pause strategy engines
    so that inconsistent state is not silently ignored.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            f"CRITICAL: Unhandled exception in OMS callback task: {exc}",
            exc_info=exc,
        )
        # H4: Escalate to RISK_HALT so strategies pause
        if bus is not None:
            try:
                bus.emit_risk_halt(
                    strategy_id="",
                    reason=f"OMS callback exception: {exc}",
                )
            except Exception:
                logger.error("Failed to emit RISK_HALT after callback exception")


def _make_order_callback_dispatcher(bus: EventBus) -> Callable[[str, Callable[[], Awaitable[object]]], object]:
    """Serialize broker callbacks per order while preserving cross-order concurrency."""
    import asyncio

    background_tasks: set[asyncio.Task] = set()
    tails: dict[str, asyncio.Task] = {}

    def _dispatch(oms_order_id: str, coro_factory: Callable[[], Awaitable[object]]) -> object:
        previous = tails.get(oms_order_id)

        async def _run() -> object:
            if previous is not None:
                try:
                    await previous
                except BaseException:
                    pass
            return await coro_factory()

        task = asyncio.create_task(_run())
        background_tasks.add(task)

        def _cleanup(done: "asyncio.Task") -> None:
            background_tasks.discard(done)
            if tails.get(oms_order_id) is done:
                tails.pop(oms_order_id, None)
            _task_exception_handler(done, bus=bus)

        task.add_done_callback(_cleanup)
        tails[oms_order_id] = task
        return task

    return _dispatch


_STATUS_TO_ORDER_STATUS: dict[str, OrderStatus] = {
    "Submitted": OrderStatus.WORKING,
    "PreSubmitted": OrderStatus.ROUTED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "ApiCancelled": OrderStatus.CANCELLED,
    "PendingCancel": OrderStatus.CANCEL_REQUESTED,
}

_ACK_FALLBACK_STATUSES = {
    OrderStatus.WORKING,
    OrderStatus.PARTIALLY_FILLED,
    OrderStatus.FILLED,
    OrderStatus.CANCEL_REQUESTED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
}


def _apply_ack_update(order, broker_ref, *, as_of: datetime) -> str:
    if order.status == OrderStatus.ROUTED and transition(order, OrderStatus.ACKED):
        order.broker_order_ref = broker_ref
        order.acked_at = as_of
        order.last_update_at = as_of
        return "emit"

    if order.status in _ACK_FALLBACK_STATUSES:
        changed = False
        if order.broker_order_ref != broker_ref:
            order.broker_order_ref = broker_ref
            changed = True
        if order.acked_at is None:
            order.acked_at = as_of
            changed = True
        if changed:
            order.last_update_at = as_of
            return "persist"
        return "noop"

    return "invalid"


def _apply_status_update(order, *, status: str, remaining: float, as_of: datetime) -> str:
    new_status = _STATUS_TO_ORDER_STATUS.get(status)
    if new_status is None:
        return "noop"

    status_changed = False
    if new_status != order.status:
        if order.status == OrderStatus.ROUTED and new_status in {
            OrderStatus.WORKING,
            OrderStatus.FILLED,
        }:
            if not transition(order, OrderStatus.ACKED):
                return "invalid"
            if order.acked_at is None:
                order.acked_at = as_of
        if new_status != order.status and not transition(order, new_status):
            return "invalid"
        status_changed = True

    remaining_int = int(remaining)
    if not status_changed and order.remaining_qty == remaining_int:
        return "noop"

    order.remaining_qty = remaining_int
    order.last_update_at = as_of
    return "emit"


async def _halt_unknown_exit_position(
    *,
    halt_trading,
    bus: EventBus,
    strategy_id: str,
    oms_order_id: str,
    instrument_symbol: str,
    role: OrderRole,
    qty: float,
    price: float,
) -> None:
    reason = (
        "Unknown open position for exit fill: "
        f"strategy={strategy_id} symbol={instrument_symbol} "
        f"role={role.value} order={oms_order_id} qty={qty} price={price}"
    )
    logger.critical(reason)
    bus.emit_risk_halt(strategy_id, reason)
    if halt_trading is not None:
        await halt_trading(reason)


async def _import_fill_through_adapter_callback(
    adapter,
    repo,
    oms_order_id: str,
    exec_report,
) -> bool:
    """Run offline broker executions through the live adapter fill pipeline."""
    on_fill = getattr(adapter, "on_fill", None)
    if on_fill is None:
        logger.error("Offline fill import requested before adapter.on_fill was wired")
        return False

    fill_ts = getattr(exec_report, "fill_time", None) or datetime.now(timezone.utc)
    commission = getattr(exec_report, "commission", 0.0) or 0.0
    result = on_fill(
        oms_order_id,
        exec_report.exec_id,
        float(exec_report.price),
        float(exec_report.qty),
        fill_ts,
        float(commission),
    )
    if result is not None and hasattr(result, "__await__"):
        imported = await result
        if isinstance(imported, bool):
            if imported and not await repo.fill_exists(exec_report.exec_id):
                logger.error(
                    "Offline fill importer reported success but fill is missing: exec_id=%s",
                    exec_report.exec_id,
                )
                return False
            return imported
    return await repo.fill_exists(exec_report.exec_id)


def _wire_adapter_callbacks(
    adapter, bus: EventBus, repo, fill_proc: FillProcessor, router,
    strategy_risk_states, portfolio_risk_state, unit_risk_dollars, open_positions,
    persist_strategy_risk_state=None,
    persist_portfolio_risk_state=None,
    paper_equity=None,
    paper_equity_pool=None,
    strat_cfg=None,
    risk_config=None,
    portfolio_urd: Optional[float] = None,
    portfolio_unit_risk_pct: float = 0.0,
    db_pool=None,
    live_equity=None,
    paper_equity_scope="paper",
    paper_initial_equity=10_000.0,
    halt_trading=None,
    family_id: str = "",
    family_strategy_ids: Optional[list[str]] = None,
    portfolio_id: str = "",
    allocation_targets: Optional[dict] = None,
    account_state_provider: Optional[Callable[[], dict]] = None,
    on_position_update=None,
) -> None:
    """Wire IBKRExecutionAdapter callbacks to OMS event bus.

    Callbacks look up the order from the repository to get the strategy_id,
    then emit appropriate events to the bus for strategy routing.
    Also wires FillProcessor for OMS order state and updates risk state.
    """
    import asyncio
    from datetime import datetime, date, timezone

    try:
        portfolio_urd_value = float(portfolio_urd)
        if portfolio_urd_value <= 0:
            portfolio_urd_value = unit_risk_dollars
    except (TypeError, ValueError):
        portfolio_urd_value = unit_risk_dollars

    def _refresh_portfolio_urd(equity: float) -> None:
        nonlocal portfolio_urd_value
        if risk_config is not None and portfolio_unit_risk_pct > 0 and equity > 0:
            portfolio_urd_value = equity * portfolio_unit_risk_pct
            risk_config.portfolio_urd = portfolio_urd_value

    dispatch_order_callback = _make_order_callback_dispatcher(bus)

    def _get_running_loop():
        """Get the running event loop, or None if not in async context."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    # H7 fix: retry config for pacing errors
    MAX_PACING_RETRIES = 3
    PACING_RETRY_BASE_DELAY_S = 2.0
    family_scope = list(family_strategy_ids or [])

    async def _emit_position_update_snapshot(
        *,
        order,
        oms_pos: Position,
        fill_data: dict,
        fill_ts: datetime,
        strat_risk,
    ) -> None:
        sid = getattr(order, "strategy_id", "")
        instr_sym = order.instrument.symbol if order.instrument else ""
        scope = family_scope or [sid]
        try:
            repo_positions = await repo.get_positions_for_strategies(scope)
        except Exception as exc:
            logger.warning("Failed to read positions for fill snapshot: %s", exc)
            repo_positions = [oms_pos]
        if not repo_positions:
            repo_positions = [oms_pos]
        positions = [
            _position_payload(
                pos,
                family_id=family_id,
                portfolio_id=portfolio_id,
                mark_price=float(fill_data.get("price", 0.0) or 0.0)
                if pos.instrument_symbol == instr_sym else 0.0,
            )
            for pos in repo_positions
        ]
        current = _position_payload(
            oms_pos,
            family_id=family_id,
            portfolio_id=portfolio_id,
            mark_price=float(fill_data.get("price", 0.0) or 0.0),
        )
        account_state = _account_state_snapshot(account_state_provider)
        payload = {
            "timestamp": fill_ts.isoformat(),
            "snapshot_id": fill_data.get("exec_id") or "",
            "source": "fill",
            "position": current,
            "positions": positions,
            "fill": fill_data,
            "order": _order_payload(order, family_id=family_id),
            "strategy_risk": _risk_state_payload(strat_risk),
            "portfolio_risk": _risk_state_payload(portfolio_risk_state),
            "allocation_targets": allocation_targets or {},
            "account_state": account_state,
            "raw_nav": _nav_from_account_state(account_state, "raw_nav"),
            "allocated_nav": _nav_from_account_state(account_state, "allocated_nav"),
        }
        try:
            bus.emit_position_update(sid, getattr(order, "oms_order_id", ""), payload)
        except Exception as exc:
            logger.warning("Failed to emit position update event: %s", exc)
        if on_position_update is None:
            return
        try:
            result = on_position_update(payload)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.warning("Failed to write position lifecycle snapshot: %s", exc)

    def on_ack(oms_order_id: str, broker_ref) -> None:
        """Handle order acknowledgment from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.debug(f"Adapter ack (no loop): {oms_order_id}")
            return

        async def _emit_ack():
            order = await repo.get_order(oms_order_id)
            if order:
                acked_at = datetime.now(timezone.utc)
                outcome = _apply_ack_update(order, broker_ref, as_of=acked_at)
                if outcome == "emit":
                    await repo.save_order(order)
                    bus.emit_order_event(order)
                    logger.debug(f"Adapter ack emitted: {oms_order_id} for {order.strategy_id}")
                elif outcome == "persist":
                    await repo.save_order(order)
                elif outcome == "invalid":
                    logger.warning(
                        f"Adapter ack: invalid transition for {oms_order_id} "
                        f"(current status={order.status.value})"
                    )
            else:
                logger.warning(f"Adapter ack for unknown order: {oms_order_id}")

        dispatch_order_callback(oms_order_id, _emit_ack)

    def on_reject(oms_order_id: str, reason: str, error_code: int, retryable: bool) -> None:
        """Handle order rejection from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.warning(f"Adapter reject (no loop): {oms_order_id} - {reason}")
            return

        async def _emit_reject():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter reject for unknown order: {oms_order_id} - {reason}")
                return

            # H5/H7 fix: retry for retryable pacing errors using persisted retry_count
            if retryable and error_code > 0:
                if order.retry_count < MAX_PACING_RETRIES:
                    order.retry_count += 1
                    delay = PACING_RETRY_BASE_DELAY_S * (2 ** (order.retry_count - 1))
                    logger.warning(
                        f"Retryable reject for {oms_order_id} (code={error_code}): "
                        f"retry {order.retry_count}/{MAX_PACING_RETRIES} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    # Re-route through the execution router
                    try:
                        if not transition(order, OrderStatus.RISK_APPROVED):
                            logger.warning(
                                f"Cannot retry {oms_order_id}: transition to RISK_APPROVED "
                                f"failed from {order.status}"
                            )
                        else:
                            order.last_update_at = datetime.now(timezone.utc)
                            await repo.save_order(order)
                            await router.route(order)
                            return
                    except Exception as e:
                        logger.error(f"Retry failed for {oms_order_id}: {e}")

            # Terminal rejection
            if transition(order, OrderStatus.REJECTED):
                order.reject_reason = reason
                order.last_update_at = datetime.now(timezone.utc)
                await repo.save_order(order)
                bus.emit_order_event(order)
                logger.warning(f"Adapter reject emitted: {oms_order_id} for {order.strategy_id} - {reason}")
            else:
                logger.warning(
                    f"Adapter reject: invalid transition for {oms_order_id} "
                    f"(current status={order.status.value})"
                )

        dispatch_order_callback(oms_order_id, _emit_reject)

    def on_fill(
        oms_order_id: str,
        exec_id: str,
        price: float,
        qty: float,
        timestamp,
        commission: float,
    ) -> None:
        """Handle fill from broker - update OMS order state, risk state, emit event."""
        loop = _get_running_loop()
        if loop is None:
            logger.info(f"Adapter fill (no loop): {oms_order_id} {qty}@{price}")
            return

        async def _emit_fill():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter fill for unknown order: {oms_order_id}")
                return False

            # 1. Update OMS order state (filled_qty, status, avg_fill_price)
            fill_ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
            inserted = await fill_proc.process_fill(oms_order_id, exec_id, price, qty, fill_ts, commission)

            # OMS-2: All downstream side effects (risk/equity/position updates,
            # strategy bus emit, paper equity, coordinator notifications) are
            # gated on `inserted`. A duplicate exec_id (IBKR replay, restart
            # cache miss) returns False and we skip everything below. This is
            # the only barrier preventing double-counted realized P&L and
            # double-released open risk after a duplicate fill.
            if not inserted:
                return False

            # 2. Emit fill event to strategy
            risk_context = getattr(order, "risk_context", None)
            fill_data = {
                "exec_id": exec_id,
                "fill_id": exec_id,
                "oms_order_id": oms_order_id,
                "price": price,
                "qty": qty,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "commission": commission,
                "client_order_id": getattr(order, 'client_order_id', ""),
                "symbol": order.instrument.symbol if order.instrument else "",
                "side": order.side.value if order.side else "",
                "order_type": order.order_type.value if order.order_type else "",
                "role": order.role.value if order.role else "",
                "requested_qty": order.qty,
                "intent_id": getattr(risk_context, "intent_id", ""),
                "risk_decision_ref": getattr(risk_context, "risk_decision_ref", ""),
                "portfolio_decision_ref": getattr(risk_context, "portfolio_decision_ref", ""),
            }
            bus.emit_fill_event(order.strategy_id, oms_order_id, fill_data)

            # 3. Update risk state
            sid = order.strategy_id
            instr_sym = order.instrument.symbol if order.instrument else ""
            pos_key = (sid, instr_sym)
            if sid not in strategy_risk_states:
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=_trade_date_for(fill_ts)
                )
            strat_risk = strategy_risk_states[sid]

            if order.role == OrderRole.ENTRY and order.risk_context:
                risk_per_contract = (
                    order.risk_context.risk_dollars / order.qty if order.qty > 0 else 0
                )
                fill_risk = risk_per_contract * qty
                fill_risk_R = fill_risk / unit_risk_dollars if unit_risk_dollars > 0 else 0
                fill_risk_portfolio_R = (
                    fill_risk / portfolio_urd_value if portfolio_urd_value > 0 else fill_risk_R
                )

                strat_risk.open_risk_dollars += fill_risk
                strat_risk.open_risk_R += fill_risk_R
                portfolio_risk_state.open_risk_dollars += fill_risk
                portfolio_risk_state.open_risk_R += fill_risk_portfolio_R

                # Track entry for exit P&L computation per symbol.
                pos = open_positions.get(pos_key)
                pv = order.instrument.point_value if order.instrument else 1.0
                if pos is None:
                    open_positions[pos_key] = {
                        "entry_price": price,
                        "risk_per_contract_R": fill_risk_R / qty if qty > 0 else 0,
                        "risk_per_contract_portfolio_R": (
                            fill_risk_portfolio_R / qty if qty > 0 else 0
                        ),
                        "point_value": pv,
                        "side": order.side,
                        "open_qty": qty,
                    }
                else:
                    old_total = pos["entry_price"] * pos["open_qty"]
                    pos["open_qty"] += qty
                    pos["entry_price"] = (old_total + price * qty) / pos["open_qty"]

                # Paper equity: deduct entry commission
                if paper_equity is not None and paper_equity[0] is not None and paper_equity_pool is not None and commission > 0:
                    try:
                        from libs.persistence.paper_equity import apply_paper_pnl
                        new_eq = await apply_paper_pnl(
                            paper_equity_pool,
                            0.0,
                            commission,
                            account_scope=paper_equity_scope,
                            initial_equity=paper_initial_equity,
                        )
                        paper_equity[0] = new_eq
                        _refresh_portfolio_urd(new_eq)
                    except Exception as e:
                        logger.warning("Paper equity entry commission failed (non-fatal): %s", e)

                # Write cross-strategy signal to shared DB
                if db_pool is not None:
                    try:
                        from ..persistence.postgres import PgStore as _PgS
                        _pg_sig = _PgS(db_pool)
                        direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
                        await _pg_sig.upsert_strategy_signal(sid, direction, fill_ts)
                    except Exception as e:
                        logger.warning("Failed to write strategy signal: %s", e)

            elif order.role in (OrderRole.EXIT, OrderRole.STOP, OrderRole.TP):
                pos = open_positions.get(pos_key)
                if pos is None:
                    await _halt_unknown_exit_position(
                        halt_trading=halt_trading,
                        bus=bus,
                        strategy_id=sid,
                        oms_order_id=oms_order_id,
                        instrument_symbol=instr_sym,
                        role=order.role,
                        qty=qty,
                        price=price,
                    )
                    return True
                if pos:
                    # Reduce open risk
                    released_R = pos["risk_per_contract_R"] * qty
                    released_dollars = released_R * unit_risk_dollars
                    released_portfolio_R = pos.get(
                        "risk_per_contract_portfolio_R",
                        released_R,
                    ) * qty

                    strat_risk.open_risk_R = max(0, strat_risk.open_risk_R - released_R)
                    strat_risk.open_risk_dollars = max(0, strat_risk.open_risk_dollars - released_dollars)
                    portfolio_risk_state.open_risk_R = max(0, portfolio_risk_state.open_risk_R - released_portfolio_R)
                    portfolio_risk_state.open_risk_dollars = max(0, portfolio_risk_state.open_risk_dollars - released_dollars)

                    # Compute realized P&L
                    pv = pos["point_value"]
                    if pos["side"] == OrderSide.BUY:
                        pnl = (price - pos["entry_price"]) * pv * qty
                    else:
                        pnl = (pos["entry_price"] - price) * pv * qty

                    pnl_R = pnl / unit_risk_dollars if unit_risk_dollars > 0 else 0
                    pnl_portfolio_R = (
                        pnl / portfolio_urd_value if portfolio_urd_value > 0 else pnl_R
                    )
                    strat_risk.daily_realized_pnl += pnl
                    strat_risk.daily_realized_R += pnl_R
                    portfolio_risk_state.daily_realized_pnl += pnl
                    portfolio_risk_state.daily_realized_R += pnl_portfolio_R
                    # Consolidated weekly tracking
                    portfolio_risk_state.weekly_realized_pnl += pnl
                    portfolio_risk_state.weekly_realized_R += pnl_portfolio_R
                    # Per-strategy daily breakdown
                    if portfolio_risk_state.strategy_daily_pnl is None:
                        portfolio_risk_state.strategy_daily_pnl = {}
                    portfolio_risk_state.strategy_daily_pnl[sid] = (
                        portfolio_risk_state.strategy_daily_pnl.get(sid, 0.0) + pnl
                    )

                    # Paper equity: apply realized P&L + exit commission
                    if paper_equity is not None and paper_equity[0] is not None and paper_equity_pool is not None:
                        try:
                            from libs.persistence.paper_equity import apply_paper_pnl
                            new_eq = await apply_paper_pnl(
                                paper_equity_pool,
                                pnl,
                                commission,
                                account_scope=paper_equity_scope,
                                initial_equity=paper_initial_equity,
                            )
                            paper_equity[0] = new_eq
                            if strat_cfg is not None and new_eq > 0:
                                strat_cfg.unit_risk_dollars = new_eq * strat_cfg.unit_risk_pct
                            _refresh_portfolio_urd(new_eq)
                            logger.info("Paper equity: $%.2f (pnl=$%.2f, comm=$%.2f)", new_eq, pnl, commission)
                        except Exception as e:
                            logger.warning("Paper equity update failed (non-fatal): %s", e)

                    # Live equity: update mutable ref so DD tiers see current equity
                    if live_equity is not None and live_equity[0] is not None:
                        live_equity[0] += pnl - commission
                        if strat_cfg is not None and live_equity[0] > 0:
                            strat_cfg.unit_risk_dollars = live_equity[0] * strat_cfg.unit_risk_pct
                        _refresh_portfolio_urd(live_equity[0])

                    pos["open_qty"] = max(0, pos["open_qty"] - qty)
                    if pos["open_qty"] <= 0:
                        del open_positions[pos_key]

            # 4. Update OMS position for FLATTEN handler and reconciliation
            pos_data = open_positions.get(pos_key)
            if pos_data:
                net = pos_data["open_qty"] if pos_data["side"] == OrderSide.BUY else -pos_data["open_qty"]
                symbol_open_risk_R = pos_data["risk_per_contract_R"] * pos_data["open_qty"]
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=net,
                    avg_price=pos_data["entry_price"],
                    realized_pnl=strat_risk.daily_realized_pnl,
                    open_risk_dollars=symbol_open_risk_R * unit_risk_dollars,
                    open_risk_R=symbol_open_risk_R,
                    last_update_at=fill_ts,
                )
            else:
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=0,
                    realized_pnl=strat_risk.daily_realized_pnl,
                    last_update_at=fill_ts,
                )
            await repo.save_position(oms_pos)
            if persist_strategy_risk_state is not None:
                await persist_strategy_risk_state(strat_risk, fill_ts)
            if persist_portfolio_risk_state is not None:
                await persist_portfolio_risk_state(fill_ts)
            await _emit_position_update_snapshot(
                order=order,
                oms_pos=oms_pos,
                fill_data=fill_data,
                fill_ts=fill_ts,
                strat_risk=strat_risk,
            )

            logger.info(f"Adapter fill processed: {oms_order_id} {qty}@{price} for {sid}/{instr_sym}")
            return True

        return dispatch_order_callback(oms_order_id, _emit_fill)

    def on_status(oms_order_id: str, status: str, remaining: float) -> None:
        """Handle status update from broker."""
        loop = _get_running_loop()
        if loop is None:
            logger.debug(f"Adapter status (no loop): {oms_order_id} {status}")
            return

        async def _emit_status():
            order = await repo.get_order(oms_order_id)
            if order:
                outcome = _apply_status_update(
                    order,
                    status=status,
                    remaining=remaining,
                    as_of=datetime.now(timezone.utc),
                )
                if outcome == "emit":
                    await repo.save_order(order)
                    bus.emit_order_event(order)
                    logger.debug(f"Adapter status emitted: {oms_order_id} {status} for {order.strategy_id}")
                elif outcome == "invalid":
                    target = _STATUS_TO_ORDER_STATUS.get(status)
                    logger.warning(
                        f"Adapter status: invalid transition for {oms_order_id} "
                        f"{order.status.value} -> {target.value if target else status}"
                    )
            else:
                logger.debug(f"Adapter status for unknown order: {oms_order_id} {status}")

        dispatch_order_callback(oms_order_id, _emit_status)

    adapter.on_ack = on_ack
    adapter.on_reject = on_reject
    adapter.on_fill = on_fill
    adapter.on_status = on_status


def _wire_adapter_callbacks_multi(
    adapter, bus: EventBus, repo, fill_proc: FillProcessor, router,
    strategy_risk_states, portfolio_risk_state,
    unit_risk_map: dict[str, float],
    open_positions: dict[tuple[str, str], dict],
    coordinator: "StrategyCoordinator",
    persist_strategy_risk_state=None,
    persist_portfolio_risk_state=None,
    portfolio_urd: float = 1.0,
    live_equity=None,
    paper_equity_pool=None,
    paper_equity_scope: str = "paper",
    paper_initial_equity: float = 10_000.0,
    halt_trading=None,
    family_id: str = "",
    family_strategy_ids: Optional[list[str]] = None,
    portfolio_id: str = "",
    allocation_targets: Optional[dict] = None,
    account_state_provider: Optional[Callable[[], dict]] = None,
    on_position_update=None,
) -> None:
    """Wire IBKRExecutionAdapter callbacks for multi-strategy OMS.

    Key differences from single-strategy:
    - unit_risk_dollars looked up per order's strategy_id
    - open_positions keyed by (strategy_id, symbol) for concurrent positions
    - coordinator notified on fills for cross-strategy signals
    """
    import asyncio
    from datetime import datetime, timezone

    dispatch_order_callback = _make_order_callback_dispatcher(bus)

    def _get_running_loop():
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    MAX_PACING_RETRIES = 3
    PACING_RETRY_BASE_DELAY_S = 2.0
    family_scope = list(family_strategy_ids or [])

    # portfolio_urd: portfolio-level unit risk dollars for normalizing R across strategies
    # (highest-priority strategy's URD, passed from build_multi_strategy_oms).

    async def _emit_position_update_snapshot(
        *,
        order,
        oms_pos: Position,
        fill_data: dict,
        fill_ts: datetime,
        strat_risk,
    ) -> None:
        sid = getattr(order, "strategy_id", "")
        instr_sym = order.instrument.symbol if order.instrument else ""
        scope = family_scope or list(unit_risk_map.keys()) or [sid]
        try:
            repo_positions = await repo.get_positions_for_strategies(scope)
        except Exception as exc:
            logger.warning("Failed to read positions for multi fill snapshot: %s", exc)
            repo_positions = [oms_pos]
        if not repo_positions:
            repo_positions = [oms_pos]
        positions = [
            _position_payload(
                pos,
                family_id=family_id,
                portfolio_id=portfolio_id,
                mark_price=float(fill_data.get("price", 0.0) or 0.0)
                if pos.instrument_symbol == instr_sym else 0.0,
            )
            for pos in repo_positions
        ]
        current = _position_payload(
            oms_pos,
            family_id=family_id,
            portfolio_id=portfolio_id,
            mark_price=float(fill_data.get("price", 0.0) or 0.0),
        )
        account_state = _account_state_snapshot(account_state_provider)
        payload = {
            "timestamp": fill_ts.isoformat(),
            "snapshot_id": fill_data.get("exec_id") or "",
            "source": "fill",
            "position": current,
            "positions": positions,
            "fill": fill_data,
            "order": _order_payload(order, family_id=family_id),
            "strategy_risk": _risk_state_payload(strat_risk),
            "portfolio_risk": _risk_state_payload(portfolio_risk_state),
            "allocation_targets": allocation_targets or {},
            "account_state": account_state,
            "raw_nav": _nav_from_account_state(account_state, "raw_nav"),
            "allocated_nav": _nav_from_account_state(account_state, "allocated_nav"),
        }
        try:
            bus.emit_position_update(sid, getattr(order, "oms_order_id", ""), payload)
        except Exception as exc:
            logger.warning("Failed to emit multi position update event: %s", exc)
        if on_position_update is None:
            return
        try:
            result = on_position_update(payload)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.warning("Failed to write multi position lifecycle snapshot: %s", exc)

    def on_ack(oms_order_id: str, broker_ref) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_ack():
            order = await repo.get_order(oms_order_id)
            if order:
                outcome = _apply_ack_update(
                    order,
                    broker_ref,
                    as_of=datetime.now(timezone.utc),
                )
                if outcome == "emit":
                    await repo.save_order(order)
                    bus.emit_order_event(order)
                elif outcome == "persist":
                    await repo.save_order(order)
                elif outcome == "invalid":
                    logger.warning(
                        f"Adapter ack: invalid transition for {oms_order_id} "
                        f"(current status={order.status.value})"
                    )

        dispatch_order_callback(oms_order_id, _emit_ack)

    def on_reject(oms_order_id: str, reason: str, error_code: int, retryable: bool) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_reject():
            order = await repo.get_order(oms_order_id)
            if not order:
                return

            # H5: use persisted retry_count instead of transient dynamic attr
            if retryable and error_code > 0:
                if order.retry_count < MAX_PACING_RETRIES:
                    order.retry_count += 1
                    delay = PACING_RETRY_BASE_DELAY_S * (2 ** (order.retry_count - 1))
                    logger.warning(
                        f"Retryable reject for {oms_order_id} (code={error_code}): "
                        f"retry {order.retry_count}/{MAX_PACING_RETRIES} in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    try:
                        if not transition(order, OrderStatus.RISK_APPROVED):
                            logger.warning(
                                f"Cannot retry {oms_order_id}: transition to RISK_APPROVED "
                                f"failed from {order.status}"
                            )
                        else:
                            order.last_update_at = datetime.now(timezone.utc)
                            await repo.save_order(order)
                            await router.route(order)
                            return
                    except Exception as e:
                        logger.error(f"Retry failed for {oms_order_id}: {e}")

            if transition(order, OrderStatus.REJECTED):
                order.reject_reason = reason
                order.last_update_at = datetime.now(timezone.utc)
                await repo.save_order(order)
                bus.emit_order_event(order)

        dispatch_order_callback(oms_order_id, _emit_reject)

    def on_fill(
        oms_order_id: str,
        exec_id: str,
        price: float,
        qty: float,
        timestamp,
        commission: float,
    ) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_fill():
            order = await repo.get_order(oms_order_id)
            if not order:
                logger.warning(f"Adapter fill for unknown order: {oms_order_id}")
                return False

            fill_ts = timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc)
            inserted = await fill_proc.process_fill(oms_order_id, exec_id, price, qty, fill_ts, commission)

            # OMS-2: Same dedupe gate as the single-strategy callback above.
            # Without this, IBKR exec replays would double-count risk across
            # the swing family's shared multi-strategy OMS.
            if not inserted:
                return False

            risk_context = getattr(order, "risk_context", None)
            fill_data = {
                "exec_id": exec_id,
                "fill_id": exec_id,
                "oms_order_id": oms_order_id,
                "price": price,
                "qty": qty,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                "commission": commission,
                "client_order_id": getattr(order, 'client_order_id', ""),
                "symbol": order.instrument.symbol if order.instrument else "",
                "side": order.side.value if order.side else "",
                "order_type": order.order_type.value if order.order_type else "",
                "role": order.role.value if order.role else "",
                "requested_qty": order.qty,
                "intent_id": getattr(risk_context, "intent_id", ""),
                "risk_decision_ref": getattr(risk_context, "risk_decision_ref", ""),
                "portfolio_decision_ref": getattr(risk_context, "portfolio_decision_ref", ""),
            }
            bus.emit_fill_event(order.strategy_id, oms_order_id, fill_data)

            # Per-strategy unit_risk_dollars lookup
            sid = order.strategy_id
            urd = unit_risk_map.get(sid, 1.0)
            instr_sym = order.instrument.symbol if order.instrument else ""
            pos_key = (sid, instr_sym)

            if sid not in strategy_risk_states:
                strategy_risk_states[sid] = StrategyRiskState(
                    strategy_id=sid, trade_date=_trade_date_for(fill_ts)
                )
            strat_risk = strategy_risk_states[sid]

            if order.role == OrderRole.ENTRY and order.risk_context:
                risk_per_contract = (
                    order.risk_context.risk_dollars / order.qty if order.qty > 0 else 0
                )
                fill_risk = risk_per_contract * qty
                fill_risk_R = fill_risk / urd if urd > 0 else 0
                # Portfolio R uses consistent base so 1R means the same dollars
                fill_risk_portfolio_R = fill_risk / portfolio_urd if portfolio_urd > 0 else 0

                strat_risk.open_risk_dollars += fill_risk
                strat_risk.open_risk_R += fill_risk_R
                portfolio_risk_state.open_risk_dollars += fill_risk
                portfolio_risk_state.open_risk_R += fill_risk_portfolio_R

                pv = order.instrument.point_value if order.instrument else 1.0
                pos = open_positions.get(pos_key)
                if pos is None:
                    open_positions[pos_key] = {
                        "entry_price": price,
                        "risk_per_contract_R": fill_risk_R / qty if qty > 0 else 0,
                        "risk_per_contract_portfolio_R": fill_risk_portfolio_R / qty if qty > 0 else 0,
                        "point_value": pv,
                        "side": order.side,
                        "open_qty": qty,
                    }
                else:
                    old_total = pos["entry_price"] * pos["open_qty"]
                    pos["open_qty"] += qty
                    pos["entry_price"] = (old_total + price * qty) / pos["open_qty"]

                # Notify coordinator of entry fill
                direction = "LONG" if order.side == OrderSide.BUY else "SHORT"
                coordinator.on_fill(
                    strategy_id=sid, symbol=instr_sym,
                    side=order.side.value, role="ENTRY", price=price,
                )
                coordinator.on_position_update(
                    strategy_id=sid, symbol=instr_sym,
                    qty=int(open_positions[pos_key]["open_qty"]),
                    direction=direction, entry_price=open_positions[pos_key]["entry_price"],
                )

                # Paper equity: deduct entry commission
                if paper_equity_pool is not None and commission > 0:
                    try:
                        from libs.persistence.paper_equity import apply_paper_pnl
                        new_eq = await apply_paper_pnl(
                            paper_equity_pool, 0.0, commission,
                            account_scope=paper_equity_scope,
                            initial_equity=paper_initial_equity,
                        )
                        if live_equity and live_equity[0] is not None:
                            live_equity[0] = new_eq
                    except Exception as _pe_exc:
                        logger.warning("Paper equity entry commission failed: %s", _pe_exc)

            elif order.role in (OrderRole.EXIT, OrderRole.STOP, OrderRole.TP):
                pos = open_positions.get(pos_key)
                if pos is None:
                    await _halt_unknown_exit_position(
                        halt_trading=halt_trading,
                        bus=bus,
                        strategy_id=sid,
                        oms_order_id=oms_order_id,
                        instrument_symbol=instr_sym,
                        role=order.role,
                        qty=qty,
                        price=price,
                    )
                    return True
                if pos:
                    released_R = pos["risk_per_contract_R"] * qty
                    released_dollars = released_R * urd
                    released_portfolio_R = pos.get("risk_per_contract_portfolio_R", released_R) * qty

                    strat_risk.open_risk_R = max(0, strat_risk.open_risk_R - released_R)
                    strat_risk.open_risk_dollars = max(0, strat_risk.open_risk_dollars - released_dollars)
                    portfolio_risk_state.open_risk_R = max(0, portfolio_risk_state.open_risk_R - released_portfolio_R)
                    portfolio_risk_state.open_risk_dollars = max(0, portfolio_risk_state.open_risk_dollars - released_dollars)

                    pv = pos["point_value"]
                    if pos["side"] == OrderSide.BUY:
                        pnl = (price - pos["entry_price"]) * pv * qty
                    else:
                        pnl = (pos["entry_price"] - price) * pv * qty

                    pnl_R = pnl / urd if urd > 0 else 0
                    pnl_portfolio_R = pnl / portfolio_urd if portfolio_urd > 0 else 0
                    strat_risk.daily_realized_pnl += pnl
                    strat_risk.daily_realized_R += pnl_R
                    portfolio_risk_state.daily_realized_pnl += pnl
                    portfolio_risk_state.daily_realized_R += pnl_portfolio_R
                    portfolio_risk_state.weekly_realized_pnl += pnl
                    portfolio_risk_state.weekly_realized_R += pnl_portfolio_R
                    if portfolio_risk_state.strategy_daily_pnl is None:
                        portfolio_risk_state.strategy_daily_pnl = {}
                    portfolio_risk_state.strategy_daily_pnl[sid] = (
                        portfolio_risk_state.strategy_daily_pnl.get(sid, 0.0) + pnl
                    )

                    # Live equity: update mutable ref so DD tiers see current equity
                    if live_equity is not None and live_equity[0] is not None:
                        live_equity[0] += pnl - commission

                    # Paper equity: persist realized P&L + exit commission
                    if paper_equity_pool is not None:
                        try:
                            from libs.persistence.paper_equity import apply_paper_pnl
                            new_eq = await apply_paper_pnl(
                                paper_equity_pool, pnl, commission,
                                account_scope=paper_equity_scope,
                                initial_equity=paper_initial_equity,
                            )
                            if live_equity and live_equity[0] is not None:
                                live_equity[0] = new_eq
                        except Exception as _pe_exc:
                            logger.warning("Paper equity update failed: %s", _pe_exc)

                    pos["open_qty"] = max(0, pos["open_qty"] - qty)
                    remaining = int(pos["open_qty"])
                    if remaining <= 0:
                        del open_positions[pos_key]

                    # Notify coordinator of position change
                    direction = "LONG" if pos["side"] == OrderSide.BUY else "SHORT"
                    coordinator.on_fill(
                        strategy_id=sid, symbol=instr_sym,
                        side=order.side.value, role=order.role.value, price=price,
                    )
                    coordinator.on_position_update(
                        strategy_id=sid, symbol=instr_sym,
                        qty=remaining, direction=direction,
                    )

            # Save OMS position
            pos_data = open_positions.get(pos_key)
            if pos_data:
                net = pos_data["open_qty"] if pos_data["side"] == OrderSide.BUY else -pos_data["open_qty"]
                # Per-position risk (not strategy total) to prevent SUM double-counting
                symbol_open_risk_R = pos_data["risk_per_contract_R"] * pos_data["open_qty"]
                symbol_open_risk_dollars = symbol_open_risk_R * urd
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=net,
                    avg_price=pos_data["entry_price"],
                    realized_pnl=strat_risk.daily_realized_pnl,
                    open_risk_dollars=symbol_open_risk_dollars,
                    open_risk_R=symbol_open_risk_R,
                    last_update_at=fill_ts,
                )
            else:
                oms_pos = Position(
                    account_id=order.account_id,
                    instrument_symbol=instr_sym,
                    strategy_id=sid,
                    net_qty=0,
                    realized_pnl=strat_risk.daily_realized_pnl,
                    last_update_at=fill_ts,
                )
            await repo.save_position(oms_pos)

            # Persist risk state to DB after fill updates
            if persist_strategy_risk_state is not None:
                await persist_strategy_risk_state(strat_risk, fill_ts)
            if persist_portfolio_risk_state is not None:
                await persist_portfolio_risk_state(fill_ts)
            await _emit_position_update_snapshot(
                order=order,
                oms_pos=oms_pos,
                fill_data=fill_data,
                fill_ts=fill_ts,
                strat_risk=strat_risk,
            )

            logger.info(f"Multi-OMS fill: {oms_order_id} {qty}@{price} for {sid}/{instr_sym}")
            return True

        return dispatch_order_callback(oms_order_id, _emit_fill)

    def on_status(oms_order_id: str, status: str, remaining: float) -> None:
        loop = _get_running_loop()
        if loop is None:
            return

        async def _emit_status():
            order = await repo.get_order(oms_order_id)
            if order:
                outcome = _apply_status_update(
                    order,
                    status=status,
                    remaining=remaining,
                    as_of=datetime.now(timezone.utc),
                )
                if outcome == "emit":
                    await repo.save_order(order)
                    bus.emit_order_event(order)
                elif outcome == "invalid":
                    target = _STATUS_TO_ORDER_STATUS.get(status)
                    logger.warning(
                        f"Adapter status: invalid transition for {oms_order_id} "
                        f"{order.status.value} -> {target.value if target else status}"
                    )

        dispatch_order_callback(oms_order_id, _emit_status)

    adapter.on_ack = on_ack
    adapter.on_reject = on_reject
    adapter.on_fill = on_fill
    adapter.on_status = on_status
