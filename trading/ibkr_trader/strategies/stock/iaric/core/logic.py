from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Sequence

from strategies.core.actions import (
    CancelAction,
    FlattenPosition,
    ReplaceProtectiveStop,
    SubmitEntry,
    SubmitMarketExit,
    SubmitProtectiveStop,
)
from strategies.core.events import DecisionEvent
from strategies.stock.iaric.execution import build_position_from_fill
from strategies.stock.iaric.models import Bar, MarketSnapshot, PBSymbolState, PositionState, VWAPLedger, WatchlistItem
from strategies.stock.iaric.signals import compute_micropressure_proxy

from .state import (
    IARICBarInput,
    IARICCoreState,
    IARICEntryRequest,
    IARICEntryAcceptance,
    IARICFill,
    IARICFlattenRequest,
    IARICOrderUpdate,
    IARICPartialExitRequest,
    IARICRouteStep,
    IARICStopUpdateRequest,
)

_TERMINAL_STATUSES = {
    "cancelled",
    "expired",
    "rejected",
    "order_cancelled",
    "order_expired",
    "order_rejected",
}


def build_core_state(engine) -> IARICCoreState:
    return IARICCoreState(
        trade_date=engine._artifact.trade_date,
        saved_at=datetime.now(timezone.utc),
        symbols=deepcopy(list(engine._symbols.values())),
        last_decision_code=engine._last_decision_code,
        meta={
            "active_symbols": sorted(engine._active_symbols),
            "order_index": deepcopy(engine._order_index),
            "pending_entry_risk": deepcopy(engine._portfolio.pending_entry_risk),
            "account_equity": engine._portfolio.account_equity,
            "base_risk_fraction": engine._portfolio.base_risk_fraction,
            "regime_allows_no_new_entries": engine._portfolio.regime_allows_no_new_entries,
            "expected_stop_cancels": sorted(engine._expected_stop_cancels),
            "last_decision_details": deepcopy(engine._last_decision_details),
            "last_bar_ts": engine._last_bar_ts,
        },
    )


def apply_core_state(engine, state: IARICCoreState) -> None:
    restored = {symbol_state.symbol: deepcopy(symbol_state) for symbol_state in state.symbols}
    for symbol, symbol_state in restored.items():
        engine._symbols[symbol] = symbol_state
        engine._markets.setdefault(symbol, MarketSnapshot(symbol=symbol))
        engine._session_vwap.setdefault(symbol, VWAPLedger())

    meta = state.meta if isinstance(state.meta, dict) else {}
    engine._active_symbols = set(meta.get("active_symbols", engine._active_symbols))
    engine._order_index = {
        str(order_id): _coerce_order_index_entry(value)
        for order_id, value in dict(meta.get("order_index", {})).items()
    }
    engine._portfolio.pending_entry_risk = {
        str(symbol): float(risk)
        for symbol, risk in dict(meta.get("pending_entry_risk", {})).items()
    }
    if "account_equity" in meta:
        engine._portfolio.account_equity = float(meta["account_equity"])
    if "base_risk_fraction" in meta:
        engine._portfolio.base_risk_fraction = float(meta["base_risk_fraction"])
    if "regime_allows_no_new_entries" in meta:
        engine._portfolio.regime_allows_no_new_entries = bool(meta["regime_allows_no_new_entries"])
    engine._expected_stop_cancels = {
        str(order_id) for order_id in meta.get("expected_stop_cancels", [])
    }
    engine._portfolio.open_positions = {
        symbol_state.symbol: deepcopy(symbol_state.position)
        for symbol_state in engine._symbols.values()
        if symbol_state.position is not None and symbol_state.in_position
    }
    engine._last_decision_code = state.last_decision_code
    engine._last_decision_details = dict(meta.get("last_decision_details", {}))
    engine._last_bar_ts = _coerce_datetime(meta.get("last_bar_ts"))


def active_symbols(state: IARICCoreState) -> list[str]:
    meta = state.meta if isinstance(state.meta, dict) else {}
    return list(meta.get("active_symbols", []))


def route_prefix(route_family: str) -> str:
    return {
        "OPEN_SCORED_ENTRY": "pb_open_scored",
        "DELAYED_CONFIRM": "pb_delayed_confirm",
        "OPENING_RECLAIM": "pb_opening_reclaim",
    }.get(str(route_family or "").upper(), "pb_opening_reclaim")


def route_enabled(settings: Any, route_family: str) -> bool:
    route_key = str(route_family or "").upper()
    v2 = bool(getattr(settings, "pb_v2_enabled", False))
    if route_key == "OPEN_SCORED_ENTRY":
        attr = "pb_v2_open_scored_enabled" if v2 else "pb_open_scored_enabled"
        return bool(getattr(settings, attr, True))
    if route_key == "DELAYED_CONFIRM":
        return bool(getattr(settings, "pb_delayed_confirm_enabled", True))
    if route_key == "OPENING_RECLAIM":
        return bool(getattr(settings, "pb_opening_reclaim_enabled", True))
    if route_key == "VWAP_BOUNCE":
        return v2 and bool(getattr(settings, "pb_v2_vwap_bounce_enabled", True))
    if route_key == "AFTERNOON_RETEST":
        return v2 and bool(getattr(settings, "pb_v2_afternoon_retest_enabled", True))
    return True


def route_setting(settings: Any, route_family: str, suffix: str, fallback_suffix: str | None = None) -> Any:
    prefix = route_prefix(route_family)
    attr = f"{prefix}_{suffix}"
    if hasattr(settings, attr):
        return getattr(settings, attr)
    if fallback_suffix is not None and hasattr(settings, fallback_suffix):
        return getattr(settings, fallback_suffix)
    raise AttributeError(f"Missing route setting for {route_family}:{suffix}")


def route_carry_profile(route_family: str) -> str:
    return route_prefix(route_family).replace("pb_", "").upper()


def route_min_daily_signal_score(settings: Any, route_family: str) -> float:
    route_key = str(route_family or "").upper()
    if route_key == "OPEN_SCORED_ENTRY":
        v2 = bool(getattr(settings, "pb_v2_enabled", False))
        attr = "pb_v2_open_scored_min_score" if v2 else "pb_open_scored_min_score"
        return float(getattr(settings, attr, 0.0))
    if route_key == "DELAYED_CONFIRM":
        return float(
            getattr(
                settings,
                "pb_delayed_confirm_min_daily_signal_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    if route_key == "OPENING_RECLAIM":
        return float(
            getattr(
                settings,
                "pb_opening_reclaim_min_daily_signal_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    if route_key == "AFTERNOON_RETEST":
        return float(
            getattr(
                settings,
                "pb_v2_afternoon_retest_min_score",
                getattr(settings, "pb_daily_signal_min_score", 0.0),
            )
        )
    return float(getattr(settings, "pb_daily_signal_min_score", 0.0))


def open_scored_eligible(settings: Any, payload: dict[str, Any] | None) -> bool:
    if not route_enabled(settings, "OPEN_SCORED_ENTRY"):
        return False
    source = payload or {}
    score = float(source.get("daily_signal_score") or 0.0)
    rank_pct = float(source.get("daily_signal_rank_pct") or 100.0)
    v2 = bool(getattr(settings, "pb_v2_enabled", False))
    min_score = route_min_daily_signal_score(settings, "OPEN_SCORED_ENTRY")
    max_rank_attr = "pb_v2_open_scored_rank_pct_max" if v2 else "pb_open_scored_rank_pct_max"
    max_rank_pct = float(getattr(settings, max_rank_attr, 100.0))
    return score >= min_score and rank_pct <= max_rank_pct


def entry_threshold(settings: Any, state: Any) -> float:
    if bool(getattr(state, "rescue_flow_candidate", False)):
        return float(max(getattr(settings, "pb_rescue_min_score", 0.0), getattr(settings, "pb_entry_score_min", 0.0)))
    if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM":
        return float(min(getattr(settings, "pb_entry_score_min", 0.0), getattr(settings, "pb_delayed_confirm_score_min", 0.0)))
    return float(getattr(settings, "pb_entry_score_min", 0.0))


def compute_volume_ratio(bar: Bar, item: WatchlistItem | None) -> float:
    if item is None:
        return 1.0
    expected = float(item.expected_5m_volume)
    if expected <= 0 and item.average_30m_volume > 0:
        expected = float(item.average_30m_volume) / 6.0
    return float(bar.volume / max(expected, 1.0))


def micropressure_label(
    bars: Sequence[Bar],
    bar_idx: int,
    reclaim_level: float,
    item: WatchlistItem,
    *,
    lookback_bars: int = 3,
) -> str:
    if bar_idx < 0 or bar_idx >= len(bars):
        return "NEUTRAL"
    span = max(int(lookback_bars), 1)
    recent = list(bars[max(0, bar_idx - (span - 1)) : bar_idx + 1])
    bullish = 0
    for sample in recent:
        label = compute_micropressure_proxy(
            sample,
            expected_volume=max(item.expected_5m_volume, 1.0),
            median20_volume=max(item.average_30m_volume / 6.0, 1.0),
            reclaim_level=reclaim_level,
        )
        if label == "ACCUMULATE":
            bullish += 1
    if bullish >= max(1, len(recent) - 1):
        return "ACCUMULATE"
    if bullish == 0 and recent and recent[-1].close < recent[-1].open:
        return "DISTRIBUTE"
    return "NEUTRAL"


def thirty_min_context_bonus(market: MarketSnapshot, *, weight: float) -> float:
    bar = market.last_30m_bar
    if bar is None:
        return 0.0
    close_pct = _close_in_range_pct(bar.high, bar.low, bar.close)
    bonus = (close_pct - 0.5) * weight
    if bar.close > bar.open:
        bonus += weight * 0.35
    elif bar.close < bar.open:
        bonus -= weight * 0.20
    return float(min(max(bonus, -weight), weight))


def compute_initial_stop(settings: Any, setup_low: float, daily_atr: float, session_atr: float) -> float:
    daily_cap = float(getattr(settings, "pb_stop_daily_atr_cap", 0.0)) * max(float(daily_atr), 0.0)
    session_buffer = float(getattr(settings, "pb_stop_session_atr_mult", 0.0)) * float(session_atr)
    buffer = min(session_buffer, daily_cap) if daily_cap > 0 else session_buffer
    return max(float(setup_low) - max(buffer, 0.01), 0.01)


def _close_in_range_pct(high: float, low: float, close: float) -> float:
    high_f = float(high)
    low_f = float(low)
    if high_f <= low_f:
        return 1.0
    return float(min(max((float(close) - low_f) / (high_f - low_f), 0.0), 1.0))


def compute_route_entry_score_bundle(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    *,
    bars: Sequence[Bar] | None = None,
    volume_ratio: float | None = None,
    micropressure: str | None = None,
    context_bonus: float | None = None,
) -> dict[str, float]:
    def _clip01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    def _peak_score(value: float, *, target: float, width: float) -> float:
        width = max(float(width), 1e-6)
        return _clip01(1.0 - abs(float(value) - float(target)) / width)

    route_family_name = getattr(state, "route_family", "") or (
        "DELAYED_CONFIRM" if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM" else "OPENING_RECLAIM"
    )
    score_family = str(getattr(settings, "pb_entry_score_family", "meanrev_sweetspot_v1") or "meanrev_sweetspot_v1").lower()
    daily_signal = min(max(float(getattr(state, "daily_signal_score", 0.0)) / 100.0, 0.0), 1.0)
    reclaim_score = 0.0
    if float(getattr(state, "stop_level", 0.0)) > 0 and bar.close > float(getattr(state, "reclaim_level", 0.0)):
        reclaim_score = min(
            max(
                (bar.close - float(getattr(state, "reclaim_level", 0.0)))
                / max(bar.close - float(getattr(state, "stop_level", 0.0)), 0.01),
                0.0,
            ),
            1.5,
        ) / 1.5
    if volume_ratio is None:
        volume_ratio = compute_volume_ratio(bar, item)
    volume_score = min(max(float(volume_ratio) / max(float(getattr(settings, "pb_ready_min_volume_ratio", 0.25)), 0.25), 0.0), 1.25) / 1.25
    vwap = market.session_vwap or bar.close
    vwap_score = 0.0
    daily_atr = float(getattr(state, "daily_atr", 0.0))
    if daily_atr > 0:
        vwap_score = min(max((bar.close - vwap) / max(daily_atr * 0.75, 0.01), 0.0), 1.0)
    cpr_score = min(max(bar.cpr, 0.0), 1.0)
    if micropressure is None:
        series = list(bars) if bars is not None else [bar]
        series_idx = bar_idx if bars is not None else len(series) - 1
        micropressure = micropressure_label(series, series_idx, float(getattr(state, "reclaim_level", 0.0)), item)
    reclaim_bars = max(bar_idx - int(getattr(state, "flush_bar_idx", 0)) + 1, 1)
    speed_score = min(max(1.0 - (reclaim_bars - 1) / 8.0, 0.0), 1.0)
    if context_bonus is None:
        context_bonus = thirty_min_context_bonus(market, weight=4.0)
    route_flag = 0.0 if route_family_name == "OPENING_RECLAIM" else 1.0

    def _bundle(
        *,
        daily_weight: float,
        reclaim_weight: float,
        volume_weight: float,
        vwap_weight: float,
        cpr_weight: float,
        speed_weight: float,
        context_low: float,
        context_high: float,
        distribute_penalty: float,
        neutral_penalty: float,
        weak_vwap_penalty_value: float,
        rescue_penalty_value: float,
        reclaim_input: float = reclaim_score,
        vwap_input: float = vwap_score,
        cpr_input: float = cpr_score,
        extension_penalty: float = 0.0,
    ) -> dict[str, float]:
        context_adjust = min(max(float(context_bonus), context_low), context_high)
        micro_penalty = distribute_penalty if micropressure == "DISTRIBUTE" else neutral_penalty if micropressure == "NEUTRAL" else 0.0
        weak_vwap_penalty = weak_vwap_penalty_value if bar.close < vwap else 0.0
        rescue_penalty = rescue_penalty_value if bool(getattr(state, "rescue_flow_candidate", False)) else 0.0
        total = (
            daily_signal * daily_weight
            + reclaim_input * reclaim_weight
            + volume_score * volume_weight
            + vwap_input * vwap_weight
            + cpr_input * cpr_weight
            + speed_score * speed_weight
            + context_adjust
            + micro_penalty
            + weak_vwap_penalty
            + rescue_penalty
            + extension_penalty
        )
        return {
            "route_family": route_flag,
            "daily_signal": float(daily_signal * daily_weight),
            "reclaim": float(reclaim_input * reclaim_weight),
            "volume": float(volume_score * volume_weight),
            "vwap_hold": float(vwap_input * vwap_weight),
            "cpr": float(cpr_input * cpr_weight),
            "speed": float(speed_score * speed_weight),
            "context_adjust": float(context_adjust),
            "micro_penalty": float(micro_penalty),
            "weak_vwap_penalty": float(weak_vwap_penalty),
            "rescue_penalty": float(rescue_penalty),
            "extension_penalty": float(extension_penalty),
            "score": float(max(total, 0.0)),
        }

    if score_family == "route_momentum_v1":
        return _bundle(
            daily_weight=45.0,
            reclaim_weight=18.0,
            volume_weight=12.0,
            vwap_weight=10.0,
            cpr_weight=10.0,
            speed_weight=8.0,
            context_low=-6.0,
            context_high=3.0,
            distribute_penalty=-12.0,
            neutral_penalty=-4.0,
            weak_vwap_penalty_value=-8.0,
            rescue_penalty_value=-8.0,
        )
    if score_family == "route_quality_v1":
        return _bundle(
            daily_weight=40.0,
            reclaim_weight=10.0,
            volume_weight=16.0,
            vwap_weight=10.0,
            cpr_weight=10.0,
            speed_weight=8.0,
            context_low=-4.0,
            context_high=2.0,
            distribute_penalty=-14.0,
            neutral_penalty=-6.0,
            weak_vwap_penalty_value=-12.0,
            rescue_penalty_value=-10.0,
        )
    if score_family == "route_early_reversal_v1":
        return _bundle(
            daily_weight=36.0,
            reclaim_weight=14.0,
            volume_weight=14.0,
            vwap_weight=12.0,
            cpr_weight=10.0,
            speed_weight=12.0,
            context_low=-4.0,
            context_high=2.0,
            distribute_penalty=-12.0,
            neutral_penalty=-5.0,
            weak_vwap_penalty_value=-10.0,
            rescue_penalty_value=-8.0,
        )

    reclaim_target = 0.55 if route_family_name == "OPENING_RECLAIM" else 0.45
    vwap_target = 0.28 if route_family_name == "OPENING_RECLAIM" else 0.20
    cpr_target = 0.68 if route_family_name == "OPENING_RECLAIM" else 0.62
    reclaim_component = _peak_score(reclaim_score, target=reclaim_target, width=0.45)
    vwap_component = _peak_score(vwap_score, target=vwap_target, width=0.28)
    cpr_component = _peak_score(cpr_score, target=cpr_target, width=0.28)
    extension_penalty = 0.0
    if reclaim_score > 0.85:
        extension_penalty -= _clip01((reclaim_score - 0.85) / 0.15) * 4.0
    if vwap_score > 0.60:
        extension_penalty -= _clip01((vwap_score - 0.60) / 0.40) * 6.0
    if cpr_score > 0.85:
        extension_penalty -= _clip01((cpr_score - 0.85) / 0.15) * 6.0
    return _bundle(
        daily_weight=54.0,
        reclaim_weight=8.0,
        volume_weight=12.0,
        vwap_weight=5.0,
        cpr_weight=6.0,
        speed_weight=8.0,
        context_low=-4.0,
        context_high=2.0,
        distribute_penalty=-12.0,
        neutral_penalty=-5.0,
        weak_vwap_penalty_value=-10.0,
        rescue_penalty_value=-8.0,
        reclaim_input=reclaim_component,
        vwap_input=vwap_component,
        cpr_input=cpr_component,
        extension_penalty=extension_penalty,
    )


def apply_entry_acceptance(state: Any, acceptance: IARICEntryAcceptance) -> None:
    state.accepted_bar_idx = int(acceptance.accepted_bar_idx)
    state.accepted_timestamp = acceptance.accepted_timestamp
    state.accepted_entry_price = float(acceptance.accepted_entry_price)
    state.accepted_entry_trigger = str(acceptance.entry_trigger)
    state.accepted_route_family = str(acceptance.route_family)
    state.accepted_score = float(acceptance.score)
    state.accepted_session_atr = float(acceptance.session_atr)
    state.accepted_score_components = dict(acceptance.score_components)


def reset_route_state(state: Any) -> None:
    reset_for_watch = getattr(state, "reset_for_watch", None)
    if callable(reset_for_watch):
        reset_for_watch()
        if hasattr(state, "last_transition_reason"):
            state.last_transition_reason = ""
        return
    state.stage = "WATCHING"
    state.intraday_setup_type = ""
    state.route_family = ""
    state.setup_low = 0.0
    state.reclaim_level = 0.0
    state.stop_level = 0.0
    state.flush_bar_idx = 0
    state.ready_bar_idx = -1
    state.acceptance_count = 0
    state.required_acceptance = 0
    state.intraday_score = 0.0
    state.target_entry_price = 0.0
    state.improvement_expires = 0
    state.invalid_reason = ""
    state.invalid_reset_bar = 0
    state.score_components = {}
    state.ready_cpr = 0.0
    state.ready_volume_ratio = 0.0
    state.ready_timestamp = None
    state.accepted_bar_idx = -1
    state.accepted_timestamp = None
    state.accepted_entry_price = 0.0
    state.accepted_entry_trigger = ""
    state.accepted_route_family = ""
    state.accepted_score = 0.0
    state.accepted_session_atr = 0.0
    state.accepted_score_components = {}
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = ""


def maybe_reset_invalidated_state(state: Any, bar_idx: int) -> bool:
    if getattr(state, "stage", "") != "INVALIDATED":
        return False
    if bar_idx < int(getattr(state, "invalid_reset_bar", 0)):
        return False
    reset_route_state(state)
    return True


def invalidate_route_state(state: Any, reason: str, reset_bar: int) -> IARICRouteStep:
    prior = str(getattr(state, "stage", "WATCHING"))
    state.stage = "INVALIDATED"
    state.invalid_reason = reason
    state.invalid_reset_bar = int(reset_bar)
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = reason
    return IARICRouteStep(prior_stage=prior, stage="INVALIDATED", reason=reason)


def _state_or_market_session_low(state: Any, market: MarketSnapshot, bar: Bar) -> float:
    state_session_low = float(getattr(state, "session_low", 0.0))
    market_session_low = float(market.session_low) if market.session_low is not None else 0.0
    base_low = state_session_low if state_session_low > 0 else market_session_low if market_session_low > 0 else bar.low
    return min(base_low, bar.low)


def advance_opening_reclaim_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if float(getattr(state, "daily_signal_score", 0.0)) < route_min_daily_signal_score(settings, "OPENING_RECLAIM"):
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_low = _state_or_market_session_low(state, market, bar)
    if state.stage == "WATCHING":
        first_bar_open = series[0].open if series else bar.open
        flush_distance = (first_bar_open - session_low) / max(session_atr, 0.01)
        flush_bar = (
            bar_idx < int(getattr(settings, "pb_flush_window_bars", 0))
            and flush_distance >= float(getattr(settings, "pb_flush_min_atr", 0.0))
            and bar.cpr <= float(getattr(settings, "pb_flush_cpr_max", 0.0))
        )
        micro = micropressure_label(series, min(bar_idx, len(series) - 1), bar.close, item)
        pm_reentry_signal = (
            bool(getattr(state, "stopped_out_today", False))
            and bool(getattr(settings, "pb_pm_reentry", False))
            and bar_idx >= int(getattr(settings, "pb_pm_reentry_after_bar", 0))
            and bar.close > bar.open
            and market.session_vwap is not None
            and bar.close >= market.session_vwap
            and micro == "ACCUMULATE"
        )
        if not (flush_bar or pm_reentry_signal):
            return None
        prior = state.stage
        state.stage = "FLUSH_LOCKED"
        if hasattr(state, "intraday_setup_type"):
            state.intraday_setup_type = (
                "PM_REENTRY"
                if pm_reentry_signal
                else "OPENING_FLUSH" if bar_idx < int(getattr(settings, "pb_opening_range_bars", 0)) else "SESSION_FLUSH"
            )
        state.route_family = "OPENING_RECLAIM"
        state.setup_low = session_low
        reclaim_anchor = max(
            bar.high - float(getattr(settings, "pb_reclaim_offset_atr", 0.0)) * session_atr,
            (market.session_vwap or bar.close) - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr,
        )
        state.reclaim_level = max(reclaim_anchor, session_low + session_atr * 0.25)
        state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
        state.flush_bar_idx = int(bar_idx)
        if hasattr(state, "last_transition_reason"):
            state.last_transition_reason = "flush_detected"
        return IARICRouteStep(prior_stage=prior, stage="FLUSH_LOCKED", reason=str(getattr(state, "intraday_setup_type", "flush_detected")))

    if state.stage == "FLUSH_LOCKED":
        state.setup_low = min(float(getattr(state, "setup_low", bar.low)), bar.low)
        reclaim_anchor = max(
            bar.high - float(getattr(settings, "pb_reclaim_offset_atr", 0.0)) * session_atr,
            (market.session_vwap or bar.close) - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr,
        )
        state.reclaim_level = max(reclaim_anchor, float(getattr(state, "setup_low", bar.low)) + session_atr * 0.25)
        state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
        if bar.close >= float(getattr(state, "reclaim_level", 0.0)) or bar.high >= float(getattr(state, "reclaim_level", 0.0)):
            prior = state.stage
            state.stage = "RECLAIMING"
            state.required_acceptance = max(1, int(getattr(settings, "pb_ready_acceptance_bars", 1)))
            state.acceptance_count = 0
            if hasattr(state, "last_transition_reason"):
                state.last_transition_reason = "reclaim_hit"
            return IARICRouteStep(prior_stage=prior, stage="RECLAIMING", reason="reclaim_hit")
        if bar_idx >= int(getattr(settings, "pb_flush_window_bars", 0)) + int(getattr(settings, "pb_ready_acceptance_bars", 0)):
            return invalidate_route_state(
                state,
                "flush_stale",
                max(bar_idx + 1, int(getattr(settings, "pb_delayed_confirm_after_bar", 0))),
            )
        return None

    if state.stage != "RECLAIMING":
        return None
    if bar.low <= float(getattr(state, "stop_level", 0.0)) or bar.close < float(getattr(state, "setup_low", 0.0)):
        reset_bar = max(
            bar_idx + 2,
            int(getattr(settings, "pb_pm_reentry_after_bar", 0)) if bool(getattr(state, "stopped_out_today", False)) else bar_idx + 2,
        )
        return invalidate_route_state(state, "reclaim_failed", reset_bar)
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), float(getattr(state, "reclaim_level", 0.0)), item)
    volume_ok = compute_volume_ratio(bar, item) >= float(getattr(settings, "pb_ready_min_volume_ratio", 0.0))
    cpr_ok = bar.cpr >= float(getattr(settings, "pb_ready_min_cpr", 0.0))
    vwap_ok = market.session_vwap is None or bar.close >= market.session_vwap - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr
    if bar.close >= float(getattr(state, "reclaim_level", 0.0)) and bar.close > bar.open and cpr_ok and volume_ok and vwap_ok and micro != "DISTRIBUTE":
        state.acceptance_count = int(getattr(state, "acceptance_count", 0)) + 1
    elif bar.close < float(getattr(state, "reclaim_level", 0.0)):
        state.acceptance_count = max(int(getattr(state, "acceptance_count", 0)) - 1, 0)
    if int(getattr(state, "acceptance_count", 0)) < max(1, int(getattr(state, "required_acceptance", 1))):
        return None
    prior = state.stage
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = max(
        float(getattr(state, "reclaim_level", 0.0)),
        bar.close * (1.0 - float(getattr(settings, "pb_improvement_discount_pct", 0.0))),
    )
    state.improvement_expires = bar_idx + max(0, int(getattr(settings, "pb_improvement_window_bars", 0)))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "acceptance_complete"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="acceptance_complete", score=state.intraday_score)


def activate_delayed_confirm_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if bool(getattr(state, "stopped_out_today", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_delayed_confirm_allow_rescue", False)):
        return None
    if not route_enabled(settings, "DELAYED_CONFIRM"):
        return None
    if float(getattr(state, "daily_signal_score", 0.0)) < route_min_daily_signal_score(settings, "DELAYED_CONFIRM"):
        return None
    if bar_idx < int(getattr(settings, "pb_delayed_confirm_after_bar", 0)):
        return None
    if getattr(state, "stage", "") != "WATCHING":
        return None
    vwap = market.session_vwap
    if vwap is None:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    session_low = _state_or_market_session_low(state, market, bar)
    close_pct = _close_in_range_pct(bar.high, bar.low, bar.close)
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), vwap, item)
    if bool(getattr(settings, "pb_v2_enabled", False)):
        min_close_pct = float(getattr(settings, "pb_v2_delayed_confirm_min_close_pct", 0.0))
        vol_ratio_min = float(getattr(settings, "pb_v2_delayed_confirm_vol_ratio", 0.0))
        volume_ok = compute_volume_ratio(bar, item) >= vol_ratio_min
        vwap_ok = bar.close >= vwap - 0.50 * session_atr
        if bar.close <= bar.open or close_pct < min_close_pct or not volume_ok or not vwap_ok or micro == "DISTRIBUTE":
            return None
    else:
        volume_ok = compute_volume_ratio(bar, item) >= max(float(getattr(settings, "pb_ready_min_volume_ratio", 0.0)) * 0.75, 0.5)
        vwap_ok = bar.close >= vwap - float(getattr(settings, "pb_ready_vwap_buffer_atr", 0.0)) * session_atr
        retest_depth = (series[0].open - session_low) / max(session_atr, 0.01)
        bounce_strength = (bar.close - session_low) / max(session_atr, 0.01)
        if (
            bar.close <= bar.open
            or close_pct < float(getattr(settings, "pb_delayed_confirm_min_close_pct", 0.0))
            or not volume_ok
            or not vwap_ok
            or micro == "DISTRIBUTE"
            or retest_depth < 0.05
            or bounce_strength < 0.20
        ):
            return None
    state.intraday_setup_type = "DELAYED_CONFIRM"
    state.route_family = "DELAYED_CONFIRM"
    state.setup_low = session_low
    state.reclaim_level = max(vwap, session_low + session_atr * 0.35)
    state.stop_level = compute_initial_stop(settings, state.setup_low, float(getattr(state, "daily_atr", 0.0)), session_atr)
    state.flush_bar_idx = max(0, bar_idx - int(getattr(settings, "pb_delayed_confirm_after_bar", 0)) + 1)
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    if state.intraday_score < float(getattr(settings, "pb_delayed_confirm_score_min", 0.0)):
        state.intraday_setup_type = ""
        state.setup_low = 0.0
        state.reclaim_level = 0.0
        state.stop_level = 0.0
        state.flush_bar_idx = 0
        state.acceptance_count = 0
        state.required_acceptance = 0
        state.intraday_score = 0.0
        state.score_components = {}
        return None
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = max(
        float(getattr(state, "reclaim_level", 0.0)),
        bar.close * (1.0 - float(getattr(settings, "pb_improvement_discount_pct", 0.0)) * 0.5),
    )
    state.improvement_expires = bar_idx + max(0, int(getattr(settings, "pb_improvement_window_bars", 0)))
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "delayed_confirm"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="delayed_confirm", score=state.intraday_score)


def activate_vwap_bounce_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if not bool(getattr(settings, "pb_v2_enabled", False)) or not bool(getattr(settings, "pb_v2_vwap_bounce_enabled", False)):
        return None
    if bool(getattr(state, "stopped_out_today", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_vwap_bounce_allow_rescue", False)):
        return None
    if getattr(state, "stage", "") != "WATCHING" or bar_idx < int(getattr(settings, "pb_v2_vwap_bounce_after_bar", 0)):
        return None
    vwap = market.session_vwap
    if vwap is None or session_atr <= 0:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    touched_below = any(sample.low < vwap for sample in series[: min(12, max(bar_idx, 0))])
    if not touched_below:
        return None
    if bar.close <= vwap or bar.close <= bar.open:
        return None
    if compute_volume_ratio(bar, item) < float(getattr(settings, "pb_v2_vwap_bounce_vol_ratio", 0.0)):
        return None
    micro = micropressure_label(series, min(bar_idx, len(series) - 1), vwap, item)
    if micro == "DISTRIBUTE":
        return None
    session_low = _state_or_market_session_low(state, market, bar)
    state.intraday_setup_type = "VWAP_BOUNCE"
    state.route_family = "VWAP_BOUNCE"
    state.setup_low = session_low
    state.reclaim_level = vwap
    state.stop_level = max(session_low - 0.25 * session_atr, 0.01)
    state.flush_bar_idx = 0
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series, micropressure=micro)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = bar.close
    state.improvement_expires = bar_idx + 2
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "vwap_bounce"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="vwap_bounce", score=state.intraday_score)


def activate_afternoon_retest_route(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if not bool(getattr(settings, "pb_v2_enabled", False)) or not bool(getattr(settings, "pb_v2_afternoon_retest_enabled", False)):
        return None
    if bool(getattr(state, "rescue_flow_candidate", False)) and not bool(getattr(settings, "pb_v2_afternoon_retest_allow_rescue", False)):
        return None
    if getattr(state, "stage", "") != "WATCHING" or bar_idx < int(getattr(settings, "pb_v2_afternoon_retest_after_bar", 0)):
        return None
    if float(getattr(state, "daily_signal_score", 0.0)) < float(getattr(settings, "pb_v2_afternoon_retest_min_score", 0.0)):
        return None
    vwap = market.session_vwap
    if vwap is None or session_atr <= 0:
        return None
    series = list(bars) if bars is not None else list(market.bars_5m)
    session_low = _state_or_market_session_low(state, market, bar)
    if bar.low < 0.95 * session_low:
        return None
    if bar.close <= vwap:
        return None
    avg_vol = float(sum(sample.volume for sample in series[: bar_idx + 1]) / max(bar_idx + 1, 1)) if series else 0.0
    if avg_vol > 0 and bar.volume > 1.5 * avg_vol:
        return None
    state.intraday_setup_type = "AFTERNOON_RETEST"
    state.route_family = "AFTERNOON_RETEST"
    state.setup_low = session_low
    state.reclaim_level = vwap
    state.stop_level = max(session_low - 0.40 * session_atr, 0.01)
    state.flush_bar_idx = 0
    state.acceptance_count = 1
    state.required_acceptance = 1
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    prior = "WATCHING"
    state.stage = "READY"
    state.ready_bar_idx = int(bar_idx)
    state.ready_cpr = float(bar.cpr)
    state.ready_volume_ratio = float(compute_volume_ratio(bar, item))
    state.ready_timestamp = bar.end_time
    state.target_entry_price = bar.close
    state.improvement_expires = bar_idx + 2
    if hasattr(state, "last_transition_reason"):
        state.last_transition_reason = "afternoon_retest"
    return IARICRouteStep(prior_stage=prior, stage="READY", reason="afternoon_retest", score=state.intraday_score)


def evaluate_ready_entry(
    settings: Any,
    state: Any,
    item: WatchlistItem,
    bar: Bar,
    market: MarketSnapshot,
    bar_idx: int,
    session_atr: float,
    *,
    bars: Sequence[Bar] | None = None,
) -> IARICRouteStep | None:
    if getattr(state, "stage", "") != "READY":
        return None
    if bar.low <= float(getattr(state, "stop_level", 0.0)):
        reset_bar = max(
            bar_idx + 2,
            int(getattr(settings, "pb_pm_reentry_after_bar", 0)) if bool(getattr(state, "stopped_out_today", False)) else bar_idx + 2,
        )
        return invalidate_route_state(state, "ready_stop_breach", reset_bar)
    series = list(bars) if bars is not None else list(market.bars_5m)
    if not series:
        series = [bar]
    score_bundle = compute_route_entry_score_bundle(settings, state, item, bar, market, bar_idx, bars=series)
    state.score_components = dict(score_bundle)
    state.intraday_score = float(score_bundle["score"])
    route_family_name = getattr(state, "route_family", "") or (
        "DELAYED_CONFIRM" if getattr(state, "intraday_setup_type", "") == "DELAYED_CONFIRM" else "OPENING_RECLAIM"
    )
    desired_entry = 0.0
    entry_trigger = ""
    if bar_idx > int(getattr(state, "ready_bar_idx", -1)) and bar_idx <= int(getattr(state, "improvement_expires", 0)) and bar.low <= float(getattr(state, "target_entry_price", 0.0)) <= bar.high:
        desired_entry = float(getattr(state, "target_entry_price", 0.0))
        entry_trigger = route_family_name
    elif (
        bar_idx > int(getattr(state, "ready_bar_idx", -1))
        and (
            bar_idx >= int(getattr(state, "improvement_expires", 0))
            or bar.close >= float(getattr(state, "reclaim_level", 0.0)) + session_atr * 0.25
        )
    ):
        desired_entry = max(bar.close, float(getattr(state, "reclaim_level", 0.0)))
        entry_trigger = route_family_name
    feasible = desired_entry > 0 and bool(entry_trigger)
    step = IARICRouteStep(prior_stage="READY", stage="READY", score=state.intraday_score, entry_feasible=feasible)
    if state.intraday_score < entry_threshold(settings, state) or not feasible:
        return step if feasible else None
    acceptance = IARICEntryAcceptance(
        accepted_bar_idx=int(bar_idx),
        accepted_timestamp=bar.end_time,
        accepted_entry_price=float(desired_entry),
        entry_trigger=str(entry_trigger),
        route_family=str(route_family_name),
        score=float(state.intraday_score),
        session_atr=float(session_atr),
        score_components=dict(state.score_components),
    )
    step.reason = "next_bar_open_fill"
    step.acceptance = acceptance
    return step


def on_bar(
    state: IARICCoreState,
    payload: IARICBarInput | None = None,
    *,
    bar_ts: datetime | None = None,
    entry_request: IARICEntryRequest | None = None,
    stop_update: IARICStopUpdateRequest | None = None,
    partial_exit_request: IARICPartialExitRequest | None = None,
    flatten_request: IARICFlattenRequest | None = None,
) -> tuple[
    IARICCoreState,
    list[SubmitEntry | ReplaceProtectiveStop | SubmitMarketExit | FlattenPosition | CancelAction],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitEntry | ReplaceProtectiveStop | SubmitMarketExit | FlattenPosition | CancelAction] = []
    events: list[DecisionEvent] = []

    if payload is not None and all(
        request is None
        for request in (entry_request, stop_update, partial_exit_request, flatten_request)
    ):
        events = _legacy_bar_events(payload)
        if payload.bar_ts is not None:
            _meta(next_state)["last_bar_ts"] = payload.bar_ts
        _update_last_decision(next_state, events)
        return next_state, [], events

    if bar_ts is not None:
        _meta(next_state)["last_bar_ts"] = bar_ts
    event_ts = bar_ts or datetime.now(timezone.utc)

    if entry_request is not None:
        actions.append(
            SubmitEntry(
                client_order_id=entry_request.client_order_id,
                symbol=entry_request.symbol,
                side="BUY",
                qty=entry_request.qty,
                order_type=entry_request.order_type,
                tif=entry_request.tif,
                limit_price=entry_request.limit_price,
                role="entry",
                risk_context={
                    "stop_for_risk": entry_request.stop_price,
                    "planned_entry_price": entry_request.limit_price,
                },
                metadata={
                    **entry_request.metadata,
                    "route": entry_request.route,
                },
            )
        )
        events.append(
            _event(
                code="ENTRY_REQUESTED",
                ts=event_ts,
                symbol=entry_request.symbol,
                details={
                    "qty": entry_request.qty,
                    "limit_price": entry_request.limit_price,
                    "stop_price": entry_request.stop_price,
                    "route": entry_request.route,
                },
            )
        )

    if stop_update is not None:
        symbol_state = _symbol_state(next_state, stop_update.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if symbol_state is not None and position is not None and position.stop_order_id:
            symbol_state.stop_level = stop_update.stop_price
            position.current_stop = stop_update.stop_price
            actions.append(
                ReplaceProtectiveStop(
                    symbol=stop_update.symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL",
                    stop_price=stop_update.stop_price,
                    qty=min(stop_update.qty, position.qty_open),
                    reason=stop_update.reason,
                )
            )
            events.append(
                _event(
                    code="STOP_REPLACEMENT_REQUESTED",
                    ts=event_ts,
                    symbol=stop_update.symbol,
                    details={
                        "stop_price": stop_update.stop_price,
                        "qty": min(stop_update.qty, position.qty_open),
                        "reason": stop_update.reason,
                    },
                )
            )

    if partial_exit_request is not None:
        symbol_state = _symbol_state(next_state, partial_exit_request.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if position is not None and position.qty_open > 0:
            actions.append(
                SubmitMarketExit(
                    client_order_id=partial_exit_request.client_order_id,
                    symbol=partial_exit_request.symbol,
                    side="SELL",
                    qty=min(partial_exit_request.qty, position.qty_open),
                    role="tp",
                    metadata={"reason": partial_exit_request.reason},
                )
            )
            events.append(
                _event(
                    code="PARTIAL_EXIT_REQUESTED",
                    ts=event_ts,
                    symbol=partial_exit_request.symbol,
                    details={
                        "qty": min(partial_exit_request.qty, position.qty_open),
                        "reason": partial_exit_request.reason,
                    },
                )
            )

    if flatten_request is not None:
        symbol_state = _symbol_state(next_state, flatten_request.symbol)
        position = symbol_state.position if symbol_state is not None else None
        if symbol_state is not None and position is not None and position.qty_open > 0:
            symbol_state.last_transition_reason = flatten_request.reason
            if symbol_state.exit_order is not None:
                if symbol_state.exit_order.role == "EXIT":
                    events.append(
                        _event(
                            code="FLATTEN_ALREADY_IN_FLIGHT",
                            ts=event_ts,
                            symbol=flatten_request.symbol,
                            details={"reason": flatten_request.reason},
                        )
                    )
                else:
                    symbol_state.pending_hard_exit = True
                    if not symbol_state.exit_order.cancel_requested:
                        symbol_state.exit_order.cancel_requested = True
                        actions.append(
                            CancelAction(
                                symbol=flatten_request.symbol,
                                target_order_id=symbol_state.exit_order.oms_order_id,
                                reason="hard_exit",
                            )
                        )
                    events.append(
                        _event(
                            code="FLATTEN_QUEUED_AFTER_CANCEL",
                            ts=event_ts,
                            symbol=flatten_request.symbol,
                            details={"reason": flatten_request.reason},
                        )
                    )
            else:
                actions.append(
                    FlattenPosition(
                        symbol=flatten_request.symbol,
                        reason=flatten_request.reason,
                        side="SELL",
                        qty=flatten_request.qty or position.qty_open,
                    )
                )
                events.append(
                    _event(
                        code="FLATTEN_REQUESTED",
                        ts=event_ts,
                        symbol=flatten_request.symbol,
                        details={
                            "reason": flatten_request.reason,
                            "qty": flatten_request.qty or position.qty_open,
                        },
                    )
                )

    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_order_update(
    state: IARICCoreState,
    update: IARICOrderUpdate,
) -> tuple[
    IARICCoreState,
    list[SubmitProtectiveStop | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitProtectiveStop | FlattenPosition] = []
    status = update.status.lower()
    symbol_state, symbol = _resolve_symbol_state(next_state, update.symbol, update.oms_order_id)
    role = _resolve_role(symbol_state, update.order_role, update.oms_order_id)
    event_ts = update.timestamp or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if not symbol:
        if update.decision_code:
            events.append(
                _event(
                    code=update.decision_code,
                    ts=event_ts,
                    symbol=update.symbol,
                    details=update.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if update.oms_order_id:
        _order_index(next_state).pop(update.oms_order_id, None)

    if status in _TERMINAL_STATUSES and symbol_state is not None:
        if role == "ENTRY":
            symbol_state.entry_order = None
            symbol_state.active_order_id = None
            _pending_entry_risk(next_state).pop(symbol, None)
            if not symbol_state.in_position:
                symbol_state.stage = "INVALIDATED"
            symbol_state.last_transition_reason = update.reason or "entry_terminal"
            events.append(
                _event(
                    code="ENTRY_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status},
                )
            )
        elif role in {"TP", "EXIT"}:
            position = symbol_state.position
            if symbol_state.exit_order is not None and symbol_state.exit_order.oms_order_id == update.oms_order_id:
                symbol_state.exit_order = None
            pending_hard_exit = symbol_state.pending_hard_exit
            symbol_state.pending_hard_exit = False
            if pending_hard_exit and position is not None and position.qty_open > 0:
                actions.append(
                    FlattenPosition(
                        symbol=symbol,
                        reason=symbol_state.last_transition_reason or "hard_exit",
                        side="SELL",
                        qty=position.qty_open,
                    )
                )
            elif (
                role == "EXIT"
                and position is not None
                and position.qty_open > 0
                and not position.stop_order_id
            ):
                actions.append(_submit_stop_action(symbol, position))
            events.append(
                _event(
                    code=f"{role}_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status},
                )
            )
        elif role == "STOP":
            position = symbol_state.position
            expected = _expected_stop_cancels(next_state)
            was_expected = update.oms_order_id in expected
            if was_expected:
                expected.discard(update.oms_order_id)
                _set_expected_stop_cancels(next_state, expected)
            if position is not None and position.stop_order_id == update.oms_order_id:
                position.stop_order_id = ""
            if not was_expected and position is not None and position.qty_open > 0:
                actions.append(
                    FlattenPosition(
                        symbol=symbol,
                        reason="stop_terminal",
                        side="SELL",
                        qty=position.qty_open,
                    )
                )
            events.append(
                _event(
                    code="STOP_TERMINAL",
                    ts=event_ts,
                    symbol=symbol,
                    details={"status": status, "expected_cancel": was_expected},
                )
            )

    if not events and update.decision_code:
        events.append(
            _event(
                code=update.decision_code,
                ts=event_ts,
                symbol=symbol,
                details=update.decision_details,
            )
        )
    _update_last_decision(next_state, events)
    return next_state, actions, events


def on_fill(
    state: IARICCoreState,
    fill: IARICFill,
) -> tuple[
    IARICCoreState,
    list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition],
    list[DecisionEvent],
]:
    next_state = deepcopy(state)
    actions: list[SubmitProtectiveStop | ReplaceProtectiveStop | FlattenPosition] = []
    symbol_state, symbol = _resolve_symbol_state(next_state, fill.symbol, fill.oms_order_id)
    role = _resolve_role(symbol_state, fill.order_role, fill.oms_order_id)
    event_ts = fill.fill_time or datetime.now(timezone.utc)
    events: list[DecisionEvent] = []

    if not symbol or symbol_state is None:
        if fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=fill.symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if fill.oms_order_id:
        _order_index(next_state).pop(fill.oms_order_id, None)

    if role == "ENTRY":
        symbol_state.entry_order = None
        symbol_state.active_order_id = None
        _pending_entry_risk(next_state).pop(symbol, None)
        stop_price = max(symbol_state.stop_level, 0.01)
        position = build_position_from_fill(
            fill_price=fill.fill_price,
            fill_qty=fill.fill_qty,
            stop_price=stop_price,
            fill_time=event_ts,
            setup_tag=f"PB_{symbol_state.route_family}",
        )
        position.entry_commission = fill.commission
        symbol_state.position = position
        symbol_state.in_position = True
        symbol_state.stage = "IN_POSITION"
        symbol_state.risk_per_share = max(fill.fill_price - stop_price, 0.01)
        actions.append(_submit_stop_action(symbol, position))
        events.append(
            _event(
                code="ENTRY_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={
                    "qty": fill.fill_qty,
                    "price": fill.fill_price,
                    "route": symbol_state.route_family,
                    "stop_price": stop_price,
                },
            )
        )
        if not events and fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    position = symbol_state.position
    if position is None or fill.fill_qty <= 0:
        if not events and fill.decision_code:
            events.append(
                _event(
                    code=fill.decision_code,
                    ts=event_ts,
                    symbol=symbol,
                    details=fill.decision_details,
                )
            )
        _update_last_decision(next_state, events)
        return next_state, actions, events

    if symbol_state.exit_order is not None and symbol_state.exit_order.oms_order_id == fill.oms_order_id:
        symbol_state.exit_order = None

    position.max_favorable_price = max(position.max_favorable_price, fill.fill_price)
    position.max_adverse_price = min(position.max_adverse_price, fill.fill_price)
    exit_qty = min(fill.fill_qty, position.qty_open)
    if exit_qty <= 0:
        _update_last_decision(next_state, events)
        return next_state, actions, events

    position.exit_commission += fill.commission
    position.realized_pnl_usd += (fill.fill_price - position.entry_price) * exit_qty
    position.qty_open = max(0, position.qty_open - exit_qty)

    if role == "TP":
        position.partial_taken = True
        symbol_state.v2_partial_taken = True
        if position.qty_open > 0 and position.stop_order_id:
            actions.append(
                ReplaceProtectiveStop(
                    symbol=symbol,
                    target_order_id=position.stop_order_id,
                    side="SELL",
                    stop_price=position.current_stop,
                    qty=position.qty_open,
                    reason="partial_resize",
                )
            )
        if symbol_state.pending_hard_exit and position.qty_open > 0:
            symbol_state.pending_hard_exit = False
            actions.append(
                FlattenPosition(
                    symbol=symbol,
                    reason=symbol_state.last_transition_reason or "hard_exit",
                    side="SELL",
                    qty=position.qty_open,
                )
            )
        events.append(
            _event(
                code="PARTIAL_EXIT_FILLED" if position.qty_open > 0 else "EXIT_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={
                    "qty": exit_qty,
                    "price": fill.fill_price,
                    "reason": fill.exit_type or "TP",
                },
            )
        )
    elif role == "EXIT":
        if position.qty_open > 0 and not position.stop_order_id:
            actions.append(_submit_stop_action(symbol, position))
            events.append(
                _event(
                    code="EXIT_PARTIALLY_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={"qty": exit_qty, "price": fill.fill_price},
                )
            )
        else:
            events.append(
                _event(
                    code="EXIT_FILLED",
                    ts=event_ts,
                    symbol=symbol,
                    details={
                        "qty": exit_qty,
                        "price": fill.fill_price,
                        "reason": symbol_state.last_transition_reason or fill.exit_type or "EXIT",
                    },
                )
            )
    elif role == "STOP":
        position.stop_order_id = ""
        if position.qty_open > 0:
            actions.append(
                FlattenPosition(
                    symbol=symbol,
                    reason="stop_unprotected",
                    side="SELL",
                    qty=position.qty_open,
                )
            )
        events.append(
            _event(
                code="STOP_FILLED",
                ts=event_ts,
                symbol=symbol,
                details={"qty": exit_qty, "price": fill.fill_price},
            )
        )

    if position.qty_open <= 0:
        reason = symbol_state.last_transition_reason or fill.exit_type or role
        if role == "STOP" or "STOP" in reason.upper():
            symbol_state.stopped_out_today = True
        symbol_state.position = None
        symbol_state.in_position = False
        symbol_state.stage = "INVALIDATED"
        symbol_state.exit_order = None
        symbol_state.pending_hard_exit = False

    if not events and fill.decision_code:
        events.append(
            _event(
                code=fill.decision_code,
                ts=event_ts,
                symbol=symbol,
                details=fill.decision_details,
            )
        )
    _update_last_decision(next_state, events)
    return next_state, actions, events


def _legacy_bar_events(payload: IARICBarInput) -> list[DecisionEvent]:
    if not payload.decision_code:
        return []
    return [
        _event(
            code=payload.decision_code,
            ts=payload.bar_ts or datetime.now(timezone.utc),
            symbol=payload.symbol,
            details=payload.decision_details,
        )
    ]


def _symbol_state(state: IARICCoreState, symbol: str) -> PBSymbolState | None:
    for symbol_state in state.symbols:
        if symbol_state.symbol == symbol:
            return symbol_state
    return None


def _resolve_symbol_state(
    state: IARICCoreState,
    symbol: str,
    oms_order_id: str,
) -> tuple[PBSymbolState | None, str]:
    if symbol:
        symbol_state = _symbol_state(state, symbol)
        if symbol_state is not None:
            return symbol_state, symbol
    if oms_order_id:
        for symbol_state in state.symbols:
            if symbol_state.entry_order and symbol_state.entry_order.oms_order_id == oms_order_id:
                return symbol_state, symbol_state.symbol
            if symbol_state.exit_order and symbol_state.exit_order.oms_order_id == oms_order_id:
                return symbol_state, symbol_state.symbol
            if (
                symbol_state.position is not None
                and symbol_state.position.stop_order_id == oms_order_id
            ):
                return symbol_state, symbol_state.symbol
    return None, ""


def _resolve_role(
    symbol_state: PBSymbolState | None,
    explicit_role: str,
    oms_order_id: str,
) -> str:
    if explicit_role and explicit_role != "UNKNOWN":
        return explicit_role
    if symbol_state is None:
        return "UNKNOWN"
    if symbol_state.entry_order and symbol_state.entry_order.oms_order_id == oms_order_id:
        return "ENTRY"
    if symbol_state.exit_order and symbol_state.exit_order.oms_order_id == oms_order_id:
        return str(symbol_state.exit_order.role or "EXIT").upper()
    if symbol_state.position and symbol_state.position.stop_order_id == oms_order_id:
        return "STOP"
    return "UNKNOWN"


def _event(
    *,
    code: str,
    ts: datetime,
    symbol: str,
    details: dict[str, Any],
) -> DecisionEvent:
    return DecisionEvent(code=code, ts=ts, symbol=symbol, timeframe="5m", details=dict(details))


def _meta(state: IARICCoreState) -> dict[str, Any]:
    if not isinstance(state.meta, dict):
        state.meta = {}
    return state.meta


def _order_index(state: IARICCoreState) -> dict[str, tuple[str, str]]:
    raw = dict(_meta(state).get("order_index", {}))
    normalized = {
        str(order_id): _coerce_order_index_entry(value)
        for order_id, value in raw.items()
    }
    _meta(state)["order_index"] = normalized
    return normalized


def _pending_entry_risk(state: IARICCoreState) -> dict[str, float]:
    raw = dict(_meta(state).get("pending_entry_risk", {}))
    normalized = {str(symbol): float(risk) for symbol, risk in raw.items()}
    _meta(state)["pending_entry_risk"] = normalized
    return normalized


def _expected_stop_cancels(state: IARICCoreState) -> set[str]:
    return {str(order_id) for order_id in _meta(state).get("expected_stop_cancels", [])}


def _set_expected_stop_cancels(state: IARICCoreState, order_ids: set[str]) -> None:
    _meta(state)["expected_stop_cancels"] = sorted(order_ids)


def _submit_stop_action(symbol: str, position: PositionState) -> SubmitProtectiveStop:
    return SubmitProtectiveStop(
        client_order_id=f"{symbol}-stop-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        symbol=symbol,
        side="SELL",
        qty=position.qty_open,
        stop_price=position.current_stop,
    )


def _update_last_decision(state: IARICCoreState, events: list[DecisionEvent]) -> None:
    if not events:
        return
    latest = events[-1]
    state.last_decision_code = latest.code
    meta = _meta(state)
    meta["last_decision_details"] = dict(latest.details)
    if latest.ts is not None:
        meta["last_bar_ts"] = meta.get("last_bar_ts", latest.ts)


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _coerce_order_index_entry(value: Any) -> tuple[str, str]:
    if isinstance(value, tuple) and len(value) == 2:
        return str(value[0]), str(value[1])
    if isinstance(value, list) and len(value) == 2:
        return str(value[0]), str(value[1])
    return "", ""
