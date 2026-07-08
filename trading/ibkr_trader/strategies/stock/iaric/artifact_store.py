"""File-backed storage for IARIC research, artifacts, and live state."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import StrategySettings
from .models import (
    HeldPositionDirective,
    HeldPositionResearch,
    IntradayStateSnapshot,
    MarketResearch,
    PBSymbolState,
    PendingOrderState,
    PositionState,
    RegimeSnapshot,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
    SectorResearch,
    SymbolIntradayState,
    WatchlistArtifact,
    WatchlistItem,
)


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def research_snapshot_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.research_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def artifact_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.artifact_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def state_path(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = root or cfg.state_dir
    return Path(base) / f"{trade_date.isoformat()}.json"


def load_research_snapshot(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> ResearchSnapshot:
    payload = _read_json(research_snapshot_path(trade_date, root=root, settings=settings))
    sectors = {
        name: SectorResearch(
            name=name,
            flow_trend_20d=float(data["flow_trend_20d"]),
            breadth_20d=float(data["breadth_20d"]),
            participation=float(data["participation"]),
        )
        for name, data in payload["sectors"].items()
    }
    symbols: dict[str, ResearchSymbol] = {}
    for symbol, data in payload["symbols"].items():
        daily_bars = [
            ResearchDailyBar(
                trade_date=date.fromisoformat(bar["trade_date"]),
                open=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                volume=float(bar["volume"]),
                event_tag=str(bar.get("event_tag", "")),
            )
            for bar in data.get("daily_bars", [])
        ]
        symbols[symbol] = ResearchSymbol(
            symbol=symbol,
            exchange=str(data["exchange"]),
            primary_exchange=str(data["primary_exchange"]),
            currency=str(data["currency"]),
            tick_size=float(data["tick_size"]),
            point_value=float(data.get("point_value", 1.0)),
            sector=str(data.get("sector", "")),
            price=float(data["price"]),
            adv20_usd=float(data["adv20_usd"]),
            median_spread_pct=float(data.get("median_spread_pct", 0.0)),
            earnings_within_sessions=(
                int(data["earnings_within_sessions"])
                if data.get("earnings_within_sessions") is not None
                else None
            ),
            blacklist_flag=bool(data.get("blacklist_flag", False)),
            halted_flag=bool(data.get("halted_flag", False)),
            severe_news_flag=bool(data.get("severe_news_flag", False)),
            etf_flag=bool(data.get("etf_flag", False)),
            adr_flag=bool(data.get("adr_flag", False)),
            preferred_flag=bool(data.get("preferred_flag", False)),
            otc_flag=bool(data.get("otc_flag", False)),
            hard_to_borrow_flag=bool(data.get("hard_to_borrow_flag", False)),
            flow_proxy_history=[float(value) for value in data.get("flow_proxy_history", [])],
            daily_bars=daily_bars,
            sector_return_20d=float(data.get("sector_return_20d", 0.0)),
            sector_return_60d=float(data.get("sector_return_60d", 0.0)),
            intraday_atr_seed=float(data.get("intraday_atr_seed", 0.0)),
            average_30m_volume=float(data.get("average_30m_volume", 0.0)),
            expected_5m_volume=float(data.get("expected_5m_volume", 0.0)),
        )
    held_positions = [
        HeldPositionResearch(
            symbol=str(item["symbol"]),
            entry_time=datetime.fromisoformat(item["entry_time"]),
            entry_price=float(item["entry_price"]),
            size=int(item["size"]),
            stop=float(item["stop"]),
            initial_r=float(item["initial_r"]),
            setup_tag=str(item.get("setup_tag", "")),
            carry_eligible_flag=bool(item.get("carry_eligible_flag", False)),
        )
        for item in payload.get("held_positions", [])
    ]
    market = payload["market"]
    return ResearchSnapshot(
        trade_date=date.fromisoformat(payload["trade_date"]),
        market=MarketResearch(
            price_ok=bool(market["price_ok"]),
            breadth_pct_above_20dma=float(market["breadth_pct_above_20dma"]),
            vix_percentile_1y=float(market["vix_percentile_1y"]),
            hy_spread_5d_bps_change=float(market["hy_spread_5d_bps_change"]),
            market_wide_institutional_selling=bool(market.get("market_wide_institutional_selling", False)),
        ),
        sectors=sectors,
        symbols=symbols,
        held_positions=held_positions,
    )


def persist_watchlist_artifact(artifact: WatchlistArtifact, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    payload = {
        "trade_date": artifact.trade_date.isoformat(),
        "generated_at": artifact.generated_at.isoformat(),
        "regime": _serialize(asdict(artifact.regime)),
        "items": [_serialize(asdict(item)) for item in artifact.items],
        "tradable": [_serialize(asdict(item)) for item in artifact.tradable],
        "overflow": [_serialize(asdict(item)) for item in artifact.overflow],
        "market_wide_institutional_selling": artifact.market_wide_institutional_selling,
        "held_positions": [_serialize(asdict(item)) for item in artifact.held_positions],
    }
    path = artifact_path(artifact.trade_date, root=root, settings=settings)
    _write_json(path, payload)
    return path


def _watchlist_item_from_dict(data: dict[str, Any]) -> WatchlistItem:
    return WatchlistItem(
        symbol=str(data["symbol"]),
        exchange=str(data["exchange"]),
        primary_exchange=str(data["primary_exchange"]),
        currency=str(data["currency"]),
        tick_size=float(data["tick_size"]),
        point_value=float(data["point_value"]),
        sector=str(data["sector"]),
        regime_score=float(data["regime_score"]),
        regime_tier=str(data["regime_tier"]),
        regime_risk_multiplier=float(data["regime_risk_multiplier"]),
        sector_score=float(data["sector_score"]),
        sector_rank_weight=float(data["sector_rank_weight"]),
        sponsorship_score=float(data["sponsorship_score"]),
        sponsorship_state=str(data["sponsorship_state"]),
        persistence=float(data["persistence"]),
        intensity_z=float(data["intensity_z"]),
        accel_z=float(data["accel_z"]),
        rs_percentile=float(data["rs_percentile"]),
        leader_pass=bool(data["leader_pass"]),
        trend_pass=bool(data["trend_pass"]),
        trend_strength=float(data["trend_strength"]),
        earnings_risk_flag=bool(data["earnings_risk_flag"]),
        blacklist_flag=bool(data["blacklist_flag"]),
        anchor_date=date.fromisoformat(data["anchor_date"]),
        anchor_type=str(data["anchor_type"]),
        acceptance_pass=bool(data["acceptance_pass"]),
        avwap_ref=float(data["avwap_ref"]),
        avwap_band_lower=float(data["avwap_band_lower"]),
        avwap_band_upper=float(data["avwap_band_upper"]),
        daily_atr_estimate=float(data["daily_atr_estimate"]),
        intraday_atr_seed=float(data["intraday_atr_seed"]),
        daily_rank=float(data["daily_rank"]),
        tradable_flag=bool(data["tradable_flag"]),
        conviction_bucket=str(data["conviction_bucket"]),
        conviction_multiplier=float(data["conviction_multiplier"]),
        recommended_risk_r=float(data["recommended_risk_r"]),
        average_30m_volume=float(data.get("average_30m_volume", 0.0)),
        expected_5m_volume=float(data.get("expected_5m_volume", 0.0)),
        entry_gap_pct=float(data.get("entry_gap_pct", 0.0)),
        flow_proxy_gate_pass=bool(data.get("flow_proxy_gate_pass", True)),
        overflow_rank=int(data["overflow_rank"]) if data.get("overflow_rank") is not None else None,
        # Pullback V2 fields
        daily_signal_score=float(data.get("daily_signal_score", 0.0)),
        trigger_types=list(data.get("trigger_types", [])),
        trigger_tier=str(data.get("trigger_tier", "STANDARD")),
        trend_tier=str(data.get("trend_tier", "STRONG")),
        rescue_flow_candidate=bool(data.get("rescue_flow_candidate", False)),
        sizing_mult=float(data.get("sizing_mult", 1.0)),
        ema10_daily=float(data.get("ema10_daily", 0.0)),
        rsi14_daily=float(data.get("rsi14_daily", 0.0)),
    )


def load_watchlist_artifact(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> WatchlistArtifact:
    payload = _read_json(artifact_path(trade_date, root=root, settings=settings))
    regime_data = payload["regime"]
    return WatchlistArtifact(
        trade_date=date.fromisoformat(payload["trade_date"]),
        generated_at=datetime.fromisoformat(payload["generated_at"]),
        regime=RegimeSnapshot(
            score=float(regime_data["score"]),
            tier=str(regime_data["tier"]),
            risk_multiplier=float(regime_data["risk_multiplier"]),
            price_ok=bool(regime_data["price_ok"]),
            breadth_ok=bool(regime_data["breadth_ok"]),
            vol_ok=bool(regime_data["vol_ok"]),
            credit_ok=bool(regime_data["credit_ok"]),
        ),
        items=[_watchlist_item_from_dict(data) for data in payload.get("items", [])],
        tradable=[_watchlist_item_from_dict(data) for data in payload.get("tradable", [])],
        overflow=[_watchlist_item_from_dict(data) for data in payload.get("overflow", [])],
        market_wide_institutional_selling=bool(payload.get("market_wide_institutional_selling", False)),
        held_positions=[
            HeldPositionDirective(
                symbol=str(data["symbol"]),
                entry_time=datetime.fromisoformat(data["entry_time"]),
                entry_price=float(data["entry_price"]),
                size=int(data["size"]),
                stop=float(data["stop"]),
                initial_r=float(data["initial_r"]),
                setup_tag=str(data.get("setup_tag", "")),
                time_stop_deadline=(
                    datetime.fromisoformat(data["time_stop_deadline"])
                    if data.get("time_stop_deadline")
                    else None
                ),
                carry_eligible_flag=bool(data.get("carry_eligible_flag", False)),
                flow_reversal_flag=bool(data.get("flow_reversal_flag", False)),
            )
            for data in payload.get("held_positions", [])
        ],
    )


def persist_intraday_state(snapshot: IntradayStateSnapshot, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    path = state_path(snapshot.trade_date, root=root, settings=settings)
    payload = {
        "trade_date": snapshot.trade_date.isoformat(),
        "saved_at": snapshot.saved_at.isoformat(),
        "symbols": [_serialize(asdict(symbol)) for symbol in snapshot.symbols],
        "last_decision_code": snapshot.last_decision_code,
        "meta": _serialize(snapshot.meta),
    }
    _write_json(path, payload)
    return path


def coerce_intraday_state_snapshot(
    payload: IntradayStateSnapshot | dict[str, Any],
) -> IntradayStateSnapshot:
    if isinstance(payload, IntradayStateSnapshot):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            "Intraday state payload must be an IntradayStateSnapshot or dict"
        )

    def _as_datetime(value: datetime | str | None) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    def _as_date(value: date | str) -> date:
        if isinstance(value, date):
            return value
        return date.fromisoformat(value)

    def _pending(data: dict[str, Any] | None) -> PendingOrderState | None:
        if not data:
            return None
        return PendingOrderState(
            oms_order_id=str(data["oms_order_id"]),
            submitted_at=_as_datetime(data["submitted_at"]),
            role=str(data["role"]),
            requested_qty=int(data["requested_qty"]),
            limit_price=float(data["limit_price"]) if data.get("limit_price") is not None else None,
            stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
            cancel_requested=bool(data.get("cancel_requested", False)),
        )

    def _position(data: dict[str, Any] | None) -> PositionState | None:
        if not data:
            return None
        return PositionState(
            entry_price=float(data["entry_price"]),
            qty_entry=int(data["qty_entry"]),
            qty_open=int(data["qty_open"]),
            final_stop=float(data["final_stop"]),
            current_stop=float(data["current_stop"]),
            entry_time=_as_datetime(data["entry_time"]),
            initial_risk_per_share=float(data["initial_risk_per_share"]),
            max_favorable_price=float(data["max_favorable_price"]),
            max_adverse_price=float(data.get("max_adverse_price", data["entry_price"])),
            partial_taken=bool(data.get("partial_taken", False)),
            stop_order_id=str(data.get("stop_order_id", "")),
            trade_id=str(data.get("trade_id", "")),
            realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
            setup_tag=str(data.get("setup_tag", "UNCLASSIFIED")),
            time_stop_deadline=_as_datetime(data.get("time_stop_deadline")),
        )

    def _pb_symbol(data: dict[str, Any]) -> PBSymbolState:
        return PBSymbolState(
            symbol=str(data["symbol"]),
            stage=str(data.get("stage", "WATCHING")),
            route_family=str(data.get("route_family", "")),
            setup_low=float(data.get("setup_low", 0.0)),
            reclaim_level=float(data.get("reclaim_level", 0.0)),
            stop_level=float(data.get("stop_level", 0.0)),
            acceptance_count=int(data.get("acceptance_count", 0)),
            required_acceptance=int(data.get("required_acceptance", 1)),
            intraday_score=float(data.get("intraday_score", 0.0)),
            score_components=dict(data.get("score_components", {})),
            bars_seen_today=int(data.get("bars_seen_today", 0)),
            session_low=float(data.get("session_low", 0.0)),
            session_high=float(data.get("session_high", 0.0)),
            in_position=bool(data.get("in_position", False)),
            position=_position(data.get("position")),
            entry_order=_pending(data.get("entry_order")),
            exit_order=_pending(data.get("exit_order")),
            pending_hard_exit=bool(data.get("pending_hard_exit", False)),
            daily_signal_score=float(data.get("daily_signal_score", 0.0)),
            trigger_types=list(data.get("trigger_types", [])),
            trigger_tier=str(data.get("trigger_tier", "STANDARD")),
            trend_tier=str(data.get("trend_tier", "STRONG")),
            rescue_flow_candidate=bool(data.get("rescue_flow_candidate", False)),
            sizing_mult=float(data.get("sizing_mult", 1.0)),
            daily_atr=float(data.get("daily_atr", 0.0)),
            entry_atr=float(data.get("entry_atr", 0.0)),
            last_1m_bar_time=_as_datetime(data.get("last_1m_bar_time")),
            last_5m_bar_time=_as_datetime(data.get("last_5m_bar_time")),
            active_order_id=str(data["active_order_id"]) if data.get("active_order_id") else None,
            last_transition_reason=str(data.get("last_transition_reason", "")),
            mfe_stage=int(data.get("mfe_stage", 0)),
            breakeven_activated=bool(data.get("breakeven_activated", False)),
            trail_active=bool(data.get("trail_active", False)),
            hold_bars=int(data.get("hold_bars", 0)),
            risk_per_share=float(data.get("risk_per_share", 0.0)),
            v2_partial_taken=bool(data.get("v2_partial_taken", False)),
            carry_decision_path=str(data.get("carry_decision_path", "")),
            consecutive_bars_below_vwap=int(data.get("consecutive_bars_below_vwap", 0)),
            ema10_daily=float(data.get("ema10_daily", 0.0)),
            rsi14_daily=float(data.get("rsi14_daily", 0.0)),
            stopped_out_today=bool(data.get("stopped_out_today", False)),
            flush_bar_idx=int(data.get("flush_bar_idx", 0)),
            ready_bar_idx=int(data.get("ready_bar_idx", -1)),
            target_entry_price=float(data.get("target_entry_price", 0.0)),
            improvement_expires=int(data.get("improvement_expires", 0)),
            invalid_reason=str(data.get("invalid_reason", "")),
            invalid_reset_bar=int(data.get("invalid_reset_bar", 0)),
            ready_cpr=float(data.get("ready_cpr", 0.0)),
            ready_volume_ratio=float(data.get("ready_volume_ratio", 0.0)),
            ready_timestamp=_as_datetime(data.get("ready_timestamp")),
            accepted_bar_idx=int(data.get("accepted_bar_idx", -1)),
            accepted_timestamp=_as_datetime(data.get("accepted_timestamp")),
            accepted_entry_price=float(data.get("accepted_entry_price", 0.0)),
            accepted_entry_trigger=str(data.get("accepted_entry_trigger", "")),
            accepted_route_family=str(data.get("accepted_route_family", "")),
            accepted_score=float(data.get("accepted_score", 0.0)),
            accepted_session_atr=float(data.get("accepted_session_atr", 0.0)),
            accepted_score_components=dict(data.get("accepted_score_components", {})),
        )

    symbols = []
    for data in payload.get("symbols", []):
        if "stage" in data:
            symbols.append(_pb_symbol(data))
        else:
            symbols.append(
                SymbolIntradayState(
                    symbol=str(data["symbol"]),
                    tier=str(data.get("tier", "COLD")),
                    fsm_state=str(data.get("fsm_state", "IDLE")),
                    in_position=bool(data.get("in_position", False)),
                    position_qty=int(data.get("position_qty", 0)),
                    avg_price=float(data["avg_price"]) if data.get("avg_price") is not None else None,
                    setup_type=str(data["setup_type"]) if data.get("setup_type") else None,
                    setup_low=float(data["setup_low"]) if data.get("setup_low") is not None else None,
                    reclaim_level=float(data["reclaim_level"]) if data.get("reclaim_level") is not None else None,
                    stop_level=float(data["stop_level"]) if data.get("stop_level") is not None else None,
                    setup_time=datetime.fromisoformat(data["setup_time"]) if data.get("setup_time") else None,
                    invalidated_at=datetime.fromisoformat(data["invalidated_at"]) if data.get("invalidated_at") else None,
                    acceptance_count=int(data.get("acceptance_count", 0)),
                    required_acceptance_count=int(data.get("required_acceptance_count", 0)),
                    location_grade=str(data["location_grade"]) if data.get("location_grade") else None,
                    session_vwap=float(data["session_vwap"]) if data.get("session_vwap") is not None else None,
                    avwap_live=float(data["avwap_live"]) if data.get("avwap_live") is not None else None,
                    sponsorship_signal=str(data.get("sponsorship_signal", "NEUTRAL")),
                    micropressure_signal=str(data.get("micropressure_signal", "NEUTRAL")),
                    micropressure_mode=str(data.get("micropressure_mode", "PROXY")),
                    flowproxy_signal=str(data.get("flowproxy_signal", "UNAVAILABLE")),
                    confidence=str(data["confidence"]) if data.get("confidence") else None,
                    last_1m_bar_time=_as_datetime(data.get("last_1m_bar_time")),
                    last_5m_bar_time=_as_datetime(data.get("last_5m_bar_time")),
                    active_order_id=str(data["active_order_id"]) if data.get("active_order_id") else None,
                    time_stop_deadline=_as_datetime(data.get("time_stop_deadline")),
                    setup_tag=str(data["setup_tag"]) if data.get("setup_tag") else None,
                    expected_volume_pct=float(data.get("expected_volume_pct", 0.0)),
                    average_30m_volume=float(data.get("average_30m_volume", 0.0)),
                    last_transition_reason=str(data.get("last_transition_reason", "")),
                    entry_order=_pending(data.get("entry_order")),
                    position=_position(data.get("position")),
                    exit_order=_pending(data.get("exit_order")),
                    pending_hard_exit=bool(data.get("pending_hard_exit", False)),
                )
            )

    return IntradayStateSnapshot(
        trade_date=_as_date(payload["trade_date"]),
        saved_at=_as_datetime(payload["saved_at"]),
        symbols=symbols,
        last_decision_code=str(payload.get("last_decision_code", "")),
        meta=dict(payload.get("meta", {})),
    )


def load_intraday_state(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> IntradayStateSnapshot:
    payload = _read_json(state_path(trade_date, root=root, settings=settings))
    return coerce_intraday_state_snapshot(payload)
