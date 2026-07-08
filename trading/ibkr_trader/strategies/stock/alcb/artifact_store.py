"""File-backed storage for ALCB research, artifacts, and live state."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .config import StrategySettings
from .models import (
    Bar,
    Box,
    BreakoutQualification,
    Campaign,
    CampaignState,
    CandidateArtifact,
    CandidateItem,
    CompressionTier,
    Direction,
    EntryType,
    HeldPositionResearch,
    IntradayStateSnapshot,
    MarketResearch,
    MarketSnapshot,
    PendingOrderState,
    PortfolioState,
    PositionState,
    QuoteSnapshot,
    Regime,
    RegimeSnapshot,
    ResearchDailyBar,
    ResearchSnapshot,
    ResearchSymbol,
    SectorResearch,
    SymbolRuntimeState,
)


def _serialize(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return _serialize(asdict(value))
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


def _daily_bar_from_dict(data: dict[str, Any]) -> ResearchDailyBar:
    return ResearchDailyBar(
        trade_date=date.fromisoformat(data["trade_date"]),
        open=float(data["open"]),
        high=float(data["high"]),
        low=float(data["low"]),
        close=float(data["close"]),
        volume=float(data["volume"]),
        event_tag=str(data.get("event_tag", "")),
    )


def _bar_from_dict(data: dict[str, Any]) -> Bar:
    return Bar(
        symbol=str(data["symbol"]),
        start_time=datetime.fromisoformat(data["start_time"]),
        end_time=datetime.fromisoformat(data["end_time"]),
        open=float(data["open"]),
        high=float(data["high"]),
        low=float(data["low"]),
        close=float(data["close"]),
        volume=float(data["volume"]),
    )


def _order_from_dict(data: dict[str, Any] | None) -> PendingOrderState | None:
    if not data:
        return None
    return PendingOrderState(
        oms_order_id=str(data["oms_order_id"]),
        submitted_at=datetime.fromisoformat(data["submitted_at"]),
        role=str(data["role"]),
        requested_qty=int(data["requested_qty"]),
        filled_qty=int(data.get("filled_qty", 0)),
        limit_price=float(data["limit_price"]) if data.get("limit_price") is not None else None,
        stop_price=float(data["stop_price"]) if data.get("stop_price") is not None else None,
        direction=(Direction(str(data["direction"])) if data.get("direction") else None),
        entry_type=(EntryType(str(data["entry_type"])) if data.get("entry_type") else None),
        entry_price=float(data["entry_price"]) if data.get("entry_price") is not None else None,
        planned_stop_price=(
            float(data["planned_stop_price"]) if data.get("planned_stop_price") is not None else None
        ),
        planned_tp1_price=(
            float(data["planned_tp1_price"]) if data.get("planned_tp1_price") is not None else None
        ),
        planned_tp2_price=(
            float(data["planned_tp2_price"]) if data.get("planned_tp2_price") is not None else None
        ),
        risk_per_share=float(data["risk_per_share"]) if data.get("risk_per_share") is not None else None,
        risk_dollars=float(data["risk_dollars"]) if data.get("risk_dollars") is not None else None,
        cancel_requested=bool(data.get("cancel_requested", False)),
    )


def _position_from_dict(data: dict[str, Any] | None) -> PositionState | None:
    if not data:
        return None
    return PositionState(
        direction=Direction(str(data["direction"])),
        entry_price=float(data["entry_price"]),
        qty_entry=int(data["qty_entry"]),
        qty_open=int(data["qty_open"]),
        final_stop=float(data["final_stop"]),
        current_stop=float(data["current_stop"]),
        entry_time=datetime.fromisoformat(data["entry_time"]),
        initial_risk_per_share=float(data["initial_risk_per_share"]),
        max_favorable_price=float(data["max_favorable_price"]),
        max_adverse_price=float(data["max_adverse_price"]),
        tp1_price=float(data["tp1_price"]),
        tp2_price=float(data["tp2_price"]),
        partial_taken=bool(data.get("partial_taken", False)),
        tp2_taken=bool(data.get("tp2_taken", False)),
        profit_funded=bool(data.get("profit_funded", False)),
        stop_order_id=str(data.get("stop_order_id", "")),
        stop_submitted_at=(
            datetime.fromisoformat(data["stop_submitted_at"])
            if data.get("stop_submitted_at")
            else None
        ),
        tp1_order_id=str(data.get("tp1_order_id", "")),
        tp2_order_id=str(data.get("tp2_order_id", "")),
        trade_id=str(data.get("trade_id", "")),
        realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
        entry_commission=float(data.get("entry_commission", 0.0)),
        exit_commission=float(data.get("exit_commission", 0.0)),
        setup_tag=str(data.get("setup_tag", "UNCLASSIFIED")),
        stale_warning_emitted=bool(data.get("stale_warning_emitted", False)),
        opened_trade_date=date.fromisoformat(data["opened_trade_date"]) if data.get("opened_trade_date") else None,
        exit_oca_group=str(data.get("exit_oca_group", "")),
    )


def _box_from_dict(data: dict[str, Any] | None) -> Box | None:
    if not data:
        return None
    return Box(
        start_date=str(data["start_date"]),
        end_date=str(data["end_date"]),
        L_used=int(data["L_used"]),
        high=float(data["high"]),
        low=float(data["low"]),
        mid=float(data["mid"]),
        height=float(data["height"]),
        containment=float(data["containment"]),
        squeeze_metric=float(data["squeeze_metric"]),
        tier=CompressionTier(str(data["tier"])),
    )


def _breakout_from_dict(data: dict[str, Any] | None) -> BreakoutQualification | None:
    if not data:
        return None
    return BreakoutQualification(
        direction=Direction(str(data["direction"])),
        breakout_date=str(data["breakout_date"]),
        structural_pass=bool(data["structural_pass"]),
        displacement_pass=bool(data["displacement_pass"]),
        disp_value=float(data["disp_value"]),
        disp_threshold=float(data["disp_threshold"]),
        breakout_rejected=bool(data["breakout_rejected"]),
        rvol_d=float(data["rvol_d"]),
        score_components={str(k): float(v) for k, v in (data.get("score_components") or {}).items()},
    )


def _campaign_from_dict(data: dict[str, Any]) -> Campaign:
    return Campaign(
        symbol=str(data["symbol"]),
        state=CampaignState(str(data.get("state", CampaignState.INACTIVE.value))),
        campaign_id=int(data.get("campaign_id", 0)),
        box_version=int(data.get("box_version", 0)),
        box=_box_from_dict(data.get("box")),
        avwap_anchor_ts=str(data["avwap_anchor_ts"]) if data.get("avwap_anchor_ts") else None,
        breakout=_breakout_from_dict(data.get("breakout")),
        dirty_since=str(data["dirty_since"]) if data.get("dirty_since") else None,
        add_count=int(data.get("add_count", 0)),
        campaign_risk_used=float(data.get("campaign_risk_used", 0.0)),
        profit_funded=bool(data.get("profit_funded", False)),
        position_open=bool(data.get("position_open", False)),
        continuation_enabled=bool(data.get("continuation_enabled", False)),
        reentry_block_same_direction=bool(data.get("reentry_block_same_direction", False)),
        last_entry_type=(
            None if data.get("last_entry_type") is None else EntryType(str(data["last_entry_type"]))
        ),
    )


def _market_snapshot_from_dict(data: dict[str, Any]) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=str(data["symbol"]),
        last_price=float(data["last_price"]) if data.get("last_price") is not None else None,
        bid=float(data.get("bid", 0.0)),
        ask=float(data.get("ask", 0.0)),
        spread_pct=float(data.get("spread_pct", 0.0)),
        session_vwap=float(data["session_vwap"]) if data.get("session_vwap") is not None else None,
        weekly_vwap=float(data["weekly_vwap"]) if data.get("weekly_vwap") is not None else None,
        avwap_live=float(data["avwap_live"]) if data.get("avwap_live") is not None else None,
        last_quote=(
            None
            if data.get("last_quote") is None
            else QuoteSnapshot(
                ts=datetime.fromisoformat(data["last_quote"]["ts"]),
                bid=float(data["last_quote"]["bid"]),
                ask=float(data["last_quote"]["ask"]),
                last=float(data["last_quote"]["last"]),
                bid_size=float(data["last_quote"].get("bid_size", 0.0)),
                ask_size=float(data["last_quote"].get("ask_size", 0.0)),
                cumulative_volume=float(data["last_quote"].get("cumulative_volume", 0.0)),
                cumulative_value=float(data["last_quote"].get("cumulative_value", 0.0)),
                vwap=float(data["last_quote"]["vwap"]) if data["last_quote"].get("vwap") is not None else None,
                is_halted=bool(data["last_quote"].get("is_halted", False)),
                spread_pct=float(data["last_quote"].get("spread_pct", 0.0)),
            )
        ),
        last_1m_bar=_bar_from_dict(data["last_1m_bar"]) if data.get("last_1m_bar") else None,
        last_30m_bar=_bar_from_dict(data["last_30m_bar"]) if data.get("last_30m_bar") else None,
        last_4h_bar=_bar_from_dict(data["last_4h_bar"]) if data.get("last_4h_bar") else None,
        minute_bars=[_bar_from_dict(item) for item in data.get("minute_bars", [])],
        bars_30m=[_bar_from_dict(item) for item in data.get("bars_30m", [])],
        bars_4h=[_bar_from_dict(item) for item in data.get("bars_4h", [])],
        daily_bars=[_daily_bar_from_dict(item) for item in data.get("daily_bars", [])],
    )


def _runtime_state_from_dict(data: dict[str, Any]) -> SymbolRuntimeState:
    return SymbolRuntimeState(
        symbol=str(data["symbol"]),
        campaign=_campaign_from_dict(data["campaign"]),
        intraday_score=int(data.get("intraday_score", 0)),
        intraday_detail={str(k): int(v) for k, v in (data.get("intraday_detail") or {}).items()},
        mode=str(data.get("mode", "NORMAL")),
        last_transition_reason=str(data.get("last_transition_reason", "")),
        last_30m_bar_time=(
            datetime.fromisoformat(data["last_30m_bar_time"])
            if data.get("last_30m_bar_time")
            else None
        ),
        entry_order=_order_from_dict(data.get("entry_order")),
        stop_order=_order_from_dict(data.get("stop_order")),
        exit_order=_order_from_dict(data.get("exit_order")),
        tp1_order=_order_from_dict(data.get("tp1_order")),
        tp2_order=_order_from_dict(data.get("tp2_order")),
        position=_position_from_dict(data.get("position")),
        pending_hard_exit=bool(data.get("pending_hard_exit", False)),
        pending_add=bool(data.get("pending_add", False)),
        last_signal_factors=dict(data.get("last_signal_factors", {})),
    )


def persist_research_snapshot(snapshot: ResearchSnapshot, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    path = research_snapshot_path(snapshot.trade_date, root=root, settings=settings)
    _write_json(path, _serialize(asdict(snapshot)))
    return path


def load_research_snapshot(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> ResearchSnapshot:
    payload = _read_json(research_snapshot_path(trade_date, root=root, settings=settings))
    return ResearchSnapshot(
        trade_date=date.fromisoformat(payload["trade_date"]),
        market=MarketResearch(
            price_ok=bool(payload["market"]["price_ok"]),
            breadth_pct_above_20dma=float(payload["market"]["breadth_pct_above_20dma"]),
            vix_percentile_1y=float(payload["market"]["vix_percentile_1y"]),
            hy_spread_5d_bps_change=float(payload["market"]["hy_spread_5d_bps_change"]),
            market_wide_institutional_selling=bool(payload["market"].get("market_wide_institutional_selling", False)),
        ),
        sectors={
            name: SectorResearch(
                name=name,
                flow_trend_20d=float(data["flow_trend_20d"]),
                breadth_20d=float(data["breadth_20d"]),
                participation=float(data["participation"]),
            )
            for name, data in payload["sectors"].items()
        },
        symbols={
            symbol: ResearchSymbol(
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
                biotech_flag=bool(data.get("biotech_flag", False)),
                flow_proxy_history=[float(value) for value in data.get("flow_proxy_history", [])],
                daily_bars=[_daily_bar_from_dict(item) for item in data.get("daily_bars", [])],
                bars_30m=[_bar_from_dict(item) for item in data.get("bars_30m", [])],
                sector_return_20d=float(data.get("sector_return_20d", 0.0)),
                sector_return_60d=float(data.get("sector_return_60d", 0.0)),
                intraday_atr_seed=float(data.get("intraday_atr_seed", 0.0)),
                average_30m_volume=float(data.get("average_30m_volume", 0.0)),
                median_30m_volume=float(data.get("median_30m_volume", 0.0)),
                expected_5m_volume=float(data.get("expected_5m_volume", 0.0)),
            )
            for symbol, data in payload["symbols"].items()
        },
        held_positions=[
            HeldPositionResearch(
                symbol=str(item["symbol"]),
                direction=str(item.get("direction", "LONG")),
                entry_time=datetime.fromisoformat(item["entry_time"]),
                entry_price=float(item["entry_price"]),
                size=int(item["size"]),
                stop=float(item["stop"]),
                initial_r=float(item["initial_r"]),
                setup_tag=str(item.get("setup_tag", "")),
                carry_eligible_flag=bool(item.get("carry_eligible_flag", False)),
            )
            for item in payload.get("held_positions", [])
        ],
    )


def _candidate_from_dict(data: dict[str, Any]) -> CandidateItem:
    campaign = _campaign_from_dict(data["campaign"])
    if not campaign.symbol:
        campaign.symbol = str(data["symbol"])
    return CandidateItem(
        symbol=str(data["symbol"]),
        exchange=str(data["exchange"]),
        primary_exchange=str(data["primary_exchange"]),
        currency=str(data["currency"]),
        tick_size=float(data["tick_size"]),
        point_value=float(data.get("point_value", 1.0)),
        sector=str(data.get("sector", "")),
        adv20_usd=float(data.get("adv20_usd", 0.0)),
        median_spread_pct=float(data.get("median_spread_pct", 0.0)),
        selection_score=int(data["selection_score"]),
        selection_detail={str(k): int(v) for k, v in data.get("selection_detail", {}).items()},
        stock_regime=str(data.get("stock_regime", Regime.TRANSITIONAL.value)),
        market_regime=str(data.get("market_regime", Regime.TRANSITIONAL.value)),
        sector_regime=str(data.get("sector_regime", Regime.TRANSITIONAL.value)),
        daily_trend_sign=int(data.get("daily_trend_sign", 0)),
        relative_strength_percentile=float(data.get("relative_strength_percentile", 0.0)),
        accumulation_score=float(data.get("accumulation_score", 0.0)),
        ttm_squeeze_bonus=int(data.get("ttm_squeeze_bonus", 0)),
        average_30m_volume=float(data.get("average_30m_volume", 0.0)),
        median_30m_volume=float(data.get("median_30m_volume", 0.0)),
        tradable_flag=bool(data.get("tradable_flag", False)),
        direction_bias=str(data.get("direction_bias", "BOTH")),
        price=float(data.get("price", 0.0)),
        earnings_risk_flag=bool(data.get("earnings_risk_flag", False)),
        campaign=campaign,
        daily_bars=[_daily_bar_from_dict(item) for item in data.get("daily_bars", [])],
        bars_30m=[_bar_from_dict(item) for item in data.get("bars_30m", [])],
    )


def persist_candidate_artifact(artifact: CandidateArtifact, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    path = artifact_path(artifact.trade_date, root=root, settings=settings)
    _write_json(path, _serialize(asdict(artifact)))
    return path


def load_candidate_artifact(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> CandidateArtifact:
    payload = _read_json(artifact_path(trade_date, root=root, settings=settings))
    regime_data = payload["regime"]
    return CandidateArtifact(
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
            market_regime=str(regime_data.get("market_regime", Regime.TRANSITIONAL.value)),
        ),
        items=[_candidate_from_dict(item) for item in payload.get("items", [])],
        tradable=[_candidate_from_dict(item) for item in payload.get("tradable", [])],
        overflow=[_candidate_from_dict(item) for item in payload.get("overflow", [])],
        long_candidates=[_candidate_from_dict(item) for item in payload.get("long_candidates", [])],
        short_candidates=[_candidate_from_dict(item) for item in payload.get("short_candidates", [])],
        market_wide_institutional_selling=bool(payload.get("market_wide_institutional_selling", False)),
    )


def persist_intraday_state(snapshot: IntradayStateSnapshot, root: Path | None = None, settings: StrategySettings | None = None) -> Path:
    path = state_path(snapshot.trade_date, root=root, settings=settings)
    _write_json(path, _serialize(asdict(snapshot)))
    return path


def load_intraday_state(trade_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> IntradayStateSnapshot:
    payload = _read_json(state_path(trade_date, root=root, settings=settings))
    return IntradayStateSnapshot(
        trade_date=date.fromisoformat(payload["trade_date"]),
        saved_at=datetime.fromisoformat(payload["saved_at"]),
        symbols=[_runtime_state_from_dict(item) for item in payload.get("symbols", [])],
        markets=[_market_snapshot_from_dict(item) for item in payload.get("markets", [])],
        last_decision_code=str(payload.get("last_decision_code", "")),
        meta=dict(payload.get("meta", {})),
    )


def load_latest_intraday_state(reference_date: date, root: Path | None = None, settings: StrategySettings | None = None) -> IntradayStateSnapshot | None:
    cfg = settings or StrategySettings()
    base = Path(root or cfg.state_dir)
    if not base.exists():
        return None
    candidates = sorted(path for path in base.glob("*.json") if path.stem <= reference_date.isoformat())
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        return load_intraday_state(date.fromisoformat(latest.stem), root=base, settings=cfg)
    except Exception:
        return None


# ── T2 engine state persistence ───────────────────────────────────────

def _t2_state_path(trade_date: date, settings: StrategySettings | None = None) -> Path:
    cfg = settings or StrategySettings()
    base = Path(cfg.state_dir) / "t2"
    return base / f"{trade_date.isoformat()}.json"


def persist_intraday_state_t2(snapshot: dict, settings: StrategySettings | None = None) -> Path:
    path = _t2_state_path(date.fromisoformat(snapshot["trade_date"]), settings=settings)
    _write_json(path, snapshot)
    return path


def load_intraday_state_t2(trade_date: date, settings: StrategySettings | None = None) -> dict | None:
    path = _t2_state_path(trade_date, settings=settings)
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None
