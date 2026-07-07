"""Nightly selection logic for ALCB."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import date, datetime, timezone
from statistics import fmean

from .artifact_store import load_research_snapshot, persist_candidate_artifact
from .config import ET, SECTOR_ETFS, StrategySettings
from .data import aggregate_bars
from .diagnostics import JsonlDiagnostics
from .models import Campaign, CampaignState, CandidateArtifact, CandidateItem, Regime, RegimeSnapshot, ResearchSnapshot, ResearchSymbol
from .signals import (
    accumulation_score,
    classify_4h_regime,
    daily_trend_sign,
    detect_compression_box,
    distribution_score,
    qualify_breakout,
    rs_percentiles,
)


def _sector_regime_name(snapshot: ResearchSnapshot, sector: str) -> str:
    data = snapshot.sectors.get(sector)
    if data is None:
        return Regime.TRANSITIONAL.value
    if data.flow_trend_20d > 0.5 and data.breadth_20d >= 0.55:
        return Regime.BULL.value
    if data.flow_trend_20d < -0.5 and data.breadth_20d <= 0.45:
        return Regime.BEAR.value
    if abs(data.flow_trend_20d) < 0.25:
        return Regime.CHOP.value
    return Regime.TRANSITIONAL.value


def compute_market_regime(snapshot: ResearchSnapshot, settings: StrategySettings) -> RegimeSnapshot:
    market = snapshot.market
    breadth_ok = market.breadth_pct_above_20dma >= 55.0
    vol_ok = market.vix_percentile_1y < 80.0
    credit_ok = market.hy_spread_5d_bps_change < 15.0
    score = (0.25 * float(market.price_ok)) + (0.30 * float(breadth_ok)) + (0.25 * float(vol_ok)) + (0.20 * float(credit_ok))
    if market.price_ok and breadth_ok and vol_ok:
        market_regime = Regime.BULL.value
    elif (not market.price_ok) and (not breadth_ok) and (not vol_ok):
        market_regime = Regime.BEAR.value
    elif not vol_ok:
        market_regime = Regime.CHOP.value
    else:
        market_regime = Regime.TRANSITIONAL.value
    if score < 0.35:
        tier = "C"
        mult = 0.0
    elif score >= 0.70:
        tier = "A"
        mult = 1.0
    else:
        tier = "B"
        mult = 0.75
    return RegimeSnapshot(
        score=score,
        tier=tier,
        risk_multiplier=mult,
        price_ok=market.price_ok,
        breadth_ok=breadth_ok,
        vol_ok=vol_ok,
        credit_ok=credit_ok,
        market_regime=market_regime,
    )


def filter_liquid_common_stocks(symbols: dict[str, ResearchSymbol], settings: StrategySettings) -> dict[str, ResearchSymbol]:
    accepted: dict[str, ResearchSymbol] = {}
    for symbol, item in symbols.items():
        if item.price < settings.min_price:
            continue
        if item.adv20_usd < settings.min_adv_usd:
            continue
        if item.median_spread_pct > settings.max_median_spread_pct:
            continue
        if item.etf_flag or item.preferred_flag or item.otc_flag:
            continue
        if item.blacklist_flag or item.halted_flag or item.severe_news_flag:
            continue
        if item.earnings_within_sessions is not None and item.earnings_within_sessions <= settings.earnings_block_days:
            continue
        if item.adr_flag and not settings.allow_adrs:
            continue
        if item.biotech_flag and not settings.allow_biotech:
            continue
        accepted[symbol] = item
    return accepted


@dataclass(slots=True)
class _PreparedSelectionFeatures:
    trend: int
    trend_avg20: float
    box: object | None
    accumulation: float
    distribution: float
    stock_regime: Regime
    sector_regime: str


def _selection_score_long(
    item: ResearchSymbol,
    prepared: _PreparedSelectionFeatures,
    rs_rank: float,
) -> tuple[int, dict[str, int]]:
    score = 0
    detail: dict[str, int] = {}
    if rs_rank >= 0.90:
        score += 3
        detail["rs"] = 3
    elif rs_rank >= 0.75:
        score += 2
        detail["rs"] = 2
    elif rs_rank >= 0.50:
        score += 1
        detail["rs"] = 1
    else:
        detail["rs"] = 0
    if prepared.trend > 0:
        score += 2
        detail["trend"] = 2
    elif prepared.trend == 0 and item.daily_bars and item.daily_bars[-1].close >= prepared.trend_avg20:
        score += 1
        detail["trend"] = 1
    else:
        detail["trend"] = 0
    if prepared.box is not None and prepared.box.tier.value == "GOOD":
        score += 2
        detail["compression"] = 2
    elif prepared.box is not None:
        score += 1
        detail["compression"] = 1
    else:
        detail["compression"] = 0
    if prepared.accumulation >= 0.8:
        score += 2
        detail["accumulation"] = 2
    elif prepared.accumulation >= 0.55:
        score += 1
        detail["accumulation"] = 1
    else:
        detail["accumulation"] = 0
    if prepared.sector_regime == Regime.BULL.value:
        score += 1
        detail["sector"] = 1
    else:
        detail["sector"] = 0
    return score, detail


def _selection_score_short(
    prepared: _PreparedSelectionFeatures,
    rs_rank: float,
) -> tuple[int, dict[str, int]]:
    score = 0
    detail: dict[str, int] = {}
    if rs_rank <= 0.10:
        score += 3
        detail["rs"] = 3
    elif rs_rank <= 0.25:
        score += 2
        detail["rs"] = 2
    elif rs_rank <= 0.50:
        score += 1
        detail["rs"] = 1
    else:
        detail["rs"] = 0
    if prepared.trend < 0:
        score += 2
        detail["trend"] = 2
    elif prepared.trend == 0:
        score += 1
        detail["trend"] = 1
    else:
        detail["trend"] = 0
    if prepared.box is not None and prepared.box.tier.value == "GOOD":
        score += 2
        detail["compression"] = 2
    elif prepared.box is not None:
        score += 1
        detail["compression"] = 1
    else:
        detail["compression"] = 0
    if prepared.distribution >= 0.8:
        score += 2
        detail["distribution"] = 2
    elif prepared.distribution >= 0.55:
        score += 1
        detail["distribution"] = 1
    else:
        detail["distribution"] = 0
    if prepared.sector_regime == Regime.BEAR.value:
        score += 1
        detail["sector"] = 1
    else:
        detail["sector"] = 0
    return score, detail


def _direction_bias(long_score: int, short_score: int) -> str:
    if long_score >= short_score + 2:
        return "LONG"
    if short_score >= long_score + 2:
        return "SHORT"
    return "BOTH"


def build_candidate_item(
    symbol: str,
    item: ResearchSymbol,
    regime: RegimeSnapshot,
    prepared: _PreparedSelectionFeatures,
    rs_rank: float,
    settings: StrategySettings,
) -> tuple[CandidateItem, int, int]:
    long_score, long_detail = _selection_score_long(item, prepared, rs_rank)
    short_score, short_detail = _selection_score_short(prepared, rs_rank)
    score = max(long_score, short_score)
    detail = long_detail if long_score >= short_score else short_detail
    box = prepared.box
    campaign = Campaign(symbol=symbol)
    if box is not None:
        campaign.state = CampaignState.COMPRESSION
        campaign.box = box
        campaign.box_version = 1
        campaign.avwap_anchor_ts = f"{box.start_date}T09:30:00-05:00"
        breakout = qualify_breakout(item.daily_bars, item.bars_30m, campaign, settings)
        if breakout is not None:
            campaign.breakout = breakout
            campaign.state = CampaignState.BREAKOUT
    return (
        CandidateItem(
            symbol=symbol,
            exchange=item.exchange,
            primary_exchange=item.primary_exchange,
            currency=item.currency,
            tick_size=item.tick_size,
            point_value=item.point_value,
            sector=item.sector,
            adv20_usd=item.adv20_usd,
            median_spread_pct=item.median_spread_pct,
            selection_score=score,
            selection_detail=detail,
            stock_regime=prepared.stock_regime.value,
            market_regime=regime.market_regime,
            sector_regime=prepared.sector_regime,
            daily_trend_sign=prepared.trend,
            relative_strength_percentile=rs_rank,
            accumulation_score=prepared.accumulation,
            ttm_squeeze_bonus=0,
            average_30m_volume=item.average_30m_volume,
            median_30m_volume=item.median_30m_volume or item.average_30m_volume,
            tradable_flag=False,
            direction_bias=_direction_bias(long_score, short_score),
            price=item.price,
            earnings_risk_flag=bool(item.earnings_within_sessions is not None and item.earnings_within_sessions <= settings.earnings_block_days),
            campaign=campaign,
            daily_bars=item.daily_bars,
            bars_30m=item.bars_30m,
        ),
        long_score,
        short_score,
        )


def _prepare_selection_features(
    universe: dict[str, ResearchSymbol],
    snapshot: ResearchSnapshot,
    settings: StrategySettings,
) -> dict[str, _PreparedSelectionFeatures]:
    sectors = {item.sector for item in universe.values()}
    sector_regimes = {
        sector: _sector_regime_name(snapshot, sector)
        for sector in sectors
    }
    prepared: dict[str, _PreparedSelectionFeatures] = {}
    for symbol, item in universe.items():
        trend = daily_trend_sign(item.daily_bars)
        box = detect_compression_box(item.daily_bars, settings)
        prepared[symbol] = _PreparedSelectionFeatures(
            trend=trend,
            trend_avg20=fmean(bar.close for bar in item.daily_bars[-20:]) if item.daily_bars else 0.0,
            box=box,
            accumulation=accumulation_score(item.daily_bars, settings=settings),
            distribution=distribution_score(item.daily_bars, settings=settings),
            stock_regime=classify_4h_regime(aggregate_bars(item.bars_30m, 8)) if item.bars_30m else Regime.TRANSITIONAL,
            sector_regime=sector_regimes.get(item.sector, Regime.TRANSITIONAL.value),
        )
    return prepared


def daily_selection_from_snapshot(
    snapshot: ResearchSnapshot,
    settings: StrategySettings | None = None,
    diagnostics: JsonlDiagnostics | None = None,
) -> CandidateArtifact:
    cfg = settings or StrategySettings()
    diag = diagnostics or JsonlDiagnostics(cfg.diagnostics_dir, enabled=False)
    regime = compute_market_regime(snapshot, cfg)
    if regime.tier == "C":
        return CandidateArtifact(
            trade_date=snapshot.trade_date,
            generated_at=datetime.now(timezone.utc),
            regime=regime,
            items=[],
            tradable=[],
            overflow=[],
            long_candidates=[],
            short_candidates=[],
            market_wide_institutional_selling=snapshot.market.market_wide_institutional_selling,
        )
    universe = filter_liquid_common_stocks(snapshot.symbols, cfg)
    universe_daily = {name: item.daily_bars for name, item in universe.items()}
    rs_ranks = rs_percentiles(universe_daily)
    prepared = _prepare_selection_features(universe, snapshot, cfg)
    long_ranked: list[tuple[int, CandidateItem]] = []
    short_ranked: list[tuple[int, CandidateItem]] = []
    all_items: list[CandidateItem] = []
    for symbol, item in universe.items():
        rs_rank = rs_ranks.get(symbol, 0.0)
        candidate, long_score, short_score = build_candidate_item(
            symbol, item, regime, prepared[symbol], rs_rank, cfg,
        )
        if candidate.campaign.box is not None:
            all_items.append(candidate)
        if long_score > 0:
            long_ranked.append((long_score, candidate))
        if short_score > 0:
            short_ranked.append((short_score, candidate))
    long_ranked.sort(key=lambda row: (row[0], row[1].selection_score, row[1].relative_strength_percentile), reverse=True)
    short_ranked.sort(key=lambda row: (row[0], row[1].selection_score, -row[1].relative_strength_percentile), reverse=True)
    long_candidates = [item for _, item in long_ranked[: cfg.selection_long_count]]
    short_candidates = [item for _, item in short_ranked[: cfg.selection_short_count]]
    tradable_map = {item.symbol: replace(item, tradable_flag=True) for item in (long_candidates + short_candidates)}
    items = [tradable_map.get(item.symbol, item) for item in sorted(all_items, key=lambda row: row.selection_score, reverse=True)]
    tradable = [tradable_map[symbol] for symbol in sorted(tradable_map, key=lambda name: tradable_map[name].selection_score, reverse=True)]
    overflow = [item for item in items if item.symbol not in tradable_map]
    diag.log_regime({"trade_date": snapshot.trade_date.isoformat(), "tier": regime.tier, "score": regime.score, "market_regime": regime.market_regime})
    return CandidateArtifact(
        trade_date=snapshot.trade_date,
        generated_at=datetime.now(timezone.utc),
        regime=regime,
        items=items,
        tradable=tradable,
        overflow=overflow,
        long_candidates=[tradable_map.get(item.symbol, item) for item in long_candidates],
        short_candidates=[tradable_map.get(item.symbol, item) for item in short_candidates],
        market_wide_institutional_selling=snapshot.market.market_wide_institutional_selling,
    )


def run_daily_selection(
    trade_date: date,
    settings: StrategySettings | None = None,
    diagnostics: JsonlDiagnostics | None = None,
) -> CandidateArtifact:
    cfg = settings or StrategySettings()
    diag = diagnostics or JsonlDiagnostics(cfg.diagnostics_dir)
    snapshot = load_research_snapshot(trade_date, settings=cfg)
    artifact = daily_selection_from_snapshot(snapshot=snapshot, settings=cfg, diagnostics=diag)
    persist_candidate_artifact(artifact, settings=cfg)
    return artifact
