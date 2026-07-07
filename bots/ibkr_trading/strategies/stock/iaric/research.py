"""Daily selection logic for IARIC v1 (pullback V2 mode).

When ``pb_v2_enabled`` is True, the selection uses 7-trigger pullback
scoring instead of T1 sponsorship/conviction.  Legacy path preserved
for ``pb_v2_enabled=False``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from statistics import fmean, median

from .artifact_store import load_research_snapshot, persist_watchlist_artifact
from .config import StrategySettings
from .diagnostics import JsonlDiagnostics
from .models import (
    HeldPositionDirective,
    RegimeSnapshot,
    ResearchSnapshot,
    ResearchSymbol,
    WatchlistArtifact,
    WatchlistItem,
)
from .signals import (
    build_daily_watchlist,
    compute_indicator_cache,
    compute_trend_tier,
    compute_trigger_tier,
)


def _entry_gap_pct(symbol: ResearchSymbol) -> float:
    if len(symbol.daily_bars) < 2:
        return 0.0
    prev_close = symbol.daily_bars[-2].close
    if prev_close <= 0:
        return 0.0
    return (symbol.daily_bars[-1].open - prev_close) / prev_close * 100.0


def _zscore_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / max(len(values), 1)
    std = variance ** 0.5
    if std <= 1e-9:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / std for key, value in values.items()}


def _percentile_rank(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    count = sum(1 for sample in values if sample <= value)
    return 100.0 * count / len(values)


def compute_market_regime(snapshot: ResearchSnapshot, settings: StrategySettings) -> RegimeSnapshot:
    market = snapshot.market
    breadth_ok = market.breadth_pct_above_20dma > settings.breadth_threshold_pct
    vol_ok = market.vix_percentile_1y < settings.vix_percentile_threshold
    credit_ok = market.hy_spread_5d_bps_change < settings.hy_spread_5d_bps_threshold
    score = (
        0.25 * float(market.price_ok)
        + 0.30 * float(breadth_ok)
        + 0.25 * float(vol_ok)
        + 0.20 * float(credit_ok)
    )
    if score < settings.tier_b_min:
        tier = "C"
        mult = 0.0
    else:
        t = min(1.0, max(0.0, (score - settings.tier_b_min) / (settings.regime_full_mult_score - settings.tier_b_min)))
        mult = 0.35 + 0.65 * (t ** 1.3)
        tier = "A" if score > settings.tier_a_min else "B"
    return RegimeSnapshot(
        score=score,
        tier=tier,
        risk_multiplier=mult,
        price_ok=market.price_ok,
        breadth_ok=breadth_ok,
        vol_ok=vol_ok,
        credit_ok=credit_ok,
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
        if item.hard_to_borrow_flag:
            continue
        if item.blacklist_flag or item.halted_flag or item.severe_news_flag:
            continue
        if item.earnings_within_sessions is not None and item.earnings_within_sessions <= 3:
            continue
        if item.adr_flag and item.median_spread_pct > settings.max_median_spread_pct * 0.75:
            continue
        accepted[symbol] = item
    return accepted


def compute_sector_scores(universe: dict[str, ResearchSymbol], snapshot: ResearchSnapshot) -> tuple[dict[str, float], dict[str, float]]:
    sectors = {name: data for name, data in snapshot.sectors.items() if name in {item.sector for item in universe.values()}}
    flow = _zscore_map({name: data.flow_trend_20d for name, data in sectors.items()})
    breadth = _zscore_map({name: data.breadth_20d for name, data in sectors.items()})
    participation = _zscore_map({name: data.participation for name, data in sectors.items()})
    scores = {
        name: (0.45 * flow.get(name, 0.0)) + (0.30 * breadth.get(name, 0.0)) + (0.25 * participation.get(name, 0.0))
        for name in sectors
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    weights: dict[str, float] = {}
    for index, (name, _) in enumerate(ranked, start=1):
        if index == 1:
            weights[name] = 1.0
        elif index == 2:
            weights[name] = 0.8
        elif index == 3:
            weights[name] = 0.6
        else:
            weights[name] = 0.3
    return scores, weights


def _persistence(symbol: ResearchSymbol) -> float:
    window = symbol.flow_proxy_history[-10:]
    if not window:
        return 0.0
    return sum(1 for value in window if value > 0) / len(window)


def _intensity(symbol: ResearchSymbol) -> float:
    flow_window = symbol.flow_proxy_history[-5:]
    if not flow_window or symbol.adv20_usd <= 0:
        return 0.0
    return sum(flow_window) / symbol.adv20_usd


def _accel(symbol: ResearchSymbol) -> float:
    history = symbol.flow_proxy_history
    if len(history) < 20:
        return 0.0
    sma5 = sum(history[-5:]) / 5.0
    sma20 = sum(history[-20:]) / 20.0
    return sma5 - sma20


def _leader_percentile(symbol: ResearchSymbol, peers: list[ResearchSymbol]) -> float:
    relative = (symbol.stock_return_20d - symbol.sector_return_20d) + (symbol.stock_return_60d - symbol.sector_return_60d)
    peer_values = [
        (peer.stock_return_20d - peer.sector_return_20d) + (peer.stock_return_60d - peer.sector_return_60d)
        for peer in peers
    ]
    return _percentile_rank(relative, peer_values)


def _trend_strength(symbol: ResearchSymbol) -> float:
    price = symbol.trend_price
    sma20 = symbol.sma20
    sma50 = symbol.sma50
    if sma20 <= 0 or sma50 <= 0:
        return 0.0
    return max(0.0, ((price - sma20) / sma20) + ((sma20 - sma50) / sma50))


def _find_sponsorship_streak_start(symbol: ResearchSymbol) -> int | None:
    history = symbol.flow_proxy_history[-40:]
    if len(history) < 5:
        return None
    for start in range(len(history) - 5, -1, -1):
        streak = history[start : start + 5]
        if len(streak) == 5 and all(value > 0 for value in streak):
            return max(0, len(symbol.daily_bars) - len(history) + start)
    return None


def _impulse_day_index(symbol: ResearchSymbol) -> int | None:
    bars = symbol.daily_bars[-40:]
    if len(bars) < 20:
        return None
    volumes = [bar.volume for bar in bars]
    for offset in range(len(bars) - 1, -1, -1):
        bar = bars[offset]
        avg20 = fmean(volumes[max(0, offset - 19) : offset + 1])
        if avg20 > 0 and bar.volume > 2.0 * avg20 and bar.cpr >= 0.7:
            return len(symbol.daily_bars) - len(bars) + offset
    return None


def select_anchor(symbol: ResearchSymbol, settings: StrategySettings) -> tuple[int, str]:
    if not symbol.daily_bars:
        raise ValueError(f"{symbol.symbol} is missing daily bars")
    streak = _find_sponsorship_streak_start(symbol)
    if streak is not None:
        return streak, "SPONSORSHIP_STREAK"
    impulse = _impulse_day_index(symbol)
    if impulse is not None:
        return impulse, "IMPULSE_DAY"
    for index in range(len(symbol.daily_bars) - 1, max(-1, len(symbol.daily_bars) - settings.anchor_lookback_sessions - 1), -1):
        bar = symbol.daily_bars[index]
        if bar.event_tag in {"BREAKOUT", "EARNINGS_CONTINUATION"}:
            return index, bar.event_tag
    return max(0, len(symbol.daily_bars) - min(len(symbol.daily_bars), settings.anchor_lookback_sessions)), "LOOKBACK_FALLBACK"


def compute_daily_avwap_approx(symbol: ResearchSymbol, anchor_index: int) -> float:
    cum_pv = 0.0
    cum_vol = 0.0
    for bar in symbol.daily_bars[anchor_index:]:
        cum_pv += bar.typical_price * bar.volume
        cum_vol += bar.volume
    return cum_pv / max(cum_vol, 1e-9)


def anchor_acceptance_pass(symbol: ResearchSymbol, anchor_index: int, avwap_ref: float, settings: StrategySettings) -> bool:
    tolerance = settings.avwap_acceptance_band_pct
    lower = avwap_ref * (1.0 - tolerance)
    upper = avwap_ref * (1.0 + tolerance)
    for bar in symbol.daily_bars[anchor_index:]:
        if bar.low <= upper and bar.high >= lower:
            return True
    return False


def _continuous_conviction(value: float) -> tuple[str, float]:
    if value <= 10.0:
        mult = 0.0
    elif value <= 35.0:
        mult = (value - 10.0) * (0.7 / 25.0)
    elif value <= 65.0:
        mult = 0.7 + (value - 35.0) * (0.3 / 30.0)
    elif value <= 90.0:
        mult = 1.0 + (value - 65.0) * (0.5 / 25.0)
    else:
        mult = 1.5

    if mult >= 1.25:
        bucket = "TOP"
    elif mult >= 0.85:
        bucket = "CORE"
    elif mult > 0.0:
        bucket = "SMALL"
    else:
        bucket = "SKIP"
    return bucket, round(mult, 4)


def build_watchlist_item(
    symbol: ResearchSymbol,
    regime: RegimeSnapshot,
    settings: StrategySettings,
    sector_scores: dict[str, float],
    sector_weights: dict[str, float],
    sponsorship_score: float,
    sponsorship_state: str,
    persistence: float,
    intensity_z: float,
    accel_z: float,
    rs_percentile: float,
    leader_pass: bool,
    trend_pass: bool,
    trend_strength: float,
    anchor_index: int,
    anchor_type: str,
    avwap_ref: float,
    acceptance_pass: bool,
    conviction_multiplier: float,
    conviction_bucket: str,
    flow_proxy_gate_pass: bool = True,
) -> WatchlistItem:
    band_pct = settings.avwap_band_pct
    avwap_proximity = 1.0 / (1.0 + abs(symbol.price - avwap_ref))
    daily_rank = (
        0.40 * sponsorship_score
        + 0.20 * (rs_percentile / 100.0)
        + 0.20 * sector_weights.get(symbol.sector, 0.3)
        + 0.20 * avwap_proximity
    )
    return WatchlistItem(
        symbol=symbol.symbol,
        exchange=symbol.exchange,
        primary_exchange=symbol.primary_exchange,
        currency=symbol.currency,
        tick_size=symbol.tick_size,
        point_value=symbol.point_value,
        sector=symbol.sector,
        regime_score=regime.score,
        regime_tier=regime.tier,
        regime_risk_multiplier=regime.risk_multiplier,
        sector_score=sector_scores.get(symbol.sector, 0.0),
        sector_rank_weight=sector_weights.get(symbol.sector, 0.3),
        sponsorship_score=sponsorship_score,
        sponsorship_state=sponsorship_state,
        persistence=persistence,
        intensity_z=intensity_z,
        accel_z=accel_z,
        rs_percentile=rs_percentile,
        leader_pass=leader_pass,
        trend_pass=trend_pass,
        trend_strength=trend_strength,
        earnings_risk_flag=(symbol.earnings_within_sessions or 99) <= 3,
        blacklist_flag=symbol.blacklist_flag,
        anchor_date=symbol.daily_bars[anchor_index].trade_date,
        anchor_type=anchor_type,
        acceptance_pass=acceptance_pass,
        avwap_ref=avwap_ref,
        avwap_band_lower=avwap_ref * (1.0 - band_pct),
        avwap_band_upper=avwap_ref * (1.0 + band_pct),
        daily_atr_estimate=symbol.daily_atr_estimate,
        intraday_atr_seed=max(symbol.intraday_atr_seed, symbol.daily_atr_estimate / max(symbol.price, 1e-9)),
        daily_rank=daily_rank,
        tradable_flag=False,
        conviction_bucket=conviction_bucket,
        conviction_multiplier=conviction_multiplier,
        recommended_risk_r=conviction_multiplier,
        average_30m_volume=max(symbol.average_30m_volume, 0.0),
        expected_5m_volume=max(symbol.expected_5m_volume, 0.0),
        entry_gap_pct=_entry_gap_pct(symbol),
        flow_proxy_gate_pass=flow_proxy_gate_pass,
    )


def _flow_reversal_flag(symbol: ResearchSymbol, lookback: int = 1) -> bool:
    window = symbol.flow_proxy_history[-lookback:]
    return len(window) == lookback and all(value < 0 for value in window)


def build_held_position_directives(snapshot: ResearchSnapshot, settings: StrategySettings) -> list[HeldPositionDirective]:
    directives: list[HeldPositionDirective] = []
    for held in snapshot.held_positions:
        symbol = snapshot.symbols.get(held.symbol)
        directives.append(
            HeldPositionDirective(
                symbol=held.symbol,
                entry_time=held.entry_time,
                entry_price=held.entry_price,
                size=held.size,
                stop=held.stop,
                initial_r=held.initial_r,
                setup_tag=held.setup_tag,
                time_stop_deadline=held.entry_time + timedelta(minutes=settings.time_stop_minutes),
                carry_eligible_flag=held.carry_eligible_flag,
                flow_reversal_flag=_flow_reversal_flag(symbol, lookback=settings.flow_reversal_lookback) if symbol else False,
            )
        )
    return directives


def _pullback_daily_selection(
    snapshot: ResearchSnapshot,
    cfg: StrategySettings,
    diag: JsonlDiagnostics,
    regime: RegimeSnapshot,
    universe: dict[str, ResearchSymbol],
    sector_scores: dict[str, float],
    sector_weights: dict[str, float],
) -> list[WatchlistItem]:
    """Pullback V2 daily selection using 7-trigger scoring."""
    # Build SPY benchmark closes for relative strength
    spy = snapshot.symbols.get("SPY")
    benchmark_closes: np.ndarray | None = None
    if spy and spy.daily_bars:
        benchmark_closes = np.array([b.close for b in spy.daily_bars], dtype=np.float64)

    # Compute indicators for all universe symbols
    indicators_cache: dict[str, dict[str, np.ndarray]] = {}
    for sym_name, sym in universe.items():
        ind = compute_indicator_cache(sym, benchmark_closes)
        if ind:
            indicators_cache[sym_name] = ind

    # Score and rank candidates
    candidates = build_daily_watchlist(universe, indicators_cache, cfg)

    # Build watchlist items for candidates
    peers_by_sector: dict[str, list[ResearchSymbol]] = {}
    for sym in universe.values():
        peers_by_sector.setdefault(sym.sector, []).append(sym)

    items: list[WatchlistItem] = []
    for sym_name, score, triggers in candidates:
        symbol = universe[sym_name]
        ind = indicators_cache.get(sym_name, {})

        # Trend tier
        trend_tier_val = compute_trend_tier(symbol, ind, cfg)
        if trend_tier_val == "EXCLUDED":
            diag.log_filter(sym_name, "trend_tier", False, "excluded")
            continue

        # CDD hard filter (research parity)
        cdd_value = 0
        if "cdd" in ind and len(ind["cdd"]) > 0:
            v = ind["cdd"][-1]
            if not np.isnan(v):
                cdd_value = int(v)
        if cdd_value > cfg.pb_cdd_max:
            diag.log_filter(sym_name, "cdd_max", False, f"cdd_{cdd_value}_exceeds_{cfg.pb_cdd_max}")
            continue

        # Trigger tier and sizing
        trigger_tier_val, sizing_mult = compute_trigger_tier(score, cfg)
        if sizing_mult <= 0:
            continue

        # Secular discount
        if trend_tier_val == "SECULAR":
            sizing_mult *= cfg.pb_v2_secular_sizing_mult

        # Flow policy: rescue candidates
        rescue_candidate = False
        if cfg.pb_flow_policy == "soft_penalty_rescue":
            if symbol.flow_proxy_history:
                last_flow = symbol.flow_proxy_history[-1:]
                if last_flow and last_flow[0] < 0:
                    if score >= cfg.pb_daily_rescue_min_score:
                        rescue_candidate = True
                        sizing_mult *= cfg.pb_rescue_size_mult
                    else:
                        diag.log_filter(sym_name, "flow", False, "negative_flow_below_rescue")
                        continue

        # Extract daily EMA10 and RSI14 for intraday exit chain
        ema10_val = 0.0
        rsi14_val = 0.0
        if "ema10" in ind and len(ind["ema10"]) > 0:
            v = float(ind["ema10"][-1])
            if not np.isnan(v):
                ema10_val = v
        if "rsi14" in ind and len(ind["rsi14"]) > 0:
            v = float(ind["rsi14"][-1])
            if not np.isnan(v):
                rsi14_val = v

        # Compute T1-compat fields for backward compatibility
        rs_percentile = _leader_percentile(symbol, peers_by_sector.get(symbol.sector, [symbol]))
        trend_strength = _trend_strength(symbol)
        persistence = _persistence(symbol)

        # Anchor (fallback for backward compat)
        anchor_index, anchor_type = select_anchor(symbol, cfg)
        avwap_ref = compute_daily_avwap_approx(symbol, anchor_index)
        band_pct = cfg.avwap_band_pct

        item = WatchlistItem(
            symbol=symbol.symbol,
            exchange=symbol.exchange,
            primary_exchange=symbol.primary_exchange,
            currency=symbol.currency,
            tick_size=symbol.tick_size,
            point_value=symbol.point_value,
            sector=symbol.sector,
            regime_score=regime.score,
            regime_tier=regime.tier,
            regime_risk_multiplier=regime.risk_multiplier,
            sector_score=sector_scores.get(symbol.sector, 0.0),
            sector_rank_weight=sector_weights.get(symbol.sector, 0.3),
            sponsorship_score=0.0,
            sponsorship_state="NEUTRAL",
            persistence=persistence,
            intensity_z=0.0,
            accel_z=0.0,
            rs_percentile=rs_percentile,
            leader_pass=True,
            trend_pass=trend_tier_val == "STRONG",
            trend_strength=trend_strength,
            earnings_risk_flag=(symbol.earnings_within_sessions or 99) <= 3,
            blacklist_flag=symbol.blacklist_flag,
            anchor_date=symbol.daily_bars[anchor_index].trade_date if symbol.daily_bars else snapshot.trade_date,
            anchor_type=anchor_type,
            acceptance_pass=True,
            avwap_ref=avwap_ref,
            avwap_band_lower=avwap_ref * (1.0 - band_pct),
            avwap_band_upper=avwap_ref * (1.0 + band_pct),
            daily_atr_estimate=symbol.daily_atr_estimate,
            intraday_atr_seed=max(symbol.intraday_atr_seed, symbol.daily_atr_estimate / max(symbol.price, 1e-9)),
            daily_rank=score / 100.0,
            tradable_flag=False,
            conviction_bucket=trigger_tier_val,
            conviction_multiplier=sizing_mult,
            recommended_risk_r=sizing_mult,
            average_30m_volume=max(symbol.average_30m_volume, 0.0),
            expected_5m_volume=max(symbol.expected_5m_volume, 0.0),
            entry_gap_pct=_entry_gap_pct(symbol),
            flow_proxy_gate_pass=not rescue_candidate,
            # Pullback V2 fields
            daily_signal_score=score,
            trigger_types=triggers,
            trigger_tier=trigger_tier_val,
            trend_tier=trend_tier_val,
            rescue_flow_candidate=rescue_candidate,
            sizing_mult=sizing_mult,
            cdd_value=cdd_value,
            ema10_daily=ema10_val,
            rsi14_daily=rsi14_val,
        )
        items.append(item)

    return items


def _legacy_daily_selection(
    snapshot: ResearchSnapshot,
    cfg: StrategySettings,
    diag: JsonlDiagnostics,
    regime: RegimeSnapshot,
    universe: dict[str, ResearchSymbol],
    sector_scores: dict[str, float],
    sector_weights: dict[str, float],
) -> list[WatchlistItem]:
    """Legacy T1 sponsorship/conviction selection (pb_v2_enabled=False)."""
    peers_by_sector: dict[str, list[ResearchSymbol]] = {}
    for symbol in universe.values():
        peers_by_sector.setdefault(symbol.sector, []).append(symbol)

    intensity_z = _zscore_map({symbol: _intensity(item) for symbol, item in universe.items()})
    accel_z = _zscore_map({symbol: _accel(item) for symbol, item in universe.items()})
    persistence_by_symbol = {symbol: _persistence(item) for symbol, item in universe.items()}
    score_by_symbol = {
        symbol: (0.40 * persistence_by_symbol[symbol]) + (0.35 * intensity_z.get(symbol, 0.0)) + (0.25 * accel_z.get(symbol, 0.0))
        for symbol in universe
    }

    sector_sponsorship: dict[str, list[float]] = {}
    for symbol, item in universe.items():
        sector_sponsorship.setdefault(item.sector, []).append(score_by_symbol[symbol])

    items: list[WatchlistItem] = []
    for symbol_name, symbol in universe.items():
        persistence = persistence_by_symbol[symbol_name]
        sponsorship_score = score_by_symbol[symbol_name]
        sector_scores_for_symbol = sorted(sector_sponsorship.get(symbol.sector, [sponsorship_score]))
        sector_median = median(sector_scores_for_symbol)
        quartile_index = max(0, int(len(sector_scores_for_symbol) * 0.75) - 1)
        top_quartile_threshold = sector_scores_for_symbol[quartile_index]
        if persistence <= 0.0:
            diag.log_filter(symbol_name, "sponsorship", False, "stale")
            continue
        if persistence < 0.6 or sponsorship_score <= sector_median:
            diag.log_filter(symbol_name, "sponsorship", False, "distribution")
            continue
        sponsorship_state = "STRONG" if sponsorship_score >= top_quartile_threshold else "NEUTRAL"

        rs_percentile = _leader_percentile(symbol, peers_by_sector.get(symbol.sector, [symbol]))
        required_rs = 65.0 if regime.tier == "A" else 75.0
        leader_pass = rs_percentile >= required_rs
        if not leader_pass:
            diag.log_filter(symbol_name, "leader", False, f"rs_lt_{required_rs}")
            continue

        trend_pass = symbol.trend_price > symbol.sma50 and symbol.sma50_slope >= 0
        trend_strength = _trend_strength(symbol)
        if not trend_pass:
            diag.log_filter(symbol_name, "trend", False, "trend_fail")
            continue

        anchor_index, anchor_type = select_anchor(symbol, cfg)
        avwap_ref = compute_daily_avwap_approx(symbol, anchor_index)
        acceptance_pass = anchor_acceptance_pass(symbol, anchor_index, avwap_ref, cfg)
        if not acceptance_pass:
            diag.log_filter(symbol_name, "anchor", False, "anchor_acceptance_fail")
            continue

        conviction_input = (
            0.35 * _percentile_rank(sponsorship_score, list(score_by_symbol.values()))
            + 0.25 * rs_percentile
            + 0.20 * (sector_weights.get(symbol.sector, 0.3) * 100.0)
            + 0.20 * min(100.0, trend_strength * 1000.0)
        )
        conviction_bucket, conviction_multiplier = _continuous_conviction(conviction_input)
        if conviction_multiplier <= 0:
            diag.log_filter(symbol_name, "conviction", False, "skip")
            continue

        flow_gate_pass = True
        if cfg.t1_entry_flow_gate and symbol.flow_proxy_history:
            last_n = symbol.flow_proxy_history[-cfg.t1_entry_flow_lookback:]
            if last_n and any(v < 0 for v in last_n):
                flow_gate_pass = False

        items.append(
            build_watchlist_item(
                symbol=symbol,
                regime=regime,
                settings=cfg,
                sector_scores=sector_scores,
                sector_weights=sector_weights,
                sponsorship_score=sponsorship_score,
                sponsorship_state=sponsorship_state,
                persistence=persistence,
                intensity_z=intensity_z.get(symbol_name, 0.0),
                accel_z=accel_z.get(symbol_name, 0.0),
                rs_percentile=rs_percentile,
                leader_pass=leader_pass,
                trend_pass=trend_pass,
                trend_strength=trend_strength,
                anchor_index=anchor_index,
                anchor_type=anchor_type,
                avwap_ref=avwap_ref,
                acceptance_pass=acceptance_pass,
                conviction_multiplier=conviction_multiplier,
                conviction_bucket=conviction_bucket,
                flow_proxy_gate_pass=flow_gate_pass,
            )
        )
    return items


def daily_selection_from_snapshot(
    snapshot: ResearchSnapshot,
    settings: StrategySettings | None = None,
    diagnostics: JsonlDiagnostics | None = None,
) -> WatchlistArtifact:
    cfg = settings or StrategySettings()
    diag = diagnostics or JsonlDiagnostics(cfg.diagnostics_dir, enabled=False)
    regime = compute_market_regime(snapshot, cfg)
    diag.log_regime({"score": regime.score, "tier": regime.tier, "risk_multiplier": regime.risk_multiplier})
    if regime.tier == "C":
        return WatchlistArtifact(
            trade_date=snapshot.trade_date,
            generated_at=datetime.now(timezone.utc),
            regime=regime,
            items=[],
            tradable=[],
            overflow=[],
            market_wide_institutional_selling=snapshot.market.market_wide_institutional_selling,
            held_positions=build_held_position_directives(snapshot, cfg),
        )

    universe = filter_liquid_common_stocks(snapshot.symbols, cfg)
    sector_scores, sector_weights = compute_sector_scores(universe, snapshot)

    # Branch: V2 pullback vs legacy T1
    if cfg.pb_v2_enabled:
        items = _pullback_daily_selection(snapshot, cfg, diag, regime, universe, sector_scores, sector_weights)
    else:
        items = _legacy_daily_selection(snapshot, cfg, diag, regime, universe, sector_scores, sector_weights)

    ranked = sorted(items, key=lambda item: item.daily_rank, reverse=True)
    tradable_ratio = 0.30 if regime.tier == "A" else 0.20
    tradable_count = min(len(ranked), max(1 if ranked else 0, int(len(ranked) * tradable_ratio)))
    tradable: list[WatchlistItem] = []
    overflow: list[WatchlistItem] = []
    for index, item in enumerate(ranked):
        flagged = replace(item, tradable_flag=index < tradable_count, overflow_rank=(index + 1 if index >= tradable_count else None))
        ranked[index] = flagged
        if index < tradable_count:
            tradable.append(flagged)
        else:
            overflow.append(flagged)

    return WatchlistArtifact(
        trade_date=snapshot.trade_date,
        generated_at=datetime.now(timezone.utc),
        regime=regime,
        items=ranked,
        tradable=tradable,
        overflow=overflow,
        market_wide_institutional_selling=snapshot.market.market_wide_institutional_selling,
        held_positions=build_held_position_directives(snapshot, cfg),
    )


def run_daily_selection(
    trading_date: date,
    settings: StrategySettings | None = None,
    diagnostics: JsonlDiagnostics | None = None,
) -> WatchlistArtifact:
    cfg = settings or StrategySettings()
    diag = diagnostics or JsonlDiagnostics(cfg.diagnostics_dir)
    snapshot = load_research_snapshot(trading_date, settings=cfg)
    artifact = daily_selection_from_snapshot(snapshot=snapshot, settings=cfg, diagnostics=diag)
    persist_watchlist_artifact(artifact, settings=cfg)
    return artifact
