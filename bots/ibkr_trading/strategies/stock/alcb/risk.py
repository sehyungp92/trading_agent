"""Risk engine helpers for ALCB."""

from __future__ import annotations

from math import floor, sqrt
from statistics import fmean

from .config import StrategySettings
from .models import (
    Campaign,
    CandidateItem,
    CompressionTier,
    Direction,
    EntryType,
    PortfolioState,
    PositionPlan,
    Regime,
)
from .signals import atr_from_bars


def is_volatile_name(item: CandidateItem) -> bool:
    if item.median_spread_pct >= 0.0035:
        return True
    atr14 = atr_from_bars(item.daily_bars, 14)
    return atr14 > 0 and item.price > 0 and (atr14 / item.price) >= 0.05


def base_risk_fraction(item: CandidateItem, settings: StrategySettings) -> float:
    return settings.volatile_base_risk_fraction if is_volatile_name(item) else settings.base_risk_fraction


def regime_mult(direction: Direction, stock_regime: Regime, market_regime: Regime) -> float:
    if direction == Direction.LONG:
        if stock_regime == Regime.BULL and market_regime in (Regime.BULL, Regime.TRANSITIONAL):
            return 1.00
        if stock_regime in (Regime.BULL, Regime.TRANSITIONAL):
            return 0.80
        if stock_regime == Regime.CHOP:
            return 0.60
        return 0.0
    if stock_regime == Regime.BEAR and market_regime in (Regime.BEAR, Regime.TRANSITIONAL):
        return 1.00
    if stock_regime in (Regime.BEAR, Regime.TRANSITIONAL):
        return 0.80
    if stock_regime == Regime.CHOP:
        return 0.60
    return 0.0


def quality_mult(campaign: Campaign, intraday_score: int, settings: StrategySettings | None = None) -> float:
    if campaign.box is None or campaign.breakout is None:
        return 0.0
    mult = 1.0
    if campaign.box.tier == CompressionTier.GOOD:
        mult *= 1.05
    elif campaign.box.tier == CompressionTier.LOOSE:
        mult *= 0.85
    disp_ratio = campaign.breakout.disp_value / campaign.breakout.disp_threshold if campaign.breakout.disp_threshold > 0 else 0.0
    if disp_ratio >= 1.25:
        mult *= 1.05
    elif disp_ratio < 1.0:
        return 0.0
    top_tier = settings.evidence_score_top_tier if settings else 6
    top_mult = settings.quality_mult_top_score if settings else 1.10
    if intraday_score >= top_tier:
        mult *= top_mult
    elif intraday_score >= 4:
        mult *= 1.05
    elif intraday_score == 3:
        mult *= 0.95
    elif intraday_score == 2:
        mult *= 0.85
    else:
        return 0.0
    return max(0.0, min(mult, 1.0))


def _returns_from_candidate(item: CandidateItem, lookback: int) -> list[float]:
    closes = [float(bar.close) for bar in item.daily_bars]
    if len(closes) < lookback + 2:
        return []
    values: list[float] = []
    for prev, current in zip(closes[-lookback - 1 : -1], closes[-lookback:]):
        if prev <= 0:
            values.append(0.0)
        else:
            values.append((current - prev) / prev)
    return values


def rolling_corr_daily(symbol_a: str, symbol_b: str, items: dict[str, CandidateItem], lookback: int) -> float:
    left = items.get(symbol_a)
    right = items.get(symbol_b)
    if left is None or right is None:
        return 0.0
    xs = _returns_from_candidate(left, lookback)
    ys = _returns_from_candidate(right, lookback)
    size = min(len(xs), len(ys))
    if size < 5:
        return 0.0
    xs = xs[-size:]
    ys = ys[-size:]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / size
    std_x = sqrt(sum((x - mean_x) ** 2 for x in xs) / size)
    std_y = sqrt(sum((y - mean_y) ** 2 for y in ys) / size)
    if std_x <= 0 or std_y <= 0:
        return 0.0
    return cov / (std_x * std_y)


def correlation_mult(symbol: str, direction: Direction, portfolio: PortfolioState, items: dict[str, CandidateItem], settings: StrategySettings) -> float:
    if not portfolio.open_positions:
        return 1.0
    max_corr = 0.0
    for other_symbol, position in portfolio.open_positions.items():
        if position.direction != direction:
            continue
        max_corr = max(max_corr, rolling_corr_daily(symbol, other_symbol, items, settings.corr_lookback))
    return 0.80 if max_corr > settings.corr_threshold else 1.0


def high_ranked_name(item: CandidateItem) -> bool:
    return item.selection_score >= 8


def _regime_from_value(value: Regime | str | None) -> Regime:
    if isinstance(value, Regime):
        return value
    if isinstance(value, str) and value in Regime._value2member_map_:
        return Regime(value)
    return Regime.TRANSITIONAL


def breakout_quality_strong(campaign: Campaign) -> bool:
    if campaign.breakout is None or campaign.breakout.disp_threshold <= 0:
        return False
    return (campaign.breakout.disp_value / campaign.breakout.disp_threshold) >= 1.10


def choose_stop(
    entry_type: EntryType,
    direction: Direction,
    item: CandidateItem,
    campaign: Campaign,
    settings: StrategySettings,
    *,
    stock_regime: Regime | str | None = None,
    market_regime: Regime | str | None = None,
) -> float:
    if campaign.box is None:
        return item.price
    atr14 = atr_from_bars(item.daily_bars, 14)
    stop_mult = settings.atr_stop_mult_volatile if is_volatile_name(item) else settings.atr_stop_mult_std
    buffer = stop_mult * atr14
    live_stock_regime = _regime_from_value(stock_regime or item.stock_regime)
    live_market_regime = _regime_from_value(market_regime or item.market_regime)
    stock_aligned = (
        direction == Direction.LONG and live_stock_regime == Regime.BULL
    ) or (
        direction == Direction.SHORT and live_stock_regime == Regime.BEAR
    )
    market_not_opposing = not (
        (direction == Direction.LONG and live_market_regime == Regime.BEAR)
        or (direction == Direction.SHORT and live_market_regime == Regime.BULL)
    )
    midpoint_allowed = (
        entry_type == EntryType.A_AVWAP_RETEST
        and campaign.box.tier == CompressionTier.GOOD
        and high_ranked_name(item)
        and stock_aligned
        and market_not_opposing
        and breakout_quality_strong(campaign)
    )
    if midpoint_allowed:
        return (campaign.box.mid - buffer) if direction == Direction.LONG else (campaign.box.mid + buffer)
    return (campaign.box.low - buffer) if direction == Direction.LONG else (campaign.box.high + buffer)


def choose_targets(direction: Direction, entry_price: float, stop_price: float, stock_regime: Regime, market_regime: Regime, settings: StrategySettings, *, quality_mult_value: float = 0.0) -> tuple[float, float]:
    r = abs(entry_price - stop_price)
    aligned = (
        (direction == Direction.LONG and stock_regime == Regime.BULL and market_regime in (Regime.BULL, Regime.TRANSITIONAL))
        or (direction == Direction.SHORT and stock_regime == Regime.BEAR and market_regime in (Regime.BEAR, Regime.TRANSITIONAL))
    )
    high_conviction = quality_mult_value >= settings.high_conviction_quality_mult_min
    if aligned:
        tp1_r = settings.tp1_aligned_r
        tp2_r = settings.tp2_aligned_r_high_conviction if high_conviction else settings.tp2_aligned_r
    else:
        tp1_r = settings.tp1_neutral_r
        tp2_r = settings.tp2_neutral_r_high_conviction if high_conviction else settings.tp2_neutral_r
    if direction == Direction.LONG:
        return entry_price + (tp1_r * r), entry_price + (tp2_r * r)
    return entry_price - (tp1_r * r), entry_price - (tp2_r * r)


def estimate_cost_buffer_per_share(item: CandidateItem, entry_price: float) -> float:
    spread_cost = max(item.median_spread_pct * entry_price, item.tick_size)
    if item.adv20_usd >= 50_000_000:
        slippage = 0.01
    elif item.adv20_usd >= 20_000_000:
        slippage = 0.02
    else:
        slippage = 0.03
    return spread_cost + slippage


def position_size(
    item: CandidateItem,
    entry_price: float,
    stop_price: float,
    direction: Direction,
    campaign: Campaign,
    intraday_score: int,
    portfolio: PortfolioState,
    items: dict[str, CandidateItem],
    settings: StrategySettings,
    *,
    entry_type: EntryType = EntryType.A_AVWAP_RETEST,
    stock_regime: Regime | str | None = None,
    market_regime: Regime | str | None = None,
) -> PositionPlan | None:
    live_stock_regime = _regime_from_value(stock_regime or item.stock_regime)
    live_market_regime = _regime_from_value(market_regime or item.market_regime)
    reg_mult = regime_mult(direction, live_stock_regime, live_market_regime)
    if reg_mult <= 0:
        return None
    q_mult = quality_mult(campaign, intraday_score, settings)
    if q_mult <= 0:
        return None
    c_mult = correlation_mult(item.symbol, direction, portfolio, items, settings)
    base = base_risk_fraction(item, settings)
    final_risk_fraction = base * reg_mult * q_mult * c_mult
    final_risk_fraction = min(max(final_risk_fraction, settings.final_risk_min_mult * base), settings.final_risk_max_mult * base)
    equity = portfolio.account_equity
    risk_dollars = equity * final_risk_fraction
    risk_per_share = abs(entry_price - stop_price) + estimate_cost_buffer_per_share(item, entry_price)
    if risk_per_share <= 0:
        return None
    qty = int(floor(risk_dollars / risk_per_share))
    if qty < 1:
        return None
    max_participation = settings.thin_participation_30m if item.adv20_usd < 50_000_000 else settings.max_participation_30m
    max_qty = int(max(item.median_30m_volume, item.average_30m_volume, 1.0) * max_participation)
    qty = min(qty, max_qty)
    if qty < 1:
        return None
    risk_dollars = qty * risk_per_share
    tp1, tp2 = choose_targets(direction, entry_price, stop_price, live_stock_regime, live_market_regime, settings, quality_mult_value=q_mult)
    return PositionPlan(
        symbol=item.symbol,
        direction=direction,
        entry_type=entry_type,
        entry_price=entry_price,
        stop_price=stop_price,
        tp1_price=tp1,
        tp2_price=tp2,
        quantity=qty,
        risk_per_share=risk_per_share,
        risk_dollars=risk_dollars,
        quality_mult=q_mult,
        regime_mult=reg_mult,
        corr_mult=c_mult,
    )


def add_position_quantity(
    item: CandidateItem,
    position: PositionPlan | None,
    portfolio: PortfolioState,
    entry_price: float,
    stop_price: float,
    settings: StrategySettings,
) -> int:
    if position is None:
        return 0
    base = portfolio.account_equity * base_risk_fraction(item, settings) * 0.5
    risk_per_share = abs(entry_price - stop_price) + estimate_cost_buffer_per_share(item, entry_price)
    if risk_per_share <= 0:
        return 0
    qty = int(base / risk_per_share)
    max_qty = int(max(item.median_30m_volume, item.average_30m_volume, 1.0) * settings.max_participation_30m)
    return max(0, min(qty, max_qty))


def event_block(item: CandidateItem) -> bool:
    return item.earnings_risk_flag


def portfolio_heat_after(plan: PositionPlan, portfolio: PortfolioState) -> float:
    current = portfolio.open_risk_dollars() + portfolio.pending_entry_risk_dollars()
    proposed = current + plan.risk_dollars * portfolio.correlation_heat_penalty(plan.symbol, plan.direction)
    if portfolio.account_equity <= 0:
        return 0.0
    return proposed / portfolio.account_equity


def sector_limit_pass(item: CandidateItem, portfolio: PortfolioState, symbol_to_sector: dict[str, str], settings: StrategySettings) -> bool:
    sector_symbols = {
        symbol
        for symbol in set(portfolio.open_positions) | set(portfolio.pending_entry_risk)
        if symbol_to_sector.get(symbol) == item.sector
    }
    return len(sector_symbols) < settings.max_positions_per_sector


def max_positions_pass(portfolio: PortfolioState, settings: StrategySettings) -> bool:
    return portfolio.occupied_slots() < settings.max_positions


def estimate_round_trip_friction(item: CandidateItem, quantity: int, entry_price: float) -> float:
    return quantity * estimate_cost_buffer_per_share(item, entry_price)


def friction_gate_pass(item: CandidateItem, plan: PositionPlan, settings: StrategySettings) -> bool:
    friction = estimate_round_trip_friction(item, plan.quantity, plan.entry_price)
    return friction <= settings.max_friction_to_risk * plan.risk_dollars


# ---------------------------------------------------------------------------
# Momentum continuation (T1) risk helpers
# ---------------------------------------------------------------------------

def momentum_regime_mult(regime_tier: str, settings: StrategySettings) -> float:
    """Sizing multiplier based on market regime tier."""
    if regime_tier == "A":
        return settings.regime_mult_a
    if regime_tier == "B":
        return settings.regime_mult_b
    return settings.regime_mult_c


def sector_sizing_mult(sector: str, settings: StrategySettings) -> float:
    """Sector-weighted sizing multiplier from P14 optimization."""
    if sector == "Financials":
        return settings.sector_mult_financials
    if sector == "Communication Services":
        return settings.sector_mult_communication
    if sector == "Industrials":
        return settings.sector_mult_industrials
    if sector == "Consumer Discretionary":
        return settings.sector_mult_consumer_disc
    if sector == "Healthcare":
        return settings.sector_mult_healthcare
    return 1.0


def conditional_entry_blocked(
    sector: str,
    entry_type: str | EntryType,
    entry_bar_index: int,
    momentum_score: int,
    settings: StrategySettings,
    score_detail: dict | None = None,
) -> bool:
    """Return True when an optimizer-configured completed-bar cohort is blocked."""
    entry_type_key = _entry_type_key(entry_type)
    sector_key = _sector_key(sector)
    bar = int(entry_bar_index)
    score = int(momentum_score)

    if bar in _int_set(settings.block_entry_bars):
        return True
    if _contains_key(settings.entry_type_bar_blocklist, f"{entry_type_key}:{bar}"):
        return True
    if _contains_key(settings.entry_score_blocklist, f"{entry_type_key}:{score}"):
        return True
    if _contains_key(settings.entry_score_blocklist, f"*:{score}"):
        return True
    if _detail_matches_any(settings.entry_detail_blocklist, entry_type_key, score, score_detail):
        return True
    if _contains_key(settings.sector_entry_blocklist, f"{sector_key}:{entry_type_key}"):
        return True
    if _contains_key(settings.sector_entry_blocklist, f"{sector_key}:*"):
        return True
    return False


def conditional_entry_size_mult(
    sector: str,
    entry_type: str | EntryType,
    entry_bar_index: int,
    momentum_score: int,
    settings: StrategySettings,
    score_detail: dict | None = None,
) -> float:
    """Multiplicative sizing overlay for completed-bar entry cohorts."""
    entry_type_key = _entry_type_key(entry_type)
    sector_key = _sector_key(sector)
    bar = int(entry_bar_index)
    score = int(momentum_score)

    mult = 1.0
    mult *= _lookup_mult(settings.entry_bar_size_mults, str(bar))
    mult *= _lookup_mult(settings.entry_type_bar_size_mults, f"{entry_type_key}:{bar}")
    mult *= _lookup_mult(settings.entry_score_size_mults, f"{entry_type_key}:{score}")
    mult *= _lookup_mult(settings.entry_score_size_mults, f"*:{score}")
    mult *= _lookup_detail_mult(settings.entry_detail_size_mults, entry_type_key, score, score_detail)
    mult *= _lookup_mult(settings.sector_entry_size_mults, f"{sector_key}:{entry_type_key}")
    mult *= _lookup_mult(settings.sector_entry_size_mults, f"{sector_key}:*")
    return max(0.0, mult)


def _entry_type_key(entry_type: str | EntryType) -> str:
    raw = getattr(entry_type, "value", entry_type)
    return str(raw).strip().upper()


def _sector_key(sector: str) -> str:
    return str(sector or "").strip().lower()


def _control_key(value: object) -> str:
    text = str(value).strip()
    if ":" not in text:
        return text.lower()
    left, right = text.split(":", 1)
    return f"{left.strip().lower()}:{right.strip().upper()}"


def _contains_key(values: object, key: str) -> bool:
    wanted = _control_key(key)
    if isinstance(values, str):
        iterable = [part.strip() for part in values.split(",") if part.strip()]
    else:
        iterable = list(values or [])
    return any(_control_key(value) == wanted for value in iterable)


def _int_set(values: object) -> set[int]:
    if isinstance(values, str):
        iterable = [part.strip() for part in values.split(",") if part.strip()]
    else:
        iterable = list(values or [])
    result: set[int] = set()
    for value in iterable:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _lookup_mult(mapping: object, key: str) -> float:
    if not isinstance(mapping, dict):
        return 1.0
    wanted = _control_key(key)
    for raw_key, raw_value in mapping.items():
        if _control_key(raw_key) != wanted:
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _detail_matches_any(values: object, entry_type_key: str, score: int, score_detail: dict | None) -> bool:
    if isinstance(values, str):
        iterable = [part.strip() for part in values.split(",") if part.strip()]
    else:
        iterable = list(values or [])
    return any(_detail_control_matches(value, entry_type_key, score, score_detail) for value in iterable)


def _lookup_detail_mult(mapping: object, entry_type_key: str, score: int, score_detail: dict | None) -> float:
    if not isinstance(mapping, dict):
        return 1.0
    mult = 1.0
    for raw_key, raw_value in mapping.items():
        if not _detail_control_matches(raw_key, entry_type_key, score, score_detail):
            continue
        try:
            mult *= float(raw_value)
        except (TypeError, ValueError):
            continue
    return mult


def _detail_control_matches(raw_key: object, entry_type_key: str, score: int, score_detail: dict | None) -> bool:
    """Match ENTRY:SCORE:detail controls against completed signal-bar score detail.

    Accepted forms are ENTRY:detail, ENTRY:SCORE:detail, *:detail, and *:SCORE:detail.
    Prefix the detail name with ! to target absent/falsy score components.
    """
    parts = [part.strip() for part in str(raw_key).split(":") if part.strip()]
    if len(parts) == 2:
        type_part, detail_part = parts
        score_part = "*"
    elif len(parts) == 3:
        type_part, score_part, detail_part = parts
    else:
        return False

    if type_part != "*" and type_part.upper() != entry_type_key:
        return False
    if score_part != "*":
        try:
            if int(score_part) != int(score):
                return False
        except (TypeError, ValueError):
            return False

    return _score_detail_present(score_detail, detail_part)


def _score_detail_present(score_detail: dict | None, detail_part: str) -> bool:
    want_absent = detail_part.startswith("!")
    detail_name = detail_part[1:] if want_absent else detail_part
    normalized = str(detail_name).strip().lower()
    values = score_detail or {}
    present = False
    for raw_key, raw_value in values.items():
        if str(raw_key).strip().lower() == normalized:
            present = bool(raw_value)
            break
    return not present if want_absent else present


def momentum_stop_price(
    entry_price: float,
    or_low: float,
    entry_bar_low: float,
    atr: float,
    settings: StrategySettings,
) -> float:
    """Compute stop price for a long momentum entry.

    Uses the lower of OR low and entry bar low, bounded by ATR.
    """
    if settings.use_or_low_stop:
        structural_stop = min(or_low, entry_bar_low)
    else:
        structural_stop = entry_bar_low

    atr_stop = entry_price - (settings.stop_atr_multiple * atr)
    # Use the tighter of structural and ATR stop (higher = tighter for longs)
    stop = max(structural_stop, atr_stop)
    # Never let stop be above entry
    return min(stop, entry_price - 0.01)


def momentum_size_mult(momentum_score: int, settings: StrategySettings) -> float:
    """Score-bucket sizing with the minimum score bucket normalized to 1.0."""
    if momentum_score <= settings.momentum_score_min:
        return 1.0
    if momentum_score >= 7:
        return settings.momentum_size_mult_score_7_plus
    if momentum_score == 6:
        return settings.momentum_size_mult_score_6
    if momentum_score == 5:
        return settings.momentum_size_mult_score_5
    if momentum_score == 4:
        return settings.momentum_size_mult_score_4
    if momentum_score == 3:
        return settings.momentum_size_mult_score_3
    return 1.0


def momentum_position_size(
    item: CandidateItem,
    entry_price: float,
    stop_price: float,
    regime_tier: str,
    momentum_score: int,
    portfolio: PortfolioState,
    settings: StrategySettings,
    *,
    entry_type: EntryType = EntryType.OR_BREAKOUT,
) -> PositionPlan | None:
    """Compute position size for a momentum entry."""
    reg_mult = momentum_regime_mult(regime_tier, settings)
    if reg_mult <= 0:
        return None

    size_mult = momentum_size_mult(momentum_score, settings)
    base = base_risk_fraction(item, settings)
    final_risk_fraction = base * reg_mult * size_mult

    equity = portfolio.account_equity
    risk_dollars = equity * final_risk_fraction
    risk_per_share = abs(entry_price - stop_price) + estimate_cost_buffer_per_share(item, entry_price)
    if risk_per_share <= 0:
        return None

    qty = int(floor(risk_dollars / risk_per_share))
    if qty < 1:
        return None

    max_participation = settings.thin_participation_30m if item.adv20_usd < 50_000_000 else settings.max_participation_30m
    max_qty = int(max(item.median_30m_volume, item.average_30m_volume, 1.0) * max_participation)
    qty = min(qty, max_qty)
    if qty < 1:
        return None

    risk_dollars = qty * risk_per_share
    return PositionPlan(
        symbol=item.symbol,
        direction=Direction.LONG,
        entry_type=entry_type,
        entry_price=entry_price,
        stop_price=stop_price,
        tp1_price=0.0,
        tp2_price=0.0,
        quantity=qty,
        risk_per_share=risk_per_share,
        risk_dollars=risk_dollars,
        quality_mult=1.0,
        regime_mult=reg_mult,
        corr_mult=1.0,
    )
