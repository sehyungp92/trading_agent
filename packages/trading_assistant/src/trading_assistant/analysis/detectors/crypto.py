"""Crypto detector behavior."""

from __future__ import annotations


from trading_assistant.schemas.detection_context import DetectionContext
from trading_assistant.schemas.strategy_suggestions import (
    SuggestionTier,
    StrategySuggestion,
)



def detect_funding_impact(
    self, bot_id: str, funding_summary: dict,
    cost_threshold: float = 0.15,
) -> list[StrategySuggestion]:
    """Detect when funding costs erode trading edge."""
    strategy_id = self._resolve_strategy_id(bot_id)
    arch_thresh = self._archetype_default(strategy_id, "funding_impact", "cost_threshold")
    if arch_thresh is not None:
        cost_threshold = arch_thresh

    suggestions: list[StrategySuggestion] = []
    funding_pct = funding_summary.get("funding_pct_of_gross", 0.0)
    funding_losers = funding_summary.get("funding_losers", [])
    funding_sample = int(funding_summary.get("coverage", 0) or 0)

    if funding_pct > cost_threshold:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.PARAMETER,
            title=f"Funding costs consuming {funding_pct:.0%} of gross PnL",
            description=(
                f"Funding paid is {funding_pct:.0%} of gross PnL (threshold: {cost_threshold:.0%}). "
                f"Consider tightening time_stop to reduce hold duration and funding exposure."
            ),
            confidence=min(0.8, 0.5 + funding_pct),
            detection_context=DetectionContext(
                detector_name="funding_impact",
                bot_id=bot_id,
                threshold_name="cost_threshold",
                threshold_value=cost_threshold,
                observed_value=funding_pct,
                sample_size=funding_sample,
            ),
        ))

    if funding_losers:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.HYPOTHESIS,
            title=f"{len(funding_losers)} trade(s) where funding exceeded PnL",
            description=(
                f"Found {len(funding_losers)} trades where cumulative funding cost "
                f"exceeded the trade's PnL. Review funding_extreme filter threshold."
            ),
            confidence=0.6,
            detection_context=DetectionContext(
                detector_name="funding_impact",
                bot_id=bot_id,
                sample_size=len(funding_losers),
            ),
        ))

    return suggestions


def detect_grade_selectivity(
    self, bot_id: str, grade_summary: dict,
    min_trades: int = 20,
) -> list[StrategySuggestion]:
    """Detect grade selectivity issues in crypto setup grading."""
    suggestions: list[StrategySuggestion] = []
    per_grade = grade_summary.get("per_grade", {})
    gap = grade_summary.get("grade_expectancy_gap", 0.0)

    a_data = per_grade.get("A", {})
    b_data = per_grade.get("B", {})
    total_trades = sum(g.get("count", 0) for g in per_grade.values())

    if total_trades < min_trades:
        return suggestions

    # B-grade negative expectancy
    if b_data.get("count", 0) >= 5 and b_data.get("avg_pnl", 0) < 0:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.HYPOTHESIS,
            title="B-grade trades have negative expectancy",
            description=(
                f"B-grade: {b_data['count']} trades, avg PnL={b_data['avg_pnl']:.4f}. "
                f"Consider disabling B-grade entries or tightening confluence requirements."
            ),
            confidence=0.7,
            detection_context=DetectionContext(detector_name="grade_selectivity"),
        ))

    # A and B within 10% - miscalibrated differential
    if a_data.get("count", 0) >= 5 and b_data.get("count", 0) >= 5:
        a_pnl = a_data.get("avg_pnl", 0)
        b_pnl = b_data.get("avg_pnl", 0)
        if a_pnl != 0 and abs(a_pnl - b_pnl) / abs(a_pnl) < 0.10:
            suggestions.append(StrategySuggestion(
                bot_id=bot_id,
                tier=SuggestionTier.PARAMETER,
                title="A/B grade performance gap is negligible",
                description=(
                    f"A avg_pnl={a_pnl:.4f}, B avg_pnl={b_pnl:.4f} -"
                    f"risk_pct differential may be miscalibrated."
                ),
                confidence=0.5,
                detection_context=DetectionContext(detector_name="grade_selectivity"),
            ))

    # Grade inversion (B outperforms A)
    if gap < 0 and a_data.get("count", 0) >= 5 and b_data.get("count", 0) >= 5:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.HYPOTHESIS,
            title="Grade criteria may be inverted - B outperforms A",
            description=(
                f"Grade expectancy gap is {gap:.4f} (A underperforms B). "
                f"Review confluence scoring criteria for grade assignment."
            ),
            confidence=0.6,
            detection_context=DetectionContext(detector_name="grade_selectivity"),
        ))

    return suggestions


def detect_confluence_quality(
    self, bot_id: str, confluence_summary: dict,
    lift_threshold: float = 0.10,
) -> list[StrategySuggestion]:
    """Detect confluence quality issues - which factors add value."""
    suggestions: list[StrategySuggestion] = []
    by_count = confluence_summary.get("by_count", {})
    by_factor = confluence_summary.get("by_factor", {})
    coverage = int(confluence_summary.get("coverage", 0) or 0)

    # Check if higher confluence count improves win rate
    sorted_counts = sorted(by_count.items(), key=lambda x: int(x[0]))
    for i in range(1, len(sorted_counts)):
        prev_key, prev_data = sorted_counts[i - 1]
        curr_key, curr_data = sorted_counts[i]
        prev_wr = prev_data.get("win_rate", 0)
        curr_wr = curr_data.get("win_rate", 0)
        curr_count = int(curr_data.get("count", 0) or 0)
        if curr_wr - prev_wr > lift_threshold and curr_count >= 5:
            suggestions.append(StrategySuggestion(
                bot_id=bot_id,
                tier=SuggestionTier.PARAMETER,
                title=f"Win rate jumps {curr_wr - prev_wr:.0%} at {curr_key} confluences",
                description=(
                    f"Win rate at {curr_key} confluences ({curr_wr:.0%}) vs "
                    f"{prev_key} ({prev_wr:.0%}). Consider raising min_confluences."
                ),
                confidence=0.6,
                detection_context=DetectionContext(
                    detector_name="confluence_quality",
                    bot_id=bot_id,
                    threshold_name="lift_threshold",
                    threshold_value=lift_threshold,
                    observed_value=curr_wr - prev_wr,
                    sample_size=curr_count,
                ),
            ))
            break  # Only report the most significant jump

    # Check for negative-lift factors
    for factor, data in by_factor.items():
        lift = data.get("lift", 0)
        if lift < - lift_threshold:
            suggestions.append(StrategySuggestion(
                bot_id=bot_id,
                tier=SuggestionTier.HYPOTHESIS,
                title=f"Confluence factor '{factor}' has negative lift ({lift:+.0%})",
                description=(
                    f"Trades WITH '{factor}' have lower win rate than trades WITHOUT it "
                    f"(lift={lift:+.4f}). Investigate whether this factor adds noise."
                ),
                confidence=0.5,
                detection_context=DetectionContext(
                    detector_name="confluence_quality",
                    bot_id=bot_id,
                    threshold_name="lift_threshold",
                    threshold_value=lift_threshold,
                    observed_value=lift,
                    sample_size=coverage,
                ),
            ))

    return suggestions


def detect_leverage_utilization(
    self, bot_id: str, leverage_summary: dict,
    utilization_warning: float = 0.80,
) -> list[StrategySuggestion]:
    """Detect leverage risk issues in crypto perpetual trading."""
    suggestions: list[StrategySuggestion] = []
    util_pct = leverage_summary.get("leverage_utilization_pct", 0)
    near_liq = leverage_summary.get("near_liquidation_count", 0)
    per_grade = leverage_summary.get("per_grade", {})
    coverage = int(leverage_summary.get("coverage", 0) or 0)

    if util_pct > utilization_warning:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.PARAMETER,
            title=f"Leverage utilization at {util_pct:.0%} of max",
            description=(
                f"Average leverage is {util_pct:.0%} of configured maximum. "
                f"Consider reducing default leverage to build safety margin."
            ),
            confidence=0.6,
            detection_context=DetectionContext(
                detector_name="leverage_utilization",
                bot_id=bot_id,
                threshold_name="utilization_warning",
                threshold_value=utilization_warning,
                observed_value=util_pct,
                sample_size=coverage,
            ),
        ))

    if near_liq > 0:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.HYPOTHESIS,
            title=f"{near_liq} trade(s) approached liquidation threshold",
            description=(
                f"Found {near_liq} trade(s) where MAE exceeded 80% of liquidation "
                f"distance. This is a critical safety flag requiring leverage reduction."
            ),
            confidence=0.9,
            detection_context=DetectionContext(
                detector_name="leverage_utilization",
                bot_id=bot_id,
                sample_size=int(near_liq),
            ),
        ))

    # Grade-leverage mismatch
    a_lev = per_grade.get("A", 0)
    b_lev = per_grade.get("B", 0)
    if b_lev >= a_lev > 0:
        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            tier=SuggestionTier.PARAMETER,
            title="B-grade trades use equal or more leverage than A-grade",
            description=(
                f"A-grade avg leverage={a_lev:.1f}, B-grade avg leverage={b_lev:.1f}. "
                f"Lower-conviction trades should use less leverage, not more."
            ),
            confidence=0.6,
            detection_context=DetectionContext(
                detector_name="leverage_utilization",
                bot_id=bot_id,
                sample_size=coverage,
            ),
        ))

    return suggestions


def detect_mtf_alignment_drift(
    self,
    bot_id: str,
    trades: list,
    min_mismatched: int = 5,
    win_rate_gap_threshold: float = 0.15,
) -> list[StrategySuggestion]:
    """Detect higher-timeframe bias disagreement hurting expectancy."""
    aligned: list = []
    mismatched: list = []
    for trade in trades:
        side = self._normalize_trade_side(self._trade_get(trade, "side", ""))
        bias = self._normalize_bias_direction(self._trade_get(trade, "bias_direction", ""))
        if not side or not bias:
            continue
        if side == bias:
            aligned.append(trade)
        else:
            mismatched.append(trade)

    if len(mismatched) < min_mismatched or not aligned:
        return []

    aligned_wr = self._win_rate(aligned)
    mismatched_wr = self._win_rate(mismatched)
    gap = aligned_wr - mismatched_wr
    aligned_avg = self._avg_pnl(aligned)
    mismatched_avg = self._avg_pnl(mismatched)
    if gap < win_rate_gap_threshold or mismatched_avg >= aligned_avg:
        return []

    return [StrategySuggestion(
        bot_id=bot_id,
        strategy_id=self._strategy_id_from_trades(bot_id, trades),
        tier=SuggestionTier.PARAMETER,
        title="Higher-timeframe bias mismatch is degrading crypto entries",
        description=(
            f"{len(mismatched)} trades disagreed with bias_direction. "
            f"Mismatched win rate {mismatched_wr:.0%} vs aligned {aligned_wr:.0%}; "
            f"avg PnL {mismatched_avg:.4f} vs {aligned_avg:.4f}. "
            "Review side/bias alignment gates before loosening entry filters."
        ),
        confidence=min(0.85, 0.55 + max(gap, 0.0)),
        detection_context=DetectionContext(
            detector_name="mtf_alignment_drift",
            bot_id=bot_id,
            threshold_name="win_rate_gap_threshold",
            threshold_value=win_rate_gap_threshold,
            observed_value=gap,
            sample_size=len(mismatched),
        ),
    )]


def detect_liquidation_proximity(
    self,
    bot_id: str,
    trades: list,
    proximity_threshold: float = 0.70,
    systemic_count: int = 3,
) -> list[StrategySuggestion]:
    """Detect trades whose MAE came too close to liquidation after leverage."""
    strategy_id = self._strategy_id_from_trades(bot_id, trades)
    arch_thresh = self._archetype_default(
        strategy_id, "liquidation_proximity", "proximity_threshold",
    )
    if arch_thresh is not None:
        proximity_threshold = arch_thresh

    offenders: list[tuple[str, float]] = []
    worst = 0.0
    for trade in trades:
        mae_r = abs(self._trade_float(trade, "mae_r", 0.0))
        leverage = self._trade_leverage(trade)
        proximity = mae_r * leverage
        worst = max(worst, proximity)
        if proximity > proximity_threshold + 1e-9:
            offenders.append((str(self._trade_get(trade, "trade_id", "")), proximity))

    if not offenders:
        return []

    tier = SuggestionTier.PARAMETER if len(offenders) >= systemic_count else SuggestionTier.HYPOTHESIS
    return [StrategySuggestion(
        bot_id=bot_id,
        strategy_id=strategy_id,
        tier=tier,
        title=f"{len(offenders)} crypto trade(s) breached liquidation proximity guard",
        description=(
            f"Worst mae_r * leverage was {worst:.2f}; guardrail is "
            f"{proximity_threshold:.2f}. Treat this as a process-quality "
            "and leverage-cap issue before tuning entries."
        ),
        confidence=0.9 if len(offenders) >= systemic_count else 0.75,
        detection_context=DetectionContext(
            detector_name="liquidation_proximity",
            bot_id=bot_id,
            threshold_name="proximity_threshold",
            threshold_value=proximity_threshold,
            observed_value=worst,
            sample_size=len(offenders),
        ),
    )]


def detect_symbol_concentration(
    self,
    bot_id: str,
    trades: list,
    concentration_threshold: float = 0.70,
    min_trades: int = 10,
) -> list[StrategySuggestion]:
    """Detect BTC/ETH/SOL loss concentration."""
    tracked = {"BTC", "ETH", "SOL"}
    loss_by_symbol: dict[str, float] = {symbol: 0.0 for symbol in tracked}
    count_by_symbol: dict[str, int] = {symbol: 0 for symbol in tracked}
    for trade in trades:
        symbol = self._base_symbol(self._trade_get(trade, "pair", ""))
        if symbol not in tracked:
            continue
        count_by_symbol[symbol] += 1
        pnl = self._trade_float(trade, "pnl", 0.0)
        if pnl < 0:
            loss_by_symbol[symbol] += abs(pnl)

    total_loss = sum(loss_by_symbol.values())
    if total_loss <= 0:
        return []

    suggestions: list[StrategySuggestion] = []
    for symbol, loss in loss_by_symbol.items():
        share = loss / total_loss if total_loss else 0.0
        if share >= concentration_threshold and count_by_symbol[symbol] >= min_trades:
            suggestions.append(StrategySuggestion(
                bot_id=bot_id,
                strategy_id=self._strategy_id_from_trades(bot_id, trades),
                tier=SuggestionTier.PARAMETER,
                title=f"{symbol} dominates crypto losses ({share:.0%})",
                description=(
                    f"{symbol} accounts for {share:.0%} of BTC/ETH/SOL gross losses "
                    f"across {count_by_symbol[symbol]} trades. Review symbol risk parity "
                    "or symbol-specific filters before changing global parameters."
                ),
                confidence=0.7,
                detection_context=DetectionContext(
                    detector_name="symbol_concentration",
                    bot_id=bot_id,
                    threshold_name="concentration_threshold",
                    threshold_value=concentration_threshold,
                    observed_value=share,
                    sample_size=count_by_symbol[symbol],
                ),
            ))
    return suggestions


def detect_session_patterns_24_7(
    self,
    bot_id: str,
    trades: list,
    min_trades: int = 10,
    negative_avg_pnl_threshold: float = 0.0,
) -> list[StrategySuggestion]:
    """Detect persistent Asia/EU/US UTC session underperformance."""
    sessions: dict[str, list] = {"Asia": [], "EU": [], "US": []}
    for trade in trades:
        ts = self._trade_timestamp(trade)
        if ts is None:
            continue
        hour = ts.hour
        if 0 <= hour <= 4:
            sessions["Asia"].append(trade)
        elif 7 <= hour <= 11:
            sessions["EU"].append(trade)
        elif 13 <= hour <= 17:
            sessions["US"].append(trade)

    suggestions: list[StrategySuggestion] = []
    for session, bucket in sessions.items():
        if len(bucket) < min_trades:
            continue
        avg_pnl = self._avg_pnl(bucket)
        net_pnl = sum(self._trade_float(t, "pnl", 0.0) for t in bucket)
        if avg_pnl < negative_avg_pnl_threshold and net_pnl < 0:
            suggestions.append(StrategySuggestion(
                bot_id=bot_id,
                strategy_id=self._strategy_id_from_trades(bot_id, trades),
                tier=SuggestionTier.HYPOTHESIS,
                title=f"{session} UTC session is negative for crypto_trader",
                description=(
                    f"{session} window has {len(bucket)} trades, net PnL {net_pnl:.4f}, "
                    f"avg PnL {avg_pnl:.4f}. Validate 24/7 session liquidity before "
                    "changing entry thresholds."
                ),
                confidence=0.6,
                detection_context=DetectionContext(
                    detector_name="session_patterns_24_7",
                    bot_id=bot_id,
                    threshold_name="min_trades",
                    threshold_value=float(min_trades),
                    observed_value=float(len(bucket)),
                    sample_size=len(bucket),
                ),
            ))
    return suggestions


def detect_funding_trend(
    self,
    bot_id: str,
    trades: list,
    cost_threshold: float = 0.15,
    rising_weeks: int = 3,
) -> list[StrategySuggestion]:
    """Detect funding cost rising as a share of gross PnL across weeks."""
    strategy_id = self._strategy_id_from_trades(bot_id, trades)
    arch_thresh = self._archetype_default(strategy_id, "funding_trend", "cost_threshold")
    if arch_thresh is not None:
        cost_threshold = arch_thresh

    weekly_by_symbol: dict[str, dict[tuple[int, int], dict[str, float]]] = {}
    count_by_symbol: dict[str, int] = {}
    for trade in trades:
        ts = self._trade_timestamp(trade)
        if ts is None:
            continue
        iso = ts.isocalendar()
        symbol = self._base_symbol(self._trade_get(trade, "pair", "")) or "UNKNOWN"
        count_by_symbol[symbol] = count_by_symbol.get(symbol, 0) + 1
        weekly = weekly_by_symbol.setdefault(symbol, {})
        bucket = weekly.setdefault((iso.year, iso.week), {"funding": 0.0, "gross": 0.0})
        bucket["funding"] += abs(self._trade_float(trade, "funding_paid", 0.0))
        bucket["gross"] += abs(self._trade_float(trade, "pnl", 0.0))

    suggestions: list[StrategySuggestion] = []
    for symbol, weekly in sorted(weekly_by_symbol.items()):
        ratios: list[float] = []
        for key in sorted(weekly):
            gross = weekly[key]["gross"]
            if gross > 0:
                ratios.append(weekly[key]["funding"] / gross)

        if len(ratios) < rising_weeks:
            continue
        recent = ratios[-rising_weeks:]
        if recent[-1] <= cost_threshold:
            continue
        if not all(a < b for a, b in zip(recent, recent[1:], strict=False)):
            continue

        suggestions.append(StrategySuggestion(
            bot_id=bot_id,
            strategy_id=strategy_id,
            tier=SuggestionTier.PARAMETER,
            title=f"{symbol} funding cost ratio rose for {rising_weeks} straight weeks",
            description=(
                f"{symbol} funding/gross PnL ratios: {', '.join(f'{r:.0%}' for r in recent)}; "
                f"latest exceeds {cost_threshold:.0%}. Review funding_threshold and "
                "time-stop settings before extending holds."
            ),
            confidence=0.75,
            detection_context=DetectionContext(
                detector_name="funding_trend",
                bot_id=bot_id,
                threshold_name="cost_threshold",
                threshold_value=cost_threshold,
                observed_value=recent[-1],
                sample_size=count_by_symbol.get(symbol, 0),
            ),
        ))
    return suggestions
