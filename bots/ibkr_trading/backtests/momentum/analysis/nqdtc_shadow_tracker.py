"""Shadow trade simulation for NQDTC v2.0 rejected candidates.

Tracks what would have happened if a gate had NOT rejected a breakout signal.
Uses the NQDTC stop lifecycle: tiered TPs, chandelier trail, stale exit, BE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from strategies.momentum.nqdtc import config as C
from strategies.momentum.nqdtc.models import Direction

logger = logging.getLogger(__name__)


@dataclass
class NQDTCShadowCandidate:
    """A candidate that was rejected by one or more gates."""

    direction: int
    filter_name: str       # first_block_reason
    time: datetime
    entry_price: float
    stop_price: float
    session: str = ""
    score: float = 0.0
    displacement: float = 0.0
    composite_regime: str = ""


@dataclass
class NQDTCShadowResult:
    """Outcome of simulating a shadow candidate."""

    candidate: NQDTCShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    reached_tp1: bool = False
    reached_tp2: bool = False


@dataclass
class NQDTCFilterStats:
    """Aggregated stats for one gate."""

    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    pct_reach_tp1: float = 0.0
    pct_reach_tp2: float = 0.0
    pct_above_1r: float = 0.0
    net_missed_ev: float = 0.0
    net_avoided_loss: float = 0.0
    virtual_max_dd_contribution: float = 0.0


class NQDTCShadowTracker:
    """Track rejected NQDTC candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[NQDTCShadowCandidate] = []
        self.results: list[NQDTCShadowResult] = []

    def record_rejection(
        self,
        direction: int,
        filter_name: str,
        time: datetime,
        entry_price: float,
        stop_price: float,
        session: str = "",
        score: float = 0.0,
        displacement: float = 0.0,
        composite_regime: str = "",
    ) -> None:
        self.rejections.append(NQDTCShadowCandidate(
            direction=direction, filter_name=filter_name,
            time=time, entry_price=entry_price, stop_price=stop_price,
            session=session, score=score, displacement=displacement,
            composite_regime=composite_regime,
        ))

    def simulate_shadows(
        self,
        five_min_opens: np.ndarray,
        five_min_highs: np.ndarray,
        five_min_lows: np.ndarray,
        five_min_closes: np.ndarray,
        five_min_times: np.ndarray,
    ) -> list[NQDTCShadowResult]:
        """Forward-simulate each rejected candidate using NQDTC exit lifecycle."""
        self.results.clear()

        for cand in self.rejections:
            result = self._simulate_one(
                cand,
                five_min_opens, five_min_highs, five_min_lows,
                five_min_closes, five_min_times,
            )
            self.results.append(result)

        return self.results

    def _simulate_one(
        self,
        cand: NQDTCShadowCandidate,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
    ) -> NQDTCShadowResult:
        """Simulate one shadow candidate with NQDTC exit lifecycle.

        Simplified lifecycle (uses conservative trade-through fill model):
        1. Fill at entry price within 3 bars (A-type TTL)
        2. Track MFE/MAE
        3. TP1 at Neutral tier (+1.0R) -> BE
        4. TP2 at Neutral tier (+2.0R)
        5. Stale exit after STALE_BARS_NORMAL 30m bars
        6. Protective stop
        """
        cand_ts = (
            np.datetime64(cand.time, "ns")
            if not isinstance(cand.time, np.datetime64)
            else cand.time
        )
        start_idx = int(np.searchsorted(times, cand_ts, side="right"))

        if start_idx >= len(times):
            return NQDTCShadowResult(candidate=cand)

        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return NQDTCShadowResult(candidate=cand)

        tick = C.NQ_SPECS[C.DEFAULT_SYMBOL]["tick"]

        # TTL: 3 5m bars (15 min)
        ttl = 3

        # Phase 1: check fill within TTL (conservative: trade-through by 1 tick)
        fill_price = 0.0
        fill_idx = -1

        for i in range(start_idx, min(start_idx + ttl, len(opens))):
            if cand.direction == Direction.LONG:
                if opens[i] <= cand.entry_price - tick:
                    fill_price = float(opens[i])
                    fill_idx = i
                    break
                elif lows[i] <= cand.entry_price - tick:
                    fill_price = cand.entry_price
                    fill_idx = i
                    break
            else:
                if opens[i] >= cand.entry_price + tick:
                    fill_price = float(opens[i])
                    fill_idx = i
                    break
                elif highs[i] >= cand.entry_price + tick:
                    fill_price = cand.entry_price
                    fill_idx = i
                    break

        if fill_idx < 0:
            return NQDTCShadowResult(candidate=cand, filled=False)

        # Phase 2: exit lifecycle
        current_stop = cand.stop_price
        mfe_price = fill_price
        mae_price = fill_price
        mfe_r = 0.0
        mae_r = 0.0
        tp1_reached = False
        tp2_reached = False
        be_triggered = False
        # Stale: 12 30m bars ≈ 72 5m bars
        stale_bars_5m = C.STALE_BARS_NORMAL * 6

        # Use Neutral exit tier targets
        tp1_r = 1.0
        tp2_r = 2.0

        for j in range(fill_idx + 1, len(closes)):
            bars_held = j - fill_idx

            # MFE / MAE
            if cand.direction == Direction.LONG:
                if highs[j] > mfe_price:
                    mfe_price = float(highs[j])
                if lows[j] < mae_price:
                    mae_price = float(lows[j])
                cur_mfe = (mfe_price - fill_price) / r_base
                cur_mae = (fill_price - mae_price) / r_base
                cur_r = (float(closes[j]) - fill_price) / r_base
            else:
                if lows[j] < mfe_price:
                    mfe_price = float(lows[j])
                if highs[j] > mae_price:
                    mae_price = float(highs[j])
                cur_mfe = (fill_price - mfe_price) / r_base
                cur_mae = (mae_price - fill_price) / r_base
                cur_r = (fill_price - float(closes[j])) / r_base

            mfe_r = max(mfe_r, cur_mfe)
            mae_r = max(mae_r, cur_mae)

            # Stop check
            if cand.direction == Direction.LONG:
                if lows[j] <= current_stop:
                    pnl_r = (current_stop - fill_price) / r_base
                    return NQDTCShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held,
                        reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                    )
            else:
                if highs[j] >= current_stop:
                    pnl_r = (fill_price - current_stop) / r_base
                    return NQDTCShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held,
                        reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                    )

            # TP1
            if not tp1_reached and cur_mfe >= tp1_r:
                tp1_reached = True
                if not be_triggered:
                    be_triggered = True
                    current_stop = fill_price

            # TP2
            if tp1_reached and not tp2_reached and cur_mfe >= tp2_r:
                tp2_reached = True

            # Stale exit
            if bars_held >= stale_bars_5m and cur_r < C.STALE_R_THRESHOLD:
                exit_px = float(closes[j])
                if cand.direction == Direction.LONG:
                    pnl_r = (exit_px - fill_price) / r_base
                else:
                    pnl_r = (fill_price - exit_px) / r_base
                return NQDTCShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=exit_px, exit_reason="STALE",
                    r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held,
                    reached_tp1=tp1_reached, reached_tp2=tp2_reached,
                )

        # End of data
        last_close = float(closes[-1])
        if cand.direction == Direction.LONG:
            pnl_r = (last_close - fill_price) / r_base
        else:
            pnl_r = (fill_price - last_close) / r_base

        return NQDTCShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
            bars_held=len(closes) - fill_idx,
            reached_tp1=tp1_reached, reached_tp2=tp2_reached,
        )

    def get_filter_summary(self) -> dict[str, NQDTCFilterStats]:
        """Compute per-gate stats from simulation results."""
        by_filter: dict[str, list[NQDTCShadowResult]] = {}
        for r in self.results:
            name = r.candidate.filter_name
            by_filter.setdefault(name, []).append(r)

        rej_counts: dict[str, int] = {}
        for c in self.rejections:
            rej_counts[c.filter_name] = rej_counts.get(c.filter_name, 0) + 1

        summaries: dict[str, NQDTCFilterStats] = {}
        for name, results in by_filter.items():
            filled = [r for r in results if r.filled]
            r_multiples = [r.r_multiple for r in filled]

            stats = NQDTCFilterStats(filter_name=name)
            stats.rejected_count = rej_counts.get(name, len(results))
            stats.simulated_count = len(results)
            stats.filled_count = len(filled)

            if filled:
                arr = np.array(r_multiples)
                stats.avg_shadow_r = float(np.mean(arr))
                stats.pct_above_1r = float(np.mean(arr > 1.0)) * 100
                stats.pct_reach_tp1 = float(np.mean([r.reached_tp1 for r in filled])) * 100
                stats.pct_reach_tp2 = float(np.mean([r.reached_tp2 for r in filled])) * 100
                stats.net_missed_ev = float(np.sum(arr[arr > 0]))
                stats.net_avoided_loss = float(np.sum(np.abs(arr[arr < 0])))
                # Max DD contribution: sum of worst losing streaks
                neg = arr[arr < 0]
                stats.virtual_max_dd_contribution = float(np.sum(neg)) if len(neg) > 0 else 0.0

            summaries[name] = stats

        return summaries

    def format_summary(self) -> str:
        """Format filter summary as a text report."""
        summaries = self.get_filter_summary()
        if not summaries:
            return "=== NQDTC Shadow Trade Summary ===\n  No shadow trades simulated."

        lines = ["=== NQDTC Shadow Trade Summary ==="]
        header = (
            f"  {'Gate':24s} {'Rej':>5s} {'Filled':>6s} {'AvgR':>7s} "
            f"{'TP1%':>5s} {'TP2%':>5s} {'>1R%':>5s} "
            f"{'MissedEV':>9s} {'Avoided':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name in sorted(summaries, key=lambda n: -summaries[n].rejected_count):
            s = summaries[name]
            lines.append(
                f"  {name:24s} {s.rejected_count:5d} {s.filled_count:6d} "
                f"{s.avg_shadow_r:+7.3f} "
                f"{s.pct_reach_tp1:4.0f}% {s.pct_reach_tp2:4.0f}% "
                f"{s.pct_above_1r:4.0f}% "
                f"{s.net_missed_ev:+9.1f} {s.net_avoided_loss:8.1f}"
            )

        lines.append("")
        total_rej = sum(s.rejected_count for s in summaries.values())
        total_filled = sum(s.filled_count for s in summaries.values())
        lines.append(f"  Total rejections: {total_rej}  Filled in sim: {total_filled}")

        return "\n".join(lines)
