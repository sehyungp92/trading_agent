"""Shadow trade simulation for VdubusNQ v4.0 rejected candidates.

Tracks what would have happened if a gate had NOT rejected an entry signal.
Uses VdubusNQ exit lifecycle: protective stop + intraday trail + stale exit,
+1R partial -> BE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from strategies.momentum.vdub import config as C
from strategies.momentum.vdub.models import Direction

logger = logging.getLogger(__name__)


@dataclass
class VdubusShadowCandidate:
    """A candidate that was rejected by one or more gates."""

    direction: int
    filter_name: str
    time: datetime
    entry_price: float
    stop_price: float
    session: str = ""
    sub_window: str = ""
    entry_type: str = ""
    daily_trend: int = 0
    vol_state: str = ""


@dataclass
class VdubusShadowResult:
    """Outcome of simulating a shadow candidate."""

    candidate: VdubusShadowCandidate
    filled: bool = False
    fill_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    reached_1r: bool = False


@dataclass
class VdubusFilterStats:
    """Aggregated stats for one gate."""

    filter_name: str
    rejected_count: int = 0
    simulated_count: int = 0
    filled_count: int = 0
    avg_shadow_r: float = 0.0
    pct_reach_1r: float = 0.0
    pct_above_1r: float = 0.0
    net_missed_ev: float = 0.0
    net_avoided_loss: float = 0.0


class VdubusShadowTracker:
    """Track rejected VdubusNQ candidates and simulate their outcomes."""

    def __init__(self):
        self.rejections: list[VdubusShadowCandidate] = []
        self.results: list[VdubusShadowResult] = []

    def record_rejection(
        self,
        direction: int,
        filter_name: str,
        time: datetime,
        entry_price: float,
        stop_price: float,
        session: str = "",
        sub_window: str = "",
        entry_type: str = "",
        daily_trend: int = 0,
        vol_state: str = "",
    ) -> None:
        self.rejections.append(VdubusShadowCandidate(
            direction=direction, filter_name=filter_name,
            time=time, entry_price=entry_price, stop_price=stop_price,
            session=session, sub_window=sub_window, entry_type=entry_type,
            daily_trend=daily_trend, vol_state=vol_state,
        ))

    def simulate_shadows(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
    ) -> list[VdubusShadowResult]:
        """Forward-simulate each rejected candidate using VdubusNQ exit lifecycle."""
        self.results.clear()

        for cand in self.rejections:
            result = self._simulate_one(cand, opens, highs, lows, closes, times)
            self.results.append(result)

        return self.results

    def _simulate_one(
        self,
        cand: VdubusShadowCandidate,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        times: np.ndarray,
    ) -> VdubusShadowResult:
        """Simulate one shadow candidate with simplified VdubusNQ lifecycle.

        Lifecycle:
        1. Stop-limit fill within TTL_BARS (3 x 15m)
        2. Protective stop
        3. +1R -> move stop to BE
        4. Stale exit after STALE_BARS_15M bars
        """
        cand_ts = (
            np.datetime64(cand.time, "ns")
            if not isinstance(cand.time, np.datetime64)
            else cand.time
        )
        start_idx = int(np.searchsorted(times, cand_ts, side="right"))

        if start_idx >= len(times):
            return VdubusShadowResult(candidate=cand)

        r_base = abs(cand.entry_price - cand.stop_price)
        if r_base <= 0:
            return VdubusShadowResult(candidate=cand)

        ttl = C.TTL_BARS

        # Phase 1: check fill within TTL
        fill_price = 0.0
        fill_idx = -1

        for i in range(start_idx, min(start_idx + ttl, len(opens))):
            if cand.direction == Direction.LONG:
                if highs[i] >= cand.entry_price:
                    fill_price = max(float(opens[i]), cand.entry_price)
                    fill_idx = i
                    break
            else:
                if lows[i] <= cand.entry_price:
                    fill_price = min(float(opens[i]), cand.entry_price)
                    fill_idx = i
                    break

        if fill_idx < 0:
            return VdubusShadowResult(candidate=cand, filled=False)

        # Phase 2: exit lifecycle
        current_stop = cand.stop_price
        mfe_price = fill_price
        mae_price = fill_price
        mfe_r = 0.0
        mae_r = 0.0
        reached_1r = False
        be_triggered = False

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
                    return VdubusShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held, reached_1r=reached_1r,
                    )
            else:
                if highs[j] >= current_stop:
                    pnl_r = (fill_price - current_stop) / r_base
                    return VdubusShadowResult(
                        candidate=cand, filled=True, fill_price=fill_price,
                        exit_price=current_stop, exit_reason="STOP",
                        r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                        bars_held=bars_held, reached_1r=reached_1r,
                    )

            # +1R -> BE
            if not reached_1r and cur_mfe >= 1.0:
                reached_1r = True
                if not be_triggered:
                    be_triggered = True
                    current_stop = fill_price

            # Stale exit
            if bars_held >= C.STALE_BARS_15M and cur_r < C.STALE_R:
                exit_px = float(closes[j])
                if cand.direction == Direction.LONG:
                    pnl_r = (exit_px - fill_price) / r_base
                else:
                    pnl_r = (fill_price - exit_px) / r_base
                return VdubusShadowResult(
                    candidate=cand, filled=True, fill_price=fill_price,
                    exit_price=exit_px, exit_reason="STALE",
                    r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
                    bars_held=bars_held, reached_1r=reached_1r,
                )

        # End of data
        last_close = float(closes[-1])
        if cand.direction == Direction.LONG:
            pnl_r = (last_close - fill_price) / r_base
        else:
            pnl_r = (fill_price - last_close) / r_base

        return VdubusShadowResult(
            candidate=cand, filled=True, fill_price=fill_price,
            exit_price=last_close, exit_reason="END_OF_DATA",
            r_multiple=pnl_r, mfe_r=mfe_r, mae_r=mae_r,
            bars_held=len(closes) - fill_idx, reached_1r=reached_1r,
        )

    def get_filter_summary(self) -> dict[str, VdubusFilterStats]:
        """Compute per-gate stats from simulation results."""
        by_filter: dict[str, list[VdubusShadowResult]] = {}
        for r in self.results:
            name = r.candidate.filter_name
            by_filter.setdefault(name, []).append(r)

        rej_counts: dict[str, int] = {}
        for c in self.rejections:
            rej_counts[c.filter_name] = rej_counts.get(c.filter_name, 0) + 1

        summaries: dict[str, VdubusFilterStats] = {}
        for name, results in by_filter.items():
            filled = [r for r in results if r.filled]
            r_multiples = [r.r_multiple for r in filled]

            stats = VdubusFilterStats(filter_name=name)
            stats.rejected_count = rej_counts.get(name, len(results))
            stats.simulated_count = len(results)
            stats.filled_count = len(filled)

            if filled:
                arr = np.array(r_multiples)
                stats.avg_shadow_r = float(np.mean(arr))
                stats.pct_above_1r = float(np.mean(arr > 1.0)) * 100
                stats.pct_reach_1r = float(np.mean([r.reached_1r for r in filled])) * 100
                stats.net_missed_ev = float(np.sum(arr[arr > 0]))
                stats.net_avoided_loss = float(np.sum(np.abs(arr[arr < 0])))

            summaries[name] = stats

        return summaries

    def format_summary(self) -> str:
        """Format filter summary as a text report."""
        summaries = self.get_filter_summary()
        if not summaries:
            return "=== VdubusNQ Shadow Trade Summary ===\n  No shadow trades simulated."

        lines = ["=== VdubusNQ Shadow Trade Summary ==="]
        header = (
            f"  {'Gate':24s} {'Rej':>5s} {'Filled':>6s} {'AvgR':>7s} "
            f"{'1R%':>5s} {'>1R%':>5s} "
            f"{'MissedEV':>9s} {'Avoided':>8s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for name in sorted(summaries, key=lambda n: -summaries[n].rejected_count):
            s = summaries[name]
            lines.append(
                f"  {name:24s} {s.rejected_count:5d} {s.filled_count:6d} "
                f"{s.avg_shadow_r:+7.3f} "
                f"{s.pct_reach_1r:4.0f}% "
                f"{s.pct_above_1r:4.0f}% "
                f"{s.net_missed_ev:+9.1f} {s.net_avoided_loss:8.1f}"
            )

        lines.append("")
        total_rej = sum(s.rejected_count for s in summaries.values())
        total_filled = sum(s.filled_count for s in summaries.values())
        lines.append(f"  Total rejections: {total_rej}  Filled in sim: {total_filled}")

        return "\n".join(lines)
