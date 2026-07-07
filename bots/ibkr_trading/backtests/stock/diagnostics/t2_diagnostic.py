"""Diagnose why IARIC T2 produces only 1 trade vs T1's 337.

Traces through the T2 FSM funnel for sample dates to find where entries are blocked.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, time
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from backtests.stock.config_iaric import IARICBacktestConfig
from backtests.stock.engine.research_replay import ResearchReplayEngine
from strategies.stock.iaric.config import StrategySettings

logging.basicConfig(level=logging.WARNING)
ET = ZoneInfo("America/New_York")
_MKT_OPEN = time(9, 30)
_MKT_CLOSE = time(16, 0)


def main():
    cfg = IARICBacktestConfig()
    settings = StrategySettings()
    if cfg.param_overrides:
        settings = replace(settings, **cfg.param_overrides)

    print(f"Config: start={cfg.start_date}, end={cfg.end_date}, tier={cfg.tier}")
    print(f"Settings: panic_drop={settings.panic_flush_drop_pct}, drift_drop={settings.drift_exhaustion_drop_pct}")
    print(f"  panic_minutes={settings.panic_flush_minutes}, drift_minutes={settings.drift_exhaustion_minutes}")
    print(f"  avwap_band_pct={settings.avwap_band_pct}")
    print()

    # Load data
    replay = ResearchReplayEngine(
        data_dir=cfg.data_dir,
        universe_config=cfg.universe,
    )
    replay.load_all_data()

    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    trading_dates = replay.tradable_dates(start, end)
    print(f"Trading dates: {len(trading_dates)} ({trading_dates[0]} to {trading_dates[-1]})")

    # Aggregate funnel stats
    total_dates = 0
    regime_c_dates = 0
    total_tradable = 0
    total_have_5m = 0
    total_have_rth_bars = 0
    total_bars_checked = 0
    total_in_avwap_band = 0
    total_panic_setups = 0
    total_drift_setups = 0
    total_setups = 0

    # Per-date detail for first 10 dates with tradable items
    detail_count = 0
    max_detail = 5

    # Sample every Nth date to keep output manageable
    sample_every = max(1, len(trading_dates) // 50)

    for day_idx, trade_date in enumerate(trading_dates):
        if day_idx == 0:
            continue
        prev_date = trading_dates[day_idx - 1]
        artifact = replay.iaric_selection_for_date(prev_date, settings)
        total_dates += 1

        if artifact.regime.tier == "C":
            regime_c_dates += 1
            continue

        tradable_map = {item.symbol: item for item in artifact.tradable}
        n_tradable = len(tradable_map)
        total_tradable += n_tradable

        if n_tradable == 0:
            continue

        show_detail = detail_count < max_detail or day_idx % sample_every == 0
        if show_detail and detail_count < max_detail:
            detail_count += 1

        day_have_5m = 0
        day_have_rth = 0
        day_bars_total = 0
        day_in_band = 0
        day_panic = 0
        day_drift = 0
        avwap_band_widths = []
        drop_from_hods = []

        for sym, item in tradable_map.items():
            bars = replay.get_5m_bar_objects_for_date(sym, trade_date)
            if not bars:
                continue
            day_have_5m += 1

            # Filter to RTH
            rth_bars = [b for b in bars if _MKT_OPEN <= b.start_time.astimezone(ET).time() < _MKT_CLOSE]
            if not rth_bars:
                continue
            day_have_rth += 1
            day_bars_total += len(rth_bars)

            avwap_lo = item.avwap_band_lower
            avwap_hi = item.avwap_band_upper
            avwap_ref = item.avwap_ref
            band_width = (avwap_hi - avwap_lo) / max(avwap_ref, 1e-9) if avwap_ref else 0
            avwap_band_widths.append(band_width)

            # Simulate session tracking
            session_high = 0.0
            hod_bar_idx = 0

            for bar_idx, bar in enumerate(rth_bars):
                if bar.high > session_high:
                    session_high = bar.high
                    hod_bar_idx = bar_idx

                # Check AVWAP band
                in_band = (avwap_lo <= bar.low <= avwap_hi) or (avwap_lo <= bar.close <= avwap_hi)
                if in_band:
                    day_in_band += 1

                    # Check setup conditions
                    if session_high > 0:
                        drop = (session_high - bar.close) / session_high
                        minutes_since_hod = (bar_idx - hod_bar_idx) * 5
                        drop_from_hods.append(drop)

                        if drop >= settings.panic_flush_drop_pct and minutes_since_hod <= settings.panic_flush_minutes:
                            day_panic += 1
                        if drop >= settings.drift_exhaustion_drop_pct and minutes_since_hod >= settings.drift_exhaustion_minutes:
                            day_drift += 1

        total_have_5m += day_have_5m
        total_have_rth_bars += day_have_rth
        total_bars_checked += day_bars_total
        total_in_avwap_band += day_in_band
        total_panic_setups += day_panic
        total_drift_setups += day_drift
        total_setups += day_panic + day_drift

        if show_detail and detail_count <= max_detail:
            avg_bw = mean(avwap_band_widths) if avwap_band_widths else 0
            max_drop = max(drop_from_hods) if drop_from_hods else 0
            print(f"[{trade_date}] regime={artifact.regime.tier} tradable={n_tradable} "
                  f"have_5m={day_have_5m} rth={day_have_rth} bars={day_bars_total} "
                  f"in_band={day_in_band} panic={day_panic} drift={day_drift} "
                  f"avg_band_width={avg_bw:.4f} max_drop={max_drop:.4f}")

    print(f"\n{'='*70}")
    print(f"FUNNEL SUMMARY ({total_dates} trading dates)")
    print(f"{'='*70}")
    print(f"  Regime C dates (skipped):     {regime_c_dates} ({100*regime_c_dates/max(total_dates,1):.1f}%)")
    print(f"  Active dates:                 {total_dates - regime_c_dates}")
    print(f"  Total tradable sym-days:      {total_tradable}")
    print(f"  Sym-days with 5m data:        {total_have_5m} ({100*total_have_5m/max(total_tradable,1):.1f}%)")
    print(f"  Sym-days with RTH bars:       {total_have_rth_bars} ({100*total_have_rth_bars/max(total_have_5m,1):.1f}%)")
    print(f"  Total RTH bars checked:       {total_bars_checked}")
    print(f"  Bars in AVWAP band:           {total_in_avwap_band} ({100*total_in_avwap_band/max(total_bars_checked,1):.1f}%)")
    print(f"  PANIC_FLUSH setups:           {total_panic_setups}")
    print(f"  DRIFT_EXHAUSTION setups:      {total_drift_setups}")
    print(f"  Total setups detected:        {total_setups}")
    print()

    if total_setups == 0:
        print("*** ZERO SETUPS DETECTED - investigating why ***")
        # Dig into a single date with most tradable to understand
        best_date = None
        best_count = 0
        for i, td in enumerate(trading_dates[:100]):
            if i == 0:
                continue
            prev_td = trading_dates[i - 1]
            art = replay.iaric_selection_for_date(prev_td, settings)
            if art.regime.tier != "C" and len(art.tradable) > best_count:
                best_count = len(art.tradable)
                best_date = td
                best_prev = prev_td
        if best_date:
            print(f"\nDeep dive: {best_date} ({best_count} tradable)")
            art = replay.iaric_selection_for_date(best_prev, settings)
            for item in art.tradable[:3]:
                sym = item.symbol
                print(f"\n  Symbol: {sym}")
                print(f"    avwap_ref={item.avwap_ref:.2f} band_lo={item.avwap_band_lower:.2f} band_hi={item.avwap_band_upper:.2f}")
                print(f"    intraday_atr_seed={item.intraday_atr_seed:.6f}")
                bars = replay.get_5m_bar_objects_for_date(sym, best_date)
                if not bars:
                    print(f"    NO 5m bars!")
                    continue
                rth = [b for b in bars if _MKT_OPEN <= b.start_time.astimezone(ET).time() < _MKT_CLOSE]
                print(f"    Total 5m bars: {len(bars)}, RTH bars: {len(rth)}")
                if rth:
                    prices = [b.close for b in rth]
                    print(f"    Price range: {min(prices):.2f} - {max(prices):.2f}")
                    print(f"    AVWAP band: {item.avwap_band_lower:.2f} - {item.avwap_band_upper:.2f}")
                    in_band = sum(1 for b in rth if item.avwap_band_lower <= b.close <= item.avwap_band_upper
                                  or item.avwap_band_lower <= b.low <= item.avwap_band_upper)
                    print(f"    Bars in band: {in_band}/{len(rth)}")

                    # Show max drops
                    sh = 0.0
                    hod_i = 0
                    max_drop_val = 0.0
                    for i, b in enumerate(rth):
                        if b.high > sh:
                            sh = b.high
                            hod_i = i
                        if sh > 0:
                            d = (sh - b.close) / sh
                            if d > max_drop_val:
                                max_drop_val = d
                    print(f"    Max drop from HOD: {max_drop_val:.4f} (need panic>={settings.panic_flush_drop_pct}, drift>={settings.drift_exhaustion_drop_pct})")

    elif total_in_avwap_band == 0:
        print("*** ZERO BARS IN AVWAP BAND - AVWAP band issue ***")


if __name__ == "__main__":
    main()
