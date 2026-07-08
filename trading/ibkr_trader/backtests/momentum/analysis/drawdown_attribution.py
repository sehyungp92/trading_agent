"""Drawdown attribution — decompose drawdown episodes by cause.

Identifies the top drawdown episodes and attributes losses to
entry type, symbol, regime, direction, and strategy (portfolio-level).
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


def drawdown_attribution_report(
    trades: list,
    equity_curve: list | np.ndarray,
    timestamps: list | np.ndarray,
    strategy_labels: list[str] | None = None,
    top_n: int = 5,
) -> str:
    """Generate drawdown attribution report.

    Args:
        trades: List of trade records with r_multiple, entry_time, exit_time,
                pnl_dollars, direction, symbol, exit_reason fields.
                Optional: setup_class/entry_type, regime, session, strategy_label.
        equity_curve: Equity values over time.
        timestamps: Corresponding timestamps.
        strategy_labels: If provided, used for portfolio-level attribution.
        top_n: Number of top DD episodes to analyze.
    """
    lines = ["=" * 60]
    lines.append("  DRAWDOWN ATTRIBUTION REPORT")
    lines.append("=" * 60)
    lines.append("")

    if not trades or len(equity_curve) < 2:
        lines.append("  Insufficient data for drawdown analysis.")
        return "\n".join(lines)

    eq = np.array(equity_curve, dtype=float)

    # ── A. Find Top DD Episodes ──
    episodes = _find_dd_episodes(eq, timestamps, top_n)

    lines.append(f"  A. TOP {min(top_n, len(episodes))} DRAWDOWN EPISODES")
    lines.append("  " + "-" * 50)

    if not episodes:
        lines.append("    No drawdown episodes found.")
    else:
        header = f"    {'#':>2s} {'Start':>12s} {'End':>12s} {'Depth R':>8s} {'Depth $':>10s} {'Trades':>6s} {'Recovery':>8s}"
        lines.append(header)
        lines.append("    " + "-" * (len(header) - 4))

        for i, ep in enumerate(episodes, 1):
            # Find trades during this episode
            ep_trades = _trades_in_window(trades, ep["start_time"], ep["end_time"])
            start_str = ep["start_time"].strftime("%Y-%m-%d") if isinstance(ep["start_time"], datetime) else str(ep["start_time"])[:10]
            end_str = ep["end_time"].strftime("%Y-%m-%d") if isinstance(ep["end_time"], datetime) else str(ep["end_time"])[:10]

            r_depth = sum(r for r in (getattr(t, 'r_multiple', 0.0) for t in ep_trades) if r < 0)

            lines.append(
                f"    {i:2d} {start_str:>12s} {end_str:>12s} "
                f"{r_depth:+8.2f} {ep['depth_dollars']:+10.0f} "
                f"{len(ep_trades):6d} {ep.get('recovery_trades', 'N/A'):>8s}"
            )

    # ── B. Drawdown Decomposition ──
    lines.append("")
    lines.append("  B. DRAWDOWN DECOMPOSITION (losses during DD episodes)")
    lines.append("  " + "-" * 50)

    # Collect all trades during DD episodes
    dd_trades = []
    for ep in episodes:
        dd_trades.extend(_trades_in_window(trades, ep["start_time"], ep["end_time"]))
    # Deduplicate by entry_time
    seen = set()
    unique_dd = []
    for t in dd_trades:
        key = (getattr(t, 'entry_time', None), getattr(t, 'symbol', ''))
        if key not in seen:
            seen.add(key)
            unique_dd.append(t)
    dd_trades = unique_dd

    losers = [t for t in dd_trades if getattr(t, 'r_multiple', 0.0) <= 0]
    total_loss_r = sum(getattr(t, 'r_multiple', 0.0) for t in losers) if losers else 0.0

    if losers and total_loss_r < 0:
        # Single-pass attribute extraction for all decomposition axes
        type_loss = defaultdict(float)
        sym_loss = defaultdict(float)
        dir_loss = defaultdict(float)
        regime_loss = defaultdict(float)
        strat_loss = defaultdict(float)

        for t in losers:
            r = getattr(t, 'r_multiple', 0.0)

            etype = getattr(t, 'setup_class', None) or getattr(t, 'entry_type', None) or getattr(t, 'entry_mode', 'unknown')
            if hasattr(etype, 'name'):
                etype = etype.name
            type_loss[str(etype)] += r

            sym_loss[getattr(t, 'symbol', 'unknown')] += r

            d = getattr(t, 'direction', 0)
            dir_loss["LONG" if d == 1 else "SHORT" if d == -1 else str(d)] += r

            regime = getattr(t, 'regime', None) or getattr(t, 'regime_at_entry', None)
            if regime is not None:
                if hasattr(regime, 'name'):
                    regime = regime.name
                regime_loss[str(regime)] += r

            if strategy_labels:
                label = getattr(t, 'strategy_label', None) or getattr(t, 'strategy', 'unknown')
                strat_loss[str(label)] += r

        def _fmt_decomp(title, decomp):
            out = [f"\n    {title}:"]
            for k, loss_r in sorted(decomp.items(), key=lambda x: x[1]):
                pct = loss_r / total_loss_r * 100 if total_loss_r != 0 else 0
                out.append(f"      {str(k):20s}: {loss_r:+7.2f}R ({pct:5.1f}% of DD losses)")
            return out

        lines.append("")
        lines.extend(_fmt_decomp("By Entry Type / Setup Class", type_loss))
        lines.extend(_fmt_decomp("By Symbol", sym_loss))
        lines.extend(_fmt_decomp("By Direction", dir_loss))
        if regime_loss:
            lines.extend(_fmt_decomp("By Regime", regime_loss))
        if strategy_labels and strat_loss:
            lines.extend(_fmt_decomp("By Strategy", strat_loss))

        # Day-of-week decomposition
        dow_loss = defaultdict(float)
        _DOW_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        for t in losers:
            r = getattr(t, 'r_multiple', 0.0)
            et = getattr(t, 'entry_time', None)
            if et is not None and hasattr(et, 'weekday'):
                dow_loss[_DOW_NAMES.get(et.weekday(), str(et.weekday()))] += r

        if dow_loss:
            lines.extend(_fmt_decomp("By Day of Week", dow_loss))

        # Session decomposition (ET-based windows)
        session_loss = defaultdict(float)
        for t in losers:
            r = getattr(t, 'r_multiple', 0.0)
            et = getattr(t, 'entry_time', None)
            if et is not None and hasattr(et, 'hour'):
                session_loss[_classify_session(et)] += r

        if session_loss:
            lines.extend(_fmt_decomp("By Session Window", session_loss))

        # Time-of-day decomposition (2-hour buckets)
        tod_loss = defaultdict(float)
        for t in losers:
            r = getattr(t, 'r_multiple', 0.0)
            et = getattr(t, 'entry_time', None)
            if et is not None and hasattr(et, 'hour'):
                bucket = (et.hour // 2) * 2
                tod_loss[f"{bucket:02d}:00-{bucket+2:02d}:00"] += r

        if tod_loss:
            lines.extend(_fmt_decomp("By Time of Day (2h buckets)", tod_loss))

        # Verdict: systematic or diffuse
        all_decomps = list(type_loss.values()) + list(sym_loss.values()) + list(dir_loss.values())
        max_pct = max(abs(v / total_loss_r * 100) for v in all_decomps) if all_decomps and total_loss_r != 0 else 0
        if max_pct > 60:
            verdict = "SYSTEMATIC (one factor > 60% of DD losses)"
        elif max_pct > 30:
            verdict = "MODERATE (dominant factor at {:.0f}%)".format(max_pct)
        else:
            verdict = "DIFFUSE (no single factor > 30%)"
        lines.append("")
        lines.append(f"    Verdict: {verdict}")
    else:
        lines.append("    No losing trades during drawdown episodes.")

    # ── B2. DD CLUSTER DETECTION (3+ losses within 24hrs) ──
    lines.append("")
    lines.append("  B2. DD CLUSTER DETECTION (3+ losses within 24 hours)")
    lines.append("  " + "-" * 50)

    clusters = _detect_dd_clusters(trades)
    if clusters:
        lines.append(f"    Found {len(clusters)} loss clusters:")
        for ci, cluster in enumerate(clusters[:10], 1):
            total_r = sum(getattr(t, 'r_multiple', 0.0) for t in cluster)
            total_pnl = sum(getattr(t, 'pnl_dollars', 0.0) for t in cluster)
            first_time = getattr(cluster[0], 'entry_time', None)
            date_str = first_time.strftime("%Y-%m-%d") if first_time and hasattr(first_time, 'strftime') else "?"
            lines.append(f"    #{ci}: {date_str} — {len(cluster)} losses, {total_r:+.2f}R, ${total_pnl:+,.0f}")
    else:
        lines.append("    No loss clusters detected.")

    # ── C. Loss Clustering ──
    lines.append("")
    lines.append("  C. LOSS CLUSTERING")
    lines.append("  " + "-" * 50)

    r_series = np.array([getattr(t, 'r_multiple', 0.0) for t in trades])
    if len(r_series) >= 10:
        # Lag-1 autocorrelation
        r_mean = np.mean(r_series)
        r_var = np.var(r_series)
        if r_var > 0:
            autocorr = np.mean((r_series[:-1] - r_mean) * (r_series[1:] - r_mean)) / r_var
        else:
            autocorr = 0.0
        lines.append(f"    Lag-1 R autocorrelation:  {autocorr:+.4f}")

        # Consecutive loss runs
        loss_runs = []
        current_run = 0
        for r in r_series:
            if r <= 0:
                current_run += 1
            else:
                if current_run > 0:
                    loss_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            loss_runs.append(current_run)

        if loss_runs:
            lines.append(f"    Max consecutive losses:   {max(loss_runs)}")
            lines.append(f"    Avg loss run length:      {np.mean(loss_runs):.1f}")
            lines.append(f"    Loss runs > 3:            {sum(1 for r in loss_runs if r > 3)}")

        if autocorr > 0.15:
            cluster_verdict = "CLUSTERED (autocorr > 0.15 — losses tend to follow losses)"
        elif autocorr < 0.05:
            cluster_verdict = "INDEPENDENT (autocorr < 0.05 — losses are random)"
        else:
            cluster_verdict = "MILD CLUSTERING (autocorr 0.05-0.15)"
        lines.append(f"    Verdict: {cluster_verdict}")
    else:
        lines.append("    Insufficient trades for clustering analysis (need >= 10).")

    # ── D. Recovery Profile ──
    lines.append("")
    lines.append("  D. RECOVERY PROFILE")
    lines.append("  " + "-" * 50)

    if episodes:
        recovery_counts = []
        for ep in episodes:
            if ep.get("recovery_idx") is not None and ep.get("trough_idx") is not None:
                rc = ep["recovery_idx"] - ep["trough_idx"]
                recovery_counts.append(rc)

        if recovery_counts:
            lines.append(f"    Avg recovery bars:       {np.mean(recovery_counts):.0f}")
            lines.append(f"    Fastest recovery:        {min(recovery_counts)} bars")
            lines.append(f"    Slowest recovery:        {max(recovery_counts)} bars")
        else:
            lines.append("    Recovery data not available for episodes.")
    else:
        lines.append("    No episodes to analyze.")

    return "\n".join(lines)


def _find_dd_episodes(
    equity: np.ndarray,
    timestamps: list | np.ndarray,
    top_n: int = 5,
) -> list[dict]:
    """Find the top N drawdown episodes by depth."""
    peak = equity[0]
    peak_idx = 0
    episodes = []
    in_dd = False
    dd_start_idx = 0

    for i in range(1, len(equity)):
        if equity[i] > peak:
            if in_dd:
                # Record completed episode
                trough_idx = dd_start_idx + int(np.argmin(equity[dd_start_idx:i]))
                depth = equity[trough_idx] - peak
                episodes.append({
                    "start_idx": dd_start_idx,
                    "trough_idx": trough_idx,
                    "recovery_idx": i,
                    "end_idx": i,
                    "depth_dollars": depth,
                    "start_time": _safe_time(timestamps, dd_start_idx),
                    "end_time": _safe_time(timestamps, i),
                    "trough_time": _safe_time(timestamps, trough_idx),
                    "recovery_trades": str(i - trough_idx),
                })
                in_dd = False
            peak = equity[i]
            peak_idx = i
        elif equity[i] < peak:
            if not in_dd:
                dd_start_idx = peak_idx
                in_dd = True

    # Handle ongoing DD at end of data
    if in_dd:
        trough_idx = dd_start_idx + int(np.argmin(equity[dd_start_idx:]))
        depth = equity[trough_idx] - peak
        episodes.append({
            "start_idx": dd_start_idx,
            "trough_idx": trough_idx,
            "recovery_idx": None,
            "end_idx": len(equity) - 1,
            "depth_dollars": depth,
            "start_time": _safe_time(timestamps, dd_start_idx),
            "end_time": _safe_time(timestamps, len(equity) - 1),
            "trough_time": _safe_time(timestamps, trough_idx),
            "recovery_trades": "ongoing",
        })

    # Sort by depth (most negative first) and take top N
    episodes.sort(key=lambda e: e["depth_dollars"])
    return episodes[:top_n]


def _trades_in_window(trades: list, start_time, end_time) -> list:
    """Return trades whose exit_time falls within the window."""
    result = []
    for t in trades:
        exit_t = getattr(t, 'exit_time', None)
        if exit_t is None:
            continue
        try:
            e = _strip_tz(exit_t)
            s = _strip_tz(start_time)
            n = _strip_tz(end_time)
            if s <= e <= n:
                result.append(t)
        except (TypeError, ValueError):
            try:
                exit_ns = np.datetime64(exit_t, 'ns') if not isinstance(exit_t, np.datetime64) else exit_t
                start_ns = np.datetime64(start_time, 'ns') if not isinstance(start_time, np.datetime64) else start_time
                end_ns = np.datetime64(end_time, 'ns') if not isinstance(end_time, np.datetime64) else end_time
                if start_ns <= exit_ns <= end_ns:
                    result.append(t)
            except (TypeError, ValueError):
                pass
    return result


def _strip_tz(dt):
    """Convert any datetime-like to naive datetime for safe comparison."""
    if isinstance(dt, datetime):
        return dt.replace(tzinfo=None)
    if hasattr(dt, 'to_pydatetime'):
        return dt.to_pydatetime().replace(tzinfo=None)
    if isinstance(dt, np.datetime64):
        import pandas as pd
        return pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None)
    return dt


def _safe_time(timestamps, idx):
    """Safely extract a timestamp."""
    if idx < 0 or idx >= len(timestamps):
        return None
    ts = timestamps[idx]
    if isinstance(ts, datetime):
        return ts
    if hasattr(ts, 'astype'):
        try:
            import pandas as pd
            return pd.Timestamp(ts).to_pydatetime()
        except Exception:
            return ts
    return ts


def _classify_session(entry_time) -> str:
    """Classify entry time into session window (approximate ET from UTC).

    Assumes UTC input, subtracts 5 hours for ET (approximate).
    """
    hour = entry_time.hour
    # Approximate ET = UTC - 5
    et_hour = (hour - 5) % 24

    if 18 <= et_hour or et_hour < 2:
        return "ETH-Asia"
    elif 2 <= et_hour < 8:
        return "ETH-Europe"
    elif 8 <= et_hour < 10:
        return "RTH-Open"
    elif 10 <= et_hour < 14:
        return "RTH-Core"
    elif 14 <= et_hour < 16:
        return "RTH-Close"
    else:
        return "Evening"


def _detect_dd_clusters(
    trades: list,
    window_hours: int = 24,
    min_losses: int = 3,
) -> list[list]:
    """Detect clusters of 3+ losses within a time window.

    Returns list of clusters, each a list of losing trade records.
    """
    from datetime import timedelta

    losers = []
    for t in trades:
        if getattr(t, 'r_multiple', 0.0) <= 0:
            et = getattr(t, 'exit_time', None) or getattr(t, 'entry_time', None)
            if et is not None:
                losers.append((et, t))

    if len(losers) < min_losses:
        return []

    losers.sort(key=lambda x: x[0])
    clusters: list[list] = []
    window = timedelta(hours=window_hours)

    i = 0
    while i < len(losers):
        cluster = [losers[i]]
        j = i + 1
        while j < len(losers):
            try:
                t0 = _strip_tz(cluster[0][0])
                tj = _strip_tz(losers[j][0])
                if tj - t0 <= window:
                    cluster.append(losers[j])
                    j += 1
                else:
                    break
            except (TypeError, ValueError):
                j += 1
                continue

        if len(cluster) >= min_losses:
            clusters.append([item[1] for item in cluster])

        i = j if j > i + 1 else i + 1

    return clusters
