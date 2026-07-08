"""Winner vs loser pre-trade signature analysis for ALCB P8 diagnostics.

Builds a logistic model on pre-trade features (RVOL, score, OR width,
AVWAP distance, sector, entry time) to predict win/loss. If AUC > 0.65,
there's untapped discrimination potential.

Usage:
    from backtests.stock.analysis.alcb_winner_prediction import (
        winner_prediction_analysis,
    )
    report = winner_prediction_analysis(trades)
    print(report)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from backtests.stock.models import TradeRecord


def _meta(t: TradeRecord, key: str, default=None):
    """Safe metadata access."""
    if t.metadata:
        return t.metadata.get(key, default)
    return default


def _entry_bar_number(t: TradeRecord) -> int:
    """Compute 5-min bar number from market open (9:30 ET = bar 1)."""
    try:
        from zoneinfo import ZoneInfo
        et = t.entry_time.astimezone(ZoneInfo("America/New_York"))
        minutes = (et.hour - 9) * 60 + et.minute - 30
        return max(1, minutes // 5 + 1)
    except Exception:
        return 0


def _or_width_pct(t: TradeRecord) -> float:
    """Compute OR width as % of price."""
    oh = _meta(t, "or_high", 0)
    ol = _meta(t, "or_low", 0)
    if oh > 0 and ol > 0:
        return (oh - ol) / oh
    return 0.0


def _avwap_dist_pct(t: TradeRecord) -> float:
    """Compute AVWAP distance as % of AVWAP."""
    avwap = _meta(t, "avwap_at_entry", 0)
    if avwap > 0:
        return (t.entry_price - avwap) / avwap
    return 0.0


def _breakout_dist_r(t: TradeRecord) -> float:
    """Compute breakout distance from level in R."""
    bl = _meta(t, "breakout_level", 0)
    if bl > 0 and t.risk_per_share > 0:
        return (t.entry_price - bl) / t.risk_per_share
    return 0.0


def winner_prediction_analysis(trades: list[TradeRecord]) -> str:
    """Analyze pre-trade feature signatures for winners vs losers.

    Returns a comprehensive report with:
    - Feature distributions for winners vs losers
    - Univariate discrimination power (AUC estimate)
    - Feature importance ranking
    - Logistic model if sklearn is available
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("WINNER vs LOSER PRE-TRADE SIGNATURE ANALYSIS")
    lines.append("=" * 70)

    winners = [t for t in trades if t.r_multiple > 0]
    losers = [t for t in trades if t.r_multiple <= 0]
    lines.append(f"\nWinners: {len(winners)} ({len(winners)/len(trades):.1%})")
    lines.append(f"Losers:  {len(losers)} ({len(losers)/len(trades):.1%})")

    # --- Feature extraction ---
    def _extract_features(t: TradeRecord) -> dict:
        return {
            "rvol": _meta(t, "rvol_at_entry", 0),
            "momentum_score": _meta(t, "momentum_score", 0),
            "or_width_pct": _or_width_pct(t),
            "avwap_dist_pct": _avwap_dist_pct(t),
            "breakout_dist_r": _breakout_dist_r(t),
            "entry_bar": _entry_bar_number(t),
            "hold_bars": t.hold_bars,
            "mfe_r": _meta(t, "mfe_r", 0),
        }

    # --- Univariate analysis ---
    lines.append("\n--- Univariate Feature Discrimination ---")
    lines.append(f"  {'Feature':<20s} {'W_mean':>8s} {'L_mean':>8s} {'Delta':>8s} {'AUC_est':>8s}")
    lines.append(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    features_to_analyze = ["rvol", "momentum_score", "or_width_pct",
                           "avwap_dist_pct", "breakout_dist_r", "entry_bar"]

    # Cache feature extraction (avoid redundant per-feature recomputation)
    w_features = [_extract_features(t) for t in winners]
    l_features = [_extract_features(t) for t in losers]

    feature_aucs: list[tuple[str, float]] = []

    for feat_name in features_to_analyze:
        w_vals = [f[feat_name] for f in w_features if f[feat_name] is not None]
        l_vals = [f[feat_name] for f in l_features if f[feat_name] is not None]

        if not w_vals or not l_vals:
            continue

        w_mean = float(np.mean(w_vals))
        l_mean = float(np.mean(l_vals))
        delta = w_mean - l_mean

        # Simple AUC estimate via Mann-Whitney U statistic
        try:
            all_vals = [(v, 1) for v in w_vals] + [(v, 0) for v in l_vals]
            all_vals.sort(key=lambda x: x[0])
            n_w, n_l = len(w_vals), len(l_vals)
            concordant = 0
            w_rank_sum = 0
            for rank, (val, label) in enumerate(all_vals, 1):
                if label == 1:
                    w_rank_sum += rank
            u = w_rank_sum - n_w * (n_w + 1) / 2
            auc = u / (n_w * n_l) if n_w * n_l > 0 else 0.5
            # Ensure AUC > 0.5 (flip if needed for interpretation)
            auc_adj = max(auc, 1 - auc)
        except Exception:
            auc_adj = 0.5

        feature_aucs.append((feat_name, auc_adj))
        lines.append(
            f"  {feat_name:<20s} {w_mean:>8.3f} {l_mean:>8.3f} "
            f"{delta:>+7.3f} {auc_adj:>7.3f}"
        )

    # --- Feature importance ranking ---
    lines.append("\n--- Feature Importance Ranking (by AUC) ---")
    feature_aucs.sort(key=lambda x: x[1], reverse=True)
    for rank, (name, auc) in enumerate(feature_aucs, 1):
        bar = "#" * int(auc * 20)
        discriminative = "***" if auc >= 0.60 else "**" if auc >= 0.55 else ""
        lines.append(f"  {rank}. {name:<20s} AUC={auc:.3f} {bar} {discriminative}")

    # --- Sector breakdown ---
    lines.append("\n--- Sector Win Rate Comparison ---")
    sector_wr: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for t in trades:
        sector = t.sector or "Unknown"
        w, total = sector_wr[sector]
        sector_wr[sector] = (w + (1 if t.r_multiple > 0 else 0), total + 1)

    for sector in sorted(sector_wr, key=lambda s: sector_wr[s][1], reverse=True):
        w, total = sector_wr[sector]
        wr = w / total if total > 0 else 0
        lines.append(f"  {sector:<30s} WR={wr:.1%} ({w}/{total})")

    # --- Entry type win rate ---
    lines.append("\n--- Entry Type Win Rate ---")
    etype_wr: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for t in trades:
        etype = _meta(t, "entry_type", t.entry_type or "UNKNOWN")
        w, total = etype_wr[etype]
        etype_wr[etype] = (w + (1 if t.r_multiple > 0 else 0), total + 1)

    for etype in sorted(etype_wr, key=lambda e: etype_wr[e][1], reverse=True):
        w, total = etype_wr[etype]
        wr = w / total if total > 0 else 0
        lines.append(f"  {etype:<25s} WR={wr:.1%} ({w}/{total})")

    # --- Logistic regression (if sklearn available) ---
    lines.append("\n--- Logistic Regression Model ---")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        # Build feature matrix
        X_rows = []
        y_rows = []
        for t in trades:
            f = _extract_features(t)
            row = [f["rvol"], f["momentum_score"], f["or_width_pct"],
                   f["avwap_dist_pct"], f["breakout_dist_r"], f["entry_bar"]]
            if any(v is None for v in row):
                continue
            X_rows.append(row)
            y_rows.append(1 if t.r_multiple > 0 else 0)

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")

        lines.append(f"  5-fold CV AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

        # Fit full model for coefficients
        model.fit(X_scaled, y)
        feat_names = ["rvol", "momentum_score", "or_width_pct",
                      "avwap_dist_pct", "breakout_dist_r", "entry_bar"]
        lines.append(f"\n  Coefficients (scaled):")
        for name, coef in sorted(zip(feat_names, model.coef_[0]),
                                  key=lambda x: abs(x[1]), reverse=True):
            direction = "+" if coef > 0 else "-"
            lines.append(f"    {name:<20s} {coef:>+7.3f} ({direction} predicts win)")

        if scores.mean() >= 0.65:
            lines.append(f"\n  >>> AUC >= 0.65: Untapped discrimination exists!")
            lines.append(f"  >>> Consider feature-based entry gating or sizing.")
        else:
            lines.append(f"\n  >>> AUC < 0.65: Pre-trade features have limited "
                        f"predictive power.")
            lines.append(f"  >>> Focus on exit optimization rather than entry "
                        f"discrimination.")

    except ImportError:
        lines.append("  sklearn not available - install for logistic regression analysis")
        lines.append("  pip install scikit-learn")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
