"""Weekly evidence-focus rotation for the monthly learning loop."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

WEEKLY_FOCUS_ROTATION_ANCHOR = date(2026, 6, 1)


class WeeklyFocusSlice(BaseModel):
    """A non-authoritative weekly evidence focus used to steer monthly search briefs."""

    cycle_week: int
    focus_id: str
    label: str
    repo_ids: list[str]
    bot_ids: list[str] = Field(default_factory=list)
    portfolio_families: list[str] = Field(default_factory=list)
    strategy_ids: list[str] = Field(default_factory=list)
    asset_scope: list[str] = Field(default_factory=list)
    monthly_handoff: str
    authority: str = "search_order_only"
    notes: list[str] = Field(default_factory=list)


_ROTATION = (
    WeeklyFocusSlice(
        cycle_week=1,
        focus_id="k_stock_and_trading_stock",
        label="k_stock_trader plus trading stock portfolio",
        repo_ids=["k_stock_trader", "trading"],
        bot_ids=["k_stock_trader", "trading"],
        portfolio_families=["k_stock_olr_kalcb", "trading_stock"],
        strategy_ids=["KALCB", "OLR", "IARIC_v1", "ALCB_v1"],
        asset_scope=["krx_equity", "us_equity"],
        monthly_handoff=(
            "Prioritize OLR/KALCB and trading stock formal bridge fixtures, adapters, "
            "and replay-evaluator coverage."
        ),
        notes=[
            "PCIM is research-only and is not part of the deployed k_stock_trader portfolio.",
            "Trading stock parity surfaces currently cover IARIC_v1 and ALCB_v1.",
        ],
    ),
    WeeklyFocusSlice(
        cycle_week=2,
        focus_id="trading_momentum",
        label="trading momentum portfolio",
        repo_ids=["trading"],
        bot_ids=["trading"],
        portfolio_families=["trading_momentum"],
        strategy_ids=["NQDTC_v2.1", "NQ_REGIME", "VdubusNQ_v4", "DownturnDominator_v1"],
        asset_scope=["futures"],
        monthly_handoff=(
            "Feed momentum and futures replay gaps into monthly phased-auto and OOS repair."
        ),
        notes=["This is the futures-trading family inside the broader trading repo."],
    ),
    WeeklyFocusSlice(
        cycle_week=3,
        focus_id="trading_swing",
        label="trading swing portfolio",
        repo_ids=["trading"],
        bot_ids=["trading"],
        portfolio_families=["trading_swing"],
        strategy_ids=["ATRSS", "AKC_HELIX", "TPC", "OVERLAY"],
        asset_scope=["equity_etf", "swing"],
        monthly_handoff=(
            "Feed swing strategy and overlay-risk candidates into monthly replay."
        ),
        notes=["Swing parity surfaces cover strategy children plus the portfolio overlay."],
    ),
    WeeklyFocusSlice(
        cycle_week=4,
        focus_id="crypto_trader",
        label="crypto_trader",
        repo_ids=["crypto_trader"],
        bot_ids=["crypto_trader"],
        portfolio_families=["crypto_portfolio"],
        strategy_ids=["crypto_trend_v1", "crypto_momentum_v1", "crypto_breakout_v1"],
        asset_scope=["crypto_perpetual"],
        monthly_handoff=(
            "Expand crypto parity fixtures and data coverage while the plugins remain "
            "shadow_validated."
        ),
        notes=[
            "BTC/1m and full-crypto bars are authoritative; plugin maturity is still shadow_validated.",
        ],
    ),
)


def weekly_focus_rotation() -> list[WeeklyFocusSlice]:
    """Return the canonical four-week focus rotation."""

    return list(_ROTATION)


def weekly_focus_for_week(week_start: str | date | datetime) -> WeeklyFocusSlice:
    """Return the rotation slice for a week-start date.

    The anchor is 2026-06-01, the first Monday after the current validation refresh.
    Week 1 is k_stock_trader plus trading stock, then momentum, swing, and crypto.
    """

    value = _as_date(week_start)
    weeks_since_anchor = (value - WEEKLY_FOCUS_ROTATION_ANCHOR).days // 7
    return _ROTATION[weeks_since_anchor % len(_ROTATION)]


def weekly_focus_payload(week_start: str | date | datetime) -> dict[str, Any]:
    """Return a JSON-ready weekly focus payload."""

    value = _as_date(week_start)
    payload = weekly_focus_for_week(value).model_dump(mode="json")
    payload["week_start"] = value.isoformat()
    return payload


def weekly_focus_rotation_payload() -> list[dict[str, Any]]:
    """Return the full rotation as JSON-ready dictionaries."""

    return [item.model_dump(mode="json") for item in _ROTATION]


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()
