from __future__ import annotations

REQUIRED_BACKTEST_INVARIANTS = (
    "completed_bar_policy",
    "next_bar_fill",
    "broker_path",
    "mtm_risk",
    "net_gross_accounting",
    "shared_capital_portfolio",
    "diagnostics",
    "timestamp_hygiene",
    "artifact_hygiene",
    "stress_gates",
)

ACCEPTED_NON_PROMOTION_STATUSES = frozenset(
    {
        "blocked_non_promotion",
        "non_promotion",
        "research_only",
        "superseded_by_portfolio_bundle",
    }
)
