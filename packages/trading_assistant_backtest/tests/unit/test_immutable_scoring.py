from __future__ import annotations

from datetime import date

import pytest

from trading_assistant_backtest.replay.types import WindowSpec
from trading_assistant_backtest.scoring.immutable import (
    compact_score_payload,
    resolve_score_profile,
    score_replay,
)
from trading_assistant_backtest.scoring.objective import capped_components


def test_crypto_plugin_resolves_strategy_overlay_components() -> None:
    profile = resolve_score_profile(
        family="crypto_portfolio",
        plugin_id="crypto-momentum-v1",
        strategy_id="btc_1m",
    )

    assert profile.profile_id == "crypto.single.momentum"
    assert capped_components(
        7,
        family="crypto_portfolio",
        plugin_id="crypto-momentum-v1",
        strategy_id="btc_1m",
    ) == [
        "returns",
        "coverage",
        "expectancy",
        "edge",
        "capture",
        "entry_quality",
        "risk",
    ]


def test_k_stock_strategy_id_resolves_source_only_overlay() -> None:
    profile = resolve_score_profile(
        family="k_stock_olr_kalcb",
        strategy_id="kalcb",
    )

    assert profile.profile_id == "k_stock.kalcb"
    assert profile.source_round == "source_only_no_output_rounds_in_checkout"
    assert [component.name for component in profile.components] == [
        "official_mtm_net_return_pct",
        "expected_total_r",
        "profit_factor",
        "avg_r",
        "entry_count",
        "mfe_capture",
        "max_drawdown_pct",
    ]


def test_non_crypto_plugin_id_resolves_family_profile_without_duplicate_mapping() -> None:
    profile = resolve_score_profile(
        plugin_id="trading-swing-family",
        strategy_id="helix",
    )

    assert profile.profile_id == "trading.swing.helix"


def test_score_replay_persists_profile_and_renormalized_components() -> None:
    profile = resolve_score_profile(
        family="trading_swing_family",
        strategy_id="tpc",
    )

    result = score_replay(
        profile=profile,
        trades=[
            {"symbol": "QQQ", "return_pct": 0.020},
            {"symbol": "QQQ", "return_pct": -0.004},
            {"symbol": "SPY", "return_pct": 0.010},
        ],
        coverage=[
            {"symbol": "QQQ", "rows": 120},
            {"symbol": "SPY", "rows": 120},
        ],
        window=WindowSpec("fold_1", date(2026, 1, 1), date(2026, 1, 31)),
        net_return=(0.020 - 0.004 + 0.010) / 3,
        max_drawdown=0.004,
        profit_factor=7.5,
        component_cap=5,
    )
    payload = result.to_payload()

    assert result.objective_score > 0.0
    assert payload["profile_id"] == "trading.swing.tpc"
    assert len(payload["renormalized_components"]) == 5
    assert [item["selected"] for item in payload["components"]] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    weight_sum = sum(
        item["renormalized_weight"] for item in payload["renormalized_components"]
    )
    assert weight_sum == pytest.approx(1.0)
    assert payload["profile"]["source_round"] == "round_8"

    compact = compact_score_payload(payload)

    assert compact["profile_id"] == "trading.swing.tpc"
    assert "profile" not in compact
    assert len(compact["renormalized_components"]) == 5
