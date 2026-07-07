from __future__ import annotations

from pathlib import Path

from backtests.scalp.auto.ivb_auction.plugin import IvbAuctionPlugin
from backtests.scalp.auto.po3_reversal.plugin import Po3ReversalPlugin
from backtests.scalp.engine.param_overrides import temporary_param_overrides
from backtests.shared.auto.cache_keys import build_cache_key
from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp.ivb_auction import gates as ivb_gates


def test_po3_plugin_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = Po3ReversalPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(plugin, "_replay_bundle", lambda: type("Bundle", (), {"cache_source_fingerprint": "po3-fp"})())

    evaluator = plugin.create_evaluate_batch(2, {})

    assert evaluator._signature_prefix == build_cache_key(
        "scalp.po3_reversal.evaluation",
        source_fingerprint="po3-fp",
        extra={
            "phase": 2,
            "analysis_symbol": "NQ",
            "trade_symbol": "MNQ",
            "confirmation_symbol": "ES",
            "symbols": ["MNQ"],
            "scoring_weights": {},
            "hard_rejects": {},
        },
    )


def test_ivb_plugin_namespaces_candidate_cache_by_source_fingerprint(monkeypatch, tmp_path: Path) -> None:
    plugin = IvbAuctionPlugin(tmp_path, max_workers=1)
    monkeypatch.setattr(plugin, "_replay_bundle", lambda: type("Bundle", (), {"cache_source_fingerprint": "ivb-fp"})())

    evaluator = plugin.create_evaluate_batch(3, {})

    assert evaluator._signature_prefix == build_cache_key(
        "scalp.ivb_auction.evaluation",
        source_fingerprint="ivb-fp",
        extra={
            "phase": 3,
            "analysis_symbol": "NQ",
            "trade_symbol": "MNQ",
            "symbols": ["MNQ"],
            "scoring_weights": {},
            "hard_rejects": {},
        },
    )


def test_scalp_param_overrides_patch_strategy_modules_for_candidate_runs() -> None:
    ivb = IVBLevels.from_bounds(130.0, 100.0)
    default_passed, _ = ivb_gates.breakout_acceptance(
        direction=1,
        close=131.0,
        high=133.0,
        low=129.0,
        ivb=ivb,
        held_seconds=60.0,
        breakout_volume=0.0,
        rolling_volume_median=50.0,
        delta_60s=None,
        rolling_delta_median=0.0,
    )

    with temporary_param_overrides({"MIN_HOLD_SECONDS": 120}, [ivb_gates]):
        overridden_passed, _ = ivb_gates.breakout_acceptance(
            direction=1,
            close=131.0,
            high=133.0,
            low=129.0,
            ivb=ivb,
            held_seconds=60.0,
            breakout_volume=0.0,
            rolling_volume_median=50.0,
            delta_60s=None,
            rolling_delta_median=0.0,
        )

    assert default_passed
    assert not overridden_passed
