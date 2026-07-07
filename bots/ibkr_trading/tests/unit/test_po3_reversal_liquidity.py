from __future__ import annotations

from strategies.scalp.po3_reversal.liquidity import detect_liquidity_pools, detect_sweep


def test_liquidity_pool_and_sweep_detection() -> None:
    pools = detect_liquidity_pools([100, 100.25, 103], [95, 95.25, 97], min_touches=2, tolerance_ticks=1)
    sell_pool = [pool for pool in pools if pool.side == "sell_side"][0]

    sweep = detect_sweep(93.0, [sell_pool], min_ticks=4)

    assert sweep.swept
    assert sweep.distance_ticks >= 4

