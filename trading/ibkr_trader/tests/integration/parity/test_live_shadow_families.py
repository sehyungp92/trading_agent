from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.parity.fixtures import load_parity_fixture
from tests.integration.parity.harness import run_layer3_family_contract
from tests.integration.parity.live_shadow_contract import assert_family_shadow_contract

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "layer3"


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("family", "fixture_name", "expected_surfaces"),
    [
        (
            "momentum",
            "momentum_family_shared_risk.json",
            {
                "momentum:NQDTC_v2.1",
                "momentum:NQ_REGIME",
                "momentum:VdubusNQ_v4",
                "momentum:DownturnDominator_v1",
            },
        ),
        ("stock", "stock_family_collision.json", {"stock:IARIC_v1", "stock:ALCB_v1"}),
        ("swing", "swing_family_overlay_rebalance.json", {"swing:ATRSS", "swing:AKC_HELIX", "swing:TPC", "swing:OVERLAY"}),
    ],
    ids=["momentum", "stock", "swing"],
)
async def test_live_shadow_family_matches_contract(
    family: str,
    fixture_name: str,
    expected_surfaces: set[str],
) -> None:
    fixture_path = FIXTURE_ROOT / fixture_name
    fixture = load_parity_fixture(fixture_path)

    assert_family_shadow_contract(
        await run_layer3_family_contract(family, fixture_path),
        expected_trades=int(fixture.get("expected_trade_count", 0)),
        expected_surfaces=expected_surfaces,
    )
