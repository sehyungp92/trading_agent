from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.parity.harness import run_layer2_contract
from tests.integration.parity.live_shadow_contract import assert_shadow_contract

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "layer2"


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "fixture_name"),
    [
        ("IARIC", "iaric_entry_fill.json"),
        ("NQ_REGIME", "nq_regime_entry_fill.json"),
        ("TPC", "tpc_entry_fill.json"),
    ],
    ids=["iaric", "nq_regime", "tpc"],
)
async def test_live_shadow_layer2_matches_offline_oms_replay_contract(
    surface: str,
    fixture_name: str,
) -> None:
    assert_shadow_contract(await run_layer2_contract(surface, FIXTURE_ROOT / fixture_name))
