from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.fixtures import normalized_source_payload
from tests.integration.parity.live_shadow_contract import source_fingerprint
from tests.integration.parity.portfolio_rules import portfolio_rules_config_from_fixture
from tests.integration.parity.source_inputs import overlay_rebalance_payload, plain, strategy_ids


def runtime_source_payload(fixture: Mapping[str, Any]) -> dict[str, Any]:
    payload = normalized_source_payload(fixture)
    payload["runtime_inputs"] = {
        "configured_strategy_ids": strategy_ids(fixture),
        "portfolio_rules": plain(portfolio_rules_config_from_fixture(fixture)),
        "overlay_rebalance": overlay_rebalance_payload(fixture),
    }
    return payload


def runtime_source_fingerprint(fixture: Mapping[str, Any]) -> str:
    return source_fingerprint(runtime_source_payload(fixture))
