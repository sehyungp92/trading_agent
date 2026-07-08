from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.integration.parity.family_surface_names import (
    family_surface_adapter_name as _family_surface_adapter_name,
)
from tests.integration.parity.replay_candidates import ReplayDecisionTimeline
from tests.integration.parity.replay_momentum_family_surface import (
    run_momentum_family_surface as _run_momentum_family_surface,
)
from tests.integration.parity.replay_stock_family_surface import (
    run_stock_family_surface as _run_stock_family_surface,
)
from tests.integration.parity.replay_swing_family_surface import (
    run_swing_family_surface as _run_swing_family_surface,
)


def run_family_portfolio_surface(
    fixture: Mapping[str, Any],
    out: ReplayDecisionTimeline,
) -> dict[str, Any]:
    family = str(fixture.get("family", "") or (fixture.get("family_config", {}) or {}).get("family", ""))
    if family == "momentum":
        return _run_momentum_family_surface(fixture, out)
    if family == "stock":
        return _run_stock_family_surface(fixture, out)
    if family == "swing":
        return _run_swing_family_surface(fixture, out)
    return {"adapter": ""}


_run_family_portfolio_surface = run_family_portfolio_surface
family_surface_adapter_name = _family_surface_adapter_name
