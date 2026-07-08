from __future__ import annotations

import pytest


def marker_selected(config: pytest.Config, marker: str) -> bool:
    return marker in (getattr(config.option, "markexpr", "") or "")


def fail_if_marker_selected_else_skip(
    request: pytest.FixtureRequest,
    *,
    marker: str,
    reason: str,
) -> None:
    if marker_selected(request.config, marker):
        pytest.fail(reason)
    pytest.skip(reason)


def require_real_parity_harness(request: pytest.FixtureRequest, surface: str) -> None:
    fail_if_marker_selected_else_skip(
        request,
        marker="parity_nightly",
        reason=(
            "parity_nightly is not production-confidence ready; "
            f"{surface} requires a real OMS-vs-replay harness, not a placeholder trace."
        ),
    )
