from __future__ import annotations

import json
from pathlib import Path

import pytest

from backtests.shared.parity.calibration_report import (
    CalibrationReportError,
    latest_calibration_report_path,
    load_calibration_report,
)
from tests.integration.parity.strict_gate import fail_if_marker_selected_else_skip, marker_selected


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parity_nightly
def test_parity_nightly_requires_verified_nqdtc_ioc_baseline(request: pytest.FixtureRequest) -> None:
    manifest_path = REPO_ROOT / "tests" / "fixtures" / "backtest_baselines" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"artifacts": []}
    verified = any(
        entry.get("id") == "momentum_nqdtc_round_1_post_ioc_final_diagnostics"
        and entry.get("regeneration", {}).get("verified_post_ioc") is True
        for entry in manifest.get("artifacts", [])
    )
    if not verified:
        fail_if_marker_selected_else_skip(
            request,
            marker="parity_nightly",
            reason=(
                "parity_nightly is not production-confidence ready; the canonical "
                "NQDTC baseline has not been verified as regenerated after the IOC fill-model change."
            ),
        )


@pytest.mark.parity_calibration
def test_parity_calibration_requires_broker_backed_report(request: pytest.FixtureRequest) -> None:
    latest = latest_calibration_report_path(root=REPO_ROOT)
    if latest is None:
        fail_if_marker_selected_else_skip(
            request,
            marker="parity_calibration",
            reason=(
                "parity_calibration is not production-confidence ready; no broker-backed "
                "docs/parity_calibration_<date>.json report exists."
            ),
        )
    try:
        load_calibration_report(latest)
    except CalibrationReportError as exc:
        if marker_selected(request.config, "parity_calibration"):
            pytest.fail(f"latest broker-backed calibration report is invalid: {latest}: {exc}")
        pytest.skip(f"latest broker-backed calibration report is invalid: {latest}: {exc}")
