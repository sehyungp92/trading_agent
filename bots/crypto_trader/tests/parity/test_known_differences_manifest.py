"""Checks for the machine-readable parity known-differences manifest."""

from __future__ import annotations

import json
from pathlib import Path


MANIFEST = Path(__file__).with_name("known_differences.json")

REQUIRED_IDS = {
    "live_actual_fill_price",
    "delayed_live_fill_handling",
    "durable_oms_missing",
    "orchestration_drift",
    "market_timestamp_policy_normalized",
    "portfolio_terminal_accounting_divergence",
    "economic_contract_normalized",
}


def test_known_differences_manifest_is_valid_json() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload


def test_known_differences_have_required_fields() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    for entry in payload:
        assert set(entry) >= {"id", "phase", "status", "description", "mitigation"}
        assert entry["id"]
        assert entry["phase"]
        assert entry["status"] in {
            "known_gap",
            "accepted_difference",
            "covered_by_regression",
        }
        assert entry["description"]
        assert entry["mitigation"]


def test_required_known_differences_are_present() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = {entry["id"] for entry in payload}
    assert REQUIRED_IDS <= ids


def test_known_difference_ids_are_unique() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ids = [entry["id"] for entry in payload]
    assert len(ids) == len(set(ids))
