"""Tests for paper shadow replay diff helpers."""

from crypto_trader.parity.shadow import compare_event_streams


def test_shadow_replay_report_passes_matching_streams() -> None:
    events = [{"decision_id": "d1", "symbol": "BTC", "side": "LONG"}]

    report = compare_event_streams(events, list(events))

    assert report.passed is True
    assert report.to_dict()["drifts"] == []


def test_shadow_replay_report_records_field_and_count_drift() -> None:
    report = compare_event_streams(
        [{"decision_id": "d1", "symbol": "BTC"}],
        [
            {"decision_id": "d2", "symbol": "BTC"},
            {"decision_id": "d3", "symbol": "ETH"},
        ],
        keys=("decision_id", "symbol"),
    )

    payload = report.to_dict()
    assert payload["passed"] is False
    assert {drift["field"] for drift in payload["drifts"]} == {"decision_id", "event_count"}
