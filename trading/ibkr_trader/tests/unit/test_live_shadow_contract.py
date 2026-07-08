from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.integration.parity.live_shadow_contract import (
    LiveShadowContract,
    ParityTrace,
    assert_merged_family_ledger,
    assert_shadow_contract,
    normalize_fingerprint_payload,
    source_fingerprint,
)


def test_shadow_contract_accepts_independent_equal_traces() -> None:
    fingerprint = source_fingerprint({"bars": [{"close": 101.0}]})
    live = ParityTrace(
        producer="live_oms",
        source_fingerprint=fingerprint,
        order_intents=[{"symbol": "QQQ", "qty": 10}],
        terminal_events=[{"event_type": "FILL", "qty": 10}],
        trade_ledger=[{"symbol": "QQQ", "entry_price": 101.0}],
        state_snapshot={"open_qty": 10},
    )
    replay = ParityTrace(
        producer="backtest_replay",
        source_fingerprint=fingerprint,
        order_intents=[{"symbol": "QQQ", "qty": 10}],
        terminal_events=[{"event_type": "FILL", "qty": 10}],
        trade_ledger=[{"symbol": "QQQ", "entry_price": 101.0}],
        state_snapshot={"open_qty": 10},
    )

    assert_shadow_contract(LiveShadowContract(surface="TPC", live=live, replay=replay))


@pytest.mark.parametrize(
    ("field_name", "live_updates", "replay_updates"),
    [
        ("source_fingerprint", {"source_fingerprint": "live"}, {"source_fingerprint": "replay"}),
        ("order_intents", {"order_intents": [{"qty": 1}]}, {"order_intents": [{"qty": 2}]}),
        ("terminal_events", {"terminal_events": [{"event_type": "FILL"}]}, {"terminal_events": []}),
        ("trade_ledger", {"trade_ledger": [{"entry_price": 1.0}]}, {"trade_ledger": [{"entry_price": 2.0}]}),
        ("state_snapshot", {"state_snapshot": {"qty": 1}}, {"state_snapshot": {"qty": 2}}),
    ],
)
def test_shadow_contract_rejects_trace_mismatches(
    field_name: str,
    live_updates: dict,
    replay_updates: dict,
) -> None:
    live = _trace(**live_updates)
    replay = _trace(**replay_updates)

    with pytest.raises(AssertionError, match=field_name):
        assert_shadow_contract(LiveShadowContract(surface="demo", live=live, replay=replay))


def test_parity_nightly_helpers_do_not_use_single_order_placeholder() -> None:
    parity_dir = Path(__file__).resolve().parents[1] / "integration" / "parity"
    text = "\n".join(path.read_text(encoding="utf-8") for path in parity_dir.glob("test_live_shadow_*.py"))
    contract_text = (parity_dir / "live_shadow_contract.py").read_text(encoding="utf-8")

    assert "run_single_order_shadow_contract" not in text
    assert "backtest_order_intents=list(live_order_intents)" not in contract_text
    assert "backtest_trade_ledger=list(live_trade_ledger)" not in contract_text


def test_source_fingerprint_normalizes_runtime_values_without_string_fallback() -> None:
    payload = {
        "runtime_inputs": {
            "as_of": datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc),
            "strategy_ids": {"NQ_REGIME", "TPC"},
        }
    }

    normalized = normalize_fingerprint_payload(payload)

    assert normalized["runtime_inputs"]["as_of"] == "2026-05-20T14:30:00+00:00"
    assert normalized["runtime_inputs"]["strategy_ids"] == ["NQ_REGIME", "TPC"]
    assert source_fingerprint(payload) == source_fingerprint(normalized)


def test_source_fingerprint_rejects_unmodeled_runtime_values() -> None:
    with pytest.raises(TypeError, match="unsupported source fingerprint value"):
        source_fingerprint({"runtime_inputs": object()})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_source_fingerprint_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(TypeError, match="non-finite"):
        source_fingerprint({"runtime_inputs": {"bad": value}})


def test_merged_family_ledger_checks_each_child_contract() -> None:
    contract = LiveShadowContract(
        surface="family-child",
        live=_trace(source_fingerprint="live"),
        replay=_trace(source_fingerprint="replay"),
    )

    with pytest.raises(AssertionError, match="source_fingerprint"):
        assert_merged_family_ledger([contract], expected_trades=1)


def test_merged_family_ledger_fails_when_configured_child_is_omitted() -> None:
    matching = LiveShadowContract(surface="family:A", live=_trace(), replay=_trace())

    with pytest.raises(AssertionError, match="child contract mismatch"):
        assert_merged_family_ledger(
            [matching],
            expected_trades=1,
            expected_surfaces={"family:A", "family:B"},
        )


def test_merged_family_ledger_checks_trade_count_and_rows() -> None:
    matching = LiveShadowContract(surface="family-child", live=_trace(), replay=_trace())
    mismatched_ledger = LiveShadowContract(
        surface="family-child-2",
        live=_trace(trade_ledger=[{"symbol": "QQQ", "entry_price": 1.0}]),
        replay=_trace(trade_ledger=[{"symbol": "QQQ", "entry_price": 2.0}]),
    )

    assert_merged_family_ledger([matching], expected_trades=1)
    with pytest.raises(AssertionError, match="trade-count"):
        assert_merged_family_ledger([matching], expected_trades=2)
    with pytest.raises(AssertionError, match="trade_ledger"):
        assert_merged_family_ledger([matching, mismatched_ledger], expected_trades=2)


def _trace(**updates) -> ParityTrace:
    payload = {
        "producer": "producer",
        "source_fingerprint": "same",
        "order_intents": [{"symbol": "QQQ", "qty": 1}],
        "terminal_events": [{"event_type": "FILL", "qty": 1}],
        "trade_ledger": [{"symbol": "QQQ", "entry_price": 1.0}],
        "state_snapshot": {"qty": 1},
    }
    payload.update(updates)
    return ParityTrace(**payload)
