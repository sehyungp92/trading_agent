from __future__ import annotations

import json
import subprocess
import urllib.request
from datetime import datetime, timezone

from crypto_trader.instrumentation.sinks import JsonlSink
from crypto_trader.instrumentation.sidecar import SidecarForwarder
from crypto_trader.instrumentation.types import (
    EventMetadata,
    GenericInstrumentationEvent,
    InstrumentedTradeEvent,
    canonical_event_envelope,
)


def test_canonical_envelope_wraps_legacy_payload() -> None:
    payload = {
        "metadata": {
            "event_id": "e1",
            "bot_id": "bot1",
            "strategy_id": "momentum",
            "exchange_timestamp": "2026-05-31T00:00:00+00:00",
        },
        "pair": "BTC",
    }

    wrapped = canonical_event_envelope("trade", payload, bot_id="bot1")

    assert wrapped["schema_version"] == "assistant_event_v1"
    assert wrapped["event_type"] == "trade"
    assert wrapped["event_id"] == "e1"
    assert wrapped["symbol"] == "BTC"
    assert wrapped["payload"]["pair"] == "BTC"
    assert wrapped["payload"]["event_id"] == "e1"
    assert wrapped["payload"]["event_type"] == "trade"
    assert wrapped["payload"]["bot_id"] == "bot1"
    assert wrapped["payload"]["strategy_id"] == "momentum"
    assert wrapped["payload"]["assistant_strategy_id"] == "MomentumPullback_M15"


def test_canonical_envelope_merges_source_for_existing_canonical_payload() -> None:
    payload = {
        "schema_version": "assistant_event_v1",
        "event_id": "e1",
        "logical_event_id": "e1",
        "event_type": "portfolio_snapshot",
        "bot_id": "bot1",
        "source": {"sink": "jsonl"},
        "payload": {"event_id": "e1"},
    }

    wrapped = canonical_event_envelope(
        "portfolio_snapshot",
        payload,
        source={"file_event_type": "portfolio_snapshot"},
    )

    assert wrapped["event_id"] == "e1"
    assert wrapped["source"] == {
        "sink": "jsonl",
        "file_event_type": "portfolio_snapshot",
    }
    assert wrapped["payload"]["event_id"] == "e1"
    assert wrapped["payload"]["event_type"] == "portfolio_snapshot"
    assert wrapped["payload"]["bot_id"] == "bot1"
    assert payload["source"] == {"sink": "jsonl"}
    assert payload["payload"] == {"event_id": "e1"}


def test_existing_canonical_envelope_duplicates_replay_keys_into_payload() -> None:
    payload = {
        "schema_version": "assistant_event_v1",
        "event_id": "risk_evt_1",
        "logical_event_id": "risk_evt_1",
        "event_type": "risk_decision",
        "bot_id": "bot1",
        "family_id": "crypto_perps",
        "portfolio_id": "paper_portfolio",
        "account_alias": "paper",
        "strategy_id": "momentum",
        "assistant_strategy_id": "MomentumPullback_M15",
        "exchange_timestamp": "2026-05-31T00:00:00+00:00",
        "local_timestamp": "2026-05-31T00:00:01+00:00",
        "deployment_id": "deploy1",
        "config_version": "cfg1",
        "code_sha": "sha1",
        "portfolio_rule_event_id": "rule1",
        "risk_decision_id": "risk1",
        "intent_id": "intent1",
        "client_order_id": "client1",
        "order_id": "order1",
        "fill_id": "fill1",
        "payload": {"action": "block"},
    }

    wrapped = canonical_event_envelope("risk_decision", payload)

    for key in (
        "event_id",
        "event_type",
        "bot_id",
        "family_id",
        "portfolio_id",
        "account_alias",
        "strategy_id",
        "assistant_strategy_id",
        "exchange_timestamp",
        "local_timestamp",
        "deployment_id",
        "config_version",
        "code_sha",
        "portfolio_rule_event_id",
        "risk_decision_id",
        "intent_id",
        "client_order_id",
        "order_id",
        "fill_id",
    ):
        assert wrapped["payload"][key] == payload[key]
    assert wrapped["payload_hash"]
    assert payload["payload"] == {"action": "block"}


def test_event_metadata_id_includes_strategy_id() -> None:
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)

    first = EventMetadata.create("bot1", "momentum", ts, "trade", "t1")
    second = EventMetadata.create("bot1", "trend", ts, "trade", "t1")

    assert first.event_id != second.event_id


def test_jsonl_sink_writes_legacy_and_date_partitioned_trade(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = InstrumentedTradeEvent(
        metadata=EventMetadata.create("bot1", "momentum", ts, "trade", "t1"),
        trade_id="t1",
        pair="BTC",
    )

    sink.write_trade(event)

    assert (tmp_path / "instrumented_trades.jsonl").exists()
    canonical_path = tmp_path / "instrumentation" / "events" / "trade" / "2026-05-31.jsonl"
    assert canonical_path.exists()
    row = json.loads(canonical_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["schema_version"] == "assistant_event_v1"
    assert row["event_type"] == "trade"
    assert row["payload"]["trade_id"] == "t1"


def test_generic_event_writes_to_canonical_type_file(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = GenericInstrumentationEvent(
        metadata=EventMetadata.create(
            "bot1",
            "portfolio",
            ts,
            "portfolio_snapshot",
            "p1",
            family_id="crypto_perps",
            portfolio_id="paper_portfolio",
            account_alias="paper",
            config_version="cfg1",
            deployment_id="deploy1",
            code_sha="sha1",
        ),
        payload={"portfolio_id": "p1", "timestamp": ts.isoformat()},
    )

    sink.write_event("portfolio_snapshot", event)

    path = tmp_path / "instrumentation" / "events" / "portfolio_snapshot" / "2026-05-31.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["event_type"] == "portfolio_snapshot"
    assert row["payload"]["portfolio_id"] == "p1"
    assert row["deployment_id"] == "deploy1"
    assert row["config_version"] == "cfg1"
    assert row["code_sha"] == "sha1"
    assert row["payload"]["deployment_id"] == "deploy1"
    assert row["payload"]["config_version"] == "cfg1"
    assert row["payload"]["code_sha"] == "sha1"
    assert row["source"]["sink"] == "jsonl"


def test_generic_order_event_payload_can_stand_alone_for_replay() -> None:
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = GenericInstrumentationEvent(
        metadata=EventMetadata.create(
            "bot1",
            "momentum",
            ts,
            "order",
            "order_1",
            family_id="crypto_perps",
            portfolio_id="paper_portfolio",
            account_alias="paper",
            config_version="cfg1",
            deployment_id="deploy1",
            code_sha="sha1",
        ),
        payload={
            "intent_id": "intent1",
            "client_order_id": "client1",
            "portfolio_rule_event_id": "rule1",
            "risk_decision_id": "risk1",
        },
    )

    row = event.to_dict()

    payload = row["payload"]
    assert payload["event_id"] == row["event_id"]
    assert payload["event_type"] == "order"
    assert payload["bot_id"] == "bot1"
    assert payload["family_id"] == "crypto_perps"
    assert payload["portfolio_id"] == "paper_portfolio"
    assert payload["account_alias"] == "paper"
    assert payload["strategy_id"] == "momentum"
    assert payload["assistant_strategy_id"] == "MomentumPullback_M15"
    assert payload["deployment_id"] == "deploy1"
    assert payload["config_version"] == "cfg1"
    assert payload["code_sha"] == "sha1"
    assert payload["intent_id"] == "intent1"
    assert payload["client_order_id"] == "client1"
    assert payload["order_id"] == "client1"
    assert payload["portfolio_rule_event_id"] == "rule1"
    assert payload["risk_decision_id"] == "risk1"
    assert row["intent_id"] == "intent1"
    assert row["client_order_id"] == "client1"
    assert row["order_id"] == "client1"
    assert row["portfolio_rule_event_id"] == "rule1"
    assert row["risk_decision_id"] == "risk1"


def test_sidecar_keeps_unread_legacy_file_when_canonical_copy_exists(tmp_path) -> None:
    sink = JsonlSink(tmp_path)
    ts = datetime(2026, 5, 31, tzinfo=timezone.utc)
    event = InstrumentedTradeEvent(
        metadata=EventMetadata.create("bot1", "momentum", ts, "trade", "t1"),
        trade_id="t1",
        pair="BTC",
    )
    sink.write_trade(event)

    forwarder = SidecarForwarder(tmp_path, "http://localhost:8000", "bot1", "secret")
    sources = forwarder._event_sources()

    canonical_source = (
        "trade",
        tmp_path / "instrumentation" / "events" / "trade" / "2026-05-31.jsonl",
        "instrumentation/events/trade/2026-05-31.jsonl",
    )
    legacy_source = (
        "instrumented_trades",
        tmp_path / "instrumented_trades.jsonl",
        "instrumented_trades",
    )
    assert canonical_source in sources
    assert legacy_source in sources

    forwarder._watermarks["instrumented_trades"] = (tmp_path / "instrumented_trades.jsonl").stat().st_size
    sources = forwarder._event_sources()
    assert canonical_source in sources
    assert legacy_source not in sources


def test_sidecar_emits_error_event_after_retry_exhaustion(tmp_path, monkeypatch) -> None:
    errors: list[dict] = []
    forwarder = SidecarForwarder(
        tmp_path,
        "http://localhost:8000",
        "bot1",
        "secret",
        error_callback=errors.append,
    )
    monkeypatch.setattr(forwarder._stop_event, "wait", lambda _wait: False)

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("relay unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    delivered = forwarder._send_batch([{"event_id": "e1"}], "instrumented_trades")

    assert delivered is False
    assert len(errors) == 1
    assert errors[0]["component"] == "sidecar"
    assert errors[0]["error_type"] == "OSError"
    assert errors[0]["event_type"] == "instrumented_trades"
    assert errors[0]["recovery_action"] == "retry_next_poll"
    assert "relay unavailable" in errors[0]["message"]


def test_sidecar_does_not_append_recursive_error_when_forwarding_errors(tmp_path, monkeypatch) -> None:
    errors: list[dict] = []
    forwarder = SidecarForwarder(
        tmp_path,
        "http://localhost:8000",
        "bot1",
        "secret",
        error_callback=errors.append,
    )
    monkeypatch.setattr(forwarder._stop_event, "wait", lambda _wait: False)

    def fail_urlopen(*_args, **_kwargs):
        raise OSError("relay unavailable")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    delivered = forwarder._send_batch([{"event_id": "err_1"}], "errors")

    assert delivered is False
    assert errors == []
    assert forwarder.status()["consecutive_send_failures"] == 1
    assert forwarder.status()["last_error"] == "relay unavailable"


def test_sidecar_status_reports_buffered_event_depth(tmp_path) -> None:
    forwarder = SidecarForwarder(
        tmp_path,
        "http://localhost:8000",
        "bot1",
        "secret",
    )
    event_file = tmp_path / "instrumented_trades.jsonl"
    event_file.write_text(
        json.dumps({"event_id": "e1"}) + "\n" + json.dumps({"event_id": "e2"}) + "\n",
        encoding="utf-8",
    )

    status = forwarder.status()

    assert status["buffered_event_count"] == 2
    assert status["oldest_buffered_event_age_seconds"] >= 0


def test_acceptance_guide_is_review_visible_to_git() -> None:
    completed = subprocess.run(
        [
            "git",
            "check-ignore",
            "docs/2026-05-31-crypto-trader-instrumentation-implementation-guide.md",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
