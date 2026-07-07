"""Tests for temporal decay on findings loading (B2)."""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta

from trading_assistant.analysis.context_builder import ContextBuilder, _apply_temporal_window, _safe_jsonl
from trading_assistant.orchestrator.jsonl_store import read_jsonl_tail


@pytest.fixture
def memory_dir(tmp_path):
    findings = tmp_path / "findings"
    findings.mkdir()
    policies = tmp_path / "policies" / "v1"
    policies.mkdir(parents=True)
    return tmp_path


def test_recent_entries_appear_before_old_ones():
    now = datetime.now(timezone.utc)
    entries = [
        {"timestamp": (now - timedelta(days=30)).isoformat(), "id": "old"},
        {"timestamp": (now - timedelta(days=1)).isoformat(), "id": "new"},
        {"timestamp": (now - timedelta(days=15)).isoformat(), "id": "mid"},
    ]
    result = _apply_temporal_window(entries)
    assert result[0]["id"] == "new"
    assert result[1]["id"] == "mid"
    assert result[2]["id"] == "old"


def test_entries_older_than_90_days_excluded():
    now = datetime.now(timezone.utc)
    entries = [
        {"timestamp": (now - timedelta(days=10)).isoformat(), "id": "recent"},
        {"timestamp": (now - timedelta(days=100)).isoformat(), "id": "old"},
    ]
    result = _apply_temporal_window(entries)
    assert len(result) == 1
    assert result[0]["id"] == "recent"


def test_max_50_entries_returned():
    now = datetime.now(timezone.utc)
    entries = [
        {"timestamp": (now - timedelta(days=i)).isoformat(), "id": str(i)}
        for i in range(60)
    ]
    result = _apply_temporal_window(entries)
    assert len(result) == 50
    # Most recent should be first
    assert result[0]["id"] == "0"


def test_custom_max_entries_returned():
    now = datetime.now(timezone.utc)
    entries = [
        {"timestamp": (now - timedelta(days=i)).isoformat(), "id": str(i)}
        for i in range(20)
    ]
    result = _apply_temporal_window(entries, max_entries=5)
    assert len(result) == 5
    assert result[0]["id"] == "0"


def test_entries_without_timestamp_included_last():
    now = datetime.now(timezone.utc)
    entries = [
        {"id": "no_ts"},
        {"timestamp": (now - timedelta(days=1)).isoformat(), "id": "with_ts"},
    ]
    result = _apply_temporal_window(entries)
    assert len(result) == 2
    assert result[0]["id"] == "with_ts"
    assert result[1]["id"] == "no_ts"


def test_empty_findings_handled_gracefully():
    result = _apply_temporal_window([])
    assert result == []


def test_safe_jsonl_reads_bounded_tail(memory_dir):
    path = memory_dir / "findings" / "corrections.jsonl"
    lines = [json.dumps({"id": i}) for i in range(1105)]
    path.write_text("\n".join(lines), encoding="utf-8")

    records = _safe_jsonl(path)

    assert len(records) == 1000
    assert records[0]["id"] == 105
    assert records[-1]["id"] == 1104


def test_jsonl_tail_seeks_from_end_with_small_chunks(memory_dir):
    path = memory_dir / "findings" / "corrections.jsonl"
    lines = [json.dumps({"id": i, "payload": "x" * 80}) for i in range(200)]
    path.write_text("\n".join(lines), encoding="utf-8")

    records = read_jsonl_tail(path, max_records=5, chunk_size=64)

    assert [record["id"] for record in records] == [195, 196, 197, 198, 199]


def test_load_corrections_applies_temporal_window(memory_dir):
    now = datetime.now(timezone.utc)
    corrections_path = memory_dir / "findings" / "corrections.jsonl"
    lines = [
        json.dumps({"timestamp": (now - timedelta(days=1)).isoformat(), "id": "recent"}),
        json.dumps({"timestamp": (now - timedelta(days=100)).isoformat(), "id": "old"}),
    ]
    corrections_path.write_text("\n".join(lines))

    ctx = ContextBuilder(memory_dir)
    result = ctx.load_corrections()
    assert len(result) == 1
    assert result[0]["id"] == "recent"


def test_load_failure_log_applies_temporal_window(memory_dir):
    now = datetime.now(timezone.utc)
    log_path = memory_dir / "findings" / "failure-log.jsonl"
    lines = [
        json.dumps({"timestamp": (now - timedelta(days=5)).isoformat(), "type": "err"}),
        json.dumps({"timestamp": (now - timedelta(days=200)).isoformat(), "type": "ancient"}),
    ]
    log_path.write_text("\n".join(lines))

    ctx = ContextBuilder(memory_dir)
    result = ctx.load_failure_log()
    assert len(result) == 1
    assert result[0]["type"] == "err"
