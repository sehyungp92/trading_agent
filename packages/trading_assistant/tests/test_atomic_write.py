# tests/test_atomic_write.py
"""Tests for the atomic JSONL rewrite utility."""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from trading_assistant.skills._atomic_write import atomic_rewrite_jsonl


class SampleModel(BaseModel):
    name: str
    value: int


class TestAtomicRewriteJsonl:
    def test_basic_rewrite(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}]
        atomic_rewrite_jsonl(path, records)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_dict_serialization(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [{"key": "value", "num": 42, "nested": {"x": True}}]
        atomic_rewrite_jsonl(path, records)

        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["nested"]["x"] is True

    def test_pydantic_model_serialization(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [SampleModel(name="foo", value=10), SampleModel(name="bar", value=20)]
        atomic_rewrite_jsonl(path, records)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"name": "foo", "value": 10}

    def test_failure_leaves_original_intact(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"original": true}\n', encoding="utf-8")

        class BadObj:
            def __str__(self):
                raise RuntimeError("serialize error")

        # Patch json.dumps to raise on the bad object
        with pytest.raises(RuntimeError):
            atomic_rewrite_jsonl(path, [BadObj()])

        # Original file should be untouched
        content = path.read_text(encoding="utf-8").strip()
        assert json.loads(content) == {"original": True}

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        atomic_rewrite_jsonl(path, [{"ok": True}])
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"old": true}\n', encoding="utf-8")
        atomic_rewrite_jsonl(path, [{"new": True}])
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data == {"new": True}

    def test_empty_records(self, tmp_path):
        path = tmp_path / "test.jsonl"
        atomic_rewrite_jsonl(path, [])
        assert path.read_text(encoding="utf-8") == ""

    def test_no_temp_files_left(self, tmp_path):
        path = tmp_path / "test.jsonl"
        atomic_rewrite_jsonl(path, [{"a": 1}])
        files = list(tmp_path.iterdir())
        assert files == [path]
