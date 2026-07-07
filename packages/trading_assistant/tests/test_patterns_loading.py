"""Tests for loading patterns_consolidated.md into context (B0)."""
from __future__ import annotations

import pytest

from trading_assistant.analysis.context_builder import ContextBuilder


@pytest.fixture
def memory_dir(tmp_path):
    findings = tmp_path / "findings"
    findings.mkdir()
    policies = tmp_path / "policies" / "v1"
    policies.mkdir(parents=True)
    return tmp_path


def test_patterns_loaded_when_file_exists(memory_dir):
    content = "# Consolidated Patterns\n- bot_alpha: 42 errors"
    (memory_dir / "findings" / "patterns_consolidated.md").write_text(content)

    ctx = ContextBuilder(memory_dir)
    result = ctx.load_consolidated_patterns()
    assert "bot_alpha: 42 errors" in result


def test_empty_string_when_file_missing(memory_dir):
    ctx = ContextBuilder(memory_dir)
    result = ctx.load_consolidated_patterns()
    assert result == ""


def test_content_included_in_pkg_data(memory_dir):
    content = "# Patterns\n- high_slippage: 10"
    (memory_dir / "findings" / "patterns_consolidated.md").write_text(content)

    ctx = ContextBuilder(memory_dir)
    pkg = ctx.base_package()
    assert "consolidated_patterns" in pkg.data
    assert "high_slippage" in pkg.data["consolidated_patterns"]
