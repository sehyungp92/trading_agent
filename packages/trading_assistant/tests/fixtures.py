# tests/fixtures.py
"""Shared pytest fixtures — imported by conftest.py for global availability.

These replace duplicated per-file fixture definitions for EventStream,
SessionStore, memory_dir, sample_package, etc.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.factories import make_sample_package


@pytest.fixture
def event_stream():
    """Provide a fresh EventStream instance."""
    from trading_assistant.orchestrator.event_stream import EventStream
    return EventStream()


@pytest.fixture
def mock_event_stream():
    """Provide a MagicMock EventStream."""
    from unittest.mock import MagicMock
    es = MagicMock()
    es.broadcast = MagicMock()
    return es


@pytest.fixture
def session_store(tmp_path: Path):
    """Provide a SessionStore backed by tmp_path."""
    from trading_assistant.orchestrator.session_store import SessionStore
    return SessionStore(store_dir=tmp_path / "sessions")


@pytest.fixture
def memory_dir(tmp_path: Path) -> Path:
    """Create a standard memory directory layout for tests."""
    mem = tmp_path / "memory"
    (mem / "policies" / "v1").mkdir(parents=True)
    (mem / "findings").mkdir(parents=True)
    return mem


@pytest.fixture
def memory_dir_with_policies(memory_dir: Path) -> Path:
    """memory_dir with standard policy files pre-populated."""
    policies = memory_dir / "policies" / "v1"
    (policies / "agent.md").write_text("Agent policy")
    (policies / "trading_rules.md").write_text("Rules")
    (policies / "soul.md").write_text("Soul")
    return memory_dir


@pytest.fixture
def sample_package():
    """Provide a default PromptPackage for agent runner tests."""
    return make_sample_package()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a standard curated data directory layout."""
    curated = tmp_path / "data" / "curated"
    curated.mkdir(parents=True)
    return curated
