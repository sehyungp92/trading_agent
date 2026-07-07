"""Benchmark case schema for analysis harness regression testing.

Compiles learning signals (validation blocks, negative outcomes, calibration
misses, transfer failures) into a unified evaluation corpus.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class BenchmarkSource(str, Enum):
    VALIDATION_BLOCK = "validation_block"
    NEGATIVE_OUTCOME = "negative_outcome"
    CALIBRATION_MISS = "calibration_miss"
    TRANSFER_FAILURE = "transfer_failure"
    DISCARDED_HARNESS_EXPERIMENT = "discarded_harness_experiment"


class BenchmarkSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BenchmarkCase(BaseModel):
    """A single benchmark case derived from a learning signal."""

    case_id: str = Field(description="Deterministic SHA256 from source:source_id, truncated to 16 chars.")
    source: BenchmarkSource
    source_id: str = Field(description="ID of the originating artifact.")
    severity: BenchmarkSeverity
    bot_id: str = ""
    agent_type: str = ""
    date: str = ""
    title: str = ""
    description: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""
    provider: str = ""
    model: str = ""
    source_run_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_refs: list[str] = Field(default_factory=list)
    case_tags: list[str] = Field(default_factory=list)
    score_profile: dict[str, float] = Field(default_factory=dict)
    input_snapshot: dict = Field(default_factory=dict)
    output_snapshot: dict = Field(default_factory=dict)

    @staticmethod
    def make_case_id(source: BenchmarkSource, source_id: str) -> str:
        raw = f"{source.value}:{source_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class BenchmarkSuite(BaseModel):
    """A collection of benchmark cases."""

    cases: list[BenchmarkCase] = Field(default_factory=list)
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_summary: dict = Field(default_factory=dict)
