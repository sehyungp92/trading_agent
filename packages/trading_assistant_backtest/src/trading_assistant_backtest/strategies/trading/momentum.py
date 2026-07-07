"""Momentum-family bridge backed by the production ``trading`` live-shadow harness."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head
from trading_assistant_backtest.strategies.trading.stock import (
    _family_event,
    _runtime_repo_copy,
    _trace_events,
    _trading_import_path,
)

PLUGIN_ID = "trading-momentum-family"
DECISION_API_VERSION = "trading_momentum_live_shadow_decision_api_v1"


def build_trading_momentum_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    fixture_paths: Iterable[str | Path],
    live_repo_path: str | Path,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
    """Build a parity report from the trading repo's momentum live-shadow fixtures."""

    repo_path = Path(live_repo_path)
    if live_repo_commit_sha:
        checkout_errors = validate_pinned_head(repo_path, live_repo_commit_sha)
        if checkout_errors:
            raise ValueError("; ".join(checkout_errors))

    live_events: list[DecisionTraceEvent] = []
    adapter_events: list[DecisionTraceEvent] = []
    evidence_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="trading-momentum-parity-") as scratch:
        runtime_repo_path = _runtime_repo_copy(repo_path, Path(scratch) / "trading")
        with _trading_import_path(runtime_repo_path):
            for fixture_path in fixture_paths:
                path = Path(fixture_path)
                fixture = json.loads(path.read_text(encoding="utf-8"))
                evidence_paths.append(str(path))
                live, adapter = _decision_traces_from_fixture(fixture, path)
                live_events.extend(live)
                adapter_events.extend(adapter)

    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )


def _decision_traces_from_fixture(
    fixture: Mapping[str, Any],
    path: Path,
) -> tuple[list[DecisionTraceEvent], list[DecisionTraceEvent]]:
    from tests.integration.parity.harness import run_layer2_contract, run_layer3_family_contract

    surface = str(fixture.get("surface") or "")
    if surface == "NQ_REGIME":
        contract = asyncio.run(run_layer2_contract("NQ_REGIME", path))
        return (
            _trace_events(contract.live, fixture, surface="momentum:NQ_REGIME"),
            _trace_events(contract.replay, fixture, surface="momentum:NQ_REGIME"),
        )
    if surface == "momentum_family":
        family = asyncio.run(run_layer3_family_contract("momentum", path))
        live_events: list[DecisionTraceEvent] = []
        adapter_events: list[DecisionTraceEvent] = []
        for child in family.children:
            live_events.extend(_trace_events(child.live, fixture, surface=child.surface))
            adapter_events.extend(_trace_events(child.replay, fixture, surface=child.surface))
        live_events.append(
            _family_event(family.live_family_state, fixture, surface="momentum:family")
        )
        adapter_events.append(
            _family_event(family.replay_family_state, fixture, surface="momentum:family")
        )
        return live_events, adapter_events
    raise ValueError(f"unsupported trading momentum parity surface: {surface!r}")

