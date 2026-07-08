from __future__ import annotations

import json

from backtests.shared.auto.phase_runner import PhaseRunner, _STRICT_PROVENANCE_SURFACES
from backtests.shared.auto.phase_state import PhaseState
from backtests.shared.auto.provenance import (
    DRIFT_STATUS_INCOMPLETE_SAVED_METRICS,
    DRIFT_STATUS_SELECTION_STALE,
    build_auto_run_provenance,
    build_file_item,
    build_phase_auto_provenance,
    build_json_item,
    classify_metric_drift,
)
from backtests.shared.auto.round_manager import RoundManager


class _NoProvenancePlugin:
    def __init__(self, name: str) -> None:
        self.name = name
        self.num_phases = 1
        self.ultimate_targets = {}
        self.initial_mutations = {}


class _ProvenancePlugin(_NoProvenancePlugin):
    def __init__(self, name: str, current_provenance, previous_round_provenance=None) -> None:
        super().__init__(name)
        self._current_provenance = current_provenance
        self.previous_round_provenance = previous_round_provenance
        self.initial_mutations = {"flags.current": True}

    def build_provenance(self):
        return self._current_provenance


def test_provenance_fingerprint_changes_when_tracked_source_hash_changes(tmp_path) -> None:
    source = tmp_path / "strategy.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    original = build_auto_run_provenance([build_file_item("strategy_source", source)])

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_auto_run_provenance([build_file_item("strategy_source", source)])

    assert changed.selection_fingerprint != original.selection_fingerprint


def test_diagnostics_only_changes_do_not_change_selection_fingerprint(tmp_path) -> None:
    source = tmp_path / "strategy.py"
    diagnostics = tmp_path / "diagnostics.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    diagnostics.write_text("FORMAT = 'old'\n", encoding="utf-8")

    original = build_auto_run_provenance(
        [
            build_file_item("strategy_source", source),
            build_file_item("diagnostics_renderer", diagnostics, scope="diagnostics"),
        ]
    )
    diagnostics.write_text("FORMAT = 'new'\n", encoding="utf-8")
    changed = build_auto_run_provenance(
        [
            build_file_item("strategy_source", source),
            build_file_item("diagnostics_renderer", diagnostics, scope="diagnostics"),
        ]
    )

    assert changed.selection_fingerprint == original.selection_fingerprint
    assert changed.diagnostics_fingerprint != original.diagnostics_fingerprint


def test_stable_json_inputs_participate_in_selection_fingerprint() -> None:
    original = build_auto_run_provenance([build_json_item("scoring_weights", {"return": 1.0, "drawdown": -0.5})])
    changed = build_auto_run_provenance([build_json_item("scoring_weights", {"return": 1.2, "drawdown": -0.5})])

    assert changed.selection_fingerprint != original.selection_fingerprint


def test_phase_auto_provenance_records_source_artifacts_and_data_contents(tmp_path) -> None:
    source = tmp_path / "source_config.json"
    source.write_text('{"enabled": true}\n', encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "bars.csv").write_text("ts,close\n2026-01-01,1\n", encoding="utf-8")

    provenance = build_phase_auto_provenance(
        "sample",
        repo_root=tmp_path,
        data_dir=data_dir,
        source_artifacts={"source_config": source},
        selection_context={"score_weights": {"return": 1.0}},
    )

    assert any(item.scope == "data" and item.kind == "tree_contents" for item in provenance.items)
    assert any(item.scope == "source_artifact" and item.name == "source_artifact:source_config" for item in provenance.items)


def test_environment_items_do_not_change_selection_fingerprint() -> None:
    original = build_auto_run_provenance(
        [
            build_json_item("scoring_weights", {"return": 1.0}),
            build_json_item("workers", {"max_workers": 1}, scope="environment"),
        ]
    )
    changed = build_auto_run_provenance(
        [
            build_json_item("scoring_weights", {"return": 1.0}),
            build_json_item("workers", {"max_workers": 4}, scope="environment"),
        ]
    )

    assert changed.selection_fingerprint == original.selection_fingerprint
    assert changed.diagnostics_fingerprint != original.diagnostics_fingerprint


def test_drift_checker_flags_missing_saved_metrics_as_incomplete() -> None:
    result = classify_metric_drift(
        {"total_trades": 10, "profit_factor": 1.5},
        {"total_trades": 10, "profit_factor": 1.5, "net_return_pct": 2.0},
        critical_metrics=("total_trades", "profit_factor", "net_return_pct"),
    )

    assert result.status == DRIFT_STATUS_INCOMPLETE_SAVED_METRICS
    assert result.missing_saved_metrics == ("net_return_pct",)


def test_drift_checker_flags_nqdtc_style_recompute_delta_as_selection_stale() -> None:
    result = classify_metric_drift(
        {"total_trades": 89, "net_return_pct": 296.43, "profit_factor": 2.193},
        {"total_trades": 94, "net_return_pct": 262.53, "profit_factor": 1.970},
        critical_metrics=("total_trades", "net_return_pct", "profit_factor"),
    )

    assert result.status == DRIFT_STATUS_SELECTION_STALE
    assert result.deltas == {
        "total_trades": {"saved": 89, "current": 94},
        "net_return_pct": {"saved": 296.43, "current": 262.53},
        "profit_factor": {"saved": 2.193, "current": 1.97},
    }


def test_drift_checker_requires_integer_trade_counts() -> None:
    result = classify_metric_drift(
        {"total_trades": 10},
        {"total_trades": 10.5},
        critical_metrics=("total_trades",),
    )

    assert result.status == DRIFT_STATUS_SELECTION_STALE


def test_phase_runner_rejects_fallback_provenance_on_strict_surfaces(tmp_path) -> None:
    for family, strategy in _STRICT_PROVENANCE_SURFACES:
        manager = RoundManager(family, strategy, base_dir=tmp_path / "output")
        round_dir = manager.get_round_dir(1)
        runner = PhaseRunner(
            plugin=_NoProvenancePlugin(strategy),
            output_dir=round_dir,
            round_manager=manager,
            round_num=1,
        )

        try:
            runner._ensure_round_spec(PhaseState())
            assert False, f"Expected strict provenance failure for {family}/{strategy}."
        except RuntimeError as exc:
            assert "requires complete provenance" in str(exc)


def test_phase_runner_can_validate_previous_lineage_provenance_override(tmp_path) -> None:
    manager = RoundManager("momentum", "sample", base_dir=tmp_path / "output")
    previous = build_auto_run_provenance([build_json_item("selection_inputs", {"round_name": "v4r1"})])
    current = build_auto_run_provenance([build_json_item("selection_inputs", {"round_name": "v5r1"})])

    round_1_dir = manager.get_round_dir(1)
    manager.write_run_summary(
        round_1_dir,
        {"flags.previous": True},
        {"total_trades": 8},
        [1],
        round_num=1,
        provenance=previous,
        provenance_status="complete",
    )
    manager.write_optimized_config(round_1_dir, {"flags.previous": True})
    manager.append_to_manifest(
        1,
        {"flags.previous": True},
        {"total_trades": 8},
        provenance=previous,
        provenance_status="complete",
    )

    round_2_dir = manager.get_round_dir(2)
    plugin = _ProvenancePlugin("sample_v5r1", current, previous_round_provenance=previous)
    runner = PhaseRunner(
        plugin=plugin,
        output_dir=round_2_dir,
        round_manager=manager,
        round_num=2,
    )

    runner._ensure_round_spec(PhaseState())

    assert runner._provenance_validation is not None
    assert runner._provenance_validation.valid
    run_spec = json.loads((round_2_dir / "run_spec.json").read_text(encoding="utf-8"))
    assert run_spec["provenance"]["selection_fingerprint"] == current.selection_fingerprint
    assert run_spec["baseline_mutations"] == {"flags.current": True}
