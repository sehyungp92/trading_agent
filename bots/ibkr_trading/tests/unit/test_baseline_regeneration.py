from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from backtests.shared.parity.baseline_regeneration import (
    build_regeneration_command,
    regenerate_manifest_entry,
    remap_regeneration_arguments,
    verify_manifest_regeneration,
)
from backtests.shared.auto.phase_state import PhaseState
from backtests.stock.auto.alcb import plugin as alcb_plugin_mod
from backtests.stock.auto.alcb.run_final_diagnostics import (
    _hydrate_final_phase_runtime_context,
)
from backtests.shared.parity.diagnostic_baselines import sha256_file
import pytest


def test_remap_regeneration_arguments_only_rewrites_output_targets(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sandbox = tmp_path / "sandbox"
    root.mkdir()
    sandbox.mkdir()

    arguments = [
        "--phase-state",
        "backtests/stock/auto/iaric/output_v4r1/phase_state.json",
        "--output",
        "backtests/stock/auto/iaric/output_v4r1/report.txt",
        "--summary-json",
        "backtests/stock/auto/iaric/output_v4r1/report.json",
        "--output-dir",
        "backtests/stock/auto/alcb/output_targeted_entry_repair_v2",
    ]

    remapped = remap_regeneration_arguments(arguments, sandbox_root=sandbox, root=root)

    assert remapped[1] == "backtests/stock/auto/iaric/output_v4r1/phase_state.json"
    assert remapped[3] == str(sandbox / "backtests/stock/auto/iaric/output_v4r1/report.txt")
    assert remapped[5] == str(sandbox / "backtests/stock/auto/iaric/output_v4r1/report.json")
    assert remapped[7] == str(sandbox / "backtests/stock/auto/alcb/output_targeted_entry_repair_v2")


def test_remap_regeneration_arguments_supports_inline_output_assignments(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sandbox = tmp_path / "sandbox"
    root.mkdir()
    sandbox.mkdir()

    remapped = remap_regeneration_arguments(
        [
            "--output=artifacts/output.txt",
            "--summary-json=artifacts/summary.json",
            "--output-dir=artifacts/output_dir",
        ],
        sandbox_root=sandbox,
        root=root,
    )

    assert remapped == [
        f"--output={sandbox / 'artifacts/output.txt'}",
        f"--summary-json={sandbox / 'artifacts/summary.json'}",
        f"--output-dir={sandbox / 'artifacts/output_dir'}",
    ]


def test_build_regeneration_command_handles_python_file_and_module(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sandbox = tmp_path / "sandbox"
    (root / "tools").mkdir(parents=True)
    sandbox.mkdir()

    file_entry = {
        "id": "file",
        "artifact_path": "artifacts/output.txt",
        "parser_kind": "momentum_performance_report",
        "sha256": "unused",
        "expected_metrics": {"total_trades": 1.0},
        "regeneration": {
            "executor": "python_file",
            "entrypoint": "tools/write_artifact.py",
            "arguments": ["--output", "artifacts/output.txt"],
            "expected_output": "artifacts/output.txt",
        },
    }
    module_entry = {
        "id": "module",
        "artifact_path": "artifacts/output.txt",
        "parser_kind": "momentum_performance_report",
        "sha256": "unused",
        "expected_metrics": {"total_trades": 1.0},
        "regeneration": {
            "executor": "python_module",
            "entrypoint": "tools.writer",
            "arguments": ["--output", "artifacts/output.txt"],
            "expected_output": "artifacts/output.txt",
        },
    }

    file_command = build_regeneration_command(file_entry, root=root, sandbox_root=sandbox, python_executable="python")
    module_command = build_regeneration_command(module_entry, root=root, sandbox_root=sandbox, python_executable="python")

    assert file_command[:2] == ["python", str(root / "tools/write_artifact.py")]
    assert file_command[-1] == str(sandbox / "artifacts/output.txt")
    assert module_command[:3] == ["python", "-m", "tools.writer"]
    assert module_command[-1] == str(sandbox / "artifacts/output.txt")


def test_regenerate_manifest_entry_executes_in_sandbox_and_validates_hash_and_metrics(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sandbox = tmp_path / "sandbox"
    script_path = root / "tools" / "write_artifact.py"
    script_path.parent.mkdir(parents=True)
    sandbox.mkdir()

    script_path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "out = Path(args.output)",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(",
                "    'PERFORMANCE SUMMARY\\n'",
                "    'Total trades: 10\\n'",
                "    'Win rate: 55.0\\n'",
                "    'Profit factor: 1.25\\n'",
                "    'Net profit: 1234.5\\n'",
                "    'Max drawdown: 4.5\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entry = {
        "id": "demo_momentum_report",
        "artifact_path": "artifacts/output.txt",
        "parser_kind": "momentum_performance_report",
        "expected_metrics": {
            "total_trades": 10.0,
            "win_rate_pct": 55.0,
            "profit_factor": 1.25,
            "net_profit": 1234.5,
            "max_drawdown_pct": 4.5,
        },
        "regeneration": {
            "executor": "python_file",
            "entrypoint": "tools/write_artifact.py",
            "arguments": ["--output", "artifacts/output.txt"],
            "expected_output": "artifacts/output.txt",
        },
    }

    manifest_entry = dict(entry)
    command = build_regeneration_command(
        {
            **entry,
            "sha256": "placeholder",
        },
        root=root,
        sandbox_root=sandbox,
        python_executable=sys.executable,
    )
    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    manifest_entry["sha256"] = sha256_file(sandbox / "artifacts/output.txt")

    verified = regenerate_manifest_entry(
        manifest_entry,
        root=root,
        sandbox_root=sandbox,
        python_executable=sys.executable,
    )

    assert verified.entry_id == "demo_momentum_report"
    assert verified.returncode == 0
    assert verified.metrics == manifest_entry["expected_metrics"]
    assert Path(verified.artifact_path).exists()
    assert json.loads(json.dumps(verified.metrics))["net_profit"] == 1234.5


def test_verify_manifest_regeneration_raises_for_unknown_artifact_ids(tmp_path: Path) -> None:
    manifest = {
        "artifacts": [
            {
                "id": "known",
                "artifact_path": "artifacts/output.txt",
                "parser_kind": "momentum_performance_report",
                "sha256": "unused",
                "expected_metrics": {"total_trades": 1.0},
                "regeneration": {
                    "executor": "python_file",
                    "entrypoint": "tools/write_artifact.py",
                    "arguments": ["--output", "artifacts/output.txt"],
                    "expected_output": "artifacts/output.txt",
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="Unknown baseline artifact id"):
        verify_manifest_regeneration(
            manifest=manifest,
            root=tmp_path,
            sandbox_root=tmp_path / "sandbox",
            artifact_ids=["missing"],
        )


def test_verify_manifest_regeneration_uses_isolated_sandboxes_per_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = {
        "artifacts": [
            {
                "id": "first",
                "artifact_path": "artifacts/first.txt",
                "parser_kind": "momentum_performance_report",
                "sha256": "unused",
                "expected_metrics": {"total_trades": 1.0},
                "regeneration": {
                    "executor": "python_file",
                    "entrypoint": "tools/write_artifact.py",
                    "arguments": ["--output", "artifacts/first.txt"],
                    "expected_output": "artifacts/first.txt",
                },
            },
            {
                "id": "second",
                "artifact_path": "artifacts/second.txt",
                "parser_kind": "momentum_performance_report",
                "sha256": "unused",
                "expected_metrics": {"total_trades": 2.0},
                "regeneration": {
                    "executor": "python_file",
                    "entrypoint": "tools/write_artifact.py",
                    "arguments": ["--output", "artifacts/second.txt"],
                    "expected_output": "artifacts/second.txt",
                },
            },
        ]
    }
    seen_sandboxes: list[Path] = []

    def _fake_regenerate(entry, *, root, sandbox_root, python_executable=None, timeout_seconds=3600):
        seen_sandboxes.append(Path(sandbox_root))
        return {
            "id": entry["id"],
            "sandbox_root": str(sandbox_root),
        }

    monkeypatch.setattr(
        "backtests.shared.parity.baseline_regeneration.regenerate_manifest_entry",
        _fake_regenerate,
    )

    results = verify_manifest_regeneration(
        manifest=manifest,
        root=tmp_path,
        sandbox_root=tmp_path / "sandbox",
    )

    assert results == [
        {"id": "first", "sandbox_root": str(tmp_path / "sandbox" / "first")},
        {"id": "second", "sandbox_root": str(tmp_path / "sandbox" / "second")},
    ]
    assert seen_sandboxes == [tmp_path / "sandbox" / "first", tmp_path / "sandbox" / "second"]


def test_alcb_final_diagnostics_restores_phase_context_after_replay_bundle_warm() -> None:
    state = PhaseState(
        completed_phases=[5],
        phase_results={5: {"final_metrics": {"net_profit": 8749.17, "profit_factor": 1.749}}},
        phase_gate_results={
            5: {
                "criteria": [
                    {"name": "net_profit", "target": 8486.6949},
                    {"name": "profit_factor", "target": 1.7136},
                ]
            }
        },
    )

    class FakePlugin:
        num_phases = 5

        def __init__(self) -> None:
            self._phase_runtime_context: dict[int, dict] = {}
            self._warmed = False

        def _replay_bundle(self):
            if not self._warmed:
                self._phase_runtime_context.clear()
                self._warmed = True
            return SimpleNamespace()

        def build_end_of_round_artifacts(self, current_state):
            del current_state
            self._replay_bundle()
            return self._phase_runtime_context.get(5, {})

    plugin = FakePlugin()

    final_phase = _hydrate_final_phase_runtime_context(plugin, state)
    restored_context = plugin.build_end_of_round_artifacts(state)

    assert final_phase == 5
    assert restored_context == {
        "base_metrics": {"net_profit": 8749.17, "profit_factor": 1.749},
        "hard_rejects": {
            "min_net_profit": 8486.6949,
            "min_pf": 1.7136,
        },
    }


def test_alcb_render_final_diagnostics_text_skips_base_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = alcb_plugin_mod.ALCBP16Plugin(tmp_path, max_workers=1)
    state = SimpleNamespace(completed_phases=[5], cumulative_mutations={"param_overrides.example": 1.0})
    run_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        alcb_plugin_mod,
        "greedy_result_from_state",
        lambda state_obj, phase, final_metrics: SimpleNamespace(
            final_mutations=dict(state_obj.cumulative_mutations),
            phase=phase,
            final_metrics=dict(final_metrics),
        ),
    )

    def fake_run_config(mutations, **kwargs):
        run_calls.append(dict(mutations))
        return {
            "metrics": {"net_profit": 1.0},
            "trades": [],
            "config": SimpleNamespace(param_overrides={}),
        }

    monkeypatch.setattr(plugin, "_run_config", fake_run_config)
    monkeypatch.setattr(
        plugin,
        "run_enhanced_diagnostics",
        lambda phase, state_obj, metrics, greedy: (
            f"phase={phase};"
            f"mutations={sorted(greedy.final_mutations.items())};"
            f"net_profit={metrics['net_profit']}"
        ),
    )

    rendered = plugin.render_final_diagnostics_text(state)

    assert rendered == "phase=5;mutations=[('param_overrides.example', 1.0)];net_profit=1.0"
    assert run_calls == [state.cumulative_mutations]
