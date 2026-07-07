from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path

from trading_assistant_backtest import monthly
from trading_assistant_backtest.monthly_execution import (
    MonthlyExecution,
    optimizer_sequence,
    repair_sequence,
    runner,
    selection_oos,
    structural_registry,
)


def test_monthly_execution_owns_manifest_execution(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_impl(manifest_path: Path, *, planner_mode: str) -> int:
        seen["manifest_path"] = manifest_path
        seen["planner_mode"] = planner_mode
        return 7

    monkeypatch.setattr(
        "trading_assistant_backtest.monthly_execution.runner.run_manifest_impl",
        fake_impl,
    )

    assert (
        MonthlyExecution().run_manifest(
            Path("run_manifest.json"),
            planner_mode="deterministic",
        )
        == 7
    )
    assert seen == {
        "manifest_path": Path("run_manifest.json"),
        "planner_mode": "deterministic",
    }


def test_monthly_execution_is_split_by_execution_concern() -> None:
    source = inspect.getsource(runner)
    monthly_source = inspect.getsource(monthly)
    optimizer_source = inspect.getsource(optimizer_sequence)

    assert "build_replay_context(" in source
    assert "write_required_artifacts(" in source
    assert "write_optimizer_artifacts(" in source
    assert "write_structural_placeholders(" in source
    assert "stdout_summary(" in source
    assert "MonthlyExecution().run_manifest(" in monthly_source
    assert "load_manifest(" not in monthly_source
    assert "ArtifactWriter(" not in monthly_source
    assert "run_repair_sequence(" in optimizer_source
    assert "def _load_accepted_mutation_chain(" not in optimizer_source
    assert "def run_repair_sequence(" in inspect.getsource(repair_sequence)
    assert "def selection_oos_evaluation(" in inspect.getsource(selection_oos)


def test_monthly_cli_help_does_not_import_execution_stack() -> None:
    code = (
        "import sys; "
        "from trading_assistant_backtest import monthly; "
        "print('runner_loaded_before_help=', "
        "'trading_assistant_backtest.monthly_execution.runner' in sys.modules); "
        "monthly.main(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    assert "runner_loaded_before_help= False" in result.stdout
    assert "Run a monthly trading-assistant backtest manifest." in result.stdout


def test_monthly_execution_structural_registries_are_single_sourced() -> None:
    root = Path(inspect.getfile(structural_registry)).parent
    constants = (
        "STRUCTURAL_PARITY_BUILDERS",
        "BRIDGE_IDS_BY_SCOPE",
        "BRIDGE_ID_BY_PLUGIN_ID",
    )

    for constant in constants:
        assignment = re.compile(rf"^{constant}(?::[^=]+)?\s=", re.MULTILINE)
        owners = [
            path.name
            for path in sorted(root.glob("*.py"))
            if assignment.search(path.read_text(encoding="utf-8"))
        ]
        assert owners == ["structural_registry.py"]

    for path in root.glob("*.py"):
        if path.name == "structural_registry.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert "_STRUCTURAL_PARITY_BUILDERS" not in source
        assert "_BRIDGE_IDS_BY_SCOPE" not in source
        assert "_BRIDGE_ID_BY_PLUGIN_ID" not in source
