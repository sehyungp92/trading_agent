"""Tests for workspace-level deployment check helpers."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


def _load_workspace_checks():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "run_workspace_checks.py"
    spec = importlib.util.spec_from_file_location("run_workspace_checks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_structure_checks():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "check_workspace_structure.py"
    spec = importlib.util.spec_from_file_location("check_workspace_structure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deployment_package_excludes_runtime_data_by_default():
    checks = _load_workspace_checks()

    assert checks._is_deployment_package_excluded(
        "packages/trading_assistant_data/data/reference/bundle.json"
    )
    assert checks._is_deployment_package_excluded(
        "packages/trading_assistant/runs/daily/run.json"
    )
    assert checks._is_deployment_package_excluded(
        "packages/trading_assistant_data/data/raw/ticks.parquet"
    )
    assert checks._is_deployment_package_excluded(
        "bots/crypto_trader/output/portfolio/rounds_manifest.json"
    )
    assert checks._is_deployment_package_excluded(
        "bots/k_stock_trader/data/backtests/output/kalcb/rounds_manifest.json"
    )
    assert checks._is_deployment_package_excluded(
        "bots/new_bot/data/backtests/output/latest/artifact.json"
    )
    assert not checks._is_deployment_package_excluded(
        "packages/trading_assistant/src/trading_assistant/orchestrator/app.py"
    )


def test_deployment_package_can_intentionally_include_data_tree():
    checks = _load_workspace_checks()

    assert not checks._is_deployment_package_excluded(
        "packages/trading_assistant_data/data/reference/bundle.json",
        include_data=True,
    )
    assert checks._is_deployment_package_excluded(
        "packages/trading_assistant_data/data/raw/ticks.parquet",
        include_data=True,
    )


def test_ruff_fb_fails_on_unapproved_regression(monkeypatch):
    checks = _load_workspace_checks()
    baseline_issue = {
        "filename": "pkg/a.py",
        "code": "F401",
        "message": "unused import",
    }
    new_issue = {
        "filename": "pkg/b.py",
        "code": "F821",
        "message": "Undefined name `oops`",
    }

    monkeypatch.setattr(checks, "_collect_ruff_fb_issues", lambda: (0, [baseline_issue, new_issue]))
    monkeypatch.setattr(checks, "_load_ruff_fb_baseline", lambda: [baseline_issue])

    assert checks.ruff_fb() == 1


def test_validation_matrix_gate_fails_on_approval_gaps():
    checks = _load_workspace_checks()

    errors = checks._validation_matrix_approval_gate({
        "ok": True,
        "approval_grade_validation_complete": False,
        "approval_remaining_gaps": [{"reason": "strategy_contract_not_approval_ready"}],
    })

    assert errors == ["approval-grade validation incomplete: 1 remaining gap(s)"]


def test_run_timed_timeout_kills_child_process(tmp_path: Path):
    checks = _load_workspace_checks()
    marker = tmp_path / "child-survived.txt"
    child_code = (
        "import pathlib, sys, time; "
        "time.sleep(2); "
        "pathlib.Path(sys.argv[1]).write_text('alive', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, {str(marker)!r}]); "
        "time.sleep(10)"
    )
    command = checks.CheckCommand(
        [sys.executable, "-c", parent_code],
        tmp_path,
        1,
    )

    assert checks._run_timed(command, tier_name="timeout-test") == 124
    time.sleep(2.5)
    assert not marker.exists()


def test_parallel_tier_returns_first_nonzero_status(tmp_path: Path, monkeypatch):
    checks = _load_workspace_checks()
    codes_by_name = {"first": 0, "second": 7, "third": 0}

    def fake_run_timed(command, *, tier_name: str):
        assert tier_name == "parallel-test"
        return codes_by_name[command.args[0]]

    monkeypatch.setattr(checks, "_run_timed", fake_run_timed)
    tier = checks.CheckTier(
        name="parallel-test",
        commands=(
            checks.CheckCommand(["first"], tmp_path, 1),
            checks.CheckCommand(["second"], tmp_path, 1),
            checks.CheckCommand(["third"], tmp_path, 1),
        ),
        parallelism=2,
    )

    assert checks._run_tier_parallel(tier) == 7


def test_structure_guard_rejects_utf8_bom(tmp_path: Path, monkeypatch):
    checks = _load_structure_checks()
    source = tmp_path / "bom.py"
    source.write_bytes(b"\xef\xbb\xbfprint('bad')\n")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    errors: list[str] = []

    assert checks._read_python_source(source, errors) is None
    assert any("UTF-8 BOM" in error for error in errors)


def test_structure_guard_reports_module_size_watch(tmp_path: Path, monkeypatch):
    checks = _load_structure_checks()
    watched = tmp_path / "pkg" / "wide.py"
    watched.parent.mkdir(parents=True)
    watched.write_text("x = 1\nx = 2\n", encoding="utf-8")
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(checks, "MODULE_SIZE_LIMITS", {"pkg/wide.py": 1})
    errors: list[str] = []

    checks._check_module_size_watch(errors)

    assert any("module-size watch limit" in error for error in errors)
