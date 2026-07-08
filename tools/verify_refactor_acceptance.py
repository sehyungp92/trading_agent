from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "migration_acceptance_report.json"
PLAN = ROOT / "docs" / "contract-first-python-workspace-monorepo-implementation-plan.md"
ROW_IDS = (
    "A00",
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "A17",
)
SCAN_SUFFIXES = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SCAN_FILENAMES = {".dockerignore", ".gitignore"}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "_ref" "erences",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "output",
}
REFERENCE_SCAN_ROOTS = (
    ROOT,
)
REFERENCE_TOKEN = "_ref" "erences"
PROVENANCE_FIELD_RE = re.compile(
    r'["\']?(source_repo|source_reference|archived_source_path)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
)
MONOREPO_PROVENANCE_PREFIXES = (
    ".github/",
    "artifacts/",
    "backtests/baselines/",
    "trading/",
    "contracts/",
    "deployments/",
    "docs/",
    "packages/",
    "tools/",
    "migration_acceptance_report.json",
    "pyproject.toml",
    "uv.lock",
)
CI_LOCKED_WORKSPACE_JOBS = {
    "contracts",
    "baselines",
    "live-configs",
    "decision-parity",
    "optimizer-compatibility",
    "backtest-integrity",
    "deployment-gate",
    "deployment-metadata",
    "docker",
    "strict-refactor-acceptance",
}


@dataclass(frozen=True)
class AcceptanceRow:
    row: str
    gate: str
    commands: tuple[tuple[str, ...], ...] = ()
    evidence_paths: tuple[str, ...] = ()
    skip_reason: str = ""
    timeout_seconds: int = 900


def main() -> int:
    args = _parser().parse_args()
    rows = _row_specs(args)
    records = [_run_row(row) for row in rows if row.row != "A17"]
    records.append(_a17_record(rows, records, strict=args.strict))
    errors = [
        f"{record['row']}: {error}"
        for record in records
        for error in record.get("errors", [])
    ]
    report = {
        "valid": not errors,
        "strict": args.strict,
        "bot": args.bot,
        "rows": records,
        "errors": errors,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify final refactor acceptance evidence.")
    parser.add_argument("--bot", choices=["all", "ibkr", "crypto", "k_stock"], default="all")
    parser.add_argument("--strict", action="store_true")
    return parser


def _row_specs(args: argparse.Namespace) -> list[AcceptanceRow]:
    return [
        AcceptanceRow(
            "A00",
            "Repository and tooling bootstrap",
            (
                (sys.executable, "tools/check_repo_bootstrap.py"),
                (sys.executable, "tools/workspace_lock_check.py"),
                (sys.executable, "tools/detect_affected_images.py", "--explain-only"),
            ),
            ("docs/repo-bootstrap-decisions.md", "pyproject.toml", "uv.lock", ".github/workflows/ci.yml"),
        ),
        AcceptanceRow(
            "A0",
            "Workspace starting-state inventory",
            ((sys.executable, "tools/freeze_optimization_baselines.py", "--inventory-only"),),
            ("docs/migration_inventory.md", "backtests/baselines/baseline_index.json"),
        ),
        AcceptanceRow(
            "A1",
            "Python/package metadata",
            ((sys.executable, "tools/workspace_import_smoke.py", "--all-packages", "--run-commands"),),
            ("pyproject.toml", "uv.lock"),
        ),
        AcceptanceRow(
            "A2",
            "IBKR latest round freeze",
            (
                (sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "ibkr"),
                (sys.executable, "tools/verify_latest_round_no_drift.py", "--bot", "ibkr", "--baseline", "backtests/baselines/baseline_index.json", "--strict"),
            ),
            ("backtests/data_portability_manifest.json", "backtests/baselines/baseline_index.json"),
        ),
        AcceptanceRow(
            "A3",
            "Crypto latest round freeze",
            (
                (sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "crypto"),
                (sys.executable, "tools/verify_latest_round_no_drift.py", "--bot", "crypto", "--baseline", "backtests/baselines/baseline_index.json", "--strict"),
            ),
            ("backtests/data_portability_manifest.json", "backtests/baselines/baseline_index.json"),
        ),
        AcceptanceRow(
            "A4",
            "Crypto portfolio bundle integrity",
            ((sys.executable, "tools/verify_live_config_promotions.py", "--bot", "crypto", "--require-portfolio-bundle"),),
            ("contracts/promotions/crypto", "deployments/crypto/generated/live_config.effective.json"),
        ),
        AcceptanceRow(
            "A5",
            "K-stock latest baseline decision",
            (
                (sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "k_stock"),
                (sys.executable, "tools/verify_latest_round_no_drift.py", "--bot", "k_stock", "--baseline", "backtests/baselines/baseline_index.json", "--strict"),
            ),
            ("backtests/data_portability_manifest.json", "backtests/baselines/baseline_index.json", "backtests/baselines/k_stock"),
        ),
        AcceptanceRow(
            "A6",
            "Live config alignment: IBKR",
            ((sys.executable, "tools/verify_live_config_promotions.py", "--bot", "ibkr", "--require-latest-round", "--require-effective-configs", "--strict"),),
            ("contracts/promotions/ibkr", "deployments/ibkr/generated/strategies.effective.json"),
        ),
        AcceptanceRow(
            "A7",
            "Live config alignment: crypto",
            ((sys.executable, "tools/verify_live_config_promotions.py", "--bot", "crypto", "--require-latest-round", "--require-effective-configs", "--strict"),),
            ("contracts/promotions/crypto", "deployments/crypto/generated/live_config.effective.json"),
        ),
        AcceptanceRow(
            "A8",
            "Live config alignment: K-stock",
            ((sys.executable, "tools/verify_live_config_promotions.py", "--bot", "k_stock", "--require-latest-round", "--require-effective-configs", "--strict"),),
            ("contracts/promotions/k_stock", "deployments/k_stock/generated/olr_kalcb.effective.json"),
        ),
        AcceptanceRow(
            "A9",
            "Strategy plugin contracts",
            ((sys.executable, "tools/verify_strategy_plugin_contracts.py", "--all"),),
            ("contracts/strategy_plugins",),
        ),
        AcceptanceRow(
            "A10",
            "Decision parity",
            ((sys.executable, "tools/run_decision_parity_matrix.py", "--promoted-only"),),
            ("artifacts/validation/decision_parity_matrix",),
        ),
        AcceptanceRow(
            "A11",
            "Assistant workspace checks",
            ((sys.executable, "tools/run_workspace_checks.py", "deployment-gate"),),
            ("artifacts/validation/backtest_integrity/validation_matrix",),
            timeout_seconds=1800,
        ),
        AcceptanceRow(
            "A12",
            "Shared contracts compatibility",
            (
                (sys.executable, "-m", "trading_contracts.cli", "validate", "--all-known-reference-artifacts", "--repo-root", "."),
                (sys.executable, "tools/verify_dependency_boundaries.py"),
            ),
            ("packages/trading_contracts", "contracts/schemas"),
        ),
        AcceptanceRow(
            "A13",
            "Optimizer compatibility",
            ((sys.executable, "tools/verify_optimizer_compatibility.py", "--bot", args.bot, "--fixture-set", "smoke"),),
            ("artifacts/validation/optimizer_compatibility",),
        ),
        AcceptanceRow(
            "A14",
            "Backtest integrity invariants",
            ((sys.executable, "tools/run_backtest_integrity_matrix.py", "--promoted-only"),),
            ("artifacts/validation/backtest_data_portability_report.json", "artifacts/validation/backtest_integrity/invariant_report.json", "artifacts/validation/backtest_integrity/thin_adapter_audit.json"),
            timeout_seconds=1200,
        ),
        AcceptanceRow(
            "A15",
            "Deployment image build",
            ((sys.executable, "tools/build_bot_image.py", "--bot", args.bot, "--emit-dependency-reports"),),
            ("deployments/ibkr/generated/dependency_report.json", "deployments/crypto/generated/dependency_report.json", "deployments/k_stock/generated/dependency_report.json"),
            timeout_seconds=3600,
        ),
        AcceptanceRow(
            "A16",
            "Deployment metadata",
            ((sys.executable, "tools/verify_deployment_metadata.py", "--bot", args.bot),),
            ("deployments", "contracts/strategy_plugins"),
        ),
        AcceptanceRow(
            "A17",
            "Final refactor acceptance",
            (
                (sys.executable, "tools/verify_backtest_data_portability.py", "--bot", "all"),
                (sys.executable, "tools/check_workspace_structure.py", "--layout", "final"),
                (sys.executable, "tools/verify_cutover_plan.py"),
                (sys.executable, "tools/verify_operational_deployment_evidence.py"),
            ),
            (
                "migration_acceptance_report.json",
                "backtests/data_portability_manifest.json",
                "artifacts/validation/backtest_data_portability_report.json",
                "deployments/cutover_plan.json",
                "deployments/operational_evidence.json",
            ),
        ),
    ]


def _run_row(row: AcceptanceRow) -> dict[str, Any]:
    if row.skip_reason:
        return _record(row, "skipped", [], [], [])
    command_records = [_run(command, timeout=row.timeout_seconds) for command in row.commands]
    command_errors = [
        " ".join(command["command"])
        for command in command_records
        if command["returncode"] != 0
    ]
    evidence_errors = _missing_evidence(row.evidence_paths)
    errors = [*command_errors, *evidence_errors]
    return _record(row, "pass" if not errors else "fail", command_records, evidence_errors, errors)


def _a17_record(rows: list[AcceptanceRow], prior_records: list[dict[str, Any]], *, strict: bool) -> dict[str, Any]:
    row = next(item for item in rows if item.row == "A17")
    command_records = [_run(command, timeout=row.timeout_seconds) for command in row.commands]
    command_errors = [
        " ".join(command["command"])
        for command in command_records
        if command["returncode"] != 0
    ]
    row_ids = {record["row"] for record in prior_records}
    missing_rows = [row_id for row_id in ROW_IDS[:-1] if row_id not in row_ids]
    failing_rows = [record["row"] for record in prior_records if record["status"] != "pass"]
    reference_failures, allowed_mentions = _reference_scan()
    checklist_failures = _checklist_failures() if strict else []
    errors = [
        *command_errors,
        *[f"missing acceptance row {row_id}" for row_id in missing_rows],
        *[f"blocked by failing row {row_id}" for row_id in failing_rows],
        *reference_failures,
        *checklist_failures,
        *(_ci_locked_workspace_failures() if strict else []),
        *_missing_evidence(row.evidence_paths, allow_self=True),
    ]
    record = _record(row, "pass" if not errors else "fail", command_records, [], errors)
    record["reference_failures"] = reference_failures
    record["allowed_reference_mentions"] = allowed_mentions
    record["blocked_by_rows"] = failing_rows
    record["checklist_failures"] = checklist_failures
    return record


def _record(
    row: AcceptanceRow,
    status: str,
    command_records: list[dict[str, Any]],
    evidence_errors: list[str],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "row": row.row,
        "gate": row.gate,
        "status": status,
        "commands": command_records,
        "evidence_paths": list(row.evidence_paths),
        "missing_evidence": evidence_errors,
        "skip_reason": row.skip_reason,
        "errors": errors,
    }


def _run(command: tuple[str, ...], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": list(command),
            "returncode": -1,
            "timeout_seconds": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-20:],
            "stderr_tail": (exc.stderr or "").splitlines()[-20:],
        }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-20:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def _env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(path) for path in sorted((ROOT / "packages").glob("*/src")) if path.exists()]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _missing_evidence(paths: tuple[str, ...], *, allow_self: bool = False) -> list[str]:
    errors = []
    for raw_path in paths:
        path = ROOT / raw_path
        if allow_self and path == REPORT:
            continue
        if not path.exists():
            errors.append(f"missing evidence path {raw_path}")
    return errors


def _reference_scan() -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    for path in _scan_files():
        relative = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        if REFERENCE_TOKEN in text:
            for line_number, line in enumerate(text.splitlines(), start=1):
                if REFERENCE_TOKEN in line:
                    failures.append(f"{relative}:{line_number} contains legacy snapshot path")
        failures.extend(_legacy_provenance_failures(relative, text))
    return failures, []


def _scan_files() -> list[Path]:
    paths: list[Path] = []
    for root in REFERENCE_SCAN_ROOTS:
        if root.is_file():
            paths.append(root)
            continue
        if not root.exists():
            continue
        for current, dirnames, filenames in os.walk(root):
            current_path = Path(current)
            if set(current_path.relative_to(root).parts) & SKIP_PARTS:
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name not in SKIP_PARTS]
            for filename in filenames:
                path = current_path / filename
                if path.suffix in SCAN_SUFFIXES or path.name in SCAN_FILENAMES:
                    paths.append(path)
    return sorted(paths)


def _legacy_provenance_failures(relative: str, text: str) -> list[str]:
    failures: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in PROVENANCE_FIELD_RE.finditer(line):
            key, value = match.groups()
            if not _is_monorepo_provenance_path(value):
                failures.append(
                    f"{relative}:{line_number} {key} points outside monorepo-owned evidence: {value}"
                )
    return failures


def _is_monorepo_provenance_path(value: str) -> bool:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return True
    if REFERENCE_TOKEN in normalized or "://" in normalized or normalized.startswith(("/", "../")):
        return False
    return normalized.startswith(MONOREPO_PROVENANCE_PREFIXES)


def _checklist_failures() -> list[str]:
    if not PLAN.exists():
        return ["missing finite checklist plan document"]
    failures: list[str] = []
    in_checklist = False
    for raw_line in PLAN.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "## Finite Implementation Checklist":
            in_checklist = True
            continue
        if in_checklist and line.startswith("## "):
            break
        if in_checklist and line.startswith("- [ ]"):
            failures.append("open finite-checklist item: " + line.removeprefix("- [ ]").strip())
    return failures


def _ci_locked_workspace_failures() -> list[str]:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.exists():
        return ["missing CI workflow for final acceptance gates"]
    text = workflow.read_text(encoding="utf-8")
    failures: list[str] = []
    for job in sorted(CI_LOCKED_WORKSPACE_JOBS):
        block = _ci_job_block(text, job)
        if not block:
            failures.append(f"CI job missing: {job}")
            continue
        if "uv sync --frozen --all-packages" not in block:
            failures.append(f"CI job {job} does not sync the locked workspace")
        raw_python_tool_runs = [
            line.strip()
            for line in block.splitlines()
            if re.search(r"run:\s*python\s+tools/", line)
        ]
        if raw_python_tool_runs:
            failures.append(f"CI job {job} runs tools without uv: {raw_python_tool_runs}")
    return failures


def _ci_job_block(text: str, job: str) -> str:
    pattern = re.compile(rf"^  {re.escape(job)}:\n(?P<body>(?:    .*\n|      .*\n|$)+)", re.MULTILINE)
    match = pattern.search(text)
    return match.group(0) if match else ""


if __name__ == "__main__":
    raise SystemExit(main())
