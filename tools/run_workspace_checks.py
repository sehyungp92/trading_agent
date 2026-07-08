from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOYMENT_MAX_BYTES = 300 * 1024 * 1024
RUFF_FB_BASELINE = ROOT / "tools" / "ruff_fb_baseline.json"
RUFF_FB_TARGETS = (
    "packages/trading_assistant/src",
    "packages/trading_assistant/tests",
    "packages/trading_assistant_data/src",
    "packages/trading_assistant_data/tests",
    "packages/trading_assistant_backtest/src",
    "packages/trading_assistant_backtest/backtests",
    "packages/trading_assistant_backtest/tests",
    "tools",
)
_DEPLOYMENT_EXCLUDE_PREFIXES = (
    ".git/",
    ".agents/",
    ".claude/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "artifacts/",
    "build/",
    "data/data/",
    "dist/",
    "htmlcov/",
    "packages/trading_assistant/.assistant/",
    "packages/trading_assistant/.claude/",
    "packages/trading_assistant/backtest_artifacts/",
    "packages/trading_assistant/data/",
    "packages/trading_assistant/runs/",
    "packages/trading_assistant_data/data/",
    "packages/trading_assistant_backtest/artifacts/",
    "venv/",
)
_DEPLOYMENT_EXCLUDE_SUFFIXES = (
    ".db",
    ".log",
    ".parquet",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
)


@dataclass(frozen=True)
class CheckCommand:
    args: Sequence[str]
    cwd: Path
    timeout_seconds: int
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class CheckTier:
    name: str
    commands: Sequence[CheckCommand]
    optional_env_var: str = ""
    parallelism: int = 1


def _workspace(name: str) -> Path:
    final_path = ROOT / "packages" / name
    if final_path.is_dir():
        return final_path
    raise FileNotFoundError(f"missing final package workspace: {final_path}")


def _run(args: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"+ ({cwd.relative_to(ROOT).as_posix()}) {' '.join(args)}")
    completed = subprocess.run(args, cwd=cwd, env=env, check=False)
    return completed.returncode


def _run_json_summary(
    args: Sequence[str],
    *,
    cwd: Path,
    summary,
    env: dict[str, str] | None = None,
    payload_gate: Callable[[dict], list[str]] | None = None,
) -> int:
    print(f"+ ({cwd.relative_to(ROOT).as_posix()}) {' '.join(args)}")
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
            print(summary(payload))
        except Exception:
            print(completed.stdout.rstrip())
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.returncode == 0 and payload_gate and isinstance(payload, dict):
        gate_errors = payload_gate(payload)
        if gate_errors:
            for error in gate_errors:
                print(error, file=sys.stderr)
            return 1
    return completed.returncode


def _normalise_ruff_filename(filename: str) -> str:
    path = Path(filename)
    try:
        if path.is_absolute():
            return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return _normalise_relative_path(filename)
    return _normalise_relative_path(filename)


def _normalise_ruff_issue(issue: dict) -> dict:
    location = issue.get("location") or {}
    return {
        "filename": _normalise_ruff_filename(str(issue.get("filename", ""))),
        "code": str(issue.get("code", "")),
        "message": str(issue.get("message", "")),
        "row": int(location.get("row") or 0),
        "column": int(location.get("column") or 0),
    }


def _ruff_issue_key(issue: dict) -> tuple[str, str, str]:
    return (
        str(issue.get("filename", "")),
        str(issue.get("code", "")),
        str(issue.get("message", "")),
    )


def _load_ruff_fb_baseline() -> list[dict]:
    if not RUFF_FB_BASELINE.exists():
        return []
    try:
        payload = json.loads(RUFF_FB_BASELINE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    issues = payload.get("issues", []) if isinstance(payload, dict) else []
    return [issue for issue in issues if isinstance(issue, dict)]


def _collect_ruff_fb_issues() -> tuple[int, list[dict]]:
    args = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "F,B",
        "--output-format",
        "json",
        *RUFF_FB_TARGETS,
    ]
    print(f"+ ({_cwd_label(ROOT)}) {_command_text(args)}")
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)
    try:
        raw_issues = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError:
        if completed.stdout.strip():
            print(completed.stdout.rstrip())
        return completed.returncode or 2, []
    if completed.returncode not in (0, 1):
        return completed.returncode, []
    issues = [_normalise_ruff_issue(issue) for issue in raw_issues if isinstance(issue, dict)]
    issues.sort(key=lambda item: (item["filename"], item["code"], item["message"], item["row"], item["column"]))
    return 0, issues


def ruff_fb(*, update_baseline: bool = False) -> int:
    code, issues = _collect_ruff_fb_issues()
    if code != 0:
        return code
    if update_baseline:
        payload = {
            "description": (
                "Approved legacy Ruff F/B baseline. The deployment gate fails "
                "when current findings exceed these filename/code/message counts."
            ),
            "command": "python tools/run_workspace_checks.py ruff-fb",
            "select": ["F", "B"],
            "targets": list(RUFF_FB_TARGETS),
            "issues": issues,
        }
        RUFF_FB_BASELINE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"ruff_fb_baseline_updated={RUFF_FB_BASELINE.relative_to(ROOT).as_posix()}")
        print(f"ruff_fb_approved_issues={len(issues)}")
        return 0
    if not issues:
        print("ruff_fb_ok=true")
        print("ruff_fb_issues=0")
        return 0

    baseline = _load_ruff_fb_baseline()
    if not baseline:
        print(
            "ruff F/B findings exist and no approved baseline is present; "
            "run with --update-baseline only after review",
            file=sys.stderr,
        )
        return 1
    current_counts = Counter(_ruff_issue_key(issue) for issue in issues)
    baseline_counts = Counter(_ruff_issue_key(issue) for issue in baseline)
    regressions = current_counts - baseline_counts
    if regressions:
        print("ruff F/B regressions beyond approved baseline:", file=sys.stderr)
        shown = 0
        for filename, code_name, message in sorted(regressions):
            print(f"{filename}: {code_name} {message}", file=sys.stderr)
            shown += 1
            if shown >= 20:
                remaining = len(regressions) - shown
                if remaining > 0:
                    print(f"... {remaining} more regression keys", file=sys.stderr)
                break
        return 1

    resolved = sum((baseline_counts - current_counts).values())
    print("ruff_fb_ok=true")
    print(f"ruff_fb_current_issues={len(issues)}")
    print(f"ruff_fb_baseline_issues={len(baseline)}")
    print(f"ruff_fb_resolved_since_baseline={resolved}")
    return 0


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    workspace_paths = [
        str(ROOT / "packages" / "trading_contracts" / "src"),
        str(ROOT / "packages" / "trading_config" / "src"),
        str(ROOT / "packages" / "trading_assistant_backtest" / "src"),
        str(ROOT / "packages" / "trading_assistant_data" / "src"),
        str(ROOT / "packages" / "trading_assistant" / "src"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join([*workspace_paths, existing] if existing else workspace_paths)
    return env


def _run_many(commands: Sequence[tuple[Sequence[str], Path, dict[str, str] | None]]) -> int:
    for args, cwd, env in commands:
        code = _run(args, cwd=cwd, env=env)
        if code != 0:
            return code
    return 0


def _command_text(args: Sequence[str]) -> str:
    return " ".join(str(arg) for arg in args)


def _cwd_label(cwd: Path) -> str:
    try:
        return cwd.relative_to(ROOT).as_posix()
    except ValueError:
        return str(cwd)


def _popen_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()


def _run_timed(command: CheckCommand, *, tier_name: str) -> int:
    print(
        f"[{tier_name}] + ({_cwd_label(command.cwd)}) {_command_text(command.args)}",
        flush=True,
    )
    started = monotonic()
    process = subprocess.Popen(
        list(command.args),
        cwd=command.cwd,
        env=command.env,
        **_popen_kwargs(),
    )
    timed_out = False
    try:
        code = process.wait(timeout=command.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(process)
        try:
            process.wait(timeout=5)
        except Exception:
            pass
        code = 124
    duration = monotonic() - started
    status = f"timeout after {command.timeout_seconds}s" if timed_out else f"exit={code}"
    print(
        f"[{tier_name}] {status}; duration={duration:.1f}s; command={_command_text(command.args)}",
        flush=True,
    )
    return code


def _run_tier_parallel(tier: CheckTier) -> int:
    results: list[int | None] = [None] * len(tier.commands)
    semaphore = threading.Semaphore(max(1, tier.parallelism))

    def _worker(index: int, command: CheckCommand) -> None:
        with semaphore:
            results[index] = _run_timed(command, tier_name=tier.name)

    threads = [
        threading.Thread(target=_worker, args=(index, command), daemon=True)
        for index, command in enumerate(tier.commands)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for code in results:
        if code:
            return int(code)
    return 0


def _run_tiers(tiers: Sequence[CheckTier]) -> int:
    for tier in tiers:
        if tier.optional_env_var and os.environ.get(tier.optional_env_var, "").lower() not in {"1", "true", "yes"}:
            print(f"[{tier.name}] skipped; set {tier.optional_env_var}=1 to run")
            continue
        tier_started = monotonic()
        print(f"[{tier.name}] starting")
        if tier.parallelism > 1:
            code = _run_tier_parallel(tier)
            if code != 0:
                print(f"[{tier.name}] failed after {monotonic() - tier_started:.1f}s", flush=True)
                return code
        else:
            for command in tier.commands:
                code = _run_timed(command, tier_name=tier.name)
                if code != 0:
                    print(
                        f"[{tier.name}] failed after {monotonic() - tier_started:.1f}s",
                        flush=True,
                    )
                    return code
        print(f"[{tier.name}] passed in {monotonic() - tier_started:.1f}s", flush=True)
    return 0


def _normalise_relative_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _is_bot_local_output_path(rel_path: str) -> bool:
    parts = rel_path.split("/")
    if len(parts) >= 3 and parts[0] == "bots" and parts[2] == "output":
        return True
    if len(parts) >= 4 and parts[0] == "bots" and parts[2:4] == ["backtests", "output"]:
        return True
    return len(parts) >= 5 and parts[0] == "bots" and parts[2:5] == [
        "data",
        "backtests",
        "output",
    ]


def _is_deployment_package_excluded(path: str | Path, *, include_data: bool = False) -> bool:
    rel_path = _normalise_relative_path(path)
    prefixes = _DEPLOYMENT_EXCLUDE_PREFIXES
    if include_data:
        prefixes = tuple(
            prefix for prefix in prefixes
            if prefix != "packages/trading_assistant_data/data/"
        )
    return (
        rel_path.startswith(prefixes)
        or _is_bot_local_output_path(rel_path)
        or _is_local_archive_path(rel_path)
        or rel_path.endswith(_DEPLOYMENT_EXCLUDE_SUFFIXES)
    )


def _is_local_archive_path(rel_path: str) -> bool:
    return any(part.startswith("_") for part in rel_path.split("/")[:-1])


def _git_package_file_list() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git ls-files failed")
    return [
        _normalise_relative_path(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def deployment_package(
    *,
    max_bytes: int = DEFAULT_DEPLOYMENT_MAX_BYTES,
    include_data: bool = False,
    json_output: bool = False,
) -> int:
    """Dry-run the deployment file set and enforce a byte-size budget."""
    try:
        candidates = _git_package_file_list()
    except RuntimeError as exc:
        print(f"deployment package check failed: {exc}", file=sys.stderr)
        return 2

    included: list[str] = []
    excluded: list[str] = []
    total_bytes = 0
    for rel_path in candidates:
        path = ROOT / rel_path
        if not path.is_file():
            continue
        if _is_deployment_package_excluded(rel_path, include_data=include_data):
            excluded.append(rel_path)
            continue
        included.append(rel_path)
        total_bytes += path.stat().st_size

    summary = {
        "ok": total_bytes <= max_bytes,
        "file_count": len(included),
        "byte_size": total_bytes,
        "max_bytes": max_bytes,
        "excluded_file_count": len(excluded),
        "include_data": include_data,
    }
    if json_output:
        print(json.dumps(summary, indent=2))
    else:
        print(f"deployment_package_ok={summary['ok']}")
        print(f"deployment_package_files={summary['file_count']}")
        print(f"deployment_package_bytes={summary['byte_size']}")
        print(f"deployment_package_max_bytes={summary['max_bytes']}")
        print(f"deployment_package_excluded_files={summary['excluded_file_count']}")
        if not summary["ok"]:
            print(
                "deployment package exceeds byte budget; raise --max-bytes "
                "intentionally or reduce shipped files"
            )
    return 0 if summary["ok"] else 1


def structure(layout: str) -> int:
    return _run(
        [sys.executable, "tools/check_workspace_structure.py", "--layout", layout],
        cwd=ROOT,
    )


def architecture_health() -> int:
    return structure("either")


def monthly_focused() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_architecture_phase_surfaces.py",
            "tests/test_context_sources.py",
            "tests/test_monthly_optimizer_runner.py",
            "tests/test_monthly_runner_contract_conformance.py",
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=_workspace("trading_assistant"),
        env=_pytest_env(),
    )


def data_contracts() -> int:
    return _run(
        [sys.executable, "-m", "pytest", "tests/test_contracts.py", "-q"],
        cwd=_workspace("trading_assistant_data"),
        env=_pytest_env(),
    )


def backtest_monthly() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_monthly_runner.py",
            "tests/unit/test_monthly_execution.py",
            "-q",
        ],
        cwd=_workspace("trading_assistant_backtest"),
        env=_pytest_env(),
    )


def backtest_approval() -> int:
    return _run(
        [sys.executable, "-m", "pytest", "tests/unit/test_approval_grade_audit.py", "-q"],
        cwd=_workspace("trading_assistant_backtest"),
        env=_pytest_env(),
    )


def loop_contracts() -> int:
    return _run(
        [sys.executable, "tools/check_loop_contracts.py"],
        cwd=ROOT,
    )


def loop_ledger() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_loop_run_ledger.py",
            "tests/test_scheduled_runs.py",
            "-q",
        ],
        cwd=_workspace("trading_assistant"),
        env=_pytest_env(),
    )


def performance_learning_ledger() -> int:
    return _run(
        [sys.executable, "tools/check_performance_learning_ledger.py"],
        cwd=ROOT,
    )


def monthly_verifier() -> int:
    return _run(
        [sys.executable, "-m", "pytest", "tests/test_monthly_evidence_verifier.py", "-q"],
        cwd=_workspace("trading_assistant"),
        env=_pytest_env(),
    )


def monthly_shadow_smoke() -> int:
    return _run(
        [sys.executable, "-m", "pytest", "tests/test_monthly_loop_shadow_smoke.py", "-q"],
        cwd=_workspace("trading_assistant"),
        env=_pytest_env(),
    )


def approval_packet_smoke() -> int:
    return _run(
        [sys.executable, "-m", "pytest", "tests/test_approval_packet_smoke.py", "-q"],
        cwd=_workspace("trading_assistant"),
        env=_pytest_env(),
    )


def manifest_validate() -> int:
    with tempfile.TemporaryDirectory(prefix="ta_manifest_validate_") as root:
        manifest_path = _write_manifest_validation_fixture(Path(root))
        return _run(
            [
                sys.executable,
                "-m",
                "trading_assistant_backtest.monthly",
                "--manifest",
                str(manifest_path),
                "--validate-only",
            ],
            cwd=_workspace("trading_assistant_backtest"),
            env=_pytest_env(),
        )


def data_reproduction_smoke() -> int:
    missing = _missing_data_reproduction_inputs()
    if missing:
        print(
            "data reproduction smoke skipped: legacy monthly bundle inputs are absent; "
            f"first missing path: {missing[0]}"
        )
        return 0
    with tempfile.TemporaryDirectory(prefix="ta_data_reproduction_") as artifact_root:
        return _run_json_summary(
            [
                sys.executable,
                "-m",
                "trading_assistant_backtest.validation.data_reproduction_run",
                "--agent-root",
                str(ROOT),
                "--artifact-root",
                artifact_root,
                "--scope",
                "crypto_trader_portfolio",
            ],
            cwd=ROOT,
            env=_pytest_env(),
            summary=lambda payload: (
                f"data_reproduction_status={payload.get('status')}; "
                f"report_count={payload.get('report_count')}; "
                "slice_count="
                f"{sum(int(report.get('slice_count') or 0) for report in payload.get('reports', []))}"
            ),
        )


def validation_matrix() -> int:
    if not _local_validation_inputs_available():
        print(
            "validation matrix skipped: monorepo bot inputs are not available in this checkout"
        )
        return 0
    with tempfile.TemporaryDirectory(prefix="ta_validation_matrix_") as artifact_root:
        return _run_json_summary(
            [
                sys.executable,
                "-m",
                "trading_assistant_backtest.validation.validation_matrix",
                "--agent-root",
                str(ROOT),
                "--artifact-root",
                artifact_root,
            ],
            cwd=ROOT,
            env=_pytest_env(),
            summary=lambda payload: (
                f"validation_matrix_ok={payload.get('ok')}; "
                "all_validation_tests_runnable_for_all_scopes="
                f"{payload.get('all_validation_tests_runnable_for_all_scopes')}; "
                f"approval_remaining_gaps={len(payload.get('approval_remaining_gaps', []))}"
            ),
            payload_gate=_validation_matrix_approval_gate,
        )


def _validation_matrix_approval_gate(payload: dict) -> list[str]:
    if payload.get("approval_grade_validation_complete") is True:
        return []
    gaps = payload.get("approval_remaining_gaps")
    if isinstance(gaps, list) and gaps:
        return [f"approval-grade validation incomplete: {len(gaps)} remaining gap(s)"]
    return ["approval-grade validation incomplete"]


def all_tests() -> int:
    env = _pytest_env()
    return _run_many(
        (
            (
                [sys.executable, "-m", "pytest", "-o", "addopts="],
                _workspace("trading_assistant"),
                env,
            ),
            ([sys.executable, "-m", "pytest"], _workspace("trading_assistant_data"), env),
            (
                [sys.executable, "-m", "pytest"],
                _workspace("trading_assistant_backtest"),
                env,
            ),
        )
    )


def _env_with_paths(*relative_paths: str) -> dict[str, str]:
    env = _pytest_env()
    extras = [str(ROOT / path) for path in relative_paths]
    env["PYTHONPATH"] = os.pathsep.join([*extras, env["PYTHONPATH"]])
    return env


def release_audit() -> int:
    commands = (
        ([sys.executable, "tools/verify_dependency_boundaries.py"], ROOT, _pytest_env()),
        ([sys.executable, "tools/verify_effective_live_configs.py"], ROOT, _pytest_env()),
        ([sys.executable, "tools/verify_cutover_plan.py"], ROOT, _pytest_env()),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "packages/trading_contracts/tests/test_relay_acceptance.py",
                "packages/trading_assistant/tests/test_relay_ingress.py",
                "packages/trading_assistant/tests/test_relay_health.py",
                "packages/trading_assistant/tests/test_startup_scripts.py",
                "packages/trading_assistant/tests/test_scheduled_shadow_report.py",
                "tools/tests/test_verify_operational_deployment_evidence.py",
                "tools/tests/test_collect_operational_evidence.py",
                "-q",
                "-o",
                "addopts=",
            ],
            ROOT,
            _pytest_env(),
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_approval_evidence_spine.py",
                "tests/unit/test_bridge_promotion.py",
                "tests/unit/test_telemetry_conformance.py",
                "-q",
                "-o",
                "addopts=",
            ],
            _workspace("trading_assistant_backtest"),
            _pytest_env(),
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "trading/crypto_trader/tests/live/test_audit_fixes.py",
                "trading/crypto_trader/tests/live/test_config.py",
                "-q",
                "-o",
                "addopts=",
            ],
            ROOT,
            _env_with_paths("trading/crypto_trader/src"),
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "trading/ibkr_trader/tests/unit/test_runtime_scaffold.py",
                "trading/ibkr_trader/tests/unit/test_market_data_policy.py",
                "trading/ibkr_trader/tests/unit/test_deployment_metadata_contract.py",
                "-q",
                "-o",
                "addopts=",
            ],
            ROOT,
            _env_with_paths(
                "trading/ibkr_trader/libs",
                "trading/ibkr_trader/apps",
                "trading/ibkr_trader/src",
            ),
        ),
        (
            [
                sys.executable,
                "-m",
                "pytest",
                "trading/k_stock_trader/tests/deployment/olr_kalcb/test_runtime_operator_script.py",
                "-q",
                "-o",
                "addopts=",
            ],
            ROOT,
            _env_with_paths("trading/k_stock_trader", "trading/k_stock_trader/src"),
        ),
    )
    return _run_many(commands)


def deployment_gate() -> int:
    env = _pytest_env()
    orchestrator_core_tests = (
        "packages/trading_assistant/tests/test_config.py",
        "packages/trading_assistant/tests/test_app_auth.py",
        "packages/trading_assistant/tests/test_preflight.py",
        "packages/trading_assistant/tests/test_integration.py",
        "packages/trading_assistant/tests/test_queue.py",
        "packages/trading_assistant/tests/test_task_registry.py",
        "packages/trading_assistant/tests/test_subagent_manager.py",
        "packages/trading_assistant/tests/test_vps_receiver.py",
        "packages/trading_assistant/tests/test_worker.py",
        "packages/trading_assistant/tests/test_startup_scripts.py",
    )
    tiers = (
        CheckTier(
            name="import-cli-package",
            commands=(
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "imports"], ROOT, 120),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "asgi-import-smoke"], ROOT, 180),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "cli-smoke"], ROOT, 120),
                CheckCommand(
                    [sys.executable, "tools/run_workspace_checks.py", "structure", "--layout", "either"],
                    ROOT,
                    180,
                ),
                CheckCommand(
                    [
                        sys.executable,
                        "tools/run_workspace_checks.py",
                        "architecture-health",
                    ],
                    ROOT,
                    180,
                ),
                CheckCommand(
                    [
                        sys.executable,
                        "tools/run_workspace_checks.py",
                        "deployment-package",
                        "--max-bytes",
                        str(DEFAULT_DEPLOYMENT_MAX_BYTES),
                    ],
                    ROOT,
                    120,
                ),
            ),
            parallelism=6,
        ),
        CheckTier(
            name="release-lint",
            commands=(
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "ruff-fb"], ROOT, 180),
            ),
        ),
        CheckTier(
            name="orchestrator-core",
            commands=(
                CheckCommand(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        *orchestrator_core_tests,
                        "-q",
                        "-o",
                        "addopts=",
                    ],
                    ROOT,
                    420,
                    env,
                ),
            ),
        ),
        CheckTier(
            name="monthly-data-backtest",
            commands=(
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "monthly-focused"], ROOT, 480),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "backtest-monthly"], ROOT, 480),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "data-contracts"], ROOT, 360),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "loop-ledger"], ROOT, 240),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "performance-learning-ledger"], ROOT, 120),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "monthly-verifier"], ROOT, 240),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "approval-packet-smoke"], ROOT, 180),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "manifest-validate"], ROOT, 120),
            ),
            parallelism=3,
        ),
        CheckTier(
            name="release-audit",
            commands=(
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "release-audit"], ROOT, 900),
            ),
        ),
        CheckTier(
            name="slow-local-only",
            optional_env_var="TA_RUN_SLOW_LOCAL_CHECKS",
            commands=(
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "data-reproduction-smoke"], ROOT, 600),
                CheckCommand([sys.executable, "tools/run_workspace_checks.py", "validation-matrix"], ROOT, 900),
            ),
            parallelism=2,
        ),
    )
    return _run_tiers(tiers)


def cli_smoke() -> int:
    env = _pytest_env()
    return _run_many(
        (
            (
                [sys.executable, "-m", "trading_assistant_data", "--help"],
                _workspace("trading_assistant_data"),
                env,
            ),
            (
                [sys.executable, "-m", "trading_assistant_backtest.monthly", "--help"],
                _workspace("trading_assistant_backtest"),
                env,
            ),
            (
                [sys.executable, "-m", "backtests.shared.monthly_repair", "--help"],
                _workspace("trading_assistant_backtest"),
                env,
            ),
        )
    )


def imports() -> int:
    commands = (
        [sys.executable, "-c", "import trading_assistant.schemas.monthly_run_manifest"],
        [sys.executable, "-c", "import trading_assistant.orchestrator.config"],
        [
            sys.executable,
            "-c",
            "import trading_assistant.skills.monthly_validation_orchestrator",
        ],
        [sys.executable, "-c", "import trading_assistant_data.cli"],
        [sys.executable, "-c", "import trading_assistant_backtest.monthly"],
    )
    return _run_many(tuple((command, ROOT, _pytest_env()) for command in commands))


def _app_import_env(**overrides: str) -> dict[str, str]:
    env = _pytest_env()
    for key in (
        "ALLOW_UNAUTHENTICATED_LOCAL",
        "APP_ENV",
        "BIND_HOST",
        "BOT_IDS",
        "DATA_DIR",
        "DEPLOYMENT_ENV",
        "DIRECT_INGEST_ONLY",
        "ENVIRONMENT",
        "ORCHESTRATOR_API_KEY",
        "RELAY_API_KEY",
        "RELAY_URL",
        "UVICORN_HOST",
    ):
        env.pop(key, None)
    env.update(overrides)
    return env


def _run_expected_failure(args: Sequence[str], *, cwd: Path, env: dict[str, str]) -> int:
    print(f"+ expect-fail ({_cwd_label(cwd)}) {_command_text(args)}")
    completed = subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        print("expected command to fail, but it succeeded", file=sys.stderr)
        return 1
    if completed.stderr.strip():
        print(completed.stderr.rstrip())
    return 0


def asgi_import_smoke() -> int:
    import_code = "import trading_assistant.orchestrator.app as target; assert target.app"
    with tempfile.TemporaryDirectory(prefix="ta_asgi_import_") as data_dir:
        commands = (
            (
                [sys.executable, "-c", import_code],
                _app_import_env(
                    ALLOW_UNAUTHENTICATED_LOCAL="true",
                    BIND_HOST="127.0.0.1",
                    UVICORN_HOST="127.0.0.1",
                    ENVIRONMENT="development",
                    DATA_DIR=data_dir,
                ),
                False,
            ),
            (
                [sys.executable, "-c", import_code],
                _app_import_env(
                    ORCHESTRATOR_API_KEY="deployment-gate-key",
                    BIND_HOST="127.0.0.1",
                    UVICORN_HOST="127.0.0.1",
                    ENVIRONMENT="production",
                    BOT_IDS="bot1",
                    DIRECT_INGEST_ONLY="true",
                    DATA_DIR=data_dir,
                ),
                False,
            ),
            (
                [sys.executable, "-c", import_code],
                _app_import_env(
                    BIND_HOST="127.0.0.1",
                    UVICORN_HOST="127.0.0.1",
                    ENVIRONMENT="development",
                    DATA_DIR=data_dir,
                ),
                True,
            ),
            (
                [sys.executable, "-c", import_code],
                _app_import_env(
                    ALLOW_UNAUTHENTICATED_LOCAL="true",
                    BIND_HOST="127.0.0.1",
                    UVICORN_HOST="0.0.0.0",
                    ENVIRONMENT="development",
                    DATA_DIR=data_dir,
                ),
                True,
            ),
        )
        for args, env, expect_failure in commands:
            if expect_failure:
                code = _run_expected_failure(args, cwd=ROOT, env=env)
            else:
                code = _run(args, cwd=ROOT, env=env)
            if code != 0:
                return code
    return 0


def _write_manifest_validation_fixture(root: Path) -> Path:
    from trading_assistant_backtest.contract_models import (
        JSON_ARTIFACTS,
        JSONL_ARTIFACTS,
        REQUIRED_BACKTEST_ARTIFACTS,
    )

    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = "monthly-ci-manifest-validate-2026-04"
    manifest_path = root / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "run_month": "2026-04",
        "mode": "incumbent_validation",
        "bot_id": "ci",
        "strategy_id": "manifest_validate",
        "latest_month_start": "2026-04-01",
        "latest_month_end": "2026-04-30",
        "market_data_manifest_path": str(root / "data_bundle.json"),
        "telemetry_manifest_path": str(root / "telemetry.json"),
        "artifact_root": str(artifact_root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "data_bundle.json").write_text("{}", encoding="utf-8")
    (root / "telemetry.json").write_text("{}", encoding="utf-8")
    for name in REQUIRED_BACKTEST_ARTIFACTS:
        path = artifact_root / name
        if name in JSON_ARTIFACTS:
            path.write_text("{}", encoding="utf-8")
        elif name in JSONL_ARTIFACTS:
            path.write_text("", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
    (artifact_root / "artifact_index.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_root": str(artifact_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _local_validation_inputs_available() -> bool:
    return (
        (ROOT / "trading" / "crypto_trader").exists()
        and (ROOT / "trading" / "k_stock_trader").exists()
        and (ROOT / "trading" / "ibkr_trader").exists()
    )


def _missing_data_reproduction_inputs() -> list[Path]:
    data_root = _workspace("trading_assistant_data")
    bundle_root = data_root / "data" / "bundles" / "monthly"
    bundle_paths = sorted(
        bundle_root.glob("*/crypto_portfolio/phased_optimizer/data_bundle_manifest.json")
    )
    if not bundle_paths:
        return [bundle_root / "<monthly-bundle-manifest>"]
    slice_index_path = bundle_paths[-1].with_name("slice_index.json")
    if not slice_index_path.exists():
        return [slice_index_path]
    slice_index = json.loads(slice_index_path.read_text(encoding="utf-8"))
    missing: list[Path] = []
    for item in slice_index.get("slices", []):
        if not isinstance(item, dict):
            continue
        for raw_path in item.get("canonical_paths", []):
            path = data_root / str(raw_path)
            if not path.exists():
                missing.append(path)
    return missing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run trading-assistant workspace checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    structure_parser = subparsers.add_parser("structure")
    structure_parser.add_argument(
        "--layout",
        choices=("current", "final", "either"),
        default="final",
    )
    command_handlers = {
        "structure": lambda parsed: structure(parsed.layout),
        "architecture-health": lambda _parsed: architecture_health(),
        "monthly-focused": lambda _parsed: monthly_focused(),
        "data-contracts": lambda _parsed: data_contracts(),
        "backtest-monthly": lambda _parsed: backtest_monthly(),
        "backtest-approval": lambda _parsed: backtest_approval(),
        "loop-contracts": lambda _parsed: loop_contracts(),
        "loop-ledger": lambda _parsed: loop_ledger(),
        "performance-learning-ledger": lambda _parsed: performance_learning_ledger(),
        "monthly-verifier": lambda _parsed: monthly_verifier(),
        "monthly-shadow-smoke": lambda _parsed: monthly_shadow_smoke(),
        "approval-packet-smoke": lambda _parsed: approval_packet_smoke(),
        "manifest-validate": lambda _parsed: manifest_validate(),
        "data-reproduction-smoke": lambda _parsed: data_reproduction_smoke(),
        "validation-matrix": lambda _parsed: validation_matrix(),
        "all-tests": lambda _parsed: all_tests(),
        "release-audit": lambda _parsed: release_audit(),
        "deployment-gate": lambda _parsed: deployment_gate(),
        "asgi-import-smoke": lambda _parsed: asgi_import_smoke(),
        "cli-smoke": lambda _parsed: cli_smoke(),
        "imports": lambda _parsed: imports(),
    }
    for command in command_handlers:
        if command != "structure":
            subparsers.add_parser(command)
    ruff_parser = subparsers.add_parser("ruff-fb")
    ruff_parser.add_argument("--update-baseline", action="store_true")
    package_parser = subparsers.add_parser("deployment-package")
    package_parser.add_argument("--max-bytes", type=int, default=DEFAULT_DEPLOYMENT_MAX_BYTES)
    package_parser.add_argument("--include-data", action="store_true")
    package_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ruff-fb":
        return ruff_fb(update_baseline=args.update_baseline)
    if args.command == "deployment-package":
        return deployment_package(
            max_bytes=args.max_bytes,
            include_data=args.include_data,
            json_output=args.json,
        )
    if args.command in command_handlers:
        return command_handlers[args.command](args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
