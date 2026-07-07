from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DEPS = {"trading-assistant", "trading-assistant-backtest"}
REFERENCE_TOKEN = "_ref" "erences"
FORBIDDEN_SMOKE_OUTPUT = (
    "KRX 로그인 실패",
    "KRX_ID",
    "KRX_PW",
    "IBKR authentication",
    "IBKR login",
    "broker login",
    "credential failure",
)
DOCKERIGNORE_REQUIRED = (
    "_[!_]*/",
    "**/_[!_]*/",
    "!**/__init__.py",
    "bots/*/output/",
    "bots/*/backtests/output/",
    "bots/*/backtests/*/data/raw/",
    "bots/crypto_trader/data/",
    "bots/*/data/backtests/output/",
)


@dataclass(frozen=True)
class BotImageSpec:
    bot: str
    package_name: str
    dockerfile: str
    compose: str
    pyproject: str
    report: str
    import_modules: tuple[str, ...]
    entrypoint_smoke: tuple[str, ...]


BOTS = {
    "ibkr": BotImageSpec(
        "ibkr",
        "ibkr-trading",
        "bots/ibkr_trading/Dockerfile",
        "deployments/ibkr/docker-compose.yml",
        "bots/ibkr_trading/pyproject.toml",
        "deployments/ibkr/generated/dependency_report.json",
        ("ibkr_trading", "apps.runtime.cli", "apps.runtime.runtime"),
        ("ibkr-trading-runtime", "--help"),
    ),
    "crypto": BotImageSpec(
        "crypto",
        "crypto-trader",
        "bots/crypto_trader/Dockerfile",
        "deployments/crypto/docker-compose.yml",
        "bots/crypto_trader/pyproject.toml",
        "deployments/crypto/generated/dependency_report.json",
        ("crypto_trader", "crypto_trader.cli"),
        ("crypto-trader", "--help"),
    ),
    "k_stock": BotImageSpec(
        "k_stock",
        "k-stock-trader",
        "bots/k_stock_trader/Dockerfile",
        "deployments/k_stock/docker-compose.yml",
        "bots/k_stock_trader/pyproject.toml",
        "deployments/k_stock/generated/dependency_report.json",
        ("k_stock_trader", "deployment.olr_kalcb.runtime"),
        ("k-stock-olr-kalcb-runtime", "--help"),
    ),
}


def main() -> int:
    args = _parser().parse_args()
    selected = list(BOTS) if args.bot == "all" else [args.bot]
    errors: list[str] = []
    reports: list[dict[str, Any]] = []
    errors.extend(_dockerignore_errors())
    errors.extend(_legacy_dockerfile_errors())
    for bot in selected:
        report, report_errors = _verify_bot(BOTS[bot])
        reports.append(report)
        errors.extend(report_errors)
        if report_errors:
            report["image_build"] = {"status": "skipped", "reason": "preflight failed"}
        elif args.preflight_only:
            report["image_build"] = {"status": "skipped", "reason": "--preflight-only"}
        else:
            build_record, build_errors = _build_and_smoke(BOTS[bot], timeout=args.timeout_seconds)
            report["image_build"] = build_record
            errors.extend(build_errors)
        if args.emit_dependency_reports:
            _write_json(ROOT / BOTS[bot].report, report)
    result = {"valid": not errors, "reports": reports, "errors": errors}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify deployable bot image definitions.")
    parser.add_argument("--bot", choices=["all", *BOTS], default="all")
    parser.add_argument("--emit-dependency-reports", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser


def _verify_bot(spec: BotImageSpec) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    dockerfile = ROOT / spec.dockerfile
    compose = ROOT / spec.compose
    pyproject = ROOT / spec.pyproject
    dependencies = _dependencies(pyproject)
    docker_text = dockerfile.read_text(encoding="utf-8") if dockerfile.exists() else ""
    if not dockerfile.exists():
        errors.append(f"{spec.bot}: missing Dockerfile {spec.dockerfile}")
    if not compose.exists():
        errors.append(f"{spec.bot}: missing compose file {spec.compose}")
    if "FROM python:3.12" not in docker_text:
        errors.append(f"{spec.bot}: Dockerfile must use the Python 3.12 live-runtime base")
    for forbidden in (REFERENCE_TOKEN, "trading_assistant_backtest/contracts", "pip install -r"):
        if forbidden in docker_text:
            errors.append(f"{spec.bot}: Dockerfile contains forbidden build input {forbidden!r}")
    leaked = sorted(FORBIDDEN_DEPS & {dep.split("[", 1)[0].lower() for dep in dependencies})
    if leaked:
        errors.append(f"{spec.bot}: live image dependencies include assistant packages: {leaked}")
    return (
        {
            "bot": spec.bot,
            "package": spec.package_name,
            "dockerfile": spec.dockerfile,
            "compose": spec.compose,
            "dependencies": dependencies,
            "assistant_packages_present": bool(leaked),
        },
        errors,
    )


def _dependencies(path: Path) -> list[str]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    return [str(dep) for dep in payload.get("project", {}).get("dependencies", [])]


def _dockerignore_errors() -> list[str]:
    path = ROOT / ".dockerignore"
    if not path.exists():
        return ["missing .dockerignore"]
    text = path.read_text(encoding="utf-8")
    return [f".dockerignore missing {pattern}" for pattern in DOCKERIGNORE_REQUIRED if pattern not in text]


def _legacy_dockerfile_errors() -> list[str]:
    deployable = {str((ROOT / spec.dockerfile).resolve()) for spec in BOTS.values()}
    errors: list[str] = []
    for path in sorted(ROOT.glob("bots/**/Dockerfile")):
        if str(path.resolve()) in deployable:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        if "trading-agent: non-deployable-legacy-dockerfile" not in text:
            errors.append(f"{relative}: legacy Dockerfile must be marked non-deployable")
        if "FROM python:3.11" in text:
            errors.append(f"{relative}: legacy Dockerfile still uses Python 3.11")
        if "trading_assistant_backtest/contracts" in text:
            errors.append(f"{relative}: legacy Dockerfile copies package-local contracts")
    return errors


def _build_and_smoke(spec: BotImageSpec, *, timeout: int) -> tuple[dict[str, Any], list[str]]:
    tag = f"trading-agent-{spec.bot}:acceptance"
    build_command = ["docker", "build", "-f", spec.dockerfile, "-t", tag, "."]
    smoke_code = "; ".join(f"import {module}" for module in spec.import_modules)
    smoke_command = ["docker", "run", "--rm", tag, "python", "-c", smoke_code]
    entrypoint_command = ["docker", "run", "--rm", tag, *spec.entrypoint_smoke]
    record: dict[str, Any] = {
        "status": "pending",
        "tag": tag,
        "build_command": build_command,
        "runtime_import_smoke_command": smoke_command,
        "entrypoint_smoke_command": entrypoint_command,
    }
    if shutil.which("docker") is None:
        record["status"] = "blocked"
        record["reason"] = "docker CLI is not installed"
        return record, [f"{spec.bot}: docker CLI is not installed"]
    build = _run(build_command, timeout=timeout)
    record["build"] = build
    if build["returncode"] != 0:
        record["status"] = "failed"
        return record, [f"{spec.bot}: docker build failed"]
    smoke = _run(smoke_command, timeout=timeout)
    record["runtime_import_smoke"] = smoke
    if smoke["returncode"] != 0:
        record["status"] = "failed"
        return record, [f"{spec.bot}: in-image runtime import smoke failed"]
    smoke_errors = _forbidden_smoke_output_errors(spec.bot, "runtime import smoke", smoke)
    if smoke_errors:
        record["status"] = "failed"
        return record, smoke_errors
    entrypoint = _run(entrypoint_command, timeout=timeout)
    record["entrypoint_smoke"] = entrypoint
    if entrypoint["returncode"] != 0:
        record["status"] = "failed"
        return record, [f"{spec.bot}: in-image entrypoint smoke failed"]
    entrypoint_errors = _forbidden_smoke_output_errors(spec.bot, "entrypoint smoke", entrypoint)
    if entrypoint_errors:
        record["status"] = "failed"
        return record, entrypoint_errors
    record["status"] = "pass"
    return record, []


def _forbidden_smoke_output_errors(bot: str, label: str, record: dict[str, Any]) -> list[str]:
    output = "\n".join(
        str(line)
        for key in ("stdout_tail", "stderr_tail")
        for line in record.get(key, [])
    )
    return [
        f"{bot}: {label} emitted broker/exchange credential output: {token}"
        for token in FORBIDDEN_SMOKE_OUTPUT
        if token in output
    ]


def _run(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": -1,
            "timeout_seconds": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-20:],
            "stderr_tail": (exc.stderr or "").splitlines()[-20:],
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "").splitlines()[-20:],
        "stderr_tail": (completed.stderr or "").splitlines()[-20:],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
