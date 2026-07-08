"""Import-smoke gate for workspace package extraction.

The script intentionally checks imports from the workspace member paths directly.
That keeps Phase 1 focused on packaging/path correctness without requiring broker
secrets, market data, or image builds.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PackageSmoke:
    name: str
    member_path: Path
    import_paths: tuple[str, ...]
    modules: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...] = ()


PACKAGE_SMOKES = {
    "ibkr-trading": PackageSmoke(
        name="ibkr-trading",
        member_path=ROOT / "trading" / "ibkr_trader",
        import_paths=(".", "src"),
        modules=("ibkr_trading", "apps.runtime.cli", "apps.runtime.runtime"),
        commands=(("python", "-m", "apps.runtime.cli", "--help"),),
    ),
    "k-stock-trader": PackageSmoke(
        name="k-stock-trader",
        member_path=ROOT / "trading" / "k_stock_trader",
        import_paths=(".", "src", "scripts"),
        modules=("k_stock_trader", "deployment.olr_kalcb.runtime"),
        commands=(("python", "scripts/run_olr_kalcb_runtime_session.py", "--help"),),
    ),
    "crypto-trader": PackageSmoke(
        name="crypto-trader",
        member_path=ROOT / "trading" / "crypto_trader",
        import_paths=("src",),
        modules=("crypto_trader", "crypto_trader.cli"),
        commands=(("python", "-m", "crypto_trader.cli", "--help"),),
    ),
    "trading-assistant": PackageSmoke(
        name="trading-assistant",
        member_path=ROOT / "packages" / "trading_assistant",
        import_paths=("src",),
        modules=("trading_assistant",),
    ),
    "trading-assistant-data": PackageSmoke(
        name="trading-assistant-data",
        member_path=ROOT / "packages" / "trading_assistant_data",
        import_paths=("src",),
        modules=("trading_assistant_data", "trading_assistant_data.cli"),
    ),
    "trading-assistant-backtest": PackageSmoke(
        name="trading-assistant-backtest",
        member_path=ROOT / "packages" / "trading_assistant_backtest",
        import_paths=("src",),
        modules=("trading_assistant_backtest", "trading_assistant_backtest.monthly"),
    ),
    "trading-contracts": PackageSmoke(
        name="trading-contracts",
        member_path=ROOT / "packages" / "trading_contracts",
        import_paths=("src",),
        modules=("trading_contracts", "trading_contracts.cli"),
        commands=(("python", "-m", "trading_contracts.cli", "--help"),),
    ),
    "trading-config": PackageSmoke(
        name="trading-config",
        member_path=ROOT / "packages" / "trading_config",
        import_paths=("src",),
        modules=("trading_config", "trading_config.cli"),
        commands=(("python", "-m", "trading_config.cli", "--help"),),
    ),
    "trading-backtest": PackageSmoke(
        name="trading-backtest",
        member_path=ROOT / "packages" / "trading_backtest",
        import_paths=("src",),
        modules=("trading_backtest", "trading_backtest.invariants"),
    ),
    "trading-optimizer": PackageSmoke(
        name="trading-optimizer",
        member_path=ROOT / "packages" / "trading_optimizer",
        import_paths=("src",),
        modules=("trading_optimizer",),
    ),
    "trading-instrumentation": PackageSmoke(
        name="trading-instrumentation",
        member_path=ROOT / "packages" / "trading_instrumentation",
        import_paths=("src",),
        modules=("trading_instrumentation", "trading_instrumentation.approval_metadata"),
    ),
    "trading-deployment": PackageSmoke(
        name="trading-deployment",
        member_path=ROOT / "packages" / "trading_deployment",
        import_paths=("src",),
        modules=("trading_deployment", "trading_deployment.metadata"),
    ),
}


def main() -> int:
    args = build_parser().parse_args()
    package_names = tuple(PACKAGE_SMOKES) if args.all_packages else tuple(args.package)
    failures: list[str] = []
    for package_name in package_names:
        smoke = PACKAGE_SMOKES[package_name]
        failures.extend(run_package_smoke(smoke, run_commands=args.run_commands))
    if failures:
        print("\nWorkspace import smoke failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Workspace import smoke passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-packages", action="store_true", help="Check every workspace member.")
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(PACKAGE_SMOKES),
        default=[],
        help="Workspace package to check. May be provided more than once.",
    )
    parser.add_argument(
        "--run-commands",
        action="store_true",
        help="Also run lightweight CLI help commands.",
    )
    return parser


def run_package_smoke(smoke: PackageSmoke, *, run_commands: bool) -> list[str]:
    failures: list[str] = []
    if not smoke.member_path.exists():
        message = f"{smoke.name} member path is missing: {smoke.member_path.relative_to(ROOT)}"
        print(f"FAIL {smoke.name} - {message}")
        return [message]
    prepend_import_paths(smoke)
    for module in smoke.modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            message = f"{smoke.name}:{module} import failed: {type(exc).__name__}: {exc}"
            print(f"FAIL {smoke.name}:{module} - {type(exc).__name__}: {exc}")
            failures.append(message)
        else:
            print(f"PASS {smoke.name}:{module}")
    if run_commands:
        for command in smoke.commands:
            failures.extend(run_command(smoke, command))
    return failures


def prepend_import_paths(smoke: PackageSmoke) -> None:
    for relative in reversed(smoke.import_paths):
        path = smoke.member_path / relative
        if path.exists():
            sys.path.insert(0, str(path))


def run_command(smoke: PackageSmoke, command: tuple[str, ...]) -> list[str]:
    env = os.environ.copy()
    paths = [
        smoke.member_path / relative
        for relative in smoke.import_paths
        if (smoke.member_path / relative).exists()
    ]
    paths.extend(sorted(path for path in (ROOT / "packages").glob("*/src") if path.exists()))
    deduped_paths = list(dict.fromkeys(str(path) for path in paths))
    pythonpath = os.pathsep.join(deduped_paths)
    env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        command,
        cwd=smoke.member_path,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=120,
    )
    label = f"{smoke.name}:{' '.join(command)}"
    if completed.returncode == 0:
        print(f"PASS {label}")
        return []
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    message = f"{label} exited {completed.returncode}: {detail[-1] if detail else 'no output'}"
    print(f"FAIL {message}")
    return [message]


if __name__ == "__main__":
    raise SystemExit(main())
