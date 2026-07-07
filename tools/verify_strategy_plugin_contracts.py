"""Validate canonical strategy plugin contracts in the monorepo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for src in (
    ROOT / "packages" / "trading_contracts" / "src",
):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from trading_contracts.legacy import validate_plugin_contract  # noqa: E402

REFERENCE_TOKEN = "_ref" "erences"


def main() -> int:
    args = build_parser().parse_args()
    paths = list(_contract_paths(args))
    if not paths:
        print("FAIL no strategy plugin contract files selected")
        return 1
    failures: list[str] = []
    for path in paths:
        try:
            contract = validate_plugin_contract(path)
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
            print(f"FAIL {path.relative_to(ROOT)} - {exc}")
            continue
        missing = contract.maturity_contract_errors()
        if missing:
            failures.append(f"{path.relative_to(ROOT)}: missing {', '.join(missing)}")
            print(f"FAIL {path.relative_to(ROOT)} - missing {', '.join(missing)}")
            continue
        path_errors = _path_errors(contract)
        if path_errors:
            failures.append(f"{path.relative_to(ROOT)}: {', '.join(path_errors)}")
            print(f"FAIL {path.relative_to(ROOT)} - {', '.join(path_errors)}")
            continue
        print(f"PASS {path.relative_to(ROOT)} - {contract.plugin_id} ({contract.maturity.value})")
    failures.extend(_runtime_metadata_path_failures())
    if failures:
        print("\nStrategy plugin contract verification failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Strategy plugin contract verification passed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Validate all canonical contracts.")
    parser.add_argument("--path", action="append", default=[], help="Specific contract path.")
    return parser


def _contract_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.all:
        paths.extend(sorted((ROOT / "contracts" / "strategy_plugins").rglob("strategy_plugin_contract.json")))
    paths.extend(Path(path) if Path(path).is_absolute() else ROOT / path for path in args.path)
    return paths


def _path_errors(contract: object) -> list[str]:
    errors: list[str] = []
    for attr in ("live_repo_path", "backtest_adapter_path"):
        value = str(getattr(contract, attr, "") or "").replace("\\", "/")
        if REFERENCE_TOKEN in value:
            errors.append(f"{attr} still points at a legacy snapshot")
            continue
        if value and not (ROOT / value).exists():
            errors.append(f"{attr} does not exist: {value}")
    for fixture in getattr(contract, "parity_fixture_set", []) or []:
        value = str(fixture).replace("\\", "/")
        if REFERENCE_TOKEN in value:
            errors.append(f"parity fixture still points at a legacy snapshot: {value}")
        if value.startswith(("bots/", "packages/")) and not (ROOT / value).exists():
            errors.append(f"parity fixture does not exist: {value}")
    return errors


def _runtime_metadata_path_failures() -> list[str]:
    failures: list[str] = []
    for path in (
        ROOT / "bots/k_stock_trader/deployment/olr_kalcb/deployment_metadata.py",
        ROOT / "bots/ibkr_trading/libs/instrumentation/deployment_metadata.py",
        ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/validation/replay_evidence_run.py",
        ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/validation/week1_decision_parity_run.py",
        ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/validation/decision_parity_run.py",
        ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/validation/approval_grade_audit.py",
        ROOT / "packages/trading_assistant_backtest/src/trading_assistant_backtest/validation/bridge_readiness.py",
    ):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if "trading_assistant_backtest/contracts" in text:
            failures.append(f"{relative}: legacy assistant-backtest contract path remains")
            print(f"FAIL {relative} - legacy assistant-backtest contract path remains")
        if REFERENCE_TOKEN in text:
            failures.append(f"{relative}: legacy snapshot path remains")
            print(f"FAIL {relative} - legacy snapshot path remains")
        if "contracts/strategy_plugins" not in text and "strategy_plugins" not in text:
            failures.append(f"{relative}: canonical strategy plugin contract path missing")
            print(f"FAIL {relative} - canonical strategy plugin contract path missing")
    for path in sorted(
        (ROOT / "packages/trading_assistant_backtest/contracts").rglob("strategy_plugin_contract.json")
    ):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if REFERENCE_TOKEN in text:
            failures.append(f"{relative}: package-local contract still points at a legacy snapshot")
            print(f"FAIL {relative} - package-local contract still points at a legacy snapshot")
    for path in sorted(
        (ROOT / "packages/trading_assistant_backtest/contracts").rglob("deployment_metadata.json")
    ):
        failures.extend(_deployment_metadata_failures(path))
    for path in sorted((ROOT / "contracts/strategy_plugins").rglob("deployment_metadata.json")):
        failures.extend(_deployment_metadata_failures(path))
    if not failures:
        print("PASS runtime metadata defaults - canonical strategy plugin contract paths")
    return failures


def _deployment_metadata_failures(path: Path) -> list[str]:
    failures: list[str] = []
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(ROOT)
    if REFERENCE_TOKEN in text:
        failures.append(f"{relative}: deployment metadata still points at a legacy snapshot")
        print(f"FAIL {relative} - deployment metadata still points at a legacy snapshot")
    if "trading_assistant_backtest/contracts" in text:
        failures.append(f"{relative}: package-local strategy contract path remains")
        print(f"FAIL {relative} - package-local strategy contract path remains")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
