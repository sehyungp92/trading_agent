"""Native monthly runner CLI adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a monthly trading-assistant backtest manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.json")
    parser.add_argument(
        "--planner-mode",
        choices=["deterministic"],
        default="deterministic",
        help="Experiment planner mode.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artifacts already emitted under artifact_root.",
    )
    args = parser.parse_args(argv)
    if args.validate_only:
        from trading_assistant_backtest.contract_loader import validate_manifest_file

        validation = validate_manifest_file(args.manifest)
        print(json.dumps({"valid": validation.valid, "errors": validation.errors}, indent=2))
        return 0 if validation.valid else 1
    return run_manifest(Path(args.manifest), planner_mode=args.planner_mode)


def run_manifest(manifest_path: Path, *, planner_mode: str = "deterministic") -> int:
    from trading_assistant_backtest.monthly_execution import MonthlyExecution

    return MonthlyExecution().run_manifest(manifest_path, planner_mode=planner_mode)


if __name__ == "__main__":
    raise SystemExit(main())
