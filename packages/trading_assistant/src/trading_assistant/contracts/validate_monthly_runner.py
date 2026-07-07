"""CLI for validating a monthly runner artifact contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trading_assistant.contracts.monthly_runner_contract import validate_manifest_file, validate_manifest_artifacts
from trading_assistant.schemas.monthly_run_manifest import MonthlyRunManifest
from trading_assistant.skills.backtest_runner_client import BacktestRunnerClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate monthly runner artifacts")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.json")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run manifest.backtest_command before validating artifacts",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if args.run:
        manifest = MonthlyRunManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        result = BacktestRunnerClient().run(manifest, manifest_path)
        errors = [] if result.success else [result.error or "runner failed"]
        if result.artifact_index is not None:
            errors.extend(validate_manifest_artifacts(
                manifest,
                result.artifact_index,
                manifest_path=manifest_path,
            ))
        payload = {
            "valid": result.success and not errors,
            "manifest_path": str(manifest_path),
            "artifact_index_path": str(Path(manifest.artifact_root) / "artifact_index.json"),
            "errors": list(dict.fromkeys(error for error in errors if error)),
        }
    else:
        validation = validate_manifest_file(manifest_path)
        payload = {
            "valid": validation.valid,
            "manifest_path": validation.manifest_path,
            "artifact_index_path": validation.artifact_index_path,
            "errors": validation.errors,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
