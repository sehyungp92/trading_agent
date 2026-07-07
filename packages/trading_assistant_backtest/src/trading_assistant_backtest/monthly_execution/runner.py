"""Native monthly runner CLI."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from trading_assistant_backtest.artifact_writer import ArtifactWriter
from trading_assistant_backtest.contract_loader import validate_manifest_file
from trading_assistant_backtest.contract_models import MonthlyRunMode
from trading_assistant_backtest.data.bundle_loader import (
    data_bundle_errors,
    load_data_bundle,
)
from trading_assistant_backtest.manifest_loader import load_manifest
from trading_assistant_backtest.monthly_execution.artifact_emitter import write_required_artifacts
from trading_assistant_backtest.monthly_execution.optimizer_sequence import (
    write_optimizer_artifacts,
)
from trading_assistant_backtest.monthly_execution.replay_context import build_replay_context
from trading_assistant_backtest.monthly_execution.report_summary import (
    raise_on_local_index_errors,
    stdout_summary,
)
from trading_assistant_backtest.monthly_execution.structural_candidates import (
    write_structural_placeholders,
)
from trading_assistant_backtest.strategies.contracts import strategy_plugin_errors


class MonthlyExecution:
    """Own the monthly manifest execution flow."""

    def run_manifest(self, manifest_path: Path, *, planner_mode: str = "deterministic") -> int:
        return run_manifest_impl(Path(manifest_path), planner_mode=planner_mode)


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
        validation = validate_manifest_file(args.manifest)
        print(json.dumps({"valid": validation.valid, "errors": validation.errors}, indent=2))
        return 0 if validation.valid else 1
    return run_manifest(Path(args.manifest), planner_mode=args.planner_mode)


def run_manifest(manifest_path: Path, *, planner_mode: str = "deterministic") -> int:
    return run_manifest_impl(manifest_path, planner_mode=planner_mode)


def run_manifest_impl(manifest_path: Path, *, planner_mode: str = "deterministic") -> int:
    started_at = datetime.now(UTC)
    manifest = load_manifest(manifest_path)
    artifact_root = Path(manifest.artifact_root).resolve()
    writer = ArtifactWriter(manifest, artifact_root)

    bundle = load_data_bundle(manifest)
    bundle_errors = [
        *data_bundle_errors(manifest, bundle),
        *strategy_plugin_errors(manifest, bundle),
    ]
    exit_code = 2 if manifest.optimizer_mode and bundle_errors else 0
    replay_context = build_replay_context(manifest, bundle, bundle_errors)

    write_required_artifacts(writer, manifest, bundle, bundle_errors, replay_context)
    if manifest.optimizer_mode:
        write_optimizer_artifacts(
            writer,
            manifest,
            manifest_path=manifest_path,
            data_errors=bundle_errors,
            planner_mode=planner_mode,
            replay_context=replay_context,
        )
    if manifest.mode == MonthlyRunMode.STRUCTURAL_REVIEW:
        write_structural_placeholders(writer, manifest, bundle)

    writer.write_text("stdout.log", stdout_summary(manifest, exit_code, bundle_errors))
    writer.write_text("stderr.log", "\n".join(bundle_errors) + ("\n" if bundle_errors else ""))
    writer.write_exit_status(
        started_at=started_at,
        exit_code=exit_code,
        error="; ".join(bundle_errors),
    )
    index = writer.write_index()
    raise_on_local_index_errors(manifest, index)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
