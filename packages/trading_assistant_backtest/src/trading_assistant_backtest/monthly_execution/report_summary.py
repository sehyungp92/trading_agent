"""Native monthly runner CLI."""

from __future__ import annotations

from trading_assistant_backtest.contract_models import (
    BacktestArtifactIndex,
    MonthlyRunManifest,
)


def raise_on_local_index_errors(
    manifest: MonthlyRunManifest, index: BacktestArtifactIndex
) -> None:
    errors = index.validation_errors(
        expected_run_id=manifest.run_id,
        expected_manifest_id=manifest.manifest_id,
        require_manifest_id=manifest.optimizer_mode,
    )
    if errors:
        raise RuntimeError("; ".join(errors))


def mode_status(manifest: MonthlyRunManifest) -> str:
    if manifest.optimizer_mode:
        return "no_adoption"
    return "no_change"


def mode_reason(manifest: MonthlyRunManifest) -> str:
    if manifest.optimizer_mode:
        return "deterministic runner found no approval-ready replay-backed candidate"
    return "incumbent validation artifacts emitted"


def no_adoption_reason(manifest: MonthlyRunManifest, data_errors: list[str]) -> str:
    if data_errors:
        return "blocked by data bundle validation: " + "; ".join(data_errors)
    if manifest.strategy_plugin_id:
        return "strategy plugin emitted no replay-backed approval-ready candidate"
    return "insufficient mature replay plugin sample size"


def stdout_summary(manifest: MonthlyRunManifest, exit_code: int, errors: list[str]) -> str:
    lines = [
        f"run_id={manifest.run_id}",
        f"mode={manifest.mode.value}",
        f"artifact_root={manifest.artifact_root}",
        f"exit_code={exit_code}",
    ]
    lines.extend(f"error={error}" for error in errors)
    return "\n".join(lines) + "\n"


def monthly_report(manifest: MonthlyRunManifest, status: str, errors: list[str]) -> str:
    lines = [
        f"# Monthly Backtest Report: {manifest.run_id}",
        "",
        f"- Mode: `{manifest.mode.value}`",
        f"- Bot: `{manifest.bot_id}`",
        f"- Strategy: `{manifest.strategy_id}`",
        f"- Status: `{status}`",
        "- Live deployment: `not_requested`",
    ]
    if errors:
        lines.append(f"- Blocking reasons: {'; '.join(errors)}")
    elif manifest.optimizer_mode:
        lines.append("- Decision: no adoption; deterministic gates require a mature replay plugin.")
    else:
        lines.append("- Decision: incumbent validation only.")
    return "\n".join(lines) + "\n"
