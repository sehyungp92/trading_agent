"""Window resolution for monthly runs."""

from __future__ import annotations

from datetime import timedelta

from trading_assistant_backtest.contract_models import MonthlyRunManifest, build_two_fold_manifest
from trading_assistant_backtest.replay.types import WindowSpec


def resolve_in_sample_window(manifest: MonthlyRunManifest) -> WindowSpec:
    selection_start = manifest.selection_oos_start or manifest.latest_month_start
    start = (
        manifest.in_sample_start
        or manifest.calibration_start
        or (selection_start - timedelta(days=90))
    )
    end = (
        manifest.in_sample_end or manifest.calibration_end or (selection_start - timedelta(days=1))
    )
    if end >= selection_start:
        end = selection_start - timedelta(days=1)
    if end < start:
        start = end - timedelta(days=1)
    return WindowSpec(name="in_sample", start=start, end=end)


def resolve_selection_oos_window(manifest: MonthlyRunManifest) -> WindowSpec:
    return WindowSpec(
        name="selection_oos",
        start=manifest.selection_oos_start or manifest.latest_month_start,
        end=manifest.selection_oos_end or manifest.latest_month_end,
    )


def build_manifest_folds(manifest: MonthlyRunManifest, evidence_paths: list[str]):
    in_sample = resolve_in_sample_window(manifest)
    selection_oos = resolve_selection_oos_window(manifest)
    return build_two_fold_manifest(
        run_id=manifest.run_id,
        run_month=manifest.run_month,
        in_sample_start=in_sample.start,
        in_sample_end=in_sample.end,
        selection_oos_start=selection_oos.start,
        selection_oos_end=selection_oos.end,
        evidence_paths=evidence_paths,
    )
