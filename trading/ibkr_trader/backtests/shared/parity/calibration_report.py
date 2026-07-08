from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REQUIRED_CALIBRATION_FAMILIES = {
    "swing": ("ATRSS", "AKC_HELIX", "TPC"),
    "momentum": ("NQDTC_v2.1", "NQ_REGIME", "VdubusNQ_v4", "DownturnDominator_v1"),
    "stock": ("IARIC_v1", "ALCB_v1"),
}
REQUIRED_CALIBRATION_STRATEGY_IDS = tuple(
    strategy_id
    for strategy_ids in REQUIRED_CALIBRATION_FAMILIES.values()
    for strategy_id in strategy_ids
)
REQUIRED_SOURCE_TABLES = {"orders", "trades", "fills", "order_events"}
REQUIRED_TOLERANCE_KEYS = {
    "backtest_source_present",
    "slippage_mean_abs_diff",
    "commission_abs_diff",
    "partial_fill_rate_abs_diff",
    "ioc_reject_rate_abs_diff",
}
REQUIRED_METRIC_KEYS = {
    "broker_fill_sample_count",
    "mean_slippage_ticks",
    "mean_commission",
    "partial_fill_rate",
    "ioc_reject_rate",
    "mean_time_to_fill_seconds",
}
MAX_FUTURE_REPORT_SKEW = timedelta(minutes=5)


class CalibrationReportError(ValueError):
    pass


def latest_calibration_report_path(*, root: Path | None = None) -> Path | None:
    base = root or Path(__file__).resolve().parents[3]
    reports = sorted((base / "docs").glob("parity_calibration_*.json"))
    return reports[-1] if reports else None


def load_calibration_report(
    path: Path,
    *,
    max_age_days: int | None = 7,
    now: datetime | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CalibrationReportError(f"{path} is not valid JSON: {exc}") from exc
    return validate_calibration_report(payload, max_age_days=max_age_days, now=now)


def validate_calibration_report(
    payload: dict[str, Any],
    *,
    max_age_days: int | None = 7,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CalibrationReportError("calibration report must be a JSON object")
    if payload.get("report_type") != "broker_backed":
        raise CalibrationReportError("calibration report must have report_type='broker_backed'")
    _validate_window(payload, max_age_days=max_age_days, now=now)

    source_tables = set(payload.get("source_tables") or [])
    missing_tables = sorted(REQUIRED_SOURCE_TABLES - source_tables)
    if missing_tables:
        raise CalibrationReportError("calibration report missing source tables: " + ", ".join(missing_tables))

    if _positive_int(payload.get("broker_fill_sample_count")) <= 0:
        raise CalibrationReportError("calibration report must include broker_fill_sample_count > 0")
    if payload.get("within_tolerance") is not True:
        raise CalibrationReportError("calibration report must have within_tolerance=true")

    expected_strategy_ids = _validate_expected_coverage(payload)
    strategy_results = payload.get("strategy_results")
    if not isinstance(strategy_results, list) or not strategy_results:
        raise CalibrationReportError("calibration report must include non-empty strategy_results")

    for index, result in enumerate(strategy_results):
        _validate_strategy_result(result, index)
    _validate_strategy_coverage(strategy_results, expected_strategy_ids)
    return payload


def _validate_window(
    payload: dict[str, Any],
    *,
    max_age_days: int | None,
    now: datetime | None,
) -> None:
    window = payload.get("window")
    if not isinstance(window, dict):
        raise CalibrationReportError("calibration report must include window object")
    end_raw = window.get("end")
    if not end_raw:
        raise CalibrationReportError("calibration report window must include end timestamp")
    end = _parse_datetime(end_raw, "window.end")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if end > current + MAX_FUTURE_REPORT_SKEW:
        raise CalibrationReportError(
            f"calibration report window.end {end.isoformat()} is in the future"
        )
    if max_age_days is None:
        return
    if end < current - timedelta(days=max_age_days):
        raise CalibrationReportError(
            f"calibration report is stale: window.end {end.isoformat()} is older than {max_age_days} days"
        )


def _validate_expected_coverage(payload: dict[str, Any]) -> set[str]:
    expected_family_ids = _required_string_list(payload, "expected_family_ids")
    expected_strategy_ids = _required_string_list(payload, "expected_strategy_ids")

    _validate_exact_members(
        "expected_family_ids",
        expected_family_ids,
        set(REQUIRED_CALIBRATION_FAMILIES),
    )
    _validate_exact_members(
        "expected_strategy_ids",
        expected_strategy_ids,
        set(REQUIRED_CALIBRATION_STRATEGY_IDS),
    )
    return set(expected_strategy_ids)


def _required_string_list(payload: dict[str, Any], field_name: str) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list) or not value:
        raise CalibrationReportError(f"calibration report must include non-empty {field_name} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise CalibrationReportError(f"calibration report {field_name} must contain only non-empty strings")
    duplicates = _duplicates(value)
    if duplicates:
        raise CalibrationReportError(
            f"calibration report {field_name} has duplicate ids: {', '.join(duplicates)}"
        )
    return value


def _validate_exact_members(field_name: str, actual: list[str], expected: set[str]) -> None:
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unknown = sorted(actual_set - expected)
    if missing:
        raise CalibrationReportError(
            f"calibration report {field_name} missing required ids: {', '.join(missing)}"
        )
    if unknown:
        raise CalibrationReportError(
            f"calibration report {field_name} has unknown ids: {', '.join(unknown)}"
        )


def _validate_strategy_coverage(strategy_results: list[Any], expected_strategy_ids: set[str]) -> None:
    strategy_ids = [str(result["strategy_id"]) for result in strategy_results]
    duplicates = _duplicates(strategy_ids)
    if duplicates:
        raise CalibrationReportError(
            "calibration report has duplicate strategy results: " + ", ".join(duplicates)
        )
    actual = set(strategy_ids)
    missing = sorted(expected_strategy_ids - actual)
    unknown = sorted(actual - expected_strategy_ids)
    if missing:
        raise CalibrationReportError(
            "calibration report missing strategy results: " + ", ".join(missing)
        )
    if unknown:
        raise CalibrationReportError(
            "calibration report has unknown strategy results: " + ", ".join(unknown)
        )


def _validate_strategy_result(result: Any, index: int) -> None:
    if not isinstance(result, dict):
        raise CalibrationReportError(f"strategy_results[{index}] must be an object")
    if not result.get("strategy_id"):
        raise CalibrationReportError(f"strategy_results[{index}] missing strategy_id")
    if _positive_int(result.get("broker_fill_sample_count")) <= 0:
        raise CalibrationReportError(
            f"strategy_results[{index}] must include broker_fill_sample_count > 0"
        )
    if _positive_int(result.get("backtest_fill_sample_count")) <= 0:
        raise CalibrationReportError(
            f"strategy_results[{index}] must include backtest_fill_sample_count > 0"
        )
    if result.get("within_tolerance") is not True:
        raise CalibrationReportError(f"strategy_results[{index}] is not within tolerance")

    _validate_metrics_object(result.get("broker_metrics"), index, "broker_metrics")
    _validate_metrics_object(result.get("backtest_metrics"), index, "backtest_metrics")

    tolerance_results = result.get("tolerance_results")
    if not isinstance(tolerance_results, dict) or not tolerance_results:
        raise CalibrationReportError(f"strategy_results[{index}] missing tolerance_results")
    keys = set(tolerance_results)
    missing = sorted(REQUIRED_TOLERANCE_KEYS - keys)
    unknown = sorted(keys - REQUIRED_TOLERANCE_KEYS)
    if missing:
        raise CalibrationReportError(
            f"strategy_results[{index}] missing tolerance keys: {', '.join(missing)}"
        )
    if unknown:
        raise CalibrationReportError(
            f"strategy_results[{index}] has unknown tolerance keys: {', '.join(unknown)}"
        )
    for name, value in tolerance_results.items():
        if value is not True:
            raise CalibrationReportError(
                f"strategy_results[{index}] tolerance {name!r} is not true"
            )


def _validate_metrics_object(metrics: Any, index: int, field_name: str) -> None:
    if not isinstance(metrics, dict):
        raise CalibrationReportError(f"strategy_results[{index}] missing {field_name}")
    missing = sorted(REQUIRED_METRIC_KEYS - set(metrics))
    if missing:
        raise CalibrationReportError(
            f"strategy_results[{index}] {field_name} missing metric keys: {', '.join(missing)}"
        )
    for key in REQUIRED_METRIC_KEYS:
        try:
            float(metrics[key])
        except (TypeError, ValueError):
            raise CalibrationReportError(
                f"strategy_results[{index}] {field_name}.{key} must be numeric"
            ) from None


def _positive_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _parse_datetime(value: Any, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationReportError(f"calibration report {field_name} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
