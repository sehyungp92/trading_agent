from __future__ import annotations

import pytest

from backtests.shared.parity.diagnostic_baselines import (
    collect_baseline_snapshot,
    default_manifest_path,
    load_manifest,
    repo_root,
    validate_manifest_entry,
)


_MANIFEST_PATH = default_manifest_path()
_MANIFEST = load_manifest() if _MANIFEST_PATH.exists() else {"artifacts": []}
_REPO_ROOT = repo_root()


def test_canonical_backtest_baseline_manifest_is_present() -> None:
    assert _MANIFEST_PATH.exists(), f"missing canonical backtest baseline manifest: {_MANIFEST_PATH}"


@pytest.mark.parametrize("entry", _MANIFEST["artifacts"], ids=lambda entry: entry["id"])
def test_canonical_backtest_baselines_are_frozen(entry: dict) -> None:
    snapshot = collect_baseline_snapshot(entry, root=_REPO_ROOT)

    assert snapshot["sha256"] == entry["sha256"]
    for metric_name, expected_value in entry["expected_metrics"].items():
        assert snapshot["metrics"][metric_name] == pytest.approx(expected_value, rel=0.0, abs=1e-9)


@pytest.mark.parametrize("entry", _MANIFEST["artifacts"], ids=lambda entry: entry["id"])
def test_canonical_backtest_manifest_entries_include_regeneration_metadata(entry: dict) -> None:
    validate_manifest_entry(entry)


def test_manifest_rejects_non_string_regeneration_arguments() -> None:
    entry = {
        "id": "demo",
        "artifact_path": "demo/output.txt",
        "parser_kind": "demo",
        "sha256": "abc",
        "expected_metrics": {"foo": 1.0},
        "regeneration": {
            "executor": "python_file",
            "entrypoint": "tools/demo.py",
            "arguments": ["--ok", 123],
            "expected_output": "demo/output.txt",
        },
    }

    with pytest.raises(ValueError, match="arguments must be a list of strings"):
        validate_manifest_entry(entry)
