from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "backtests" / "data_portability_manifest.json"
REPORT_PATH = ROOT / "artifacts" / "validation" / "backtest_data_portability_report.json"
REFERENCE_TOKEN = "_ref" "erences"
DOCKERIGNORE_REQUIRED = (
    "bots/*/backtests/*/data/raw/",
    "bots/crypto_trader/data/",
    "bots/*/data/backtests/output/",
)
GITIGNORE_REQUIRED = ("bots/*/data/backtests/output/",)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    bot: str
    path: str
    storage_class: str
    required: bool
    required_for: tuple[str, ...]
    description: str
    canonical_replacement_path: str = ""


DATASETS = (
    DatasetSpec(
        "ibkr_stock_raw_market_data",
        "ibkr",
        "bots/ibkr_trading/backtests/stock/data/raw",
        "git_lfs",
        True,
        ("A2", "A13", "A14", "A17"),
        "Historical stock-family market data required to reproduce frozen IBKR optimizer rounds.",
    ),
    DatasetSpec(
        "ibkr_momentum_raw_market_data",
        "ibkr",
        "bots/ibkr_trading/backtests/momentum/data/raw",
        "git_lfs",
        True,
        ("A2", "A13", "A14", "A17"),
        "Historical momentum-family market data required to reproduce frozen IBKR optimizer rounds.",
    ),
    DatasetSpec(
        "ibkr_swing_raw_market_data",
        "ibkr",
        "bots/ibkr_trading/backtests/swing/data/raw",
        "git_lfs",
        True,
        ("A2", "A13", "A14", "A17"),
        "Historical swing-family market data required to reproduce frozen IBKR optimizer rounds.",
    ),
    DatasetSpec(
        "ibkr_regime_raw_market_data",
        "ibkr",
        "bots/ibkr_trading/backtests/regime/data/raw",
        "git_lfs",
        True,
        ("A2", "A13", "A14", "A17"),
        "Historical regime-family market data required to reproduce frozen IBKR optimizer rounds.",
    ),
    DatasetSpec(
        "crypto_market_data",
        "crypto",
        "bots/crypto_trader/data",
        "git_lfs",
        True,
        ("A3", "A4", "A13", "A14", "A17"),
        "Hyperliquid candle, funding, and asset metadata used by frozen crypto strategy and portfolio rounds.",
    ),
    DatasetSpec(
        "k_stock_frozen_baseline_evidence",
        "k_stock",
        "backtests/baselines/k_stock",
        "source_control",
        True,
        ("A5", "A13", "A14", "A17"),
        "Frozen accepted K-stock latest-round evidence. The multi-gigabyte generated output tree is not source truth.",
    ),
    DatasetSpec(
        "k_stock_restored_local_output_evidence",
        "k_stock",
        "bots/k_stock_trader/data/backtests/output",
        "local_generated_excluded",
        False,
        ("A5",),
        "Optional restored local output tree used only for local audit cross-checks.",
        canonical_replacement_path="backtests/baselines/k_stock",
    ),
)


def main() -> int:
    args = _parser().parse_args()
    if args.write_manifest:
        _write_manifest()
    errors: list[str] = []
    manifest = _read_manifest(errors)
    records: list[dict[str, Any]] = []
    if manifest:
        errors.extend(_forbidden_reference_errors(manifest))
        records.extend(_verify_datasets(manifest, args.bot, errors))
    errors.extend(_repo_policy_errors())
    report = {
        "valid": not errors,
        "bot": args.bot,
        "manifest_path": _relative(MANIFEST_PATH),
        "datasets": records,
        "policy_checks": {
            "dockerignore_required": list(DOCKERIGNORE_REQUIRED),
            "gitignore_required": list(GITIGNORE_REQUIRED),
            "parquet_lfs_required": True,
        },
        "errors": errors,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _write_json(REPORT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify backtest data required for latest-round reproduction is portable.")
    parser.add_argument("--bot", choices=["all", "ibkr", "crypto", "k_stock"], default="all")
    parser.add_argument("--write-manifest", action="store_true", help="Refresh the monorepo-owned data manifest.")
    return parser


def _write_manifest() -> None:
    payload = {
        "schema_version": 1,
        "manifest_path": _relative(MANIFEST_PATH),
        "generated_by": "tools/verify_backtest_data_portability.py --write-manifest",
        "policy": {
            "repo_owned_paths_only": True,
            "no_legacy_snapshot_paths": True,
            "large_generated_outputs": "exclude_from_source_control_and_docker_contexts; freeze selected accepted evidence under backtests/baselines",
        },
        "datasets": [_dataset_manifest_record(spec) for spec in DATASETS],
    }
    _write_json(MANIFEST_PATH, payload)


def _dataset_manifest_record(spec: DatasetSpec) -> dict[str, Any]:
    path = ROOT / spec.path
    record: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "bot": spec.bot,
        "path": spec.path,
        "storage_class": spec.storage_class,
        "required": spec.required,
        "required_for": list(spec.required_for),
        "description": spec.description,
    }
    if spec.canonical_replacement_path:
        record["canonical_replacement_path"] = spec.canonical_replacement_path
    if path.exists():
        record.update(_tree_snapshot(path))
    else:
        record.update({"exists": False, "file_count": 0, "size_bytes": 0, "tree_sha256": ""})
    return record


def _read_manifest(errors: list[str]) -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        errors.append(f"missing data portability manifest {_relative(MANIFEST_PATH)}")
        return None
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid data portability manifest JSON: {exc}")
        return None
    if manifest.get("schema_version") != 1:
        errors.append(f"unsupported data portability manifest schema {manifest.get('schema_version')!r}")
    return manifest


def _verify_datasets(manifest: dict[str, Any], bot: str, errors: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifests_by_id = {
        str(record.get("dataset_id")): record
        for record in manifest.get("datasets", [])
        if isinstance(record, dict)
    }
    for spec in DATASETS:
        if bot != "all" and spec.bot != bot:
            continue
        expected = manifests_by_id.get(spec.dataset_id)
        if expected is None:
            errors.append(f"{spec.dataset_id}: missing manifest record")
            records.append({"dataset_id": spec.dataset_id, "status": "fail", "reason": "missing_manifest_record"})
            continue
        records.append(_verify_dataset_record(spec, expected, errors))
    return records


def _verify_dataset_record(spec: DatasetSpec, expected: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    identity_errors: list[str] = []
    content_errors: list[str] = []
    for key, value in (
        ("bot", spec.bot),
        ("path", spec.path),
        ("storage_class", spec.storage_class),
        ("required", spec.required),
    ):
        if expected.get(key) != value:
            identity_errors.append(f"manifest {key} is {expected.get(key)!r}, expected {value!r}")
    path = ROOT / spec.path
    if not path.exists():
        if identity_errors:
            status = "fail"
        elif spec.required:
            content_errors.append(f"required dataset path missing: {spec.path}")
            status = "fail"
        else:
            status = "skipped_optional"
        record_errors = [*identity_errors, *content_errors]
        errors.extend(f"{spec.dataset_id}: {error}" for error in record_errors)
        return {
            "dataset_id": spec.dataset_id,
            "bot": spec.bot,
            "path": spec.path,
            "required": spec.required,
            "status": status,
            "errors": record_errors,
        }

    actual = _tree_snapshot(path)
    for key in ("file_count", "size_bytes", "tree_sha256"):
        if actual[key] != expected.get(key):
            content_errors.append(f"{key} drift: actual {actual[key]!r}, expected {expected.get(key)!r}")
    if actual.get("file_records") != expected.get("file_records"):
        content_errors.append("file_records drift")
    record_errors = [*identity_errors, *content_errors]
    if spec.required:
        errors.extend(f"{spec.dataset_id}: {error}" for error in record_errors)
    else:
        errors.extend(f"{spec.dataset_id}: {error}" for error in identity_errors)
    status = "pass"
    if identity_errors or (spec.required and content_errors):
        status = "fail"
    elif content_errors:
        status = "observed_optional_drift"
    return {
        "dataset_id": spec.dataset_id,
        "bot": spec.bot,
        "path": spec.path,
        "required": spec.required,
        "storage_class": spec.storage_class,
        "file_count": actual["file_count"],
        "size_bytes": actual["size_bytes"],
        "tree_sha256": actual["tree_sha256"],
        "status": status,
        "errors": record_errors,
    }


def _tree_snapshot(root: Path) -> dict[str, Any]:
    file_records = []
    total_size = 0
    for path in _iter_files(root):
        size = path.stat().st_size
        total_size += size
        file_records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": _file_sha256(path),
            }
        )
    digest = hashlib.sha256()
    for record in file_records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "exists": True,
        "file_count": len(file_records),
        "size_bytes": total_size,
        "tree_sha256": digest.hexdigest(),
        "file_records": file_records,
    }


def _iter_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_policy_errors() -> list[str]:
    errors: list[str] = []
    dockerignore = ROOT / ".dockerignore"
    if not dockerignore.exists():
        errors.append("missing .dockerignore")
    else:
        text = dockerignore.read_text(encoding="utf-8")
        errors.extend(f".dockerignore missing {pattern}" for pattern in DOCKERIGNORE_REQUIRED if pattern not in text)
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        errors.append("missing .gitignore")
    else:
        text = gitignore.read_text(encoding="utf-8")
        errors.extend(f".gitignore missing {pattern}" for pattern in GITIGNORE_REQUIRED if pattern not in text)
    gitattributes = ROOT / ".gitattributes"
    if not gitattributes.exists():
        errors.append("missing .gitattributes")
    elif "*.parquet filter=lfs" not in gitattributes.read_text(encoding="utf-8"):
        errors.append(".gitattributes must route parquet historical data through Git LFS")
    return errors


def _forbidden_reference_errors(payload: Any) -> list[str]:
    errors: list[str] = []
    forbidden_fragments = _forbidden_fragments()
    for path, value in _walk_strings(payload):
        normalized = value.replace("\\", "/")
        for fragment in forbidden_fragments:
            if fragment and fragment in normalized:
                errors.append(f"manifest string {path} points outside monorepo-owned evidence")
                break
    return errors


def _forbidden_fragments() -> tuple[str, ...]:
    sibling_names = ("trading", "crypto_trader", "k_stock_trader", "trading_assistant_agent")
    sibling_paths = tuple(str((ROOT.parent / name).resolve()).replace("\\", "/") for name in sibling_names)
    return (REFERENCE_TOKEN, *sibling_paths)


def _walk_strings(payload: Any, prefix: str = "$") -> list[tuple[str, str]]:
    if isinstance(payload, str):
        return [(prefix, payload)]
    if isinstance(payload, list):
        values: list[tuple[str, str]] = []
        for index, item in enumerate(payload):
            values.extend(_walk_strings(item, f"{prefix}[{index}]"))
        return values
    if isinstance(payload, dict):
        values = []
        for key, item in payload.items():
            values.extend(_walk_strings(item, f"{prefix}.{key}"))
        return values
    return []


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


if __name__ == "__main__":
    sys.exit(main())
