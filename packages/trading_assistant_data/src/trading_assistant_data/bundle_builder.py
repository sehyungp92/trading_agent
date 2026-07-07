"""Build monthly data bundles and compatibility exports."""

from __future__ import annotations

import json
from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .io import read_json, write_json
from .manifests import (
    DataBundleManifest,
    DataBundleSlice,
    DataBundleStatus,
    MarketDataManifest,
    load_bundle_manifest,
    load_market_manifest,
    write_model,
)
from .normalization import DEFAULT_FEE_MODEL, DEFAULT_SLIPPAGE_MODEL
from .repo import git_branch, git_commit_exists, is_git_commit_sha
from .validation import validate_market_manifest


@dataclass(frozen=True)
class BundleBuildResult:
    bundle_path: Path
    slice_index_path: Path
    bundle: DataBundleManifest
    canonical_paths: list[Path]


def build_bundle(
    *,
    repo_root: Path,
    run_month: str,
    bot_id: str,
    strategy_id: str,
    slice_manifest_paths: list[Path] | None = None,
    requirements_path: Path | None = None,
    dry_run: bool = False,
) -> BundleBuildResult:
    if not slice_manifest_paths and requirements_path is None:
        raise ValueError("build-bundle requires --slice-manifest or --requirements-file")
    requirements = _load_bundle_requirements(repo_root, requirements_path)
    manifests = _select_slice_manifests(repo_root, run_month, slice_manifest_paths, requirements_path)
    if not manifests:
        raise ValueError(f"no slice manifests found for run month {run_month}")
    data_commit_sha, source_errors = _common_source_version(repo_root, manifests)
    slices = [
        DataBundleSlice(
            manifest_path=_rel(path, repo_root),
            manifest_id=manifest.manifest_id,
            source=manifest.source,
            market=manifest.market,
            symbol=manifest.symbol,
            timeframe=manifest.timeframe,
            start_ts=manifest.start_ts,
            end_ts=manifest.end_ts,
            checksum=manifest.checksum,
            calendar=manifest.session_calendar,
            authoritative=manifest.usable_for_authoritative_validation,
        )
        for path, manifest in manifests
    ]
    fee_model_version, fee_errors = _single_policy_value(
        [manifest.fee_model_version for _path, manifest in manifests],
        field_name="fee_model_version",
        default=DEFAULT_FEE_MODEL,
    )
    slippage_model_version, slippage_errors = _single_policy_value(
        [manifest.slippage_model_version for _path, manifest in manifests],
        field_name="slippage_model_version",
        default=DEFAULT_SLIPPAGE_MODEL,
    )
    adjustment_policy, adjustment_errors = _bundle_adjustment_policy(manifests)
    blocking_reasons = [
        *source_errors,
        *_non_authoritative_slice_reasons(manifests),
        *_slice_manifest_validation_reasons(repo_root, manifests),
        *_missing_required_slice_reasons(manifests, requirements),
        *fee_errors,
        *slippage_errors,
        *adjustment_errors,
    ]
    status = DataBundleStatus.AUTHORITATIVE if not blocking_reasons else DataBundleStatus.DIAGNOSTICS_ONLY
    diagnostics = "; ".join(blocking_reasons)
    bundle = DataBundleManifest(
        data_repo_path=".",
        data_repo_commit_sha=data_commit_sha,
        data_repo_branch=git_branch(repo_root),
        slice_manifests=slices,
        calendars=sorted({item.calendar for item in slices if item.calendar}),
        fee_model_version=fee_model_version,
        slippage_model_version=slippage_model_version,
        adjustment_policy=adjustment_policy,
        status=status,
        diagnostics_only_reason=diagnostics,
    )
    bundle_root = Path(repo_root) / "data" / "bundles" / "monthly" / run_month / bot_id / strategy_id
    bundle_path = bundle_root / "data_bundle_manifest.json"
    slice_index_path = bundle_root / "slice_index.json"
    canonical_paths = _canonical_paths_for_manifests(repo_root, [manifest.manifest_id for _path, manifest in manifests])
    if not dry_run:
        write_model(bundle_path, bundle)
        write_json(
            slice_index_path,
            {
                "schema_version": "bundle_slice_index_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bundle_checksum": bundle.bundle_checksum,
                "slices": [
                    {
                        "manifest_id": manifest.manifest_id,
                        "manifest_path": _rel(path, repo_root),
                        "canonical_paths": [
                            _rel(canonical_path, repo_root)
                            for canonical_path in _canonical_paths_for_manifests(repo_root, [manifest.manifest_id])
                        ],
                    }
                    for path, manifest in manifests
                ],
            },
        )
    return BundleBuildResult(
        bundle_path=bundle_path,
        slice_index_path=slice_index_path,
        bundle=bundle,
        canonical_paths=canonical_paths,
    )


def export_filesystem(
    *,
    repo_root: Path,
    run_month: str,
    bundle_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    bundle_path = bundle_manifest_path or _latest_bundle_for_month(repo_root, run_month)
    bundle = load_bundle_manifest(bundle_path)
    exported: list[str] = []
    slice_index = read_json(Path(bundle_path).with_name("slice_index.json"))
    by_id = {item["manifest_id"]: item for item in slice_index.get("slices", [])}
    for item in bundle.slice_manifests:
        if item.timeframe.startswith("funding_"):
            continue
        index_entry = by_id.get(item.manifest_id, {})
        canonical_paths = [Path(repo_root) / path for path in index_entry.get("canonical_paths", [])]
        if not canonical_paths:
            canonical_paths = _canonical_paths_for_manifests(repo_root, [item.manifest_id])
        frames = [pd.read_parquet(path) for path in canonical_paths if path.exists()]
        if not frames:
            continue
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
        output_path = (
            Path(repo_root)
            / "data"
            / "export"
            / "filesystem"
            / item.market
            / item.symbol
            / item.timeframe
            / f"{run_month}.parquet"
        )
        exported.append(str(output_path))
        if dry_run:
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, engine="pyarrow", index=False)
    return {
        "bundle_manifest_path": str(bundle_path),
        "run_month": run_month,
        "exported": exported,
        "dry_run": dry_run,
    }


def export_single_slice_coverage(
    *,
    repo_root: Path,
    run_month: str,
    bot_id: str,
    strategy_id: str,
    bundle_manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict:
    bundle_path = bundle_manifest_path or _latest_bundle_for_month(repo_root, run_month)
    bundle = load_bundle_manifest(bundle_path)
    output_path = (
        Path(repo_root)
        / "data"
        / "export"
        / "manifests"
        / bot_id
        / strategy_id
        / f"{run_month}.coverage_manifest.json"
    )
    if len(bundle.slice_manifests) != 1:
        return {
            "status": "skipped",
            "reason": "bundle contains multiple slices; direct DataBundleManifest handoff required",
            "path": str(output_path),
        }
    slice_item = bundle.slice_manifests[0]
    manifest_path = Path(repo_root) / slice_item.manifest_path
    manifest = load_market_manifest(manifest_path)
    if bundle.status != DataBundleStatus.AUTHORITATIVE:
        manifest.usable_for_authoritative_validation = False
        reasons = list(manifest.blocking_reasons)
        reasons.append(f"bundle is {bundle.status.value}: {bundle.diagnostics_only_reason}")
        manifest.blocking_reasons = reasons
    if not dry_run:
        write_model(output_path, manifest)
    return {
        "status": "written" if not dry_run else "planned",
        "path": str(output_path),
        "manifest_id": manifest.manifest_id,
        "bundle_checksum": bundle.bundle_checksum,
        "dry_run": dry_run,
    }


def audit_coverage(repo_root: Path, *, run_month: str) -> dict:
    manifests = _select_slice_manifests(repo_root, run_month, None, None)
    return {
        "run_month": run_month,
        "slice_count": len(manifests),
        "authoritative_count": sum(1 for _path, item in manifests if item.usable_for_authoritative_validation),
        "diagnostics_only": [
            {
                "manifest_path": _rel(path, repo_root),
                "symbol": item.symbol,
                "timeframe": item.timeframe,
                "blocking_reasons": item.blocking_reasons,
            }
            for path, item in manifests
            if not item.usable_for_authoritative_validation
        ],
    }


def _select_slice_manifests(
    repo_root: Path,
    run_month: str,
    slice_manifest_paths: list[Path] | None,
    requirements_path: Path | None,
) -> list[tuple[Path, MarketDataManifest]]:
    requirements = _load_bundle_requirements(repo_root, requirements_path)
    indexed_paths = _indexed_slice_manifest_paths(repo_root)
    if slice_manifest_paths:
        candidates = [Path(path) if Path(path).is_absolute() else Path(repo_root) / path for path in slice_manifest_paths]
        unindexed = [path for path in candidates if path.resolve() not in indexed_paths]
        if unindexed:
            preview = ", ".join(_rel(path, repo_root) for path in unindexed[:5])
            raise ValueError(f"slice manifests are not present in slice_index.json: {preview}")
    else:
        candidates = sorted(_indexed_slice_manifest_paths(repo_root, requirements=requirements))
    month_start, month_end = _month_window(run_month)
    selected: list[tuple[Path, MarketDataManifest]] = []
    for path in candidates:
        manifest = load_market_manifest(path)
        if requirements and not _matches_any_requirement(manifest, requirements):
            continue
        if (
            manifest.end_ts < month_start or manifest.start_ts > month_end
        ) and not _allow_archived_requirement_without_month_overlap(manifest, requirements):
            continue
        selected.append((path, manifest))
    if requirements:
        selected = _prefer_authoritative_identity_slices(selected, month_start, month_end)
    selected = _drop_contained_duplicate_slices(selected)
    selected.sort(key=lambda item: (item[1].source, item[1].market, item[1].symbol, item[1].timeframe))
    return selected


def _prefer_authoritative_identity_slices(
    slices: list[tuple[Path, MarketDataManifest]],
    month_start: datetime,
    month_end: datetime,
) -> list[tuple[Path, MarketDataManifest]]:
    by_identity: dict[tuple[str, str, str, str], list[tuple[Path, MarketDataManifest]]] = {}
    for item in slices:
        by_identity.setdefault(_slice_identity(item[1]), []).append(item)
    selected: list[tuple[Path, MarketDataManifest]] = []
    for items in by_identity.values():
        valid_full_month = [
            item for item in items if _valid_authoritative(item[1]) and _covers_month(item[1], month_start, month_end)
        ]
        if valid_full_month:
            selected.extend(_best_full_month_slices(valid_full_month))
            continue
        authoritative = [
            item for item in items if _valid_authoritative(item[1])
        ]
        selected.extend(authoritative or items)
    return selected


def _valid_authoritative(manifest: MarketDataManifest) -> bool:
    return (
        manifest.usable_for_authoritative_validation is True
        and validate_market_manifest(manifest).valid
    )


def _covers_month(
    manifest: MarketDataManifest,
    month_start: datetime,
    month_end: datetime,
) -> bool:
    if manifest.timeframe.lower() in {"1d", "daily"}:
        return manifest.start_ts.date() <= month_start.date() and manifest.end_ts.date() >= month_end.date()
    return manifest.start_ts <= month_start and manifest.end_ts >= month_end - pd.Timedelta(minutes=1).to_pytimedelta()


def _best_full_month_slices(
    slices: list[tuple[Path, MarketDataManifest]],
) -> list[tuple[Path, MarketDataManifest]]:
    return [
        max(
            slices,
            key=lambda item: (
                item[1].generated_at,
                item[1].actual_bars,
                item[1].start_ts,
                item[1].end_ts,
            ),
        )
    ]


def _drop_contained_duplicate_slices(
    slices: list[tuple[Path, MarketDataManifest]],
) -> list[tuple[Path, MarketDataManifest]]:
    filtered: list[tuple[Path, MarketDataManifest]] = []
    for path, manifest in slices:
        if any(
            other_manifest is not manifest
            and _slice_identity(other_manifest) == _slice_identity(manifest)
            and _strictly_contains(other_manifest, manifest)
            for _other_path, other_manifest in slices
        ):
            continue
        filtered.append((path, manifest))
    return filtered


def _allow_archived_requirement_without_month_overlap(
    manifest: MarketDataManifest,
    requirements: list[dict[str, str]],
) -> bool:
    if not requirements:
        return False
    lineage = manifest.lineage or {}
    return (
        manifest.source == "ibkr"
        and manifest.market == "us_equity"
        and lineage.get("strategy_data_family") == "trading_stock"
        and lineage.get("authority_status")
        == "archived_ibkr_stock_updater_parquet_exact_declared_request"
        and lineage.get("archive_import_policy")
        == "source_owned_trading_stock_ibkr_updater_parquet_v1"
        and manifest.usable_for_authoritative_validation
    )


def _slice_identity(manifest: MarketDataManifest) -> tuple[str, str, str, str]:
    return (manifest.source, manifest.market, manifest.symbol.upper(), manifest.timeframe)


def _strictly_contains(outer: MarketDataManifest, inner: MarketDataManifest) -> bool:
    return (
        outer.start_ts <= inner.start_ts
        and outer.end_ts >= inner.end_ts
        and (outer.start_ts < inner.start_ts or outer.end_ts > inner.end_ts)
    )


def _indexed_slice_manifest_paths(
    repo_root: Path,
    *,
    requirements: list[dict[str, str]] | None = None,
) -> set[Path]:
    index_path = Path(repo_root) / "data" / "manifests" / "slices" / "slice_index.json"
    if not index_path.exists():
        return set()
    payload = read_json(index_path)
    paths: set[Path] = set()
    for item in payload.get("slices", []):
        if requirements and not _index_entry_matches_any_requirement(item, requirements):
            continue
        manifest_path = str(item.get("manifest_path", "")).strip()
        if not manifest_path:
            continue
        path = Path(repo_root) / manifest_path
        if path.exists():
            paths.add(path.resolve())
    return paths


def _index_entry_matches_any_requirement(
    item: dict,
    requirements: list[dict[str, str]],
) -> bool:
    return any(
        _matches_requirement_field(str(item.get("source", "")), requirement["source"])
        and _matches_requirement_field(str(item.get("market", "")), requirement["market"])
        and _matches_requirement_field(
            str(item.get("symbol", "")).upper(), requirement["symbol"].upper()
        )
        and _matches_requirement_field(str(item.get("timeframe", "")), requirement["timeframe"])
        for requirement in requirements
    )


def _canonical_paths_for_manifests(repo_root: Path, manifest_ids: list[str]) -> list[Path]:
    index_path = Path(repo_root) / "data" / "manifests" / "slices" / "slice_index.json"
    if not index_path.exists():
        return []
    index = read_json(index_path)
    wanted = set(manifest_ids)
    paths: list[Path] = []
    for item in index.get("slices", []):
        if item.get("manifest_id") in wanted:
            paths.extend(Path(repo_root) / path for path in item.get("canonical_paths", []))
    return paths


def _latest_bundle_for_month(repo_root: Path, run_month: str) -> Path:
    candidates = sorted(
        (Path(repo_root) / "data" / "bundles" / "monthly" / run_month).rglob("data_bundle_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"no data bundle manifest found for {run_month}")
    return candidates[0]


def _month_window(run_month: str) -> tuple[datetime, datetime]:
    year, month = (int(part) for part in run_month.split("-", 1))
    last_day = monthrange(year, month)[1]
    return (
        datetime(year, month, 1, tzinfo=timezone.utc),
        datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc),
    )


def _single_policy_value(values: list[str], *, field_name: str, default: str) -> tuple[str, list[str]]:
    cleaned = sorted({value for value in values if value})
    if len(cleaned) == 1:
        return cleaned[0], []
    if not cleaned:
        return default, [f"{field_name} missing across bundle"]
    return ",".join(cleaned), [f"{field_name} mismatch across bundle: {','.join(cleaned)}"]


def _bundle_adjustment_policy(manifests: list[tuple[Path, MarketDataManifest]]) -> tuple[str, list[str]]:
    values = sorted({manifest.adjustment_policy for _path, manifest in manifests if manifest.adjustment_policy})
    if len(values) == 1:
        return values[0], []
    if not values:
        return "mixed_adjustment_policy", ["adjustment_policy missing across bundle"]
    has_flow_panel = any("flow" in manifest.timeframe.lower() for _path, manifest in manifests)
    has_price_bars = any("flow" not in manifest.timeframe.lower() for _path, manifest in manifests)
    if has_flow_panel and has_price_bars:
        return f"mixed_adjustment_policy:{','.join(values)}", []
    return ",".join(values), [f"adjustment_policy mismatch across bundle: {','.join(values)}"]


def _common_source_version(
    repo_root: Path,
    manifests: list[tuple[Path, MarketDataManifest]],
) -> tuple[str, list[str]]:
    versions = sorted({manifest.source_version for _path, manifest in manifests if manifest.source_version})
    if len(versions) != 1:
        return "", ["slice source_version missing or mixed across bundle"]
    version = versions[0]
    if not is_git_commit_sha(version):
        return "", [f"slice source_version is not a git commit SHA: {version}"]
    if not git_commit_exists(repo_root, version):
        return version, [f"slice source_version commit is not available in this data repo: {version}"]
    return version, []


def _non_authoritative_slice_reasons(manifests: list[tuple[Path, MarketDataManifest]]) -> list[str]:
    reasons = [
        f"{manifest.symbol}:{manifest.timeframe}:{','.join(manifest.blocking_reasons or ['not authoritative'])}"
        for _path, manifest in manifests
        if not manifest.usable_for_authoritative_validation
    ]
    return [f"non-authoritative slices: {'; '.join(reasons)}"] if reasons else []


def _slice_manifest_validation_reasons(
    repo_root: Path,
    manifests: list[tuple[Path, MarketDataManifest]],
) -> list[str]:
    reasons = []
    for path, manifest in manifests:
        report = validate_market_manifest(manifest)
        if report.valid:
            continue
        reasons.append(f"{_rel(path, repo_root)}: {', '.join(report.errors)}")
    return [f"invalid slice manifests: {'; '.join(reasons)}"] if reasons else []


def _load_bundle_requirements(repo_root: Path, requirements_path: Path | None) -> list[dict[str, str]]:
    if requirements_path is None:
        return []
    path = Path(requirements_path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"requirements file must contain an object: {path}")
    raw_requirements = payload.get("requirements") or payload.get("slices") or []
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError(f"requirements file has no requirements: {path}")
    requirements: list[dict[str, str]] = []
    for item in raw_requirements:
        if not isinstance(item, dict):
            raise ValueError(f"requirement entry must be an object: {path}")
        requirement = {
            key: str(item.get(key, "")).strip()
            for key in ("source", "market", "symbol", "timeframe")
        }
        missing = [key for key, value in requirement.items() if not value]
        if missing:
            raise ValueError(f"requirement missing fields {missing}: {path}")
        requirement["symbol"] = requirement["symbol"].upper()
        for key in (
            "strategy_data_family",
            "family",
            "session_policy",
            "use_rth",
            "primary_exchange",
            "data_role",
        ):
            value = str(item.get(key, "")).strip()
            if value:
                requirement[key] = value
        if requirement.get("family") and not requirement.get("strategy_data_family"):
            requirement["strategy_data_family"] = requirement["family"]
        requirements.append(requirement)
    return requirements


def _matches_any_requirement(manifest: MarketDataManifest, requirements: list[dict[str, str]]) -> bool:
    return any(_manifest_matches_requirement(manifest, requirement) for requirement in requirements)


def _missing_required_slice_reasons(
    manifests: list[tuple[Path, MarketDataManifest]],
    requirements: list[dict[str, str]],
) -> list[str]:
    missing = [
        requirement
        for requirement in requirements
        if _is_concrete_requirement(requirement)
        and not any(_manifest_matches_requirement(manifest, requirement) for _path, manifest in manifests)
    ]
    if not missing:
        return []
    examples = ", ".join(_requirement_label(item) for item in missing[:10])
    suffix = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
    return [f"missing required slices: {examples}{suffix}"]


def _manifest_matches_requirement(
    manifest: MarketDataManifest,
    requirement: dict[str, str],
) -> bool:
    return (
        _matches_requirement_field(manifest.source, requirement["source"])
        and _matches_requirement_field(manifest.market, requirement["market"])
        and _matches_requirement_field(manifest.symbol.upper(), requirement["symbol"].upper())
        and _matches_requirement_field(manifest.timeframe, requirement["timeframe"])
        and _matches_optional_manifest_requirements(manifest, requirement)
    )


def _is_concrete_requirement(requirement: dict[str, str]) -> bool:
    return all(requirement.get(key, "") != "*" for key in ("source", "market", "symbol", "timeframe"))


def _requirement_label(requirement: dict[str, str]) -> str:
    label = (
        f"{requirement['source']}/{requirement['market']}/"
        f"{requirement['symbol']}/{requirement['timeframe']}"
    )
    family = requirement.get("strategy_data_family") or requirement.get("family") or ""
    role = requirement.get("data_role", "")
    if family or role:
        suffix = ":".join(item for item in (family, role) if item)
        label = f"{label} ({suffix})"
    return label


def _matches_requirement_field(value: str, requirement: str) -> bool:
    return requirement == "*" or value == requirement


def _matches_optional_manifest_requirements(
    manifest: MarketDataManifest,
    requirement: dict[str, str],
) -> bool:
    lineage = manifest.lineage or {}
    family = requirement.get("strategy_data_family") or requirement.get("family") or ""
    if family and family not in {
        lineage.get("strategy_data_family", ""),
        lineage.get("legacy_family", ""),
        lineage.get("data_family", ""),
    }:
        return False
    session_policy = requirement.get("session_policy", "")
    if session_policy and lineage.get("session_policy", "") != session_policy:
        return False
    use_rth = requirement.get("use_rth", "")
    if use_rth and str(lineage.get("use_rth", "")).lower() != use_rth.lower():
        return False
    primary_exchange = requirement.get("primary_exchange", "")
    if primary_exchange and primary_exchange != _lineage_primary_exchange(lineage):
        return False
    return True


def _lineage_primary_exchange(lineage: dict[str, str]) -> str:
    direct = str(lineage.get("primary_exchange", "")).upper().strip()
    if direct:
        return direct
    params = str(lineage.get("source_request_params_json", "")).strip()
    if not params:
        return ""
    try:
        payload = json.loads(params)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("primary_exchange", "")).upper().strip()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
