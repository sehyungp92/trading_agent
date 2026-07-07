"""Finalize slice manifests against a committed data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bundle_builder import _canonical_paths_for_manifests, _rel, _select_slice_manifests
from .checksums import parquet_content_checksum
from .manifests import MarketDataManifest, write_model
from .slices import SliceWrite
from .slices.writer import update_slice_index
from .repo import git_commit_exists, git_commit_sha, git_dirty_paths
from .validation import market_manifest_errors

SOURCE_VERSION_REASON = "source_version is not a data repo commit SHA"


@dataclass(frozen=True)
class FinalizeSlicesResult:
    data_commit_sha: str
    updated: list[str]
    skipped: list[dict[str, str]]
    dry_run: bool

    def to_dict(self) -> dict:
        return {
            "data_commit_sha": self.data_commit_sha,
            "updated": self.updated,
            "skipped": self.skipped,
            "updated_count": len(self.updated),
            "skipped_count": len(self.skipped),
            "dry_run": self.dry_run,
        }


def finalize_slice_manifests(
    *,
    repo_root: Path,
    run_month: str,
    slice_manifest_paths: list[Path] | None = None,
    requirements_path: Path | None = None,
    data_commit_sha: str | None = None,
    dry_run: bool = False,
) -> FinalizeSlicesResult:
    data_commit = data_commit_sha or git_commit_sha(repo_root)
    if not data_commit or not git_commit_exists(repo_root, data_commit):
        raise ValueError("finalize-slices requires a valid data repo commit SHA")
    manifests = _select_slice_manifests(repo_root, run_month, slice_manifest_paths, requirements_path)
    if not manifests:
        raise ValueError(f"no slice manifests found for run month {run_month}")

    updated: list[str] = []
    skipped: list[dict[str, str]] = []
    slice_writes: list[SliceWrite] = []
    for manifest_path, manifest in manifests:
        canonical_paths = _canonical_paths_for_manifests(repo_root, [manifest.manifest_id])
        if not canonical_paths:
            skipped.append({"manifest_path": _rel(manifest_path, repo_root), "reason": "canonical paths missing"})
            continue
        dirty_paths = git_dirty_paths(repo_root, canonical_paths)
        if dirty_paths:
            skipped.append(
                {
                    "manifest_path": _rel(manifest_path, repo_root),
                    "reason": "canonical paths are not committed cleanly: " + ", ".join(dirty_paths[:5]),
                }
            )
            continue
        finalized = _finalized_manifest(manifest, data_commit, canonical_paths)
        if market_manifest_errors(finalized):
            skipped.append(
                {
                    "manifest_path": _rel(manifest_path, repo_root),
                    "reason": "; ".join(market_manifest_errors(finalized)),
                }
            )
            continue
        if not dry_run:
            write_model(manifest_path, finalized)
        updated.append(_rel(manifest_path, repo_root))
        slice_writes.append(SliceWrite(manifest_path, canonical_paths, finalized))
    if not dry_run:
        update_slice_index(repo_root, slice_writes)
    return FinalizeSlicesResult(data_commit, updated, skipped, dry_run)


def _finalized_manifest(
    manifest: MarketDataManifest,
    data_commit_sha: str,
    canonical_paths: list[Path],
) -> MarketDataManifest:
    checksum = parquet_content_checksum(canonical_paths[0])
    blocking_reasons = [
        reason
        for reason in manifest.blocking_reasons
        if reason != SOURCE_VERSION_REASON and not reason.startswith("bundle is ")
    ]
    payload = manifest.model_dump()
    payload.update(
        {
            "checksum": checksum,
            "source_version": data_commit_sha,
            "blocking_reasons": blocking_reasons,
            "usable_for_authoritative_validation": not blocking_reasons,
        }
    )
    finalized = MarketDataManifest.model_validate(payload)
    finalized.usable_for_authoritative_validation = not market_manifest_errors(finalized)
    return finalized
