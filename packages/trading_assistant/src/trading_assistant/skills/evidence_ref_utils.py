"""Shared validation for evidence references used by offline learning tools."""
from __future__ import annotations

from pathlib import Path

VIRTUAL_EVIDENCE_PREFIXES = ("benchmark:", "retrospective:", "outcome:", "transfer:")


def is_virtual_evidence_ref(ref: str) -> bool:
    """Return True for stable ledger-style references that are not filesystem paths."""
    return str(ref or "").strip().startswith(VIRTUAL_EVIDENCE_PREFIXES)


def evidence_ref_exists_within_roots(ref: str, roots: list[Path]) -> bool:
    """Check that a concrete evidence path exists and is contained by an allowed root.

    Absolute paths must resolve under one of the supplied roots. Relative paths are
    resolved against each root and must not contain traversal segments.
    """
    value = str(ref or "").strip()
    if not value:
        return False
    normalized = value.replace("\\", "/")
    if ".." in normalized.split("/"):
        return False
    if is_virtual_evidence_ref(value):
        return True

    clean_roots = [Path(root).resolve() for root in roots if root]
    path = Path(value)
    if path.is_absolute():
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(_is_relative_to(resolved, root) for root in clean_roots) and resolved.exists()

    for root in clean_roots:
        candidate = (root / path).resolve()
        if _is_relative_to(candidate, root) and candidate.exists():
            return True
    return False


def evidence_ref_allowed(
    ref: str,
    *,
    allowed_refs: list[str] | None = None,
    roots: list[Path] | None = None,
    require_allowed: bool = False,
) -> bool:
    """Return True when a ref is virtual or resolves to an allowed contained path."""
    value = str(ref or "").strip()
    if not value:
        return False
    allowed = [str(item).strip() for item in (allowed_refs or []) if str(item).strip()]
    root_list = roots or []
    if require_allowed and not _matches_allowed_ref(value, allowed, root_list):
        return False
    if is_virtual_evidence_ref(value):
        return True
    return evidence_ref_exists_within_roots(value, root_list)


def _matches_allowed_ref(ref: str, allowed_refs: list[str], roots: list[Path]) -> bool:
    if ref in allowed_refs:
        return True
    if is_virtual_evidence_ref(ref):
        return False
    ref_paths = _resolved_candidates(ref, roots)
    if not ref_paths:
        return False
    allowed_paths = [
        candidate
        for allowed in allowed_refs
        if not is_virtual_evidence_ref(allowed)
        for candidate in _resolved_candidates(allowed, roots)
    ]
    return bool(set(ref_paths) & set(allowed_paths))


def _resolved_candidates(ref: str, roots: list[Path]) -> list[Path]:
    value = str(ref or "").strip()
    if not value or ".." in value.replace("\\", "/").split("/"):
        return []
    path = Path(value)
    clean_roots: list[Path] = []
    for root in roots:
        if not root:
            continue
        try:
            clean_roots.append(Path(root).resolve())
        except OSError:
            continue
    candidates: list[Path] = []
    if path.is_absolute():
        try:
            resolved = path.resolve()
        except OSError:
            return []
        if not clean_roots or any(_is_relative_to(resolved, root) for root in clean_roots):
            candidates.append(resolved)
        return candidates
    for root in clean_roots:
        try:
            candidate = (root / path).resolve()
        except OSError:
            continue
        if _is_relative_to(candidate, root):
            candidates.append(candidate)
    return candidates


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
