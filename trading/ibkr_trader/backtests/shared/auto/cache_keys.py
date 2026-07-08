from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_signature(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_signature_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mutation_subset(
    mutations: dict[str, Any],
    *,
    exact_keys: Iterable[str] = (),
    prefixes: Iterable[str] = (),
) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for key in exact_keys:
        if key in mutations:
            subset[key] = mutations[key]

    for prefix in prefixes:
        normalized = prefix if prefix.endswith(".") else f"{prefix}."
        for key, value in mutations.items():
            if key == prefix or key.startswith(normalized):
                subset[key] = value

    return {key: subset[key] for key in sorted(subset)}


def build_cache_key(
    namespace: str,
    *,
    source_fingerprint: str = "",
    mutations: dict[str, Any] | None = None,
    mutation_exact_keys: Iterable[str] = (),
    mutation_prefixes: Iterable[str] = (),
    extra: dict[str, Any] | None = None,
) -> str:
    exact_keys = tuple(mutation_exact_keys)
    prefixes = tuple(mutation_prefixes)
    if mutations is None:
        mutation_payload: dict[str, Any] | None = None
    elif exact_keys or prefixes:
        mutation_payload = mutation_subset(
            mutations,
            exact_keys=exact_keys,
            prefixes=prefixes,
        )
    else:
        mutation_payload = {key: mutations[key] for key in sorted(mutations)}

    return stable_signature(
        {
            "namespace": namespace,
            "source_fingerprint": source_fingerprint,
            "mutations": mutation_payload,
            "extra": extra or {},
        }
    )


def fingerprint_paths(paths: Iterable[Path], *, root: Path | None = None) -> str:
    entries: list[dict[str, Any]] = []
    for raw_path in sorted({Path(path) for path in paths}, key=lambda path: str(path).lower()):
        path = Path(raw_path)
        exists = path.exists()
        stat = path.stat() if exists else None
        try:
            rel = str(path.relative_to(root)) if root is not None else str(path)
        except ValueError:
            rel = str(path)
        entries.append(
            {
                "path": rel.replace("\\", "/"),
                "exists": exists,
                "size": stat.st_size if stat is not None else None,
                "mtime_ns": stat.st_mtime_ns if stat is not None else None,
            }
        )
    return stable_signature(entries)


def fingerprint_tree(
    root: Path,
    *,
    patterns: Iterable[str] = ("*.parquet",),
    recursive: bool = True,
) -> str:
    root = Path(root)
    paths: list[Path] = []
    for pattern in patterns:
        if recursive:
            paths.extend(root.rglob(pattern))
        else:
            paths.extend(root.glob(pattern))
    return fingerprint_paths(paths, root=root)


def _signature_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)
