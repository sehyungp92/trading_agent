from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache_keys import fingerprint_tree, stable_signature

PROVENANCE_SCHEMA_VERSION = 1

PROVENANCE_SCOPES = {
    "selection",
    "diagnostics",
    "data",
    "source_artifact",
    "environment",
}
SELECTION_FINGERPRINT_SCOPES = {"selection", "data", "source_artifact"}

DRIFT_STATUS_CURRENT = "current"
DRIFT_STATUS_DIAGNOSTICS_STALE = "diagnostics_stale"
DRIFT_STATUS_SELECTION_STALE = "selection_stale"
DRIFT_STATUS_INCOMPLETE_SAVED_METRICS = "incomplete_saved_metrics"

DEFAULT_CRITICAL_METRICS = (
    "total_trades",
    "win_rate",
    "profit_factor",
    "max_drawdown_pct",
    "net_return_pct",
    "sharpe_ratio",
    "calmar_ratio",
)

_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class ProvenanceItem:
    name: str
    kind: str
    fingerprint: str
    paths: tuple[str, ...] = ()
    scope: str = "selection"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.scope not in PROVENANCE_SCOPES:
            raise ValueError(f"Unknown provenance scope {self.scope!r}; expected one of {sorted(PROVENANCE_SCOPES)}.")
        object.__setattr__(self, "paths", tuple(str(path).replace("\\", "/") for path in self.paths))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "paths": list(self.paths),
            "scope": self.scope,
            "notes": self.notes,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProvenanceItem":
        return cls(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            fingerprint=str(data.get("fingerprint", "")),
            paths=tuple(str(path) for path in data.get("paths", ()) or ()),
            scope=str(data.get("scope", "selection")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class AutoRunProvenance:
    schema_version: int
    selection_fingerprint: str
    diagnostics_fingerprint: str
    items: tuple[ProvenanceItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_fingerprint": self.selection_fingerprint,
            "diagnostics_fingerprint": self.diagnostics_fingerprint,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AutoRunProvenance":
        return cls(
            schema_version=int(data.get("schema_version", PROVENANCE_SCHEMA_VERSION)),
            selection_fingerprint=str(data.get("selection_fingerprint", "")),
            diagnostics_fingerprint=str(data.get("diagnostics_fingerprint", "")),
            items=tuple(ProvenanceItem.from_mapping(item) for item in data.get("items", ()) or ()),
        )


@dataclass(frozen=True)
class ProvenanceValidationResult:
    valid: bool
    status: str
    previous_round: int | None = None
    current_round: int | None = None
    selection_drift: bool = False
    diagnostics_drift: bool = False
    changed_items: tuple[str, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "previous_round": self.previous_round,
            "current_round": self.current_round,
            "selection_drift": self.selection_drift,
            "diagnostics_drift": self.diagnostics_drift,
            "changed_items": list(self.changed_items),
            "message": self.message,
        }


@dataclass(frozen=True)
class MetricDriftResult:
    status: str
    missing_saved_metrics: tuple[str, ...] = ()
    missing_current_metrics: tuple[str, ...] = ()
    deltas: dict[str, dict[str, Any]] | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "missing_saved_metrics": list(self.missing_saved_metrics),
            "missing_current_metrics": list(self.missing_current_metrics),
            "deltas": dict(self.deltas or {}),
            "message": self.message,
        }


class ProvenanceValidationError(RuntimeError):
    def __init__(self, result: ProvenanceValidationResult):
        self.result = result
        super().__init__(result.message or result.status)


def coerce_provenance(value: AutoRunProvenance | Mapping[str, Any] | None) -> AutoRunProvenance | None:
    if value is None:
        return None
    if isinstance(value, AutoRunProvenance):
        return value
    return AutoRunProvenance.from_mapping(value)


def build_auto_run_provenance(
    items: Iterable[ProvenanceItem],
    *,
    schema_version: int = PROVENANCE_SCHEMA_VERSION,
) -> AutoRunProvenance:
    item_tuple = tuple(sorted(items, key=_item_sort_key))
    selection_payload = {
        "schema_version": schema_version,
        "items": [item.to_dict() for item in item_tuple if item.scope in SELECTION_FINGERPRINT_SCOPES],
    }
    diagnostics_payload = {
        "schema_version": schema_version,
        "items": [item.to_dict() for item in item_tuple],
    }
    return AutoRunProvenance(
        schema_version=schema_version,
        selection_fingerprint=stable_signature(selection_payload),
        diagnostics_fingerprint=stable_signature(diagnostics_payload),
        items=item_tuple,
    )


def build_json_item(
    name: str,
    value: Any,
    *,
    scope: str = "selection",
    kind: str = "stable_json",
    notes: str = "",
) -> ProvenanceItem:
    return ProvenanceItem(
        name=name,
        kind=kind,
        fingerprint=stable_signature(_json_safe(value)),
        scope=scope,
        notes=notes,
    )


def build_file_item(
    name: str,
    path: Path | str,
    *,
    root: Path | str | None = None,
    scope: str = "selection",
    kind: str = "file_contents",
    notes: str = "",
) -> ProvenanceItem:
    path_obj = Path(path)
    root_obj = Path(root) if root is not None else None
    return ProvenanceItem(
        name=name,
        kind=kind,
        fingerprint=fingerprint_file_contents(path_obj, root=root_obj),
        paths=(_display_path(path_obj, root=root_obj),),
        scope=scope,
        notes=notes,
    )


def build_tree_item(
    name: str,
    root: Path | str,
    *,
    patterns: Iterable[str] = ("*.py",),
    recursive: bool = True,
    scope: str = "selection",
    kind: str = "tree_contents",
    display_root: Path | str | None = None,
    notes: str = "",
) -> ProvenanceItem:
    root_obj = Path(root)
    display_root_obj = Path(display_root) if display_root is not None else None
    return ProvenanceItem(
        name=name,
        kind=kind,
        fingerprint=fingerprint_tree_contents(
            root_obj,
            patterns=patterns,
            recursive=recursive,
            display_root=display_root_obj,
        ),
        paths=(_display_path(root_obj, root=display_root_obj),),
        scope=scope,
        notes=notes,
    )


def build_tree_metadata_item(
    name: str,
    root: Path | str,
    *,
    patterns: Iterable[str] = ("*.parquet",),
    recursive: bool = True,
    scope: str = "data",
    kind: str = "tree_metadata",
    notes: str = "",
) -> ProvenanceItem:
    root_obj = Path(root)
    return ProvenanceItem(
        name=name,
        kind=kind,
        fingerprint=fingerprint_tree(root_obj, patterns=patterns, recursive=recursive),
        paths=(_display_path(root_obj),),
        scope=scope,
        notes=notes,
    )


def fingerprint_file_contents(path: Path | str, *, root: Path | str | None = None) -> str:
    path_obj = Path(path)
    root_obj = Path(root) if root is not None else None
    return stable_signature(
        {
            "algorithm": "sha256",
            "entries": [_content_entry(path_obj, root=root_obj)],
        }
    )


def fingerprint_tree_contents(
    root: Path | str,
    *,
    patterns: Iterable[str] = ("*.py",),
    recursive: bool = True,
    display_root: Path | str | None = None,
) -> str:
    root_obj = Path(root)
    display_root_obj = Path(display_root) if display_root is not None else None
    if not root_obj.exists():
        return stable_signature(
            {"root": _display_path(root_obj, root=display_root_obj), "exists": False, "entries": []}
        )

    paths: set[Path] = set()
    for pattern in tuple(patterns):
        iterator = root_obj.rglob(pattern) if recursive else root_obj.glob(pattern)
        paths.update(path for path in iterator if path.is_file())

    entries = [_content_entry(path, root=root_obj) for path in sorted(paths, key=lambda item: str(item).lower())]
    return stable_signature(
        {
            "root": _display_path(root_obj, root=display_root_obj),
            "exists": True,
            "algorithm": "sha256",
            "entries": entries,
        }
    )


def build_fallback_provenance(
    *,
    plugin_name: str,
    execution_context: Mapping[str, Any],
    shared_auto_dir: Path,
) -> AutoRunProvenance:
    items: list[ProvenanceItem] = [
        build_json_item(
            "plugin",
            {"name": plugin_name},
            scope="selection",
            kind="plugin_identity",
            notes="Fallback provenance because the plugin does not implement build_provenance().",
        ),
        build_json_item("execution_context", dict(execution_context), scope="environment"),
        build_tree_item("shared_auto_package", shared_auto_dir, patterns=("*.py",), scope="selection"),
    ]

    data_dir = execution_context.get("data_dir")
    if data_dir:
        items.append(
            build_tree_metadata_item(
                "data_dir",
                Path(str(data_dir)),
                patterns=("*.csv", "*.json", "*.parquet"),
                scope="data",
                notes="Fallback metadata fingerprint; plugins should provide content-based data provenance.",
            )
        )
    return build_auto_run_provenance(items)


def build_phase_auto_provenance(
    plugin_name: str,
    *,
    repo_root: Path | str,
    code_paths: Iterable[Path | str] = (),
    code_dirs: Iterable[Path | str] = (),
    data_dir: Path | str | None = None,
    data_patterns: Iterable[str] = ("*.parquet", "*.csv", "*.json"),
    source_artifacts: Mapping[str, Path | str] | None = None,
    selection_context: Mapping[str, Any] | None = None,
    diagnostics_paths: Mapping[str, Path | str] | None = None,
    extra_items: Iterable[ProvenanceItem] = (),
) -> AutoRunProvenance:
    """Build run-boundary provenance for phased-auto plugins and promotion scripts."""
    root = Path(repo_root)
    items: list[ProvenanceItem] = [
        build_json_item("plugin", {"name": plugin_name}, kind="plugin_identity"),
        build_tree_item("shared_auto_package", Path(__file__).resolve().parent, display_root=root),
    ]
    if selection_context:
        items.append(build_json_item("selection_context", dict(selection_context)))

    for path in code_paths:
        path_obj = Path(path)
        items.append(build_file_item(_provenance_path_name("code", path_obj, root), path_obj, root=root))
    for path in code_dirs:
        path_obj = Path(path)
        items.append(build_tree_item(_provenance_path_name("code_tree", path_obj, root), path_obj, display_root=root))

    if data_dir is not None:
        items.append(
            build_tree_item(
                "data_dir",
                Path(data_dir),
                patterns=data_patterns,
                scope="data",
                display_root=root,
                notes="Content fingerprint for selection-relevant replay data files.",
            )
        )

    for name, path in sorted((source_artifacts or {}).items()):
        items.append(
            build_file_item(
                f"source_artifact:{name}",
                Path(path),
                root=root,
                scope="source_artifact",
                kind="source_artifact_contents",
            )
        )

    for name, path in sorted((diagnostics_paths or {}).items()):
        items.append(build_file_item(f"diagnostics:{name}", Path(path), root=root, scope="diagnostics"))

    items.extend(extra_items)
    return build_auto_run_provenance(items)


def diff_provenance_items(
    previous: AutoRunProvenance,
    current: AutoRunProvenance,
    *,
    include_diagnostics: bool,
) -> tuple[str, ...]:
    previous_items = _item_map(previous.items, include_diagnostics=include_diagnostics)
    current_items = _item_map(current.items, include_diagnostics=include_diagnostics)
    changed: list[str] = []

    for key in sorted(previous_items.keys() | current_items.keys()):
        old = previous_items.get(key)
        new = current_items.get(key)
        if old is None:
            changed.append(f"added {key}")
        elif new is None:
            changed.append(f"removed {key}")
        elif old.fingerprint != new.fingerprint or old.paths != new.paths:
            changed.append(f"changed {key}")
    return tuple(changed)


def classify_metric_drift(
    saved_metrics: Mapping[str, Any] | None,
    current_metrics: Mapping[str, Any] | None,
    *,
    critical_metrics: Iterable[str] = DEFAULT_CRITICAL_METRICS,
    float_abs_tolerance: float = 1e-9,
    selection_relevant: bool = True,
) -> MetricDriftResult:
    saved = dict(saved_metrics or {})
    current = dict(current_metrics or {})
    keys = tuple(critical_metrics)

    missing_saved = tuple(key for key in keys if saved.get(key) is None)
    missing_current = tuple(key for key in keys if current.get(key) is None)
    if missing_saved:
        return MetricDriftResult(
            status=DRIFT_STATUS_INCOMPLETE_SAVED_METRICS,
            missing_saved_metrics=missing_saved,
            missing_current_metrics=missing_current,
            message=f"Saved metrics are missing required fields: {', '.join(missing_saved)}.",
        )

    deltas: dict[str, dict[str, Any]] = {}
    for key in keys:
        if key in missing_current:
            continue
        saved_value = saved.get(key)
        current_value = current.get(key)
        if not _metric_values_match(key, saved_value, current_value, float_abs_tolerance=float_abs_tolerance):
            deltas[key] = {"saved": saved_value, "current": current_value}

    if missing_current:
        return MetricDriftResult(
            status=DRIFT_STATUS_SELECTION_STALE if selection_relevant else DRIFT_STATUS_DIAGNOSTICS_STALE,
            missing_current_metrics=missing_current,
            deltas=deltas,
            message=f"Current recompute metrics are missing required fields: {', '.join(missing_current)}.",
        )
    if deltas:
        return MetricDriftResult(
            status=DRIFT_STATUS_SELECTION_STALE if selection_relevant else DRIFT_STATUS_DIAGNOSTICS_STALE,
            deltas=deltas,
            message=f"Metric drift detected for {', '.join(sorted(deltas))}.",
        )
    return MetricDriftResult(status=DRIFT_STATUS_CURRENT, deltas={})


def _sha256_file(path: Path, *, stat: Any | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    file_stat = stat if stat is not None else path.stat()
    cache_key = (str(path.resolve()), int(file_stat.st_size), int(file_stat.st_mtime_ns))
    cached = _FILE_HASH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = value
    return value


def _content_entry(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    is_file = path.is_file() if exists else False
    return {
        "path": _display_path(path, root=root),
        "exists": exists,
        "is_file": is_file,
        "size": stat.st_size if stat is not None else None,
        "sha256": _sha256_file(path, stat=stat) if is_file else None,
    }


def _display_path(path: Path, *, root: Path | None = None) -> str:
    try:
        rendered = path.relative_to(root) if root is not None else path
    except ValueError:
        rendered = path
    return str(rendered).replace("\\", "/")


def _provenance_path_name(prefix: str, path: Path, root: Path) -> str:
    return f"{prefix}:{_display_path(path, root=root)}"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value


def _metric_values_match(key: str, saved: Any, current: Any, *, float_abs_tolerance: float) -> bool:
    if key == "total_trades":
        try:
            saved_number = float(saved)
            current_number = float(current)
        except (TypeError, ValueError):
            return saved == current
        return saved_number.is_integer() and current_number.is_integer() and int(saved_number) == int(current_number)
    try:
        return abs(float(saved) - float(current)) <= float_abs_tolerance
    except (TypeError, ValueError):
        return saved == current


def _item_sort_key(item: ProvenanceItem) -> tuple[str, str, str, tuple[str, ...]]:
    return (item.scope, item.kind, item.name, item.paths)


def _item_key(item: ProvenanceItem) -> str:
    return f"{item.scope}:{item.kind}:{item.name}"


def _item_map(items: Iterable[ProvenanceItem], *, include_diagnostics: bool) -> dict[str, ProvenanceItem]:
    return {
        _item_key(item): item
        for item in items
        if include_diagnostics or item.scope != "diagnostics"
    }
