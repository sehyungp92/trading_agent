"""Path containment helpers."""

from __future__ import annotations

from pathlib import Path

WORKSPACE_NAMES = frozenset(
    {
        "trading_assistant",
        "trading_assistant_data",
        "trading_assistant_backtest",
    }
)


def package_root() -> Path:
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == "src":
        return package_dir.parent.parent
    return package_dir


def monorepo_root() -> Path:
    root = package_root()
    return root.parent.parent if root.parent.name == "packages" else root.parent


def workspace_root(agent_root: Path, name: str) -> Path:
    root = Path(agent_root).resolve()
    final_path = root / "packages" / name
    if final_path.exists():
        return final_path
    return root / name


def resolve_workspace_path(agent_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0] in WORKSPACE_NAMES:
        return workspace_root(agent_root, parts[0]).joinpath(*parts[1:])
    return Path(agent_root).resolve() / candidate


def normalize_workspace_path(agent_root: Path, path: str | Path) -> Path:
    """Map historical workspace-root paths to the active checkout layout."""

    text = str(path or "").strip()
    if not text:
        return Path()

    root = Path(agent_root).resolve()
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(root)
        except (OSError, ValueError):
            return candidate
        return resolve_workspace_path(root, relative)

    workspace_relative = workspace_relative_path(text)
    if workspace_relative is not None:
        return resolve_workspace_path(root, workspace_relative)
    return resolve_workspace_path(root, candidate)


def workspace_relative_path(text: str) -> Path | None:
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts):
        if part in WORKSPACE_NAMES:
            return Path(*parts[index:])
    return None


def resolve_under(root: Path, path: str | Path, *, label: str = "path") -> Path:
    """Resolve `path` and require it to remain under `root`."""
    root = Path(root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes artifact root: {resolved}") from exc
    return resolved


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
