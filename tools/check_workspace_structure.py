from __future__ import annotations

import argparse
import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FINAL_ROOT_DIRECTORIES = ("packages", "docs", "artifacts")
FINAL_ROOT_FILES = ("README.md",)
FINAL_SHARED_PACKAGES = (
    "trading_backtest",
    "trading_config",
    "trading_contracts",
    "trading_deployment",
    "trading_instrumentation",
    "trading_optimizer",
)
FINAL_SHARED_BEHAVIOR_MODULES = {
    "trading_backtest": ("src/trading_backtest/invariants.py",),
    "trading_deployment": ("src/trading_deployment/metadata.py",),
    "trading_instrumentation": ("src/trading_instrumentation/approval_metadata.py",),
    "trading_optimizer": (
        "src/trading_optimizer/archived_smoke.py",
        "src/trading_optimizer/phase_runner_adapters.py",
    ),
}

LAYOUTS = {
    "current": {
        "trading_assistant": ROOT / "trading_assistant",
        "trading_assistant_data": ROOT / "trading_assistant_data",
        "trading_assistant_backtest": ROOT / "trading_assistant_backtest",
    },
    "final": {
        "trading_assistant": ROOT / "packages" / "trading_assistant",
        "trading_assistant_data": ROOT / "packages" / "trading_assistant_data",
        "trading_assistant_backtest": ROOT / "packages" / "trading_assistant_backtest",
    },
}

RUNTIME_DIRS = {
    "current": {
        "trading_assistant": [
            "analysis",
            "comms",
            "contracts",
            "orchestrator",
            "schemas",
            "skills",
        ],
        "trading_assistant_data": ["src/trading_assistant_data"],
        "trading_assistant_backtest": ["src/trading_assistant_backtest", "backtests"],
    },
    "final": {
        "trading_assistant": ["src/trading_assistant"],
        "trading_assistant_data": ["src/trading_assistant_data"],
        "trading_assistant_backtest": ["src/trading_assistant_backtest", "backtests"],
    },
}

LEGACY_CONTROL_IMPORT_ROOTS = {
    "analysis",
    "comms",
    "contracts",
    "orchestrator",
    "schemas",
    "skills",
}
FINAL_CONTROL_NAMESPACE_SCAN_DIRS = ("src/trading_assistant", "tests")

FORBIDDEN_IMPORTS = {
    "current": {
        "trading_assistant": {"trading_assistant_data", "trading_assistant_backtest"},
        "trading_assistant_data": {"trading_assistant", "trading_assistant_backtest"},
        "trading_assistant_backtest": {"trading_assistant", "trading_assistant_data"},
    },
    "final": {
        "trading_assistant": {
            "trading_assistant_data",
            "trading_assistant_backtest",
        },
        "trading_assistant_data": {"trading_assistant", "trading_assistant_backtest"},
        "trading_assistant_backtest": {"trading_assistant", "trading_assistant_data"},
    },
}

MODULE_SIZE_LIMITS = {
    "packages/trading_assistant/src/trading_assistant/orchestrator/app.py": 800,
    "packages/trading_assistant/src/trading_assistant/orchestrator/handlers.py": 120,
    "packages/trading_assistant/src/trading_assistant/orchestrator/_handler_implementation.py": 250,
    "packages/trading_assistant/src/trading_assistant/analysis/context_builder.py": 900,
    "packages/trading_assistant/src/trading_assistant/analysis/strategy_engine.py": 1300,
    "packages/trading_assistant_data/src/trading_assistant_data/normalization.py": 2300,
    "packages/trading_assistant_backtest/src/trading_assistant_backtest/monthly.py": 120,
    "packages/trading_assistant_backtest/src/trading_assistant_backtest/monthly_execution/runner.py": 180,
}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_python_source(path: Path, errors: list[str]) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        errors.append(f"could not read {_rel(path)}: {exc}")
        return None
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append(f"{_rel(path)} starts with a UTF-8 BOM; remove the BOM")
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"could not decode {_rel(path)} as UTF-8: {exc}")
        return None


def _check_exists(workspaces: dict[str, Path], errors: list[str]) -> None:
    for name, path in workspaces.items():
        if not path.is_dir():
            errors.append(f"missing workspace directory: {name}")
        if not (path / "pyproject.toml").is_file():
            errors.append(f"missing pyproject.toml: {name}")

    data_src = workspaces["trading_assistant_data"] / "src" / "trading_assistant_data"
    backtest_src = (
        workspaces["trading_assistant_backtest"] / "src" / "trading_assistant_backtest"
    )
    if not (data_src / "__init__.py").is_file():
        errors.append("missing data src package")
    if not (backtest_src / "__init__.py").is_file():
        errors.append("missing backtest src package")


def _check_final_root_layout(errors: list[str]) -> None:
    for relative in FINAL_ROOT_DIRECTORIES:
        if not (ROOT / relative).is_dir():
            errors.append(f"missing final root directory: {relative}")
    for relative in FINAL_ROOT_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing final root file: {relative}")
    for package in FINAL_SHARED_PACKAGES:
        package_root = ROOT / "packages" / package
        if not package_root.is_dir():
            errors.append(f"missing final shared package: packages/{package}")
            continue
        if not (package_root / "pyproject.toml").is_file():
            errors.append(f"missing pyproject.toml: packages/{package}")
        if not (package_root / "src" / package / "__init__.py").is_file():
            errors.append(f"missing src package: packages/{package}/src/{package}")
        for module in FINAL_SHARED_BEHAVIOR_MODULES.get(package, ()):
            if not (package_root / module).is_file():
                errors.append(f"missing shared behavior module: packages/{package}/{module}")


def _check_no_legacy_workspace_roots(errors: list[str]) -> None:
    for name, path in LAYOUTS["current"].items():
        if path.exists():
            errors.append(
                "obsolete top-level workspace exists in final layout: "
                f"{name} ({_rel(path)})"
            )


def _check_packaging(layout: str, workspaces: dict[str, Path], errors: list[str]) -> None:
    if layout == "final":
        control_pyproject = workspaces["trading_assistant"] / "pyproject.toml"
        if not control_pyproject.is_file():
            return
        control = _load_toml(workspaces["trading_assistant"] / "pyproject.toml")
        control_where = (
            control.get("tool", {})
            .get("setuptools", {})
            .get("packages", {})
            .get("find", {})
            .get("where")
        )
        if control_where != ["src"]:
            errors.append("trading_assistant should discover packages from src in final layout")
        control_include = (
            control.get("tool", {})
            .get("setuptools", {})
            .get("packages", {})
            .get("find", {})
            .get("include", [])
        )
        if "trading_assistant*" not in control_include:
            errors.append("trading_assistant package discovery should include trading_assistant*")

        control_src = workspaces["trading_assistant"] / "src" / "trading_assistant"
        if not (control_src / "__init__.py").is_file():
            errors.append("missing control-plane src package")

    data_pyproject = workspaces["trading_assistant_data"] / "pyproject.toml"
    backtest_pyproject = workspaces["trading_assistant_backtest"] / "pyproject.toml"
    if not data_pyproject.is_file() or not backtest_pyproject.is_file():
        return

    data = _load_toml(workspaces["trading_assistant_data"] / "pyproject.toml")
    data_build_backend = data.get("build-system", {}).get("build-backend")
    if data_build_backend != "setuptools.build_meta":
        errors.append("trading_assistant_data should use setuptools.build_meta")
    data_where = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("where")
    )
    if data_where != ["src"]:
        errors.append("trading_assistant_data should discover packages from src")

    backtest = _load_toml(workspaces["trading_assistant_backtest"] / "pyproject.toml")
    backtest_build_backend = backtest.get("build-system", {}).get("build-backend")
    if backtest_build_backend != "hatchling.build":
        errors.append("trading_assistant_backtest should use hatchling.build")
    wheel_packages = (
        backtest.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    required = {"src/trading_assistant_backtest", "backtests"}
    missing = sorted(required - set(wheel_packages))
    if missing:
        errors.append(
            "trading_assistant_backtest wheel package list is missing: "
            + ", ".join(missing)
        )


def _iter_python_files(
    workspace: str,
    *,
    workspaces: dict[str, Path],
    runtime_dirs: dict[str, list[str]],
) -> list[Path]:
    root = workspaces[workspace]
    files: list[Path] = []
    for relative in runtime_dirs[workspace]:
        directory = root / relative
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    return sorted(files)


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.partition(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.partition(".")[0])
    return roots


def _check_import_boundaries(
    *,
    workspaces: dict[str, Path],
    runtime_dirs: dict[str, list[str]],
    forbidden_imports: dict[str, set[str]],
    errors: list[str],
    violation_label: str = "forbidden workspace package(s)",
) -> None:
    for workspace, forbidden in forbidden_imports.items():
        for path in _iter_python_files(
            workspace,
            workspaces=workspaces,
            runtime_dirs=runtime_dirs,
        ):
            source = _read_python_source(path, errors)
            if source is None:
                continue
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                errors.append(f"could not parse {_rel(path)}: {exc}")
                continue

            illegal = sorted(_import_roots(tree) & forbidden)
            if illegal:
                errors.append(
                    f"{_rel(path)} imports {violation_label}: "
                    + ", ".join(illegal)
                )


def _check_final_control_namespace_imports(
    workspaces: dict[str, Path],
    errors: list[str],
) -> None:
    _check_import_boundaries(
        workspaces=workspaces,
        runtime_dirs={"trading_assistant": list(FINAL_CONTROL_NAMESPACE_SCAN_DIRS)},
        forbidden_imports={"trading_assistant": LEGACY_CONTROL_IMPORT_ROOTS},
        errors=errors,
        violation_label="legacy control-plane root(s)",
    )


def _check_module_size_watch(errors: list[str]) -> None:
    for relative, max_lines in MODULE_SIZE_LIMITS.items():
        path = ROOT / relative
        if not path.exists():
            continue
        source = _read_python_source(path, errors)
        if source is None:
            continue
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"could not parse {_rel(path)}: {exc}")
            continue
        line_count = len(source.splitlines())
        if line_count > max_lines:
            errors.append(
                f"{relative} has {line_count} lines; module-size watch limit is {max_lines}"
            )


def _check_layout(layout: str) -> list[str]:
    errors: list[str] = []
    workspaces = LAYOUTS[layout]
    if layout == "final":
        _check_final_root_layout(errors)
        _check_no_legacy_workspace_roots(errors)
    _check_exists(workspaces, errors)
    _check_packaging(layout, workspaces, errors)
    _check_import_boundaries(
        workspaces=workspaces,
        runtime_dirs=RUNTIME_DIRS[layout],
        forbidden_imports=FORBIDDEN_IMPORTS[layout],
        errors=errors,
    )
    if layout == "final":
        _check_final_control_namespace_imports(workspaces, errors)
        _check_module_size_watch(errors)
    return errors


def _check_transition_layout() -> tuple[str, list[str]]:
    """Validate an intentional in-flight migration layout.

    Data and backtest may already live under packages/ while the control plane
    may still be either top-level, packages/ with its old package roots, or
    fully migrated to packages/.../src/trading_assistant.
    """

    errors: list[str] = []
    workspaces: dict[str, Path] = {}
    runtime_dirs: dict[str, list[str]] = {}
    forbidden_imports: dict[str, set[str]] = {}
    states: dict[str, str] = {}

    for name in LAYOUTS["current"]:
        final_path = LAYOUTS["final"][name]
        current_path = LAYOUTS["current"][name]
        workspace = final_path if final_path.is_dir() else current_path
        workspaces[name] = workspace

        if name == "trading_assistant":
            if (workspace / "src" / "trading_assistant").is_dir():
                state = "final"
                runtime_dirs[name] = RUNTIME_DIRS["final"][name]
                forbidden_imports[name] = FORBIDDEN_IMPORTS["final"][name]
            else:
                state = "transitional" if workspace == final_path else "current"
                runtime_dirs[name] = RUNTIME_DIRS["current"][name]
                forbidden_imports[name] = FORBIDDEN_IMPORTS["current"][name]
        else:
            state = "final" if workspace == final_path else "current"
            runtime_dirs[name] = RUNTIME_DIRS[state][name]
            forbidden_imports[name] = FORBIDDEN_IMPORTS[state][name]
        states[name] = state

    _check_exists(workspaces, errors)

    if states["trading_assistant"] == "final":
        _check_packaging("final", workspaces, errors)
    else:
        _check_packaging("current", workspaces, errors)

    if states["trading_assistant"] == "transitional":
        missing = [
            relative
            for relative in RUNTIME_DIRS["current"]["trading_assistant"]
            if not (workspaces["trading_assistant"] / relative).is_dir()
        ]
        if missing:
            errors.append(
                "transitional control workspace is missing runtime dirs: "
                + ", ".join(missing)
            )

    _check_import_boundaries(
        workspaces=workspaces,
        runtime_dirs=runtime_dirs,
        forbidden_imports=forbidden_imports,
        errors=errors,
    )
    _check_module_size_watch(errors)
    label = ", ".join(f"{name}={state}" for name, state in sorted(states.items()))
    return label, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate trading assistant workspace layout.")
    parser.add_argument(
        "--layout",
        choices=("current", "final", "either"),
        default="final",
        help=(
            "Workspace layout to validate. The final packages/ layout is the "
            "supported checkout shape; current/either are historical migration checks."
        ),
    )
    args = parser.parse_args(argv)

    if args.layout == "either":
        current_errors = _check_layout("current")
        if not current_errors:
            print("workspace structure OK (current layout)")
            return 0
        final_errors = _check_layout("final")
        if not final_errors:
            print("workspace structure OK (final layout)")
            return 0
        transition_label, transition_errors = _check_transition_layout()
        if not transition_errors:
            print(f"workspace structure OK (transition layout: {transition_label})")
            return 0
        for error in current_errors:
            print(f"CURRENT ERROR: {error}")
        for error in final_errors:
            print(f"FINAL ERROR: {error}")
        for error in transition_errors:
            print(f"TRANSITION ERROR: {error}")
        return 1

    errors = _check_layout(args.layout)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"workspace structure OK ({args.layout} layout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
