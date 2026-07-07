from __future__ import annotations

import json
import ast
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "backtests" / "baselines"
BASELINE_INDEX = BASELINES / "baseline_index.json"
INVENTORY_DOC = ROOT / "docs" / "migration_inventory.md"
STRATEGY_CONTRACTS = ROOT / "contracts" / "strategy_plugins"
PROMOTION_DRAFTS = ROOT / "contracts" / "promotions" / "draft"
PORTED_SOURCE_ROOTS = {
    "crypto_trader": ROOT / "bots" / "crypto_trader",
    "ibkr_trading": ROOT / "bots" / "ibkr_trading",
    "k_stock_trader": ROOT / "bots" / "k_stock_trader",
    "trading_assistant": ROOT / "packages" / "trading_assistant",
    "trading_assistant_backtest": ROOT / "packages" / "trading_assistant_backtest",
    "trading_assistant_data": ROOT / "packages" / "trading_assistant_data",
}


KEY_METRIC_NAMES = (
    "total_trades",
    "win_rate",
    "win_rate_pct",
    "profit_factor",
    "net_return_pct",
    "max_drawdown_pct",
    "max_dd_pct",
    "sharpe_ratio",
    "sharpe",
    "calmar_ratio",
    "calmar",
    "expectancy_r",
    "avg_r",
)

TEXT_ARTIFACT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)


IBKR_PROMOTION_MAP = {
    "ATRSS": ("swing", "atrss"),
    "AKC_HELIX": ("swing", "helix"),
    "TPC": ("swing", "tpc"),
    "IARIC_v1": ("stock", "iaric"),
    "ALCB_v1": ("stock", "alcb"),
    "NQDTC_v2.1": ("momentum", "nqdtc"),
    "NQ_REGIME": ("momentum", "nq_regime"),
    "VdubusNQ_v4": ("momentum", "vdubus"),
    "DownturnDominator_v1": ("momentum", "downturn"),
}

K_STOCK_PROMOTION_MAP = {
    "kalcb": "kalcb",
    "olr": "olr",
    "olr_kalcb_portfolio": "portfolio_synergy",
}


@dataclass(frozen=True)
class FileRecord:
    role: str
    path: str
    sha256: str
    size_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    data = path.read_bytes()
    if _is_text_artifact(path, data):
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _is_text_artifact(path: Path, data: bytes) -> bool:
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def canonical_json_hash_path(path: Path) -> str:
    payload = read_json(path)
    return canonical_json_hash(payload)


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def copy_with_record(source: Path, destination: Path, role: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    record = {
        "role": role,
        "source_path": rel(destination),
        "baseline_path": rel(destination),
        "sha256": file_sha256(source),
        "size_bytes": source.stat().st_size,
    }
    if source.suffix.lower() == ".json":
        record["canonical_json_sha256"] = canonical_json_hash_path(source)
    return record


def selected_files() -> list[FileRecord]:
    patterns = {
        "pyproject": ("pyproject.toml",),
        "requirements": ("requirements.txt",),
        "pytest": ("pytest.ini",),
        "dockerfile": ("Dockerfile",),
        "compose": ("docker-compose.yml", "docker-compose.yaml"),
        "contract": ("strategy_plugin_contract.json",),
        "live_config": (
            "config/strategies.yaml",
            "config/live_config.example.json",
            "config/kalcb.yaml",
            "config/optimization/kalcb.yaml",
            "config/optimization/olr.yaml",
            "config/olr_kalcb/olr_deployment_universe_103.yaml",
        ),
    }
    records: list[FileRecord] = []
    for source_root in sorted(PORTED_SOURCE_ROOTS.values()):
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            normalized = path.relative_to(source_root).as_posix()
            name = path.name
            role = None
            for candidate_role, names in patterns.items():
                if name in names or normalized in names:
                    role = candidate_role
                    break
            if role is None and "/contracts/" in f"/{normalized}":
                role = "contract"
            if role is None and "/scripts/" in f"/{normalized}":
                role = "script"
            if role is None:
                continue
            records.append(
                FileRecord(
                    role=role,
                    path=rel(path),
                    sha256=file_sha256(path),
                    size_bytes=path.stat().st_size,
                )
            )
    return records


def source_roots_inventory() -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    for source_root in sorted(PORTED_SOURCE_ROOTS.values()):
        if not source_root.is_dir():
            continue
        files = [path for path in source_root.rglob("*") if path.is_file() and ".git" not in path.parts]
        git_dir = source_root / ".git"
        git_head = None
        if git_dir.exists():
            git_head = _git_capture(["git", "-C", str(source_root), "rev-parse", "HEAD"])
        roots.append(
            {
                "path": rel(source_root),
                "file_count": len(files),
                "nested_git": git_dir.exists(),
                "git_head": git_head,
            }
        )
    return roots


def _git_capture(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def artifact_roots_inventory() -> list[dict[str, Any]]:
    candidates = [
        ("ibkr_backtest_baselines", BASELINES / "ibkr"),
        ("ibkr_baseline_fixture", ROOT / "bots" / "ibkr_trading" / "tests" / "fixtures" / "backtest_baselines"),
        ("crypto_backtest_baselines", BASELINES / "crypto"),
        ("k_stock_backtest_baselines", BASELINES / "k_stock"),
        ("assistant_contracts", STRATEGY_CONTRACTS),
    ]
    roots: list[dict[str, Any]] = []
    for role, path in candidates:
        files = []
        if path.exists():
            files = [child for child in path.rglob("*") if child.is_file()]
        roots.append(
            {
                "role": role,
                "path": rel(path) if path.exists() else path.relative_to(ROOT).as_posix(),
                "exists": path.exists(),
                "file_count": len(files),
            }
        )
    return roots


def direct_ibkr_round_manifests() -> list[Path]:
    output = BASELINES / "ibkr"
    manifests = []
    for path in output.rglob("rounds_manifest.json"):
        relative = path.relative_to(output)
        if len(relative.parts) == 3:
            manifests.append(path)
    return sorted(manifests)


def crypto_strategy_round_manifests() -> list[Path]:
    output = BASELINES / "crypto"
    return sorted(
        path
        for path in output.glob("*/rounds_manifest.json")
        if path.parent.name in {"momentum", "trend", "breakout"}
    )


def k_stock_round_manifests() -> list[Path]:
    output = BASELINES / "k_stock"
    if not output.exists():
        return []
    return sorted(path for path in output.glob("*/rounds_manifest.json"))


def choose_latest_round(
    rounds: list[dict[str, Any]],
    explicit_latest_round: Any | None = None,
) -> dict[str, Any] | None:
    candidates = [r for r in rounds if not r.get("archived") and not r.get("invalidated")]
    if explicit_latest_round is not None:
        explicit = [
            r
            for r in candidates
            if _round_sort_value(r.get("round")) == _round_sort_value(explicit_latest_round)
        ]
        if explicit:
            candidates = explicit
    if not candidates:
        return None
    return max(candidates, key=lambda r: (_round_sort_value(r.get("round")), str(r.get("timestamp", ""))))


def _round_sort_value(raw: Any) -> tuple[int, str]:
    try:
        return (int(raw), "")
    except Exception:
        return (-1, str(raw))


def key_metrics(round_record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(round_record.get("metrics") or {})
    for name in KEY_METRIC_NAMES:
        if name in round_record and name not in metrics:
            metrics[name] = round_record[name]
    return {name: metrics[name] for name in KEY_METRIC_NAMES if name in metrics}


def parse_ibkr_enabled_strategies() -> dict[str, dict[str, Any]]:
    path = ROOT / "bots" / "ibkr_trading" / "config" / "strategies.yaml"
    if not path.exists():
        return {}
    strategies: dict[str, dict[str, Any]] = {}
    in_strategies = False
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("strategies:"):
            in_strategies = True
            continue
        if not in_strategies:
            continue
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            current = line.strip().rstrip(":").strip('"')
            strategies[current] = {"registry_key": current}
            continue
        if current and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip().strip('"')
            if key in {"strategy_id", "family", "module_path", "enabled"}:
                strategies[current][key] = parse_scalar(value)
    return strategies


def parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def parse_kalcb_alignment(source_root: Path | None = None) -> dict[str, Any]:
    source = source_root or ROOT / "bots" / "k_stock_trader"
    live_path = source / "config" / "kalcb.yaml"
    optimizer_path = source / "config" / "optimization" / "kalcb.yaml"
    universe_path = source / "config" / "olr_kalcb" / "olr_deployment_universe_103.yaml"
    default_path = source / "strategy_kalcb" / "config.py"
    phase_base_path = source / "backtests" / "strategies" / "kalcb" / "phase_candidates.py"
    live_size = _extract_nested_size(live_path, ("kalcb", "frontier"), "size")
    optimizer_size = _extract_initial_mutation_value(
        optimizer_path, "kalcb.frontier.size"
    ) or _extract_nested_size(optimizer_path, ("kalcb", "frontier"), "size")
    universe_size = _extract_simple_key(universe_path, "symbol_count")
    default_size = _extract_python_class_int(default_path, "KALCBConfig", "frontier_size")
    phase_base_size = _extract_python_dict_value(
        phase_base_path,
        "BASE_MUTATIONS",
        "kalcb.frontier.size",
    )
    sizes = {
        "live_frontier_size": live_size,
        "optimizer_frontier_size": optimizer_size,
        "strategy_default_frontier_size": default_size,
        "phase_optimizer_base_frontier_size": phase_base_size,
        "deployment_universe_size": universe_size,
    }
    status = "aligned" if set(sizes.values()) == {103} else "blocked_alignment_finding"
    return {
        "status": status,
        "live_config_path": rel(live_path),
        "live_config_sha256": file_sha256(live_path) if live_path.exists() else None,
        "optimizer_config_path": rel(optimizer_path),
        "optimizer_config_sha256": file_sha256(optimizer_path) if optimizer_path.exists() else None,
        "strategy_default_path": rel(default_path),
        "strategy_default_sha256": file_sha256(default_path) if default_path.exists() else None,
        "phase_optimizer_base_path": rel(phase_base_path),
        "phase_optimizer_base_sha256": file_sha256(phase_base_path) if phase_base_path.exists() else None,
        "deployment_universe_path": rel(universe_path),
        "deployment_universe_sha256": file_sha256(universe_path) if universe_path.exists() else None,
        **sizes,
        "decision": (
            "KALCB frontier size must be 103 in live config, optimizer base mutation, "
            "strategy default, phase optimizer base, and deployment universe before A8/promotion."
            if status != "aligned"
            else "No alignment finding."
        ),
    }


def _extract_simple_key(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith(prefix):
            return _maybe_int(line.split(":", 1)[1].strip())
    return None


def _extract_initial_mutation_value(path: Path, key: str) -> int | None:
    if not path.exists():
        return None
    prefix = f"{key}:"
    in_initial = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("initial_mutations:"):
            in_initial = True
            continue
        if in_initial and line and not line.startswith(" "):
            in_initial = False
        if in_initial and line.strip().startswith(prefix):
            return _maybe_int(line.split(":", 1)[1].strip())
    return None


def _extract_nested_size(path: Path, sections: tuple[str, ...], key: str) -> int | None:
    if not path.exists():
        return None
    stack: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if stripped.endswith(":"):
            stack.append((indent, stripped[:-1]))
            continue
        if ":" in stripped and stripped.split(":", 1)[0] == key:
            active = tuple(item[1] for item in stack)
            if active[-len(sections) :] == sections:
                return _maybe_int(stripped.split(":", 1)[1].strip())
    return None


def _maybe_int(value: str) -> int | None:
    try:
        return int(value.strip().strip('"'))
    except Exception:
        return None


def _extract_python_class_int(path: Path, class_name: str, attr_name: str) -> int | None:
    if not path.exists():
        return None
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            target = None
            value = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target, value = item.target.id, item.value
            elif isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
                target, value = item.targets[0].id, item.value
            if target == attr_name and value is not None:
                try:
                    return int(ast.literal_eval(value))
                except Exception:
                    return None
    return None


def _extract_python_dict_value(path: Path, dict_name: str, key: str) -> int | None:
    if not path.exists():
        return None
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == dict_name
            for target in node.targets
        ):
            try:
                payload = ast.literal_eval(node.value)
            except Exception:
                return None
            if isinstance(payload, dict) and key in payload:
                try:
                    return int(payload[key])
                except Exception:
                    return None
    return None


def source_path_from_record(record: dict[str, Any]) -> Path:
    return ROOT / record["source_path"]


def baseline_path_from_record(record: dict[str, Any]) -> Path:
    return ROOT / record["baseline_path"]


def load_baseline_index() -> dict[str, Any]:
    if not BASELINE_INDEX.exists():
        raise FileNotFoundError(f"{BASELINE_INDEX} does not exist; run tools/freeze_optimization_baselines.py first")
    return read_json(BASELINE_INDEX)


def iter_baseline_records(index: dict[str, Any], bot: str | None = None) -> Iterable[dict[str, Any]]:
    for record in index.get("baselines", []):
        if bot in {None, "all"} or record.get("bot") == bot:
            yield record


def print_result(ok: bool, label: str, detail: str) -> None:
    prefix = "PASS" if ok else "FAIL"
    print(f"{prefix} {label} - {detail}")


def ensure_no_path_escape(path: Path) -> None:
    resolved = path.resolve()
    if not str(resolved).startswith(str(ROOT.resolve())):
        raise ValueError(f"Refusing to write outside workspace: {resolved}")
