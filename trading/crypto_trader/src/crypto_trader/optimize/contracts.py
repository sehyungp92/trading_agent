"""Optimization contract hashing and preflight checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from crypto_trader.backtest.config import BacktestConfig
from crypto_trader.backtest.profiles import (
    LIVE_PARITY_PROFILE,
    BacktestEconomicProfile,
    assert_backtest_config_matches_profile,
    profile_hash,
)
from crypto_trader.data.store import ParquetStore
from crypto_trader.optimize.types import GateCriterion, PhaseSpec


CONTRACT_SCHEMA_VERSION = "optimizer_contract_v1"


def stable_plain(value: Any) -> Any:
    """Convert common project objects into stable JSON-compatible values."""
    if is_dataclass(value):
        return stable_plain(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): stable_plain(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return [stable_plain(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [stable_plain(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(stable_plain(value), sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def strategy_config_hash(config: Any) -> str:
    payload = config.to_dict() if hasattr(config, "to_dict") else stable_plain(config)
    return stable_hash(payload)


def portfolio_config_hash(config: Any | None) -> str:
    if config is None:
        return ""
    payload = config.to_dict() if hasattr(config, "to_dict") else stable_plain(config)
    return stable_hash(payload)


def _file_fingerprint(base_dir: Path, path: Path) -> dict[str, Any]:
    rel_path = str(path.relative_to(base_dir)) if path.is_relative_to(base_dir) else str(path)
    if not path.exists():
        return {"path": rel_path, "exists": False}
    stat = path.stat()
    return {
        "path": rel_path,
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def data_snapshot_fingerprint(
    data_dir: Path | str,
    *,
    symbols: list[str],
    timeframes: list[str],
    include_funding: bool,
) -> dict[str, Any]:
    """Build a cheap fingerprint for the data files used by an optimization."""
    base = Path(data_dir).resolve()
    files: list[dict[str, Any]] = []
    for symbol in symbols:
        coin = str(symbol).upper()
        for timeframe in timeframes:
            files.append(_file_fingerprint(base, base / "candles" / coin / f"{timeframe}.parquet"))
        if include_funding:
            files.append(_file_fingerprint(base, base / "funding" / f"{coin}.parquet"))
    return {
        "method": "file_stat_v1",
        "files": files,
        "hash": stable_hash(files),
    }


def required_timeframes(strategy_type: str) -> list[str]:
    if strategy_type == "trend":
        return ["15m", "1h", "1d"]
    if strategy_type == "breakout":
        return ["30m", "4h"]
    return ["15m", "1h", "4h"]


def summarize_phase_spec(
    spec: PhaseSpec,
    *,
    include_candidates: bool = True,
) -> dict[str, Any]:
    """Summarize the contract-relevant parts of a phase spec."""
    payload: dict[str, Any] = {
        "phase_num": spec.phase_num,
        "name": spec.name,
        "scoring_weights": spec.scoring_weights,
        "hard_rejects": spec.hard_rejects,
        "gate_criteria": [criterion_to_dict(criterion) for criterion in spec.gate_criteria],
        "has_gate_criteria_fn": spec.gate_criteria_fn is not None,
        "min_delta": spec.min_delta,
        "max_rounds": spec.max_rounds,
        "prune_threshold": spec.prune_threshold,
        "focus": spec.focus,
    }
    if include_candidates:
        payload["candidate_count"] = len(spec.candidates)
        payload["candidates"] = [
            {"name": candidate.name, "mutations": candidate.mutations}
            for candidate in spec.candidates
        ]
    return payload


def criterion_to_dict(criterion: GateCriterion) -> dict[str, Any]:
    return {
        "metric": criterion.metric,
        "operator": criterion.operator,
        "threshold": criterion.threshold,
        "weight": criterion.weight,
    }


def collect_phase_contract(plugin: Any) -> dict[str, Any]:
    """Collect static phase contract data from a plugin."""
    from crypto_trader.optimize.phase_state import PhaseState

    state = PhaseState()
    initial = getattr(plugin, "initial_mutations", {})
    if initial:
        state.cumulative_mutations.update(initial)

    phases = []
    for phase in range(1, int(plugin.num_phases) + 1):
        phases.append(
            summarize_phase_spec(
                plugin.get_phase_spec(phase, state),
                include_candidates=False,
            )
        )
    return {"num_phases": int(plugin.num_phases), "phases": phases}


def _module_source_hash(module: Any) -> str:
    path_text = getattr(module, "__file__", None)
    if not path_text:
        return ""
    path = Path(path_text)
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    if not path.exists() or path.suffix != ".py":
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plugin_code_identity(plugin: Any | None) -> dict[str, str]:
    if plugin is None:
        return {}
    try:
        module = importlib.import_module(plugin.__class__.__module__)
    except (ImportError, AttributeError):
        return {}
    return {
        "module": plugin.__class__.__module__,
        "class": plugin.__class__.__qualname__,
        "source_hash": _module_source_hash(module),
    }


def _plugin_scoring_ceilings(plugin: Any | None) -> dict[str, Any]:
    if plugin is None:
        return {}
    try:
        module = importlib.import_module(plugin.__class__.__module__)
    except (ImportError, AttributeError):
        return {}
    preferred_names = (
        "SCORING_CEILINGS",
        "IMMUTABLE_SCORING_CEILINGS",
    )
    round_names = sorted(
        name for name in vars(module)
        if re.fullmatch(r"ROUND\d+_(?:IMMUTABLE_)?SCORING_CEILINGS", name)
    )
    for name in (*preferred_names, *round_names):
        if hasattr(module, name):
            return dict(getattr(module, name))
    return {}


def build_optimization_contract(
    *,
    strategy_type: str,
    strategy_config: Any,
    backtest_config: BacktestConfig,
    data_dir: Path | str,
    profile: BacktestEconomicProfile = LIVE_PARITY_PROFILE,
    plugin: Any | None = None,
    portfolio_config: Any | None = None,
    scoring_weights: dict[str, float] | None = None,
    hard_rejects: dict[str, Any] | None = None,
    gate_criteria: dict[int, list[GateCriterion]] | None = None,
    scoring_ceilings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a stable optimization contract payload and hash."""
    symbols = [str(symbol).upper() for symbol in (backtest_config.symbols or getattr(strategy_config, "symbols", []))]
    timeframes = required_timeframes(strategy_type)
    phase_contract = collect_phase_contract(plugin) if plugin is not None else {}
    scoring_contract = {
        "scoring_weights": scoring_weights or {},
        "hard_rejects": hard_rejects or {},
        "gate_criteria": {
            str(phase): [criterion_to_dict(criterion) for criterion in criteria]
            for phase, criteria in (gate_criteria or {}).items()
        },
        "scoring_ceilings": scoring_ceilings or _plugin_scoring_ceilings(plugin),
    }
    payload = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "strategy_type": strategy_type,
        "economic_profile": profile.to_dict(),
        "code_identity": {
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "plugin": _plugin_code_identity(plugin),
        },
        "profile_hash": profile_hash(profile),
        "data_dir": str(Path(data_dir).resolve()),
        "data_window": {
            "start_date": backtest_config.start_date.isoformat() if backtest_config.start_date else None,
            "end_date": backtest_config.end_date.isoformat() if backtest_config.end_date else None,
        },
        "symbols": symbols,
        "required_timeframes": timeframes,
        "data_fingerprint": data_snapshot_fingerprint(
            data_dir,
            symbols=symbols,
            timeframes=timeframes,
            include_funding=backtest_config.apply_funding,
        ),
        "backtest_config": stable_plain(backtest_config),
        "strategy_config_hash": strategy_config_hash(strategy_config),
        "portfolio_config_hash": portfolio_config_hash(portfolio_config),
        "phase_contract": phase_contract,
        "scoring_contract": scoring_contract,
    }
    payload["contract_hash"] = stable_hash(payload)
    return payload


def phase_checkpoint_context(
    contract: dict[str, Any] | None,
    *,
    phase: int,
    spec: PhaseSpec,
    scoring_weights: dict[str, float],
    validation_mode: str,
) -> str:
    """Build the checkpoint context passed to the greedy optimizer."""
    payload = {
        "contract_hash": (contract or {}).get("contract_hash", ""),
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "phase": phase,
        "spec": summarize_phase_spec(spec),
        "active_scoring_weights": scoring_weights,
        "validation_mode": validation_mode,
        "contract": contract or {},
    }
    return stable_json(payload)


def check_stale_artifacts(
    output_dir: Path | str,
    *,
    contract_hash: str,
    strict: bool = True,
) -> None:
    """Fail when existing optimizer artifacts belong to another contract."""
    if not strict:
        return
    output = Path(output_dir)
    if not output.exists():
        return

    paths = [
        output / "run_spec.json",
        output / "phase_state.json",
        output / "optimized_config.json",
        *output.glob("phase_*_greedy_checkpoint.json"),
    ]
    stale: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stale.append(f"{path.name}: unreadable")
            continue
        found = str(
            payload.get("contract_hash")
            or payload.get("metadata", {}).get("contract_hash")
            or ""
        )
        if found != contract_hash:
            stale.append(f"{path.name}: {found or '<missing>'} != {contract_hash}")

    manifest_path = output.parent / "rounds_manifest.json"
    round_match = re.fullmatch(r"round_(\d+)", output.name)
    if manifest_path.exists() and round_match:
        round_num = int(round_match.group(1))
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            stale.append("rounds_manifest.json: unreadable")
        else:
            for entry in manifest.get("rounds", []):
                try:
                    entry_round = int(entry.get("round", -1))
                except (TypeError, ValueError):
                    continue
                if entry_round == round_num:
                    found = str(entry.get("contract_hash") or "")
                    if found != contract_hash:
                        stale.append(
                            f"rounds_manifest.json round {round_num}: "
                            f"{found or '<missing>'} != {contract_hash}"
                        )
                    break
    if stale:
        raise RuntimeError("Stale optimization artifacts refused: " + "; ".join(stale))


def run_optimization_preflight(
    *,
    contract: dict[str, Any],
    backtest_config: BacktestConfig,
    data_dir: Path | str,
    output_dir: Path | str | None = None,
    profile: BacktestEconomicProfile = LIVE_PARITY_PROFILE,
    validation_mode: str = "strict",
) -> None:
    """Run cheap checks before an optimization search starts."""
    strict = validation_mode == "strict"
    if strict:
        assert_backtest_config_matches_profile(backtest_config, profile=profile)

    if output_dir is not None:
        check_stale_artifacts(
            output_dir,
            contract_hash=str(contract.get("contract_hash") or ""),
            strict=strict,
        )

    store = ParquetStore(base_dir=Path(data_dir))
    missing: list[str] = []
    symbols = [str(symbol).upper() for symbol in contract.get("symbols", [])]
    for symbol in symbols:
        for timeframe in contract.get("required_timeframes", []):
            df = store.load_candles(symbol, timeframe)
            if df is None or df.empty:
                missing.append(f"candles/{symbol}/{timeframe}")
        if backtest_config.apply_funding:
            funding = store.load_funding(symbol)
            if funding is None or funding.empty:
                missing.append(f"funding/{symbol}")
    if missing:
        raise RuntimeError("Optimization preflight missing required data: " + ", ".join(missing))
