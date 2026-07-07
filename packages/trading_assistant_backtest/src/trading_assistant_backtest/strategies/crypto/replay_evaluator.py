"""Replay-backed monthly evaluator for the crypto shadow portfolio."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from trading_assistant_backtest.auto.types import Candidate, CandidateEvaluation
from trading_assistant_backtest.contract_models import DataBundleManifest, MonthlyRunManifest
from trading_assistant_backtest.replay.types import ReplayResult, WindowSpec
from trading_assistant_backtest.scoring.immutable import (
    compact_score_payload,
    resolve_score_profile,
    score_replay,
)
from trading_assistant_backtest.strategies.approval import adoption_enabled_for_manifest
from trading_assistant_backtest.strategies.plugin_semantics import (
    build_confirmatory_variants_for_scope,
    build_phase_specs_for_scope,
    build_repair_candidates_for_scope,
    effective_parameter_patch,
    evaluated_patch_payload,
    patch_int,
    patch_number,
    round_n_plus_1_payload,
)

REPLAY_ENGINE_VERSION = "crypto_bar_replay_v1"


@dataclass
class CryptoReplayBaseline:
    manifest: MonthlyRunManifest
    data_bundle: DataBundleManifest
    bundle_manifest_path: Path
    bundle_id: str
    bundle_checksum: str
    strategy_plugin_id: str
    threshold_bps: float = 0.0
    position_weight: float = 1.0
    max_symbols: int = 3
    adoption_enabled: bool = False


@dataclass
class CryptoReplayPlugin:
    plugin_id: str
    strategy_id: str
    family: str = "crypto_portfolio"
    supported_symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    supported_timeframes: list[str] = field(
        default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    )
    _baseline: CryptoReplayBaseline | None = None
    _incumbent: ReplayResult | None = None

    def load_baseline(
        self, manifest: MonthlyRunManifest, data_bundle: DataBundleManifest
    ) -> CryptoReplayBaseline:
        baseline = CryptoReplayBaseline(
            manifest=manifest,
            data_bundle=data_bundle,
            bundle_manifest_path=Path(
                manifest.data_bundle_manifest_path or manifest.market_data_manifest_path
            ),
            bundle_id=data_bundle.bundle_id,
            bundle_checksum=data_bundle.bundle_checksum,
            strategy_plugin_id=manifest.strategy_plugin_id or self.plugin_id,
            adoption_enabled=adoption_enabled_for_manifest(manifest),
        )
        self._baseline = baseline
        return baseline

    def run_incumbent(self, window: WindowSpec, baseline: Any) -> ReplayResult:
        replay_baseline = _as_baseline(baseline)
        result = _run_bar_replay(
            manifest=replay_baseline.manifest,
            data_bundle=replay_baseline.data_bundle,
            bundle_manifest_path=replay_baseline.bundle_manifest_path,
            window=window,
            threshold_bps=replay_baseline.threshold_bps,
            position_weight=replay_baseline.position_weight,
            max_symbols=replay_baseline.max_symbols,
            candidate_id="incumbent",
        )
        self._incumbent = result
        return result

    def run_diagnostics(self, replay: ReplayResult) -> dict[str, Any]:
        return {
            "schema_version": "crypto_replay_diagnostics_v1",
            "replay_engine_version": REPLAY_ENGINE_VERSION,
            "trade_count": replay.trade_count,
            "net_return": replay.net_return,
            "max_drawdown": replay.max_drawdown,
            "profit_factor": replay.profit_factor,
            "objective_score": replay.objective_score,
            "objective_profile_id": replay.diagnostics.get("objective_profile_id", ""),
            "immutable_score": compact_score_payload(replay.diagnostics.get("immutable_score")),
            "coverage": replay.diagnostics.get("coverage", {}),
            "symbols": replay.diagnostics.get("symbols", []),
            "timeframes": replay.diagnostics.get("timeframes", []),
        }

    def build_phase_specs(
        self, diagnostics: Any, experiment_plan: Any, search_brief: Any
    ) -> list[Any]:
        return build_phase_specs_for_scope(
            scope_family=self.family,
            plugin_id=self.plugin_id,
            strategy_id=self.strategy_id,
            diagnostics=diagnostics,
            experiment_plan=experiment_plan,
            search_brief=search_brief,
        )

    def evaluate_candidate(self, candidate: Candidate, window: WindowSpec) -> CandidateEvaluation:
        if self._baseline is None:
            return CandidateEvaluation(
                candidate=candidate,
                objective_score=0.0,
                passed=False,
                reasons=["replay baseline has not been loaded for this plugin"],
            )
        incumbent_score = self._incumbent.objective_score if self._incumbent is not None else 0.0
        params = _candidate_params(candidate)
        patch_payload = evaluated_patch_payload(
            candidate,
            params,
            scope_family=self.family,
        )
        result = _run_bar_replay(
            manifest=self._baseline.manifest,
            data_bundle=self._baseline.data_bundle,
            bundle_manifest_path=self._baseline.bundle_manifest_path,
            window=window,
            threshold_bps=params["threshold_bps"],
            position_weight=params["position_weight"],
            max_symbols=params["max_symbols"],
            candidate_id=candidate.candidate_id,
            patch_payload=patch_payload,
        )
        improvement = result.objective_score - incumbent_score
        reasons = [
            (
                f"replay-backed evaluation score={result.objective_score:.8f}; "
                f"baseline={incumbent_score:.8f}; delta={improvement:.8f}"
            )
        ]
        passed = result.trade_count > 0 and improvement > 0.0
        if not self._baseline.adoption_enabled:
            passed = False
            reasons.append(
                "shadow_validated replay evidence retained; candidate adoption remains disabled"
            )
        elif not passed:
            reasons.append("candidate did not improve the replayed incumbent objective")
        return CandidateEvaluation(
            candidate=Candidate(
                candidate_id=candidate.candidate_id,
                family=candidate.family,
                payload={
                    **candidate.payload,
                    **patch_payload,
                    "replay_result": _replay_summary(result),
                },
            ),
            objective_score=result.objective_score,
            passed=passed,
            reasons=reasons,
        )

    def build_repair_candidates(self, failure_analysis: Any, round_chain: Any) -> list[Candidate]:
        return build_repair_candidates_for_scope(
            scope_family=self.family,
            plugin_id=self.plugin_id,
            strategy_id=self.strategy_id,
            failure_analysis=failure_analysis,
            round_chain=round_chain,
        )

    def build_confirmatory_variants(self, primary: Candidate, context: Any) -> list[Candidate]:
        return build_confirmatory_variants_for_scope(
            scope_family=self.family,
            primary=primary,
            context=context,
        )

    def write_round_n_plus_1(self, candidate: Candidate, output_dir: Path) -> Any:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = round_n_plus_1_payload(candidate)
        manifest_path = output_dir / "candidate_manifest.json"
        config_patch_path = output_dir / "config_patch.json"
        rollback_path = output_dir / "rollback_plan.json"
        manifest_path.write_text(
            json.dumps(payload["candidate_manifest"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        config_patch_path.write_text(
            json.dumps(payload["config_patch"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rollback_path.write_text(
            json.dumps(payload["rollback_plan"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        recommendation_path = output_dir / "round_n_plus_1_recommendation.json"
        recommendation_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "path": str(recommendation_path),
            "candidate_manifest_path": str(manifest_path),
            "config_patch_path": str(config_patch_path),
            "rollback_plan_path": str(rollback_path),
            "next_config_hash": payload["next_config_hash"],
            "parameter_patch_fingerprint": payload.get("parameter_patch_fingerprint", ""),
            "evaluated_patch_fingerprint": payload.get("evaluated_patch_fingerprint", ""),
        }

    def run_decision_parity(self, candidate: Candidate, fixtures: list[Path]) -> Any:
        return None


def _as_baseline(value: Any) -> CryptoReplayBaseline:
    if not isinstance(value, CryptoReplayBaseline):
        raise TypeError("crypto replay baseline is unavailable")
    return value


def _candidate_params(candidate: Candidate) -> dict[str, float | int]:
    patch = effective_parameter_patch(candidate, scope_family="crypto_portfolio")
    threshold_bps = 0.0
    position_weight = 1.0
    max_symbols = 3
    threshold_bps += patch_number(patch, "filter_threshold_bps_delta")
    threshold_bps += patch_number(patch, "exit_threshold_bps_delta")
    threshold_bps += patch_number(patch, "local_parameter_delta")
    threshold_bps += patch_number(patch, "stop_tighten_bps") * 0.10
    if patch.get("regime_filter"):
        threshold_bps *= 1.15
    if patch.get("session_gate"):
        threshold_bps *= 1.10
    if patch.get("direction") == "loosen":
        threshold_bps -= 1.0
    elif patch.get("direction") == "tighten":
        threshold_bps += 1.0
    position_weight *= patch_number(patch, "position_weight_multiplier", 1.0)
    max_symbols += patch_int(patch, "max_symbols_delta")
    return {
        "threshold_bps": max(0.0, threshold_bps),
        "position_weight": max(0.0, position_weight),
        "max_symbols": max(1, max_symbols),
    }


def _run_bar_replay(
    *,
    manifest: MonthlyRunManifest,
    data_bundle: DataBundleManifest,
    bundle_manifest_path: Path,
    window: WindowSpec,
    threshold_bps: float,
    position_weight: float,
    max_symbols: int,
    candidate_id: str,
    patch_payload: dict[str, Any] | None = None,
) -> ReplayResult:
    frames = _load_bundle_frames(data_bundle, bundle_manifest_path)
    trades: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    start_ts = datetime.combine(window.start, time.min, tzinfo=UTC)
    end_ts = datetime.combine(window.end, time.max, tzinfo=UTC)
    for key, frame in sorted(frames.items()):
        item = frame.loc[
            (frame["timestamp_utc"] >= start_ts) & (frame["timestamp_utc"] <= end_ts)
        ].copy()
        coverage.append(
            {
                "symbol": key[0],
                "timeframe": key[1],
                "rows": int(len(item)),
                "start_ts": item["timestamp_utc"].min().isoformat() if not item.empty else "",
                "end_ts": item["timestamp_utc"].max().isoformat() if not item.empty else "",
            }
        )
        if len(item) < 2:
            continue
        if len({trade["symbol"] for trade in trades}) >= max_symbols:
            break
        trade = _trade_from_frame(
            item,
            symbol=key[0],
            timeframe=key[1],
            threshold_bps=threshold_bps,
            position_weight=position_weight,
            candidate_id=candidate_id,
        )
        if trade is None:
            continue
        trades.append(trade)
        orders.extend(
            [
                {
                    "candidate_id": candidate_id,
                    "symbol": trade["symbol"],
                    "timeframe": trade["timeframe"],
                    "action": "entry",
                    "side": trade["side"],
                    "timestamp_utc": trade["entry_ts"],
                },
                {
                    "candidate_id": candidate_id,
                    "symbol": trade["symbol"],
                    "timeframe": trade["timeframe"],
                    "action": "exit",
                    "side": trade["side"],
                    "timestamp_utc": trade["exit_ts"],
                },
            ]
        )
    returns = [float(trade["return_pct"]) for trade in trades]
    net_return = sum(returns) / max(len(returns), 1)
    max_drawdown = _max_drawdown(returns)
    wins = [value for value in returns if value > 0]
    losses = [-value for value in returns if value < 0]
    profit_factor = sum(wins) / sum(losses) if losses else (sum(wins) if wins else 0.0)
    score = score_replay(
        profile=resolve_score_profile(
            family="crypto_portfolio",
            plugin_id=manifest.strategy_plugin_id,
            strategy_id=manifest.strategy_id,
        ),
        trades=trades,
        coverage=coverage,
        window=window,
        net_return=net_return,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        component_cap=manifest.score_component_cap,
    )
    return ReplayResult(
        run_id=manifest.run_id,
        window=window,
        trade_count=len(trades),
        net_return=net_return,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        objective_score=score.objective_score,
        trades=trades,
        orders=orders,
        diagnostics={
            "schema_version": "crypto_bar_replay_result_v1",
            "replay_engine_version": REPLAY_ENGINE_VERSION,
            "candidate_id": candidate_id,
            "bundle_id": data_bundle.bundle_id,
            "bundle_checksum": data_bundle.bundle_checksum,
            "symbols": sorted({key[0] for key in frames}),
            "timeframes": sorted({key[1] for key in frames}),
            "coverage": coverage,
            "threshold_bps": threshold_bps,
            "position_weight": position_weight,
            "max_symbols": max_symbols,
            "objective_profile_id": score.profile.profile_id,
            "immutable_score": score.to_payload(),
            **(patch_payload or {}),
            "trade_hash": _stable_hash(trades),
            "order_hash": _stable_hash(orders),
        },
    )


def _load_bundle_frames(
    data_bundle: DataBundleManifest, bundle_manifest_path: Path
) -> dict[tuple[str, str], pd.DataFrame]:
    slice_index = _read_slice_index(bundle_manifest_path)
    by_id = {
        str(item.get("manifest_id") or ""): item
        for item in slice_index.get("slices", [])
        if isinstance(item, dict)
    }
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for item in data_bundle.slice_manifests:
        if item.market != "crypto_perp":
            continue
        if item.timeframe.startswith("funding_"):
            continue
        index_item = by_id.get(item.manifest_id, {})
        paths = [Path(path) for path in index_item.get("canonical_paths", [])]
        if not paths:
            continue
        loaded = [
            pd.read_parquet(path, engine="pyarrow")
            for path in _resolve_existing_paths(data_bundle, bundle_manifest_path, paths)
        ]
        if not loaded:
            continue
        frame = pd.concat(loaded, ignore_index=True)
        frame = _normalize_ohlcv_frame(frame)
        if frame is None:
            continue
        frame = frame.sort_values("timestamp_utc")
        frame = frame.drop_duplicates(subset=["timestamp_utc"], keep="last")
        frames[(item.symbol.upper(), item.timeframe)] = frame
    return frames


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame | None:
    timestamp_column = _first_existing_column(frame, ("timestamp_utc", "timestamp", "time", "ts"))
    open_column = _first_existing_column(frame, ("open", "Open", "o"))
    high_column = _first_existing_column(frame, ("high", "High", "h"))
    low_column = _first_existing_column(frame, ("low", "Low", "l"))
    close_column = _first_existing_column(frame, ("close", "Close", "c"))
    volume_column = _first_existing_column(frame, ("volume", "Volume", "v", "vol"))
    if not all((timestamp_column, open_column, high_column, low_column, close_column)):
        return None
    normalized = frame.copy()
    normalized["timestamp_utc"] = _parse_timestamp_column(normalized[timestamp_column])
    normalized["open"] = pd.to_numeric(normalized[open_column], errors="coerce")
    normalized["high"] = pd.to_numeric(normalized[high_column], errors="coerce")
    normalized["low"] = pd.to_numeric(normalized[low_column], errors="coerce")
    normalized["close"] = pd.to_numeric(normalized[close_column], errors="coerce")
    normalized["volume"] = (
        pd.to_numeric(normalized[volume_column], errors="coerce") if volume_column else 0.0
    )
    normalized = normalized.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    return normalized


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    return next((name for name in names if name in frame.columns), "")


def _parse_timestamp_column(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", utc=True)
    return pd.to_datetime(values, utc=True)


def _read_slice_index(bundle_manifest_path: Path) -> dict[str, Any]:
    path = Path(bundle_manifest_path).with_name("slice_index.json")
    if not path.exists():
        return {"slices": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"slices": []}


def _resolve_existing_paths(
    data_bundle: DataBundleManifest,
    bundle_manifest_path: Path,
    paths: list[Path],
) -> list[Path]:
    roots = _candidate_roots(data_bundle, bundle_manifest_path)
    resolved: list[Path] = []
    for raw_path in paths:
        candidates = [raw_path] if raw_path.is_absolute() else [root / raw_path for root in roots]
        for candidate in candidates:
            if candidate.exists():
                resolved.append(candidate.resolve())
                break
    return resolved


def _candidate_roots(data_bundle: DataBundleManifest, bundle_manifest_path: Path) -> list[Path]:
    roots: list[Path] = []
    if data_bundle.data_repo_path:
        raw = Path(data_bundle.data_repo_path)
        if raw.is_absolute():
            roots.append(raw)
        else:
            bundle_roots = [bundle_manifest_path.parent, *bundle_manifest_path.parents]
            roots.extend(
                parent / raw for parent in bundle_roots
            )
    roots.extend([bundle_manifest_path.parent, *bundle_manifest_path.parents])
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _trade_from_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    threshold_bps: float,
    position_weight: float,
    candidate_id: str,
) -> dict[str, Any] | None:
    first = frame.iloc[0]
    last = frame.iloc[-1]
    entry = float(first["close"])
    exit_ = float(last["close"])
    if entry <= 0.0:
        return None
    raw_return = (exit_ - entry) / entry
    threshold = threshold_bps / 10_000.0
    if abs(raw_return) < threshold:
        return None
    side = "long" if raw_return > 0 else "short"
    signed_return = abs(raw_return) * position_weight
    return {
        "candidate_id": candidate_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "entry_ts": _iso(first["timestamp_utc"]),
        "exit_ts": _iso(last["timestamp_utc"]),
        "entry_price": entry,
        "exit_price": exit_,
        "return_pct": signed_return,
        "holding_bars": int(len(frame)),
    }


def _max_drawdown(returns: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = max(worst, (peak - equity) / peak if peak else 0.0)
    return worst


def _replay_summary(result: ReplayResult) -> dict[str, Any]:
    return {
        "trade_count": result.trade_count,
        "net_return": result.net_return,
        "max_drawdown": result.max_drawdown,
        "profit_factor": result.profit_factor,
        "objective_score": result.objective_score,
        "objective_profile_id": result.diagnostics.get("objective_profile_id", ""),
        "immutable_score": compact_score_payload(result.diagnostics.get("immutable_score")),
        "trade_hash": result.diagnostics.get("trade_hash", ""),
        "order_hash": result.diagnostics.get("order_hash", ""),
        "coverage": result.diagnostics.get("coverage", []),
        "parameter_patch": result.diagnostics.get("parameter_patch", {}),
        "evaluated_parameter_patch": result.diagnostics.get("evaluated_parameter_patch", {}),
        "parameter_patch_fingerprint": result.diagnostics.get(
            "parameter_patch_fingerprint",
            "",
        ),
        "evaluated_patch_fingerprint": result.diagnostics.get(
            "evaluated_patch_fingerprint",
            "",
        ),
        "evaluated_parameters": result.diagnostics.get("evaluated_parameters", {}),
    }


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _iso(value: Any) -> str:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
