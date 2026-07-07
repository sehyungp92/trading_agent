"""KRX OLR/KALCB bridge backed by the production ``k_stock_trader`` code."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from trading_assistant_backtest.contract_models import DecisionParityReport, MonthlyRunManifest
from trading_assistant_backtest.replay.decision_trace import DecisionTraceEvent
from trading_assistant_backtest.replay.parity import decision_parity_report_from_traces
from trading_assistant_backtest.strategies.live_clone import validate_pinned_head

PLUGIN_ID = "k-stock-olr-kalcb"
DECISION_API_VERSION = "k_stock_olr_kalcb_artifact_replay_decision_api_v1"


def build_k_stock_olr_kalcb_decision_parity_report(
    manifest: MonthlyRunManifest,
    *,
    candidate_id: str,
    fixture_paths: Iterable[str | Path],
    live_repo_path: str | Path,
    live_repo_commit_sha: str = "",
    backtest_adapter_commit_sha: str = "",
) -> DecisionParityReport:
    """Build a formal parity report for OLR/KALCB artifact decision surfaces."""

    repo_path = Path(live_repo_path)
    if live_repo_commit_sha:
        checkout_errors = validate_pinned_head(repo_path, live_repo_commit_sha)
        if checkout_errors:
            raise ValueError("; ".join(checkout_errors))

    live_events: list[DecisionTraceEvent] = []
    adapter_events: list[DecisionTraceEvent] = []
    evidence_paths: list[str] = []
    with tempfile.TemporaryDirectory(prefix="k-stock-olr-kalcb-parity-") as scratch:
        scratch_root = Path(scratch)
        with _k_stock_import_path(repo_path):
            for fixture_path in fixture_paths:
                path = Path(fixture_path)
                fixture = json.loads(path.read_text(encoding="utf-8"))
                evidence_paths.append(str(path))
                live, adapter = _decision_traces_from_fixture(
                    fixture,
                    scratch_root=scratch_root / _safe_name(path.stem),
                )
                live_events.extend(live)
                adapter_events.extend(adapter)

    return decision_parity_report_from_traces(
        manifest,
        candidate_id=candidate_id,
        live_events=live_events,
        adapter_events=adapter_events,
        evidence_paths=evidence_paths,
        live_repo_commit_sha=live_repo_commit_sha,
        backtest_adapter_commit_sha=backtest_adapter_commit_sha,
    )


def _decision_traces_from_fixture(
    fixture: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> tuple[list[DecisionTraceEvent], list[DecisionTraceEvent]]:
    fixture_id = str(fixture.get("fixture_id") or "").lower()
    if "kalcb" in fixture_id:
        return _kalcb_traces(fixture, scratch_root=scratch_root)
    if "olr" in fixture_id:
        return _olr_traces(fixture, scratch_root=scratch_root)
    raise ValueError(f"unknown OLR/KALCB fixture id: {fixture.get('fixture_id')!r}")


def _kalcb_traces(
    fixture: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> tuple[list[DecisionTraceEvent], list[DecisionTraceEvent]]:
    from backtests.strategies.kalcb import replay_cache as kalcb_replay_cache
    from deployment.olr_kalcb.artifacts import generate_kalcb_daily
    from strategy_kalcb.artifact_store import KALCBArtifactStore
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.research import candidate_config_fingerprint

    trade_date = date(2026, 2, 2)
    rows = _daily_fixture(trade_date)
    config_payload = {
        "kalcb.research.min_adv20_krw": 1_000_000,
        "kalcb.research.top_long_count": 2,
    }
    config = KALCBConfig.from_mapping(config_payload)
    sector_map = {"005930": "SEMIS", "000660": "SEMIS", "035420": "INTERNET"}
    source_fingerprint = str(fixture.get("source_fingerprint") or "fixture-source")
    config_hash = candidate_config_fingerprint(config, {}, sector_map)

    live_snapshot, _result = generate_kalcb_daily(
        rows,
        trade_date,
        config=config,
        sector_map=sector_map,
        artifact_root=scratch_root / "live_kalcb_store",
        source_fingerprint=source_fingerprint,
        candidate_config_hash=config_hash,
    )
    adapter_snapshot = kalcb_replay_cache._load_or_build_snapshot(
        trade_date,
        rows,
        config,
        source_fingerprint=source_fingerprint,
        candidate_config_hash=config_hash,
        requested_universe_count=len(rows),
        data_available_symbols=sorted(str(symbol).zfill(6) for symbol in rows),
        unavailable_symbols=(),
        sector_map=sector_map,
        store=KALCBArtifactStore(scratch_root / "adapter_kalcb_store"),
    )
    return (
        _snapshot_events(
            "KALCB",
            fixture,
            live_snapshot,
            config_payload=config_payload,
            config_hash=config_hash,
        ),
        _snapshot_events(
            "KALCB",
            fixture,
            adapter_snapshot,
            config_payload=config_payload,
            config_hash=config_hash,
        ),
    )


def _olr_traces(
    fixture: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> tuple[list[DecisionTraceEvent], list[DecisionTraceEvent]]:
    from backtests.strategies.olr import replay_cache as olr_replay_cache
    from backtests.strategies.olr.research_sweep import OLRResearchSweepDataset
    from strategy_common.clock import KST
    from strategy_common.market import MarketBar
    from strategy_olr.config import OLRConfig
    from strategy_olr.research import final_candidate_config_fingerprint
    from strategy_olr.research_generator import (
        generate_afternoon_candidate_snapshot,
        generate_candidate_snapshot,
    )

    trade_date = date(2026, 2, 2)
    rows = _daily_fixture(trade_date)
    bars = _bars_fixture(MarketBar, KST, trade_date)
    config_payload = {
        "olr.research.min_adv20_krw": 1_000_000,
        "olr.research.top_long_count": 2,
        "olr.signal.daily_min_score": 0.0,
        "olr.afternoon.top_n": 1,
        "olr.afternoon.score_mode": "momentum",
    }
    config = OLRConfig.from_mapping(config_payload)
    source_fingerprint = str(fixture.get("source_fingerprint") or "fixture-source")
    daily = generate_candidate_snapshot(
        rows,
        trade_date,
        config=config,
        artifact_root=None,
        source_fingerprint=source_fingerprint,
    )
    live_snapshot = generate_afternoon_candidate_snapshot(
        daily,
        bars,
        config=config,
        artifact_root=scratch_root / "live_olr_store",
    )
    dataset = _olr_dataset(
        OLRResearchSweepDataset,
        trade_date,
        rows,
        bars,
        config_payload,
        source_fingerprint,
    )
    stage1_hash = olr_replay_cache._stage1_config_hash(config, {})
    olr_replay_cache._stage1_snapshots(
        dataset,
        {},
        stage1_hash,
        {"artifact_root": str(scratch_root / "adapter_olr_stage1_store")},
    )
    adapter_snapshot = olr_replay_cache._load_or_build_stage2_snapshots(
        dataset,
        config,
        {},
        (trade_date,),
        stage1_hash,
        olr_replay_cache._candidate_config_hash(config, {}),
        {"artifact_root": str(scratch_root / "adapter_olr_store")},
    )[trade_date]
    config_hash = final_candidate_config_fingerprint(config)
    return (
        _snapshot_events(
            "OLR",
            fixture,
            live_snapshot,
            config_payload=config_payload,
            config_hash=config_hash,
        ),
        _snapshot_events(
            "OLR",
            fixture,
            adapter_snapshot,
            config_payload=config_payload,
            config_hash=config_hash,
        ),
    )


def _snapshot_events(
    strategy_id: str,
    fixture: Mapping[str, Any],
    snapshot: Any,
    *,
    config_payload: Mapping[str, Any],
    config_hash: str,
) -> list[DecisionTraceEvent]:
    payload = snapshot.to_json_dict() if hasattr(snapshot, "to_json_dict") else _clean(snapshot)
    candidates = _candidate_rows(payload)
    fixture_id = str(fixture.get("fixture_id") or strategy_id.lower())
    key = f"{strategy_id}:{fixture_id}"
    ts = _parse_ts(f"{payload.get('trade_date', '2026-02-02')}T00:00:00+00:00")
    metadata = dict(payload.get("metadata") or {})
    top = candidates[0] if candidates else {}
    return [
        _event(
            ts,
            "signals",
            key,
            {
                "artifact_hash": payload.get("artifact_hash", ""),
                "candidate_count": len(candidates),
                "source_fingerprint": payload.get("source_fingerprint", ""),
                "stage": metadata.get("artifact_stage", ""),
                "top_symbol": top.get("symbol", ""),
            },
        ),
        _event(
            ts,
            "filters",
            key,
            {
                "bar_count": fixture.get("bar_count", 0),
                "config_hash": config_hash,
                "date_window": fixture.get("date_window", {}),
                "input_hash": fixture.get("input_hash", ""),
                "symbols": fixture.get("symbols", []),
            },
        ),
        _event(ts, "entries", key, {"candidates": candidates}),
        _event(
            ts,
            "exits",
            key,
            {
                "action": "no_exit_order",
                "surface": "artifact_candidate_generation",
                "strategy_id": strategy_id,
            },
        ),
        _event(ts, "stops", key, {"stop_config": _stop_config(config_payload)}),
        _event(
            ts,
            "sizing",
            key,
            {
                "candidate_count": len(candidates),
                "selected_symbols": [row.get("symbol", "") for row in candidates],
                "sizing_config": _sizing_config(config_payload),
            },
        ),
        _event(
            ts,
            "risk_caps",
            key,
            {
                "config_hash": config_hash,
                "risk_config": _risk_config(config_payload),
                "universe_symbols": fixture.get("symbols", []),
            },
        ),
        _event(
            ts,
            "order_intent",
            key,
            {
                "intents": [
                    {
                        "action": "candidate_entry",
                        "rank": row.get("rank", index),
                        "strategy_id": strategy_id,
                        "symbol": row.get("symbol", ""),
                    }
                    for index, row in enumerate(candidates, start=1)
                ]
            },
        ),
    ]


def _candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(payload.get("candidates") or (), start=1):
        row = dict(item)
        rows.append(
            {
                "symbol": str(row.get("symbol") or "").zfill(6),
                "rank": int(row.get("rank") or index),
                "sector": str(row.get("sector") or ""),
                "daily_atr": row.get("daily_atr", 0),
                "selection_score": row.get("selection_score", row.get("momentum_score", 0)),
            }
        )
    return rows


def _daily_fixture(trade_date: date) -> dict[str, list[dict[str, Any]]]:
    return {
        "005930": _rows(trade_date, start=5_000, drift=45),
        "000660": _rows(trade_date, start=4_800, drift=28),
        "035420": _rows(trade_date, start=4_000, drift=15),
    }


def _rows(
    trade_date: date,
    *,
    start: float,
    drift: float,
    days: int = 80,
) -> list[dict[str, Any]]:
    first = trade_date - timedelta(days=days)
    rows = []
    for index in range(days):
        day = first + timedelta(days=index)
        close = start + drift * index
        rows.append(
            {
                "date": day.isoformat(),
                "open": close - max(drift * 0.5, 1.0),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return rows


def _bars_fixture(
    MarketBar: type,
    KST: Any,
    trade_date: date,
) -> dict[tuple[date, str], tuple[Any, ...]]:
    return {
        (trade_date, "005930"): (
            _bar(MarketBar, KST, "005930", trade_date, 9, 0, 100.0, 103.0, 99.0, 102.0),
            _bar(MarketBar, KST, "005930", trade_date, 14, 25, 102.0, 105.0, 101.0, 104.0),
        ),
        (trade_date, "000660"): (
            _bar(MarketBar, KST, "000660", trade_date, 9, 0, 100.0, 101.0, 99.0, 100.5),
            _bar(MarketBar, KST, "000660", trade_date, 14, 25, 100.5, 101.0, 100.0, 100.7),
        ),
    }


def _bar(
    MarketBar: type,
    KST: Any,
    symbol: str,
    trade_date: date,
    hour: int,
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Any:
    return MarketBar(
        symbol=symbol,
        timestamp=datetime(
            trade_date.year,
            trade_date.month,
            trade_date.day,
            hour,
            minute,
            tzinfo=KST,
        ),
        timeframe="5m",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=10_000,
        is_completed=True,
        source="fixture",
        source_fingerprint="fixture-source",
    )


def _olr_dataset(
    OLRResearchSweepDataset: type,
    trade_date: date,
    rows: dict[str, list[dict[str, Any]]],
    bars: dict[tuple[date, str], tuple[Any, ...]],
    config: Mapping[str, Any],
    source_fingerprint: str,
) -> Any:
    symbols = tuple(sorted(rows))
    intraday_symbols = {symbol for _day, symbol in bars}
    return OLRResearchSweepDataset(
        config=dict(config),
        source_fingerprint=source_fingerprint,
        daily_source_fingerprint=source_fingerprint,
        intraday_source_fingerprint=source_fingerprint,
        data_root=Path("."),
        daily_data_root=Path("."),
        timeframe="5m",
        symbols=symbols,
        requested_symbols=symbols,
        excluded_symbols={},
        intraday_available_symbols=tuple(sorted(intraday_symbols)),
        intraday_unavailable_symbols=tuple(
            symbol for symbol in symbols if symbol not in intraday_symbols
        ),
        daily_by_symbol=rows,
        flow_by_symbol={},
        foreign_flow_by_symbol={},
        institutional_flow_by_symbol={},
        index_by_code={},
        sector_map={},
        trading_dates=(trade_date,),
        bars_by_key=bars,
        train_start=trade_date,
        train_end=trade_date,
        holdout_start=trade_date + timedelta(days=1),
        coverage_report={},
    )


def _stop_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(config.items())
        if ".risk." in str(key) or ".exit." in str(key) or "stop" in str(key)
    }


def _sizing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(config.items())
        if "top_" in str(key) or "budget" in str(key) or "size" in str(key)
    }


def _risk_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(config.items())
        if ".risk." in str(key) or "max" in str(key) or "min" in str(key)
    }


@contextmanager
def _k_stock_import_path(repo_path: Path) -> Iterator[None]:
    text = str(repo_path.resolve())
    already_present = text in sys.path
    if not already_present:
        sys.path.insert(0, text)
    prior_modules = _pop_modules(
        (
            "backtests",
            "deployment",
            "kis_core",
            "oms_client",
            "strategy_common",
            "strategy_kalcb",
            "strategy_olr",
        )
    )
    try:
        yield
    finally:
        _pop_modules(
            (
                "backtests",
                "deployment",
                "kis_core",
                "oms_client",
                "strategy_common",
                "strategy_kalcb",
                "strategy_olr",
            )
        )
        sys.modules.update(prior_modules)
        if not already_present:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


def _pop_modules(prefixes: tuple[str, ...]) -> dict[str, Any]:
    modules = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for name in modules:
        sys.modules.pop(name, None)
    return modules


def _parse_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _event(
    ts: datetime,
    dimension: str,
    key: str,
    payload: Mapping[str, Any],
) -> DecisionTraceEvent:
    return DecisionTraceEvent(ts=ts, dimension=dimension, key=key, payload=_clean(payload))


def _clean(value: Any) -> Any:
    if is_dataclass(value):
        return _clean(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in sorted(value.items()) if item is not None}
    if isinstance(value, list | tuple):
        return [_clean(item) for item in value]
    return value


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

