"""Build diagnostics-only strategy discovery packets from monthly evidence."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from trading_assistant.schemas.discovery import (
    StrategyDiscoveryCluster,
    StrategyDiscoveryPacket,
    TradeReference,
)
from trading_assistant.schemas.learning_sufficiency import LearningSufficiencyManifest


MIN_DISCOVERY_CLUSTER_COUNT = 2
MIN_DISCOVERY_AFTER_COST_ESTIMATE = 0.01


class StrategyDiscoveryPacketBuilder:
    """Derive new-strategy discovery context from existing curated artifacts."""

    def __init__(
        self,
        curated_dir: Path,
        *,
        min_cluster_count: int = MIN_DISCOVERY_CLUSTER_COUNT,
        min_after_cost_estimate: float = MIN_DISCOVERY_AFTER_COST_ESTIMATE,
    ) -> None:
        self.curated_dir = Path(curated_dir)
        self.min_cluster_count = max(1, min_cluster_count)
        self.min_after_cost_estimate = max(0.0, float(min_after_cost_estimate))

    def build(
        self,
        *,
        run_id: str,
        run_month: str,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
        learning_sufficiency_manifest_path: Path | str = "",
    ) -> StrategyDiscoveryPacket:
        records = self._load_window_records(
            bot_id=bot_id,
            strategy_id=strategy_id,
            window_start=window_start,
            window_end=window_end,
        )
        learning = _load_learning_sufficiency(learning_sufficiency_manifest_path)
        missed_clusters = self._missed_opportunity_clusters(
            records["missed"],
            records["trades"],
        )
        denominator_clusters = self._denominator_clusters(
            records["funnels"],
            records["trades"],
            bot_id=bot_id,
            strategy_id=strategy_id,
        )
        evidence_paths = _dedupe([
            *(str(path) for path in records["paths"]),
            *(str(learning_sufficiency_manifest_path) for _ in [0] if str(learning_sufficiency_manifest_path)),
        ])
        return StrategyDiscoveryPacket(
            run_id=run_id,
            run_month=run_month,
            bot_id=bot_id,
            strategy_id=strategy_id,
            learning_sufficiency_manifest_path=str(learning_sufficiency_manifest_path or ""),
            supported_learning_capabilities=(
                learning.supported_learning_capabilities if learning is not None else []
            ),
            blocked_learning_capabilities=(
                learning.blocked_learning_capabilities if learning is not None else []
            ),
            missed_opportunity_clusters=missed_clusters,
            denominator_clusters=denominator_clusters,
            evidence_paths=evidence_paths,
        )

    def write(self, packet: StrategyDiscoveryPacket, artifact_root: Path) -> Path | None:
        path = Path(artifact_root) / "strategy_discovery_packet.json"
        if not self.has_material_clusters(packet):
            if path.exists():
                path.unlink()
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(packet.model_dump_json(indent=2), encoding="utf-8")
        return path

    def has_material_clusters(self, packet: StrategyDiscoveryPacket) -> bool:
        return any(
            cluster.support_count >= self.min_cluster_count
            and cluster.control_count > 0
            and cluster.estimated_after_cost_pnl >= self.min_after_cost_estimate
            for cluster in [*packet.missed_opportunity_clusters, *packet.denominator_clusters]
        )

    def _load_window_records(
        self,
        *,
        bot_id: str,
        strategy_id: str,
        window_start: date,
        window_end: date,
    ) -> dict[str, list[Any]]:
        trades: list[dict[str, Any]] = []
        missed: list[dict[str, Any]] = []
        funnels: list[dict[str, Any]] = []
        paths: list[Path] = []
        current = window_start
        while current <= window_end:
            date_str = current.isoformat()
            bot_dir = self.curated_dir / date_str / bot_id
            trade_path = bot_dir / "trades.jsonl"
            missed_path = bot_dir / "missed.jsonl"
            funnel_snapshot_path = bot_dir / "funnel_snapshots.jsonl"
            funnel_analysis_path = bot_dir / "funnel_analysis.json"
            regime_path = bot_dir / "regime_analysis.json"

            for record in _read_jsonl(trade_path, date_str=date_str):
                if _matches_strategy(record, strategy_id):
                    trades.append(record)
                    paths.append(trade_path)
            for record in _read_jsonl(missed_path, date_str=date_str):
                if _matches_strategy(record, strategy_id):
                    missed.append(record)
                    paths.append(missed_path)
            for record in _read_jsonl(funnel_snapshot_path, date_str=date_str):
                if _matches_strategy(record, strategy_id):
                    funnels.append(record)
                    paths.append(funnel_snapshot_path)
            funnel_analysis = _read_json(funnel_analysis_path)
            if isinstance(funnel_analysis, dict):
                funnels.extend(_funnel_records_from_analysis(
                    funnel_analysis,
                    date_str=date_str,
                    path=funnel_analysis_path,
                    strategy_id=strategy_id,
                ))
                paths.append(funnel_analysis_path)
            if regime_path.exists():
                paths.append(regime_path)
            current += timedelta(days=1)
        return {
            "trades": trades,
            "missed": missed,
            "funnels": funnels,
            "paths": _dedupe_paths(paths),
        }

    def _missed_opportunity_clusters(
        self,
        missed: list[dict[str, Any]],
        trades: list[dict[str, Any]],
    ) -> list[StrategyDiscoveryCluster]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in missed:
            grouped[_missed_cluster_key(record)].append(record)

        clusters: list[StrategyDiscoveryCluster] = []
        for (symbol, regime, setup_key), items in sorted(grouped.items()):
            if len(items) < self.min_cluster_count:
                continue
            estimate, source = _missed_after_cost_estimate(items)
            if estimate < self.min_after_cost_estimate:
                continue
            control = _control_slice(
                trades,
                symbol=symbol,
                regime=regime,
                setup_key=setup_key,
            )
            bot_id = _text_first(items[0], "bot_id")
            strategy_id = _text_first(items[0], "strategy_id")
            clusters.append(StrategyDiscoveryCluster(
                source="missed_opportunity",
                bot_id=bot_id,
                strategy_id=strategy_id,
                symbol=symbol,
                regime=regime,
                setup_key=setup_key,
                support_count=len(items),
                missed_count=len(items),
                control_count=int(control.get("control_count", 0)),
                estimated_after_cost_pnl=round(estimate, 4),
                estimated_after_cost_pnl_source=source,
                evidence=[_missed_reference(item) for item in items[:10]],
                evidence_paths=_dedupe([_text_first(item, "_evidence_path") for item in items]),
                control_slice=control,
                replay_plan=(
                    f"Replay entry logic for {symbol}/{regime}/{setup_key} against held-out "
                    "monthly windows before promotion."
                ),
                shadow_plan=(
                    f"Shadow-log hypothetical entries for {symbol}/{regime}/{setup_key} for "
                    "one completed month before approval routing."
                ),
            ))
        return clusters

    def _denominator_clusters(
        self,
        funnels: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        *,
        bot_id: str,
        strategy_id: str,
    ) -> list[StrategyDiscoveryCluster]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in funnels:
            stage, lost_count, denominator = _funnel_dropoff(record)
            if not stage or lost_count <= 0 or denominator <= 0:
                continue
            enriched = dict(record)
            enriched["_dropoff_stage"] = stage
            enriched["_lost_count"] = lost_count
            enriched["_denominator"] = denominator
            grouped[stage].append(enriched)

        clusters: list[StrategyDiscoveryCluster] = []
        avg_net = _average_net_after_cost(trades)
        for stage, items in sorted(grouped.items()):
            if len(items) < self.min_cluster_count:
                continue
            denominator = sum(int(item.get("_denominator") or 0) for item in items)
            lost_count = sum(int(item.get("_lost_count") or 0) for item in items)
            estimate = lost_count * avg_net
            if estimate < self.min_after_cost_estimate:
                continue
            control = {
                "control_type": "realized_trade_after_cost_average",
                "control_count": len(trades),
                "avg_realized_net_pnl": round(avg_net, 4),
                "denominator_count": denominator,
                "lost_count": lost_count,
            }
            clusters.append(StrategyDiscoveryCluster(
                source="denominator_snapshot",
                bot_id=bot_id,
                strategy_id=strategy_id,
                symbol="",
                regime="",
                setup_key=stage,
                support_count=len(items),
                denominator_count=denominator,
                control_count=len(trades),
                estimated_after_cost_pnl=round(estimate, 4),
                estimated_after_cost_pnl_source="funnel_dropoff_x_realized_avg_net_pnl",
                evidence=[
                    TradeReference(
                        date=str(item.get("_date") or ""),
                        bot_id=bot_id,
                        trade_id=stage,
                        pnl=0.0,
                        note=f"dropoff={int(item.get('_lost_count') or 0)} denominator={int(item.get('_denominator') or 0)}",
                    )
                    for item in items[:10]
                ],
                evidence_paths=_dedupe([_text_first(item, "_evidence_path") for item in items]),
                control_slice=control,
                replay_plan=(
                    f"Replay a candidate strategy that addresses the {stage} funnel drop-off "
                    "against held-out denominator snapshots."
                ),
                shadow_plan=(
                    f"Shadow-count would-enter events at {stage} for one completed month "
                    "before any approval routing."
                ),
            ))
        return clusters


def _load_learning_sufficiency(path: Path | str) -> LearningSufficiencyManifest | None:
    if not str(path):
        return None
    try:
        return LearningSufficiencyManifest.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
    except Exception:
        return None


def _read_jsonl(path: Path, *, date_str: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            record = dict(record)
            record["_date"] = date_str
            record["_evidence_path"] = str(path)
            records.append(record)
    return records


def _read_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _funnel_records_from_analysis(
    payload: dict[str, Any],
    *,
    date_str: str,
    path: Path,
    strategy_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    per_strategy = payload.get("per_strategy_breakdown")
    if isinstance(per_strategy, dict) and strategy_id in per_strategy:
        data = per_strategy.get(strategy_id)
        if isinstance(data, dict):
            records.append({
                **data,
                "strategy_id": strategy_id,
                "_date": date_str,
                "_evidence_path": str(path),
            })
            return records
    records.append({
        **payload,
        "_date": date_str,
        "_evidence_path": str(path),
    })
    return records


def _matches_strategy(record: dict[str, Any], strategy_id: str) -> bool:
    observed = str(record.get("strategy_id") or "")
    return not strategy_id or not observed or observed == strategy_id


def _missed_cluster_key(record: dict[str, Any]) -> tuple[str, str, str]:
    symbol = _text_first(record, "pair", "symbol", "instrument") or "unknown"
    regime = _text_first(record, "market_regime", "regime", "macro_regime") or "unknown"
    setup_key = (
        _text_first(record, "setup_key", "setup", "signal", "blocked_by", "block_reason", "root_cause")
        or "unknown_setup"
    )
    return symbol, regime, setup_key


def _missed_after_cost_estimate(items: list[dict[str, Any]]) -> tuple[float, str]:
    total = 0.0
    source = "missed_counterfactual_estimate"
    for item in items:
        value = _number_first(
            item,
            "would_have_pnl_net",
            "would_have_pnl",
            "estimated_pnl_net",
            "estimated_pnl",
            "counterfactual_pnl",
            "outcome_24h",
            "outcome_4h",
            "outcome_1h",
        )
        total += value
    return total, source


def _missed_reference(record: dict[str, Any]) -> TradeReference:
    return TradeReference(
        date=_text_first(record, "_date"),
        bot_id=_text_first(record, "bot_id"),
        trade_id=_text_first(record, "opportunity_id", "logical_event_id", "signal_id"),
        pnl=_number_first(record, "would_have_pnl_net", "would_have_pnl", "estimated_pnl", "outcome_24h"),
        regime=_text_first(record, "market_regime", "regime", "macro_regime"),
        signal_strength=_number_first(record, "signal_strength", "confidence"),
        note=_text_first(record, "blocked_by", "block_reason", "signal"),
    )


def _control_slice(
    trades: list[dict[str, Any]],
    *,
    symbol: str,
    regime: str,
    setup_key: str,
) -> dict[str, Any]:
    matching = [
        trade for trade in trades
        if (_text_first(trade, "pair", "symbol", "instrument") or "unknown") == symbol
        and (_text_first(trade, "market_regime", "regime", "macro_regime") or "unknown") == regime
    ]
    if not matching:
        matching = [
            trade for trade in trades
            if (_text_first(trade, "pair", "symbol", "instrument") or "unknown") == symbol
        ]
    net_values = [_trade_net_after_cost(trade) for trade in matching]
    wins = sum(1 for value in net_values if value > 0)
    return {
        "control_type": "realized_trades_same_symbol_regime",
        "symbol": symbol,
        "regime": regime,
        "setup_key": setup_key,
        "control_count": len(matching),
        "avg_net_pnl": round(sum(net_values) / len(net_values), 4) if net_values else 0.0,
        "win_rate": round(wins / len(net_values), 4) if net_values else 0.0,
    }


def _funnel_dropoff(record: dict[str, Any]) -> tuple[str, int, int]:
    totals = _stage_totals(record.get("stage_totals") or record.get("funnel") or record)
    pairs = [
        ("bars_to_indicators", "bars_received", "indicators_ready"),
        ("indicators_to_setups", "indicators_ready", "setups_detected"),
        ("setups_to_confirmations", "setups_detected", "confirmations"),
        ("confirmations_to_entries", "confirmations", "entries_attempted"),
        ("entries_to_fills", "entries_attempted", "fills"),
        ("fill_to_close", "fills", "trades_closed"),
    ]
    best_stage = ""
    best_lost = 0
    best_denominator = 0
    for label, before, after in pairs:
        denominator = totals.get(before, 0)
        lost = max(0, denominator - totals.get(after, 0))
        if denominator > 0 and lost > best_lost:
            best_stage = label
            best_lost = lost
            best_denominator = denominator
    return best_stage, best_lost, best_denominator


def _stage_totals(payload: object) -> dict[str, int]:
    stages = {
        "bars_received": 0,
        "indicators_ready": 0,
        "setups_detected": 0,
        "confirmations": 0,
        "entries_attempted": 0,
        "fills": 0,
        "trades_closed": 0,
    }
    aliases = {
        "signals_generated": "bars_received",
        "setups_seen": "setups_detected",
        "setups_qualified": "setups_detected",
        "confirmations_passed": "confirmations",
        "entries_taken": "entries_attempted",
    }
    if not isinstance(payload, dict):
        return stages
    for key, value in payload.items():
        stage = aliases.get(str(key), str(key))
        if stage not in stages:
            continue
        if isinstance(value, dict):
            stages[stage] += sum(_int(item) for item in value.values())
        else:
            stages[stage] += _int(value)
    return stages


def _average_net_after_cost(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return sum(_trade_net_after_cost(trade) for trade in trades) / len(trades)


def _trade_net_after_cost(trade: dict[str, Any]) -> float:
    explicit = _number_or_none(trade.get("net_pnl"))
    if explicit is not None:
        return explicit
    realized = _number_or_none(trade.get("realized_pnl_net"))
    if realized is not None:
        return realized
    gross = _number_first(trade, "pnl", "gross_pnl", "price_pnl_gross")
    costs = sum(_number_first(trade, key) for key in (
        "fees_paid",
        "total_fees",
        "commission",
        "tax",
        "spread_cost",
        "borrow_cost",
        "funding_paid",
    ))
    return gross - costs


def _text_first(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value):
            return str(value)
    return ""


def _number_first(record: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _number_or_none(record.get(key))
        if value is not None:
            return value
    return 0.0


def _number_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _dedupe_paths(values: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(value)
    return result
