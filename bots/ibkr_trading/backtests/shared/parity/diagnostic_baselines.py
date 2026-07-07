from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_manifest_path() -> Path:
    return repo_root() / "tests" / "fixtures" / "backtest_baselines" / "manifest.json"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or default_manifest_path()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    for entry in artifacts:
        validate_manifest_entry(entry)
    return manifest


def resolve_repo_path(relative_path: str, *, root: Path | None = None) -> Path:
    return (root or repo_root()) / Path(relative_path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def normalized_sha256_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return hashlib.sha256(normalize_artifact_text(text).encode("utf-8")).hexdigest().upper()


def load_summary_source(entry: dict[str, Any], *, root: Path | None = None) -> Any | None:
    source = entry.get("summary_source")
    if source:
        source_path = resolve_repo_path(source["path"], root=root)
        kind = source.get("kind", "json")
        if kind != "json":
            raise ValueError(f"Unsupported summary source kind: {kind}")
        return json.loads(source_path.read_text(encoding="utf-8"))

    artifact_path = resolve_repo_path(entry["artifact_path"], root=root)
    phase_manifest = artifact_path.parent / "phase_run_manifest.json"
    if phase_manifest.exists():
        return json.loads(phase_manifest.read_text(encoding="utf-8"))
    return None


def collect_baseline_snapshot(
    entry: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    base_root = root or repo_root()
    artifact_path = resolve_repo_path(entry["artifact_path"], root=base_root)
    summary_data = load_summary_source(entry, root=base_root)
    text = artifact_path.read_text(encoding="utf-8")
    return {
        "sha256": sha256_file(artifact_path),
        "metrics": parse_diagnostic_metrics(
            text,
            parser_kind=entry["parser_kind"],
            summary_data=summary_data,
        ),
    }


def validate_manifest_entry(entry: dict[str, Any]) -> None:
    required_fields = ("id", "artifact_path", "parser_kind", "sha256", "expected_metrics", "regeneration")
    missing = [field for field in required_fields if field not in entry]
    if missing:
        raise ValueError(f"Baseline manifest entry missing required fields: {', '.join(missing)}")

    expected_metrics = entry.get("expected_metrics")
    if not isinstance(expected_metrics, dict) or not expected_metrics:
        raise ValueError(f"Baseline manifest entry {entry.get('id')} has no expected_metrics payload")

    regeneration = entry.get("regeneration")
    if not isinstance(regeneration, dict):
        raise ValueError(f"Baseline manifest entry {entry.get('id')} must define regeneration metadata")

    executor = regeneration.get("executor")
    if executor not in {"python_file", "python_module", "manual"}:
        raise ValueError(
            f"Baseline manifest entry {entry.get('id')} has unsupported regeneration executor: {executor}"
        )

    entrypoint = regeneration.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError(f"Baseline manifest entry {entry.get('id')} must define a regeneration entrypoint")

    arguments = regeneration.get("arguments", [])
    if not isinstance(arguments, list) or any(not isinstance(arg, str) or not arg.strip() for arg in arguments):
        raise ValueError(
            f"Baseline manifest entry {entry.get('id')} regeneration arguments must be a list of strings"
        )

    notes = regeneration.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"Baseline manifest entry {entry.get('id')} regeneration notes must be a string")

    expected_output = regeneration.get("expected_output")
    if expected_output != entry["artifact_path"]:
        raise ValueError(
            f"Baseline manifest entry {entry.get('id')} regeneration expected_output must match artifact_path"
        )


def parse_diagnostic_metrics(
    text: str,
    *,
    parser_kind: str,
    summary_data: Any | None = None,
) -> dict[str, float]:
    if parser_kind in {"swing_helix_summary_json", "swing_strategy_summary_json"}:
        payload = json.loads(text)
        return {
            "total_trades": _to_number(payload["total_trades"]),
            "win_rate_pct": _to_number(payload["win_rate_pct"]),
            "profit_factor": _to_number(payload["profit_factor"]),
            "total_pnl": _to_number(payload["total_pnl"]),
            "total_r": _to_number(payload["total_r"]),
            "net_return_pct": _to_number(payload["net_return_pct"]),
            "final_equity": _to_number(payload["final_equity"]),
            "max_drawdown_pct": _to_number(payload["max_drawdown_pct"]),
        }
    if parser_kind == "swing_aggregate_summary":
        block = _section_between(text, "--- AGGREGATE SUMMARY ---", ["PER-SYMBOL SUMMARY"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win rate"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "total_pnl": _extract_labeled_number(block, "Total PnL"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
        }
    if parser_kind == "swing_helix_summary":
        block = _section_between(text, "Fee-net trade count:", ["A) PER-SYMBOL TRADE SUMMARY"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win Rate"),
            "profit_factor": _extract_labeled_number(block, "Profit Factor"),
            "total_pnl": _extract_labeled_number(block, "Total PnL"),
            "net_return_pct": _extract_labeled_number(block, "Net Return"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max Drawdown"),
        }
    if parser_kind == "brs_topline":
        block = _section_between(text, "A) Topline", ["B) Strength / Weakness Snapshot"])
        return {
            "campaigns": _extract_labeled_number(block, "Campaigns"),
            "fee_net_pnl": _extract_labeled_number(block, "Fee-net PnL"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
            "composite_score": _extract_labeled_number(block, "Composite score"),
        }
    if parser_kind == "swing_portfolio_synergy_round_summary":
        return {
            "final_equity": _extract_labeled_number(text, "Final equity"),
            "net_return_pct": _extract_labeled_number(text, "Net return"),
            "profit_factor": _extract_labeled_number(text, "Profit factor"),
            "win_rate_pct": _extract_labeled_number(text, "Win rate"),
            "max_drawdown_pct": _extract_labeled_number(text, "Max DD"),
            "sharpe": _extract_labeled_number(text, "Sharpe"),
            "sortino": _extract_labeled_number(text, "Sortino"),
            "calmar": _extract_labeled_number(text, "Calmar"),
            "entry_signals_fired": _extract_labeled_number(text, "Portfolio entry opportunities fired"),
            "entries_accepted": _extract_labeled_number(text, "Portfolio entries accepted"),
            "entries_blocked": _extract_labeled_number(text, "Portfolio entries blocked"),
            "score_total": _extract_score_total(text),
        }
    if parser_kind == "stock_portfolio_synergy_round_summary":
        return {
            "final_equity": _extract_labeled_number(text, "Final equity"),
            "net_return_pct": _extract_labeled_number(text, "Net return"),
            "profit_factor": _extract_labeled_number(text, "Profit factor"),
            "win_rate_pct": _extract_labeled_number(text, "Win rate"),
            "max_drawdown_pct": _extract_labeled_number(text, "Max DD"),
            "sharpe": _extract_labeled_number(text, "Sharpe"),
            "sortino": _extract_labeled_number(text, "Sortino"),
            "calmar": _extract_labeled_number(text, "Calmar"),
            "entry_signals_fired": _extract_labeled_number(text, "Fired entries"),
            "entries_accepted": _extract_labeled_number(text, "Accepted entries"),
            "entries_blocked": _extract_labeled_number(text, "Blocked entries"),
            "score_total": _extract_score_total(text),
        }
    if parser_kind == "momentum_performance_summary":
        summary_block = _section_between(text, "PERFORMANCE SUMMARY", ["FUNNEL"])
        return {
            "composite_score": _extract_labeled_number(text, "Composite Score"),
            "total_trades": _extract_labeled_number(summary_block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(summary_block, "Win rate"),
            "profit_factor": _extract_labeled_number(summary_block, "Profit factor"),
            "net_profit": _extract_labeled_number(summary_block, "Net profit"),
            "max_drawdown_pct": _extract_labeled_number(summary_block, "Max drawdown"),
        }
    if parser_kind == "momentum_performance_report":
        return {
            "total_trades": _extract_labeled_number(text, "Total trades"),
            "win_rate_pct": _extract_labeled_number(text, "Win rate"),
            "profit_factor": _extract_labeled_number(text, "Profit factor"),
            "net_profit": _extract_labeled_number(text, "Net profit"),
            "max_drawdown_pct": _extract_labeled_number(text, "Max drawdown"),
        }
    if parser_kind == "downturn_summary":
        block = _section_between(text, "--- Summary ---", ["--- Per-Engine Breakdown ---"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win rate"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "net_return_pct": _extract_labeled_number(block, "Net return"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
            "correction_pnl_pct": _extract_labeled_number(text, "Correction PnL"),
        }
    if parser_kind == "nqdtc_round_final_diagnostics":
        block = _section_between(text, "--- D1: Core Performance ---", ["--- D2: Exit Efficiency ---"])
        return {
            "total_trades": _extract_labeled_number(block, "Total trades"),
            "win_rate_pct": _extract_labeled_number(block, "Win rate"),
            "profit_factor": _extract_labeled_number(block, "Profit factor"),
            "net_return_pct": _extract_labeled_number(block, "Net return"),
            "max_drawdown_pct": _extract_labeled_number(block, "Max drawdown"),
            "sharpe": _extract_labeled_number(block, "Sharpe"),
            "calmar": _extract_labeled_number(block, "Calmar"),
        }
    if parser_kind == "stock_alcb_round_summary":
        core = _require_match(
            re.search(
                r"Core:\s*trades=(?P<trades>[+\-]?\d+(?:\.\d+)?),\s*"
                r"net_profit=(?P<pnl>[+\-]?\d+(?:,\d{3})*(?:\.\d+)?),.*?"
                r"pf=(?P<pf>[+\-]?\d+(?:\.\d+)?),\s*dd=(?P<dd>[+\-]?\d+(?:\.\d+)?)%",
                text,
            ),
            "ALCB core summary",
        )
        overview = _section_between(text, "1. Overview", ["2. Signal Funnel"])
        return {
            "total_trades": _to_number(core.group("trades")),
            "total_pnl": _to_number(core.group("pnl")),
            "profit_factor": _to_number(core.group("pf")),
            "max_drawdown_pct": _to_number(core.group("dd")),
            "win_rate_pct": _extract_labeled_number(overview, "Win Rate"),
        }
    if parser_kind == "stock_iaric_phase_summary":
        if not isinstance(summary_data, dict) or "live_metrics" not in summary_data:
            raise ValueError("IARIC baseline requires summary JSON with live_metrics")
        live_metrics = summary_data["live_metrics"]
        return {
            "total_trades": _to_number(live_metrics["n"]),
            "win_rate_pct": _to_number(live_metrics["wr"]) * 100.0,
            "profit_factor": _to_number(live_metrics["pf"]),
            "total_pnl": _to_number(live_metrics["pnl"]),
            "max_drawdown_r": _extract_labeled_number(text, "Max drawdown"),
        }
    raise ValueError(f"Unsupported parser kind: {parser_kind}")


def normalize_artifact_text(text: str) -> str:
    normalized_lines = [
        line.rstrip()
        for line in text.splitlines()
        if not re.match(r"^Generated:\s+.+$", line)
        and not re.match(r"^Data load:\s+.+$", line)
    ]
    return "\n".join(normalized_lines).strip() + "\n"


def _section_between(text: str, start_marker: str, end_markers: list[str]) -> str:
    start_index = text.find(start_marker)
    if start_index == -1:
        raise ValueError(f"Missing section start marker: {start_marker}")
    section = text[start_index + len(start_marker):]
    end_positions = [section.find(marker) for marker in end_markers if section.find(marker) != -1]
    if end_positions:
        section = section[: min(end_positions)]
    return section


def _extract_labeled_number(text: str, label: str) -> float:
    match = _require_match(
        re.search(
            rf"{re.escape(label)}\s*[:=]\s*\$?(?P<value>[+\-]?\d+(?:,\d{{3}})*(?:\.\d+)?)%?",
            text,
            flags=re.IGNORECASE,
        ),
        label,
    )
    return _to_number(match.group("value"))


def _extract_score_total(text: str) -> float:
    score_block = _section_between(text, "Score:", [])
    match = _require_match(
        re.search(r"^\s*total\s+(?P<value>[+\-]?\d+(?:\.\d+)?)\s*$", score_block, flags=re.MULTILINE),
        "score total",
    )
    return _to_number(match.group("value"))


def _to_number(value: Any) -> float:
    return float(str(value).replace(",", "").strip())


def _require_match(match: re.Match[str] | None, label: str) -> re.Match[str]:
    if match is None:
        raise ValueError(f"Could not parse metric block: {label}")
    return match
