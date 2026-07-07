from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .phase_state import _atomic_write_json, _utc_now_iso


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
    return entries


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _seconds_since(timestamp: str | None) -> float | None:
    dt = _parse_iso(timestamp)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()


def _process_snapshot(pid: int | None) -> dict[str, Any]:
    if pid is None:
        return {"pid": None, "alive": None}

    if os.name == "nt":
        script = (
            f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; "
            "if ($p) { "
            "@{alive=$true; pid=$p.Id; cpu=$p.CPU; ws=$p.WS; start_time=$p.StartTime.ToString('o')} | ConvertTo-Json -Compress "
            "} else { "
            f"@{{alive=$false; pid={pid}}} | ConvertTo-Json -Compress "
            "}"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            data = json.loads(result.stdout.strip() or "{}")
            if isinstance(data, dict):
                return data
        except Exception:
            return {"pid": pid, "alive": None}

    try:
        os.kill(pid, 0)
    except OSError:
        return {"pid": pid, "alive": False}
    return {"pid": pid, "alive": True}


def _status_from_activity(action: str | None, process_alive: bool | None) -> str:
    mapping = {
        "phase_start": "phase_started",
        "greedy_start": "greedy_running",
        "greedy_complete": "greedy_complete",
        "gate_check": "gate_checked",
        "diagnostics_run": "diagnostics_complete",
        "analysis_complete": "analysis_complete",
        "decision_improve_scoring": "greedy_retry_running",
        "decision_improve_diagnostics": "diagnostic_retry_running",
        "decision_advance": "phase_completed",
        "end_of_round": "completed",
    }
    status = mapping.get(action or "", "unknown")
    if process_alive is False and status not in {"completed", "phase_completed"}:
        return "stopped"
    return status


def _recent_phase_files(output_dir: Path, phase: int, limit: int = 8) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if not path.name.startswith(f"phase_{phase}"):
            continue
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "last_updated": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "size_bytes": stat.st_size,
            }
        )
    items.sort(key=lambda item: item["last_updated"], reverse=True)
    return items[:limit]


def _failed_gate_criteria(state: dict[str, Any], phase: int) -> list[dict[str, Any]]:
    gate = ((state.get("phase_gate_results") or {}).get(str(phase)) or {})
    criteria = gate.get("criteria") or []
    return [criterion for criterion in criteria if not criterion.get("passed", True)]


def _phase_focus(activity_entries: list[dict[str, Any]], phase: int) -> str:
    for entry in reversed(activity_entries):
        if int(entry.get("phase", 0)) != phase:
            continue
        if entry.get("action") == "phase_start":
            return str(entry.get("focus", ""))
    return ""


def _phase_start_value(activity_entries: list[dict[str, Any]], phase: int, key: str) -> Any:
    for entry in reversed(activity_entries):
        if int(entry.get("phase", 0)) != phase:
            continue
        if entry.get("action") == "phase_start":
            return entry.get(key)
    return None


def build_progress_snapshot(output_dir: Path, pid: int | None) -> dict[str, Any]:
    state = _read_json(output_dir / "phase_state.json")
    progress = _read_json(output_dir / "progress.json")
    activity = _read_jsonl(output_dir / "phase_activity_log.jsonl")
    process = _process_snapshot(pid)

    current_phase = int(progress.get("current_phase") or state.get("current_phase") or 0)
    completed_phases = list(state.get("completed_phases") or [])
    phase_progress = ((progress.get("phases") or {}).get(str(current_phase)) or {}) if current_phase else {}
    recent_activity = [entry for entry in activity if int(entry.get("phase", 0)) == current_phase][-8:]
    last_activity = recent_activity[-1] if recent_activity else {}
    focus = str(phase_progress.get("focus") or _phase_focus(activity, current_phase))
    candidate_count = phase_progress.get("candidate_count")
    if candidate_count is None and current_phase:
        candidate_count = _phase_start_value(activity, current_phase, "candidate_count")
    latest_analysis = _read_json(output_dir / f"phase_{current_phase}_analysis.json") if current_phase else {}
    status = str(
        phase_progress.get("status")
        or _status_from_activity(last_activity.get("action"), process.get("alive"))
    )

    snapshot = {
        "generated_at": _utc_now_iso(),
        "output_dir": str(output_dir),
        "process": process,
        "status": status,
        "round_completed": (output_dir / "round_evaluation.txt").exists(),
        "current_phase": current_phase,
        "completed_phases": completed_phases,
        "focus": focus,
        "candidate_count": candidate_count,
        "phase_started_at": phase_progress.get("phase_started_at") or ((state.get("phase_timestamps") or {}).get(str(current_phase), {}) or {}).get("started"),
        "phase_elapsed_seconds": _seconds_since(
            phase_progress.get("phase_started_at")
            or ((state.get("phase_timestamps") or {}).get(str(current_phase), {}) or {}).get("started")
        ),
        "scoring_retries": ((state.get("scoring_retries") or {}).get(str(current_phase), 0) if current_phase else 0),
        "diagnostic_retries": ((state.get("diagnostic_retries") or {}).get(str(current_phase), 0) if current_phase else 0),
        "last_activity": last_activity,
        "seconds_since_last_activity": _seconds_since(last_activity.get("timestamp")),
        "phase_progress": phase_progress,
        "failed_gate_criteria": _failed_gate_criteria(state, current_phase) if current_phase else [],
        "latest_analysis": {
            "recommendation": latest_analysis.get("recommendation"),
            "reason": latest_analysis.get("recommendation_reason"),
            "scoring_assessment": latest_analysis.get("scoring_assessment"),
            "diagnostic_gaps": latest_analysis.get("diagnostic_gaps"),
        } if latest_analysis else {},
        "recent_files": _recent_phase_files(output_dir, current_phase) if current_phase else [],
        "recent_activity": recent_activity,
    }
    return snapshot


def render_progress_text(snapshot: dict[str, Any]) -> str:
    process = snapshot.get("process") or {}
    last_activity = snapshot.get("last_activity") or {}
    latest_analysis = snapshot.get("latest_analysis") or {}
    lines = [
        f"Generated: {snapshot.get('generated_at', '')}",
        f"Round dir: {snapshot.get('output_dir', '')}",
        (
            f"Process: pid={process.get('pid')} alive={process.get('alive')} "
            f"cpu={process.get('cpu', 'n/a')} ws={process.get('ws', 'n/a')}"
        ),
        (
            f"Status: {snapshot.get('status', 'unknown')} | "
            f"phase={snapshot.get('current_phase', 0)} | "
            f"completed={snapshot.get('completed_phases', [])}"
        ),
        (
            f"Focus: {snapshot.get('focus', '') or 'n/a'} | "
            f"candidates={snapshot.get('candidate_count', 'n/a')} | "
            f"phase elapsed={_format_duration(snapshot.get('phase_elapsed_seconds'))}"
        ),
        (
            f"Retries: scoring={snapshot.get('scoring_retries', 'n/a')} "
            f"diagnostic={snapshot.get('diagnostic_retries', 'n/a')}"
        ),
        (
            f"Last activity: {last_activity.get('action', 'n/a')} at {last_activity.get('timestamp', 'n/a')} "
            f"({ _format_duration(snapshot.get('seconds_since_last_activity')) } ago)"
        ),
    ]

    phase_progress = snapshot.get("phase_progress") or {}
    greedy_progress = phase_progress.get("greedy_progress") or {}
    if greedy_progress:
        lines.append(
            "Greedy progress: "
            f"event={phase_progress.get('last_event', 'n/a')} "
            f"completed={greedy_progress.get('completed_candidates', 'n/a')}/"
            f"{greedy_progress.get('total_candidates', 'n/a')} "
            f"chunk={greedy_progress.get('chunk_index', 'n/a')}/"
            f"{greedy_progress.get('total_chunks', 'n/a')}"
        )

    failed_gate_criteria = snapshot.get("failed_gate_criteria") or []
    if failed_gate_criteria:
        first_failure = failed_gate_criteria[0]
        lines.append(
            "Latest gate miss: "
            f"{first_failure.get('name')} actual={first_failure.get('actual')} "
            f"target={first_failure.get('target')}"
        )

    if latest_analysis:
        lines.append(
            "Latest analysis: "
            f"{latest_analysis.get('recommendation', 'n/a')} | "
            f"{latest_analysis.get('reason', 'n/a')}"
        )

    recent_files = snapshot.get("recent_files") or []
    if recent_files:
        lines.append("Recent phase files:")
        for item in recent_files[:6]:
            lines.append(
                f"  - {item['name']} | {item['last_updated']} | {item['size_bytes']} bytes"
            )

    recent_activity = snapshot.get("recent_activity") or []
    if recent_activity:
        lines.append("Recent activity:")
        for item in recent_activity[-6:]:
            lines.append(
                f"  - {item.get('timestamp', 'n/a')} | {item.get('action', 'n/a')}"
            )

    return "\n".join(lines) + "\n"


def write_snapshot(output_dir: Path, snapshot: dict[str, Any]) -> None:
    _atomic_write_json(snapshot, output_dir / "live_progress.json")
    (output_dir / "live_progress.txt").write_text(render_progress_text(snapshot), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write live round progress summaries.")
    parser.add_argument("--output-dir", required=True, help="Round output directory to monitor.")
    parser.add_argument("--pid", type=int, default=None, help="Optional python PID for liveness checks.")
    parser.add_argument("--interval", type=float, default=30.0, help="Refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Write one snapshot and exit.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        snapshot = build_progress_snapshot(output_dir, args.pid)
        write_snapshot(output_dir, snapshot)
        if args.once:
            return 0
        if snapshot.get("round_completed"):
            return 0
        process = snapshot.get("process") or {}
        if process.get("alive") is False:
            return 0
        time.sleep(max(5.0, float(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
