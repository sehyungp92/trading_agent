from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .phase_state import _atomic_write_json, _utc_now_iso


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
    return cleaned[:80] or "backup"


_PHASE_FILE_RE = re.compile(r"^phase_(\d+)(?:_|\.|$)")


def _safe_remove(path: Path) -> None:
    """Remove a file, falling back to truncation on Windows lock errors."""
    try:
        path.unlink()
    except PermissionError:
        try:
            path.write_bytes(b"")
        except Exception:
            pass


class PhaseLogger:
    def __init__(self, output_dir: Path, round_name: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.round_name = round_name
        self._phase_loggers: dict[int, logging.Logger] = {}

    def get_phase_logger(self, phase: int) -> logging.Logger:
        if phase in self._phase_loggers:
            return self._phase_loggers[phase]

        path_key = str(self.output_dir.resolve()).lower().encode("utf-8")
        path_hash = hashlib.md5(path_key).hexdigest()[:12]
        logger_name = f"research.backtests.shared.auto.phase.{path_hash}.{phase}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(self.output_dir / f"phase_{phase}.log", mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        self._phase_loggers[phase] = logger
        return logger

    def log_activity(self, phase: int, action: str, details: dict) -> None:
        path = self.output_dir / "phase_activity_log.jsonl"
        entry = json.dumps(
            {"timestamp": _utc_now_iso(), "phase": phase, "action": action, **details},
            default=str,
        )
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(entry + "\n")

    def read_activity_log(self) -> list[dict]:
        """Read activity entries from JSONL (preferred) or legacy JSON array."""
        jsonl_path = self.output_dir / "phase_activity_log.jsonl"
        json_path = self.output_dir / "phase_activity_log.json"
        entries: list[dict] = []
        if jsonl_path.exists():
            with open(jsonl_path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        if json_path.exists():
            with open(json_path, encoding="utf-8") as handle:
                try:
                    legacy = json.load(handle)
                    if isinstance(legacy, list):
                        entries = legacy + entries
                except json.JSONDecodeError:
                    pass
        return entries

    def update_progress(self, phase: int, summary: dict) -> None:
        path = self.output_dir / "progress.json"
        current = {}
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                try:
                    current = json.load(handle)
                except json.JSONDecodeError:
                    current = {}
        current.setdefault("phases", {})
        current["last_updated"] = _utc_now_iso()
        current["current_phase"] = phase
        current["phases"][str(phase)] = summary
        _atomic_write_json(current, path)

    def close(self, from_phase: int = 0) -> None:
        for phase, logger in list(self._phase_loggers.items()):
            if phase < from_phase:
                continue
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            self._phase_loggers.pop(phase, None)

    def prune_progress(self, keep_phases: set[int], *, current_phase: int = 0) -> None:
        path = self.output_dir / "progress.json"
        current = {}
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                try:
                    current = json.load(handle)
                except json.JSONDecodeError:
                    current = {}
        phases = current.get("phases", {})
        current["phases"] = {
            str(int(key)): value
            for key, value in phases.items()
            if int(key) in keep_phases
        }
        current["last_updated"] = _utc_now_iso()
        current["current_phase"] = current_phase
        _atomic_write_json(current, path)

    def save_phase_output(self, phase: int, kind: str, data: str | dict) -> None:
        suffix = ".txt" if isinstance(data, str) else ".json"
        path = self.output_dir / f"phase_{phase}_{kind}{suffix}"
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            _atomic_write_json(data, path)

    def clear_generated_outputs(self, from_phase: int) -> None:
        self.close(from_phase=from_phase)
        for path in self.output_dir.iterdir():
            if not path.is_file():
                continue
            if path.name.endswith("_greedy_checkpoint.json"):
                continue
            match = _PHASE_FILE_RE.match(path.name)
            if match and int(match.group(1)) >= from_phase:
                _safe_remove(path)
        for name in ("round_evaluation.txt", "round_final_diagnostics.txt", "phase_activity_log.jsonl"):
            target = self.output_dir / name
            if target.exists():
                _safe_remove(target)

    def backup_state(self, state_path: Path, label: str) -> None:
        if not state_path.exists():
            return
        safe_label = _safe_label(label)
        backup_path = state_path.with_name(f"phase_state_pre_{safe_label}_backup.json")
        if backup_path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = state_path.with_name(f"phase_state_pre_{safe_label}_{stamp}_backup.json")
        shutil.copy2(state_path, backup_path)
