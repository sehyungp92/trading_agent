"""JSONL activity logging and per-phase output management for optimization."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog


class PhaseLogger:
    """Appends structured log entries to a JSONL file and manages per-phase outputs."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = output_dir / "phase_activity_log.jsonl"
        self._log = structlog.get_logger("optimize")
        self._phase_handlers: dict[int, logging.Handler] = {}

    def log_event(
        self,
        event: str,
        phase: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a structured event to JSONL and structlog."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        if phase is not None:
            entry["phase"] = phase
        entry.update(kwargs)

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        # Sanitize non-ASCII in string kwargs for cp949 console (Windows)
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                safe_kwargs[k] = v.replace("\u2014", "--").replace("\u2013", "-")
            else:
                safe_kwargs[k] = v
        self._log.info(event, phase=phase, **safe_kwargs)

    def log_phase_start(self, phase: int, name: str, num_candidates: int) -> None:
        self.log_event(
            "phase.start",
            phase=phase,
            name=name,
            num_candidates=num_candidates,
        )

    def log_phase_end(
        self,
        phase: int,
        name: str,
        accepted: int,
        final_score: float,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.log_event(
            "phase.end",
            phase=phase,
            name=name,
            accepted=accepted,
            final_score=final_score,
            metrics=metrics,
        )

    def log_experiment_result(
        self,
        phase: int,
        experiment_name: str,
        score: float,
        accepted: bool,
        rejected: bool = False,
        reject_reason: str = "",
    ) -> None:
        self.log_event(
            "experiment.result",
            phase=phase,
            experiment=experiment_name,
            score=score,
            accepted=accepted,
            rejected=rejected,
            reject_reason=reject_reason,
        )

    def log_greedy_round(
        self,
        phase: int,
        round_num: int,
        best_name: str,
        best_score: float,
        improvement: float,
    ) -> None:
        self.log_event(
            "greedy.round",
            phase=phase,
            round=round_num,
            best=best_name,
            score=best_score,
            improvement=improvement,
        )

    def log_gate_result(
        self,
        phase: int,
        passed: bool,
        failures: list[str] | None = None,
        failure_category: str | None = None,
    ) -> None:
        self.log_event(
            "gate.result",
            phase=phase,
            passed=passed,
            failures=failures,
            failure_category=failure_category,
        )

    def log_retry(
        self,
        phase: int,
        retry_type: str,
        retry_count: int,
        reason: str = "",
    ) -> None:
        self.log_event(
            "phase.retry",
            phase=phase,
            retry_type=retry_type,
            retry_count=retry_count,
            reason=reason,
        )

    def log_analysis(
        self,
        phase: int,
        recommendation: str,
        summary: str,
    ) -> None:
        self.log_event(
            "phase.analysis",
            phase=phase,
            recommendation=recommendation,
            summary=summary,
        )

    # ── Per-phase output management ──────────────────────────────────

    def get_phase_logger(self, phase: int) -> logging.Logger:
        """Get or create a per-phase file logger."""
        logger_name = f"optimize.phase_{phase}"
        logger = logging.getLogger(logger_name)

        if phase not in self._phase_handlers:
            log_path = self._output_dir / f"phase_{phase}.log"
            handler = logging.FileHandler(str(log_path), mode="a")
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            )
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            self._phase_handlers[phase] = handler

        return logger

    def save_phase_output(
        self, phase: int, kind: str, data: str | dict
    ) -> Path:
        """Save per-phase output as text or JSON file.

        Returns the path of the saved file.
        """
        if isinstance(data, dict):
            path = self._output_dir / f"phase_{phase}_{kind}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        else:
            path = self._output_dir / f"phase_{phase}_{kind}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        return path

    def update_progress(self, phase: int, summary: dict) -> None:
        """Maintain a progress.json file with latest phase status."""
        progress_path = self._output_dir / "progress.json"

        progress: dict[str, Any] = {}
        if progress_path.exists():
            try:
                with open(progress_path, encoding="utf-8") as f:
                    progress = json.load(f)
            except (json.JSONDecodeError, OSError):
                progress = {}

        progress[str(phase)] = summary
        progress["last_updated"] = datetime.now(timezone.utc).isoformat()

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, default=str)

    def backup_state(self, state_path: Path, label: str) -> Path | None:
        """Create a timestamped backup copy of the state file."""
        if not state_path.exists():
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_name = f"phase_state_backup_{label}_{ts}.json"
        backup_path = self._output_dir / backup_name
        shutil.copy2(str(state_path), str(backup_path))
        return backup_path

    def clear_generated_outputs(self, from_phase: int) -> None:
        """Clean up generated output files from a given phase onward."""
        for path in self._output_dir.iterdir():
            if not path.is_file():
                continue
            name = path.name
            for p in range(from_phase, from_phase + 20):
                if name.startswith(f"phase_{p}_"):
                    path.unlink(missing_ok=True)
                    break

    def close(self, from_phase: int = 0) -> None:
        """Close per-phase file handlers."""
        for phase, handler in list(self._phase_handlers.items()):
            if phase >= from_phase:
                handler.close()
                logger = logging.getLogger(f"optimize.phase_{phase}")
                logger.removeHandler(handler)
                del self._phase_handlers[phase]
