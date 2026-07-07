"""Benchmark compiler — unifies learning signals into a replayable regression corpus."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trading_assistant.schemas.benchmark_case import (
    BenchmarkCase,
    BenchmarkSeverity,
    BenchmarkSource,
    BenchmarkSuite,
)

logger = logging.getLogger(__name__)


class BenchmarkCompiler:
    """Compile learning signals into benchmark cases with run/artifact linkage."""

    def __init__(self, findings_dir: Path, runs_dir: Path | None = None) -> None:
        self._findings_dir = Path(findings_dir)
        self._output_path = self._findings_dir / "benchmark_cases.jsonl"
        if self._findings_dir.name == "findings" and self._findings_dir.parent.name == "memory":
            self._root_dir = self._findings_dir.parent.parent
        else:
            self._root_dir = self._findings_dir.parent
        self._runs_dir = Path(runs_dir) if runs_dir is not None else self._root_dir / "runs"
        self._suggestions_by_id: dict[str, dict] = {}

    def compile(self, lookback_days: int = 90) -> BenchmarkSuite:
        """Compile all supported learning sources into a benchmark suite."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        self._suggestions_by_id = self._load_suggestions_by_id()
        cases: list[BenchmarkCase] = []

        cases.extend(self._compile_validation_blocks(cutoff))
        cases.extend(self._compile_negative_outcomes(cutoff))
        cases.extend(self._compile_calibration_misses(cutoff))
        cases.extend(self._compile_transfer_failures(cutoff))
        cases.extend(self._compile_discarded_harness_experiments(cutoff))

        seen: set[str] = set()
        unique: list[BenchmarkCase] = []
        for case in cases:
            if case.case_id in seen:
                continue
            seen.add(case.case_id)
            unique.append(case)

        summary = {
            "validation_blocks": sum(1 for c in unique if c.source == BenchmarkSource.VALIDATION_BLOCK),
            "negative_outcomes": sum(1 for c in unique if c.source == BenchmarkSource.NEGATIVE_OUTCOME),
            "calibration_misses": sum(1 for c in unique if c.source == BenchmarkSource.CALIBRATION_MISS),
            "transfer_failures": sum(1 for c in unique if c.source == BenchmarkSource.TRANSFER_FAILURE),
            "discarded_harness_experiments": sum(
                1 for c in unique if c.source == BenchmarkSource.DISCARDED_HARNESS_EXPERIMENT
            ),
        }
        return BenchmarkSuite(cases=unique, source_summary=summary)

    def compile_and_save(self, lookback_days: int = 90) -> int:
        """Compile all cases and append any newly discovered ones."""
        suite = self.compile(lookback_days)
        return self.save_suite(suite)

    def save_suite(self, suite: BenchmarkSuite) -> int:
        """Append any new cases from an already-compiled suite."""
        if not suite.cases:
            return 0

        existing_ids = self._load_existing_ids()
        new_cases = [case for case in suite.cases if case.case_id not in existing_ids]
        if not new_cases:
            return 0

        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._output_path.open("a", encoding="utf-8") as handle:
            for case in new_cases:
                handle.write(case.model_dump_json() + "\n")

        logger.info("Compiled %d new benchmark cases (sources=%s)", len(new_cases), suite.source_summary)
        return len(new_cases)

    def _compile_validation_blocks(self, cutoff: datetime) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for entry in self._load_jsonl("validation_log.jsonl"):
            ts = self._parse_ts(entry)
            if ts and ts < cutoff:
                continue

            approved = entry.get("approved_count", 0)
            blocked = entry.get("blocked_count", 0)
            total = approved + blocked
            if total < 4 or blocked == 0:
                continue
            block_rate = blocked / total
            if block_rate < 0.75:
                continue

            blocked_details = entry.get("blocked_details", [])
            titles = [detail.get("title", "") for detail in blocked_details[:5] if detail.get("title")]
            categories = [detail.get("category", "") for detail in blocked_details if detail.get("category")]
            reasons = [detail.get("reason", "") for detail in blocked_details if detail.get("reason")]
            run_id = entry.get("run_id", "")
            source_id = f"vlog:{entry.get('date', '')}:{entry.get('timestamp', '')}:{run_id}"
            artifact_snapshot = self._run_artifact_snapshot(run_id)

            cases.append(BenchmarkCase(
                case_id=BenchmarkCase.make_case_id(BenchmarkSource.VALIDATION_BLOCK, source_id),
                source=BenchmarkSource.VALIDATION_BLOCK,
                source_id=source_id,
                severity=BenchmarkSeverity.HIGH,
                agent_type=entry.get("agent_type", ""),
                date=entry.get("date", ""),
                provider=entry.get("provider", ""),
                model=entry.get("model", ""),
                source_run_id=run_id,
                title=f"High block rate: {blocked}/{total} suggestions blocked",
                description=f"Blocked: {', '.join(titles)}" if titles else "",
                expected_behavior="Most suggestions should pass validation",
                actual_behavior=f"{blocked}/{total} blocked ({block_rate:.0%})",
                artifact_refs=self._artifact_refs(run_id),
                case_tags=self._case_tags(
                    "source:validation_block",
                    self._workflow_tag(entry.get("agent_type", "")),
                    *[self._bot_tag(bot) for bot in self._bot_values(entry.get("bot_ids", ""))],
                    *[self._category_tag(category) for category in categories],
                    *[self._reason_tag(reason) for reason in reasons],
                ),
                score_profile={
                    "blocked_ratio": round(block_rate, 4),
                    "total_suggestions": float(total),
                },
                input_snapshot={
                    "approved_count": approved,
                    "blocked_count": blocked,
                    "bot_ids": entry.get("bot_ids", ""),
                    "run_id": run_id,
                },
                output_snapshot={"blocked_details": blocked_details, **artifact_snapshot},
            ))
        return cases

    def _compile_negative_outcomes(self, cutoff: datetime) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for entry in self._load_jsonl("outcomes.jsonl"):
            ts = self._parse_ts(entry)
            if ts and ts < cutoff:
                continue
            if entry.get("verdict") != "negative":
                continue
            if entry.get("measurement_quality") not in ("high", "medium"):
                continue

            suggestion_id = entry.get("suggestion_id", "")
            suggestion = self._suggestions_by_id.get(suggestion_id, {})
            detection_context = suggestion.get("detection_context") or {}
            if not isinstance(detection_context, dict):
                detection_context = {}

            source_run_id = entry.get("source_run_id", "") or suggestion.get("source_report_id", "")
            bot_id = entry.get("bot_id", "") or suggestion.get("bot_id", "")
            category = entry.get("category", "") or suggestion.get("category", "")
            provider = entry.get("source_provider", "") or detection_context.get("source_provider", "")
            model = entry.get("source_model", "") or detection_context.get("source_model", "")
            workflow = self._infer_workflow(source_run_id)
            pnl_delta = float(entry.get("pnl_after", 0.0) - entry.get("pnl_before", 0.0))
            artifact_snapshot = self._run_artifact_snapshot(source_run_id)

            cases.append(BenchmarkCase(
                case_id=BenchmarkCase.make_case_id(BenchmarkSource.NEGATIVE_OUTCOME, f"outcome:{suggestion_id}"),
                source=BenchmarkSource.NEGATIVE_OUTCOME,
                source_id=f"outcome:{suggestion_id}",
                severity=BenchmarkSeverity.CRITICAL,
                bot_id=bot_id,
                agent_type=workflow,
                date=entry.get("measurement_date", ""),
                provider=provider,
                model=model,
                source_run_id=source_run_id,
                title=f"Negative outcome for suggestion {suggestion_id}",
                description=f"PnL delta: {pnl_delta:+.2f}, quality: {entry.get('measurement_quality', '')}",
                expected_behavior="Implemented suggestion should improve performance",
                actual_behavior=f"Negative verdict (PnL delta: {pnl_delta:+.2f})",
                artifact_refs=self._artifact_refs(source_run_id),
                case_tags=self._case_tags(
                    "source:negative_outcome",
                    "verdict:negative",
                    self._workflow_tag(workflow),
                    self._bot_tag(bot_id),
                    self._category_tag(category),
                ),
                score_profile={
                    "pnl_delta": round(pnl_delta, 4),
                    "target_metric_delta": float(entry.get("target_metric_delta", 0.0)),
                },
                input_snapshot={
                    "suggestion_id": suggestion_id,
                    "implemented_date": entry.get("implemented_date", ""),
                    "category": category,
                    "source_run_id": source_run_id,
                },
                output_snapshot={
                    "pnl_before": entry.get("pnl_before", 0),
                    "pnl_after": entry.get("pnl_after", 0),
                    "win_rate_before": entry.get("win_rate_before", 0),
                    "win_rate_after": entry.get("win_rate_after", 0),
                    "verdict": entry.get("verdict", ""),
                    "measurement_quality": entry.get("measurement_quality", ""),
                    **artifact_snapshot,
                },
            ))
        return cases

    def _compile_calibration_misses(self, cutoff: datetime) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for entry in self._load_jsonl("recalibrations.jsonl"):
            ts = self._parse_ts(entry)
            if ts and ts < cutoff:
                continue

            revised = float(entry.get("revised_confidence", 0.0))
            original = float(entry.get("original_confidence", entry.get("confidence", 0.5)))
            delta = abs(revised - original)
            if delta < 0.3:
                continue

            suggestion_id = entry.get("suggestion_id", entry.get("id", ""))
            suggestion = self._suggestions_by_id.get(suggestion_id, {})
            detection_context = suggestion.get("detection_context") or {}
            if not isinstance(detection_context, dict):
                detection_context = {}
            source_run_id = entry.get("source_run_id", "") or suggestion.get("source_report_id", "")
            workflow = self._infer_workflow(source_run_id)
            bot_id = entry.get("bot_id", "") or suggestion.get("bot_id", "")
            category = entry.get("category", "") or suggestion.get("category", "")
            artifact_snapshot = self._run_artifact_snapshot(source_run_id)

            cases.append(BenchmarkCase(
                case_id=BenchmarkCase.make_case_id(BenchmarkSource.CALIBRATION_MISS, f"recalib:{suggestion_id}"),
                source=BenchmarkSource.CALIBRATION_MISS,
                source_id=f"recalib:{suggestion_id}",
                severity=BenchmarkSeverity.MEDIUM,
                bot_id=bot_id,
                agent_type=workflow,
                date=entry.get("date", ""),
                provider=detection_context.get("source_provider", ""),
                model=detection_context.get("source_model", ""),
                source_run_id=source_run_id,
                title=f"Calibration miss: {original:.0%} -> {revised:.0%} ({delta:+.0%})",
                description="; ".join(entry.get("lessons_learned", [])) or "",
                expected_behavior=f"Confidence should be close to actual ({revised:.0%})",
                actual_behavior=f"Original confidence {original:.0%} was off by {delta:.0%}",
                artifact_refs=self._artifact_refs(source_run_id),
                case_tags=self._case_tags(
                    "source:calibration_miss",
                    self._workflow_tag(workflow),
                    self._bot_tag(bot_id),
                    self._category_tag(category),
                ),
                score_profile={"confidence_delta": round(delta, 4)},
                input_snapshot={
                    "suggestion_id": suggestion_id,
                    "category": category,
                    "original_confidence": original,
                    "source_run_id": source_run_id,
                },
                output_snapshot={
                    "revised_confidence": revised,
                    "delta": delta,
                    **artifact_snapshot,
                },
            ))
        return cases

    def _compile_transfer_failures(self, cutoff: datetime) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for entry in self._load_jsonl("transfer_outcomes.jsonl"):
            ts = self._parse_ts(entry)
            if ts and ts < cutoff:
                continue
            if entry.get("verdict") != "negative":
                continue

            source_run_id = entry.get("source_run_id", "") or entry.get("proposal_run_id", "")
            target_bot = entry.get("target_bot", "")
            pattern_id = entry.get("pattern_id", "")
            artifact_snapshot = self._run_artifact_snapshot(source_run_id)

            cases.append(BenchmarkCase(
                case_id=BenchmarkCase.make_case_id(
                    BenchmarkSource.TRANSFER_FAILURE,
                    f"transfer:{pattern_id}:{target_bot}",
                ),
                source=BenchmarkSource.TRANSFER_FAILURE,
                source_id=f"transfer:{pattern_id}:{target_bot}",
                severity=BenchmarkSeverity.HIGH,
                bot_id=target_bot,
                agent_type=self._infer_workflow(source_run_id),
                provider=entry.get("source_provider", ""),
                model=entry.get("source_model", ""),
                source_run_id=source_run_id,
                title=f"Failed transfer: {pattern_id} -> {target_bot}",
                description=(
                    f"PnL delta: {float(entry.get('pnl_delta_7d', 0.0)):+.2f}, "
                    f"WR delta: {float(entry.get('win_rate_delta_7d', 0.0)):+.2%}"
                ),
                expected_behavior="Transferred pattern should maintain or improve performance",
                actual_behavior=f"Negative verdict for pattern {pattern_id} on {target_bot}",
                artifact_refs=self._artifact_refs(source_run_id),
                case_tags=self._case_tags(
                    "source:transfer_failure",
                    self._bot_tag(target_bot),
                    "transfer",
                    "verdict:negative",
                    "regime:mismatch" if entry.get("regime_matched") is False else "",
                ),
                score_profile={
                    "pnl_delta_7d": float(entry.get("pnl_delta_7d", 0.0)),
                    "win_rate_delta_7d": float(entry.get("win_rate_delta_7d", 0.0)),
                },
                input_snapshot={
                    "pattern_id": pattern_id,
                    "source_bot": entry.get("source_bot", ""),
                    "target_bot": target_bot,
                },
                output_snapshot={
                    "pnl_delta_7d": entry.get("pnl_delta_7d", 0),
                    "win_rate_delta_7d": entry.get("win_rate_delta_7d", 0),
                    "regime_matched": entry.get("regime_matched", False),
                    "verdict": entry.get("verdict", ""),
                    **artifact_snapshot,
                },
            ))
        return cases

    def _compile_discarded_harness_experiments(self, cutoff: datetime) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for entry in self._load_jsonl("discarded_harness_experiments.jsonl"):
            ts = self._parse_ts(entry)
            if ts and ts < cutoff:
                continue
            experiment_id = str(entry.get("experiment_id", "") or "").strip()
            variant_name = str(entry.get("variant_name", "") or "").strip()
            if not experiment_id or not variant_name:
                continue

            governance = [
                str(item).strip()
                for item in entry.get("governance_regressions", [])
                if str(item).strip()
            ]
            score_delta = float(entry.get("score_delta", 0.0) or 0.0)
            severity = (
                BenchmarkSeverity.CRITICAL
                if governance
                else BenchmarkSeverity.HIGH if score_delta <= -0.05 else BenchmarkSeverity.MEDIUM
            )
            changed_components = [
                str(item).strip()
                for item in entry.get("changed_components", [])
                if str(item).strip()
            ]
            warnings = [
                str(item).strip()
                for item in entry.get("future_warning_tags", [])
                if str(item).strip()
            ]
            program_ref = str(entry.get("program_ref", "") or "").strip()

            cases.append(BenchmarkCase(
                case_id=BenchmarkCase.make_case_id(
                    BenchmarkSource.DISCARDED_HARNESS_EXPERIMENT,
                    f"harness:{experiment_id}",
                ),
                source=BenchmarkSource.DISCARDED_HARNESS_EXPERIMENT,
                source_id=f"harness:{experiment_id}",
                severity=severity,
                agent_type="harness_evaluation",
                date=str(entry.get("recorded_at", ""))[:10],
                title=f"Discarded harness experiment: {variant_name}",
                description=str(entry.get("discard_reason", "") or entry.get("hypothesis", "")),
                expected_behavior="Future harness experiments should avoid this rejected pattern.",
                actual_behavior=f"Discarded with score delta {score_delta:+.4f}",
                artifact_refs=[program_ref] if program_ref else [],
                case_tags=self._case_tags(
                    "source:discarded_harness_experiment",
                    self._variant_tag(variant_name),
                    *[self._component_tag(component) for component in changed_components],
                    *[self._governance_tag(item) for item in governance],
                    *warnings,
                ),
                score_profile={
                    "score_delta": score_delta,
                    "aggregate_score": float(entry.get("aggregate_score", 0.0) or 0.0),
                    "baseline_score": float(entry.get("baseline_score", 0.0) or 0.0),
                },
                input_snapshot={
                    "hypothesis": entry.get("hypothesis", ""),
                    "changed_components": changed_components,
                    "anti_overfitting_assessment": entry.get("anti_overfitting_assessment", ""),
                    "program_ref": program_ref,
                },
                output_snapshot={
                    "kept": bool(entry.get("kept", False)),
                    "discard_reason": entry.get("discard_reason", ""),
                    "governance_regressions": governance,
                    "future_warning_tags": warnings,
                },
            ))
        return cases

    def _load_suggestions_by_id(self) -> dict[str, dict]:
        return {
            entry.get("suggestion_id", ""): entry
            for entry in self._load_jsonl("suggestions.jsonl")
            if entry.get("suggestion_id")
        }

    def _artifact_refs(self, run_id: str) -> list[str]:
        if not run_id:
            return []
        run_dir = self._runs_dir / run_id
        if not run_dir.exists():
            return []
        refs: list[str] = []
        for name in (
            "metadata.json",
            "instructions.md",
            "response.md",
            "parsed_analysis.json",
            "validator_notes.md",
            "daily_report.md",
            "weekly_report.md",
            "discovery_report.md",
        ):
            artifact = run_dir / name
            if artifact.exists():
                refs.append(str((Path("runs") / run_id / name).as_posix()))
        return refs

    def _run_artifact_snapshot(self, run_id: str) -> dict:
        if not run_id:
            return {}
        run_dir = self._runs_dir / run_id
        if not run_dir.exists():
            return {}

        snapshot: dict = {}
        parsed = self._read_json(run_dir / "parsed_analysis.json")
        if isinstance(parsed, dict):
            snapshot["parsed_analysis"] = parsed

        response = self._read_text(run_dir / "response.md", max_chars=80_000)
        if response:
            snapshot["raw_response"] = response

        validator_notes = self._read_text(run_dir / "validator_notes.md", max_chars=20_000)
        if validator_notes:
            snapshot["validator_notes"] = validator_notes
        return snapshot

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _read_text(path: Path, *, max_chars: int) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")[:max_chars]
        except OSError:
            return ""

    def _load_jsonl(self, filename: str) -> list[dict]:
        path = self._findings_dir / filename
        if not path.exists():
            return []
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries

    def _load_existing_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self._output_path.exists():
            return ids
        for line in self._output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line).get("case_id", ""))
            except (json.JSONDecodeError, ValueError):
                continue
        return ids

    @staticmethod
    def _parse_ts(entry: dict) -> datetime | None:
        for key in ("timestamp", "created_at", "recorded_at", "measurement_date", "measured_at", "transferred_at", "date"):
            value = entry.get(key)
            if not value or not isinstance(value, str):
                continue
            try:
                if "T" in value:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _infer_workflow(run_id: str) -> str:
        lower = str(run_id or "").lower()
        prefixes = {
            "daily": "daily_analysis",
            "weekly": "weekly_analysis",
            "monthly": "monthly_validation",
            "triage": "triage",
            "discovery": "discovery_analysis",
            "reasoning": "outcome_reasoning",
        }
        for prefix, workflow in prefixes.items():
            if lower.startswith(prefix):
                return workflow
        return ""

    @staticmethod
    def _case_tags(*values: str) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _slug(value: str) -> str:
        text = str(value or "").strip().lower()
        chars: list[str] = []
        prev_sep = False
        for char in text:
            if char.isalnum():
                chars.append(char)
                prev_sep = False
            elif not prev_sep:
                chars.append("_")
                prev_sep = True
        return "".join(chars).strip("_")

    @classmethod
    def _workflow_tag(cls, workflow: str) -> str:
        slug = cls._slug(workflow)
        return f"workflow:{slug}" if slug else ""

    @classmethod
    def _bot_tag(cls, bot_id: str) -> str:
        slug = cls._slug(bot_id)
        return f"bot:{slug}" if slug else ""

    @classmethod
    def _category_tag(cls, category: str) -> str:
        slug = cls._slug(category)
        return f"category:{slug}" if slug else ""

    @classmethod
    def _reason_tag(cls, reason: str) -> str:
        slug = cls._slug(reason)
        return f"reason:{slug}" if slug else ""

    @classmethod
    def _component_tag(cls, component: str) -> str:
        slug = cls._slug(component)
        return f"component:{slug}" if slug else ""

    @classmethod
    def _governance_tag(cls, issue: str) -> str:
        slug = cls._slug(issue)
        return f"governance:{slug}" if slug else ""

    @classmethod
    def _variant_tag(cls, variant: str) -> str:
        slug = cls._slug(variant)
        return f"variant:{slug}" if slug else ""

    @staticmethod
    def _bot_values(bot_ids: str | list[str]) -> list[str]:
        if isinstance(bot_ids, list):
            return [str(bot).strip() for bot in bot_ids if str(bot).strip()]
        return [bot.strip() for bot in str(bot_ids or "").split(",") if bot.strip()]
