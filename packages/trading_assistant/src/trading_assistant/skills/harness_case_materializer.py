"""Materialize benchmark cases into replayable harness case folders."""
from __future__ import annotations

from pathlib import Path

from trading_assistant.schemas.benchmark_case import BenchmarkCase, BenchmarkSuite
from trading_assistant.schemas.harness_learning import HarnessExecutionInput
from trading_assistant.skills.evidence_ref_utils import evidence_ref_exists_within_roots


class HarnessCaseMaterializationError(ValueError):
    """Raised when a benchmark case lacks enough provenance to freeze safely."""


class HarnessCaseMaterializer:
    """Convert compiled benchmark cases into frozen executable case folders."""

    def __init__(self, findings_dir: Path, materialized_root: Path | None = None) -> None:
        self._findings_dir = Path(findings_dir)
        self._root_dir = _infer_root_dir(self._findings_dir)
        self._materialized_root = materialized_root or (
            self._findings_dir / "harness_cases" / "materialized"
        )

    def materialize_suite(
        self,
        suite: BenchmarkSuite,
        *,
        strict: bool = True,
    ) -> list[Path]:
        """Materialize every case and return created case folder paths."""

        paths: list[Path] = []
        for case in suite.cases:
            paths.append(self.materialize_case(case, strict=strict))
        return paths

    def materialize_case(self, case: BenchmarkCase, *, strict: bool = True) -> Path:
        """Write a frozen execution input for one benchmark case."""

        if strict:
            self._validate_provenance(case)

        case_dir = self._materialized_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        execution_input = self.execution_input_for_case(case)
        (case_dir / "execution_input.json").write_text(
            execution_input.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (case_dir / "benchmark_case.json").write_text(
            case.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return case_dir

    def execution_input_for_case(self, case: BenchmarkCase) -> HarnessExecutionInput:
        """Build the immutable execution input model for a case."""

        workflow = case.agent_type or _tag_value(case.case_tags, "workflow")
        bot_id = case.bot_id or _tag_value(case.case_tags, "bot")
        prompt_inputs = dict(case.input_snapshot or {})
        retrieval_profile = {
            "tags": list(case.case_tags),
            "query_terms": _query_terms(case),
            "agent_type": workflow,
            "bot_id": bot_id,
        }
        recorded_response = str(
            (case.output_snapshot or {}).get("raw_response")
            or (case.output_snapshot or {}).get("response_md")
            or (case.output_snapshot or {}).get("response")
            or ""
        )
        recorded_structured = (case.output_snapshot or {}).get("parsed_analysis") or (
            case.output_snapshot or {}
        ).get("structured_output") or {}
        if not isinstance(recorded_structured, dict):
            recorded_structured = {}

        return HarnessExecutionInput(
            case_id=case.case_id,
            workflow=workflow,
            bot_id=bot_id,
            strategy_id=str(prompt_inputs.get("strategy_id") or _tag_value(case.case_tags, "strategy")),
            source_run_id=case.source_run_id,
            run_metadata={
                "source": case.source.value,
                "source_id": case.source_id,
                "severity": case.severity.value,
                "provider": case.provider,
                "model": case.model,
                "date": case.date,
                "title": case.title,
            },
            prompt_inputs=prompt_inputs,
            retrieval_profile=retrieval_profile,
            allowed_artifacts=list(case.artifact_refs),
            expected_behavior=case.expected_behavior,
            forbidden_behavior=_forbidden_behavior(case),
            recorded_response=recorded_response,
            recorded_structured_output=recorded_structured,
        )

    def _validate_provenance(self, case: BenchmarkCase) -> None:
        if not case.source_id:
            raise HarnessCaseMaterializationError(f"{case.case_id} has no source_id")
        has_snapshot = bool(case.input_snapshot or case.output_snapshot or case.score_profile)
        has_refs = bool(case.artifact_refs or case.source_run_id)
        if not has_snapshot and not has_refs:
            raise HarnessCaseMaterializationError(
                f"{case.case_id} lacks artifact refs, source run, or frozen snapshots"
            )
        invalid = [
            ref for ref in case.artifact_refs
            if not self._artifact_exists(ref)
        ]
        if invalid:
            raise HarnessCaseMaterializationError(
                f"{case.case_id} has missing artifact refs: {', '.join(invalid)}"
            )

    def _artifact_exists(self, ref: str) -> bool:
        return evidence_ref_exists_within_roots(ref, [self._root_dir, self._findings_dir])


def _tag_value(tags: list[str], prefix: str) -> str:
    needle = f"{prefix}:"
    for tag in tags:
        value = str(tag)
        if value.startswith(needle):
            return value[len(needle):]
    return ""


def _query_terms(case: BenchmarkCase) -> list[str]:
    terms = [
        case.title,
        case.description,
        case.expected_behavior,
        case.actual_behavior,
        *[tag.replace(":", " ") for tag in case.case_tags],
    ]
    return [term for term in terms if str(term).strip()][:12]


def _forbidden_behavior(case: BenchmarkCase) -> list[str]:
    text = " ".join([case.expected_behavior, case.actual_behavior]).lower()
    forbidden = []
    for phrase in (
        "approval bypass",
        "bypass approval",
        "hallucinated artifact",
        "direct live trading command",
        "policy edit",
        "deterministic gate failure",
    ):
        if phrase in text:
            forbidden.append(phrase)
    return forbidden


def _infer_root_dir(findings_dir: Path) -> Path:
    if findings_dir.name == "findings" and findings_dir.parent.name == "memory":
        return findings_dir.parent.parent
    return findings_dir.parent
