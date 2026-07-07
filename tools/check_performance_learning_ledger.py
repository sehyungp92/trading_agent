from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "packages" / "trading_assistant" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trading_assistant.schemas.performance_learning_ledger import (  # noqa: E402
    AuthorityLevel,
    DecisionStage,
    LearningLayer,
    PerformanceLearningRecord,
    PerformanceMetricDeltas,
    PerformanceRecordType,
    SourceCadence,
)
from trading_assistant.skills.performance_learning_ledger import (  # noqa: E402
    PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME,
    PerformanceLearningLedgerStore,
    PerformanceLearningProjector,
    validate_performance_learning_records,
)
from trading_assistant.skills.performance_learning_relevance import (  # noqa: E402
    performance_learning_bot_relevance_keys,
    performance_learning_record_relation_keys,
)
from trading_assistant.testing.performance_learning_fixtures import (  # noqa: E402
    write_performance_learning_sources,
)


SOURCE_LEDGER_NAMES = (
    "proposal_ledger.jsonl",
    "strategy_change_ledger.jsonl",
    "portfolio_outcomes.jsonl",
)


def run_checks(*, memory_dir: Path, project_first: bool = False) -> list[str]:
    findings_dir = memory_dir / "findings"
    ledger_path = findings_dir / "performance_learning_ledger.jsonl"
    try:
        if project_first:
            PerformanceLearningProjector(findings_dir).project_to_ledger()
    except ValueError as exc:
        return [f"AM-14 performance_learning_ledger: malformed or invalid source JSONL row: {exc}"]
    store = PerformanceLearningLedgerStore(ledger_path)
    try:
        records = store.read(strict=True)
    except ValueError as exc:
        return [f"AM-14 performance_learning_ledger: malformed or invalid JSONL row: {exc}"]
    messages = validate_performance_learning_records(records)
    messages.extend(_check_bot_scoped_portfolio_prompt_summaries(store, records))
    refresh_error_path = findings_dir / PERFORMANCE_LEARNING_REFRESH_ERROR_FILENAME
    if refresh_error_path.exists():
        messages.append(
            "AM-14 performance_learning_ledger: runtime refresh failure marker exists "
            f"at {refresh_error_path}; fix source-ledger errors and refresh the projection."
        )
    missing_source_ledgers = _missing_source_ledgers(findings_dir)
    if missing_source_ledgers:
        messages.append(
            "AM-14 performance_learning_ledger: checked ledger is not backed by current "
            "source ledgers; missing or empty " + ", ".join(missing_source_ledgers)
        )
    try:
        current_records = PerformanceLearningProjector(findings_dir).build_records()
    except ValueError as exc:
        return [
            *messages,
            f"AM-14 performance_learning_ledger: malformed or invalid source JSONL row: {exc}",
        ]
    messages.extend(_check_projection_freshness(records, current_records, findings_dir))
    for record in records:
        if not record.source_records:
            messages.append(
                "AM-14 performance_learning_ledger: record "
                f"{record.record_id} lacks source_records lineage."
            )
        messages.extend(_check_source_record_paths(record, findings_dir))
        messages.extend(_check_stage_context_boundaries(record))
        if (
            record.decision_stage
            in {
                DecisionStage.PROPOSED,
                DecisionStage.EVALUATED,
                DecisionStage.APPROVED,
                DecisionStage.DEPLOYED,
                DecisionStage.REJECTED,
            }
            and record.realized_after_cost_deltas.has_any()
        ):
            messages.append(
                "AM-14 performance_learning_ledger: decision-stage record "
                f"{record.record_id} contains realized outcome deltas; keep outcome stages separate."
            )
        if (
            record.decision_stage in {DecisionStage.MEASURED, DecisionStage.FOLLOW_UP}
            and record.expected_deltas.has_any()
        ):
            messages.append(
                "AM-14 performance_learning_ledger: outcome-stage record "
                f"{record.record_id} contains expected deltas; keep proposal/evaluation stages separate."
            )
    messages.extend(_run_integration_projection_fixture())
    return messages


def _check_projection_freshness(records: list, current_records: list, findings_dir: Path) -> list[str]:
    messages: list[str] = []
    if not current_records:
        return [
            "AM-14 performance_learning_ledger: projector produced no current records from "
            "the checked source ledgers."
        ]
    messages.extend(_check_duplicate_current_ids(current_records, findings_dir))
    ledger_by_id = {record.record_id: _record_fingerprint(record, findings_dir) for record in records}
    current_by_id = {record.record_id: _record_fingerprint(record, findings_dir) for record in current_records}
    missing = sorted(set(current_by_id) - set(ledger_by_id))
    stale = sorted(
        record_id for record_id in set(current_by_id) & set(ledger_by_id)
        if current_by_id[record_id] != ledger_by_id[record_id]
    )
    extra = sorted(set(ledger_by_id) - set(current_by_id))
    if missing:
        messages.append(
            "AM-14 performance_learning_ledger: ledger is missing current source projections "
            + ", ".join(missing[:10])
        )
    if stale:
        messages.append(
            "AM-14 performance_learning_ledger: ledger has stale projection rows "
            + ", ".join(stale[:10])
        )
    if extra:
        messages.append(
            "AM-14 performance_learning_ledger: ledger has rows not backed by current source ledgers "
            + ", ".join(extra[:10])
        )
    return messages


def _run_integration_projection_fixture() -> list[str]:
    with TemporaryDirectory(prefix="performance_learning_check_") as root:
        memory = Path(root) / "memory"
        findings = memory / "findings"
        write_performance_learning_sources(findings)
        before = _source_snapshots(findings)
        records = PerformanceLearningProjector(findings).project_to_ledger()
        after = _source_snapshots(findings)
        stored = PerformanceLearningLedgerStore(
            findings / "performance_learning_ledger.jsonl"
        ).read(strict=True)
        messages = validate_performance_learning_records(stored)
        messages.extend(_check_bot_scoped_portfolio_prompt_summaries(
            PerformanceLearningLedgerStore(findings / "performance_learning_ledger.jsonl"),
            stored,
        ))
        messages.extend(_check_projection_freshness(stored, records, findings))
        for record in stored:
            messages.extend(_check_source_record_paths(record, findings))
            messages.extend(_check_stage_context_boundaries(record))
        if before != after:
            messages.append("AM-14 integration fixture: projector mutated source ledgers.")
        if not any(record.data_bundle_id for record in stored):
            messages.append("AM-14 integration fixture: data bundle metadata was not source-derived.")
        if not any(record.verifier_version for record in stored):
            messages.append("AM-14 integration fixture: verifier version was not source-derived.")
        if not any(record.artifact_authority_version for record in stored):
            messages.append("AM-14 integration fixture: artifact authority version was not source-derived.")
        if not any(record.portfolio_context.has_any() for record in stored):
            messages.append("AM-14 integration fixture: portfolio metrics context was not source-derived.")
        messages.extend(_run_prompt_relevance_fixture())
        return messages


def _run_prompt_relevance_fixture() -> list[str]:
    with TemporaryDirectory(prefix="performance_learning_prompt_check_") as root:
        ledger = PerformanceLearningLedgerStore(
            Path(root) / "memory" / "findings" / "performance_learning_ledger.jsonl"
        )
        base_time = datetime(2026, 6, 20, tzinfo=timezone.utc)
        strategy_record = PerformanceLearningRecord(
            record_type=PerformanceRecordType.STRATEGY,
            scope="strat1",
            bot_id="bot1",
            strategy_id="strat1",
            source_cadence=SourceCadence.MONTHLY,
            learning_layer=LearningLayer.TRADING_AUTHORITY,
            authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
            decision_stage=DecisionStage.MEASURED,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.04),
            summary="Bot strategy lesson must remain visible",
            event_time=base_time,
        )
        relevant_portfolio = PerformanceLearningRecord(
            record_id="portfolio-relevant-newer",
            record_type=PerformanceRecordType.PORTFOLIO,
            scope="core_portfolio",
            bot_id="PORTFOLIO",
            portfolio_id="core_portfolio",
            source_cadence=SourceCadence.FOLLOW_UP,
            learning_layer=LearningLayer.PERSISTENCE_CONFIRMATION,
            authority_level=AuthorityLevel.PERSISTENCE_CONFIRMATION,
            decision_stage=DecisionStage.FOLLOW_UP,
            realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.03),
            summary="Relevant portfolio lesson for strat1",
            event_time=base_time + timedelta(minutes=20),
            portfolio_context={
                "allocation_weights": {"strat1": 0.38, "other_strategy": 0.62},
                "marginal_contribution": {"strat1": 0.021},
            },
        )
        unrelated_portfolio_records = [
            PerformanceLearningRecord(
                record_id=f"portfolio-unrelated-newer-{index}",
                record_type=PerformanceRecordType.PORTFOLIO,
                scope=f"portfolio-{index}",
                bot_id="PORTFOLIO",
                portfolio_id=f"portfolio-{index}",
                source_cadence=SourceCadence.FOLLOW_UP,
                learning_layer=LearningLayer.PERSISTENCE_CONFIRMATION,
                authority_level=AuthorityLevel.PERSISTENCE_CONFIRMATION,
                decision_stage=DecisionStage.FOLLOW_UP,
                realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01 * index),
                summary=f"Unrelated portfolio lesson {index}",
                event_time=base_time + timedelta(minutes=index),
                portfolio_context={
                    "allocation_weights": {f"other_strategy_{index}": 0.4},
                    "marginal_contribution": {f"other_strategy_{index}": 0.01},
                },
            )
            for index in range(1, 13)
        ]
        unrelated_global_records = [
            PerformanceLearningRecord(
                record_id=f"global-unrelated-newer-{index}",
                record_type=PerformanceRecordType.STRATEGY,
                scope=f"global-strategy-{index}",
                strategy_id=f"global-strategy-{index}",
                source_cadence=SourceCadence.MONTHLY,
                learning_layer=LearningLayer.TRADING_AUTHORITY,
                authority_level=AuthorityLevel.MONTHLY_REPLAY_AUTHORITY,
                decision_stage=DecisionStage.MEASURED,
                realized_after_cost_deltas=PerformanceMetricDeltas(objective=0.01),
                summary=f"Generic strategy lesson {index}",
                event_time=base_time + timedelta(hours=1, minutes=index),
            )
            for index in range(1, 13)
        ]
        ledger.append_records([
            strategy_record,
            relevant_portfolio,
            *unrelated_portfolio_records,
            *unrelated_global_records,
        ])
        summaries = ledger.recent_summaries(bot_id="bot1", limit=10)
        portfolio_count = sum(
            summary.get("record_type") == PerformanceRecordType.PORTFOLIO.value
            for summary in summaries
        )
        messages: list[str] = []
        if not any(summary.get("record_id") == strategy_record.record_id for summary in summaries):
            messages.append(
                "AM-14 prompt relevance fixture: newer non-bot rows crowded out "
                "bot-specific strategy learning."
            )
        if not any(summary.get("record_id") == relevant_portfolio.record_id for summary in summaries):
            messages.append(
                "AM-14 prompt relevance fixture: bot-scoped prompt summaries omitted "
                "relevant portfolio performance-learning context."
            )
        if any(str(summary.get("record_id", "")).startswith("portfolio-unrelated") for summary in summaries):
            messages.append(
                "AM-14 prompt relevance fixture: bot-scoped prompt summaries admitted "
                "unrelated portfolio performance-learning context."
            )
        if portfolio_count > 3:
            messages.append(
                "AM-14 prompt relevance fixture: unrelated portfolio rows exceeded "
                "the bounded context budget."
            )
        return messages


def _missing_source_ledgers(findings_dir: Path) -> list[str]:
    return [
        name
        for name in SOURCE_LEDGER_NAMES
        if not (
            (findings_dir / name).exists()
            and (findings_dir / name).is_file()
            and (findings_dir / name).stat().st_size > 0
        )
    ]


def _source_snapshots(findings: Path) -> dict[str, str]:
    snapshots: dict[str, str] = {}
    for name in SOURCE_LEDGER_NAMES:
        path = findings / name
        snapshots[name] = path.read_text(encoding="utf-8") if path.exists() else ""
    return snapshots


def _record_fingerprint(record, findings_dir: Path) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("generated_at", None)
    _normalize_paths(payload, findings_dir)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _check_source_record_paths(record, findings_dir: Path) -> list[str]:
    messages: list[str] = []
    for source in record.source_records:
        label = f"{source.kind}:{source.id}" if source.id else source.kind
        if not source.path:
            messages.append(
                "AM-14 performance_learning_ledger: record "
                f"{record.record_id} source {label} lacks a backing path."
            )
            continue
        if not _source_path_exists(source.path, findings_dir):
            messages.append(
                "AM-14 performance_learning_ledger: record "
                f"{record.record_id} source {label} points to missing path {source.path}."
            )
    return messages


def _check_duplicate_current_ids(current_records: list, findings_dir: Path) -> list[str]:
    by_id: dict[str, list[str]] = {}
    for record in current_records:
        by_id.setdefault(record.record_id, []).append(_record_fingerprint(record, findings_dir))
    duplicate_ids = sorted(record_id for record_id, fingerprints in by_id.items() if len(fingerprints) > 1)
    if not duplicate_ids:
        return []
    return [
        "AM-14 performance_learning_ledger: projector emitted duplicate current record_id "
        + ", ".join(duplicate_ids[:10])
        + "; distinct proposal outcomes and strategy verdicts must remain prompt-visible."
    ]


def _check_stage_context_boundaries(record) -> list[str]:
    if record.decision_stage not in {
        DecisionStage.PROPOSED,
        DecisionStage.EVALUATED,
        DecisionStage.APPROVED,
        DecisionStage.DEPLOYED,
        DecisionStage.REJECTED,
    }:
        return []
    messages: list[str] = []
    if record.intended_learning_effects.outcome_prior_update:
        messages.append(
            "AM-14 performance_learning_ledger: decision-stage record "
            f"{record.record_id} contains outcome-prior learning effects."
        )
    if record.portfolio_context.has_any():
        messages.append(
            "AM-14 performance_learning_ledger: decision-stage record "
            f"{record.record_id} contains realized portfolio interaction context."
        )
    hindsight_paths = [
        path for path in [*record.evidence_paths, *[source.path for source in record.source_records]]
        if _is_hindsight_context_path(path)
    ]
    hindsight_sources = [
        source.kind for source in record.source_records
        if source.kind in HINDSIGHT_CONTEXT_SOURCE_KINDS
    ]
    if hindsight_paths or hindsight_sources:
        messages.append(
            "AM-14 performance_learning_ledger: decision-stage record "
            f"{record.record_id} contains outcome-only context "
            + ", ".join(sorted({*hindsight_paths, *hindsight_sources})[:10])
        )
    return messages


def _check_bot_scoped_portfolio_prompt_summaries(
    store: PerformanceLearningLedgerStore,
    records: list,
) -> list[str]:
    if not any(record.record_type == PerformanceRecordType.PORTFOLIO for record in records):
        return []
    bot_ids = sorted({
        record.bot_id for record in records
        if record.bot_id and record.bot_id.upper() != "PORTFOLIO"
    })
    messages: list[str] = []
    for bot_id in bot_ids:
        summaries = store.recent_summaries(bot_id=bot_id, limit=10)
        if not any(summary.get("record_type") == PerformanceRecordType.PORTFOLIO.value for summary in summaries):
            messages.append(
                "AM-14 performance_learning_ledger: bot-scoped prompt summaries for "
                f"{bot_id} omit portfolio performance-learning records."
            )
        if any(
            record.bot_id == bot_id and record.record_type == PerformanceRecordType.STRATEGY
            for record in records
        ) and not any(
            summary.get("bot_id") == bot_id
            and summary.get("record_type") == PerformanceRecordType.STRATEGY.value
            for summary in summaries
        ):
            messages.append(
                "AM-14 performance_learning_ledger: bot-scoped prompt summaries for "
                f"{bot_id} omit bot-specific strategy learning records."
            )
        portfolio_count = sum(
            summary.get("record_type") == PerformanceRecordType.PORTFOLIO.value
            for summary in summaries
        )
        if portfolio_count > 3:
            messages.append(
                "AM-14 performance_learning_ledger: bot-scoped prompt summaries for "
                f"{bot_id} include too many portfolio records and may crowd out strategy lessons."
            )
        portfolio_summary_ids = {
            str(summary.get("record_id", ""))
            for summary in summaries
            if summary.get("record_type") == PerformanceRecordType.PORTFOLIO.value
        }
        bot_relevance_keys = performance_learning_bot_relevance_keys(records, bot_id)
        for record in records:
            if (
                record.record_type == PerformanceRecordType.PORTFOLIO
                and record.record_id in portfolio_summary_ids
                and bot_relevance_keys
                and not (
                    performance_learning_record_relation_keys(
                        record,
                        include_portfolio_context=True,
                    )
                    & bot_relevance_keys
                )
            ):
                messages.append(
                    "AM-14 performance_learning_ledger: bot-scoped prompt summaries for "
                    f"{bot_id} include portfolio record {record.record_id} without an explicit "
                    "bot, strategy, allocation, proposal, or source relevance signal."
                )
    return messages


def _source_path_exists(raw: str, findings_dir: Path) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([
            ROOT / text,
            findings_dir / text,
            findings_dir.parent / text,
            findings_dir.parent.parent / text,
        ])
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return True
        except OSError:
            continue
    return False


def _normalize_paths(value, findings_dir: Path) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if isinstance(item, str) and _looks_like_path_key(key):
                value[key] = _normalize_path_text(item, findings_dir)
            else:
                _normalize_paths(item, findings_dir)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str):
                value[index] = _normalize_path_text(item, findings_dir)
            else:
                _normalize_paths(item, findings_dir)


def _looks_like_path_key(key: str) -> bool:
    return key == "path" or key.endswith("_path") or key.endswith("_paths")


def _normalize_path_text(raw: str, findings_dir: Path) -> str:
    text = str(raw or "").replace("\\", "/")
    if not text:
        return text
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([ROOT / raw, findings_dir / raw])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            continue
    return text


def _is_hindsight_context_path(raw: str) -> bool:
    return Path(str(raw or "").replace("\\", "/")).name.lower() in HINDSIGHT_CONTEXT_FILENAMES


HINDSIGHT_CONTEXT_SOURCE_KINDS = {
    "outcome_priors_snapshot",
    "portfolio_follow_up_outcome",
    "portfolio_metrics",
    "strategy_follow_up_verdict",
    "strategy_monthly_verdict",
}


HINDSIGHT_CONTEXT_FILENAMES = {
    "outcome_priors.json",
    "outcome_priors_snapshot.json",
    "portfolio_follow_up_outcome.json",
    "portfolio_metrics.json",
    "portfolio_rolling_metrics.json",
    "portfolio_risk_card.json",
    "portfolio_synergy.json",
    "strategy_monthly_verdict.json",
}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AM-14 performance-learning ledger coverage.")
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=ROOT / "packages" / "trading_assistant" / "memory",
    )
    parser.add_argument(
        "--project-first",
        action="store_true",
        help="append projection records from existing fixture/source ledgers before checking",
    )
    args = parser.parse_args(argv)
    issues = run_checks(memory_dir=args.memory_dir, project_first=args.project_first)
    if issues:
        print("performance-learning-ledger checks failed:")
        for issue in issues:
            print(issue)
            print(
                "  remediation: regenerate a non-placeholder performance_learning_ledger.jsonl "
                "from proposal, strategy-change, portfolio-outcome, monthly artifact, and loop-run sources."
            )
        return 1
    print(
        "performance-learning-ledger checks passed: AM-14 strategy/portfolio "
        "source authority, raw source/projection JSONL strictness, "
        "source-backed projection freshness, stable record identity, "
        "duplicate-current-ID guards, stage-gated enrichment, expected/realized "
        "deltas, weekly attribution, relevance-safe portfolio prompt surfacing, "
        "context, and stage separation hold"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
