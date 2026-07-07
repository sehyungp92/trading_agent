"""Weekly provider scoring and safe learned-routing recommendations."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ProviderRouteScorer:
    """Compute workflow-specific provider scores from attributed learning data."""

    def __init__(self, findings_dir: Path) -> None:
        self._findings_dir = Path(findings_dir)
        self._path = self._findings_dir / "provider_route_scores.jsonl"

    def recompute(self) -> list[dict]:
        suggestions = self._load_suggestions_by_id()
        stats: dict[tuple[str, str, str], dict] = {}

        def _bucket(workflow: str, provider: str, model: str) -> dict:
            key = (workflow, provider, model)
            return stats.setdefault(key, {
                "validation": [],
                "validation_weight": [],
                "outcomes": [],
                "outcome_weight": [],
                "calibration": [],
                "benchmarks": [],
                "benchmark_weight": [],
                "sample_count": 0,
            })

        def _add_samples(bucket: dict, count: object = 1) -> None:
            bucket["sample_count"] += max(1, int(_float(count, default=1.0)))

        for entry in self._load_jsonl("validation_log.jsonl"):
            workflow = str(entry.get("agent_type", "") or "").strip()
            provider = str(entry.get("provider", "") or "").strip()
            model = str(entry.get("model", "") or "").strip()
            total = int(entry.get("approved_count", 0) + entry.get("blocked_count", 0))
            if not workflow or not provider or total <= 0:
                continue
            bucket = _bucket(workflow, provider, model)
            bucket["validation"].append(entry.get("approved_count", 0) / total)
            bucket["validation_weight"].append(total)
            _add_samples(bucket)

        for entry in self._load_jsonl("outcomes.jsonl"):
            suggestion_id = entry.get("suggestion_id", "")
            suggestion = suggestions.get(suggestion_id, {})
            detection = suggestion.get("detection_context") or {}
            if not isinstance(detection, dict):
                detection = {}

            provider = str(entry.get("source_provider", "") or detection.get("source_provider", "")).strip()
            model = str(entry.get("source_model", "") or detection.get("source_model", "")).strip()
            workflow = self._infer_workflow(entry.get("source_run_id", "") or suggestion.get("source_report_id", ""))
            verdict = str(entry.get("verdict", "") or "").strip()
            if not workflow or not provider or verdict not in {"positive", "negative"}:
                continue

            quality = str(entry.get("measurement_quality", "") or "").lower()
            weight = 1.0 if quality == "high" else 0.7 if quality == "medium" else 0.4
            bucket = _bucket(workflow, provider, model)
            bucket["outcomes"].append(1.0 if verdict == "positive" else 0.0)
            bucket["outcome_weight"].append(weight)
            _add_samples(bucket)

        for entry in self._load_jsonl("monthly_outcomes.jsonl"):
            provider = str(
                entry.get("source_provider")
                or entry.get("provider")
                or entry.get("monthly_provider")
                or ""
            ).strip()
            model = str(
                entry.get("source_model")
                or entry.get("model")
                or entry.get("monthly_model")
                or ""
            ).strip()
            workflow = str(entry.get("workflow") or "").strip() or self._infer_workflow(
                entry.get("run_id", "")
            ) or "monthly_validation"
            verdict = str(entry.get("verdict", "") or "").strip().lower()
            if not workflow or not provider:
                continue
            score = _monthly_verdict_score(verdict)
            if score is None:
                continue
            source = str(entry.get("source") or "").strip().lower()
            confidence = self._clamp(_float(entry.get("confidence", 0.0)))
            weight = 1.5 if source == "follow_up" else 1.2
            if confidence:
                weight *= max(0.5, confidence)
            bucket = _bucket(workflow, provider, model)
            bucket["outcomes"].append(score)
            bucket["outcome_weight"].append(weight)
            _add_samples(bucket)

        for entry in self._load_jsonl("recalibrations.jsonl"):
            suggestion_id = entry.get("suggestion_id", entry.get("id", ""))
            suggestion = suggestions.get(suggestion_id, {})
            detection = suggestion.get("detection_context") or {}
            if not isinstance(detection, dict):
                detection = {}

            provider = str(detection.get("source_provider", "") or "").strip()
            model = str(detection.get("source_model", "") or "").strip()
            workflow = self._infer_workflow(entry.get("source_run_id", "") or suggestion.get("source_report_id", ""))
            if not workflow or not provider:
                continue

            revised = _float(entry.get("revised_confidence", 0.0))
            original = _float(entry.get("original_confidence", entry.get("confidence", 0.5)))
            bucket = _bucket(workflow, provider, model)
            bucket["calibration"].append(max(0.0, 1.0 - abs(revised - original)))
            _add_samples(bucket)

        for entry in self._load_jsonl("provider_benchmark_results.jsonl"):
            workflow = str(entry.get("workflow", "") or "").strip()
            provider = str(entry.get("provider", "") or "").strip()
            model = str(entry.get("model", "") or "").strip()
            if not workflow or not provider:
                continue
            failures = entry.get("governance_failures") or []
            if isinstance(failures, list) and failures:
                score = 0.0
            else:
                score = self._clamp(_float(entry.get("benchmark_score", entry.get("score", 0.0))))
            weight = _float(entry.get("sample_count", entry.get("benchmark_count", 1)), default=1.0)
            bucket = _bucket(workflow, provider, model)
            bucket["benchmarks"].append(score)
            bucket["benchmark_weight"].append(max(weight, 1.0))
            _add_samples(bucket, entry.get("sample_count", entry.get("benchmark_count", 1)))

        for entry in self._load_jsonl("harness_eval_results.jsonl"):
            provider = str(entry.get("provider", "") or "").strip()
            workflow = str(entry.get("workflow", "") or "").strip()
            if not provider or not workflow:
                continue
            model = str(entry.get("model", "") or "").strip()
            failures = entry.get("governance_failures") or []
            score = 0.0 if isinstance(failures, list) and failures else self._clamp(
                _float(entry.get("aggregate_score", 0.0))
            )
            weight = _float(entry.get("benchmark_count", 1), default=1.0)
            bucket = _bucket(workflow, provider, model)
            bucket["benchmarks"].append(score)
            bucket["benchmark_weight"].append(max(weight, 1.0))
            _add_samples(bucket, entry.get("benchmark_count", 1))

        for filename in (
            "monthly_model_review_validations.jsonl",
            "model_review_validations.jsonl",
        ):
            for entry in self._load_jsonl(filename):
                workflow = str(entry.get("workflow") or "monthly_model_review").strip()
                provider = str(entry.get("provider") or entry.get("source_provider") or "").strip()
                model = str(entry.get("model") or entry.get("source_model") or "").strip()
                if not workflow or not provider:
                    continue
                issues = entry.get("issues") or []
                score = 1.0 if bool(entry.get("valid")) else 0.0
                weight = max(_float(entry.get("sample_count", 1), default=1.0), len(issues) or 1)
                bucket = _bucket(workflow, provider, model)
                bucket["benchmarks"].append(score)
                bucket["benchmark_weight"].append(weight)
                _add_samples(bucket, entry.get("sample_count", len(issues) or 1))

        scores: list[dict] = []
        recorded_at = datetime.now(timezone.utc).isoformat()
        for (workflow, provider, model), bucket in sorted(stats.items()):
            validation_score = self._weighted_mean(bucket["validation"], bucket["validation_weight"])
            outcome_score = self._weighted_mean(bucket["outcomes"], bucket["outcome_weight"])
            calibration_score = self._mean(bucket["calibration"])
            benchmark_score = self._weighted_mean(bucket["benchmarks"], bucket["benchmark_weight"])

            components: list[tuple[float, float]] = []
            if validation_score is not None:
                components.append((0.3, validation_score))
            if outcome_score is not None:
                components.append((0.4, outcome_score))
            if calibration_score is not None:
                components.append((0.15, calibration_score))
            if benchmark_score is not None:
                components.append((0.15, benchmark_score))
            if not components:
                continue

            weight_total = sum(weight for weight, _ in components)
            composite = sum(weight * value for weight, value in components) / weight_total
            scores.append({
                "workflow": workflow,
                "provider": provider,
                "model": model,
                "composite_score": round(composite, 4),
                "validation_pass_rate": round(validation_score, 4) if validation_score is not None else None,
                "outcome_quality": round(outcome_score, 4) if outcome_score is not None else None,
                "calibration_accuracy": round(calibration_score, 4) if calibration_score is not None else None,
                "benchmark_quality": round(benchmark_score, 4) if benchmark_score is not None else None,
                "sample_count": int(bucket["sample_count"]),
                "recorded_at": recorded_at,
            })

        self._write_scores(scores)
        return scores

    def load_scores(self) -> list[dict]:
        return self._load_jsonl(self._path.name)

    def recommend_provider(
        self,
        workflow: str,
        requested_provider: str = "",
        *,
        min_samples: int | None = None,
        min_score_gap: float = 0.08,
    ) -> dict | None:
        if min_samples is None:
            min_samples = 10 if workflow in {"monthly_validation", "monthly_model_review"} else 5
        scores = [
            score for score in self.load_scores()
            if score.get("workflow") == workflow and score.get("sample_count", 0) >= min_samples
        ]
        if not scores:
            return None

        scores.sort(key=lambda item: (item.get("composite_score", 0.0), item.get("sample_count", 0)), reverse=True)
        best = scores[0]
        if not requested_provider or best.get("provider") == requested_provider:
            return dict(best)

        requested = next((score for score in scores if score.get("provider") == requested_provider), None)
        if requested is None:
            return None
        best_score = _float(best.get("composite_score", 0.0))
        requested_score = _float(requested.get("composite_score", 0.0))
        if best_score < requested_score + min_score_gap:
            return None
        recommendation = dict(best)
        recommendation["requested_provider"] = requested_provider
        recommendation["requested_model"] = requested.get("model") or ""
        recommendation["requested_composite_score"] = round(requested_score, 4)
        recommendation["score_gap"] = round(best_score - requested_score, 4)
        recommendation["min_score_gap"] = min_score_gap
        recommendation["rollback_condition"] = (
            f"revert to {requested_provider} if {workflow} score gap falls below "
            f"{min_score_gap:.2f}, benchmark quality turns negative, or validation failures recur"
        )
        return recommendation

    def _write_scores(self, scores: list[dict]) -> None:
        self._findings_dir.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for score in scores:
                handle.write(json.dumps(score) + "\n")

    def _load_jsonl(self, filename: str) -> list[dict]:
        path = self._findings_dir / filename
        if not path.exists():
            return []
        result: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def _load_suggestions_by_id(self) -> dict[str, dict]:
        return {
            entry.get("suggestion_id", ""): entry
            for entry in self._load_jsonl("suggestions.jsonl")
            if entry.get("suggestion_id")
        }

    @staticmethod
    def _infer_workflow(run_id: str) -> str:
        lower = str(run_id or "").lower()
        mapping = {
            "daily": "daily_analysis",
            "weekly": "weekly_analysis",
            "monthly-model-review": "monthly_model_review",
            "monthly_model_review": "monthly_model_review",
            "monthly-validation": "monthly_validation",
            "monthly_validation": "monthly_validation",
            "monthly": "monthly_validation",
            "triage": "triage",
            "discovery": "discovery_analysis",
            "reasoning": "outcome_reasoning",
        }
        for prefix, workflow in mapping.items():
            if lower.startswith(prefix):
                return workflow
        return ""

    @staticmethod
    def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
        if not values or not weights or len(values) != len(weights):
            return None
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
        return sum(value * weight for value, weight in zip(values, weights, strict=False)) / total_weight

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, value))


def _monthly_verdict_score(verdict: str) -> float | None:
    if verdict in {"keep", "positive"}:
        return 1.0
    if verdict in {"watch", "inconclusive"}:
        return 0.5
    if verdict in {"repair", "rollback", "quarantine", "negative"}:
        return 0.0
    return None


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
