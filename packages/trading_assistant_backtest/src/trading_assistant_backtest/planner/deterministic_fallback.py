"""Deterministic experiment plan builder."""

from __future__ import annotations

from pathlib import Path

from trading_assistant_backtest.contract_models import MonthlyRunManifest, OptimizerExperimentPlan
from trading_assistant_backtest.scoring.objective import capped_components


def build_deterministic_plan(
    manifest: MonthlyRunManifest, artifact_root: Path
) -> OptimizerExperimentPlan:
    evidence_paths = [
        str(artifact_root / "gap_attribution.json"),
        str(artifact_root / "objective_breakdown.json"),
    ]
    if manifest.monthly_search_brief_path:
        evidence_paths.append(manifest.monthly_search_brief_path)
    families = _candidate_families(manifest)
    return OptimizerExperimentPlan(
        run_id=manifest.run_id,
        objective_version=manifest.objective_version,
        score_components=capped_components(
            manifest.score_component_cap,
            plugin_id=manifest.strategy_plugin_id,
            strategy_id=manifest.strategy_id,
        ),
        phase_order=["diagnostics", "signal_quality", "trade_management"],
        candidate_families=families,
        gate_expectations=[
            "positive purged folds",
            "cost sensitivity passes",
            "outlier concentration is bounded",
            "portfolio risk constraints pass",
        ],
        overfit_risks=[
            "latest completed month is selection-OOS only",
            "weekly hints may reorder families but cannot satisfy gates",
            "rollback and negative-prior families require extra caution",
        ],
        evidence_paths=evidence_paths,
        source_weekly_signal_ids=manifest.source_weekly_signal_ids
        or [
            str(item)
            for item in manifest.monthly_search_guidance.get("source_weekly_signal_ids", [])
        ],
    )


def _candidate_families(manifest: MonthlyRunManifest) -> list[dict]:
    families: list[str] = []
    guidance = manifest.monthly_search_guidance or {}
    requirements = guidance.get("plan_requirements") if isinstance(guidance, dict) else {}
    if not isinstance(requirements, dict):
        requirements = {}
    for key in ("candidate_families", "rollback_families", "negative_prior_families"):
        for family in requirements.get(key, []) or []:
            if str(family) not in families:
                families.append(str(family))
    for key in ("seed_candidates", "priority_families", "rollback_candidates", "negative_priors"):
        for item in guidance.get(key, []) or []:
            family = item.get("family") if isinstance(item, dict) else item
            if str(family) and str(family) not in families:
                families.append(str(family))
    if not families:
        families = ["filter_repair"]
    return [{"family": family, "phase": "signal_quality"} for family in families]
