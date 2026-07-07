"""Shared monthly optimization semantics for replay-backed strategy plugins."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from trading_assistant_backtest.auto.types import Candidate, PhaseSpec


DEFAULT_PHASE_FAMILIES = {
    "crypto_portfolio": {
        "signal_quality": ["filter_repair", "momentum_threshold_repair"],
        "trade_management": ["exit_repair", "risk_size_repair", "portfolio_cap_repair"],
    },
    "trading_momentum_family": {
        "signal_quality": ["filter_repair", "regime_filter_repair", "entry_quality_gate"],
        "trade_management": ["exit_repair", "risk_size_repair", "session_repair"],
    },
    "trading_swing_family": {
        "signal_quality": ["filter_repair", "entry_quality_gate", "regime_filter_repair"],
        "trade_management": ["exit_repair", "stop_take_profit_repair", "risk_size_repair"],
    },
    "trading_stock_family": {
        "signal_quality": ["signal_threshold_repair", "filter_repair", "entry_quality_gate"],
        "trade_management": ["exit_repair", "risk_size_repair", "portfolio_cap_repair"],
    },
    "k_stock_olr_kalcb": {
        "signal_quality": ["signal_threshold_repair", "filter_repair", "entry_quality_gate"],
        "trade_management": ["session_repair", "exit_repair", "risk_size_repair"],
    },
}


def build_phase_specs_for_scope(
    *,
    scope_family: str,
    plugin_id: str,
    strategy_id: str,
    diagnostics: Any,
    experiment_plan: Any,
    search_brief: Any,
) -> list[PhaseSpec]:
    phase_families = _phase_families_from_plan(experiment_plan)
    if not phase_families:
        phase_families = DEFAULT_PHASE_FAMILIES.get(scope_family, DEFAULT_PHASE_FAMILIES["crypto_portfolio"])
    phase_order = _phase_order(search_brief, experiment_plan, phase_families)
    metadata = {
        "target_scope": {
            "strategy_plugin_id": plugin_id,
            "strategy_id": strategy_id,
            "scope_family": scope_family,
        },
        "source_evidence_paths": _evidence_paths(experiment_plan),
        "weekly_signal_attribution": _weekly_signal_attribution(search_brief, experiment_plan),
        "candidate_payloads_by_family": {
            family: _family_payload(scope_family, family, diagnostics, search_brief)
            for families in phase_families.values()
            for family in families
        },
    }
    return [
        PhaseSpec(
            phase_id=phase,
            candidate_families=_dedupe(phase_families[phase]),
            metadata=metadata,
        )
        for phase in phase_order
        if phase_families.get(phase)
    ]


def build_repair_candidates_for_scope(
    *,
    scope_family: str,
    plugin_id: str,
    strategy_id: str,
    failure_analysis: Any,
    round_chain: Any,
) -> list[Candidate]:
    analysis = failure_analysis if isinstance(failure_analysis, dict) else {}
    primary = str(analysis.get("primary_failure") or "unknown")
    if primary in {"none", "data_contract"}:
        return []
    families = _repair_families(primary, scope_family)
    candidates: list[Candidate] = []
    for index, family in enumerate(families, start=1):
        candidate_id = f"repair-{_safe(scope_family)}-{_safe(primary)}-{_safe(family)}-{index}"
        family_payload = _family_payload(scope_family, family, analysis, {})
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                family=family,
                payload={
                    "repair_type": "targeted",
                    "primary_failure": primary,
                    "target_scope": {
                        "strategy_plugin_id": plugin_id,
                        "strategy_id": strategy_id,
                        "scope_family": scope_family,
                    },
                    "parameter_patch": family_payload["parameter_patch"],
                    "expected_mechanism": family_payload["expected_mechanism"],
                    "source_evidence_paths": analysis.get("evidence_paths", []),
                    "rollback_plan_ref": f"rollback:{candidate_id}:restore_round_n",
                },
            )
        )
    for mutation in _round_chain_items(round_chain):
        mutation_id = str(mutation.get("mutation_id") or mutation.get("candidate_id") or "")
        if not mutation_id:
            continue
        candidates.append(
            Candidate(
                candidate_id=f"rollback-{_safe(mutation_id)}",
                family="rollback",
                payload={
                    "repair_type": "accepted_mutation_rollback",
                    "mutation_id": mutation_id,
                    "target_scope": {
                        "strategy_plugin_id": plugin_id,
                        "strategy_id": strategy_id,
                        "scope_family": scope_family,
                    },
                    "parameter_patch": {
                        "family": "rollback",
                        "scope_family": scope_family,
                        "rollback_mutation_id": mutation_id,
                        "parameter_diff": mutation.get("parameter_diff", {}),
                    },
                    "source_evidence_paths": mutation.get("original_evidence_paths", []),
                    "rollback_plan_ref": f"rollback:{mutation_id}",
                },
            )
        )
    return _dedupe_candidates(candidates)


def build_confirmatory_variants_for_scope(
    *,
    scope_family: str,
    primary: Candidate,
    context: Any,
) -> list[Candidate]:
    variants = [
        Candidate(
            candidate_id=f"{primary.candidate_id}-confirm",
            family=primary.family,
            payload={
                **primary.payload,
                "variant_type": "primary_confirmatory",
                "source_candidate_id": primary.candidate_id,
                "expected_mechanism": primary.payload.get(
                    "expected_mechanism",
                    f"confirm primary {primary.family} mutation",
                ),
            },
        )
    ]
    for index, direction in enumerate(("loosen", "tighten"), start=1):
        candidate_id = f"{primary.candidate_id}-{direction}-{index}"
        primary_patch = effective_parameter_patch(primary, scope_family=scope_family)
        parameter_patch = {
            **primary_patch,
            "source_candidate_id": primary.candidate_id,
            "direction": direction,
            "local_parameter_delta": (
                patch_number(primary_patch, "local_parameter_delta", 0.0)
                + (-1.0 if direction == "loosen" else 1.0)
            ),
        }
        if "filter_threshold_bps_delta" in primary_patch:
            parameter_patch["filter_threshold_bps_delta"] = (
                patch_number(primary_patch, "filter_threshold_bps_delta")
                + (-2.0 if direction == "loosen" else 2.0)
            )
        if "position_weight_multiplier" in primary_patch:
            parameter_patch["position_weight_multiplier"] = max(
                0.1,
                patch_number(primary_patch, "position_weight_multiplier", 1.0)
                * (1.1 if direction == "loosen" else 0.9),
            )
        variants.append(
            Candidate(
                candidate_id=candidate_id,
                family=f"{primary.family}_{direction}",
                payload={
                    **primary.payload,
                    "variant_type": f"local_parameter_{direction}",
                    "source_candidate_id": primary.candidate_id,
                    "parameter_patch": parameter_patch,
                    "rollback_plan_ref": f"rollback:{candidate_id}:restore_primary_candidate",
                    "confirmatory_context_hash": _stable_hash(context),
                },
            )
        )
    return variants


def round_n_plus_1_payload(candidate: Candidate) -> dict[str, Any]:
    if not str(candidate.payload.get("evaluated_patch_fingerprint") or "").strip():
        raise ValueError(
            "round_N+1 recommendation requires a candidate evaluated with a concrete patch"
        )
    evaluated_patch = _dict_payload(candidate.payload.get("evaluated_parameter_patch"))
    config_patch = evaluated_patch or effective_parameter_patch(candidate)
    applied = _dict_payload(candidate.payload.get("evaluated_parameters"))
    patch_fingerprint = str(
        candidate.payload.get("parameter_patch_fingerprint")
        or patch_fingerprint_for(config_patch)
    )
    evaluated_fingerprint = str(
        candidate.payload.get("evaluated_patch_fingerprint")
        or evaluated_patch_fingerprint(config_patch, applied)
    )
    next_config_hash = hashlib.sha256(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "config_patch": config_patch,
                "evaluated_parameters": applied,
                "evaluated_patch_fingerprint": evaluated_fingerprint,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "round_n_plus_1_recommendation_v1",
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "next_config_hash": next_config_hash,
        "config_patch": config_patch,
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_fingerprint,
        "evaluated_parameters": applied,
        "strategy_patch": candidate.payload.get("structural_patch_ref", ""),
        "adapter_patch": candidate.payload.get("adapter_patch_ref", ""),
        "rollback_plan": {
            "rollback_plan_ref": candidate.payload.get("rollback_plan_ref", ""),
            "restore": "round_N accepted config",
        },
        "candidate_manifest": candidate.payload,
        "live_deployment_status": "optimized_backtest_recommendation",
    }


def effective_parameter_patch(candidate: Candidate, *, scope_family: str = "") -> dict[str, Any]:
    patch = _dict_payload(candidate.payload.get("parameter_patch"))
    if not patch:
        patch = {"family": candidate.family}
    patch.setdefault("family", candidate.family)
    if scope_family:
        patch.setdefault("scope_family", scope_family)
    return _normalize_patch(patch)


def patch_fingerprint_for(patch: dict[str, Any]) -> str:
    return _stable_hash(_normalize_patch(patch))


def evaluated_patch_fingerprint(
    patch: dict[str, Any],
    applied_parameters: dict[str, Any],
) -> str:
    return _stable_hash(
        {
            "parameter_patch": _normalize_patch(patch),
            "evaluated_parameters": _normalize_patch(applied_parameters),
        }
    )


def evaluated_patch_payload(
    candidate: Candidate,
    applied_parameters: dict[str, Any],
    *,
    scope_family: str = "",
) -> dict[str, Any]:
    patch = effective_parameter_patch(candidate, scope_family=scope_family)
    applied = _normalize_patch(applied_parameters)
    patch_fingerprint = patch_fingerprint_for(patch)
    return {
        "parameter_patch": patch,
        "evaluated_parameter_patch": patch,
        "evaluated_parameters": applied,
        "parameter_patch_fingerprint": patch_fingerprint,
        "evaluated_patch_fingerprint": evaluated_patch_fingerprint(patch, applied),
        "evaluated_patch_schema_version": "candidate_evaluated_patch_v1",
    }


def patch_number(patch: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = patch.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def patch_int(patch: dict[str, Any], key: str, default: int = 0) -> int:
    value = patch.get(key, default)
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _phase_families_from_plan(experiment_plan: Any) -> dict[str, list[str]]:
    phase_families: dict[str, list[str]] = {}
    for item in getattr(experiment_plan, "candidate_families", []) or []:
        if isinstance(item, dict):
            family = str(item.get("family") or item.get("candidate_family") or "").strip()
            phase = str(item.get("phase") or "signal_quality").strip() or "signal_quality"
        else:
            family = str(item).strip()
            phase = "signal_quality"
        if family:
            phase_families.setdefault(phase, []).append(family)
    return {phase: _dedupe(families) for phase, families in phase_families.items()}


def _phase_order(search_brief: Any, experiment_plan: Any, phase_families: dict[str, list[str]]) -> list[str]:
    hints: list[str] = []
    if isinstance(search_brief, dict):
        hints.extend(str(item) for item in search_brief.get("phase_order_hints", []) if str(item))
    hints.extend(str(item) for item in getattr(experiment_plan, "phase_order", []) or [] if str(item))
    hints.extend(phase for phase in phase_families if phase not in hints)
    return [phase for phase in _dedupe(hints) if phase in phase_families]


def _evidence_paths(experiment_plan: Any) -> list[str]:
    return [str(path) for path in getattr(experiment_plan, "evidence_paths", []) or [] if str(path)]


def _weekly_signal_attribution(search_brief: Any, experiment_plan: Any) -> list[dict[str, str]]:
    ids = [str(item) for item in getattr(experiment_plan, "source_weekly_signal_ids", []) or []]
    if isinstance(search_brief, dict):
        ids.extend(str(item) for item in search_brief.get("source_weekly_signal_ids", []) or [])
    return [{"source_weekly_signal_id": item} for item in _dedupe(ids) if item]


def _family_payload(scope_family: str, family: str, diagnostics: Any, search_brief: Any) -> dict[str, Any]:
    family_text = family.lower()
    patch: dict[str, Any] = {"family": family, "scope_family": scope_family}
    if "filter" in family_text:
        patch.update({"filter_threshold_bps_delta": -5.0 if "loosen" in family_text else 5.0})
    elif "exit" in family_text or "stop" in family_text:
        patch.update({"exit_threshold_bps_delta": 5.0, "stop_tighten_bps": 10.0})
    elif "risk" in family_text or "size" in family_text:
        patch.update({"position_weight_multiplier": 0.75})
    elif "session" in family_text:
        patch.update({"session_gate": "skip_low_liquidity_tail"})
    elif "regime" in family_text:
        patch.update({"regime_filter": "require_positive_discrimination"})
    else:
        patch.update({"local_parameter_delta": 1})
    return {
        "parameter_patch": patch,
        "expected_mechanism": _mechanism(family, scope_family),
        "diagnostic_snapshot": _diagnostic_snapshot(diagnostics),
    }


def _dict_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _normalize_patch(value: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, dict):
            normalized[str(key)] = _normalize_patch(item)
        elif isinstance(item, list):
            normalized[str(key)] = [
                _normalize_patch(element)
                if isinstance(element, dict)
                else _normalize_scalar(element)
                for element in item
            ]
        else:
            normalized[str(key)] = _normalize_scalar(item)
    return normalized


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 10)
    return str(value)


def _mechanism(family: str, scope_family: str) -> str:
    normalized = family.lower().replace("_", " ")
    if "filter" in normalized:
        return f"{scope_family}: rebalance filter strictness against purged fold signal quality"
    if "entry" in normalized or "signal" in normalized:
        return f"{scope_family}: improve entry discrimination without lowering fold support"
    if "exit" in normalized or "stop" in normalized:
        return f"{scope_family}: reduce adverse exit behavior while preserving expectancy"
    if "risk" in normalized or "size" in normalized:
        return f"{scope_family}: improve risk-adjusted objective through bounded sizing"
    if "session" in normalized:
        return f"{scope_family}: remove weak session exposure identified by replay diagnostics"
    if "regime" in normalized:
        return f"{scope_family}: avoid regimes with poor discrimination or execution quality"
    if "rollback" in normalized:
        return f"{scope_family}: ablate a prior accepted mutation implicated by OOS degradation"
    return f"{scope_family}: bounded monthly optimizer mutation for {normalized}"


def _repair_families(primary_failure: str, scope_family: str) -> list[str]:
    if primary_failure == "missing_replay_evaluator":
        return ["replay_evaluator_enablement"]
    if "under_trading" in primary_failure:
        return ["filter_loosen_repair", "session_repair"]
    if "drawdown" in primary_failure:
        return ["risk_size_repair", "stop_take_profit_repair"]
    if primary_failure == "shadow_replay_candidate_gate":
        return ["filter_repair", "exit_repair"]
    return DEFAULT_PHASE_FAMILIES.get(scope_family, DEFAULT_PHASE_FAMILIES["crypto_portfolio"])[
        "signal_quality"
    ][:2]


def _round_chain_items(round_chain: Any) -> list[dict[str, Any]]:
    if isinstance(round_chain, dict):
        items = round_chain.get("accepted_mutations") or round_chain.get("mutations") or []
    else:
        items = round_chain
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _diagnostic_snapshot(diagnostics: Any) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    keys = ("trade_count", "net_return", "max_drawdown", "profit_factor", "objective_score")
    return {key: diagnostics.get(key) for key in keys if key in diagnostics}


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        result.append(candidate)
    return result


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")[:64] or "candidate"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
