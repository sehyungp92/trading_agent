"""Shared optimizer P6/P7 evidence checks for approval-grade promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trading_assistant_backtest.file_hashes import sha256_file
from trading_assistant_backtest.strategies.plugin_semantics import (
    evaluated_patch_fingerprint,
    patch_fingerprint_for,
)
from trading_assistant_backtest.paths import normalize_workspace_path, resolve_workspace_path

OPTIMIZER_RUN_MANIFEST = "optimizer_run_manifest.json"
OPTIMIZER_RUN_MANIFEST_VERSION = "optimizer_approval_run_manifest_v1"
FORBIDDEN_OPTIMIZER_ROOT_PARTS = {"monthly_smoke"}
FORBIDDEN_OPTIMIZER_MODE_TOKENS = {"smoke", "monthly_smoke", "smoke_repair"}
APPROVAL_OPTIMIZER_ROOTS = (
    Path("trading_assistant_backtest/artifacts/validation/optimizer"),
    Path("trading_assistant_backtest/artifacts/validation/approval_grade/optimizer"),
    Path("trading_assistant_backtest/artifacts/monthly/approval_grade"),
)


def optimizer_evidence_checks(
    scope_id: str,
    agent_root: Path,
    *,
    expected_context: dict[str, Any] | None = None,
    manifest_index: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    root = latest_optimizer_artifact_root(
        scope_id,
        agent_root,
        expected_context=expected_context,
        manifest_index=manifest_index,
    )
    if root is None:
        missing = [
            "missing explicit approval-grade optimizer_run_manifest.json for promoted scope"
        ]
        return [
            _check(
                f"{scope_id}:optimizer_p6_true_fold_scoring_complete",
                False,
                missing,
            ),
            _check(
                f"{scope_id}:optimizer_p7_repair_confirmatory_round_complete",
                False,
                missing,
            ),
        ]
    manifest_errors = optimizer_run_manifest_errors(
        root,
        scope_id,
        agent_root=agent_root,
        expected_context=expected_context,
    )
    p6_errors = [*manifest_errors, *p6_optimizer_errors(root)]
    p7_errors = [*manifest_errors, *p7_optimizer_errors(root)]
    return [
        _check(
            f"{scope_id}:optimizer_p6_true_fold_scoring_complete",
            not p6_errors,
            p6_errors,
        ),
        _check(
            f"{scope_id}:optimizer_p7_repair_confirmatory_round_complete",
            not p7_errors,
            p7_errors,
        ),
    ]


def optimizer_readiness_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    optimizer = [check for check in checks if ":optimizer_p" in check.get("name", "")]
    return {
        "ready": bool(optimizer) and all(check["passed"] for check in optimizer),
        "checks": optimizer,
    }


def build_optimizer_manifest_index(agent_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Index explicit approval optimizer manifest locations once per audit run."""

    agent_root = Path(agent_root).resolve()
    index: dict[str, list[dict[str, Any]]] = {}
    for relative_root in APPROVAL_OPTIMIZER_ROOTS:
        root = resolve_workspace_path(agent_root, relative_root)
        if not root.exists():
            continue
        for manifest_path in root.rglob(OPTIMIZER_RUN_MANIFEST):
            manifest = _read_json(manifest_path)
            if not manifest:
                continue
            scope_ids = _manifest_scope_ids(manifest)
            if not scope_ids:
                continue
            entry = {
                "root": manifest_path.parent,
                "manifest_path": manifest_path,
                "manifest": manifest,
                "mtime": manifest_path.stat().st_mtime,
            }
            for scope_id in scope_ids:
                index.setdefault(scope_id, []).append(entry)
    return index


def latest_optimizer_artifact_root(
    scope_id: str,
    agent_root: Path,
    *,
    expected_context: dict[str, Any] | None = None,
    manifest_index: dict[str, list[dict[str, Any]]] | None = None,
) -> Path | None:
    index = manifest_index if manifest_index is not None else build_optimizer_manifest_index(agent_root)
    entries = list(index.get(scope_id, []))
    if not entries:
        return None
    context_matches = [
        entry for entry in entries
        if not _context_errors(entry["manifest"], expected_context)
    ]
    if context_matches:
        entries = context_matches
    latest = max(entries, key=lambda entry: entry["mtime"])
    return Path(latest["root"])


def optimizer_run_manifest_errors(
    root: Path,
    scope_id: str,
    *,
    agent_root: Path,
    expected_context: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    manifest_path = root / OPTIMIZER_RUN_MANIFEST
    manifest = _read_json(manifest_path)
    if not manifest:
        return [f"{OPTIMIZER_RUN_MANIFEST} missing or invalid"]
    if manifest.get("schema_version") != OPTIMIZER_RUN_MANIFEST_VERSION:
        errors.append(
            f"{OPTIMIZER_RUN_MANIFEST} schema_version must be {OPTIMIZER_RUN_MANIFEST_VERSION}"
        )
    if not _manifest_scope_matches(manifest, scope_id):
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} scope_id does not match {scope_id}")
    if manifest.get("approval_grade_optimizer_run") is not True:
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} must mark approval_grade_optimizer_run=true")
    if manifest.get("smoke_mode") is not False:
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} must mark smoke_mode=false")
    errors.extend(_context_errors(manifest, expected_context))
    mode_tokens = {
        str(manifest.get(key) or "").strip().lower()
        for key in ("run_mode", "optimizer_mode", "artifact_mode")
    }
    forbidden_modes = sorted(mode_tokens & FORBIDDEN_OPTIMIZER_MODE_TOKENS)
    if forbidden_modes:
        errors.append(
            f"{OPTIMIZER_RUN_MANIFEST} contains smoke optimizer mode(s): "
            + ", ".join(forbidden_modes)
        )
    forbidden_root_parts = sorted(
        part for part in root.parts if part.lower() in FORBIDDEN_OPTIMIZER_ROOT_PARTS
    )
    if forbidden_root_parts:
        errors.append(
            f"{OPTIMIZER_RUN_MANIFEST} must not live under smoke artifact root(s): "
            + ", ".join(forbidden_root_parts)
        )
    artifact_root = str(manifest.get("artifact_root") or "").strip()
    if artifact_root:
        try:
            if normalize_workspace_path(agent_root, artifact_root).resolve() != root.resolve():
                errors.append(f"{OPTIMIZER_RUN_MANIFEST} artifact_root does not match selected root")
        except OSError:
            errors.append(f"{OPTIMIZER_RUN_MANIFEST} artifact_root is not resolvable")
    else:
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} missing artifact_root")
    required = (
        "run_id",
        "run_month",
        "data_bundle_checksum",
        "strategy_plugin_contract_hash",
        "deployment_metadata_hash",
        "run_manifest_hash",
        "strategy_plugin_contract_path",
        "deployment_metadata_path",
        "run_manifest_path",
    )
    missing = [field for field in required if not str(manifest.get(field) or "").strip()]
    if missing:
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} missing required fields: " + ", ".join(missing))
    _append_hash_path_errors(
        manifest,
        "strategy_plugin_contract_path",
        "strategy_plugin_contract_hash",
        errors,
        agent_root,
    )
    _append_hash_path_errors(
        manifest,
        "deployment_metadata_path",
        "deployment_metadata_hash",
        errors,
        agent_root,
    )
    _append_hash_path_errors(
        manifest,
        "run_manifest_path",
        "run_manifest_hash",
        errors,
        agent_root,
    )
    run_manifest_path = normalize_workspace_path(
        agent_root,
        str(manifest.get("run_manifest_path") or ""),
    )
    run_manifest = _read_json(run_manifest_path)
    if run_manifest:
        for field in ("run_id", "run_month", "bot_id", "strategy_id"):
            expected = str(manifest.get(field) or "").strip()
            actual = str(run_manifest.get(field) or "").strip()
            if expected and actual and expected != actual:
                errors.append(f"{OPTIMIZER_RUN_MANIFEST} {field} does not match run_manifest.json")
        manifest_bundle = str(manifest.get("data_bundle_checksum") or "").strip()
        run_bundle = str(
            run_manifest.get("data_bundle_checksum")
            or run_manifest.get("data_manifest_checksum")
            or ""
        ).strip()
        if manifest_bundle and run_bundle and manifest_bundle != run_bundle:
            errors.append(
                f"{OPTIMIZER_RUN_MANIFEST} data_bundle_checksum does not match run_manifest.json"
            )
    return errors


def p6_optimizer_errors(root: Path) -> list[str]:
    errors: list[str] = []
    fold_manifest = _read_json(root / "fold_manifest.json")
    fold_score = _read_json(root / "fold_score_matrix.json")
    fold_candidate_results = root / "fold_candidate_results.jsonl"
    if not fold_manifest:
        errors.append("fold_manifest.json missing or invalid")
    elif len(fold_manifest.get("folds") or []) != 2:
        errors.append("fold_manifest.json must contain exactly two folds")
    elif not all((fold or {}).get("purged") is True for fold in fold_manifest.get("folds", [])):
        errors.append("fold_manifest.json folds must be purged")
    if not fold_score:
        errors.append("fold_score_matrix.json missing or invalid")
    else:
        if fold_score.get("selection_oos_excluded_from_first_pass") is not True:
            errors.append("fold_score_matrix must prove selection-OOS exclusion")
        if len(fold_score.get("scoring_windows") or []) != 2:
            errors.append("fold_score_matrix must include two scoring windows")
        if int(fold_score.get("candidate_count") or 0) <= 0:
            errors.append("fold_score_matrix must include at least one scored candidate")
        candidates = fold_score.get("candidates") or []
        if isinstance(candidates, list) and candidates:
            if int(fold_score.get("candidate_count") or 0) != len(candidates):
                errors.append("fold_score_matrix candidate_count must match candidates length")
    if not fold_candidate_results.exists():
        errors.append("fold_candidate_results.jsonl missing")
    elif not fold_candidate_results.read_text(encoding="utf-8").strip():
        errors.append("fold_candidate_results.jsonl is empty")
    else:
        rows = _read_jsonl(fold_candidate_results)
        fold_ids = {str(row.get("fold_id") or "") for row in rows if row.get("fold_id")}
        if len(fold_ids) < 2:
            errors.append("fold_candidate_results.jsonl must cover both purged folds")
        if any(row.get("selection_oos_used_in_first_pass") is True for row in rows):
            errors.append("fold_candidate_results.jsonl must exclude selection-OOS first pass")
        missing_replay_rows = [
            row for row in rows if not isinstance(row.get("candidate"), dict)
        ]
        if missing_replay_rows:
            errors.append("fold_candidate_results.jsonl rows must include candidate replay summaries")
        elif any(
            not str((row.get("candidate") or {}).get("evaluated_patch_fingerprint") or "")
            for row in rows
        ):
            errors.append(
                "fold_candidate_results.jsonl candidate replays must include evaluated_patch_fingerprint"
            )
        elif any(
            not str((row.get("candidate") or {}).get("parameter_patch_fingerprint") or "")
            for row in rows
        ):
            errors.append(
                "fold_candidate_results.jsonl candidate replays must include parameter_patch_fingerprint"
            )
        errors.extend(_fold_patch_fingerprint_errors(rows))
    return errors


def p7_optimizer_errors(root: Path) -> list[str]:
    errors: list[str] = []
    selection = _read_json(root / "selection_oos_evaluation.json")
    trigger = _read_json(root / "selection_oos_repair_trigger.json")
    confirmatory = _read_json(root / "confirmatory_rerank.json")
    rounds = _read_json(root / "rounds_manifest.json")
    recommendation = _read_json(root / "round_n_plus_1_recommendation.json")
    failure = _read_json(root / "repair_failure_attribution.json")
    accepted_chain = _read_json(root / "accepted_mutation_chain.json")
    repair_checkpoint = _read_json(root / "repair_checkpoint.json")
    repair_results = root / "repair_candidate_results.jsonl"
    selected_candidates = _read_json_list(root / "selected_candidates.json")
    if not selection:
        errors.append("selection_oos_evaluation.json missing or invalid")
    elif selection.get("selection_oos_used_after_fold_ranking") is not True:
        errors.append("selection_oos_evaluation must run after fold ranking")
    elif selection.get("selection_oos_used_in_first_pass") is True:
        errors.append("selection_oos_evaluation must not mark selection-OOS as first pass")
    if not trigger:
        errors.append("selection_oos_repair_trigger.json missing or invalid")
    else:
        if trigger.get("status") not in {"triggered", "not_triggered"}:
            errors.append("selection_oos_repair_trigger status must be deterministic")
        if not isinstance(trigger.get("thresholds"), dict):
            errors.append("selection_oos_repair_trigger thresholds missing")
        if not isinstance(trigger.get("measured_degradation"), dict):
            errors.append("selection_oos_repair_trigger measured_degradation missing")
    if not failure:
        errors.append("repair_failure_attribution.json missing or invalid")
    if not accepted_chain:
        errors.append("accepted_mutation_chain.json missing or invalid")
    elif not isinstance(accepted_chain.get("accepted_mutations", []), list):
        errors.append("accepted_mutation_chain accepted_mutations must be a list")
    if not repair_checkpoint:
        errors.append("repair_checkpoint.json missing or invalid")
    elif bool(repair_checkpoint.get("repair_triggered")) != bool(trigger.get("triggered")):
        errors.append("repair_checkpoint repair_triggered must match trigger status")
    if not repair_results.exists():
        errors.append("repair_candidate_results.jsonl missing")
    elif bool(trigger.get("triggered")) and not repair_results.read_text(encoding="utf-8").strip():
        errors.append("triggered repair requires repair_candidate_results.jsonl rows")
    if not confirmatory:
        errors.append("confirmatory_rerank.json missing or invalid")
    else:
        has_primary = bool(confirmatory.get("primary_candidate_id"))
        has_variants = bool(confirmatory.get("variants"))
        has_no_adoption = bool(confirmatory.get("no_adoption_reason"))
        if has_primary and not has_variants:
            errors.append("confirmatory variants are empty despite a primary candidate")
        if not has_primary and not has_no_adoption:
            errors.append("empty confirmatory variants require a deterministic no-adoption reason")
        if bool(confirmatory.get("repair_triggered")) != bool(trigger.get("triggered")):
            errors.append("confirmatory repair_triggered must match measured trigger")
    if not rounds:
        errors.append("rounds_manifest.json missing or invalid")
    else:
        adopted = bool(rounds.get("adopted_candidate_id"))
        no_adoption = bool(rounds.get("no_adoption_reason"))
        if adopted == no_adoption:
            errors.append("rounds_manifest must have exactly one adoption or no-adoption reason")
    if not recommendation:
        errors.append("round_n_plus_1_recommendation.json missing or invalid")
    else:
        status = recommendation.get("status")
        if status not in {"optimized_backtest_recommendation", "no_adoption"}:
            errors.append("round_n_plus_1_recommendation status is invalid")
        if status == "optimized_backtest_recommendation":
            adopted_id = str(recommendation.get("adopted_candidate_id") or "")
            if not adopted_id:
                errors.append("round_n_plus_1_recommendation adopted_candidate_id missing")
            if adopted_id and adopted_id != str(confirmatory.get("adopted_candidate_id") or ""):
                errors.append("round_n_plus_1_recommendation does not match confirmatory adoption")
            if adopted_id and adopted_id != str(rounds.get("adopted_candidate_id") or ""):
                errors.append("round_n_plus_1_recommendation does not match rounds manifest")
            selected = [
                row
                for row in selected_candidates
                if str(row.get("candidate_id") or "") == adopted_id
            ]
            if len(selected) != 1:
                errors.append("optimized recommendation requires exactly one selected candidate row")
            else:
                selected_row = selected[0]
                adopted_source = str(confirmatory.get("adopted_source") or "")
                if adopted_source and adopted_source != str(selected_row.get("source") or ""):
                    errors.append("confirmatory adopted_source does not match selected candidate source")
                round_source = _round_adopted_source(rounds, adopted_id)
                if round_source and round_source != str(selected_row.get("source") or ""):
                    errors.append("rounds manifest source does not match selected candidate source")
                selected_fingerprint = _candidate_evaluated_patch_fingerprint(selected_row)
                recommendation_fingerprint = str(
                    recommendation.get("evaluated_patch_fingerprint") or ""
                )
                selected_parameter_fingerprint = _candidate_parameter_patch_fingerprint(
                    selected_row
                )
                recommendation_parameter_fingerprint = str(
                    recommendation.get("parameter_patch_fingerprint") or ""
                )
                if not selected_fingerprint:
                    errors.append("selected candidate missing evaluated_patch_fingerprint")
                if not recommendation_fingerprint:
                    errors.append("round_n_plus_1_recommendation missing evaluated_patch_fingerprint")
                if (
                    selected_fingerprint
                    and recommendation_fingerprint
                    and selected_fingerprint != recommendation_fingerprint
                ):
                    errors.append(
                        "round_n_plus_1_recommendation patch fingerprint does not match selected candidate"
                    )
                if not selected_parameter_fingerprint:
                    errors.append("selected candidate missing parameter_patch_fingerprint")
                if not recommendation_parameter_fingerprint:
                    errors.append("round_n_plus_1_recommendation missing parameter_patch_fingerprint")
                if (
                    selected_parameter_fingerprint
                    and recommendation_parameter_fingerprint
                    and selected_parameter_fingerprint != recommendation_parameter_fingerprint
                ):
                    errors.append(
                        "round_n_plus_1_recommendation parameter patch fingerprint does not match selected candidate"
                    )
                _append_selected_patch_errors(selected_row, errors)
                _append_config_patch_errors(recommendation, errors)
        if status == "no_adoption" and not str(recommendation.get("no_adoption_reason") or ""):
            errors.append("round_n_plus_1_recommendation no-adoption reason missing")
    return errors


def _check(name: str, passed: bool, errors: list[str]) -> dict[str, Any]:
    return {"name": name, "passed": passed, "errors": errors}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _candidate_evaluated_patch_fingerprint(row: dict[str, Any]) -> str:
    direct = str(row.get("evaluated_patch_fingerprint") or "")
    if direct:
        return direct
    raw = row.get("raw_payload")
    if not isinstance(raw, dict):
        return ""
    candidate_payload = raw.get("candidate_payload")
    if isinstance(candidate_payload, dict):
        return str(candidate_payload.get("evaluated_patch_fingerprint") or "")
    return str(raw.get("evaluated_patch_fingerprint") or "")


def _candidate_parameter_patch_fingerprint(row: dict[str, Any]) -> str:
    direct = str(row.get("parameter_patch_fingerprint") or "")
    if direct:
        return direct
    raw = row.get("raw_payload")
    if not isinstance(raw, dict):
        return ""
    candidate_payload = raw.get("candidate_payload")
    if isinstance(candidate_payload, dict):
        return str(candidate_payload.get("parameter_patch_fingerprint") or "")
    return str(raw.get("parameter_patch_fingerprint") or "")


def _candidate_parameter_patch(row: dict[str, Any]) -> dict[str, Any]:
    return _candidate_payload_dict(
        row,
        (
            "evaluated_parameter_patch",
            "parameter_patch",
            "config_patch",
        ),
    )


def _candidate_evaluated_parameters(row: dict[str, Any]) -> dict[str, Any]:
    return _candidate_payload_dict(row, ("evaluated_parameters",))


def _candidate_payload_dict(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, dict):
            return value
    raw = row.get("raw_payload")
    if isinstance(raw, dict):
        for key in keys:
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        candidate_payload = raw.get("candidate_payload")
        if isinstance(candidate_payload, dict):
            for key in keys:
                value = candidate_payload.get(key)
                if isinstance(value, dict):
                    return value
    return {}


def _round_adopted_source(rounds: dict[str, Any], adopted_id: str) -> str:
    for record in rounds.get("records") or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("candidate_id") or "") == adopted_id:
            return str(record.get("source") or "")
    return ""


def _append_config_patch_errors(
    recommendation: dict[str, Any],
    errors: list[str],
) -> None:
    path_text = str(recommendation.get("config_patch_path") or "")
    if not path_text:
        errors.append("round_n_plus_1_recommendation missing config_patch_path")
        return
    path = Path(path_text)
    if not path.exists():
        errors.append("round_n_plus_1_recommendation config_patch_path does not exist")
        return
    patch = _read_json(path)
    if not patch:
        errors.append("round_n_plus_1 config patch is missing or invalid")
        return
    patch_fingerprint = patch_fingerprint_for(patch)
    expected = str(recommendation.get("parameter_patch_fingerprint") or "")
    if not expected:
        errors.append("round_n_plus_1_recommendation missing parameter_patch_fingerprint")
    elif patch_fingerprint != expected:
        errors.append("round_n_plus_1 config patch fingerprint does not match recommendation")
    evaluated_parameters = recommendation.get("evaluated_parameters")
    if not isinstance(evaluated_parameters, dict) or not evaluated_parameters:
        errors.append("round_n_plus_1_recommendation missing evaluated_parameters")
        return
    expected_evaluated = str(recommendation.get("evaluated_patch_fingerprint") or "")
    recomputed_evaluated = evaluated_patch_fingerprint(patch, evaluated_parameters)
    if expected_evaluated and recomputed_evaluated != expected_evaluated:
        errors.append(
            "round_n_plus_1 evaluated patch fingerprint does not match config patch and evaluated parameters"
        )


def _append_selected_patch_errors(row: dict[str, Any], errors: list[str]) -> None:
    patch = _candidate_parameter_patch(row)
    evaluated_parameters = _candidate_evaluated_parameters(row)
    parameter_fingerprint = _candidate_parameter_patch_fingerprint(row)
    evaluated_fingerprint = _candidate_evaluated_patch_fingerprint(row)
    if not patch:
        errors.append("selected candidate missing concrete parameter_patch")
        return
    recomputed_parameter = patch_fingerprint_for(patch)
    if parameter_fingerprint and parameter_fingerprint != recomputed_parameter:
        errors.append("selected candidate parameter patch fingerprint is not canonical")
    if not evaluated_parameters:
        errors.append("selected candidate missing evaluated_parameters")
        return
    recomputed_evaluated = evaluated_patch_fingerprint(patch, evaluated_parameters)
    if evaluated_fingerprint and evaluated_fingerprint != recomputed_evaluated:
        errors.append("selected candidate evaluated patch fingerprint is not canonical")


def _stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str)
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _manifest_scope_matches(manifest: dict[str, Any], scope_id: str) -> bool:
    return scope_id in _manifest_scope_ids(manifest)


def _manifest_scope_ids(manifest: dict[str, Any]) -> set[str]:
    if not manifest:
        return set()
    aliases = {
        str(manifest.get("scope_id") or "").strip(),
        str(manifest.get("portfolio_id") or "").strip(),
        str(manifest.get("bot_id") or "").strip(),
        str(manifest.get("strategy_id") or "").strip(),
        str(manifest.get("strategy_plugin_id") or "").strip(),
    }
    scope_aliases = manifest.get("scope_aliases") or []
    if isinstance(scope_aliases, list):
        aliases.update(str(alias).strip() for alias in scope_aliases)
    return {alias for alias in aliases if alias}


def _context_errors(
    manifest: dict[str, Any],
    expected_context: dict[str, Any] | None,
) -> list[str]:
    if not expected_context:
        return []
    errors: list[str] = []
    expected_run_month = str(expected_context.get("run_month") or "").strip()
    if expected_run_month:
        actual_run_month = str(manifest.get("run_month") or "").strip()
        if actual_run_month != expected_run_month:
            errors.append(
                f"{OPTIMIZER_RUN_MANIFEST} run_month {actual_run_month!r} "
                f"does not match promoted validation run_month {expected_run_month!r}"
            )

    expected_checksums = _string_set(expected_context.get("data_bundle_checksums"))
    if expected_checksums:
        actual_checksums = _manifest_data_bundle_checksums(manifest)
        if actual_checksums != expected_checksums:
            errors.append(
                f"{OPTIMIZER_RUN_MANIFEST} data bundle checksum set "
                f"{sorted(actual_checksums)} does not match promoted validation "
                f"checksum set {sorted(expected_checksums)}"
            )

    expected_contract_hashes = _string_map(expected_context.get("bridge_contract_hashes"))
    if expected_contract_hashes:
        actual_contract_hashes = _manifest_bridge_hashes(
            manifest,
            plural_keys=("bridge_contract_hashes", "strategy_plugin_contract_hashes"),
            single_key="strategy_plugin_contract_hash",
        )
        errors.extend(
            _hash_map_errors(
                "strategy contract",
                actual_contract_hashes,
                expected_contract_hashes,
            )
        )

    expected_metadata_hashes = _string_map(expected_context.get("deployment_metadata_hashes"))
    if expected_metadata_hashes:
        actual_metadata_hashes = _manifest_bridge_hashes(
            manifest,
            plural_keys=("bridge_deployment_metadata_hashes", "deployment_metadata_hashes"),
            single_key="deployment_metadata_hash",
        )
        errors.extend(
            _hash_map_errors(
                "deployment metadata",
                actual_metadata_hashes,
                expected_metadata_hashes,
            )
        )
    return errors


def _manifest_data_bundle_checksums(manifest: dict[str, Any]) -> set[str]:
    values = _string_set(manifest.get("data_bundle_checksums"))
    single = str(manifest.get("data_bundle_checksum") or "").strip()
    if single:
        values.add(single)
    return values


def _manifest_bridge_hashes(
    manifest: dict[str, Any],
    *,
    plural_keys: tuple[str, ...],
    single_key: str,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in plural_keys:
        value = manifest.get(key)
        if isinstance(value, dict):
            hashes.update(
                {
                    str(item_key): str(item_value).strip()
                    for item_key, item_value in value.items()
                    if str(item_key).strip() and str(item_value).strip()
                }
            )
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    bridge_id = str(
                        item.get("bridge_id")
                        or item.get("scope_id")
                        or item.get("id")
                        or ""
                    ).strip()
                    digest = str(item.get("hash") or item.get("sha256") or "").strip()
                    if bridge_id and digest:
                        hashes[bridge_id] = digest
    single = str(manifest.get(single_key) or "").strip()
    if single:
        for alias in _manifest_scope_ids(manifest):
            hashes.setdefault(alias, single)
    return hashes


def _hash_map_errors(
    label: str,
    actual: dict[str, str],
    expected: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    missing = sorted(bridge_id for bridge_id in expected if bridge_id not in actual)
    if missing:
        errors.append(
            f"{OPTIMIZER_RUN_MANIFEST} missing promoted {label} hash(es) for: "
            + ", ".join(missing)
        )
    mismatched = sorted(
        bridge_id
        for bridge_id, expected_hash in expected.items()
        if bridge_id in actual and actual[bridge_id] != expected_hash
    )
    if mismatched:
        errors.append(
            f"{OPTIMIZER_RUN_MANIFEST} promoted {label} hash mismatch for: "
            + ", ".join(mismatched)
        )
    return errors


def _string_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    item = str(value or "").strip()
    return {item} if item else set()


def _string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def _append_hash_path_errors(
    manifest: dict[str, Any],
    path_key: str,
    hash_key: str,
    errors: list[str],
    agent_root: Path,
) -> None:
    path_text = str(manifest.get(path_key) or "").strip()
    expected = str(manifest.get(hash_key) or "").strip()
    if not path_text or not expected:
        return
    path = normalize_workspace_path(agent_root, path_text)
    if not path.exists() or not path.is_file():
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} {path_key} does not exist")
        return
    digest = _file_sha256(path)
    if digest != expected:
        errors.append(f"{OPTIMIZER_RUN_MANIFEST} {hash_key} does not match {path_key}")


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _fold_patch_fingerprint_errors(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_candidate: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        candidate = row.get("candidate")
        if not candidate_id or not isinstance(candidate, dict):
            continue
        patch = _candidate_parameter_patch(candidate)
        evaluated_parameters = _candidate_evaluated_parameters(candidate)
        parameter_fingerprint = _candidate_parameter_patch_fingerprint(candidate)
        evaluated_fingerprint = _candidate_evaluated_patch_fingerprint(candidate)
        fingerprints = by_candidate.setdefault(
            candidate_id,
            {"parameter": set(), "evaluated": set()},
        )
        if parameter_fingerprint:
            fingerprints["parameter"].add(parameter_fingerprint)
        if evaluated_fingerprint:
            fingerprints["evaluated"].add(evaluated_fingerprint)
        if not patch:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} missing concrete parameter_patch"
            )
            continue
        recomputed_parameter = patch_fingerprint_for(patch)
        if parameter_fingerprint and parameter_fingerprint != recomputed_parameter:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} parameter patch fingerprint is not canonical"
            )
        if not evaluated_parameters:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} missing evaluated_parameters"
            )
            continue
        recomputed_evaluated = evaluated_patch_fingerprint(patch, evaluated_parameters)
        if evaluated_fingerprint and evaluated_fingerprint != recomputed_evaluated:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} evaluated patch fingerprint is not canonical"
            )
    for candidate_id, fingerprints in by_candidate.items():
        if len(fingerprints["parameter"]) > 1:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} has inconsistent parameter_patch_fingerprint across folds"
            )
        if len(fingerprints["evaluated"]) > 1:
            errors.append(
                f"fold_candidate_results.jsonl candidate {candidate_id} has inconsistent evaluated_patch_fingerprint across folds"
            )
    return list(dict.fromkeys(errors))
