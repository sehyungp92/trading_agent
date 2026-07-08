from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .phase_state import _atomic_write_json, _utc_now_iso, load_phase_state
from .provenance import (
    AutoRunProvenance,
    ProvenanceValidationError,
    ProvenanceValidationResult,
    coerce_provenance,
    diff_provenance_items,
)

_PHASE_OUTPUT_RE = re.compile(r"^phase_\d+.*\.(?:json|txt|log)$")
_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")

_CANONICAL_METRIC_KEYS = {
    "total_trades": ("total_trades", "trades"),
    "win_rate": ("win_rate",),
    "profit_factor": ("profit_factor", "pf"),
    "max_drawdown_pct": ("max_drawdown_pct", "max_dd_pct"),
    "net_return_pct": ("net_return_pct", "return_pct"),
    "sharpe_ratio": ("sharpe_ratio", "sharpe"),
    "calmar_ratio": ("calmar_ratio", "calmar", "calmar_r"),
}

_PERCENT_RATIO_METRICS = {"max_drawdown_pct", "net_return_pct", "win_rate"}
_BOOTSTRAP_EXTRA_FILES = {
    "phase_activity_log.jsonl",
    "phase_run_manifest.json",
    "progress.json",
    "round_evaluation.txt",
}

RUN_SPEC_FILENAME = "run_spec.json"
RUN_SUMMARY_FILENAME = "run_summary.json"
OPTIMIZED_CONFIG_FILENAME = "optimized_config.json"
ROUND_DIAGNOSTICS_FILENAME = "round_final_diagnostics.txt"
ROUND_EVALUATION_FILENAME = "round_evaluation.txt"
PHASE_STATE_FILENAME = "phase_state.json"
DIAGNOSTICS_SUMMARY_FILENAME = "diagnostics_summary.json"
MANIFEST_FILENAME = "rounds_manifest.json"


def _coerce_number(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return value
    if number.is_integer():
        return int(number)
    return number


def _coerce_percent(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return number


def _first_metric(metrics: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def canonicalize_metrics(final_metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = final_metrics or {}
    canonical: dict[str, Any] = {}
    for key, aliases in _CANONICAL_METRIC_KEYS.items():
        value = _first_metric(metrics, aliases)
        if key in _PERCENT_RATIO_METRICS:
            canonical[key] = _coerce_percent(value)
        else:
            canonical[key] = _coerce_number(value)
    return canonical


class RoundManager:
    """Manages centralized round directories for one strategy."""

    def __init__(
        self,
        family: str,
        strategy: str,
        base_dir: Path | None = None,
    ):
        self.family = family
        self.strategy = strategy
        self.base_dir = Path(base_dir) if base_dir is not None else self.default_base_dir()
        self.strategy_dir = self.base_dir / family / strategy
        self.manifest_path = self.strategy_dir / MANIFEST_FILENAME

    @staticmethod
    def default_base_dir() -> Path:
        return Path(__file__).resolve().parents[2] / "output"

    def round_path(self, round_num: int) -> Path:
        return self.strategy_dir / f"round_{round_num}"

    def get_round_dir(self, round_num: int) -> Path:
        round_dir = self.round_path(round_num)
        round_dir.mkdir(parents=True, exist_ok=True)
        return round_dir

    def run_spec_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / RUN_SPEC_FILENAME

    def run_summary_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / RUN_SUMMARY_FILENAME

    def optimized_config_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / OPTIMIZED_CONFIG_FILENAME

    def diagnostics_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_DIAGNOSTICS_FILENAME

    def diagnostics_summary_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / DIAGNOSTICS_SUMMARY_FILENAME

    def evaluation_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / ROUND_EVALUATION_FILENAME

    def phase_state_path(self, round_dir: Path) -> Path:
        return Path(round_dir) / PHASE_STATE_FILENAME

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {
                "family": self.family,
                "strategy": self.strategy,
                "rounds": [],
            }
        data = json.loads(self.manifest_path.read_text(encoding="utf-8-sig"))
        data.setdefault("family", self.family)
        data.setdefault("strategy", self.strategy)
        data.setdefault("rounds", [])
        return data

    def get_latest_round(self) -> int:
        rounds = self.load_manifest().get("rounds", [])
        latest = 0
        for entry in rounds:
            if entry.get("archived"):
                continue
            try:
                latest = max(latest, int(entry.get("round", 0)))
            except (TypeError, ValueError):
                continue
        return latest

    def resolve_round(
        self,
        requested_round: int | None,
        *,
        for_write: bool,
        expected_phases: int | None = None,
    ) -> tuple[int, Path]:
        if requested_round is not None and requested_round < 1:
            raise ValueError(f"Round numbers must be positive integers, got {requested_round}.")

        latest = self.get_latest_round()
        if requested_round is not None:
            round_dir = self.round_path(requested_round)
            if not for_write and not round_dir.exists():
                raise FileNotFoundError(f"Round {requested_round} does not exist at {round_dir}.")
            if not for_write:
                return requested_round, round_dir

            if requested_round > latest + 1:
                raise ValueError(
                    f"Cannot create round {requested_round} for {self.family}/{self.strategy}: "
                    f"latest recorded round is {latest}, so the next writable round is {latest + 1}."
                )

            if requested_round <= latest:
                if not round_dir.exists():
                    raise FileNotFoundError(
                        f"Round {requested_round} is recorded in {self.manifest_path} "
                        f"but its directory is missing: {round_dir}"
                    )
                if self._round_is_in_progress(requested_round, expected_phases):
                    return requested_round, self.get_round_dir(requested_round)
                raise FileExistsError(
                    f"Round {requested_round} for {self.family}/{self.strategy} is already complete. "
                    f"Use round {latest + 1} for the next optimization pass."
                )

            return requested_round, self.get_round_dir(requested_round)
        if not for_write:
            if latest == 0:
                raise FileNotFoundError(f"No rounds exist yet under {self.strategy_dir}.")
            round_dir = self.round_path(latest)
            if not round_dir.exists():
                raise FileNotFoundError(
                    f"Latest recorded round {latest} is missing its directory under {self.strategy_dir}."
                )
            return latest, round_dir

        if latest == 0:
            return 1, self.get_round_dir(1)
        latest_dir = self.round_path(latest)
        if not latest_dir.exists():
            raise FileNotFoundError(
                f"Latest recorded round {latest} is missing its directory under {self.strategy_dir}."
            )
        if self._round_is_in_progress(latest, expected_phases):
            return latest, self.get_round_dir(latest)
        return latest + 1, self.get_round_dir(latest + 1)

    def get_previous_mutations(
        self,
        current_round: int | None = None,
        *,
        current_provenance: AutoRunProvenance | dict[str, Any] | None = None,
        allow_diagnostics_only_drift: bool = True,
    ) -> dict[str, Any]:
        previous_round = self.get_latest_round() if current_round is None else current_round - 1
        if previous_round < 1:
            raise FileNotFoundError(f"No previous round exists for {self.family}/{self.strategy}.")
        if current_provenance is not None:
            validation_round = current_round if current_round is not None else previous_round + 1
            result = self.validate_previous_round_provenance(
                validation_round,
                current_provenance,
                allow_diagnostics_only_drift=allow_diagnostics_only_drift,
            )
            if not result.valid:
                raise ProvenanceValidationError(result)

        config_path = self.optimized_config_path(self.round_path(previous_round))
        if not config_path.exists():
            raise FileNotFoundError(f"Missing optimized config for round {previous_round}: {config_path}")

        data = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if isinstance(data.get("mutations"), dict):
                return dict(data["mutations"])
            if isinstance(data.get("cumulative_mutations"), dict):
                return dict(data["cumulative_mutations"])
            return dict(data)
        raise TypeError(f"Unexpected optimized config payload in {config_path}")

    def validate_previous_round_provenance(
        self,
        current_round: int,
        current: AutoRunProvenance | dict[str, Any],
        *,
        allow_diagnostics_only_drift: bool = True,
    ) -> ProvenanceValidationResult:
        current_provenance = coerce_provenance(current)
        if current_provenance is None:
            raise ValueError("Current provenance is required for prior-round validation.")

        previous_round = current_round - 1
        if previous_round < 1:
            return ProvenanceValidationResult(
                valid=True,
                status="no_previous_round",
                previous_round=None,
                current_round=current_round,
                message=f"Round {current_round} has no prior round to validate.",
            )

        previous = self.load_round_provenance(previous_round)
        if previous is None:
            return ProvenanceValidationResult(
                valid=False,
                status="missing_previous_provenance",
                previous_round=previous_round,
                current_round=current_round,
                selection_drift=True,
                message=(
                    f"Round {previous_round} for {self.family}/{self.strategy} has no saved provenance; "
                    f"refusing to reuse its optimized config for round {current_round}."
                ),
            )

        if previous.schema_version != current_provenance.schema_version:
            return ProvenanceValidationResult(
                valid=False,
                status="schema_version_drift",
                previous_round=previous_round,
                current_round=current_round,
                selection_drift=True,
                changed_items=("provenance_schema_version",),
                message=(
                    f"Round {previous_round} provenance schema {previous.schema_version} does not match "
                    f"current schema {current_provenance.schema_version}."
                ),
            )

        if previous.selection_fingerprint != current_provenance.selection_fingerprint:
            changed_items = diff_provenance_items(previous, current_provenance, include_diagnostics=False)
            return ProvenanceValidationResult(
                valid=False,
                status="selection_drift",
                previous_round=previous_round,
                current_round=current_round,
                selection_drift=True,
                changed_items=changed_items,
                message=(
                    f"Selection provenance changed between round {previous_round} and round {current_round}; "
                    f"changed items: {', '.join(changed_items) if changed_items else 'selection_fingerprint'}."
                ),
            )

        if previous.diagnostics_fingerprint != current_provenance.diagnostics_fingerprint:
            changed_items = diff_provenance_items(previous, current_provenance, include_diagnostics=True)
            return ProvenanceValidationResult(
                valid=allow_diagnostics_only_drift,
                status="diagnostics_drift",
                previous_round=previous_round,
                current_round=current_round,
                diagnostics_drift=True,
                changed_items=changed_items,
                message=(
                    f"Diagnostics provenance changed between round {previous_round} and round {current_round}; "
                    f"changed items: {', '.join(changed_items) if changed_items else 'diagnostics_fingerprint'}."
                ),
            )

        return ProvenanceValidationResult(
            valid=True,
            status="current",
            previous_round=previous_round,
            current_round=current_round,
            message=f"Round {previous_round} provenance matches current round {current_round}.",
        )

    def load_round_provenance(self, round_num: int) -> AutoRunProvenance | None:
        round_dir = self.round_path(round_num)
        for path in (self.run_summary_path(round_dir), self.run_spec_path(round_dir)):
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            provenance = coerce_provenance(payload.get("provenance"))
            if provenance is not None:
                return provenance

        manifest_entry = self._active_manifest_entry(round_num)
        if not manifest_entry:
            return None
        if not manifest_entry.get("selection_fingerprint") or not manifest_entry.get("diagnostics_fingerprint"):
            return None
        return AutoRunProvenance(
            schema_version=int(manifest_entry.get("provenance_schema_version", 1)),
            selection_fingerprint=str(manifest_entry["selection_fingerprint"]),
            diagnostics_fingerprint=str(manifest_entry["diagnostics_fingerprint"]),
            items=(),
        )

    def archive_rounds(
        self,
        round_nums: Iterable[int],
        *,
        reason: str,
        archive_root: Path | None = None,
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_base = Path(archive_root) if archive_root is not None else self.strategy_dir / "archived_rounds"
        archive_dir = archive_base / f"{timestamp}_{_slug(reason)}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        manifest = self.load_manifest()
        requested = {int(round_num) for round_num in round_nums}
        for entry in manifest.setdefault("rounds", []):
            try:
                entry_round = int(entry.get("round", 0))
            except (TypeError, ValueError):
                continue
            if entry_round in requested and not entry.get("archived"):
                entry["archived"] = True
                entry["archived_at_utc"] = _utc_now_iso()
                entry["archive_reason"] = reason

        for round_num in sorted(requested):
            source = self.round_path(round_num)
            if not source.exists():
                continue
            target = archive_dir / source.name
            suffix = 1
            while target.exists():
                target = archive_dir / f"{source.name}_{suffix}"
                suffix += 1
            shutil.move(str(source), str(target))

        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest, self.manifest_path)
        return archive_dir

    def write_run_spec(
        self,
        round_dir: Path,
        round_num: int,
        strategy_name: str,
        *,
        description: str = "",
        scoring_weights: dict[str, Any] | None = None,
        baseline_mutations: dict[str, Any] | None = None,
        baseline_source: Path | str | None = None,
        execution_context: dict[str, Any] | None = None,
        provenance: AutoRunProvenance | dict[str, Any] | None = None,
        provenance_status: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        path = self.run_spec_path(round_dir)
        if path.exists() and not overwrite:
            self._validate_existing_provenance_file(path, provenance)
            return path

        provenance_payload = _provenance_payload(provenance)
        payload = {
            "family": self.family,
            "strategy": self.strategy,
            "strategy_name": strategy_name,
            "round": round_num,
            "description": description,
            "generated_at_utc": _utc_now_iso(),
            "baseline_source": str(Path(baseline_source).resolve()) if baseline_source else (
                str(self.optimized_config_path(self.round_path(round_num - 1)).resolve())
                if round_num > 1 else None
            ),
            "baseline_mutation_count": len(baseline_mutations or {}),
            "baseline_mutations": dict(baseline_mutations or {}),
            "scoring_weights": dict(scoring_weights or {}),
            "execution_context": dict(execution_context or {}),
        }
        if provenance_payload is not None:
            payload["provenance"] = provenance_payload
        if provenance_status is not None:
            payload["provenance_status"] = provenance_status
        _atomic_write_json(payload, path)
        return path

    def write_run_summary(
        self,
        round_dir: Path,
        cumulative_mutations: dict[str, Any],
        final_metrics: dict[str, Any] | None,
        completed_phases: list[int],
        *,
        round_num: int | None = None,
        source_diagnostics: Path | str | None = None,
        source_phase_state: Path | str | None = None,
        provenance: AutoRunProvenance | dict[str, Any] | None = None,
        provenance_status: str | None = None,
        provenance_validation: ProvenanceValidationResult | dict[str, Any] | None = None,
    ) -> Path:
        resolved_round = round_num if round_num is not None else self._round_num_from_dir(round_dir)
        provenance_payload = _provenance_payload(provenance)
        payload = {
            "family": self.family,
            "strategy": self.strategy,
            "round": resolved_round,
            "generated_at_utc": _utc_now_iso(),
            "completed_phases": list(completed_phases),
            "mutation_count": len(cumulative_mutations),
            "cumulative_mutations": dict(cumulative_mutations),
            "headline_metrics": canonicalize_metrics(final_metrics),
            "final_metrics": dict(final_metrics or {}),
            "source_diagnostics": str(Path(source_diagnostics).resolve()) if source_diagnostics else None,
            "source_phase_state": str(Path(source_phase_state).resolve()) if source_phase_state else None,
        }
        if provenance_payload is not None:
            payload["provenance"] = provenance_payload
        if provenance_status is not None:
            payload["provenance_status"] = provenance_status
        if provenance_validation is not None:
            payload["provenance_validation"] = (
                provenance_validation.to_dict()
                if isinstance(provenance_validation, ProvenanceValidationResult)
                else dict(provenance_validation)
            )
        path = self.run_summary_path(round_dir)
        _atomic_write_json(payload, path)
        return path

    def write_optimized_config(self, round_dir: Path, cumulative_mutations: dict[str, Any]) -> Path:
        path = self.optimized_config_path(round_dir)
        _atomic_write_json(dict(cumulative_mutations), path)
        return path

    def append_to_manifest(
        self,
        round_num: int,
        cumulative_mutations: dict[str, Any],
        final_metrics: dict[str, Any] | None,
        *,
        provenance: AutoRunProvenance | dict[str, Any] | None = None,
        provenance_status: str | None = None,
    ) -> Path:
        manifest = self.load_manifest()
        current_provenance = coerce_provenance(provenance)
        entry = {
            "round": round_num,
            "timestamp": _utc_now_iso(),
            "mutations_count": len(cumulative_mutations),
            "mutations": dict(cumulative_mutations),
        }
        entry.update(canonicalize_metrics(final_metrics))
        if current_provenance is not None:
            entry.update(
                {
                    "selection_fingerprint": current_provenance.selection_fingerprint,
                    "diagnostics_fingerprint": current_provenance.diagnostics_fingerprint,
                    "provenance_schema_version": current_provenance.schema_version,
                }
            )
        if provenance_status is not None:
            entry["provenance_status"] = provenance_status

        rounds = manifest.setdefault("rounds", [])
        replaced = False
        for index, existing in enumerate(rounds):
            if int(existing.get("round", 0)) == round_num and not existing.get("archived"):
                rounds[index] = entry
                replaced = True
                break
        if not replaced:
            rounds.append(entry)
        rounds.sort(key=lambda item: int(item.get("round", 0)))

        self.strategy_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(manifest, self.manifest_path)
        return self.manifest_path

    def _validate_existing_provenance_file(
        self,
        path: Path,
        provenance: AutoRunProvenance | dict[str, Any] | None,
    ) -> None:
        current = coerce_provenance(provenance)
        if current is None:
            return
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        existing = coerce_provenance(payload.get("provenance"))
        if existing is None:
            result = ProvenanceValidationResult(
                valid=False,
                status="missing_existing_spec_provenance",
                selection_drift=True,
                message=f"Existing run spec {path} has no provenance and cannot be reused silently.",
            )
            raise ProvenanceValidationError(result)
        if existing.selection_fingerprint != current.selection_fingerprint:
            changed_items = diff_provenance_items(existing, current, include_diagnostics=False)
            result = ProvenanceValidationResult(
                valid=False,
                status="existing_spec_selection_drift",
                selection_drift=True,
                changed_items=changed_items,
                message=(
                    f"Existing run spec {path} was created with different selection provenance; "
                    f"changed items: {', '.join(changed_items) if changed_items else 'selection_fingerprint'}."
                ),
            )
            raise ProvenanceValidationError(result)
        if existing.diagnostics_fingerprint != current.diagnostics_fingerprint:
            changed_items = diff_provenance_items(existing, current, include_diagnostics=True)
            result = ProvenanceValidationResult(
                valid=False,
                status="existing_spec_diagnostics_drift",
                diagnostics_drift=True,
                changed_items=changed_items,
                message=(
                    f"Existing run spec {path} was created with different diagnostics provenance; "
                    f"changed items: {', '.join(changed_items) if changed_items else 'diagnostics_fingerprint'}."
                ),
            )
            raise ProvenanceValidationError(result)

    def _active_manifest_entry(self, round_num: int) -> dict[str, Any] | None:
        for entry in self.load_manifest().get("rounds", []):
            try:
                entry_round = int(entry.get("round", 0))
            except (TypeError, ValueError):
                continue
            if entry_round == round_num and not entry.get("archived"):
                return entry
        return None

    @classmethod
    def bootstrap_round_1(
        cls,
        family: str,
        strategy: str,
        mutations: dict[str, Any],
        diagnostics_src_path: Path,
        phase_state_src_path: Path | None = None,
        *,
        diagnostics_summary_src_path: Path | None = None,
        base_dir: Path | None = None,
        final_metrics: dict[str, Any] | None = None,
        completed_phases: list[int] | None = None,
        artifacts_dir: Path | None = None,
        baseline_source: Path | str | None = None,
    ) -> Path:
        manager = cls(family, strategy, base_dir=base_dir)
        round_dir = manager.get_round_dir(1)

        diagnostics_src = Path(diagnostics_src_path)
        shutil.copy2(diagnostics_src, manager.diagnostics_path(round_dir))

        phase_state_src = Path(phase_state_src_path) if phase_state_src_path else None
        if phase_state_src and phase_state_src.exists():
            shutil.copy2(phase_state_src, manager.phase_state_path(round_dir))

        diagnostics_summary_src = Path(diagnostics_summary_src_path) if diagnostics_summary_src_path else None
        if diagnostics_summary_src and diagnostics_summary_src.exists():
            shutil.copy2(diagnostics_summary_src, manager.diagnostics_summary_path(round_dir))

        if artifacts_dir:
            excluded_sources = {
                diagnostics_src.resolve(),
                *( [phase_state_src.resolve()] if phase_state_src and phase_state_src.exists() else [] ),
                *( [diagnostics_summary_src.resolve()] if diagnostics_summary_src and diagnostics_summary_src.exists() else [] ),
            }
            manager.copy_bootstrap_artifacts(Path(artifacts_dir), round_dir, exclude_paths=excluded_sources)

        manager.write_run_spec(
            round_dir,
            1,
            strategy_name=strategy,
            description="Round 1 bootstrap from latest post-fix optimized results",
            baseline_mutations=mutations,
            baseline_source=baseline_source or phase_state_src or diagnostics_src,
            overwrite=True,
        )
        manager.write_run_summary(
            round_dir,
            mutations,
            final_metrics or {},
            completed_phases or [],
            round_num=1,
            source_diagnostics=diagnostics_src,
            source_phase_state=phase_state_src,
        )
        manager.write_optimized_config(round_dir, mutations)
        manager.append_to_manifest(1, mutations, final_metrics or {})
        return round_dir

    def copy_bootstrap_artifacts(
        self,
        source_dir: Path,
        round_dir: Path,
        *,
        exclude_paths: set[Path] | None = None,
    ) -> None:
        source_dir = Path(source_dir)
        if not source_dir.exists():
            return
        excluded = {path.resolve() for path in (exclude_paths or set())}
        for path in sorted(source_dir.iterdir()):
            if not path.is_file():
                continue
            if path.resolve() in excluded:
                continue
            name = path.name
            if name in {PHASE_STATE_FILENAME, ROUND_DIAGNOSTICS_FILENAME, DIAGNOSTICS_SUMMARY_FILENAME}:
                continue
            if name in _BOOTSTRAP_EXTRA_FILES or _PHASE_OUTPUT_RE.match(name):
                shutil.copy2(path, Path(round_dir) / name)

    def _round_is_in_progress(self, round_num: int, expected_phases: int | None) -> bool:
        round_dir = self.round_path(round_num)
        state_path = self.phase_state_path(round_dir)
        summary_path = self.run_summary_path(round_dir)

        if not state_path.exists():
            return not summary_path.exists()

        state = load_phase_state(state_path)
        if expected_phases is None:
            return not summary_path.exists()
        return len(state.completed_phases) < expected_phases or not summary_path.exists()

    @staticmethod
    def _round_num_from_dir(round_dir: Path) -> int:
        match = _ROUND_DIR_RE.match(Path(round_dir).name)
        if not match:
            raise ValueError(f"Could not infer round number from directory: {round_dir}")
        return int(match.group(1))


def _provenance_payload(provenance: AutoRunProvenance | dict[str, Any] | None) -> dict[str, Any] | None:
    current = coerce_provenance(provenance)
    return current.to_dict() if current is not None else None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug[:80] or "archived"
