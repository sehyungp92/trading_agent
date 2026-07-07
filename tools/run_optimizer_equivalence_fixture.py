from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_SRC = ROOT / "packages" / "trading_optimizer" / "src"
if str(OPTIMIZER_SRC) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER_SRC))

from trading_optimizer.phase_runner_adapters import (  # noqa: E402
    LEGACY_PHASE_RUNNERS,
    LegacyPhaseRunnerAdapter,
)


SCOPES = {
    "trading_stock_family": "ibkr",
    "trading_momentum_family": "ibkr",
    "trading_swing_family": "ibkr",
    "crypto_trader_portfolio": "crypto",
    "k_stock_olr_kalcb": "k_stock",
}
BOT_PATHS = {
    "ibkr": ROOT / "bots" / "ibkr_trading",
    "crypto": ROOT / "bots" / "crypto_trader" / "src",
    "k_stock": ROOT / "bots" / "k_stock_trader",
}
EVIDENCE_ROOT = ROOT / "artifacts" / "validation" / "optimizer_compatibility"
DIMENSIONS = (
    "cumulative_mutations",
    "gate_decisions",
    "selected_candidates",
    "canonical_round_outputs",
)


def main() -> int:
    args = _parser().parse_args()
    if args.runner:
        return _run_one(args.scope, args.fixture_set, args.runner, args.result_path)
    return _write_evidence(args.scope, args.fixture_set)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate optimizer runner equivalence evidence.")
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--fixture-set", default="smoke")
    parser.add_argument("--runner", choices=["legacy", "adapter"], default=None)
    parser.add_argument("--result-path", type=Path, default=None)
    return parser


def _write_evidence(scope: str, fixture_set: str) -> int:
    results: dict[str, dict[str, Any]] = {}
    command_records: dict[str, dict[str, Any]] = {}
    for runner in ("legacy", "adapter"):
        result_path = EVIDENCE_ROOT / "_runner_results" / f"{scope}.{fixture_set}.{runner}.json"
        command = [
            sys.executable,
            "tools/run_optimizer_equivalence_fixture.py",
            "--scope",
            scope,
            "--fixture-set",
            fixture_set,
            "--runner",
            runner,
            "--result-path",
            result_path.as_posix(),
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        command_records[runner] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout.splitlines()[-20:],
            "stderr_tail": completed.stderr.splitlines()[-20:],
            "result_path": _relative(result_path),
        }
        if completed.returncode != 0 or not result_path.exists():
            return _emit_failure(scope, fixture_set, command_records)
        results[runner] = _read_json(result_path)

    legacy_outputs = results["legacy"]["canonical_outputs"]
    adapter_outputs = results["adapter"]["canonical_outputs"]
    comparisons = []
    errors = []
    for dimension in DIMENSIONS:
        legacy_payload = legacy_outputs.get(dimension)
        adapter_payload = adapter_outputs.get(dimension)
        status = "pass" if legacy_payload == adapter_payload else "fail"
        if status != "pass":
            errors.append(f"{dimension} mismatch")
        comparisons.append({
            "dimension": dimension,
            "status": status,
            "legacy_hash": _hash_payload(legacy_payload),
            "adapter_hash": _hash_payload(adapter_payload),
            "notes": "same-input legacy runner and shared adapter runner outputs match",
        })

    bot = SCOPES[scope]
    spec = LEGACY_PHASE_RUNNERS[bot]
    evidence = {
        "schema_version": "optimizer_runner_equivalence_matrix_v1",
        "scope_id": scope,
        "fixture_set": fixture_set,
        "status": "pass" if not errors else "fail",
        "legacy_runner": f"{spec.module_path}.{spec.class_name}",
        "adapter_runner": "trading_optimizer.phase_runner_adapters.LegacyPhaseRunnerAdapter",
        "wrapped_legacy_runners": [
            {
                "bot": bot,
                "module_path": spec.module_path,
                "class_name": spec.class_name,
                "source_path": spec.source_path,
                "source_sha256": _file_hash(ROOT / spec.source_path),
            }
        ],
        "execution_evidence": {
            "legacy_command": command_records["legacy"],
            "adapter_command": command_records["adapter"],
            "input_hashes": {
                "fixture": _hash_payload(_fixture_descriptor(scope, fixture_set)),
                "legacy_runner_source": _file_hash(ROOT / spec.source_path),
                "shared_adapter_source": _file_hash(
                    ROOT / "packages" / "trading_optimizer" / "src"
                    / "trading_optimizer" / "phase_runner_adapters.py"
                ),
            },
            "output_hashes": {
                "legacy": _hash_payload(legacy_outputs),
                "adapter": _hash_payload(adapter_outputs),
            },
            "compared_payloads": {
                dimension: {
                    "legacy": legacy_outputs.get(dimension),
                    "adapter": adapter_outputs.get(dimension),
                }
                for dimension in DIMENSIONS
            },
        },
        "comparisons": comparisons,
        "errors": errors,
    }
    path = EVIDENCE_ROOT / f"{scope}.{fixture_set}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _run_one(scope: str, fixture_set: str, runner: str, result_path: Path | None) -> int:
    bot = SCOPES[scope]
    bot_path = BOT_PATHS[bot]
    sys.path.insert(0, str(bot_path))
    output_dir = EVIDENCE_ROOT / "_runs" / scope / fixture_set / runner
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plugin = _fixture_plugin(bot, scope)
    state = _execute_runner(bot, plugin, output_dir, runner)
    result = {
        "scope": scope,
        "fixture_set": fixture_set,
        "runner": runner,
        "bot": bot,
        "fixture": _fixture_descriptor(scope, fixture_set),
        "canonical_outputs": _canonical_outputs(state),
    }
    if result_path is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _execute_runner(bot: str, plugin: Any, output_dir: Path, runner: str) -> Any:
    spec = LEGACY_PHASE_RUNNERS[bot]
    kwargs = _runner_kwargs(bot)
    if runner == "adapter":
        return LegacyPhaseRunnerAdapter(spec).run(plugin, output_dir, **kwargs)
    module = __import__(spec.module_path, fromlist=[spec.class_name])
    runner_cls = getattr(module, spec.class_name)
    phase_runner = runner_cls(plugin, output_dir, **kwargs)
    if bot == "crypto":
        return phase_runner.run_all_phases(phase_runner.load_state())
    return phase_runner.run_all_phases()


def _runner_kwargs(bot: str) -> dict[str, Any]:
    if bot == "crypto":
        return {
            "min_delta": 0.0,
            "max_retries": 0,
            "max_diagnostic_retries": 0,
            "validation_mode": "dev",
        }
    return {
        "max_rounds": 2,
        "min_delta": 0.0,
        "max_retries": 0,
        "max_diagnostic_retries": 0,
    }


def _fixture_plugin(bot: str, scope: str) -> Any:
    if bot == "crypto":
        return _crypto_fixture(scope)
    if bot == "k_stock":
        return _k_stock_fixture(scope)
    return _ibkr_fixture(scope)


def _ibkr_fixture(scope: str) -> Any:
    from backtests.shared.auto.plugin import PhaseAnalysisPolicy, PhaseSpec
    from backtests.shared.auto.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

    class FixturePlugin:
        name = scope
        num_phases = 1
        ultimate_targets = {"score": 1.0}
        initial_mutations = {"base": 1}

        def get_phase_spec(self, phase: int, state: Any) -> Any:
            return PhaseSpec(
                focus="fixture",
                candidates=[
                    Experiment("alpha", {"alpha": 1}),
                    Experiment("beta", {"beta": 2}),
                ],
                gate_criteria_fn=lambda metrics: [
                    GateCriterion("score", 1.0, float(metrics.get("score", 0.0)), True)
                ],
                scoring_weights={"score": 1.0},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(min_effective_score_delta_pct=0.0),
                max_rounds=2,
                prune_threshold=0.0,
                reject_streak_limit=2,
            )

        def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], **_: Any) -> Any:
            def evaluate(candidates: list[Any], current: dict[str, Any]) -> list[Any]:
                rows = []
                for candidate in candidates:
                    merged = {**current, **candidate.mutations}
                    score = 1.0 + len(merged) / 10 + (0.2 if "beta" in candidate.mutations else 0.1)
                    rows.append(ScoredCandidate(candidate.name, score, False, "", {"score": score}))
                return rows

            return evaluate

        def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
            return {"score": 1.0 + len(mutations) / 10}

        def run_phase_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def run_enhanced_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def build_end_of_round_artifacts(self, state: Any) -> Any:
            return EndOfRoundArtifacts("fixture final diagnostics", {}, "pass")

    return FixturePlugin()


def _crypto_fixture(scope: str) -> Any:
    from crypto_trader.optimize.types import (
        EndOfRoundArtifacts,
        Experiment,
        GateCriterion,
        PhaseAnalysisPolicy,
        PhaseSpec,
        ScoredCandidate,
    )

    class FixturePlugin:
        name = scope
        num_phases = 1
        ultimate_targets = {"score": 1.0}
        initial_mutations = {"base": 1}

        def get_phase_spec(self, phase: int, state: Any) -> Any:
            return PhaseSpec(
                phase_num=phase,
                name="fixture",
                candidates=[
                    Experiment("alpha", {"alpha": 1}),
                    Experiment("beta", {"beta": 2}),
                ],
                scoring_weights={"score": 1.0},
                hard_rejects={},
                gate_criteria=[GateCriterion("score", ">=", 1.0)],
                gate_criteria_fn=lambda metrics: [GateCriterion("score", ">=", 1.0)],
                analysis_policy=PhaseAnalysisPolicy(
                    max_scoring_retries=0,
                    max_diagnostic_retries=0,
                    min_effective_score_delta_pct=0.0,
                ),
                min_delta=0.0,
                focus="fixture",
                max_rounds=2,
                prune_threshold=0.0,
            )

        def create_evaluate_batch(
            self,
            phase: int,
            cumulative_mutations: dict[str, Any],
            scoring_weights: dict[str, float] | None = None,
            hard_rejects: dict[str, Any] | None = None,
        ) -> Any:
            def evaluate(candidates: list[Any], current: dict[str, Any]) -> list[Any]:
                rows = []
                for candidate in candidates:
                    merged = {**current, **candidate.mutations}
                    score = 1.0 + len(merged) / 10 + (0.2 if "beta" in candidate.mutations else 0.1)
                    rows.append(ScoredCandidate(candidate, score, {"score": score}, False, ""))
                return rows

            return evaluate

        def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
            return {"score": 1.0 + len(mutations) / 10}

        def run_phase_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def run_enhanced_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def build_end_of_round_artifacts(self, state: Any) -> Any:
            return EndOfRoundArtifacts("fixture final diagnostics", {}, "pass")

    return FixturePlugin()


def _k_stock_fixture(scope: str) -> Any:
    from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
    from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate

    hygiene = {
        "same_bar_fill_count": 0.0,
        "forced_replay_close_count": 0.0,
        "rejected_order_count": 0.0,
        "end_open_position_count": 0.0,
    }

    class FixturePlugin:
        name = scope
        num_phases = 1
        ultimate_targets = {"score": 1.0}
        initial_mutations = {"base": 1}
        execution_context = {
            "source_fingerprint": "fixture",
            "feature_manifest_hash": "fixture",
            "candidate_snapshot_hash": "fixture",
            "capability_level": "fixture",
        }
        config = {"allow_synthetic_promotion": True}

        def get_phase_spec(self, phase: int, state: Any) -> Any:
            return PhaseSpec(
                focus="fixture",
                candidates=[
                    Experiment("alpha", {"alpha": 1}),
                    Experiment("beta", {"beta": 2}),
                ],
                gate_criteria_fn=lambda metrics: [
                    GateCriterion("score", 1.0, float(metrics.get("score", 0.0)), True)
                ],
                scoring_weights={"score": 1.0},
                hard_rejects={},
                analysis_policy=PhaseAnalysisPolicy(min_effective_score_delta_pct=0.0),
                max_rounds=2,
                prune_threshold=0.0,
                reject_streak_limit=2,
                phase_metric_basis="fixture",
                primary_promotion_metric="score",
                official_metric_keys=("score",),
                promotion_requires_audit_pass=False,
            )

        def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], **_: Any) -> Any:
            def evaluate(candidates: list[Any], current: dict[str, Any]) -> list[Any]:
                rows = []
                for candidate in candidates:
                    merged = {**current, **candidate.mutations}
                    score = 1.0 + len(merged) / 10 + (0.2 if "beta" in candidate.mutations else 0.1)
                    rows.append(ScoredCandidate(candidate.name, score, False, "", {"score": score, **hygiene}))
                return rows

            return evaluate

        def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float | str]:
            return {
                "score": 1.0 + len(mutations) / 10,
                "official_metric_basis": "fixture",
                "source_fingerprint": "fixture",
                "feature_manifest_hash": "fixture",
                "candidate_snapshot_hash": "fixture",
                **hygiene,
            }

        def run_phase_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def run_enhanced_diagnostics(self, *_: Any) -> str:
            return "fixture diagnostics pass"

        def build_end_of_round_artifacts(self, state: Any) -> Any:
            return EndOfRoundArtifacts("fixture final diagnostics", {}, "pass")

    return FixturePlugin()


def _canonical_outputs(state: Any) -> dict[str, Any]:
    phase_results = _strip_nondeterminism(_jsonable(getattr(state, "phase_results", {})))
    gate_results = _strip_nondeterminism(_jsonable(getattr(state, "phase_gate_results", {})))
    selected = {
        str(phase): {
            "accepted_count": result.get("accepted_count"),
            "adoption_reason": result.get("adoption_reason"),
            "kept_features": result.get("kept_features", []),
            "new_mutations": result.get("new_mutations", {}),
        }
        for phase, result in sorted(phase_results.items(), key=lambda item: str(item[0]))
        if isinstance(result, dict)
    }
    return {
        "cumulative_mutations": _strip_nondeterminism(_jsonable(getattr(state, "cumulative_mutations", {}))),
        "gate_decisions": gate_results,
        "selected_candidates": selected,
        "canonical_round_outputs": phase_results,
    }


def _strip_nondeterminism(value: Any) -> Any:
    volatile_keys = {
        "elapsed_seconds",
        "generated_at",
        "generated_at_utc",
        "timestamp",
        "updated_at",
        "started_at",
        "completed_at",
    }
    if isinstance(value, dict):
        return {
            str(key): _strip_nondeterminism(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            if str(key) not in volatile_keys
        }
    if isinstance(value, list):
        return [_strip_nondeterminism(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _fixture_descriptor(scope: str, fixture_set: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "fixture_set": fixture_set,
        "bot": SCOPES[scope],
        "num_phases": 1,
        "initial_mutations": {"base": 1},
        "candidates": [
            {"name": "alpha", "mutations": {"alpha": 1}},
            {"name": "beta", "mutations": {"beta": 2}},
        ],
        "dimensions": list(DIMENSIONS),
    }


def _emit_failure(scope: str, fixture_set: str, command_records: dict[str, Any]) -> int:
    evidence = {
        "schema_version": "optimizer_runner_equivalence_matrix_v1",
        "scope_id": scope,
        "fixture_set": fixture_set,
        "status": "fail",
        "execution_evidence": command_records,
        "comparisons": [],
        "errors": ["legacy or adapter fixture execution failed"],
    }
    path = EVIDENCE_ROOT / f"{scope}.{fixture_set}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _hash_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
