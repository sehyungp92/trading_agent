from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from backtests.momentum.auto.nq_regime.phase_candidates import BASE_MUTATIONS, build_round5_seed_from_configs
from backtests.momentum.auto.nq_regime.plugin import NqRegimePlugin
from backtests.momentum.auto.nq_regime.scoring import IMMUTABLE_WEIGHTS
from backtests.shared.auto.phase_runner import PhaseRunner
from backtests.shared.auto.phase_state import _utc_now_iso


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(IMMUTABLE_WEIGHTS) > 7:
        raise ValueError("round_5 scoring must not use more than 7 components")

    repo_root = Path(__file__).resolve().parents[4]
    output_root = repo_root / "backtests" / "output" / "momentum" / "nq_regime"
    output_dir = output_root / "round_5"
    output_dir.mkdir(parents=True, exist_ok=True)
    round4_config_paths = {
        "round_4a": output_root / "round_4a" / "optimized_config.json",
        "round_4b": output_root / "round_4b" / "optimized_config.json",
        "round_4c": output_root / "round_4c" / "optimized_config.json",
    }
    round4_configs = {name: _load_json(path) for name, path in round4_config_paths.items()}
    baseline_mutations = build_round5_seed_from_configs(
        round4_configs["round_4a"],
        round4_configs["round_4b"],
        round4_configs["round_4c"],
    )

    plugin = NqRegimePlugin(
        data_dir=repo_root / "backtests" / "momentum" / "data" / "raw",
        initial_equity=10_000.0,
        max_workers=2,
        analysis_symbol="NQ",
        trade_symbol="MNQ",
    )
    plugin.initial_mutations = dict(baseline_mutations)
    source_fingerprint = plugin.source_fingerprint()
    source_fingerprint_parts = plugin.source_fingerprint_parts()

    run_spec = {
        "family": "momentum",
        "strategy": "nq_regime",
        "strategy_name": "nq_regime",
        "round": "5",
        "description": "round_5 synergistic all-component alpha/frequency optimization after isolated round_4a/4b/4c diagnostics",
        "generated_at_utc": _utc_now_iso(),
        "baseline_source": (
            "round_4a optimized_config base with round_4b STRUCTURAL* overlay, "
            "round_4c REVERSION* overlay, all modules re-enabled, and round_5 synergy/risk guards"
        ),
        "baseline_source_config_paths": {name: str(path.resolve()) for name, path in round4_config_paths.items()},
        "baseline_source_config_hashes": {name: _file_sha256(path) for name, path in round4_config_paths.items()},
        "baseline_source_mutation_counts": {name: len(config) for name, config in round4_configs.items()},
        "baseline_mutation_count": len(baseline_mutations),
        "baseline_mutations": dict(baseline_mutations),
        "embedded_baseline_matches_loaded_seed": baseline_mutations == BASE_MUTATIONS,
        "scoring_weights": dict(IMMUTABLE_WEIGHTS),
        "score_component_count": len(IMMUTABLE_WEIGHTS),
        "max_workers": 2,
        "strategy_implementation_lessons_alignment": [
            "optimizer-only changes; shared strategy core and execution model remain untouched",
            "source-fingerprinted replay bundle and phase/cache context retained",
            "cohort-pure module guardrails prevent single-component/liquidity-only overfit selection",
            "raw return extraction is treated as an optimization proxy, not benchmark-relative alpha",
        ],
        "source_fingerprint": source_fingerprint,
        "source_fingerprint_parts": source_fingerprint_parts,
        "replay_command": "python -m backtests.momentum.auto.nq_regime.run",
    }
    (output_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2), encoding="utf-8")

    runner = PhaseRunner(
        plugin=plugin,
        output_dir=output_dir,
        round_name="round_5",
        max_rounds=10,
        min_delta=0.003,
        max_retries=0,
        max_diagnostic_retries=0,
    )
    state = runner.run_all_phases()
    final_metrics = plugin.compute_final_metrics(state.cumulative_mutations)
    (output_dir / "optimized_config.json").write_text(json.dumps(state.cumulative_mutations, indent=2), encoding="utf-8")
    summary = {
        "family": "momentum",
        "strategy": "nq_regime",
        "round": "5",
        "generated_at_utc": _utc_now_iso(),
        "baseline_source_config_hashes": run_spec["baseline_source_config_hashes"],
        "completed_phases": state.completed_phases,
        "mutation_count": len(state.cumulative_mutations),
        "cumulative_mutations": state.cumulative_mutations,
        "final_metrics": final_metrics,
        "source_fingerprint": plugin.source_fingerprint(),
        "source_fingerprint_parts": plugin.source_fingerprint_parts(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    reproducibility_manifest = {
        "family": "momentum",
        "strategy": "nq_regime",
        "round": "5",
        "generated_at_utc": summary["generated_at_utc"],
        "source_fingerprint": summary["source_fingerprint"],
        "source_fingerprint_parts": summary["source_fingerprint_parts"],
        "optimized_config_path": str((output_dir / "optimized_config.json").resolve()),
        "run_summary_path": str((output_dir / "run_summary.json").resolve()),
        "baseline_source_config_paths": run_spec["baseline_source_config_paths"],
        "baseline_source_config_hashes": run_spec["baseline_source_config_hashes"],
        "score_component_count": run_spec["score_component_count"],
        "replay_command": run_spec["replay_command"],
        "final_metrics": final_metrics,
    }
    (output_dir / "reproducibility_manifest.json").write_text(json.dumps(reproducibility_manifest, indent=2), encoding="utf-8")
    plugin.close_pool()


if __name__ == "__main__":
    main()
