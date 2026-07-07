from __future__ import annotations

from crypto_trader.optimize.revalidation import (
    build_manifest_checkpoints,
    build_momentum_cleaned_seed_mutations,
    local_perturbation_values,
    normalize_hard_rejects,
    parse_gate_criteria,
)


def test_normalize_hard_rejects_supports_dict_and_list_formats() -> None:
    raw = {
        "total_trades": {"operator": ">=", "threshold": 18},
        "profit_factor": ["<=", 2.5],
    }

    normalized = normalize_hard_rejects(raw)

    assert normalized == {
        "total_trades": (">=", 18.0),
        "profit_factor": ("<=", 2.5),
    }


def test_parse_gate_criteria_builds_dataclasses() -> None:
    parsed = parse_gate_criteria(
        {
            "1": [
                {"metric": "total_trades", "operator": ">=", "threshold": 20},
                {"metric": "profit_factor", "operator": ">=", "threshold": 1.2, "weight": 2},
            ]
        }
    )

    assert 1 in parsed
    assert parsed[1][0].metric == "total_trades"
    assert parsed[1][0].threshold == 20.0
    assert parsed[1][1].weight == 2.0


def test_build_manifest_checkpoints_accumulates_cumulatively() -> None:
    manifest = {
        "rounds": [
            {"round": 1, "mutations": {"a.b": 1}, "profit_factor": 1.2},
            {"round": 2, "mutations": {"c.d": 2, "a.b": 3}, "profit_factor": 1.4},
        ]
    }

    checkpoints = build_manifest_checkpoints(manifest)

    assert [checkpoint.label for checkpoint in checkpoints] == [
        "base",
        "round_1_cumulative",
        "round_2_cumulative",
    ]
    assert checkpoints[1].mutations == {"a.b": 1}
    assert checkpoints[2].mutations == {"a.b": 3, "c.d": 2}
    assert checkpoints[2].source_manifest_metrics == {"profit_factor": 1.4}


def test_local_perturbation_values_handles_tp_fraction_and_ints() -> None:
    assert local_perturbation_values("exits.tp1_frac", 0.25) == [0.15, 0.2, 0.3, 0.35]
    assert local_perturbation_values("exits.time_stop_bars", 12) == [10, 11, 13, 14]


def test_build_momentum_cleaned_seed_mutations_keeps_only_structural_survivors() -> None:
    winner_mutations = {
        "setup.min_confluences_b": 1,
        "bias.h1_adx_threshold": 10.0,
        "trail.trail_buffer_tight": 0.25,
        "risk.risk_pct_b": 0.014,
        "setup.reject_extended_reaction": True,
    }
    ablation_support = {
        "setup.min_confluences_b": True,
        "bias.h1_adx_threshold": False,
        "trail.trail_buffer_tight": True,
        "risk.risk_pct_b": True,
        "setup.reject_extended_reaction": True,
    }

    cleaned = build_momentum_cleaned_seed_mutations(winner_mutations, ablation_support)

    assert cleaned == {
        "setup.min_confluences_b": 1,
        "setup.reject_extended_reaction": True,
    }
