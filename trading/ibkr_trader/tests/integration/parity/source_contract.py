from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceFieldContract:
    path: str
    scope: str
    included_in_fingerprint: bool
    consumption_kind: str
    allowed_families: tuple[str, ...] = ()
    live_consumer: str = ""
    replay_consumer: str = ""
    post_order_consumer: str = ""
    derived_from: str = ""
    validation: str = ""


CONSUMPTION_KINDS = {
    "live_replay",
    "post_order",
    "derived_runtime",
    "validation",
    "excluded_metadata",
    "rejected",
}


SOURCE_FIELD_CONTRACTS: tuple[SourceFieldContract, ...] = (
    SourceFieldContract(
        "schema_version",
        "all",
        True,
        "validation",
        live_consumer="fixtures.load_parity_fixture",
        replay_consumer="fixtures.load_parity_fixture",
        validation="must equal schema v2",
    ),
    SourceFieldContract(
        "surface",
        "all",
        True,
        "live_replay",
        live_consumer="live_runners runner selection",
        replay_consumer="replay_runners runner selection",
    ),
    SourceFieldContract(
        "family",
        "all",
        True,
        "live_replay",
        live_consumer="family_resolver and coordinator registry fallback",
        replay_consumer="family_resolver and replay family surface selection",
        validation="must match family_config.family or strategy_config.family when those are present",
    ),
    SourceFieldContract(
        "clock_start",
        "all",
        True,
        "live_replay",
        live_consumer="live event/runtime clock",
        replay_consumer="replay OMS/event clock",
        validation="timezone-aware timestamp",
    ),
    SourceFieldContract(
        "instruments",
        "all",
        True,
        "live_replay",
        live_consumer="instrument registry and OMS order hydration",
        replay_consumer="instrument registry and OMS order hydration",
    ),
    SourceFieldContract(
        "bars",
        "all",
        True,
        "live_replay",
        live_consumer="live engine bar providers",
        replay_consumer="strategy core replay inputs",
    ),
    SourceFieldContract(
        "higher_timeframe_bars",
        "all",
        True,
        "live_replay",
        live_consumer="live higher-timeframe providers",
        replay_consumer="strategy core replay inputs",
    ),
    SourceFieldContract(
        "artifacts",
        "all",
        True,
        "live_replay",
        live_consumer="runtime artifact providers and idle input builders",
        replay_consumer="replay artifact and idle input builders",
    ),
    SourceFieldContract(
        "artifacts.nq_regime",
        "momentum",
        True,
        "live_replay",
        live_consumer="source_inputs.nq_daily_context and nq_live_context",
        replay_consumer="source_inputs.nq_daily_context and nq_live_context",
        derived_from="artifacts",
    ),
    SourceFieldContract(
        "artifacts.iaric",
        "stock",
        True,
        "live_replay",
        live_consumer="source_inputs.iaric_artifact and iaric_quote",
        replay_consumer="source_inputs.iaric_artifact and iaric_quote",
        derived_from="artifacts",
    ),
    SourceFieldContract(
        "artifacts.<idle>.idle_market_input",
        "configured idle children",
        True,
        "live_replay",
        live_consumer="live idle fetch/callback seeding",
        replay_consumer="replay idle strategy core adapters",
        derived_from="artifacts",
        validation="configured idle children must have bar-backed idle_market_input",
    ),
    SourceFieldContract(
        "artifacts.overlay_rebalance",
        "swing",
        True,
        "live_replay",
        live_consumer="overlay rebalance provider",
        replay_consumer="overlay replay planner",
        derived_from="artifacts",
    ),
    SourceFieldContract(
        "strategy_config",
        "all",
        True,
        "live_replay",
        live_consumer="engine and OMS configuration",
        replay_consumer="strategy core and OMS configuration",
    ),
    SourceFieldContract(
        "strategy_config.config_overrides",
        "all",
        True,
        "live_replay",
        live_consumer="engine settings/config construction",
        replay_consumer="strategy core settings/config construction",
        derived_from="strategy_config",
    ),
    SourceFieldContract(
        "family_config",
        "layer3",
        True,
        "live_replay",
        live_consumer="coordinator and OMS family configuration",
        replay_consumer="family replay surfaces and OMS family configuration",
    ),
    SourceFieldContract(
        "family_config.strategies",
        "layer3",
        True,
        "live_replay",
        live_consumer="coordinator child discovery and OMS risk config",
        replay_consumer="replay child discovery and OMS risk config",
        derived_from="family_config",
    ),
    SourceFieldContract(
        "family_config.portfolio_rules",
        "layer3",
        True,
        "live_replay",
        live_consumer="portfolio_rules_config_from_fixture",
        replay_consumer="portfolio_rules_config_from_fixture",
        derived_from="family_config",
    ),
    SourceFieldContract(
        "initial_repository_state",
        "layer3",
        True,
        "live_replay",
        live_consumer="repository and OMS risk hydration",
        replay_consumer="repository and family replay hydration",
    ),
    SourceFieldContract(
        "initial_repository_state.orders",
        "layer3",
        True,
        "live_replay",
        live_consumer="oms_hydration.hydrate_repository_from_fixture",
        replay_consumer="oms_hydration.hydrate_repository_from_fixture",
        derived_from="initial_repository_state",
        validation="enum values and risk_context fields must be valid",
    ),
    SourceFieldContract(
        "initial_repository_state.positions",
        "layer3",
        True,
        "live_replay",
        live_consumer="oms_hydration.hydrate_repository_from_fixture",
        replay_consumer="oms_hydration.hydrate_repository_from_fixture and family replay surfaces",
        derived_from="initial_repository_state",
    ),
    SourceFieldContract(
        "initial_strategy_state",
        "all",
        True,
        "live_replay",
        live_consumer="engine hydration",
        replay_consumer="strategy core replay hydration",
    ),
    SourceFieldContract(
        "initial_strategy_state.<strategy_id>",
        "all",
        True,
        "live_replay",
        live_consumer="strategy-specific engine hydration",
        replay_consumer="strategy-specific core replay hydration",
        derived_from="initial_strategy_state",
    ),
    SourceFieldContract(
        "initial_family_state",
        "layer3",
        True,
        "live_replay",
        live_consumer="overlay engine starting state",
        replay_consumer="overlay replay starting state",
        allowed_families=("swing",),
        validation="momentum and stock must remain empty; swing may contain overlay only",
    ),
    SourceFieldContract(
        "initial_family_state.overlay",
        "swing",
        True,
        "live_replay",
        live_consumer="overlay engine starting state",
        replay_consumer="overlay replay starting state",
        derived_from="initial_family_state",
    ),
    SourceFieldContract(
        "account_state",
        "all",
        True,
        "live_replay",
        live_consumer="OMS/equity providers",
        replay_consumer="replay OMS/equity providers",
    ),
    SourceFieldContract(
        "broker_event_script",
        "all",
        True,
        "post_order",
        live_consumer="fake broker adapter after live order submission",
        replay_consumer="fake broker adapter after replay OMS submission",
        post_order_consumer="broker script application",
        validation="events must target submitted orders with order_match",
    ),
    SourceFieldContract(
        "runtime_inputs.configured_strategy_ids",
        "all",
        True,
        "derived_runtime",
        live_consumer="live source payload",
        replay_consumer="replay source payload",
        derived_from="family_config.strategies and strategy_config.strategy_id",
    ),
    SourceFieldContract(
        "runtime_inputs.portfolio_rules",
        "layer3",
        True,
        "derived_runtime",
        live_consumer="OMS/coordinator risk config",
        replay_consumer="replay OMS/family risk config",
        derived_from="family_config.portfolio_rules",
    ),
    SourceFieldContract(
        "runtime_inputs.overlay_rebalance",
        "swing",
        True,
        "derived_runtime",
        live_consumer="overlay provider payload",
        replay_consumer="overlay planner replay payload",
        derived_from="artifacts.overlay_rebalance and initial_family_state.overlay",
    ),
    SourceFieldContract(
        "timezone",
        "all",
        False,
        "excluded_metadata",
        validation="not consumed; fixture timestamps must be timezone-aware",
    ),
    SourceFieldContract(
        "market_calendar",
        "all",
        False,
        "excluded_metadata",
        validation="not consumed by the current fixture calendar override",
    ),
    SourceFieldContract(
        "expected_fill_model",
        "all",
        False,
        "excluded_metadata",
        validation="assertion metadata only until deterministic fill defaults consume it",
    ),
    SourceFieldContract(
        "expected_normalized_outputs",
        "all",
        False,
        "excluded_metadata",
        validation="expected comparison metadata only",
    ),
    SourceFieldContract(
        "expected_trade_count",
        "layer3",
        False,
        "excluded_metadata",
        validation="assertion metadata only",
    ),
    SourceFieldContract(
        "orders",
        "all",
        False,
        "rejected",
        validation="removed schema v1 source field",
    ),
    SourceFieldContract(
        "strategy_inputs",
        "all",
        False,
        "rejected",
        validation="removed scripted decision source field",
    ),
    SourceFieldContract(
        "entry_actions",
        "all",
        False,
        "rejected",
        validation="removed scripted decision source field",
    ),
    SourceFieldContract(
        "parity_entry_signals",
        "all",
        False,
        "rejected",
        validation="removed scripted decision source field",
    ),
    SourceFieldContract(
        "generated_decision_trace",
        "all",
        False,
        "rejected",
        validation="removed generated decision artifact",
    ),
)


_CONTRACT_BY_PATH = {row.path: row for row in SOURCE_FIELD_CONTRACTS}
SOURCE_TOP_LEVEL_KEYS = tuple(
    row.path
    for row in SOURCE_FIELD_CONTRACTS
    if "." not in row.path and row.included_in_fingerprint
)
EXCLUDED_TOP_LEVEL_KEYS = frozenset(
    row.path
    for row in SOURCE_FIELD_CONTRACTS
    if "." not in row.path and row.consumption_kind == "excluded_metadata"
)
REJECTED_TOP_LEVEL_KEYS = frozenset(
    row.path
    for row in SOURCE_FIELD_CONTRACTS
    if "." not in row.path and row.consumption_kind == "rejected"
)
SCRIPTED_DECISION_KEYS = frozenset(
    {"strategy_inputs", "entry_actions", "parity_entry_signals", "generated_decision_trace"}
)
ALLOWED_TOP_LEVEL_KEYS = (
    set(SOURCE_TOP_LEVEL_KEYS)
    | set(EXCLUDED_TOP_LEVEL_KEYS)
    | set(REJECTED_TOP_LEVEL_KEYS)
)
RUNTIME_INPUT_CONTRACT_PATHS = tuple(
    row.path
    for row in SOURCE_FIELD_CONTRACTS
    if row.path.startswith("runtime_inputs.")
)


class SourceContractError(ValueError):
    pass


def validate_contract_table() -> None:
    path_counts = Counter(row.path for row in SOURCE_FIELD_CONTRACTS)
    duplicate_paths = {path for path, count in path_counts.items() if count > 1}
    if duplicate_paths:
        raise SourceContractError(f"duplicate source contract path(s): {sorted(duplicate_paths)}")
    invalid = [row.path for row in SOURCE_FIELD_CONTRACTS if row.consumption_kind not in CONSUMPTION_KINDS]
    if invalid:
        raise SourceContractError(f"invalid source contract consumption kind(s): {invalid}")
    for row in SOURCE_FIELD_CONTRACTS:
        _validate_contract_row(row)


def _validate_contract_row(row: SourceFieldContract) -> None:
    if row.consumption_kind in {"excluded_metadata", "rejected"} and row.included_in_fingerprint:
        raise SourceContractError(f"{row.path} cannot be fingerprinted and {row.consumption_kind}")
    if row.consumption_kind == "live_replay" and not (row.live_consumer and row.replay_consumer):
        raise SourceContractError(f"{row.path} must declare live and replay consumers")
    if row.consumption_kind == "post_order" and not row.post_order_consumer:
        raise SourceContractError(f"{row.path} must declare a post-order consumer")
    if row.consumption_kind == "derived_runtime":
        if not row.path.startswith("runtime_inputs."):
            raise SourceContractError(f"{row.path} derived runtime row must live under runtime_inputs")
        if not (row.derived_from and row.live_consumer and row.replay_consumer):
            raise SourceContractError(f"{row.path} must declare source and runtime consumers")
    if row.path.startswith("runtime_inputs.") and row.consumption_kind != "derived_runtime":
        raise SourceContractError(f"{row.path} runtime input row must be derived_runtime")


def validate_top_level_source_fields(payload: Mapping[str, Any], path: Path) -> None:
    for key in sorted(str(key) for key in payload):
        if key in REJECTED_TOP_LEVEL_KEYS:
            row = _CONTRACT_BY_PATH[key]
            raise SourceContractError(f"{path} uses removed source field: {key} ({row.validation})")
    unknown = sorted(str(key) for key in payload if str(key) not in ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise SourceContractError(
            f"{path} contains uncontracted top-level source field(s): {', '.join(unknown)}"
        )


def validate_family_alias(payload: Mapping[str, Any], path: Path) -> None:
    top_level = payload.get("family")
    expected = None
    family_config = payload.get("family_config", {}) or {}
    strategy_config = payload.get("strategy_config", {}) or {}
    if isinstance(family_config, Mapping) and family_config.get("family"):
        expected = family_config.get("family")
    elif isinstance(strategy_config, Mapping) and strategy_config.get("family"):
        expected = strategy_config.get("family")
    if top_level in (None, "") or expected in (None, ""):
        return
    if str(top_level) != str(expected):
        raise SourceContractError(
            f"{path} top-level family={top_level!r} diverges from configured family={expected!r}"
        )
