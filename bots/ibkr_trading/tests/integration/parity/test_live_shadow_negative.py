from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tests.integration.parity.fixtures import fixture_source_fingerprint, load_parity_fixture
from tests.integration.parity.harness import run_layer3_family_contract
from tests.integration.parity.live_runners import run_layer2_live_trace, run_layer3_family_live_trace
from tests.integration.parity.live_shadow_contract import (
    FamilyShadowContract,
    LiveShadowContract,
    assert_family_shadow_contract,
    assert_shadow_contract,
)
from tests.integration.parity.replay_runners import run_layer2_replay_trace, run_layer3_family_replay_trace

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "parity"
TPC_FIXTURE = FIXTURE_ROOT / "layer2" / "tpc_entry_fill.json"
MOMENTUM_FAMILY_FIXTURE = FIXTURE_ROOT / "layer3" / "momentum_family_shared_risk.json"
SWING_FAMILY_FIXTURE = FIXTURE_ROOT / "layer3" / "swing_family_overlay_rebalance.json"
STOCK_FAMILY_FIXTURE = FIXTURE_ROOT / "layer3" / "stock_family_collision.json"
IDLE_MARKET_INPUT_TIMESTAMP = "2026-05-20T14:30:00+00:00"


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_tampered_replay_quantity_fails_order_intent_comparison() -> None:
    fixture = load_parity_fixture(TPC_FIXTURE)
    live = await run_layer2_live_trace(fixture)
    replay = run_layer2_replay_trace(fixture)
    tampered_orders = [dict(row) for row in replay.order_intents]
    tampered_orders[0]["qty"] = int(tampered_orders[0]["qty"]) + 1

    with pytest.raises(AssertionError, match="order_intents"):
        assert_shadow_contract(
            LiveShadowContract(
                surface="TPC",
                live=live,
                replay=replace(replay, order_intents=tampered_orders),
            )
        )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_missing_replay_broker_fill_fails_event_or_ledger_comparison() -> None:
    fixture = load_parity_fixture(TPC_FIXTURE)
    live = await run_layer2_live_trace(fixture)
    replay = run_layer2_replay_trace(fixture)

    with pytest.raises(AssertionError, match="terminal_events"):
        assert_shadow_contract(
            LiveShadowContract(
                surface="TPC",
                live=live,
                replay=replace(replay, terminal_events=[], trade_ledger=[]),
            )
        )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_omitted_configured_family_child_fails_family_assertion() -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    contract = await run_layer3_family_contract("momentum", MOMENTUM_FAMILY_FIXTURE)
    expected = {f"momentum:{item['id']}" for item in fixture["family_config"]["strategies"]}

    with pytest.raises(AssertionError, match="child contract mismatch"):
        assert_family_shadow_contract(
            FamilyShadowContract(
                family=contract.family,
                children=contract.children[:-1],
                live_family_state=contract.live_family_state,
                replay_family_state=contract.replay_family_state,
            ),
            expected_trades=int(fixture.get("expected_trade_count", 0)),
            expected_surfaces=expected,
        )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_altered_initial_family_state_changes_fingerprint_and_state_comparison() -> None:
    fixture = load_parity_fixture(SWING_FAMILY_FIXTURE)
    altered = deepcopy(fixture)
    altered["initial_family_state"]["overlay"]["rebalance_due"] = False

    live = await run_layer3_family_live_trace(fixture)
    replay = run_layer3_family_replay_trace(altered)

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(altered)
    assert (live.state_snapshot or {})["family_state"] != (replay.state_snapshot or {})["family_state"]

    with pytest.raises(AssertionError, match="family state mismatch"):
        assert_family_shadow_contract(
            FamilyShadowContract(
                family="swing",
                children=[],
                live_family_state=(live.state_snapshot or {})["family_state"],
                replay_family_state=(replay.state_snapshot or {})["family_state"],
            ),
            expected_trades=0,
            expected_surfaces=set(),
        )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_family_replay_surface_quantity_decision_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    original = replay_runners._run_family_portfolio_surface

    def tampered_surface(current_fixture, out):
        surface = deepcopy(original(current_fixture, out))
        for decision in surface.get("decisions", []) or []:
            if str(decision.get("status", "")).lower() in {"accepted", "reduced"}:
                decision["approved_qty"] = int(decision.get("approved_qty", 0) or 0) + 1
                surface["accepted_quantities"] = {decision["strategy_id"]: decision["approved_qty"]}
                return surface
        raise AssertionError("stock fixture did not produce an accepted/reduced family decision")

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", tampered_surface)

    live = await run_layer3_family_live_trace(fixture)
    replay = replay_runners.run_layer3_family_replay_trace(fixture)
    assert any(
        row.get("strategy_id") == "IARIC_v1" and row.get("qty") == 2
        for row in replay.order_intents
    )
    with pytest.raises(AssertionError, match="order_intents"):
        assert_shadow_contract(
            LiveShadowContract(
                surface="stock_family",
                live=live,
                replay=replay,
            )
        )


@pytest.mark.parity_nightly
def test_family_replay_surface_must_emit_decisions_for_generated_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    def empty_surface(current_fixture, out):
        return {
            "adapter": "regressed_stock_portfolio_replay",
            "decisions": [],
            "accepted_counts": {},
            "blocked_counts": {},
            "blocked_reasons": {},
            "accepted_quantities": {},
        }

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", empty_surface)

    with pytest.raises(AssertionError, match="emitted no decisions"):
        replay_runners.run_layer3_family_replay_trace(fixture)


@pytest.mark.parity_nightly
def test_family_replay_surface_rejects_unknown_decision_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    original = replay_runners._run_family_portfolio_surface

    def invalid_status_surface(current_fixture, out):
        surface = deepcopy(original(current_fixture, out))
        for decision in surface.get("decisions", []) or []:
            decision["status"] = "blocked"
            return surface
        raise AssertionError("stock fixture did not produce a family decision")

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", invalid_status_surface)

    with pytest.raises(AssertionError, match="unsupported family replay decision status"):
        replay_runners.run_layer3_family_replay_trace(fixture)


@pytest.mark.parity_nightly
@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"status": "blocked"}, "unsupported family replay decision status"),
        ({"approved_qty": 1.5}, "approved_qty must be an integer"),
        ({"status": "accepted", "approved_qty": 1}, "accepted family replay decision must preserve quantity"),
        ({"status": "reduced", "approved_qty": 3}, "reduced family replay decision"),
        ({"status": "rejected", "approved_qty": 1}, "rejected family replay decision"),
    ],
)
def test_family_replay_rejects_invalid_decision_payload(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict,
    match: str,
) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    original = replay_runners._run_family_portfolio_surface

    def invalid_surface(current_fixture, out):
        surface = deepcopy(original(current_fixture, out))
        decision = surface["decisions"][0]
        decision.update(updates)
        return surface

    async def forbidden_sink(*_args, **_kwargs):
        raise AssertionError("invalid family decision reached replay OMS")

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", invalid_surface)
    monkeypatch.setattr(replay_runners, "run_replay_oms_sink", forbidden_sink)

    with pytest.raises(AssertionError, match=match):
        replay_runners.run_layer3_family_replay_trace(fixture)


@pytest.mark.parity_nightly
def test_family_replay_rejects_duplicate_candidate_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    original = replay_runners._run_family_portfolio_surface

    def duplicate_surface(current_fixture, out):
        surface = deepcopy(original(current_fixture, out))
        surface["decisions"].append(deepcopy(surface["decisions"][0]))
        return surface

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", duplicate_surface)

    with pytest.raises(AssertionError, match="duplicate decision"):
        replay_runners.run_layer3_family_replay_trace(fixture)


@pytest.mark.parity_nightly
def test_swing_family_replay_reduction_drives_canonical_order_quantity() -> None:
    fixture = load_parity_fixture(SWING_FAMILY_FIXTURE)
    fixture["initial_repository_state"] = {
        "positions": [
            {
                "strategy_id": "ATRSS",
                "account_id": "ACCT-PARITY",
                "symbol": "QQQ",
                "net_qty": 1,
                "avg_price": 100.0,
                "open_risk_dollars": 1_650.0,
                "open_risk_R": 1.0,
                "last_update_at": "2026-05-20T14:29:00+00:00",
            }
        ]
    }

    replay = run_layer3_family_replay_trace(fixture)

    tpc_entries = [
        row
        for row in replay.order_intents
        if row.get("strategy_id") == "TPC" and row.get("order_role") == "ENTRY"
    ]
    assert tpc_entries and tpc_entries[0]["qty"] == 5
    family_state = replay.state_snapshot["family_state"]
    assert family_state["portfolio_surface"]["accepted_quantities"]["TPC"] == 5


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_initial_repository_positions_seed_family_risk_before_decision() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    live = await run_layer3_family_live_trace(fixture)
    replay = run_layer3_family_replay_trace(fixture)

    for trace in (live, replay):
        family_state = (trace.state_snapshot or {})["family_state"]
        strategy_risk = family_state["risk_state"]["strategy"]
        assert strategy_risk["ALCB_v1"]["open_risk_dollars"] == 100
        assert family_state["portfolio_surface"]["accepted_quantities"]["IARIC_v1"] == 1
        entry_orders = [
            row
            for row in trace.order_intents
            if row.get("strategy_id") == "IARIC_v1"
            and row.get("order_role") == "ENTRY"
        ]
        assert entry_orders and entry_orders[0]["qty"] == 1


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_replay_oms_hydrates_initial_order_status_and_risk_context() -> None:
    fixture = _fixture_with_initial_working_entry_order()

    live = await run_layer3_family_live_trace(fixture)
    replay = run_layer3_family_replay_trace(fixture)

    for trace in (live, replay):
        family_state = (trace.state_snapshot or {})["family_state"]
        assert family_state["repository"]["order_counts"]["ALCB_v1:ENTRY:WORKING"] == 1
        [portfolio_risk] = family_state["risk_state"]["portfolio"]
        assert portfolio_risk["pending_entry_risk_R"] > 0.0

    assert (live.state_snapshot or {})["family_state"] == (replay.state_snapshot or {})["family_state"]


@pytest.mark.parity_nightly
def test_initial_working_repository_order_changes_family_risk_state() -> None:
    fixture = _fixture_with_initial_working_entry_order(risk_dollars=200.0)
    changed = deepcopy(fixture)
    changed["initial_repository_state"]["orders"][0]["risk_context"]["risk_dollars"] = 300.0

    replay = run_layer3_family_replay_trace(fixture)
    changed_replay = run_layer3_family_replay_trace(changed)
    risk = (replay.state_snapshot or {})["family_state"]["risk_state"]["portfolio"][0]
    changed_risk = (changed_replay.state_snapshot or {})["family_state"]["risk_state"]["portfolio"][0]

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(changed)
    assert risk["pending_entry_risk_R"] != changed_risk["pending_entry_risk_R"]


@pytest.mark.parity_nightly
def test_stock_family_fixture_proves_reduced_family_decision_changes_canonical_output() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    out = replay_runners._ReplayDecisionTimeline(fixture)
    replay_runners._replay_iaric(fixture, out)
    replay_runners._replay_idle_market_children(fixture, out)
    surface = replay_runners._run_family_portfolio_surface(fixture, out)
    decisions = list(surface.get("decisions", []) or [])
    decision = next(item for item in decisions if item["strategy_id"] == "IARIC_v1")

    assert decision["status"] == "reduced"
    assert decision["original_qty"] == 3
    assert decision["approved_qty"] == 1

    replay = replay_runners.run_layer3_family_replay_trace(fixture)
    entry_orders = [
        row
        for row in replay.order_intents
        if row.get("strategy_id") == "IARIC_v1" and row.get("order_role") == "ENTRY"
    ]
    assert entry_orders and entry_orders[0]["qty"] == decision["approved_qty"]
    fills = [
        row
        for row in replay.terminal_events
        if row.get("strategy_id") == "IARIC_v1" and row.get("event_type") == "FILL"
    ]
    ledger = [row for row in replay.trade_ledger if row.get("strategy_id") == "IARIC_v1"]
    assert fills and fills[0]["qty"] == decision["approved_qty"]
    assert ledger and ledger[0]["qty"] == decision["approved_qty"]


@pytest.mark.parity_nightly
def test_momentum_family_fixture_proves_family_decision_changes_canonical_output() -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    out = replay_runners._ReplayDecisionTimeline(fixture)
    replay_runners._replay_nq_regime(fixture, out)
    replay_runners._replay_idle_market_children(fixture, out)
    surface = replay_runners._run_family_portfolio_surface(fixture, out)
    decisions = list(surface.get("decisions", []) or [])
    decision = next(item for item in decisions if item["strategy_id"] == "NQ_REGIME")

    assert decision["status"] == "reduced"
    assert decision["original_qty"] == 2
    assert decision["approved_qty"] == 1

    replay = replay_runners.run_layer3_family_replay_trace(fixture)
    entry_orders = [
        row
        for row in replay.order_intents
        if row.get("strategy_id") == "NQ_REGIME" and row.get("order_role") == "ENTRY"
    ]
    assert entry_orders and entry_orders[0]["qty"] == decision["approved_qty"]
    fills = [
        row
        for row in replay.terminal_events
        if row.get("strategy_id") == "NQ_REGIME" and row.get("event_type") == "FILL"
    ]
    ledger = [row for row in replay.trade_ledger if row.get("strategy_id") == "NQ_REGIME"]
    assert fills and fills[0]["qty"] == decision["approved_qty"]
    assert ledger and ledger[0]["qty"] == decision["approved_qty"]


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_momentum_family_contract_fails_if_surface_returns_accepted_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)

    from tests.integration.parity import replay_runners

    original = replay_runners._run_family_portfolio_surface

    def accepted_only_surface(current_fixture, out):
        surface = deepcopy(original(current_fixture, out))
        for decision in surface.get("decisions", []) or []:
            if decision["strategy_id"] == "NQ_REGIME":
                decision["status"] = "accepted"
                decision["approved_qty"] = decision["original_qty"]
        surface["accepted_quantities"] = {
            decision["strategy_id"]: decision["approved_qty"]
            for decision in surface.get("decisions", []) or []
            if str(decision.get("status", "")).lower() in {"accepted", "reduced"}
        }
        return surface

    monkeypatch.setattr(replay_runners, "_run_family_portfolio_surface", accepted_only_surface)

    live = await run_layer3_family_live_trace(fixture)
    replay = replay_runners.run_layer3_family_replay_trace(fixture)
    assert any(
        row.get("strategy_id") == "NQ_REGIME"
        and row.get("order_role") == "ENTRY"
        and row.get("qty") == 2
        for row in replay.order_intents
    )
    with pytest.raises(AssertionError, match="order_intents"):
        assert_shadow_contract(
            LiveShadowContract(
                surface="momentum_family",
                live=live,
                replay=replay,
            )
        )


@pytest.mark.parity_nightly
def test_layer3_consumed_repository_collision_state_changes_family_decision_and_output() -> None:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["initial_repository_state"]["positions"][0]["net_qty"] = 0

    from tests.integration.parity import replay_runners

    def decisions_for(payload: dict) -> list[dict]:
        out = replay_runners._ReplayDecisionTimeline(payload)
        replay_runners._replay_iaric(payload, out)
        replay_runners._replay_idle_market_children(payload, out)
        return list(replay_runners._run_family_portfolio_surface(payload, out).get("decisions", []) or [])

    base_decision = next(item for item in decisions_for(fixture) if item["strategy_id"] == "IARIC_v1")
    changed_decision = next(item for item in decisions_for(changed) if item["strategy_id"] == "IARIC_v1")
    base_replay = replay_runners.run_layer3_family_replay_trace(fixture)
    changed_replay = replay_runners.run_layer3_family_replay_trace(changed)

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(changed)
    assert base_decision["status"] == "reduced"
    assert base_decision["approved_qty"] == 1
    assert changed_decision["status"] == "accepted"
    assert changed_decision["approved_qty"] == changed_decision["original_qty"] == 3
    assert [
        row["qty"]
        for row in base_replay.order_intents
        if row.get("strategy_id") == "IARIC_v1" and row.get("order_role") == "ENTRY"
    ] != [
        row["qty"]
        for row in changed_replay.order_intents
        if row.get("strategy_id") == "IARIC_v1" and row.get("order_role") == "ENTRY"
    ]


@pytest.mark.parity_nightly
def test_momentum_consumed_repository_state_changes_family_decision_and_output() -> None:
    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    changed["initial_repository_state"]["positions"][0]["net_qty"] = 0

    from tests.integration.parity import replay_runners

    def decision_and_qty(payload: dict) -> tuple[dict, int]:
        out = replay_runners._ReplayDecisionTimeline(payload)
        replay_runners._replay_nq_regime(payload, out)
        replay_runners._replay_idle_market_children(payload, out)
        decision = next(
            item
            for item in replay_runners._run_family_portfolio_surface(payload, out).get("decisions", [])
            if item["strategy_id"] == "NQ_REGIME"
        )
        replay = replay_runners.run_layer3_family_replay_trace(payload)
        [entry_order] = [
            row
            for row in replay.order_intents
            if row.get("strategy_id") == "NQ_REGIME" and row.get("order_role") == "ENTRY"
        ]
        return decision, int(entry_order["qty"])

    base_decision, base_qty = decision_and_qty(fixture)
    changed_decision, changed_qty = decision_and_qty(changed)

    assert fixture_source_fingerprint(fixture) != fixture_source_fingerprint(changed)
    assert base_decision["status"] == "reduced"
    assert base_decision["approved_qty"] == base_qty == 1
    assert changed_decision["status"] == "accepted"
    assert changed_decision["approved_qty"] == changed_qty == changed_decision["original_qty"] == 2


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_path", "idle_children"),
    [
        (
            MOMENTUM_FAMILY_FIXTURE,
            ("NQDTC_v2.1", "VdubusNQ_v4", "DownturnDominator_v1"),
        ),
        (STOCK_FAMILY_FIXTURE, ("ALCB_v1",)),
        (SWING_FAMILY_FIXTURE, ("ATRSS", "AKC_HELIX")),
    ],
)
async def test_configured_idle_children_consume_deterministic_market_inputs(
    fixture_path: Path,
    idle_children: tuple[str, ...],
) -> None:
    fixture = load_parity_fixture(fixture_path)
    live = await run_layer3_family_live_trace(fixture)
    replay = run_layer3_family_replay_trace(fixture)

    for trace in (live, replay):
        strategy_state = (trace.state_snapshot or {}).get("strategy_state", {})
        for strategy_id in idle_children:
            assert strategy_state[strategy_id]["last_bar_ts"] == IDLE_MARKET_INPUT_TIMESTAMP
            assert strategy_state[strategy_id]["last_decision_code"] == "IDLE_MARKET_OBSERVED"
            assert strategy_state[strategy_id]["last_decision_details"]["bar_count"] >= 1
            assert "idle_market_input" not in strategy_state[strategy_id]
            assert not any(
                row.get("strategy_id") == strategy_id
                for row in trace.order_intents
            )


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_idle_child_live_trace_does_not_use_post_cycle_observation_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.parity import live_runners
    from strategies.core import idle_market

    if hasattr(live_runners, "_install_idle_market_decision_hook"):
        monkeypatch.setattr(
            live_runners,
            "_install_idle_market_decision_hook",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("post-cycle idle hook was used")
            ),
        )
    assert not hasattr(live_runners, "_install_idle_market_observation_input")
    assert "_idle_market_observation_input" not in inspect.getsource(live_runners)
    assert "_idle_market_observation_input" not in inspect.getsource(idle_market)

    for fixture_path, idle_children in (
        (MOMENTUM_FAMILY_FIXTURE, ("NQDTC_v2.1", "VdubusNQ_v4", "DownturnDominator_v1")),
        (STOCK_FAMILY_FIXTURE, ("ALCB_v1",)),
        (SWING_FAMILY_FIXTURE, ("ATRSS", "AKC_HELIX")),
    ):
        fixture = load_parity_fixture(fixture_path)
        trace = await live_runners.run_layer3_family_live_trace(fixture)
        strategy_state = (trace.state_snapshot or {}).get("strategy_state", {})
        for strategy_id in idle_children:
            state = strategy_state[strategy_id]
            assert state["last_decision_code"] == "IDLE_MARKET_OBSERVED"
            assert state["last_decision_details"]["bar_count"] >= 1
            assert "last_ohlcv" in state["last_decision_details"]
            assert "idle_market_input" not in state
            assert not any(row.get("strategy_id") == strategy_id for row in trace.order_intents)


@pytest.mark.parity_nightly
@pytest.mark.asyncio
async def test_idle_child_live_cycle_requires_seeded_market_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration.parity import live_idle
    from tests.integration.parity import live_runners

    fixture = load_parity_fixture(MOMENTUM_FAMILY_FIXTURE)

    def no_seed(engine, _market_input):
        original_fetch = engine._fetch_bars

        async def empty_fetch(*_args, **_kwargs):
            return None

        engine._fetch_bars = empty_fetch
        return lambda: setattr(engine, "_fetch_bars", original_fetch)

    monkeypatch.setattr(live_idle, "_seed_nqdtc_market_input", no_seed)

    with pytest.raises(AssertionError, match="did not record IDLE_MARKET_OBSERVED"):
        await live_runners.run_layer3_family_live_trace(fixture)


@pytest.mark.parity_nightly
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_path", "strategy_id", "artifact_key"),
    [
        (MOMENTUM_FAMILY_FIXTURE, "NQDTC_v2.1", "nqdtc"),
        (MOMENTUM_FAMILY_FIXTURE, "VdubusNQ_v4", "vdub"),
        (MOMENTUM_FAMILY_FIXTURE, "DownturnDominator_v1", "downturn"),
        (STOCK_FAMILY_FIXTURE, "ALCB_v1", "alcb"),
        (SWING_FAMILY_FIXTURE, "ATRSS", "atrss"),
        (SWING_FAMILY_FIXTURE, "AKC_HELIX", "akc_helix"),
    ],
)
async def test_idle_child_live_state_consumes_seeded_ohlcv(
    monkeypatch: pytest.MonkeyPatch,
    fixture_path: Path,
    strategy_id: str,
    artifact_key: str,
) -> None:
    from tests.integration.parity import live_runners

    if hasattr(live_runners, "_install_idle_market_decision_hook"):
        monkeypatch.setattr(
            live_runners,
            "_install_idle_market_decision_hook",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("post-cycle idle hook was used")
            ),
        )

    fixture = load_parity_fixture(fixture_path)
    changed = deepcopy(fixture)
    bar = changed["artifacts"][artifact_key]["idle_market_input"]["bars"][0]
    bar["close"] = float(bar["close"]) + 1.25

    base_trace = await live_runners.run_layer3_family_live_trace(fixture)
    changed_trace = await live_runners.run_layer3_family_live_trace(changed)
    base_state = (base_trace.state_snapshot or {})["strategy_state"][strategy_id]
    changed_state = (changed_trace.state_snapshot or {})["strategy_state"][strategy_id]

    assert "idle_market_input" not in base_state
    assert base_state["last_decision_code"] == "IDLE_MARKET_OBSERVED"
    assert changed_state["last_decision_code"] == "IDLE_MARKET_OBSERVED"
    assert base_state["last_decision_details"] != changed_state["last_decision_details"]
    assert changed_state["last_decision_details"]["last_ohlcv"]["close"] == bar["close"]


@pytest.mark.parity_nightly
@pytest.mark.parametrize(
    ("fixture_path", "strategy_id", "artifact_key"),
    [
        (MOMENTUM_FAMILY_FIXTURE, "NQDTC_v2.1", "nqdtc"),
        (MOMENTUM_FAMILY_FIXTURE, "VdubusNQ_v4", "vdub"),
        (MOMENTUM_FAMILY_FIXTURE, "DownturnDominator_v1", "downturn"),
        (STOCK_FAMILY_FIXTURE, "ALCB_v1", "alcb"),
        (SWING_FAMILY_FIXTURE, "ATRSS", "atrss"),
        (SWING_FAMILY_FIXTURE, "AKC_HELIX", "akc_helix"),
    ],
)
def test_idle_child_replay_state_consumes_bar_backed_market_input(
    fixture_path: Path,
    strategy_id: str,
    artifact_key: str,
) -> None:
    fixture = load_parity_fixture(fixture_path)
    changed = deepcopy(fixture)
    bar = changed["artifacts"][artifact_key]["idle_market_input"]["bars"][0]
    bar["close"] = float(bar["close"]) + 1.25

    from tests.integration.parity import replay_runners

    base_state = replay_runners._run_idle_market_core(fixture, strategy_id)
    changed_state = replay_runners._run_idle_market_core(changed, strategy_id)

    assert "idle_market_input" not in base_state
    assert base_state["last_decision_code"] == "IDLE_MARKET_OBSERVED"
    assert changed_state["last_decision_code"] == "IDLE_MARKET_OBSERVED"
    assert base_state["last_decision_details"] != changed_state["last_decision_details"]
    assert changed_state["last_decision_details"]["last_ohlcv"]["close"] == bar["close"]


def _fixture_with_initial_working_entry_order(*, risk_dollars: float = 200.0) -> dict:
    fixture = load_parity_fixture(STOCK_FAMILY_FIXTURE)
    changed = deepcopy(fixture)
    initial = changed.setdefault("initial_repository_state", {})
    initial["orders"] = [
        {
            "oms_order_id": "INIT-WORKING-ALCB",
            "client_order_id": "INIT-WORKING-ALCB",
            "strategy_id": "ALCB_v1",
            "account_id": "ACCT-PARITY",
            "symbol": "MSFT",
            "side": "BUY",
            "qty": 2,
            "order_type": "LIMIT",
            "limit_price": 411.0,
            "tif": "DAY",
            "role": "ENTRY",
            "status": "WORKING",
            "filled_qty": 0.0,
            "remaining_qty": 2.0,
            "avg_fill_price": 0.0,
            "risk_context": {
                "planned_entry_price": 411.0,
                "stop_for_risk": 401.0,
                "risk_budget_tag": "ALCB_v1",
                "risk_dollars": risk_dollars,
                "portfolio_size_mult": 1.0,
                "unit_risk_dollars": 702.0,
            },
        }
    ]
    return changed
