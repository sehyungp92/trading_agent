"""Canonical taxonomy for strategy decision codes.

Engines and coordinators emit a ``last_decision_code`` string at every
heartbeat. Without a shared vocabulary, the dashboard and watchdog cannot
distinguish "evaluated, no signal" from "blocked by risk gate" from
"broker down" — they all collapse into ad-hoc literals.

Use the constants below when setting ``engine._last_decision_code`` so the
watchdog (``apps/watchdog/checks.py``) and dashboard
(``apps/dashboard/src/components/StrategyDiagnostics.tsx``) can render
classification reliably.

JSONB key contract for ``last_decision_details``
------------------------------------------------
The watchdog reads these keys; engines and the OMS-service-sampling code
in family coordinators populate them. Drift here breaks no-trade
observability silently — change with care.

* ``liveness.bars_processed``  (int, monotonic)
* ``liveness.symbol_freshness``  (dict[symbol -> ISO8601 timestamp])
* ``liveness.last_rebalance_date``  (overlay only, "YYYY-MM-DD")
* ``oms_health.submitted`` / ``accepted`` / ``denied`` / ``consecutive_denials``  (int)
* ``ib_farm_status``  (dict[farm -> "OK" | other])

The OMS counters are owned by ``OMSService`` (see
``libs/oms/services/oms_service.py``); coordinators sample them at
heartbeat time. The freshness dicts are owned by individual engines.
"""
from __future__ import annotations

from typing import Final

# Strategy is alive but has nothing to evaluate (engine hasn't seen a bar
# this cycle). Default at construction time.
IDLE: Final[str] = "IDLE"

# Engine processed a bar and chose not to enter. Quiet-day "normal" code.
EVALUATED_NO_SIGNAL: Final[str] = "EVALUATED_NO_SIGNAL"

# Engine generated a signal that will be (or was just) submitted as an intent.
SIGNAL_EMITTED: Final[str] = "SIGNAL_EMITTED"

# Intent submitted to OMS, awaiting acceptance / fill.
ENTRY_SUBMITTED: Final[str] = "ENTRY_SUBMITTED"

# Entry filled; managing position.
ENTRY_FILLED: Final[str] = "ENTRY_FILLED"
MANAGING_POSITION: Final[str] = "MANAGING_POSITION"

# Risk gateway (libs/oms/risk/gateway.py) denied the intent — heat cap,
# daily stop, session block, account gate, holiday, etc.
BLOCKED_RISK_GATE: Final[str] = "BLOCKED_RISK_GATE"

# Cross-strategy portfolio rule (libs/oms/risk/portfolio_rules.py) denied
# the intent — directional cap, symbol collision, drawdown tier, etc.
BLOCKED_PORTFOLIO_RULE: Final[str] = "BLOCKED_PORTFOLIO_RULE"

# Engine refused to evaluate because incoming bar data is stale.
BLOCKED_DATA_STALE: Final[str] = "BLOCKED_DATA_STALE"

# Engine refused to evaluate because the broker (IBKR) is disconnected.
BLOCKED_BROKER_DOWN: Final[str] = "BLOCKED_BROKER_DOWN"

# Strategy passed pre-trade halt guards but is gated by the family halt
# logic (e.g., cooldown after a daily stop).
HALT_GUARDED: Final[str] = "HALT_GUARDED"

# Operator or risk system has put the strategy into stand-down.
STAND_DOWN: Final[str] = "STAND_DOWN"


_KNOWN: Final[frozenset[str]] = frozenset({
    IDLE,
    EVALUATED_NO_SIGNAL,
    SIGNAL_EMITTED,
    ENTRY_SUBMITTED,
    ENTRY_FILLED,
    MANAGING_POSITION,
    BLOCKED_RISK_GATE,
    BLOCKED_PORTFOLIO_RULE,
    BLOCKED_DATA_STALE,
    BLOCKED_BROKER_DOWN,
    HALT_GUARDED,
    STAND_DOWN,
})


def is_known(code: str | None) -> bool:
    """Return True if ``code`` is a recognised decision code.

    Unknown codes are accepted (engines may emit legacy or strategy-specific
    refinements) but ``HeartbeatService`` logs a warning when they appear,
    so drift is visible.
    """
    return code is not None and code in _KNOWN
