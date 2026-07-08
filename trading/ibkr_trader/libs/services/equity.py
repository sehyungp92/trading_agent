"""Live NetLiquidation resolver.

EQUITY-1 / SWING-1 / SWING-2 cluster: a single helper used by every family
coordinator to resolve account equity for live (and paper-fallback) sizing.

Before this module existed, three different code paths tried to read
NetLiquidation:
  - swing/coordinator.py used the literal $100_000 in live mode (no read at all)
  - momentum/coordinator.py and stock/coordinator.py both tried `accountValues()`
    once, then silently fell back to $100_000 with a `WARNING` log on any
    exception or empty list

The fallback is dangerous: account values populate asynchronously after the
IB session connects, and a startup race can produce an empty `accountValues()`
even on a healthy account. The result was that every order in the racing
session would size against $100k regardless of the real account NAV.

This helper polls until either a usable USD NetLiquidation arrives for the
configured account or the timeout expires; in live mode the callers raise on
the timeout rather than continuing with placeholder equity.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def resolve_live_nlv(
    session,
    *,
    account_id: Optional[str] = None,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.5,
) -> float:
    """Resolve USD NetLiquidation for the configured account.

    Args:
        session: a UnifiedIBSession (with `.ib`).
        account_id: the configured IB account; if None, falls back to the
            first managed account. Live coordinators should always pass the
            configured `IBKRConfig.profile.account_id`.
        timeout_s: bounded wait for `accountValues()` to populate.
        poll_interval_s: poll cadence inside the wait.

    Returns the NLV as a float. Raises RuntimeError on:
      - no IB session (or disconnected)
      - no managed accounts
      - configured account not found in managed accounts
      - timeout waiting for NetLiquidation to appear
      - non-positive NLV
    """
    if session is None or not getattr(session, "ib", None):
        raise RuntimeError("resolve_live_nlv: no IB session available")

    ib = session.ib
    if not ib.isConnected():
        raise RuntimeError("resolve_live_nlv: IB session is not connected")

    # Resolve which account we expect to read.
    accounts = list(ib.managedAccounts() or [])
    if not accounts:
        raise RuntimeError("resolve_live_nlv: no managed accounts available")
    target = account_id or accounts[0]
    if account_id and account_id not in accounts:
        raise RuntimeError(
            f"resolve_live_nlv: configured account {account_id!r} not in "
            f"managedAccounts={accounts}"
        )

    deadline = asyncio.get_event_loop().time() + timeout_s
    nlv: Optional[float] = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            for item in ib.accountValues():
                if (
                    item.tag == "NetLiquidation"
                    and item.currency == "USD"
                    and item.account == target
                ):
                    try:
                        nlv = float(item.value)
                    except (TypeError, ValueError):
                        nlv = None
                    break
        except Exception as exc:
            # Don't burn the budget on transient errors — log once and continue
            logger.debug("resolve_live_nlv accountValues read failed: %s", exc)

        if nlv is not None and nlv > 0:
            logger.info(
                "Live NetLiquidation resolved: $%.2f (account=%s)", nlv, target,
            )
            return nlv
        await asyncio.sleep(poll_interval_s)

    raise RuntimeError(
        f"resolve_live_nlv: timed out after {timeout_s:.1f}s waiting for "
        f"USD NetLiquidation on account {target}"
    )
