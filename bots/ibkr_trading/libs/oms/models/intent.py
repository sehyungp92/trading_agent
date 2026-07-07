"""Intent models for strategy -> OMS communication."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .order import OMSOrder


class IntentType(Enum):
    NEW_ORDER = "NEW_ORDER"
    PREAPPROVED_ORDER = "PREAPPROVED_ORDER"
    CANCEL_ORDER = "CANCEL_ORDER"
    REPLACE_ORDER = "REPLACE_ORDER"
    FLATTEN = "FLATTEN"


class IntentResult(Enum):
    ACCEPTED = "ACCEPTED"
    DENIED = "DENIED"


@dataclass(frozen=True)
class PreapprovedFamilyDecision:
    """Authoritative family replay approval used to bypass only family risk."""

    candidate_key: str
    family_surface: str
    strategy_id: str
    symbol: str
    side: str
    role: str
    sequence: int
    original_qty: int
    approved_qty: int
    status: str
    reason: str = ""


@dataclass
class Intent:
    intent_type: IntentType
    strategy_id: str
    # For NEW_ORDER: full OMSOrder
    order: Optional["OMSOrder"] = None
    # For CANCEL/REPLACE: target oms_order_id
    target_oms_order_id: Optional[str] = None
    # For REPLACE: new parameters
    new_qty: Optional[int] = None
    new_limit_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    # For FLATTEN: optional instrument filter
    instrument_symbol: Optional[str] = None
    # For PREAPPROVED_ORDER: provenance and approval contract from a family
    # replay/backtest surface. Generic PREAPPROVED_ORDER submissions are denied.
    preapproved_family_decision: Optional[PreapprovedFamilyDecision] = None


@dataclass
class IntentReceipt:
    result: IntentResult
    intent_id: str
    oms_order_id: Optional[str] = None
    denial_reason: Optional[str] = None
