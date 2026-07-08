"""IB error code classification."""
from .models.types import RejectCategory

# Reference: https://interactivebrokers.github.io/tws-api/message_codes.html
_ERROR_MAP: dict[int, RejectCategory] = {
    103: RejectCategory.DUPLICATE,
    104: RejectCategory.PERMISSIONS,
    110: RejectCategory.INVALID_PRICE,
    135: RejectCategory.PERMISSIONS,
    161: RejectCategory.PACING,
    200: RejectCategory.CONTRACT,
    201: RejectCategory.RISK,
    202: RejectCategory.UNKNOWN,
    399: RejectCategory.PACING,
    10147: RejectCategory.PACING,
    # Farm connectivity (reqId=-1, never reach OMS reject logic)
    2103: RejectCategory.TRANSIENT,  # Market data farm connection is broken
    2104: RejectCategory.TRANSIENT,  # Market data farm connection is OK
    2108: RejectCategory.TRANSIENT,  # Market data farm connection is inactive
    2119: RejectCategory.TRANSIENT,  # Market data farm is connecting
}


def classify_error(code: int, message: str) -> tuple[RejectCategory, bool]:
    """Returns (category, is_retryable)."""
    cat = _ERROR_MAP.get(code, RejectCategory.UNKNOWN)
    retryable = cat in {RejectCategory.PACING, RejectCategory.TRANSIENT}
    return cat, retryable
