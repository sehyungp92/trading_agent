"""Per-instrument/venue allowed order flags."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AllowedOrderFlags:
    """Per-instrument/venue allowed flags."""

    allow_outside_rth: bool = False
    allow_oca: bool = True
    allow_bracket: bool = True
    max_order_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"LIMIT", "MARKET", "STOP", "STOP_LIMIT"})
    )


EXCHANGE_FLAGS: dict[str, AllowedOrderFlags] = {
    "CME": AllowedOrderFlags(allow_outside_rth=True),
    "COMEX": AllowedOrderFlags(allow_outside_rth=True),
    "NYMEX": AllowedOrderFlags(allow_outside_rth=True),
    "CBOT": AllowedOrderFlags(allow_outside_rth=True),
}
