"""Reconciliation components."""
from .discrepancy_policy import DiscrepancyAction, DiscrepancyPolicy
from .sync import Discrepancy, ReconcilerSync

__all__ = [
    "DiscrepancyAction",
    "DiscrepancyPolicy",
    "Discrepancy",
    "ReconcilerSync",
    "SnapshotFetcher",
]


def __getattr__(name: str):
    if name == "SnapshotFetcher":
        from .snapshots import SnapshotFetcher
        return SnapshotFetcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
