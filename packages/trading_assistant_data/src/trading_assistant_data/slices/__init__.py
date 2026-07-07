"""Data slice product interfaces."""

from trading_assistant_data.slices.authority import (
    SliceAuthorityStatus,
    authority_status_for_manifest,
    is_authoritative_slice_manifest,
)
from trading_assistant_data.slices.coverage import SliceCoverageReport
from trading_assistant_data.slices.product import (
    CanonicalSlice,
    DataSliceProduct,
    SliceRequest,
    SliceWrite,
    timestamps_sorted_unique_utc,
)
from trading_assistant_data.slices.writer import update_slice_index

__all__ = [
    "CanonicalSlice",
    "DataSliceProduct",
    "SliceAuthorityStatus",
    "SliceCoverageReport",
    "SliceRequest",
    "SliceWrite",
    "authority_status_for_manifest",
    "is_authoritative_slice_manifest",
    "timestamps_sorted_unique_utc",
    "update_slice_index",
]
