from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

DataT = TypeVar("DataT")


@dataclass(frozen=True)
class ReplayBundle(Generic[DataT]):
    data: DataT
    cache_key: str
    cache_source_fingerprint: str
