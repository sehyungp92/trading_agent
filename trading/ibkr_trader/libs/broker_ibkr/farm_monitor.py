"""Farm connectivity monitor that detects IBKR farm state transitions."""
from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)

_FARM_CODES = {2103, 2104, 2108, 2119}
_FARM_NAME_RE = re.compile(r":(\w+)\s*$")

_CODE_TO_STATUS = {
    2103: "BROKEN",
    2104: "OK",
    2108: "INACTIVE",
    2119: "CONNECTING",
}


class FarmStatus(Enum):
    OK = "OK"
    BROKEN = "BROKEN"
    INACTIVE = "INACTIVE"
    CONNECTING = "CONNECTING"


class FarmMonitor:
    """Tracks per-farm connectivity status and fires callbacks on recovery."""

    def __init__(self, ib: object) -> None:
        self._ib = ib
        self._farm_status: dict[str, FarmStatus] = {}
        self.on_farm_recovered: Callable[[str], None] | None = None

    def start(self) -> None:
        self._ib.errorEvent += self._on_error  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._ib.errorEvent -= self._on_error  # type: ignore[attr-defined]

    def get_status(self, farm_name: str) -> FarmStatus | None:
        return self._farm_status.get(farm_name)

    def all_statuses(self) -> dict[str, str]:
        """Return {farm_name: status_value} for all known farms."""
        return {name: status.value for name, status in self._farm_status.items()}

    def _on_error(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        contract: object = None,
    ) -> None:
        if errorCode not in _FARM_CODES:
            return

        farm_name = self._parse_farm_name(errorString)
        if not farm_name:
            return

        new_status = FarmStatus(_CODE_TO_STATUS[errorCode])
        old_status = self._farm_status.get(farm_name)

        if old_status == new_status:
            return

        self._farm_status[farm_name] = new_status
        logger.info("Farm %s: %s -> %s", farm_name, old_status, new_status.value)

        if new_status == FarmStatus.OK and self.on_farm_recovered:
            self.on_farm_recovered(farm_name)

    @staticmethod
    def _parse_farm_name(msg: str) -> str:
        match = _FARM_NAME_RE.search(msg)
        return match.group(1) if match else ""

