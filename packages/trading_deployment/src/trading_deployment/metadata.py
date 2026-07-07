from __future__ import annotations

from hashlib import sha256
from typing import Mapping

FAIL_CLOSED_CHECKS = (
    "image_version",
    "materialized_config_hash",
    "promotion_hash",
    "strategy_plugin_contract_hash",
)


def combined_artifact_hash(hashes: Mapping[str, str]) -> str:
    digest = sha256()
    for path, value in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()
