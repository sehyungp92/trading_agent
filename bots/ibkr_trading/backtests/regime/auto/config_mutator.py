"""Config mutation for regime greedy optimization."""
from __future__ import annotations

from dataclasses import replace

from regime.config import MetaConfig


def mutate_meta_config(base: MetaConfig, overrides: dict) -> MetaConfig:
    """Apply parameter overrides to a MetaConfig via dataclasses.replace."""
    return replace(base, **overrides)
