"""Objective gate helpers."""

from __future__ import annotations


def enforce_score_component_cap(score_components: list[str], cap: int) -> None:
    if len(score_components) > cap:
        raise ValueError(f"score component cap exceeded: {len(score_components)} > {cap}")
