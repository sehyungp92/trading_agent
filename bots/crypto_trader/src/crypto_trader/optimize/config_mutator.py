"""Config mutation via dot-notation paths and dataclasses.replace()."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any


def apply_mutations(
    base_config: Any,
    mutations: dict[str, Any],
) -> Any:
    """Apply dot-notation mutations to a dataclass config, returning a new copy.

    Works with any nested-dataclass config (MomentumConfig, TrendConfig, etc.).
    Example mutations: {"entry.entry_on_break": True, "stops.atr_buffer_mult": 0.4}
    """
    if not mutations:
        return copy.deepcopy(base_config)

    # Group mutations by section
    section_updates: dict[str, dict[str, Any]] = {}
    for key, value in mutations.items():
        parts = key.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Mutation key must be 'section.field', got: {key!r}")
        section, field_name = parts
        if not hasattr(base_config, section):
            raise ValueError(f"Unknown config section: {section!r}")
        section_updates.setdefault(section, {})[field_name] = value

    # Build replacement kwargs
    replace_kwargs: dict[str, Any] = {}
    for section, fields in section_updates.items():
        current_sub = getattr(base_config, section)
        # Validate field names
        for field_name in fields:
            if not hasattr(current_sub, field_name):
                raise ValueError(
                    f"Unknown field {field_name!r} in section {section!r}"
                )
        replace_kwargs[section] = replace(current_sub, **fields)

    return replace(base_config, **replace_kwargs)


def merge_mutations(
    base: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Merge two mutation dicts. Overlay values win on conflict."""
    merged = dict(base)
    merged.update(overlay)
    return merged
