from __future__ import annotations

from .models import IvbSignalScore


def score_signal(**components: float | None) -> IvbSignalScore:
    available = {name: value for name, value in components.items() if value is not None}
    if not available:
        return IvbSignalScore(total=0.0)
    total = sum(max(0.0, min(1.0, float(value))) for value in available.values())
    score = 100.0 * total / len(available)
    footprint_names = {"absorption_quality", "delta_confirmation"}
    footprint_available = any(name in available for name in footprint_names)
    size_multiplier = 0.5 + min(score, 100.0) / 200.0
    cleaned_names = tuple(name.replace("_quality", "").replace("_confirmation", "") for name in available)
    return IvbSignalScore(
        total=round(score, 6),
        size_multiplier=size_multiplier,
        footprint_available=footprint_available,
        available_components=cleaned_names,
    )
