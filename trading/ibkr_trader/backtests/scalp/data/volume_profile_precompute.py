from __future__ import annotations

from dataclasses import dataclass, field

from strategies.scalp._shared.levels import IVBLevels
from strategies.scalp._shared.volume_profile import compute_volume_profile

from .preprocessing import NumpyBars


@dataclass
class IvbProfileCache:
    levels_by_date: dict[str, IVBLevels] = field(default_factory=dict)


def precompute_ivb_levels(bars: NumpyBars) -> IvbProfileCache:
    from strategies.scalp._shared.time_utils import to_et

    by_date: dict[str, dict[str, list[float]]] = {}
    for idx, raw_ts in enumerate(bars.times):
        ts = _np_to_datetime(raw_ts)
        et = to_et(ts)
        if not (et.hour == 9 and et.minute >= 30 or et.hour == 10 and et.minute == 0):
            continue
        if et.hour == 10 and et.minute >= 0:
            continue
        key = et.date().isoformat()
        bucket = by_date.setdefault(key, {"highs": [], "lows": [], "prices": [], "volumes": []})
        bucket["highs"].append(float(bars.highs[idx]))
        bucket["lows"].append(float(bars.lows[idx]))
        bucket["prices"].append(float(bars.closes[idx]))
        bucket["volumes"].append(float(bars.volumes[idx]))
    levels: dict[str, IVBLevels] = {}
    for key, bucket in by_date.items():
        if not bucket["highs"]:
            continue
        profile = compute_volume_profile(bucket["prices"], bucket["volumes"])
        levels[key] = IVBLevels.from_bounds(
            max(bucket["highs"]),
            min(bucket["lows"]),
            poc=profile.poc,
            vah=profile.vah,
            val=profile.val,
        )
    return IvbProfileCache(levels_by_date=levels)


def _np_to_datetime(value):
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()

