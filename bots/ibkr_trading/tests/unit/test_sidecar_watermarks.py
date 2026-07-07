from __future__ import annotations

import json
import os

import pytest

from strategies.momentum.instrumentation.src.sidecar import Sidecar as MomentumSidecar
from strategies.swing.instrumentation.src.sidecar import Sidecar as SwingSidecar


def _config(tmp_path, bot_id: str) -> dict:
    return {
        "bot_id": bot_id,
        "data_dir": str(tmp_path),
        "sidecar": {
            "relay_url": "http://relay.local/events",
            "batch_size": 10,
        },
    }


@pytest.mark.parametrize("sidecar_cls,bot_id", [
    (MomentumSidecar, "momentum_nq_01"),
    (SwingSidecar, "swing_multi_01"),
])
def test_sidecar_resends_mutated_jsonl_line(tmp_path, sidecar_cls, bot_id):
    missed_dir = tmp_path / "missed"
    missed_dir.mkdir()
    path = missed_dir / "events.jsonl"
    path.write_text(json.dumps({"trade_id": "t1", "reason": "old"}) + "\n")

    sidecar = sidecar_cls(_config(tmp_path, bot_id))
    sent_batches: list[list[dict]] = []
    sidecar._send_batch = lambda events: sent_batches.append(list(events)) or True

    sidecar.run_once()
    path.write_text(json.dumps({"trade_id": "t1", "reason": "new"}) + "\n")
    sidecar.run_once()

    assert len(sent_batches) == 2
    assert sent_batches[0][0]["event_id"] != sent_batches[1][0]["event_id"]
    watermark = sidecar.watermarks[str(path)]
    assert watermark["kind"] == "jsonl"
    assert len(watermark["line_hashes"]) == 1


@pytest.mark.parametrize("sidecar_cls,bot_id", [
    (MomentumSidecar, "momentum_nq_01"),
    (SwingSidecar, "swing_multi_01"),
])
def test_sidecar_resends_mutated_daily_json_with_preserved_mtime(tmp_path, sidecar_cls, bot_id):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    path = daily_dir / "daily_2026-05-10.json"
    path.write_text(json.dumps({"date": "2026-05-10", "value": 1}))
    first_stat = path.stat()

    sidecar = sidecar_cls(_config(tmp_path, bot_id))
    sent_batches: list[list[dict]] = []
    sidecar._send_batch = lambda events: sent_batches.append(list(events)) or True

    sidecar.run_once()
    path.write_text(json.dumps({"date": "2026-05-10", "value": 2}))
    os.utime(path, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
    sidecar.run_once()

    assert len(sent_batches) == 2
    watermark = sidecar.watermarks[str(path)]
    assert watermark["kind"] == "json"
    assert watermark["hash"]
