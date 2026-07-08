from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module

import pytest

from libs.instrumentation.lineage import LineageContext


@pytest.mark.parametrize(
    "module_name",
    [
        "strategies.stock.instrumentation.src.event_metadata",
        "strategies.momentum.instrumentation.src.event_metadata",
        "strategies.swing.instrumentation.src.event_metadata",
    ],
)
def test_event_metadata_v2_fields_round_trip(module_name: str) -> None:
    metadata_module = import_module(module_name)
    exchange_ts = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)

    meta = metadata_module.create_event_metadata(
        bot_id="bot1",
        event_type="trade",
        payload_key="trade_1:exit",
        exchange_timestamp=exchange_ts,
        data_source_id="ibkr_execution",
        bar_id="2026-05-31T12:00Z_1h",
        schema_version="event_metadata_v2",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        trace_id="trace_signal_1",
    )

    assert meta.event_id == metadata_module.compute_event_id(
        "bot1",
        exchange_ts.isoformat(),
        "trade",
        "trade_1:exit",
    )
    assert meta.event_type == "trade"
    assert meta.payload_key == "trade_1:exit"
    assert meta.schema_version == "event_metadata_v2"
    assert meta.strategy_id == "IARIC_v1"
    assert meta.family_id == "stock"
    assert meta.portfolio_id == "paper_default"
    assert meta.trace_id == "trace_signal_1"
    assert meta.bar_id == "2026-05-31T12:00Z_1h"

    as_dict = meta.to_dict()
    assert as_dict["event_type"] == "trade"
    assert as_dict["payload_key"] == "trade_1:exit"
    assert as_dict["schema_version"] == "event_metadata_v2"
    assert as_dict["trace_id"] == "trace_signal_1"


@pytest.mark.parametrize(
    "module_name",
    [
        "strategies.stock.instrumentation.src.event_metadata",
        "strategies.momentum.instrumentation.src.event_metadata",
        "strategies.swing.instrumentation.src.event_metadata",
    ],
)
def test_event_metadata_v2_defaults_preserve_legacy_id_inputs(module_name: str) -> None:
    metadata_module = import_module(module_name)
    exchange_ts = datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc)

    meta = metadata_module.create_event_metadata(
        "bot1",
        "order",
        "order_1",
        exchange_ts,
        "ibkr_execution",
    )

    assert meta.event_id == metadata_module.compute_event_id(
        "bot1",
        exchange_ts.isoformat(),
        "order",
        "order_1",
    )
    assert meta.schema_version == "event_metadata_v2"
    assert meta.trace_id.startswith("trace_")
    assert meta.event_type == "order"
    assert meta.payload_key == "order_1"


@pytest.mark.parametrize(
    "module_name",
    [
        "strategies.stock.instrumentation.src.event_metadata",
        "strategies.momentum.instrumentation.src.event_metadata",
        "strategies.swing.instrumentation.src.event_metadata",
    ],
)
def test_event_metadata_v2_can_hydrate_from_lineage(module_name: str) -> None:
    metadata_module = import_module(module_name)
    lineage = LineageContext(
        bot_id="bot1",
        strategy_id="IARIC_v1",
        family_id="stock",
        portfolio_id="paper_default",
        trace_id="trace_lineage",
    )

    meta = metadata_module.create_event_metadata(
        "bot1",
        "filter_decision",
        "filter_1",
        datetime(2026, 5, 31, 12, 0, tzinfo=timezone.utc),
        "market_data",
        lineage=lineage,
    )

    assert meta.strategy_id == "IARIC_v1"
    assert meta.family_id == "stock"
    assert meta.portfolio_id == "paper_default"
    assert meta.trace_id == "trace_lineage"
