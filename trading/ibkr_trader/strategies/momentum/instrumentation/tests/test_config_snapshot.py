import enum
import types
from strategies.momentum.instrumentation.src.config_snapshot import snapshot_config_module


def test_snapshot_captures_uppercase_constants():
    mod = types.ModuleType("fake_config")
    mod.STOP_MULT = 2.5
    mod.DOW_BLOCKED = {0, 2}
    mod.SESSION_SIZE_MULT = {"RTH": 1.0, "ETH": 0.6}
    mod._private = "hidden"
    mod.some_function = lambda: None

    result = snapshot_config_module(mod)
    assert result["STOP_MULT"] == 2.5
    assert result["DOW_BLOCKED"] == [0, 2]  # sets -> sorted lists
    assert result["SESSION_SIZE_MULT"] == {"RTH": 1.0, "ETH": 0.6}
    assert "_private" not in result
    assert "some_function" not in result


def test_snapshot_handles_enum_values():
    mod = types.ModuleType("fake_config")

    class Side(enum.Enum):
        LONG = "LONG"

    mod.DEFAULT_SIDE = Side.LONG
    result = snapshot_config_module(mod)
    assert result["DEFAULT_SIDE"] == "LONG"
