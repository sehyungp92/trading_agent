from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

_OPERATOR_SCRIPT = Path("scripts") / "run_olr_kalcb_runtime_session.py"
_EXECUTION_MODES = {"dry_run", "paper", "live"}
_PREFLIGHT_MODES = {"artifact_only", "artifact_only_stage1"}
_MARKET_DATA_SOURCES = {"auto", "external_completed_bars", "kis_websocket"}


class LaunchConfigError(ValueError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = list(argv) if argv is not None else build_watch_args(os.environ)
    except LaunchConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 64
    return int(_load_operator().main(args))


def build_watch_args(env: Mapping[str, str]) -> list[str]:
    trade_date = _required(env, "OLR_KALCB_TRADE_DATE", "OLR_KALCB_TRADE_DATE must be set before starting the runtime service")
    mode = _env(env, "OLR_KALCB_RUNTIME_MODE", "dry_run")
    if mode in _PREFLIGHT_MODES:
        raise LaunchConfigError(
            f"OLR_KALCB_RUNTIME_MODE={mode} is preflight-only; run the artifact preflight gate before starting dry_run, paper, or live."
        )
    if mode not in _EXECUTION_MODES:
        raise LaunchConfigError(f"Unsupported OLR_KALCB_RUNTIME_MODE={mode}; expected dry_run, paper, or live.")

    market_data_source = _env(env, "OLR_KALCB_MARKET_DATA_SOURCE", "auto")
    if market_data_source not in _MARKET_DATA_SOURCES:
        raise LaunchConfigError(
            f"Unsupported OLR_KALCB_MARKET_DATA_SOURCE={market_data_source}; expected auto, external_completed_bars, or kis_websocket."
        )
    effective_source = _effective_market_data_source(market_data_source, mode)
    bars_parquet = _env(env, "OLR_KALCB_BARS_PARQUET", "")
    if effective_source == "external_completed_bars" and not bars_parquet:
        raise LaunchConfigError(
            "OLR_KALCB_BARS_PARQUET must point to completed 5m bars when using external_completed_bars"
        )

    session_root = _env(env, "OLR_KALCB_SESSION_ROOT", f"data/paper_live/olr_kalcb/{trade_date}")
    args = [
        "watch-bars",
        "--trade-date",
        trade_date,
        "--mode",
        mode,
        "--market-data-source",
        market_data_source,
        "--session-root",
        session_root,
        "--health-checks-json",
        _env(env, "OLR_KALCB_HEALTH_CHECKS_JSON", _session_file(session_root, "health_checks.json")),
        "--account-state-json",
        _env(env, "OLR_KALCB_ACCOUNT_STATE_JSON", _session_file(session_root, "account_state.json")),
        "--positions-json",
        _env(env, "OLR_KALCB_POSITIONS_JSON", _session_file(session_root, "positions.json")),
        "--poll-seconds",
        _env(env, "OLR_KALCB_POLL_SECONDS", "15"),
        "--oms-url",
        _env(env, "OMS_URL", "http://oms:8000"),
        "--assistant-event-data-dir",
        _env(env, "ASSISTANT_EVENT_DATA_DIR", "instrumentation/data"),
    ]
    _add_optional(args, "--bars-parquet", bars_parquet)
    _add_optional(args, "--kis-ws-url", _env(env, "OLR_KALCB_KIS_WS_URL", ""))
    _add_optional(args, "--ws-ledger-path", _env(env, "OLR_KALCB_WS_LEDGER_PATH", _env(env, "KIS_WS_LEDGER_PATH", "")))
    _add_optional(args, "--output-json", _env(env, "OLR_KALCB_OUTPUT_JSON", ""))
    _add_optional(args, "--deployment-metadata-json", _env(env, "OLR_KALCB_DEPLOYMENT_METADATA_PATH", ""))
    _add_optional(args, "--strategy-plugin-contract", _env(env, "OLR_KALCB_STRATEGY_PLUGIN_CONTRACT", ""))
    _add_optional(args, "--deployment-metadata-environment", _env(env, "OLR_KALCB_DEPLOYMENT_METADATA_ENV", "paper_vps"))
    if _env(env, "OLR_KALCB_ONCE", "0") == "1":
        args.extend(["--once", "--close-session-after-once"])
    return args


def _load_operator() -> ModuleType:
    candidates = (Path.cwd() / _OPERATOR_SCRIPT, Path(__file__).resolve().parents[2] / _OPERATOR_SCRIPT)
    script = next((path for path in candidates if path.is_file()), candidates[0])
    spec = importlib.util.spec_from_file_location("_k_stock_olr_kalcb_runtime_operator", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load OLR/KALCB runtime operator: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required(env: Mapping[str, str], name: str, message: str) -> str:
    value = _env(env, name, "")
    if not value:
        raise LaunchConfigError(message)
    return value


def _env(env: Mapping[str, str], name: str, default: str) -> str:
    return str(env.get(name, default) or "").strip()


def _effective_market_data_source(source: str, mode: str) -> str:
    if source != "auto":
        return source
    return "kis_websocket" if mode in {"paper", "live"} else "external_completed_bars"


def _session_file(session_root: str, filename: str) -> str:
    return f"{session_root.rstrip('/')}/{filename}"


def _add_optional(args: list[str], flag: str, value: str) -> None:
    if value:
        args.extend([flag, value])


if __name__ == "__main__":
    raise SystemExit(main())
