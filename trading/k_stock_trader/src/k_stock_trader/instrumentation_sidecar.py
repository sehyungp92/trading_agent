from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any

from instrumentation.src.sidecar import Sidecar


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    sidecar = Sidecar(_config(args))
    if args.once:
        sidecar.run_once()
        return 0

    logging.getLogger(__name__).info(
        "K-stock instrumentation sidecar started for %s", args.data_dir
    )
    try:
        while True:
            sidecar.run_once()
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("K-stock instrumentation sidecar stopped")
        return 0


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bot_id": args.bot_id,
        "data_dir": str(args.data_dir),
        "sidecar": {
            "relay_url": args.relay_url,
            "hmac_secret_env": args.hmac_secret_env,
            "batch_size": args.batch_size,
            "retry_max": args.retry_max,
            "retry_backoff_base_seconds": args.retry_backoff_base_seconds,
            "buffer_dir": str(args.buffer_dir),
            "poll_interval_seconds": args.poll_seconds,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward K-stock instrumentation events to the assistant relay.")
    parser.add_argument("--bot-id", default=os.environ.get("K_STOCK_SIDECAR_BOT_ID", "k_stock_trader"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("ASSISTANT_EVENT_DATA_DIR", "instrumentation/data")),
    )
    parser.add_argument(
        "--relay-url",
        default=os.environ.get("SIDECAR_RELAY_URL") or os.environ.get("RELAY_URL", ""),
    )
    parser.add_argument(
        "--hmac-secret-env",
        default=os.environ.get("K_STOCK_SIDECAR_HMAC_SECRET_ENV", "INSTRUMENTATION_HMAC_SECRET"),
    )
    parser.add_argument("--batch-size", type=int, default=_int_env("K_STOCK_SIDECAR_BATCH_SIZE", 50))
    parser.add_argument("--retry-max", type=int, default=_int_env("K_STOCK_SIDECAR_RETRY_MAX", 5))
    parser.add_argument(
        "--retry-backoff-base-seconds",
        type=int,
        default=_int_env("K_STOCK_SIDECAR_RETRY_BACKOFF_SECONDS", 10),
    )
    parser.add_argument(
        "--buffer-dir",
        type=Path,
        default=Path(os.environ.get("K_STOCK_SIDECAR_BUFFER_DIR", "instrumentation/data/.sidecar_buffer")),
    )
    parser.add_argument("--poll-seconds", type=int, default=_int_env("K_STOCK_SIDECAR_POLL_SECONDS", 15))
    parser.add_argument("--once", action="store_true", default=_bool_env("K_STOCK_SIDECAR_ONCE"))
    parser.add_argument("--log-level", default=os.environ.get("K_STOCK_SIDECAR_LOG_LEVEL", "INFO"))
    return parser


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
