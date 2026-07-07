#!/usr/bin/env sh
set -eu

if [ -z "${OLR_KALCB_TRADE_DATE:-}" ]; then
  echo "OLR_KALCB_TRADE_DATE is required, for example 2026-07-06" >&2
  exit 64
fi

set -- k-stock-olr-kalcb-runtime watch-bars \
  --trade-date "${OLR_KALCB_TRADE_DATE}" \
  --mode "${OLR_KALCB_RUNTIME_MODE:-dry_run}" \
  --market-data-source "${OLR_KALCB_MARKET_DATA_SOURCE:-auto}" \
  --poll-seconds "${OLR_KALCB_POLL_SECONDS:-15}" \
  --oms-url "${OMS_URL:-http://oms:8000}" \
  --assistant-event-data-dir "${ASSISTANT_EVENT_DATA_DIR:-instrumentation/data}"

if [ -n "${OLR_KALCB_BARS_PARQUET:-}" ]; then
  set -- "$@" --bars-parquet "${OLR_KALCB_BARS_PARQUET}"
fi

if [ "${OLR_KALCB_ONCE:-0}" = "1" ]; then
  set -- "$@" --once --close-session-after-once
fi

exec "$@"
