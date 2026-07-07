#!/usr/bin/env sh
set -eu

if [ -z "${OLR_KALCB_TRADE_DATE:-}" ]; then
  echo "OLR_KALCB_TRADE_DATE is required, for example 2026-07-06" >&2
  exit 64
fi

exec k-stock-olr-kalcb-runtime preflight \
  --trade-date "${OLR_KALCB_TRADE_DATE}" \
  --mode "${OLR_KALCB_PREFLIGHT_MODE:-artifact_only_stage1}" \
  --assistant-event-data-dir "${ASSISTANT_EVENT_DATA_DIR:-instrumentation/data}" \
  --output-json "${OLR_KALCB_PREFLIGHT_OUTPUT:-data/paper_live/olr_kalcb_preflight.json}"
