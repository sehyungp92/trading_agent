#!/bin/bash
set -euo pipefail
cd /opt/trading
source venv/bin/activate

# Load HMAC secrets from file into RELAY_SHARED_SECRETS env var
# (the relay app reads RELAY_SHARED_SECRETS, not RELAY_SECRETS_FILE)
SECRETS_FILE="${RELAY_SECRETS_FILE:-/opt/trading/config/relay_secrets.json}"
if [ -f "$SECRETS_FILE" ]; then
    export RELAY_SHARED_SECRETS="$(cat "$SECRETS_FILE")"
fi

export RELAY_DB_PATH="${RELAY_DB_PATH:-/opt/trading/data/relay.db}"
mkdir -p "$(dirname "$RELAY_DB_PATH")"

exec uvicorn apps.relay.app:app \
  --host 127.0.0.1 \
  --port 8001 \
  --workers 1 \
  --log-level info
