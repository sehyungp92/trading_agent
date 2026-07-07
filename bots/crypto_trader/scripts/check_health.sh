#!/bin/bash
# Check if the trader container is healthy and running

CONTAINER="crypto-trader"
COMPOSE_DIR="/home/trader/projects/crypto_trader"

STATUS=$(docker inspect --format='{{.State.Status}}' $CONTAINER 2>/dev/null)
HEALTH=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER 2>/dev/null)

if [ "$STATUS" != "running" ]; then
    echo "ALERT: Container $CONTAINER is $STATUS"
    # Optional: send notification (email, Telegram, Discord webhook)
    # curl -s -X POST "https://api.telegram.org/bot$BOT_TOKEN/sendMessage" \
    #   -d chat_id=$CHAT_ID -d text="Crypto trader is DOWN: $STATUS"
    exit 1
fi

if [ "$HEALTH" = "unhealthy" ]; then
    echo "ALERT: Container $CONTAINER is unhealthy"
    docker compose -f "$COMPOSE_DIR/docker-compose.yml" restart trader
    exit 1
fi

# Check instrumentation health report assessment
ASSESSMENT=$(docker compose -f "$COMPOSE_DIR/docker-compose.yml" exec -T trader \
    cat data/live_state/health_reports.jsonl 2>/dev/null | tail -1 | \
    python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('assessment','unknown'))" 2>/dev/null)

if [ "$ASSESSMENT" = "critical" ]; then
    echo "ALERT: Trading system assessment is CRITICAL"
    # Optional: send notification
    exit 1
elif [ "$ASSESSMENT" = "degraded" ]; then
    echo "WARNING: Trading system assessment is DEGRADED"
fi

echo "OK: $CONTAINER is $STATUS ($HEALTH) — assessment: ${ASSESSMENT:-n/a}"
