FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY bots/ibkr_trading/pyproject.toml /app/bots/ibkr_trading/
COPY bots/ibkr_trading/apps /app/bots/ibkr_trading/apps
COPY bots/ibkr_trading/config /app/bots/ibkr_trading/config
COPY bots/ibkr_trading/libs /app/bots/ibkr_trading/libs
COPY bots/ibkr_trading/regime /app/bots/ibkr_trading/regime
COPY bots/ibkr_trading/src /app/bots/ibkr_trading/src
COPY bots/ibkr_trading/strategies /app/bots/ibkr_trading/strategies
RUN uv pip install --system /app/bots/ibkr_trading

WORKDIR /app/bots/ibkr_trading
CMD ["python", "-m", "apps.runtime.cli", "--help"]
