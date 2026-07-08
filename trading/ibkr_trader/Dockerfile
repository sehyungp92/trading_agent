FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY packages/trading_contracts/pyproject.toml /app/packages/trading_contracts/
COPY packages/trading_contracts/src /app/packages/trading_contracts/src
RUN uv pip install --system /app/packages/trading_contracts

COPY trading/ibkr_trader/pyproject.toml /app/trading/ibkr_trader/
COPY trading/ibkr_trader/apps/__init__.py /app/trading/ibkr_trader/apps/
COPY trading/ibkr_trader/apps/runtime /app/trading/ibkr_trader/apps/runtime
COPY trading/ibkr_trader/apps/watchdog /app/trading/ibkr_trader/apps/watchdog
COPY trading/ibkr_trader/config /app/trading/ibkr_trader/config
COPY trading/ibkr_trader/libs /app/trading/ibkr_trader/libs
COPY trading/ibkr_trader/regime /app/trading/ibkr_trader/regime
COPY trading/ibkr_trader/src /app/trading/ibkr_trader/src
COPY trading/ibkr_trader/strategies /app/trading/ibkr_trader/strategies
RUN uv pip install --system /app/trading/ibkr_trader

WORKDIR /app/trading/ibkr_trader
CMD ["python", "-m", "apps.runtime.cli", "--help"]
