FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY packages/trading_contracts/pyproject.toml /app/packages/trading_contracts/
COPY packages/trading_contracts/src /app/packages/trading_contracts/src
RUN uv pip install --system /app/packages/trading_contracts

COPY trading/crypto_trader/pyproject.toml /app/trading/crypto_trader/
COPY trading/crypto_trader/config /app/trading/crypto_trader/config
COPY trading/crypto_trader/infra /app/trading/crypto_trader/infra
COPY trading/crypto_trader/src /app/trading/crypto_trader/src
RUN uv pip install --system "/app/trading/crypto_trader[live]"

WORKDIR /app/trading/crypto_trader
CMD ["crypto-trader", "--help"]
