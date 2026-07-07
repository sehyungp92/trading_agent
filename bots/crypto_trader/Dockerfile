FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY bots/crypto_trader/pyproject.toml /app/bots/crypto_trader/
COPY bots/crypto_trader/config /app/bots/crypto_trader/config
COPY bots/crypto_trader/infra /app/bots/crypto_trader/infra
COPY bots/crypto_trader/src /app/bots/crypto_trader/src
RUN uv pip install --system "/app/bots/crypto_trader[live]"

WORKDIR /app/bots/crypto_trader
CMD ["crypto-trader", "--help"]
