FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY bots/k_stock_trader/pyproject.toml /app/bots/k_stock_trader/
COPY bots/k_stock_trader/config /app/bots/k_stock_trader/config
COPY bots/k_stock_trader/deployment /app/bots/k_stock_trader/deployment
COPY bots/k_stock_trader/infra /app/bots/k_stock_trader/infra
COPY bots/k_stock_trader/instrumentation /app/bots/k_stock_trader/instrumentation
COPY bots/k_stock_trader/kis_core /app/bots/k_stock_trader/kis_core
COPY bots/k_stock_trader/oms /app/bots/k_stock_trader/oms
COPY bots/k_stock_trader/oms_client /app/bots/k_stock_trader/oms_client
COPY bots/k_stock_trader/schemas /app/bots/k_stock_trader/schemas
COPY bots/k_stock_trader/src /app/bots/k_stock_trader/src
COPY bots/k_stock_trader/strategy_common /app/bots/k_stock_trader/strategy_common
COPY bots/k_stock_trader/strategy_kalcb /app/bots/k_stock_trader/strategy_kalcb
COPY bots/k_stock_trader/strategy_olr /app/bots/k_stock_trader/strategy_olr
COPY bots/k_stock_trader/strategy_pcim /app/bots/k_stock_trader/strategy_pcim
RUN uv pip install --system /app/bots/k_stock_trader

WORKDIR /app/bots/k_stock_trader
CMD ["k-stock-olr-kalcb-runtime", "--help"]
