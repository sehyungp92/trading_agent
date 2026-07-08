FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir uv

COPY packages/trading_contracts/pyproject.toml /app/packages/trading_contracts/
COPY packages/trading_contracts/src /app/packages/trading_contracts/src
RUN uv pip install --system /app/packages/trading_contracts

COPY trading/k_stock_trader/pyproject.toml /app/trading/k_stock_trader/
COPY trading/k_stock_trader/config /app/trading/k_stock_trader/config
COPY trading/k_stock_trader/deployment /app/trading/k_stock_trader/deployment
COPY trading/k_stock_trader/infra /app/trading/k_stock_trader/infra
COPY trading/k_stock_trader/instrumentation /app/trading/k_stock_trader/instrumentation
COPY trading/k_stock_trader/kis_core /app/trading/k_stock_trader/kis_core
COPY trading/k_stock_trader/oms /app/trading/k_stock_trader/oms
COPY trading/k_stock_trader/oms_client /app/trading/k_stock_trader/oms_client
COPY trading/k_stock_trader/schemas /app/trading/k_stock_trader/schemas
COPY trading/k_stock_trader/scripts /app/trading/k_stock_trader/scripts
COPY trading/k_stock_trader/src /app/trading/k_stock_trader/src
COPY trading/k_stock_trader/strategy_common /app/trading/k_stock_trader/strategy_common
COPY trading/k_stock_trader/strategy_kalcb /app/trading/k_stock_trader/strategy_kalcb
COPY trading/k_stock_trader/strategy_olr /app/trading/k_stock_trader/strategy_olr
COPY trading/k_stock_trader/strategy_pcim /app/trading/k_stock_trader/strategy_pcim
RUN uv pip install --system /app/trading/k_stock_trader

WORKDIR /app/trading/k_stock_trader
CMD ["k-stock-olr-kalcb-runtime", "--help"]
