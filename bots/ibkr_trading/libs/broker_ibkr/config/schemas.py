"""Pydantic models for configuration validation."""
from typing import Optional
from pydantic import BaseModel, Field


class IBKRProfile(BaseModel):
    host: str = "127.0.0.1"
    port: int = 4002  # 4002=paper, 4001=live
    client_id: int = 1
    account_id: str = ""
    is_gateway: bool = True
    readonly: bool = False
    reconnect_max_retries: int = 10
    reconnect_base_delay_s: float = 1.0
    reconnect_max_delay_s: float = 60.0
    pacing_orders_per_sec: float = 5.0
    pacing_messages_per_sec: float = 50.0


class ContractTemplate(BaseModel):
    symbol: str
    sec_type: str = "FUT"
    exchange: str
    currency: str = "USD"
    multiplier: float
    tick_size: float
    tick_value: float
    trading_class: Optional[str] = None
    primary_exchange: Optional[str] = None


class ExchangeRoute(BaseModel):
    root_symbol: str
    exchange: str
    primary_exchange: Optional[str] = None
    trading_class: Optional[str] = None
    local_symbol_pattern: Optional[str] = None
