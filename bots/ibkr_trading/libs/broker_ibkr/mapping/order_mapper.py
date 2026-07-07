"""Maps OMS canonical order fields to IB Order objects."""
from ib_async import LimitOrder, MarketOrder, StopOrder, StopLimitOrder, Order


class OrderMapper:
    """Maps OMS canonical order fields to IB Order objects."""

    @staticmethod
    def to_ib_order(
        action: str,  # "BUY" or "SELL"
        order_type: str,  # "LIMIT", "MARKET", "STOP", "STOP_LIMIT"
        qty: int,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: str = "DAY",
        account: str = "",
        oca_group: str = "",
        oca_type: int = 0,
        outside_rth: bool = False,
        transmit: bool = True,
        order_ref: str = "",
    ) -> Order:
        """Build an IB Order from canonical fields."""
        if order_type == "MARKET":
            order = MarketOrder(action=action, totalQuantity=qty)
        elif order_type == "LIMIT":
            if limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            order = LimitOrder(action=action, totalQuantity=qty, lmtPrice=limit_price)
        elif order_type == "STOP":
            if stop_price is None:
                raise ValueError("STOP order requires stop_price")
            order = StopOrder(action=action, totalQuantity=qty, stopPrice=stop_price)
        elif order_type == "STOP_LIMIT":
            if stop_price is None or limit_price is None:
                raise ValueError("STOP_LIMIT order requires both stop_price and limit_price")
            order = StopLimitOrder(
                action=action,
                totalQuantity=qty,
                stopPrice=stop_price,
                lmtPrice=limit_price,
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

        order.tif = tif
        order.account = account
        order.outsideRth = outside_rth
        order.transmit = transmit
        if oca_group:
            order.ocaGroup = oca_group
            order.ocaType = oca_type
        if order_ref:
            order.orderRef = order_ref
        return order
