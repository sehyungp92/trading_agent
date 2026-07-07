from __future__ import annotations

from trading_assistant_data.sources.kis.read_only_client import KisReadOnlyClient


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"output2": []}


def test_kis_client_ports_reference_daily_chart_params(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url, *, headers, params, timeout):
        captured.update({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = KisReadOnlyClient("https://example.test", "key", "secret", "token")

    client.get_daily_chart("005930", "20250512", "20260512")

    assert captured["headers"]["tr_id"] == "FHKST03010100"
    assert captured["params"]["FID_INPUT_ISCD"] == "005930"
    assert captured["params"]["FID_ORG_ADJ_PRC"] == "0"


def test_kis_client_ports_reference_minute_chart_params(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url, *, headers, params, timeout):
        captured.update({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = KisReadOnlyClient("https://example.test", "key", "secret", "token")

    client.get_minute_chart("005930", input_hour_hhmmss="153000", include_previous=True)

    assert captured["headers"]["tr_id"] == "FHKST03010200"
    assert captured["params"]["FID_INPUT_ISCD"] == "005930"
    assert captured["params"]["FID_INPUT_HOUR_1"] == "153000"
    assert captured["params"]["FID_PW_DATA_INCU_YN"] == "Y"
    assert "FID_PERIOD_DIV_CODE" not in captured["params"]


def test_kis_client_ports_reference_historical_minute_page_params(monkeypatch) -> None:
    captured: dict = {}

    def fake_get(url, *, headers, params, timeout):
        captured.update({"url": url, "headers": headers, "params": params, "timeout": timeout})
        return _Response()

    monkeypatch.setattr("requests.get", fake_get)
    client = KisReadOnlyClient("https://example.test", "key", "secret", "token")

    client.get_historical_minute_page(
        "005930",
        date_yyyymmdd="20260512",
        hour_hhmmss="153000",
        market_code="J",
        include_previous=True,
    )

    assert captured["url"].endswith("/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice")
    assert captured["headers"]["tr_id"] == "FHKST03010230"
    assert captured["params"]["FID_COND_MRKT_DIV_CODE"] == "J"
    assert captured["params"]["FID_INPUT_ISCD"] == "005930"
    assert captured["params"]["FID_INPUT_DATE_1"] == "20260512"
    assert captured["params"]["FID_INPUT_HOUR_1"] == "153000"
    assert captured["params"]["FID_PW_DATA_INCU_YN"] == "Y"
    assert captured["params"]["FID_FAKE_TICK_INCU_YN"] == ""
