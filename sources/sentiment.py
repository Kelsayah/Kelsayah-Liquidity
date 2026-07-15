import json
from urllib.request import Request, urlopen

import pandas as pd

from utils.persistence import load_series, log_data_error, mark_series, save_series


FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"
CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
MARKET_FNG_URL = "https://feargreedchart.com/api/?action=all"


def classify_fear_greed(value: float) -> str:
    if value <= 24:
        return "Miedo extremo"
    if value <= 44:
        return "Miedo"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Codicia"
    return "Codicia extrema"


def parse_cnn_fear_greed(payload: dict) -> pd.Series:
    """Convierte la respuesta de CNN en una serie diaria 0-100."""
    values = {}
    historical = payload.get("fear_and_greed_historical", {})
    for point in historical.get("data", []) if isinstance(historical, dict) else []:
        timestamp = point.get("x", point.get("timestamp"))
        score = point.get("y", point.get("score"))
        if timestamp is not None and score is not None:
            unit = "ms" if float(timestamp) > 10_000_000_000 else "s"
            values[pd.to_datetime(float(timestamp), unit=unit).normalize()] = float(score)

    current = payload.get("fear_and_greed", {})
    if isinstance(current, dict) and current.get("score") is not None:
        timestamp = current.get("timestamp") or pd.Timestamp.now().timestamp()
        if isinstance(timestamp, str):
            day = pd.to_datetime(timestamp, utc=True).tz_localize(None).normalize()
        else:
            unit = "ms" if float(timestamp) > 10_000_000_000 else "s"
            day = pd.to_datetime(float(timestamp), unit=unit).normalize()
        values[day] = float(current["score"])
    if not values:
        raise ValueError("CNN Fear & Greed no devolvió observaciones")
    return pd.Series(values, name="S&P 500 Fear & Greed").sort_index()


def parse_market_fear_greed(payload: dict) -> pd.Series:
    values = {
        pd.to_datetime(point["date"]).normalize(): float(point["score"])
        for point in payload.get("recent", [])
        if point.get("date") and point.get("score") is not None
    }
    if not values:
        raise ValueError("FearGreedChart no devolvió observaciones")
    return pd.Series(values, name="S&P 500 Fear & Greed").sort_index()


def get_sp500_fear_greed_history(observation_start=None) -> pd.Series:
    errors = []
    try:
        request = Request(CNN_FNG_URL, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=20) as response:
            series = parse_cnn_fear_greed(json.load(response))
        save_series("sp500_fear_greed", series, "CNN")
        provider, source, error_text = "CNN", "live", None
    except Exception as error:
        log_data_error("CNN", "S&P 500 Fear & Greed", error)
        errors.append(f"CNN: {type(error).__name__}: {error}")
        try:
            request = Request(MARKET_FNG_URL, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
            })
            with urlopen(request, timeout=20) as response:
                series = parse_market_fear_greed(json.load(response))
            save_series("sp500_fear_greed", series, "FearGreedChart.com")
            provider, source, error_text = "FearGreedChart.com", "live", None
        except Exception as fallback_error:
            log_data_error("FearGreedChart.com", "S&P 500 Fear & Greed", fallback_error)
            errors.append(f"FearGreedChart.com: {type(fallback_error).__name__}: {fallback_error}")
            series, metadata = load_series("sp500_fear_greed")
            provider = metadata.get("provider", "caché local")
            source, error_text = "cache", " | ".join(errors)
    if observation_start is not None:
        series = series.loc[series.index >= pd.Timestamp(observation_start)]
    if series.empty:
        raise ValueError("S&P 500 Fear & Greed no cubre el periodo solicitado")
    return mark_series(series, provider, source, error_text)


def get_crypto_fear_greed_history(observation_start=None) -> pd.Series:
    try:
        request = Request(FNG_URL, headers={"User-Agent": "GlobalLiquidityMonitor/1.0"})
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        points = payload.get("data", [])
        if not points:
            raise ValueError("Fear & Greed no devolvió observaciones")
        series = pd.Series(
            {
                pd.to_datetime(int(point["timestamp"]), unit="s").normalize(): float(point["value"])
                for point in points
            },
            name="Crypto Fear & Greed",
        ).sort_index()
        save_series("crypto_fear_greed", series, "Alternative.me")
        source, error_text = "live", None
    except Exception as error:
        log_data_error("Alternative.me", "Crypto Fear & Greed", error)
        series, _ = load_series("crypto_fear_greed")
        source, error_text = "cache", f"{type(error).__name__}: {error}"
    if observation_start is not None:
        series = series.loc[series.index >= pd.Timestamp(observation_start)]
    if series.empty:
        raise ValueError("Fear & Greed no cubre el periodo solicitado")
    return mark_series(series, "Alternative.me", source, error_text)
