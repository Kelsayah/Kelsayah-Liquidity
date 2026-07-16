import yfinance as yf
import pandas as pd
import time
from utils.persistence import load_series, log_data_error, mark_series, save_series
from utils.yfinance_cache import configure_yfinance_cache


configure_yfinance_cache()


def get_market_data(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)

        data = ticker.history(
            period="1mo",
            interval="1d",
            auto_adjust=False,
            timeout=20,
            raise_errors=True,
        )

        if data.empty:
            return {
                "price": None,
                "change": None,
                "change_pct": None,
                "error": f"No se recibieron datos para {symbol}",
            }

        close_series = data["Close"].dropna()

        if len(close_series) < 2:
            return {
                "price": None,
                "change": None,
                "change_pct": None,
                "error": f"No hay suficientes cierres para {symbol}",
            }

        current_price = float(close_series.iloc[-1])
        previous_price = float(close_series.iloc[-2])

        change = current_price - previous_price
        change_pct = (change / previous_price) * 100

        cached = close_series.copy()
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        save_series(f"yahoo_{symbol}", cached, "Yahoo Finance")
        return {
            "price": current_price,
            "change": change,
            "change_pct": change_pct,
            "error": None,
            "data_source": "live",
        }

    except Exception as error:
        log_data_error("Yahoo Finance", symbol, error)
        try:
            cached, _ = load_series(f"yahoo_{symbol}")
            if len(cached) < 2:
                raise ValueError("Caché insuficiente")
            current_price = float(cached.iloc[-1])
            previous_price = float(cached.iloc[-2])
            change = current_price - previous_price
            return {
                "price": current_price,
                "change": change,
                "change_pct": (change / previous_price) * 100,
                "error": None,
                "data_source": "cache",
                "cache_error": f"{type(error).__name__}: {error}",
            }
        except Exception:
            return {
                "price": None, "change": None, "change_pct": None,
                "error": f"{type(error).__name__}: {error}",
                "data_source": "unavailable",
            }


def get_market_history(symbol: str, start) -> pd.Series:
    """Descarga cierres ajustados y devuelve una serie cronológica limpia."""
    data = None
    errors = []
    for attempt in range(3):
        try:
            data = yf.download(
                symbol,
                start=start,
                auto_adjust=True,
                progress=False,
                timeout=25,
                threads=False,
            )
            if data is not None and not data.empty:
                break
            errors.append("respuesta vacía")
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))

    if data is None or data.empty:
        try:
            data = yf.Ticker(symbol).history(
                start=start,
                auto_adjust=True,
                timeout=25,
                raise_errors=True,
            )
        except Exception as error:
            errors.append(f"Ticker.history: {type(error).__name__}: {error}")

    if data is None or data.empty:
        detail = " | ".join(errors[-3:])
        error = ValueError(f"No se recibieron datos históricos para {symbol}. {detail}")
        log_data_error("Yahoo Finance", symbol, error)
        try:
            series, _ = load_series(f"yahoo_{symbol}")
            series = series.loc[series.index >= pd.Timestamp(start)]
            if series.empty:
                raise ValueError("La caché no cubre el periodo solicitado")
            series.name = symbol
            return mark_series(series, "Yahoo Finance", "cache", str(error))
        except Exception:
            raise error

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float).sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = symbol
    save_series(f"yahoo_{symbol}", close, "Yahoo Finance")
    return mark_series(close, "Yahoo Finance", "live")
