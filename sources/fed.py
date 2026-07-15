import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from utils.persistence import load_series, log_data_error, mark_series, save_series


load_dotenv()


def get_fred_client() -> Fred:
    api_key = os.getenv("FRED_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró FRED_API_KEY en el entorno o en los secretos de Streamlit"
        )

    return Fred(api_key=api_key)


def get_fred_series(series_id: str) -> dict:
    try:
        series = get_fred_history(series_id)

        if series is None or series.empty:
            return {
                "value": None,
                "previous": None,
                "change_pct": None,
                "date": None,
                "error": f"No se recibieron datos para {series_id}",
            }

        series = series.dropna()

        if len(series) < 2:
            return {
                "value": None,
                "previous": None,
                "change_pct": None,
                "date": None,
                "error": f"No hay datos suficientes para {series_id}",
            }

        current_value = float(series.iloc[-1])
        previous_value = float(series.iloc[-2])
        latest_date = pd.Timestamp(series.index[-1])

        if previous_value == 0:
            change_pct = 0.0
        else:
            change_pct = (
                (current_value - previous_value) / previous_value
            ) * 100

        return {
            "value": current_value,
            "previous": previous_value,
            "change_pct": change_pct,
            "date": latest_date,
            "error": None,
            "data_source": series.attrs.get("data_status", {}).get("source", "live"),
        }

    except Exception as error:
        return {
            "value": None,
            "previous": None,
            "change_pct": None,
            "date": None,
            "error": f"{type(error).__name__}: {error}",
        }


def get_fred_history(series_id: str, observation_start=None) -> pd.Series:
    """Devuelve una serie FRED limpia y ordenada cronológicamente."""
    cache_key = f"fred_{series_id}"
    try:
        fred = get_fred_client()
        series = fred.get_series(series_id, observation_start=observation_start)
        if series is None or series.empty:
            raise ValueError(f"No se recibieron datos para {series_id}")
        series = series.dropna().astype(float).sort_index()
        series.index = pd.to_datetime(series.index)
        series.name = series_id
        save_series(cache_key, series, "FRED")
        return mark_series(series, "FRED", "live")
    except Exception as error:
        log_data_error("FRED", series_id, error)
        try:
            series, _ = load_series(cache_key)
            if observation_start is not None:
                series = series.loc[series.index >= pd.Timestamp(observation_start)]
            if series.empty:
                raise ValueError("La caché no cubre el periodo solicitado")
            series.name = series_id
            return mark_series(series, "FRED", "cache", f"{type(error).__name__}: {error}")
        except Exception:
            raise error
