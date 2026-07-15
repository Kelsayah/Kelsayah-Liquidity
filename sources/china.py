import json
from urllib.request import Request, urlopen

import pandas as pd
from utils.persistence import load_series, log_data_error, mark_series, save_series


CHINA_M2_URL = "https://chinadata.live/api/v2/data/china-m2-money-supply"


def get_china_m2_history(observation_start=None) -> pd.Series:
    try:
        request = Request(CHINA_M2_URL, headers={"User-Agent": "GlobalLiquidityMonitor/1.0"})
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        points = payload.get("data", {}).get("data", [])
        if not points:
            raise ValueError("La fuente de China no devolvió observaciones")
        series = pd.Series(
            {pd.Timestamp(f'{point["date"]}-01'): float(point["value"]) for point in points},
            name="China M2",
        ).sort_index()
        save_series("china_m2", series, "PBoC/NBS mirror")
        source = "live"
        error_text = None
    except Exception as error:
        log_data_error("China Data", "China M2", error)
        series, _ = load_series("china_m2")
        series.name = "China M2"
        source = "cache"
        error_text = f"{type(error).__name__}: {error}"
    if observation_start is not None:
        series = series.loc[series.index >= pd.Timestamp(observation_start)]
    if series.empty:
        raise ValueError("China M2 no cubre el periodo solicitado")
    return mark_series(series, "China M2", source, error_text)
