from __future__ import annotations

from datetime import datetime
from urllib.request import Request, urlopen

import pandas as pd

from sources.fed import get_fred_history


BLS_CALENDAR_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

US_MACRO_SERIES = {
    "IPC general": "CPIAUCSL",
    "IPC subyacente": "CPILFESL",
    "Nóminas no agrícolas": "PAYEMS",
    "Desempleo": "UNRATE",
    "Salario medio por hora": "CES0500000003",
    "Ventas minoristas": "RSAFS",
    "Producción industrial": "INDPRO",
}


def build_us_macro_history(months: int = 12) -> pd.DataFrame:
    """Construye una tabla mensual comparable a partir de series oficiales de FRED."""
    raw = {name: get_fred_history(series_id) for name, series_id in US_MACRO_SERIES.items()}
    monthly = pd.concat(raw, axis=1).sort_index().resample("MS").last()

    result = pd.DataFrame(index=monthly.index)
    result["IPC interanual"] = monthly["IPC general"].pct_change(12) * 100
    result["IPC subyacente interanual"] = monthly["IPC subyacente"].pct_change(12) * 100
    result["Nóminas no agrícolas"] = monthly["Nóminas no agrícolas"].diff()
    result["Desempleo"] = monthly["Desempleo"]
    result["Salarios interanual"] = monthly["Salario medio por hora"].pct_change(12) * 100
    result["Ventas minoristas mensual"] = monthly["Ventas minoristas"].pct_change() * 100
    result["Producción industrial interanual"] = monthly["Producción industrial"].pct_change(12) * 100
    result = result.dropna(how="all").tail(months)
    result.attrs["source_status"] = {
        name: series.attrs.get("data_status", {}).get("source", "live")
        for name, series in raw.items()
    }
    return result


def latest_us_macro(history: pd.DataFrame) -> dict[str, dict]:
    definitions = {
        "IPC interanual": ("%", 1),
        "IPC subyacente interanual": ("%", 1),
        "Nóminas no agrícolas": ("mil", 0),
        "Desempleo": ("%", 1),
        "Salarios interanual": ("%", 1),
        "Ventas minoristas mensual": ("%", 1),
    }
    latest = {}
    for column, (unit, decimals) in definitions.items():
        series = history[column].dropna()
        if series.empty:
            continue
        latest[column] = {
            "value": float(series.iloc[-1]),
            "previous": float(series.iloc[-2]) if len(series) > 1 else None,
            "date": pd.Timestamp(series.index[-1]),
            "unit": unit,
            "decimals": decimals,
        }
    return latest


def _unfold_ics(text: str) -> list[str]:
    unfolded: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _parse_ics_datetime(value: str) -> pd.Timestamp:
    value = value.strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M", "%Y%m%d"):
        try:
            return pd.Timestamp(datetime.strptime(value.rstrip("Z"), fmt))
        except ValueError:
            continue
    raise ValueError(f"Fecha BLS desconocida: {value}")


def get_bls_release_calendar(limit: int = 12) -> pd.DataFrame:
    """Lee las próximas publicaciones desde el calendario oficial del BLS."""
    request = Request(BLS_CALENDAR_URL, headers={"User-Agent": "GlobalLiquidityMonitor/0.7"})
    with urlopen(request, timeout=15) as response:
        text = response.read().decode("utf-8", errors="replace")

    events: list[dict] = []
    current: dict[str, str] | None = None
    for line in _unfold_ics(text):
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT" and current is not None:
            summary = current.get("SUMMARY", "")
            start = current.get("DTSTART")
            if start and any(name in summary for name in (
                "Consumer Price Index", "Employment Situation", "Producer Price Index",
                "Job Openings and Labor Turnover", "Employment Cost Index",
            )):
                events.append({"Publicación": summary, "Fecha": _parse_ics_datetime(start)})
            current = None
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.split(";", 1)[0]] = value.replace("\\,", ",")

    today = pd.Timestamp.today().normalize()
    frame = pd.DataFrame(events)
    if frame.empty:
        return frame
    frame = frame.loc[frame["Fecha"] >= today].sort_values("Fecha").head(limit).reset_index(drop=True)
    return frame

