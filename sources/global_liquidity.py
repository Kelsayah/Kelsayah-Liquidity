import pandas as pd


def build_global_liquidity_index(
    fed_assets: pd.Series,
    ecb_assets: pd.Series,
    boj_assets: pd.Series,
    eurusd: pd.Series,
    usdjpy: pd.Series,
    china_m2: pd.Series | None = None,
    usdcny: pd.Series | None = None,
) -> pd.DataFrame:
    """Construye GLI v0.1 en billones USD con FED, BCE y BoJ."""
    raw = pd.concat(
        {
            "FED": fed_assets,
            "BCE": ecb_assets,
            "BoJ": boj_assets,
            "EURUSD": eurusd,
            "USDJPY": usdjpy,
        },
        axis=1,
        sort=False,
    ).sort_index()
    weekly = raw.resample("W-FRI").last().ffill().dropna()
    if weekly.empty:
        raise ValueError("No existen fechas comunes suficientes para calcular el GLI")
    if (weekly["USDJPY"] <= 0).any():
        raise ValueError("USDJPY debe ser mayor que cero")

    result = pd.DataFrame(index=weekly.index)
    # WALCL y ECBASSETSW: millones de moneda local -> billones USD.
    result["FED"] = weekly["FED"] / 1_000_000
    result["BCE"] = weekly["BCE"] * weekly["EURUSD"] / 1_000_000
    # JPNASSETS: centenas de millones JPY -> billones USD.
    result["BoJ"] = weekly["BoJ"] * 0.0001 / weekly["USDJPY"]
    result["GLI Bancos Centrales"] = result[["FED", "BCE", "BoJ"]].sum(axis=1)
    if china_m2 is not None and usdcny is not None:
        china = pd.concat({"M2": china_m2, "USDCNY": usdcny}, axis=1, sort=False)
        china = china.resample("W-FRI").last().ffill().dropna()
        result = result.join(china, how="left").ffill().dropna()
        if (result["USDCNY"] <= 0).any():
            raise ValueError("USDCNY debe ser mayor que cero")
        result["China M2"] = result["M2"] * 0.0001 / result["USDCNY"]
        result = result.drop(columns=["M2", "USDCNY"])
    else:
        result["China M2"] = 0.0
    result["GLI"] = result["GLI Bancos Centrales"] + result["China M2"]
    return result


def build_gli_trends(gli: pd.Series) -> pd.DataFrame:
    if gli.empty or gli.iloc[0] == 0:
        raise ValueError("El GLI no contiene datos normalizables")
    normalized = gli.divide(gli.iloc[0]).multiply(100)
    trends = pd.DataFrame({"GLI": normalized})
    for span in (10, 20, 50, 200):
        trends[f"EMA {span}"] = (
            gli.ewm(span=span, adjust=False).mean().divide(gli.iloc[0]).multiply(100)
        )
    return trends


def build_tradingview_view(
    gli: pd.Series,
    mode: str = "Nivel",
    smoothing: int = 1,
    offset_weeks: int = 0,
) -> pd.Series:
    """Transforma el GLI semanal a vistas habituales de indicadores de TradingView."""
    if gli.empty:
        raise ValueError("El GLI no contiene datos")
    periods = {
        "Variación interanual": 52,
        "Variación 6 meses": 26,
        "Variación 3 meses": 13,
        "Variación mensual": 4,
    }
    if mode == "Nivel":
        result = gli.astype(float).copy()
        result.name = "GLI · T USD"
    elif mode in periods:
        result = gli.astype(float).pct_change(periods[mode]).multiply(100)
        result.name = f"GLI · {mode} (%)"
    else:
        raise ValueError(f"Modo GLI no reconocido: {mode}")
    if smoothing < 1:
        raise ValueError("El suavizado debe ser al menos 1")
    if smoothing > 1:
        result = result.rolling(smoothing, min_periods=1).mean()
    if offset_weeks:
        result = result.shift(offset_weeks, freq="W-FRI")
    return result.dropna()
