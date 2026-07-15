import pandas as pd


def compare_liquidity_with_asset(
    liquidity: pd.Series,
    asset: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alinea a frecuencia semanal, normaliza a base 100 y calcula EMAs del activo."""
    frame = pd.concat(
        {"US Net Liquidity": liquidity, "Activo": asset},
        axis=1,
        sort=False,
    ).sort_index()
    weekly = frame.resample("W-FRI").last().ffill().dropna()
    if weekly.empty:
        raise ValueError("No existen fechas comunes entre liquidez y activo")

    base = weekly.iloc[0]
    if (base == 0).any():
        raise ValueError("No se puede normalizar una serie cuyo valor inicial es cero")

    normalized = weekly.divide(base).multiply(100)
    trends = pd.DataFrame(index=weekly.index)
    trends["Activo"] = normalized["Activo"]
    for span in (10, 20, 50, 200):
        ema = weekly["Activo"].ewm(span=span, adjust=False).mean()
        trends[f"EMA {span}"] = ema.divide(base["Activo"]).multiply(100)
    return normalized, trends
