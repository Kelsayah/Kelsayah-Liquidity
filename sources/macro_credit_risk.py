import pandas as pd


def classify_macro_risk(score: float) -> str:
    if score < 25:
        return "Riesgo bajo"
    if score < 45:
        return "Riesgo moderado"
    if score < 60:
        return "Riesgo elevado"
    if score < 80:
        return "Riesgo alto"
    return "Riesgo extremo"


def _risk_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=frame.index)
    scores["Crédito high yield"] = pd.cut(
        frame["Spread high yield"], [-float("inf"), 3, 4, 5, 7, float("inf")],
        labels=[15, 30, 50, 75, 95], right=False,
    ).astype(float)
    scores["Condiciones financieras"] = pd.cut(
        frame["NFCI"], [-float("inf"), -0.5, 0, 0.5, 1, float("inf")],
        labels=[15, 30, 60, 80, 95], right=False,
    ).astype(float)
    scores["Curva 10Y–2Y"] = pd.cut(
        frame["Curva 10Y–2Y"], [-float("inf"), 0, 0.5, 1, float("inf")],
        labels=[80, 50, 30, 20], right=False,
    ).astype(float)
    scores["Desempleo"] = pd.cut(
        frame["Brecha desempleo"], [-float("inf"), 0.3, 0.5, float("inf")],
        labels=[25, 65, 90], right=False,
    ).astype(float)
    scores["Inflación"] = pd.cut(
        frame["Inflación interanual"], [-float("inf"), 2.5, 3.5, 5, float("inf")],
        labels=[25, 45, 70, 90], right=False,
    ).astype(float)
    scores["Actividad industrial"] = pd.cut(
        frame["Producción industrial interanual"], [-float("inf"), -2, 0, 2, float("inf")],
        labels=[85, 65, 35, 20], right=False,
    ).astype(float)
    return scores


def calculate_macro_credit_risk(series: dict[str, pd.Series]) -> dict:
    required = {
        "Curva 10Y–2Y", "Spread high yield", "NFCI", "IPC",
        "Desempleo", "Producción industrial",
    }
    missing = required.difference(series)
    if missing:
        raise ValueError(f"Faltan series: {', '.join(sorted(missing))}")

    weekly = pd.concat(series, axis=1).sort_index().resample("W-FRI").last().ffill()
    if weekly.empty:
        raise ValueError("No hay datos suficientes para calcular el riesgo macro")
    today = pd.Timestamp.today().normalize()
    if weekly.index[-1] > today:
        adjusted_index = list(weekly.index)
        adjusted_index[-1] = today
        weekly.index = pd.DatetimeIndex(adjusted_index)
    weekly["Inflación interanual"] = weekly["IPC"].pct_change(52).multiply(100)
    weekly["Producción industrial interanual"] = weekly["Producción industrial"].pct_change(52).multiply(100)
    unemployment_3m = weekly["Desempleo"].rolling(13, min_periods=1).mean()
    weekly["Brecha desempleo"] = unemployment_3m - unemployment_3m.rolling(52, min_periods=13).min()
    weekly = weekly.dropna(subset=[
        "Inflación interanual", "Producción industrial interanual", "Brecha desempleo",
    ])
    scores = _risk_scores(weekly)
    weights = {
        "Crédito high yield": 0.25,
        "Condiciones financieras": 0.20,
        "Curva 10Y–2Y": 0.20,
        "Desempleo": 0.15,
        "Inflación": 0.10,
        "Actividad industrial": 0.10,
    }
    risk_history = sum(scores[name] * weight for name, weight in weights.items())
    risk_history.name = "Riesgo macro y crédito"
    latest = weekly.iloc[-1]
    score = float(risk_history.iloc[-1])
    readings = {
        "Crédito high yield": f'{latest["Spread high yield"]:.2f}%',
        "Condiciones financieras": f'{latest["NFCI"]:.2f}',
        "Curva 10Y–2Y": f'{latest["Curva 10Y–2Y"]:+.2f} pp',
        "Desempleo": f'{latest["Desempleo"]:.2f}% · brecha {latest["Brecha desempleo"]:+.2f} pp',
        "Inflación": f'{latest["Inflación interanual"]:.2f}% interanual',
        "Actividad industrial": f'{latest["Producción industrial interanual"]:+.2f}% interanual',
    }
    details = pd.DataFrame([
        {
            "Factor": name,
            "Lectura": readings[name],
            "Riesgo": int(scores[name].iloc[-1]),
            "Peso": f"{weight:.0%}",
            "Aportación": round(scores[name].iloc[-1] * weight, 1),
        }
        for name, weight in weights.items()
    ])
    return {
        "score": round(score, 1),
        "classification": classify_macro_risk(score),
        "details": details,
        "history": risk_history,
        "data": weekly,
        "date": weekly.index[-1],
    }
