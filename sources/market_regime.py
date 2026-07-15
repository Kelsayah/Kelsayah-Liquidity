import pandas as pd


def _latest(series: pd.Series) -> pd.Series:
    clean = series.dropna().astype(float).sort_index()
    if clean.empty:
        raise ValueError("Serie sin datos")
    return clean


def classify_market_regime(score: float) -> str:
    if score >= 75:
        return "Risk-on fuerte"
    if score >= 55:
        return "Risk-on moderado"
    if score >= 45:
        return "Neutral"
    if score >= 25:
        return "Risk-off"
    return "Crisis de liquidez"


def calculate_market_regime(
    gli: pd.Series,
    sp500: pd.Series,
    vix: pd.Series,
    dxy: pd.Series,
    fed_rate: pd.Series,
    sentiment: pd.Series,
) -> dict:
    """Calcula una puntuación explicable de régimen de mercado entre 0 y 100."""
    gli, sp500, vix = _latest(gli), _latest(sp500), _latest(vix)
    dxy, fed_rate, sentiment = _latest(dxy), _latest(fed_rate), _latest(sentiment)

    gli_change = gli.pct_change(4).iloc[-1] if len(gli) > 4 else 0.0
    gli_ema50 = gli.ewm(span=50, adjust=False).mean().iloc[-1]
    liquidity_score = 25 + 35 * (gli_change > 0) + 40 * (gli.iloc[-1] >= gli_ema50)

    ema20 = sp500.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = sp500.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = sp500.ewm(span=200, adjust=False).mean().iloc[-1]
    trend_score = 10 + 30 * (sp500.iloc[-1] >= ema20) + 30 * (sp500.iloc[-1] >= ema50) + 30 * (sp500.iloc[-1] >= ema200)

    vix_value = vix.iloc[-1]
    volatility_score = 90 if vix_value <= 15 else 70 if vix_value <= 20 else 45 if vix_value <= 25 else 25 if vix_value <= 35 else 10

    dxy_change = dxy.pct_change(min(12, len(dxy) - 1)).iloc[-1] if len(dxy) > 1 else 0.0
    dollar_score = 80 if dxy_change <= -0.02 else 65 if dxy_change < 0 else 45 if dxy_change <= 0.02 else 25

    rate_change = fed_rate.iloc[-1] - fed_rate.iloc[-min(14, len(fed_rate))]
    rates_score = 80 if rate_change <= -0.25 else 60 if rate_change <= 0 else 40 if rate_change <= 0.25 else 25
    sentiment_score = max(0.0, min(100.0, float(sentiment.iloc[-1])))

    factors = {
        "Liquidez global": (float(liquidity_score), 0.25),
        "Tendencia S&P 500": (float(trend_score), 0.25),
        "Volatilidad VIX": (float(volatility_score), 0.20),
        "Dólar DXY": (float(dollar_score), 0.10),
        "Tipos FED": (float(rates_score), 0.10),
        "Sentimiento": (float(sentiment_score), 0.10),
    }
    score = sum(value * weight for value, weight in factors.values())
    details = pd.DataFrame([
        {"Factor": name, "Puntuación": round(value), "Peso": f"{weight:.0%}", "Aportación": round(value * weight, 1)}
        for name, (value, weight) in factors.items()
    ])
    return {
        "score": round(score, 1),
        "regime": classify_market_regime(score),
        "details": details,
        "date": min(series.index[-1] for series in (gli, sp500, vix, dxy, fed_rate, sentiment)),
    }
