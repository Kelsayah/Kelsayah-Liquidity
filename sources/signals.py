import pandas as pd


def compare_gli_with_asset(
    gli: pd.Series,
    asset: pd.Series,
    lag_weeks: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Compara GLI adelantado con un activo y devuelve base 100 y correlación."""
    shifted = gli.shift(lag_weeks) if lag_weeks else gli
    frame = pd.concat({"GLI": shifted, "Activo": asset}, axis=1, sort=False)
    weekly = frame.resample("W-FRI").last().ffill().dropna()
    if len(weekly) < 3:
        raise ValueError("No hay observaciones suficientes para comparar")
    normalized = weekly.divide(weekly.iloc[0]).multiply(100)
    correlation = weekly.pct_change().dropna().corr().iloc[0, 1]
    return normalized, float(correlation)


def interpret_liquidity(gli: pd.Series, asset: pd.Series | None = None) -> dict:
    """Genera señales deterministas y auditables, sin depender de una API de IA."""
    if len(gli) < 26:
        return {"regime": "Sin datos", "score": 0, "messages": ["Histórico insuficiente."]}
    ema20 = gli.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = gli.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = gli.ewm(span=200, adjust=False).mean().iloc[-1]
    change4 = gli.pct_change(4).iloc[-1] * 100
    score = int(gli.iloc[-1] > ema20) + int(ema20 > ema50) + int(ema50 > ema200)
    score -= int(change4 < 0)
    regime = "Expansivo" if score >= 3 else "Contractivo" if score <= 0 else "Neutral"
    messages = [
        f"El GLI cambia {change4:+.2f}% en cuatro semanas.",
        f"La estructura EMA 20/50/200 es {'favorable' if ema20 > ema50 > ema200 else 'mixta o débil'}.",
    ]
    if asset is not None and len(asset) >= 20:
        asset_change = asset.pct_change(4).iloc[-1] * 100
        if change4 > 0 > asset_change:
            messages.append("La liquidez mejora mientras el activo retrocede: posible divergencia positiva.")
        elif change4 < 0 < asset_change:
            messages.append("El activo sube pese a una liquidez decreciente: divergencia de riesgo.")
    return {"regime": regime, "score": score, "change4": change4, "messages": messages}


def build_markdown_report(gli: pd.DataFrame, analysis: dict, asset_name: str) -> str:
    latest = gli.iloc[-1]
    lines = [
        "# Global Liquidity Monitor · Informe automático",
        "",
        f"Fecha de datos: {gli.index[-1].date().isoformat()}",
        f"GLI ampliado: {latest['GLI']:.2f} T USD",
        f"GLI bancos centrales: {latest['GLI Bancos Centrales']:.2f} T USD",
        f"China M2 (proxy): {latest['China M2']:.2f} T USD",
        f"Régimen: {analysis['regime']}",
        f"Activo analizado: {asset_name}",
        "",
        "## Lectura",
        "",
    ]
    lines.extend(f"- {message}" for message in analysis["messages"])
    lines.extend([
        "", "## Metodología", "",
        "El GLI ampliado agrega balances de FED, BCE y BoJ más China M2 como proxy, todos convertidos a USD.",
        "Es una herramienta analítica orientativa y no constituye asesoramiento financiero.",
    ])
    return "\n".join(lines)
