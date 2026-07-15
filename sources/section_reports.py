from datetime import date


def _probabilities(weighted_scenarios: list[tuple[str, float, str]]) -> list[dict]:
    total = sum(max(0.1, weight) for _, weight, _ in weighted_scenarios)
    probabilities = [round(max(0.1, weight) / total * 100) for _, weight, _ in weighted_scenarios]
    probabilities[-1] += 100 - sum(probabilities)
    return [
        {"Escenario": name, "Probabilidad": probability, "Qué implicaría": description}
        for (name, _, description), probability in zip(weighted_scenarios, probabilities)
    ]


def _market_scenarios(score: float) -> list[dict]:
    neutral = max(18, 100 - abs(score - 50) * 1.5)
    return _probabilities([
        ("Continuación favorable", max(8, score), "Liquidez y tendencia sostienen los activos de riesgo."),
        ("Consolidación lateral", neutral, "Señales mixtas y movimiento dentro de un rango."),
        ("Corrección / risk-off", max(8, 100 - score), "Aumentan volatilidad, dólar y presión vendedora."),
    ])


def build_section_report(section: str, context: dict) -> dict:
    market_score = float(context["market_score"])
    macro_score = float(context["macro_score"])
    gli_change = float(context["gli_change4"])
    inflation = float(context["inflation"])
    fed_change = float(context["fed_rate_change"])

    if section == "Resumen":
        combined = market_score * 0.60 + (100 - macro_score) * 0.40
        situation = (
            f"El mercado está en régimen {context['market_regime'].lower()} ({market_score:.0f}/100), "
            f"mientras el riesgo macro y de crédito es {context['macro_regime'].lower()} ({macro_score:.0f}/100). "
            f"La liquidez global cambia {gli_change:+.2f}% en cuatro semanas."
        )
        signals = [
            f"Régimen de mercado: {context['market_regime']}.",
            f"Riesgo macro y crédito: {context['macro_regime']}.",
            f"VIX: {context['vix']:.2f}.",
            f"S&P 500 {'por encima' if context['sp500_above_ema200'] else 'por debajo'} de su EMA 200 semanal.",
        ]
        scenarios = _market_scenarios(combined)
    elif section == "Liquidez global":
        liquidity_score = max(0, min(100, 50 + (20 if gli_change > 0 else -20) + (20 if context["gli_above_ema200"] else -20)))
        situation = (
            f"El GLI registra una variación de {gli_change:+.2f}% en cuatro semanas y se encuentra "
            f"{'sobre' if context['gli_above_ema200'] else 'bajo'} su EMA 200. "
            "La señal mide dirección y tendencia de la liquidez, no el nivel absoluto de los mercados."
        )
        signals = [
            f"GLI: {context['gli_latest']:.2f} T USD.",
            f"Variación de cuatro semanas: {gli_change:+.2f}%.",
            f"Tendencia EMA 200: {'alcista' if context['gli_above_ema200'] else 'bajista'}.",
            f"DXY, cambio aproximado de 12 semanas: {context['dxy_change12']:+.2f}%.",
        ]
        scenarios = _probabilities([
            ("Expansión de liquidez", max(8, liquidity_score), "El GLI mejora y crea un viento favorable para activos de riesgo."),
            ("Estabilización", max(20, 100 - abs(liquidity_score - 50) * 1.5), "La liquidez oscila sin una dirección dominante."),
            ("Contracción de liquidez", max(8, 100 - liquidity_score), "El GLI pierde tendencia y aumenta el riesgo de divergencias."),
        ])
    elif section == "Política monetaria":
        cut = 30 + max(0, macro_score - 40) * 0.5 + max(0, -fed_change) * 30
        hold = max(20, 60 - abs(fed_change) * 25)
        hike = 12 + max(0, inflation - 2.5) * 12 + max(0, fed_change) * 30
        situation = (
            f"El tipo efectivo de la FED está en {context['fed_rate']:.2f}% y ha cambiado "
            f"{fed_change:+.2f} puntos en unas 13 semanas. La inflación interanual se sitúa en {inflation:.2f}%."
        )
        signals = [
            f"Inflación: {inflation:.2f}% interanual.",
            f"Cambio reciente del Fed Funds: {fed_change:+.2f} pp.",
            f"Curva 10Y–2Y: {context['yield_curve']:+.2f} pp.",
            f"Riesgo macro agregado: {macro_score:.0f}/100.",
        ]
        scenarios = _probabilities([
            ("Recortes / política más flexible", cut, "Los tipos bajan o la comunicación se vuelve más acomodaticia."),
            ("Pausa prolongada", hold, "Los bancos centrales esperan más datos antes de actuar."),
            ("Tipos altos o nuevas subidas", hike, "La inflación obliga a mantener una política restrictiva."),
        ])
    elif section == "Mercados":
        situation = (
            f"El régimen de mercado es {context['market_regime'].lower()} con {market_score:.0f}/100. "
            f"El VIX está en {context['vix']:.2f} y el S&P 500 se mantiene "
            f"{'sobre' if context['sp500_above_ema200'] else 'bajo'} su EMA 200 semanal."
        )
        signals = [
            f"Puntuación de régimen: {market_score:.0f}/100.",
            f"VIX: {context['vix']:.2f}.",
            f"Fear & Greed S&P 500: {context['sentiment']:.0f}/100.",
            f"Liquidez global a cuatro semanas: {gli_change:+.2f}%.",
        ]
        scenarios = _market_scenarios(market_score)
    elif section == "Riesgo macro y crédito":
        situation = (
            f"El riesgo macro y de crédito es {context['macro_regime'].lower()} con {macro_score:.0f}/100. "
            f"El spread high yield está en {context['hy_spread']:.2f}% y el NFCI en {context['nfci']:.2f}."
        )
        signals = [
            f"Spread high yield: {context['hy_spread']:.2f}%.",
            f"NFCI: {context['nfci']:.2f}.",
            f"Curva 10Y–2Y: {context['yield_curve']:+.2f} pp.",
            f"Inflación: {inflation:.2f}% y desempleo: {context['unemployment']:.2f}%.",
        ]
        scenarios = _probabilities([
            ("Aterrizaje suave", max(8, 100 - macro_score), "Crédito estable y desaceleración sin deterioro fuerte del empleo."),
            ("Desaceleración", max(22, 100 - abs(macro_score - 50) * 1.4), "Crecimiento más débil con tensión financiera contenida."),
            ("Estrés / recesión", max(8, macro_score), "Se amplían spreads y empeoran empleo y condiciones financieras."),
        ])
    else:
        raise ValueError(f"No existe informe para la sección {section}")

    generated = date.today().strftime("%d/%m/%Y")
    markdown = [f"# Informe · {section}", "", f"Generado: {generated}", "", "## Situación actual", "", situation, "", "## Señales", ""]
    markdown.extend(f"- {signal}" for signal in signals)
    markdown.extend(["", "## Escenarios orientativos", ""])
    markdown.extend(f"- **{row['Escenario']} ({row['Probabilidad']}%)**: {row['Qué implicaría']}" for row in scenarios)
    markdown.extend(["", "Los porcentajes son estimaciones heurísticas basadas en los indicadores del dashboard; no son probabilidades estadísticas calibradas ni recomendaciones de inversión."])
    return {"situation": situation, "signals": signals, "scenarios": scenarios, "markdown": "\n".join(markdown)}
