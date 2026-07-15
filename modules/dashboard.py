from datetime import date

import pandas as pd
import streamlit as st

from modules.cards import draw_fred_card, draw_market_card
from sources.analytics import compare_liquidity_with_asset
from sources.china import get_china_m2_history
from sources.fed import get_fred_history, get_fred_series
from sources.global_liquidity import (
    build_custom_gli, build_global_liquidity_index, build_gli_trends,
    build_tradingview_view,
)
from sources.liquidity import (
    build_us_net_liquidity_history,
    calculate_us_net_liquidity,
)
from sources.markets import get_market_data, get_market_history
from sources.policy_rates import (
    build_policy_comparison, classify_policy, get_china_lpr_history, rate_change,
)
from sources.signals import build_markdown_report, compare_gli_with_asset, interpret_liquidity
from sources.sentiment import (
    classify_fear_greed, get_crypto_fear_greed_history,
    get_sp500_fear_greed_history,
)
from utils.constants import (
    BITCOIN, BOJ_BALANCE, BOJ_CALL_RATE, DXY, ECB_BALANCE, ECB_DEPOSIT_RATE,
    EURUSD, FED_BALANCE, FED_FUNDS_RATE, GOLD, M2, NASDAQ, REVERSE_REPO,
    SP500, TGA, USDCNY, USDJPY, US10Y, VIX,
)

PERIOD_YEARS = {"1 año": 1, "3 años": 3, "5 años": 5, "10 años": 10}
COMPARISON_ASSETS = {
    "S&P 500": SP500,
    "Nasdaq": NASDAQ,
    "Bitcoin": BITCOIN,
    "Oro": GOLD,
}


@st.cache_data(ttl=300)
def load_market_data() -> dict:
    return {
        "VIX": get_market_data(VIX), "DXY": get_market_data(DXY),
        "S&P 500": get_market_data(SP500), "Nasdaq": get_market_data(NASDAQ),
        "Bitcoin": get_market_data(BITCOIN), "Oro": get_market_data(GOLD),
        "US 10Y": get_market_data(US10Y),
    }


@st.cache_data(ttl=3600)
def load_fred_data() -> dict:
    return {
        "Balance FED": get_fred_series(FED_BALANCE),
        "Reverse Repo": get_fred_series(REVERSE_REPO),
        "TGA": get_fred_series(TGA), "M2": get_fred_series(M2),
    }


@st.cache_data(ttl=3600)
def load_liquidity_history(years: int) -> pd.DataFrame:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    return build_us_net_liquidity_history(
        get_fred_history(FED_BALANCE, start),
        get_fred_history(TGA, start),
        get_fred_history(REVERSE_REPO, start),
    )


@st.cache_data(ttl=3600)
def load_comparison(years: int, asset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    liquidity = load_liquidity_history(years)["US Net Liquidity"]
    asset = get_market_history(COMPARISON_ASSETS[asset_name], start)
    normalized, trends = compare_liquidity_with_asset(liquidity, asset)
    normalized = normalized.rename(columns={"Activo": asset_name})
    trends = trends.rename(columns={"Activo": asset_name})
    return normalized, trends


@st.cache_data(ttl=3600)
def load_asset_history(years: int, asset_name: str) -> pd.Series:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    return get_market_history(COMPARISON_ASSETS[asset_name], start)


@st.cache_data(ttl=3600)
def load_global_liquidity(years: int) -> pd.DataFrame:
    # Margen adicional para que las EMAs largas no empiecen sin contexto.
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 5))
    fed = get_fred_history(FED_BALANCE, start)
    ecb = get_fred_history(ECB_BALANCE, start)
    boj = get_fred_history(BOJ_BALANCE, start)
    china = get_china_m2_history(start)
    tga = get_fred_history(TGA, start)
    reverse_repo = get_fred_history(REVERSE_REPO, start)
    usa_m2 = get_fred_history(M2, start)
    result = build_global_liquidity_index(
        fed, ecb, boj,
        get_market_history(EURUSD, start),
        get_market_history(USDJPY, start),
        china,
        get_market_history(USDCNY, start),
    )
    extra = pd.concat({
        "TGA": tga / 1_000_000,
        "Reverse Repo": reverse_repo / 1_000,
        "USA M2": usa_m2 / 1_000,
    }, axis=1).resample("W-FRI").last().ffill()
    result = result.join(extra, how="left").ffill()
    statuses = {
        "FED": fed.attrs.get("data_status", {}).get("source", "live"),
        "BCE": ecb.attrs.get("data_status", {}).get("source", "live"),
        "BoJ": boj.attrs.get("data_status", {}).get("source", "live"),
        "China": china.attrs.get("data_status", {}).get("source", "live"),
    }
    for provider, status in statuses.items():
        result[f"_{provider} status"] = status
    # Columnas internas constantes: sobreviven a la serialización de la caché.
    result["_Europe source value"] = float(ecb.iloc[-1])
    result["_Europe source date"] = ecb.index[-1]
    result["_Japan source value"] = float(boj.iloc[-1])
    result["_Japan source date"] = boj.index[-1]
    result["_China source value"] = float(china.iloc[-1])
    result["_China source date"] = china.index[-1]
    visible_start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    visible = result.loc[result.index >= visible_start].copy()
    return visible


@st.cache_data(ttl=3600)
def load_ecb_rate_history(years: int) -> pd.Series:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    rate = get_fred_history(ECB_DEPOSIT_RATE, start)
    return rate.resample("W-FRI").last().ffill().dropna().rename("Tipo depósito BCE")


@st.cache_data(ttl=3600)
def load_fed_rate_history(years: int) -> pd.Series:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    rate = get_fred_history(FED_FUNDS_RATE, start)
    return rate.resample("W-FRI").last().ffill().dropna().rename("Fed Funds efectivo")


@st.cache_data(ttl=3600)
def load_policy_rates(years: int) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 3))
    fed = get_fred_history(FED_FUNDS_RATE, start).rename("FED")
    ecb = get_fred_history(ECB_DEPOSIT_RATE, start).rename("BCE")
    boj = get_fred_history(BOJ_CALL_RATE, start).rename("BoJ")
    china_frame = get_china_lpr_history()
    china_1y = china_frame["China LPR 1 año"].rename("China LPR 1A")
    comparison = build_policy_comparison({
        "FED": fed, "BCE": ecb, "BoJ": boj, "China LPR 1A": china_1y,
    })
    visible_start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    comparison = comparison.loc[comparison.index >= visible_start]
    dates = {
        "FED": fed.index[-1], "BCE": ecb.index[-1], "BoJ": boj.index[-1],
        "China LPR 1A": china_frame.index[-1],
    }
    return comparison, dates


@st.cache_data(ttl=3600)
def load_fear_greed(years: int) -> pd.Series:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    return get_crypto_fear_greed_history(start)


@st.cache_data(ttl=3600)
def load_sp500_fear_greed(years: int) -> pd.Series:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    return get_sp500_fear_greed_history(start)


def draw_header(section: str) -> None:
    st.markdown(f"""
        <div class="dashboard-header">
            <h1 class="dashboard-title">🌍 Global Liquidity Monitor</h1>
            <p class="dashboard-subtitle">{section} · liquidez y condiciones financieras</p>
        </div>
    """, unsafe_allow_html=True)


def draw_summary(market_data: dict, fred_data: dict) -> dict:
    net = calculate_us_net_liquidity(
        fred_data["Balance FED"], fred_data["TGA"], fred_data["Reverse Repo"]
    )
    st.markdown('<div class="section-title">Resumen general</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    with cols[0]:
        if net.get("error") or net.get("value") is None:
            st.metric("US Net Liquidity", "Sin datos")
        else:
            st.metric("US Net Liquidity", f'{net["value"]:,.1f} B USD',
                      f'{net["change"]:+,.1f} B USD')
    with cols[1]:
        draw_market_card("VIX", market_data["VIX"], inverse_delta=True)
    with cols[2]:
        draw_market_card("DXY", market_data["DXY"], inverse_delta=True)
    with cols[3]:
        draw_market_card("US 10Y", market_data["US 10Y"], suffix="%", inverse_delta=True)
    return net


def draw_fed_cards(fred_data: dict) -> None:
    st.markdown('<div class="section-title">Liquidez de Estados Unidos</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    specs = [
        ("Balance FED", 1000, " B USD", False, 1),
        ("Reverse Repo", 1, " B USD", True, 1),
        ("TGA", 1000, " B USD", True, 1),
        ("M2", 1000, " T USD", False, 2),
    ]
    for col, (label, divisor, suffix, inverse, decimals) in zip(cols, specs):
        with col:
            draw_fred_card(label, fred_data[label], divisor, suffix, decimals, inverse)


def draw_market_cards(market_data: dict) -> None:
    st.markdown('<div class="section-title">Mercados</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    specs = [("S&P 500", "", 2), ("Nasdaq", "", 2),
             ("Bitcoin", " USD", 0), ("Oro", " USD", 2)]
    for col, (label, suffix, decimals) in zip(cols, specs):
        with col:
            draw_market_card(label, market_data[label], suffix=suffix, decimals=decimals)


def _sentiment_gauge(title: str, history: pd.Series) -> None:
    value = max(0.0, min(100.0, float(history.iloc[-1])))
    angle = -90 + value * 1.8
    status = history.attrs.get("data_status", {}).get("source", "live")
    provider = history.attrs.get("data_status", {}).get("provider", "Fuente externa")
    label = classify_fear_greed(value)
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.05rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">{title}</div>
      <div style="position:relative;max-width:390px;height:190px;margin:auto;overflow:hidden">
        <div style="position:absolute;width:100%;aspect-ratio:1;left:0;top:0;border-radius:50%;background:conic-gradient(from 270deg,#dc2626 0deg 36deg,#f97316 36deg 72deg,#eab308 72deg 108deg,#84cc16 108deg 144deg,#16a34a 144deg 180deg,transparent 180deg);transform:rotate(0deg)"></div>
        <div style="position:absolute;width:72%;aspect-ratio:1;left:14%;top:14%;border-radius:50%;background:#111827"></div>
        <div style="position:absolute;width:4px;height:112px;background:#f8fafc;left:calc(50% - 2px);bottom:0;transform-origin:50% 100%;transform:rotate({angle:.1f}deg);border-radius:4px;box-shadow:0 0 8px #000"></div>
        <div style="position:absolute;width:18px;height:18px;background:#f8fafc;border-radius:50%;left:calc(50% - 9px);bottom:-9px"></div>
        <div style="position:absolute;left:0;right:0;bottom:22px;font-size:2.5rem;font-weight:800;color:#f8fafc">{value:.0f}</div>
      </div>
      <div style="font-weight:700;color:#e5e7eb">{label}</div>
      <div style="font-size:.78rem;color:#94a3b8;margin-top:5px">{history.index[-1].strftime('%d/%m/%Y')} · {provider} · {'En directo' if status == 'live' else 'Caché local'}</div>
    </div>
    """, unsafe_allow_html=True)


def draw_fear_greed(period: str) -> None:
    st.markdown('<div class="section-title">Fear & Greed · sentimiento de mercado</div>', unsafe_allow_html=True)
    histories, errors = {}, []
    for title, loader in (
        ("S&P 500 Fear & Greed", load_sp500_fear_greed),
        ("Crypto Fear & Greed", load_fear_greed),
    ):
        try:
            histories[title] = loader(PERIOD_YEARS[period])
        except Exception as error:
            errors.append(f"{title}: {type(error).__name__}: {error}")

    columns = st.columns(2)
    for column, title in zip(columns, ("S&P 500 Fear & Greed", "Crypto Fear & Greed")):
        with column:
            if title in histories:
                _sentiment_gauge(title, histories[title])
            else:
                st.info(f"{title}: sin datos disponibles")

    if histories:
        comparison = pd.concat(histories.values(), axis=1).sort_index().ffill()
        st.markdown("#### Evolución comparada")
        st.line_chart(comparison, y_label="Índice 0–100")
        st.caption(
            "Fuentes: CNN con respaldo de FearGreedChart.com (S&P 500) y Alternative.me (cripto). 0 indica miedo extremo y "
            "100 codicia extrema. Son indicadores de sentimiento, no señales de compra o venta."
        )
    for error in errors:
        st.warning(error)


def draw_history(period: str) -> None:
    st.markdown('<div class="section-title">Evolución histórica</div>', unsafe_allow_html=True)
    try:
        history = load_liquidity_history(PERIOD_YEARS[period])
        if history.empty:
            st.info("No hay histórico disponible para el periodo seleccionado.")
            return
        st.line_chart(history[["US Net Liquidity"]], color="#38bdf8")
        with st.expander("Ver componentes del cálculo"):
            st.line_chart(history[["Balance FED", "TGA", "Reverse Repo"]])
        st.caption(
            "Cifras en miles de millones de USD. Series alineadas al cierre semanal; "
            "los valores se mantienen hasta la siguiente publicación disponible."
        )
    except Exception as error:
        st.warning(f"No se pudo cargar el histórico: {type(error).__name__}: {error}")


def draw_market_comparison(period: str) -> None:
    st.markdown(
        '<div class="section-title">Liquidez frente a mercados</div>',
        unsafe_allow_html=True,
    )
    asset_name = st.selectbox(
        "Activo para comparar",
        list(COMPARISON_ASSETS),
        key="comparison_asset",
    )
    try:
        normalized, trends = load_comparison(PERIOD_YEARS[period], asset_name)
        st.line_chart(normalized)
        st.caption(
            "Comparación base 100 desde la primera fecha común. "
            "Las dos curvas muestran evolución relativa, no importes absolutos."
        )

        with st.expander("Tendencia del activo · EMA 10, 20, 50 y 200", expanded=True):
            st.line_chart(trends)
            latest = trends.iloc[-1]
            signal = "Alcista" if latest[asset_name] > latest["EMA 200"] else "Bajista"
            st.metric("Tendencia frente a EMA 200", signal)
            st.caption(
                "Medias móviles exponenciales calculadas sobre cierres semanales "
                "y expresadas en la misma escala base 100."
            )
    except Exception as error:
        st.warning(f"No se pudo construir la comparación: {type(error).__name__}: {error}")


def draw_global_liquidity(period: str) -> None:
    st.markdown('<div class="section-title">Global Liquidity Index · v1.0</div>', unsafe_allow_html=True)
    try:
        history = load_global_liquidity(PERIOD_YEARS[period])
        if history.empty:
            st.info("No hay datos globales para el periodo seleccionado.")
            return

        latest = history["GLI"].iloc[-1]
        monthly = history["GLI"].pct_change(4).iloc[-1] * 100
        trends = build_gli_trends(history["GLI"])
        above_200 = trends["GLI"].iloc[-1] >= trends["EMA 200"].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("GLI", f"{latest:,.2f} T USD")
        c2.metric("Variación 4 semanas", "Sin datos" if pd.isna(monthly) else f"{monthly:+.2f}%")
        c3.metric("Tendencia EMA 200", "Alcista" if above_200 else "Bajista")

        europe = {
            "value": history["_Europe source value"].iloc[-1],
            "date": pd.Timestamp(history["_Europe source date"].iloc[-1]),
        }
        japan = {
            "value": history["_Japan source value"].iloc[-1],
            "date": pd.Timestamp(history["_Japan source date"].iloc[-1]),
        }
        china = {
            "value": history["_China source value"].iloc[-1],
            "date": pd.Timestamp(history["_China source date"].iloc[-1]),
        }
        ecb_rate = load_ecb_rate_history(PERIOD_YEARS[period])
        fed_rate = load_fed_rate_history(PERIOD_YEARS[period])
        j0, j1, j2 = st.columns(3)
        with j0:
            if europe:
                europe_trillion_eur = europe["value"] / 1_000_000
                st.metric(
                    "BCE · último balance",
                    f"{europe_trillion_eur:,.2f} T EUR",
                    f'{history["BCE"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {europe["date"].strftime("%d/%m/%Y")}')
        with j1:
            if japan:
                japan_trillion_yen = japan["value"] * 0.0001
                st.metric(
                    "Banco de Japón · último dato",
                    f"{japan_trillion_yen:,.2f} T JPY",
                    f'{history["BoJ"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {japan["date"].strftime("%d/%m/%Y")}')
        with j2:
            if china:
                china_trillion_cny = china["value"] * 0.0001
                st.metric(
                    "China M2 · último dato",
                    f"{china_trillion_cny:,.2f} T CNY",
                    f'{history["China M2"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {china["date"].strftime("%d/%m/%Y")}')

        rate1, rate2 = st.columns([1, 2])
        with rate1:
            st.metric("Tipo de depósito BCE", f"{ecb_rate.iloc[-1]:.2f}%")
            st.caption(f'Fecha del tipo: {ecb_rate.index[-1].strftime("%d/%m/%Y")}')
        with rate2:
            st.line_chart(ecb_rate, y_label="Porcentaje")
            st.caption("Evolución semanal del tipo de la facilidad de depósito del BCE.")

        us1, us2 = st.columns([1, 2])
        with us1:
            st.metric("Fed Funds efectivo", f"{fed_rate.iloc[-1]:.2f}%")
            st.caption(f'Fecha del tipo: {fed_rate.index[-1].strftime("%d/%m/%Y")}')
        with us2:
            st.line_chart(fed_rate, y_label="Porcentaje")
            st.caption("Evolución semanal del tipo efectivo de los fondos federales de EE. UU.")

        st.line_chart(history[["GLI"]], y_label="Billones USD")
        with st.expander("Composición global"):
            st.area_chart(history[["FED", "BCE", "BoJ", "China M2"]], y_label="Billones USD")
        with st.expander("Tendencia GLI · EMA 10, 20, 50 y 200", expanded=True):
            st.line_chart(trends, y_label="Base 100")

        with st.expander("Vista TradingView · Global Liquidity Index", expanded=True):
            st.caption(
                "Adaptación nativa basada en la metodología pública del indicador de "
                "QuantitativeAlpha. Conserva nuestras fuentes y fechas de actualización."
            )
            available_components = [
                "Balance FED", "Cuenta del Tesoro (TGA)", "Reverse Repo",
                "Balance BCE", "Balance BoJ", "Oferta monetaria EE. UU.",
                "Oferta monetaria China",
            ]
            selected_components = st.multiselect(
                "Componentes del índice",
                available_components,
                default=["Balance FED", "Balance BCE", "Balance BoJ", "Oferta monetaria China"],
                key="tradingview_gli_components",
            )
            st.caption(
                "TGA y Reverse Repo se restan; el resto se suma. Pendientes de una fuente estable: "
                "balance PBoC, otros bancos centrales y oferta monetaria de Europa y Japón."
            )
            tv1, tv2, tv3 = st.columns([2, 1, 1])
            mode = tv1.selectbox(
                "Lectura",
                ["Nivel", "Variación interanual", "Variación 6 meses", "Variación 3 meses", "Variación mensual"],
                key="tradingview_gli_mode",
            )
            smoothing = tv2.selectbox(
                "Suavizado", [1, 2, 4, 8, 13], index=2,
                format_func=lambda value: "Sin suavizado" if value == 1 else f"{value} semanas",
                key="tradingview_gli_smoothing",
            )
            offset = tv3.selectbox(
                "Desplazamiento", [0, 30, 60, 90],
                format_func=lambda value: "Sin desplazar" if value == 0 else f"{value} días",
                key="tradingview_gli_offset",
            )
            if not selected_components:
                st.warning("Selecciona al menos un componente para calcular el índice.")
            else:
                custom_gli = build_custom_gli(history, selected_components)
                tv_series = build_tradingview_view(custom_gli, mode, smoothing, offset)
            if selected_components and tv_series.empty:
                st.info("El periodo elegido todavía no permite calcular esta variación.")
            elif selected_components:
                value = tv_series.iloc[-1]
                unit = "T USD" if mode == "Nivel" else "%"
                st.metric(f"Última lectura · {mode}", f"{value:,.2f} {unit}")
                st.line_chart(tv_series, y_label=unit)
            st.link_button(
                "Abrir indicador original en TradingView",
                "https://www.tradingview.com/script/lG8KoR4f-Global-Liquidity-Index/",
                use_container_width=True,
            )
            st.caption(
                "No ejecuta código Pine externo: aplica las mismas vistas analíticas al GLI "
                "del dashboard. El desplazamiento mueve la curva hacia delante en días naturales."
            )

        st.caption(
            "GLI ampliado = activos de FED + BCE + BoJ + China M2 como proxy, convertidos a USD. "
            "La composición separa el proxy chino para evitar confundir M2 con activos de banco central."
        )
        status_labels = []
        for provider in ("FED", "BCE", "BoJ", "China"):
            status = history[f"_{provider} status"].iloc[-1]
            status_labels.append(f"{provider}: {'en directo' if status == 'live' else 'caché local'}")
        st.caption("Estado de fuentes · " + " · ".join(status_labels))
    except Exception as error:
        st.warning(f"No se pudo calcular el GLI: {type(error).__name__}: {error}")


def draw_gli_intelligence(period: str) -> None:
    st.markdown('<div class="section-title">GLI frente a mercados y señales</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    asset_name = left.selectbox("Activo GLI", list(COMPARISON_ASSETS), key="gli_asset")
    lag = right.selectbox("Adelanto del GLI", [0, 4, 8, 12, 16], format_func=lambda x: f"{x} semanas")
    try:
        years = PERIOD_YEARS[period]
        history = load_global_liquidity(years)
        asset = load_asset_history(years, asset_name)
        comparison, correlation = compare_gli_with_asset(history["GLI"], asset, lag)
        comparison = comparison.rename(columns={"Activo": asset_name})
        st.metric("Correlación de variaciones semanales", f"{correlation:+.2f}")
        st.line_chart(comparison, y_label="Base 100")

        aligned_asset = asset.resample("W-FRI").last().ffill()
        analysis = interpret_liquidity(history["GLI"], aligned_asset)
        regime = analysis["regime"]
        if regime == "Expansivo":
            st.success(f"Régimen de liquidez: {regime}")
        elif regime == "Contractivo":
            st.error(f"Régimen de liquidez: {regime}")
        else:
            st.info(f"Régimen de liquidez: {regime}")
        for message in analysis["messages"]:
            st.write(f"• {message}")

        threshold = st.number_input(
            "Alertar si la variación de 4 semanas cae por debajo de (%)",
            min_value=-20.0, max_value=5.0, value=-1.0, step=0.5,
        )
        if analysis.get("change4", 0) <= threshold:
            st.error(f"Alerta activa: GLI {analysis['change4']:+.2f}% en cuatro semanas.")
        else:
            st.success("Sin alerta de contracción activa.")

        report = build_markdown_report(history, analysis, asset_name)
        st.download_button(
            "Descargar informe automático",
            data=report.encode("utf-8"),
            file_name=f"informe-liquidez-{date.today().isoformat()}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    except Exception as error:
        st.warning(f"No se pudo generar el análisis GLI: {type(error).__name__}: {error}")


def draw_policy_comparison(period: str) -> None:
    st.markdown('<div class="section-title">Política monetaria global</div>', unsafe_allow_html=True)
    try:
        rates, source_dates = load_policy_rates(PERIOD_YEARS[period])
        rows = []
        for name in rates.columns:
            series = rates[name].dropna()
            rows.append({
                "Banco / referencia": name,
                "Tipo actual": f"{series.iloc[-1]:.2f}%",
                "3 meses": _format_rate_change(rate_change(series, 3)),
                "6 meses": _format_rate_change(rate_change(series, 6)),
                "12 meses": _format_rate_change(rate_change(series, 12)),
                "Orientación": classify_policy(series),
                "Fecha": pd.Timestamp(source_dates[name]).strftime("%d/%m/%Y"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.line_chart(rates, y_label="Porcentaje")

        china = get_china_lpr_history().iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("China LPR 1 año", f'{china["China LPR 1 año"]:.2f}%')
        c2.metric("China LPR 5 años", f'{china["China LPR 5 años"]:.2f}%')
        st.caption(
            "FED: tipo efectivo · BCE: facilidad de depósito · BoJ: call rate overnight · "
            "China: Loan Prime Rate. Los cambios se expresan en puntos porcentuales."
        )
    except Exception as error:
        st.warning(f"No se pudo cargar la comparativa de tipos: {type(error).__name__}: {error}")


def _format_rate_change(value: float) -> str:
    return "Sin datos" if pd.isna(value) else f"{value:+.2f} pp"


def draw_connection_status(market_data: dict, fred_data: dict, net: dict) -> None:
    with st.expander("Estado de las conexiones"):
        errors = {
            **{f"Mercado · {k}": v["error"] for k, v in market_data.items() if v.get("error")},
            **{f"FRED · {k}": v["error"] for k, v in fred_data.items() if v.get("error")},
        }
        if net.get("error"):
            errors["Cálculo · US Net Liquidity"] = net["error"]
        if not errors:
            st.success("Yahoo Finance, FRED y el cálculo de liquidez funcionan correctamente.")
        else:
            st.warning("Algunos indicadores no se pudieron descargar o calcular.")
            for name, error in errors.items():
                st.write(f"**{name}:** {error}")


def draw_dashboard(section: str = "Resumen", period: str = "3 años") -> None:
    draw_header(section)
    market_data = load_market_data()
    fred_data = load_fred_data()
    if section == "Resumen":
        net = draw_summary(market_data, fred_data)
    else:
        net = calculate_us_net_liquidity(
            fred_data["Balance FED"], fred_data["TGA"], fred_data["Reverse Repo"]
        )

    if section == "Resumen":
        draw_global_liquidity(period)
        draw_policy_comparison(period)
    elif section == "Liquidez global":
        draw_global_liquidity(period)
        draw_fed_cards(fred_data)
        draw_history(period)
        draw_gli_intelligence(period)
    elif section == "Política monetaria":
        draw_policy_comparison(period)
    elif section == "Mercados":
        draw_market_cards(market_data)
        draw_fear_greed(period)
        draw_market_comparison(period)
        draw_gli_intelligence(period)
    elif section == "Datos y diagnóstico":
        draw_connection_status(market_data, fred_data, net)
        st.caption("Registro local: cache/diagnostics.log")

    st.caption(
        "US Net Liquidity = Balance FED − TGA − Reverse Repo. "
        "Aproximación orientativa; no es una medida oficial de la Reserva Federal."
    )
    if section != "Datos y diagnóstico":
        draw_connection_status(market_data, fred_data, net)
