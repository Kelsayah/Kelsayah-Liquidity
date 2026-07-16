Exit code: 0
Wall time: 0.5 seconds
Total output lines: 1197
Output:
from datetime import date
import json
from urllib.parse import quote_plus

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
from sources.market_regime import calculate_market_regime
from sources.macro_credit_risk import calculate_macro_credit_risk
from sources.section_reports import build_section_report
from sources.report_pdf import build_report_pdf
from sources.index_breadth import (
    EMA_PERIODS, build_index_technical, download_index_histories,
    download_sp500_breadth,
)
from sources.us_economy import build_us_macro_history, get_bls_release_calendar, latest_us_macro
from sources.policy_rates import (
    build_policy_comparison, classify_policy, get_china_lpr_history, rate_change,
)
from sources.signals import build_markdown_report, compare_gli_with_asset, interpret_liquidity
from sources.sentiment import (
    classify_fear_greed, get_crypto_fear_greed_history,
    get_sp500_fear_greed_history,
)
from utils.constants import (
    BITCOIN, BOJ_BALANCE, BOJ_CALL_RATE, DAX, DXY, ECB_BALANCE, ECB_DEPOSIT_RATE,
    EURUSD, FED_BALANCE, FED_FUNDS_RATE, GOLD, IBEX35, KOSPI, M2, NASDAQ, REVERSE_REPO,
    SP500, TGA, USDCNY, USDJPY, US10Y, VIX, FINANCIAL_CONDITIONS,
    HIGH_YIELD_SPREAD, US_CPI, US_INDUSTRIAL_PRODUCTION, US_UNEMPLOYMENT,
    YIELD_CURVE_10Y2Y, RSP, SHANGHAI, SPY,
)

PERIOD_YEARS = {"1 aÃ±o": 1, "3 aÃ±os": 3, "5 aÃ±os": 5, "10 aÃ±os": 10}
COMPARISON_ASSETS = {
    "S&P 500": SP500,
    "Nasdaq": NASDAQ,
    "Bitcoin": BITCOIN,
    "Oro": GOLD,
}
GLOBAL_INDICES = {
    "S&P 500": SP500,
    "Nasdaq Composite": NASDAQ,
    "IBEX 35": IBEX35,
    "DAX": DAX,
    "Shanghai Composite": SHANGHAI,
    "KOSPI": KOSPI,
}
TRADINGVIEW_INDICES = {
    "S&P 500": "SP:SPX",
    "Nasdaq Composite": "NASDAQ:IXIC",
    "IBEX 35": "BME:IBC",
    "DAX": "XETR:DAX",
    "Shanghai Composite": "SSE:000001",
    "KOSPI": "KRX:KOSPI",
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
    # Columnas internas constantes: sobreviven a la serializaciÃ³n de la cachÃ©.
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
    return rate.resample("W-FRI").last().ffill().dropna().rename("Tipo depÃ³sito BCE")


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
    china_1y = china_frame["China LPR 1 aÃ±o"].rename("China LPR 1A")
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


@st.cache_data(ttl=3600)
def load_macro_credit_risk(years: int) -> dict:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 3))
    return calculate_macro_credit_risk({
        "Curva 10Yâ€“2Y": get_fred_history(YIELD_CURVE_10Y2Y, start),
        "Spread high yield": get_fred_history(HIGH_YIELD_SPREAD, start),
        "NFCI": get_fred_history(FINANCIAL_CONDITIONS, start),
        "IPC": get_fred_history(US_CPI, start),
        "Desempleo": get_fred_history(US_UNEMPLOYMENT, start),
        "ProducciÃ³n industrial": get_fred_history(US_INDUSTRIAL_PRODUCTION, start),
    })


@st.cache_data(ttl=3600)
def load_index_analyses(years: int) -> tuple[dict, dict]:
    # La EMA 200 mensual necesita aproximadamente 17 aÃ±os de contexto.
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=20)
    analyses, errors = {}, {}
    try:
        histories = download_index_histories(GLOBAL_INDICES, start)
    except Exception as error:
        return {}, {"Descarga global": f"{type(error).__name__}: {error}"}
    for name, history in histories.items():
        try:
            analyses[name] = build_index_technical(history)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    return analyses, errors


@st.cache_data(ttl=21600)
def load_sp500_breadth_data() -> dict:
    return download_sp500_breadth()


@st.cache_data(ttl=3600)
def load_equal_weight_comparison(years: int) -> pd.DataFrame:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 3))
    histories = download_index_histories({
        "S&P 500 Â· SPY": SPY,
        "S&P 500 equiponderado Â· RSP": RSP,
    }, start)
    comparison = pd.concat(histories, axis=1).dropna()
    return comparison.divide(comparison.iloc[0]).multiply(100)


def draw_header(section: str) -> None:
    st.markdown(f"""
        <div class="dashboard-header">
            <h1 class="dashboard-title">ðŸŒ Global Liquidity Monitor</h1>
            <p class="dashboard-subtitle">{section} Â· liquidez y condiciones financieras</p>
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
      <div style="font-size:.78rem;color:#94a3b8;margin-top:5px">{history.index[-1].strftime('%d/%m/%Y')} Â· {provider} Â· {'En directo' if status == 'live' else 'CachÃ© local'}</div>
    </div>
    """, unsafe_allow_html=True)


def _regime_gauge(score: float, label: str, date_value: pd.Timestamp) -> None:
    value = max(0.0, min(100.0, float(score)))
    angle = -90 + value * 1.8
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">RÃ©gimen de mercado</div>
      <div style="position:relative;max-width:520px;height:245px;margin:auto;overflow:hidden">
        <div style="position:absolute;width:100%;aspect-ratio:1;left:0;top:0;border-radius:50%;background:conic-gradient(from 270deg,#991b1b 0deg 36deg,#dc2626 36deg 72deg,#eab308 72deg 108deg,#65a30d 108deg 144deg,#16a34a 144deg 180deg,transparent 180deg)"></div>
        <div style="position:absolute;width:72%;aspect-ratio:1;left:14%;top:14%;border-radius:50%;background:#111827"></div>
        <div style="position:absolute;width:5px;height:145px;background:#f8fafc;left:calc(50% - 2px);bottom:0;transform-origin:50% 100%;transform:rotate({angle:.1f}deg);border-radius:4px;box-shadow:0 0 8px #000"></div>
        <div style="position:absolute;width:20px;height:20px;background:#f8fafc;border-radius:50%;left:calc(50% - 10px);bottom:-10px"></div>
        <div style="position:absolute;left:0;right:0;bottom:28px;font-size:3rem;font-weight:800;color:#f8fafc">{value:.0f}</div>
      </div>
      <div style="font-size:1.15rem;font-weight:800;color:#e5e7eb">{label}</div>
      <div style="font-size:.8rem;color:#94a3b8;margin-top:6px">Datos comunes hasta {pd.Timestamp(date_value).strftime('%d/%m/%Y')}</div>
    </div>
    """, unsafe_allow_html=True)


def draw_market_regime(period: str) -> None:
    st.markdown('<div class="section-title">RÃ©gimen de mercado Â· puntuaciÃ³n 0â€“100</div>', unsafe_allow_html=True)
    try:
        years = PERIOD_YEARS[period]
        result = calculate_market_regime(
            load_global_liquidity(years)["GLI"],
            load_asset_history(years, "S&P 500"),
            get_market_history(VIX, pd.Timestamp(date.today()) - pd.DateOffset(years=years)),
            get_market_history(DXY, pd.Timestamp(date.today()) - pd.DateOffset(years=years)),
            load_fed_rate_history(years),
            load_sp500_fear_greed(years),
        )
        _regime_gauge(result["score"], result["regime"], result["date"])
        st.dataframe(result["details"], hide_index=True, use_container_width=True)
        st.caption(
            "75â€“100 Risk-on fuerte Â· 55â€“74 Risk-on moderado Â· 45â€“54 Neutral Â· "
            "25â€“44 Risk-off Â· 0â€“24 Crisis de liquidez. Es una lectura orientativa, no una recomendaciÃ³n."
        )
    except Exception as error:
        st.warning(f"No se pudo calcular el rÃ©gimen de mercado: {type(error).__name__}: {error}")


def _macro_risk_gauge(score: float, label: str, date_value: pd.Timestamp) -> None:
    value = max(0.0, min(100.0, float(score)))
    angle = -90 + value * 1.8
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">Riesgo macro y de crÃ©dito</div>
      <div style="position:relative;max-width:520px;height:245px;margin:auto;overflow:hidden">
        <div style="position:absolute;width:100%;aspect-ratio:1;left:0;top:0;border-radius:50%;background:conic-gradient(from 270deg,#16a34a 0deg 36deg,#65a30d 36deg 72deg,#eab308 72deg 108deg,#f97316 108deg 144deg,#dc2626 144deg 180deg,transparent 180deg)"></div>
        <div style="position:absolute;width:72%;aspect-ratio:1;left:14%;top:14%;border-radius:50%;background:#111827"></div>
        <div style="position:absolute;width:5px;height:145px;background:#f8fafc;left:calc(50% - 2px);bottom:0;transform-origin:50% 100%;transform:rotate({angle:.1f}deg);border-radius:4px;box-shadow:0 0 8px #000"></div>
        <div style="position:absolute;width:20px;height:20px;background:#f8fafc;border-radius:50%;left:calc(50% - 10px);bottom:-10px"></div>
        <div style="position:absolute;left:0;right:0;bottom:28px;font-size:3rem;font-weight:800;color:#f8fafc">{value:.0f}</div>
      </div>
      <div style="font-size:1.15rem;font-weight:800;color:#e5e7eb">{label}</div>
      <div style="font-size:.8rem;color:#94a3b8;margin-top:6px">Actualizado: {pd.Timestamp(date_value).strftime('%d/%m/%Y')}</div>
    </div>
    """, unsafe_allow_html=True)


def draw_macro_credit_risk(period: str) -> None:
    st.markdown('<div class="section-title">Riesgo macroeconÃ³mico y de crÃ©dito</div>', unsafe_allow_html=True)
    try:
        result = load_macro_credit_risk(PERIOD_YEARS[period])
        _macro_risk_gauge(result["score"], result["classification"], result["date"])

        st.markdown("#### Factores que forman el indicador")
        st.dataframe(result["details"], hide_index=True, use_container_width=True)

        st.markdown("#### EvoluciÃ³n del riesgo agregado")
        st.line_chart(result["history"], y_label="Riesgo 0â€“100")

        credit_tab, conditions_tab, macro_tab = st.tabs([
            "CrÃ©dito y curva", "Condiciones financieras", "Ciclo macro",
        ])
        data = result["data"]
        with credit_tab:
            c1, c2 = st.columns(2)
            c1.metric("Spread high yield", f'{data["Spread high yield"].iloc[-1]:.2f}%')
            c2.metric("Curva 10Yâ€“2Y", f'{data["Curva 10Yâ€“2Y"].iloc[-1]:+.2f} pp')
            st.line_chart(data[["Spread high yield", "Curva 10Yâ€“2Y"]], y_label="Porcentaje")
            st.caption("Un spread high yield creciente seÃ±ala mayor tensiÃ³n crediticia. Una curva invertida eleva el riesgo cÃ­clico.")
        with conditions_tab:
            st.metric("Chicago Fed NFCI", f'{data["NFCI"].iloc[-1]:.2f}')
            st.line_chart(data[["NFCI"]], y_label="Ãndice")
            st.caption("NFCI positivo = condiciones mÃ¡s tensas que la media; negativo = condiciones mÃ¡s laxas.")
        with macro_ta…4511 tokens truncated… Repo"]])
        st.caption(
            "Cifras en miles de millones de USD. Series alineadas al cierre semanal; "
            "los valores se mantienen hasta la siguiente publicaciÃ³n disponible."
        )
    except Exception as error:
        st.warning(f"No se pudo cargar el histÃ³rico: {type(error).__name__}: {error}")


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
            "ComparaciÃ³n base 100 desde la primera fecha comÃºn. "
            "Las dos curvas muestran evoluciÃ³n relativa, no importes absolutos."
        )

        with st.expander("Tendencia del activo Â· EMA 10, 20, 50 y 200", expanded=True):
            st.line_chart(trends)
            latest = trends.iloc[-1]
            signal = "Alcista" if latest[asset_name] > latest["EMA 200"] else "Bajista"
            st.metric("Tendencia frente a EMA 200", signal)
            st.caption(
                "Medias mÃ³viles exponenciales calculadas sobre cierres semanales "
                "y expresadas en la misma escala base 100."
            )
    except Exception as error:
        st.warning(f"No se pudo construir la comparaciÃ³n: {type(error).__name__}: {error}")


def draw_global_liquidity(period: str) -> None:
    st.markdown('<div class="section-title">Global Liquidity Index Â· v1.0</div>', unsafe_allow_html=True)
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
        c2.metric("VariaciÃ³n 4 semanas", "Sin datos" if pd.isna(monthly) else f"{monthly:+.2f}%")
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
                    "BCE Â· Ãºltimo balance",
                    f"{europe_trillion_eur:,.2f} T EUR",
                    f'{history["BCE"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {europe["date"].strftime("%d/%m/%Y")}')
        with j1:
            if japan:
                japan_trillion_yen = japan["value"] * 0.0001
                st.metric(
                    "Banco de JapÃ³n Â· Ãºltimo dato",
                    f"{japan_trillion_yen:,.2f} T JPY",
                    f'{history["BoJ"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {japan["date"].strftime("%d/%m/%Y")}')
        with j2:
            if china:
                china_trillion_cny = china["value"] * 0.0001
                st.metric(
                    "China M2 Â· Ãºltimo dato",
                    f"{china_trillion_cny:,.2f} T CNY",
                    f'{history["China M2"].iloc[-1]:,.2f} T USD aprox.',
                )
                st.caption(f'Fecha del dato: {china["date"].strftime("%d/%m/%Y")}')

        rate1, rate2 = st.columns([1, 2])
        with rate1:
            st.metric("Tipo de depÃ³sito BCE", f"{ecb_rate.iloc[-1]:.2f}%")
            st.caption(f'Fecha del tipo: {ecb_rate.index[-1].strftime("%d/%m/%Y")}')
        with rate2:
            st.line_chart(ecb_rate, y_label="Porcentaje")
            st.caption("EvoluciÃ³n semanal del tipo de la facilidad de depÃ³sito del BCE.")

        us1, us2 = st.columns([1, 2])
        with us1:
            st.metric("Fed Funds efectivo", f"{fed_rate.iloc[-1]:.2f}%")
            st.caption(f'Fecha del tipo: {fed_rate.index[-1].strftime("%d/%m/%Y")}')
        with us2:
            st.line_chart(fed_rate, y_label="Porcentaje")
            st.caption("EvoluciÃ³n semanal del tipo efectivo de los fondos federales de EE. UU.")

        st.line_chart(history[["GLI"]], y_label="Billones USD")
        with st.expander("ComposiciÃ³n global"):
            st.area_chart(history[["FED", "BCE", "BoJ", "China M2"]], y_label="Billones USD")
        with st.expander("Tendencia GLI Â· EMA 10, 20, 50 y 200", expanded=True):
            st.line_chart(trends, y_label="Base 100")

        with st.expander("Vista TradingView Â· Global Liquidity Index", expanded=True):
            st.caption(
                "AdaptaciÃ³n nativa basada en la metodologÃ­a pÃºblica del indicador de "
                "QuantitativeAlpha. Conserva nuestras fuentes y fechas de actualizaciÃ³n."
            )
            available_components = [
                "Balance FED", "Cuenta del Tesoro (TGA)", "Reverse Repo",
                "Balance BCE", "Balance BoJ", "Oferta monetaria EE. UU.",
                "Oferta monetaria China",
            ]
            selected_components = st.multiselect(
                "Componentes del Ã­ndice",
                available_components,
                default=["Balance FED", "Balance BCE", "Balance BoJ", "Oferta monetaria China"],
                key="tradingview_gli_components",
            )
            st.caption(
                "TGA y Reverse Repo se restan; el resto se suma. Pendientes de una fuente estable: "
                "balance PBoC, otros bancos centrales y oferta monetaria de Europa y JapÃ³n."
            )
            tv1, tv2, tv3 = st.columns([2, 1, 1])
            mode = tv1.selectbox(
                "Lectura",
                ["Nivel", "VariaciÃ³n interanual", "VariaciÃ³n 6 meses", "VariaciÃ³n 3 meses", "VariaciÃ³n mensual"],
                key="tradingview_gli_mode",
            )
            smoothing = tv2.selectbox(
                "Suavizado", [1, 2, 4, 8, 13], index=2,
                format_func=lambda value: "Sin suavizado" if value == 1 else f"{value} semanas",
                key="tradingview_gli_smoothing",
            )
            offset = tv3.selectbox(
                "Desplazamiento", [0, 30, 60, 90],
                format_func=lambda value: "Sin desplazar" if value == 0 else f"{value} dÃ­as",
                key="tradingview_gli_offset",
            )
            if not selected_components:
                st.warning("Selecciona al menos un componente para calcular el Ã­ndice.")
            else:
                custom_gli = build_custom_gli(history, selected_components)
                tv_series = build_tradingview_view(custom_gli, mode, smoothing, offset)
            if selected_components and tv_series.empty:
                st.info("El periodo elegido todavÃ­a no permite calcular esta variaciÃ³n.")
            elif selected_components:
                value = tv_series.iloc[-1]
                unit = "T USD" if mode == "Nivel" else "%"
                st.metric(f"Ãšltima lectura Â· {mode}", f"{value:,.2f} {unit}")
                st.line_chart(tv_series, y_label=unit)
            st.link_button(
                "Abrir indicador original en TradingView",
                "https://www.tradingview.com/script/lG8KoR4f-Global-Liquidity-Index/",
                use_container_width=True,
            )
            st.caption(
                "No ejecuta cÃ³digo Pine externo: aplica las mismas vistas analÃ­ticas al GLI "
                "del dashboard. El desplazamiento mueve la curva hacia delante en dÃ­as naturales."
            )

        st.caption(
            "GLI ampliado = activos de FED + BCE + BoJ + China M2 como proxy, convertidos a USD. "
            "La composiciÃ³n separa el proxy chino para evitar confundir M2 con activos de banco central."
        )
        status_labels = []
        for provider in ("FED", "BCE", "BoJ", "China"):
            status = history[f"_{provider} status"].iloc[-1]
            status_labels.append(f"{provider}: {'en directo' if status == 'live' else 'cachÃ© local'}")
        st.caption("Estado de fuentes Â· " + " Â· ".join(status_labels))
    except Exception as error:
        st.warning(f"No se pudo calcular el GLI: {type(error).__name__}: {error}")


def draw_gli_intelligence(period: str) -> None:
    st.markdown('<div class="section-title">GLI frente a mercados y seÃ±ales</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    asset_name = left.selectbox("Activo GLI", list(COMPARISON_ASSETS), key="gli_asset")
    lag = right.selectbox("Adelanto del GLI", [0, 4, 8, 12, 16], format_func=lambda x: f"{x} semanas")
    try:
        years = PERIOD_YEARS[period]
        history = load_global_liquidity(years)
        asset = load_asset_history(years, asset_name)
        comparison, correlation = compare_gli_with_asset(history["GLI"], asset, lag)
        comparison = comparison.rename(columns={"Activo": asset_name})
        st.metric("CorrelaciÃ³n de variaciones semanales", f"{correlation:+.2f}")
        st.line_chart(comparison, y_label="Base 100")

        aligned_asset = asset.resample("W-FRI").last().ffill()
        analysis = interpret_liquidity(history["GLI"], aligned_asset)
        regime = analysis["regime"]
        if regime == "Expansivo":
            st.success(f"RÃ©gimen de liquidez: {regime}")
        elif regime == "Contractivo":
            st.error(f"RÃ©gimen de liquidez: {regime}")
        else:
            st.info(f"RÃ©gimen de liquidez: {regime}")
        for message in analysis["messages"]:
            st.write(f"â€¢ {message}")

        threshold = st.number_input(
            "Alertar si la variaciÃ³n de 4 semanas cae por debajo de (%)",
            min_value=-20.0, max_value=5.0, value=-1.0, step=0.5,
        )
        if analysis.get("change4", 0) <= threshold:
            st.error(f"Alerta activa: GLI {analysis['change4']:+.2f}% en cuatro semanas.")
        else:
            st.success("Sin alerta de contracciÃ³n activa.")

        report = build_markdown_report(history, analysis, asset_name)
        st.download_button(
            "Descargar informe automÃ¡tico",
            data=report.encode("utf-8"),
            file_name=f"informe-liquidez-{date.today().isoformat()}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    except Exception as error:
        st.warning(f"No se pudo generar el anÃ¡lisis GLI: {type(error).__name__}: {error}")


def draw_policy_comparison(period: str) -> None:
    st.markdown('<div class="section-title">PolÃ­tica monetaria global</div>', unsafe_allow_html=True)
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
                "OrientaciÃ³n": classify_policy(series),
                "Fecha": pd.Timestamp(source_dates[name]).strftime("%d/%m/%Y"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.line_chart(rates, y_label="Porcentaje")

        china = get_china_lpr_history().iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("China LPR 1 aÃ±o", f'{china["China LPR 1 aÃ±o"]:.2f}%')
        c2.metric("China LPR 5 aÃ±os", f'{china["China LPR 5 aÃ±os"]:.2f}%')
        st.caption(
            "FED: tipo efectivo Â· BCE: facilidad de depÃ³sito Â· BoJ: call rate overnight Â· "
            "China: Loan Prime Rate. Los cambios se expresan en puntos porcentuales."
        )
    except Exception as error:
        st.warning(f"No se pudo cargar la comparativa de tipos: {type(error).__name__}: {error}")


def _format_rate_change(value: float) -> str:
    return "Sin datos" if pd.isna(value) else f"{value:+.2f} pp"


def draw_connection_status(market_data: dict, fred_data: dict, net: dict) -> None:
    with st.expander("Estado de las conexiones"):
        errors = {
            **{f"Mercado Â· {k}": v["error"] for k, v in market_data.items() if v.get("error")},
            **{f"FRED Â· {k}": v["error"] for k, v in fred_data.items() if v.get("error")},
        }
        if net.get("error"):
            errors["CÃ¡lculo Â· US Net Liquidity"] = net["error"]
        if not errors:
            st.success("Yahoo Finance, FRED y el cÃ¡lculo de liquidez funcionan correctamente.")
        else:
            st.warning("Algunos indicadores no se pudieron descargar o calcular.")
            for name, error in errors.items():
                st.write(f"**{name}:** {error}")


@st.cache_data(ttl=3600)
def load_us_macro_history() -> pd.DataFrame:
    return build_us_macro_history(12)


@st.cache_data(ttl=21600)
def load_bls_calendar() -> pd.DataFrame:
    return get_bls_release_calendar(14)


def draw_us_economy() -> None:
    st.markdown('<div class="section-title">Ãšltimos datos de la economÃ­a de EEUU</div>', unsafe_allow_html=True)
    st.caption("InflaciÃ³n, empleo, salarios y actividad. HistÃ³rico oficial de los Ãºltimos 12 meses.")
    try:
        history = load_us_macro_history()
        latest = latest_us_macro(history)
        labels = [
            "IPC interanual", "IPC subyacente interanual", "NÃ³minas no agrÃ­colas",
            "Desempleo", "Salarios interanual", "Ventas minoristas mensual",
        ]
        for start in range(0, len(labels), 3):
            columns = st.columns(3)
            for column, label in zip(columns, labels[start:start + 3]):
                item = latest.get(label)
                if not item:
                    column.metric(label, "Sin datos")
                    continue
                decimals = item["decimals"]
                suffix = "%" if item["unit"] == "%" else " mil"
                value = f'{item["value"]:,.{decimals}f}{suffix}'
                delta = None if item["previous"] is None else f'{item["value"] - item["previous"]:+,.{decimals}f}'
                column.metric(label, value, delta)
                column.caption(f'Dato de {item["date"].strftime("%m/%Y")} Â· variaciÃ³n frente al dato anterior')

        st.markdown('<div class="section-title">Tabla de los Ãºltimos 12 meses</div>', unsafe_allow_html=True)
        table = history.reset_index().rename(columns={"index": "Mes"})
        table["Mes"] = table["Mes"].dt.strftime("%m/%Y")
        st.dataframe(
            table.style.format({
                "IPC interanual": "{:.1f}%",
                "IPC subyacente interanual": "{:.1f}%",
                "NÃ³minas no agrÃ­colas": "{:+,.0f} mil",
                "Desempleo": "{:.1f}%",
                "Salarios interanual": "{:.1f}%",
                "Ventas minoristas mensual": "{:+.1f}%",
                "ProducciÃ³n industrial interanual": "{:+.1f}%",
            }, na_rep="â€”"),
            hide_index=True, use_container_width=True,
        )

        chart_data = history[["IPC interanual", "IPC subyacente interanual", "Salarios interanual"]]
        st.line_chart(chart_data, y_label="VariaciÃ³n interanual (%)")
        source_status = history.attrs.get("source_status", {})
        cached = [name for name, status in source_status.items() if status != "live"]
        st.caption("Fuente: FRED / BLS." + (f" En cachÃ©: {', '.join(cached)}." if cached else " Datos en directo."))
    except Exception as error:
        st.warning(f"No se pudieron cargar los datos econÃ³micos: {type(error).__name__}: {error}")

    st.markdown('<div class="section-title">PrÃ³ximas publicaciones</div>', unsafe_allow_html=True)
    try:
        calendar = load_bls_calendar()
        if calendar.empty:
            st.info("El calendario oficial no ha publicado todavÃ­a nuevas fechas.")
        else:
            shown = calendar.copy()
            shown["Fecha"] = shown["Fecha"].dt.strftime("%d/%m/%Y Â· %H:%M ET")
            st.dataframe(shown, hide_index=True, use_container_width=True)
            next_release = calendar.iloc[0]
            days = max(0, (next_release["Fecha"].normalize() - pd.Timestamp.today().normalize()).days)
            st.info(f'PrÃ³xima: {next_release["PublicaciÃ³n"]} Â· {next_release["Fecha"].strftime("%d/%m/%Y")} Â· dentro de {days} dÃ­as.')
        st.link_button("Abrir calendario oficial del BLS", "https://www.bls.gov/schedule/", use_container_width=True)
        st.caption("Horas expresadas en horario del Este de EEUU (ET). El BLS puede revisar las fechas.")
    except Exception as error:
        st.warning(f"No se pudo consultar el calendario BLS: {type(error).__name__}: {error}")
        st.link_button("Consultar calendario oficial", "https://www.bls.gov/schedule/", use_container_width=True)


def draw_dashboard(section: str = "Resumen", period: str = "3 aÃ±os") -> None:
    draw_header(section)
    market_data = load_market_data() if section in {"Resumen", "Mercados", "Datos y diagnÃ³stico"} else {}
    fred_data = load_fred_data() if section in {"Resumen", "Liquidez global", "Datos y diagnÃ³stico"} else {}
    if section == "Resumen":
        net = draw_summary(market_data, fred_data)
    elif fred_data:
        net = calculate_us_net_liquidity(
            fred_data["Balance FED"], fred_data["TGA"], fred_data["Reverse Repo"]
        )
    else:
        net = {"error": None}

    if section == "Resumen":
        draw_global_liquidity(period)
        draw_market_regime(period)
        draw_policy_comparison(period)
    elif section == "Liquidez global":
        draw_global_liquidity(period)
        draw_fed_cards(fred_data)
        draw_history(period)
        draw_gli_intelligence(period)
    elif section == "PolÃ­tica monetaria":
        draw_policy_comparison(period)
    elif section == "Mercados":
        draw_market_cards(market_data)
        draw_market_regime(period)
        draw_fear_greed(period)
        draw_market_comparison(period)
        draw_gli_intelligence(period)
    elif section == "Riesgo macro y crÃ©dito":
        draw_macro_credit_risk(period)
    elif section == "Ãndices y amplitud":
        draw_indices_breadth(period)
    elif section == "EconomÃ­a EEUU":
        draw_us_economy()
    elif section == "Datos y diagnÃ³stico":
        draw_connection_status(market_data, fred_data, net)
        st.caption("Registro local: cache/diagnostics.log")

    if section in {
        "Resumen", "Liquidez global", "PolÃ­tica monetaria", "Mercados",
        "Riesgo macro y crÃ©dito",
    }:
        draw_generated_report(section, period)

    st.caption(
        "US Net Liquidity = Balance FED âˆ’ TGA âˆ’ Reverse Repo. "
        "AproximaciÃ³n orientativa; no es una medida oficial de la Reserva Federal."
    )

