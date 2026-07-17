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

PERIOD_YEARS = {"1 año": 1, "3 años": 3, "5 años": 5, "10 años": 10}
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


@st.cache_data(ttl=3600)
def load_macro_credit_risk(years: int) -> dict:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 3))
    return calculate_macro_credit_risk({
        "Curva 10Y–2Y": get_fred_history(YIELD_CURVE_10Y2Y, start),
        "Spread high yield": get_fred_history(HIGH_YIELD_SPREAD, start),
        "NFCI": get_fred_history(FINANCIAL_CONDITIONS, start),
        "IPC": get_fred_history(US_CPI, start),
        "Desempleo": get_fred_history(US_UNEMPLOYMENT, start),
        "Producción industrial": get_fred_history(US_INDUSTRIAL_PRODUCTION, start),
    })


@st.cache_data(ttl=3600)
def load_index_analyses(years: int) -> tuple[dict, dict]:
    # La EMA 200 mensual necesita aproximadamente 17 años de contexto.
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
        "S&P 500 · SPY": SPY,
        "S&P 500 equiponderado · RSP": RSP,
    }, start)
    comparison = pd.concat(histories, axis=1).dropna()
    return comparison.divide(comparison.iloc[0]).multiply(100)


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


def _regime_gauge(score: float, label: str, date_value: pd.Timestamp) -> None:
    value = max(0.0, min(100.0, float(score)))
    angle = -90 + value * 1.8
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">Régimen de mercado</div>
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
    st.markdown('<div class="section-title">Régimen de mercado · puntuación 0–100</div>', unsafe_allow_html=True)
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
            "75–100 Risk-on fuerte · 55–74 Risk-on moderado · 45–54 Neutral · "
            "25–44 Risk-off · 0–24 Crisis de liquidez. Es una lectura orientativa, no una recomendación."
        )
    except Exception as error:
        st.warning(f"No se pudo calcular el régimen de mercado: {type(error).__name__}: {error}")


def _macro_risk_gauge(score: float, label: str, date_value: pd.Timestamp) -> None:
    value = max(0.0, min(100.0, float(score)))
    angle = -90 + value * 1.8
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">Riesgo macro y de crédito</div>
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
    st.markdown('<div class="section-title">Riesgo macroeconómico y de crédito</div>', unsafe_allow_html=True)
    try:
        result = load_macro_credit_risk(PERIOD_YEARS[period])
        _macro_risk_gauge(result["score"], result["classification"], result["date"])

        st.markdown("#### Factores que forman el indicador")
        st.dataframe(result["details"], hide_index=True, use_container_width=True)

        st.markdown("#### Evolución del riesgo agregado")
        st.line_chart(result["history"], y_label="Riesgo 0–100")

        credit_tab, conditions_tab, macro_tab = st.tabs([
            "Crédito y curva", "Condiciones financieras", "Ciclo macro",
        ])
        data = result["data"]
        with credit_tab:
            c1, c2 = st.columns(2)
            c1.metric("Spread high yield", f'{data["Spread high yield"].iloc[-1]:.2f}%')
            c2.metric("Curva 10Y–2Y", f'{data["Curva 10Y–2Y"].iloc[-1]:+.2f} pp')
            st.line_chart(data[["Spread high yield", "Curva 10Y–2Y"]], y_label="Porcentaje")
            st.caption("Un spread high yield creciente señala mayor tensión crediticia. Una curva invertida eleva el riesgo cíclico.")
        with conditions_tab:
            st.metric("Chicago Fed NFCI", f'{data["NFCI"].iloc[-1]:.2f}')
            st.line_chart(data[["NFCI"]], y_label="Índice")
            st.caption("NFCI positivo = condiciones más tensas que la media; negativo = condiciones más laxas.")
        with macro_tab:
            macro = data[["Inflación interanual", "Desempleo", "Producción industrial interanual"]]
            st.line_chart(macro, y_label="Porcentaje")
            st.caption("La brecha de desempleo compara su media reciente con el mínimo del último año.")

        st.caption(
            "0–24 riesgo bajo · 25–44 moderado · 45–59 elevado · 60–79 alto · 80–100 extremo. "
            "Fuentes: FRED, Reserva Federal de Chicago, Tesoro de EE. UU. e ICE BofA."
        )
    except Exception as error:
        st.warning(f"No se pudo calcular el riesgo macro y de crédito: {type(error).__name__}: {error}")


def _index_breadth_gauge(score: float, label: str) -> None:
    value = max(0.0, min(100.0, float(score)))
    angle = -90 + value * 1.8
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #293449;border-radius:18px;padding:18px 18px 12px;text-align:center">
      <div style="font-size:1.15rem;font-weight:700;color:#e5e7eb;margin-bottom:8px">Amplitud y confirmación global</div>
      <div style="position:relative;max-width:520px;height:245px;margin:auto;overflow:hidden">
        <div style="position:absolute;width:100%;aspect-ratio:1;left:0;top:0;border-radius:50%;background:conic-gradient(from 270deg,#dc2626 0deg 36deg,#f97316 36deg 72deg,#eab308 72deg 108deg,#84cc16 108deg 144deg,#16a34a 144deg 180deg,transparent 180deg)"></div>
        <div style="position:absolute;width:72%;aspect-ratio:1;left:14%;top:14%;border-radius:50%;background:#111827"></div>
        <div style="position:absolute;width:5px;height:145px;background:#f8fafc;left:calc(50% - 2px);bottom:0;transform-origin:50% 100%;transform:rotate({angle:.1f}deg);border-radius:4px;box-shadow:0 0 8px #000"></div>
        <div style="position:absolute;width:20px;height:20px;background:#f8fafc;border-radius:50%;left:calc(50% - 10px);bottom:-10px"></div>
        <div style="position:absolute;left:0;right:0;bottom:28px;font-size:3rem;font-weight:800;color:#f8fafc">{value:.0f}</div>
      </div>
      <div style="font-size:1.15rem;font-weight:800;color:#e5e7eb">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _ema_chart(frame: pd.DataFrame, years: int, title: str):
    visible_start = pd.Timestamp(date.today()) - pd.DateOffset(years=years)
    visible = frame.loc[frame.index >= visible_start].copy()
    visible.index.name = "Fecha"
    long = visible.reset_index().melt("Fecha", var_name="Serie", value_name="Valor").dropna()
    price = alt.Chart(long[long["Serie"] == "Precio"]).mark_line(
        color="#f8fafc", strokeWidth=2.8,
    ).encode(
        x=alt.X("Fecha:T", title=None), y=alt.Y("Valor:Q", title="Nivel"),
        tooltip=[alt.Tooltip("Fecha:T"), alt.Tooltip("Valor:Q", format=",.2f")],
    )
    emas = alt.Chart(long[long["Serie"] != "Precio"]).mark_line(
        strokeDash=[7, 5], strokeWidth=1.6,
    ).encode(
        x=alt.X("Fecha:T", title=None), y=alt.Y("Valor:Q", title="Nivel"),
        color=alt.Color(
            "Serie:N",
            scale=alt.Scale(
                domain=[f"EMA {period}" for period in EMA_PERIODS],
                range=["#22d3ee", "#a78bfa", "#f59e0b", "#fb7185", "#22c55e"],
            ),
            legend=alt.Legend(orient="bottom", columns=5),
        ),
        tooltip=["Serie:N", alt.Tooltip("Fecha:T"), alt.Tooltip("Valor:Q", format=",.2f")],
    )
    return (price + emas).properties(height=430, title=title).interactive()


def _draw_tradingview_index(name: str) -> None:
    symbol = TRADINGVIEW_INDICES[name]
    configuration = json.dumps({
        "autosize": True,
        "symbol": symbol,
        "interval": "D",
        "timezone": "Europe/Madrid",
        "theme": "dark",
        "style": "1",
        "locale": "es",
        "withdateranges": True,
        "hide_side_toolbar": False,
        "allow_symbol_change": True,
        "save_image": False,
        "studies": [],
        "show_popup_button": True,
        "popup_width": "1200",
        "popup_height": "700",
        "support_host": "https://www.tradingview.com",
    })
    widget = f"""
    <div class="tradingview-widget-container" style="height:620px;width:100%">
      <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {configuration}
      </script>
    </div>
    """
    components.html(widget, height=640, scrolling=False)
    st.link_button(
        f"Abrir {name} en TradingView",
        f"https://www.tradingview.com/chart/?symbol={quote_plus(symbol)}",
        use_container_width=True,
    )


def _draw_indices_report(context: dict) -> None:
    st.markdown('<div class="section-title">Informe automático y escenarios</div>', unsafe_allow_html=True)
    report = build_section_report("Índices y amplitud", context)
    st.info(report["situation"])
    with st.expander("Señales que sustentan el diagnóstico", expanded=True):
        for signal in report["signals"]:
            st.write(f"• {signal}")
    scenarios = pd.DataFrame(report["scenarios"])
    scenarios["Probabilidad"] = scenarios["Probabilidad"].map(lambda value: f"{value}%")
    st.dataframe(scenarios, hide_index=True, use_container_width=True)
    st.caption("Estimaciones heurísticas derivadas de tendencia, amplitud y confirmación; no son recomendaciones de inversión.")
    st.download_button(
        "Descargar informe PDF",
        data=build_report_pdf("Índices y amplitud", report),
        file_name=f"informe-indices-amplitud-{date.today().isoformat()}.pdf",
        mime="application/pdf", use_container_width=True,
        key="download_report_pdf_indices_amplitud",
    )


def draw_indices_breadth(period: str) -> None:
    st.markdown('<div class="section-title">Índices globales · tendencia y amplitud</div>', unsafe_allow_html=True)
    years = PERIOD_YEARS[period]
    analyses, errors = load_index_analyses(years)
    if not analyses:
        st.warning("No se pudo cargar ninguno de los índices globales.")
        return

    rows = []
    for name, result in analyses.items():
        latest = result["daily"]["Precio"].iloc[-1]
        signal = "Alcista" if result["score"] >= 70 else "Neutral" if result["score"] >= 40 else "Bajista"
        rows.append({
            "Índice": name, "Último": f"{latest:,.2f}",
            "Diario": result["daily_score"], "Semanal": result["weekly_score"],
            "Mensual": result["monthly_score"], "Confirmación": result["score"],
            "Tendencia": signal,
        })
    summary = pd.DataFrame(rows)
    index_score = float(summary["Confirmación"].mean())

    load_full_breadth = st.toggle(
        "Calcular amplitud completa de las 500 empresas del S&P 500",
        value=False,
        help="La primera descarga puede tardar entre 45 y 90 segundos; después se conserva seis horas en caché.",
        key="load_full_sp500_breadth",
    )
    breadth, breadth_error = None, "Activa el cálculo completo para ver los componentes internos."
    if load_full_breadth:
        try:
            with st.spinner("Calculando amplitud de los componentes del S&P 500..."):
                breadth = load_sp500_breadth_data()
        except Exception as error:
            breadth_error = f"{type(error).__name__}: {error}"

    equal_weight, equal_error = None, None
    try:
        equal_weight = load_equal_weight_comparison(years)
    except Exception as error:
        equal_error = f"{type(error).__name__}: {error}"

    breadth_score = index_score
    breadth_values = {"Sobre EMA 20": index_score, "Sobre EMA 50": index_score, "Sobre EMA 200": index_score}
    new_highs = new_lows = 0
    if breadth:
        breadth_values = breadth["breadth"].iloc[-1].to_dict()
        breadth_score = float(pd.Series(breadth_values).mean())
        new_highs = int(breadth["highs_lows"]["Nuevos máximos"].iloc[-1])
        new_lows = int(breadth["highs_lows"]["Nuevos mínimos"].iloc[-1])

    rsp_relative = 100.0
    equal_score = 50.0
    if equal_weight is not None and not equal_weight.empty:
        rsp_relative = float(equal_weight.iloc[-1, 1] / equal_weight.iloc[-1, 0] * 100)
        equal_score = 70.0 if rsp_relative >= 100 else 35.0
    global_score = index_score * 0.50 + breadth_score * 0.35 + equal_score * 0.15
    label = "Amplitud fuerte" if global_score >= 70 else "Amplitud saludable" if global_score >= 55 else "Amplitud neutral" if global_score >= 40 else "Amplitud débil"
    _index_breadth_gauge(global_score, label)

    st.markdown("#### Confirmación entre índices y temporalidades")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    selected = st.selectbox("Índice para analizar", list(analyses), key="selected_global_index")
    result = analyses[selected]
    daily_tab, weekly_tab, monthly_tab = st.tabs(["Diario", "Semanal", "Mensual"])
    with daily_tab:
        st.altair_chart(_ema_chart(result["daily"], years, f"{selected} · diario"), use_container_width=True)
    with weekly_tab:
        st.altair_chart(_ema_chart(result["weekly"], years, f"{selected} · semanal"), use_container_width=True)
    with monthly_tab:
        st.altair_chart(_ema_chart(result["monthly"], years, f"{selected} · mensual"), use_container_width=True)
    st.caption("Precio en línea continua. EMA 10, 20, 34, 50 y 200 en líneas discontinuas; todas calculadas sobre su propia temporalidad.")

    with st.expander("Gráfico oficial de TradingView", expanded=False):
        _draw_tradingview_index(selected)

    st.markdown("#### Divergencias detectadas")
    divergences = []
    for name, item in analyses.items():
        time_scores = [item["daily_score"], item["weekly_score"], item["monthly_score"]]
        if max(time_scores) - min(time_scores) >= 40:
            divergences.append(f"{name}: las temporalidades no confirman la misma tendencia ({item['daily_score']}/{item['weekly_score']}/{item['monthly_score']}).")
        if item["score"] <= index_score - 25:
            divergences.append(f"{name}: debilidad relativa frente al conjunto global.")
    if rsp_relative < 98:
        divergences.append("RSP rinde claramente peor que SPY: el avance depende más de las compañías de mayor capitalización.")
    if not divergences:
        st.success("No se observan divergencias relevantes entre índices, temporalidades o ponderaciones.")
    else:
        for message in divergences:
            st.warning(message)

    st.markdown("#### Amplitud interna del S&P 500")
    if breadth:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Sobre EMA 20", f'{breadth_values["Sobre EMA 20"]:.1f}%')
        b2.metric("Sobre EMA 50", f'{breadth_values["Sobre EMA 50"]:.1f}%')
        b3.metric("Sobre EMA 200", f'{breadth_values["Sobre EMA 200"]:.1f}%')
        b4.metric("Cobertura", f'{breadth["coverage"]} empresas')
        st.line_chart(breadth["breadth"], y_label="Porcentaje")
        ad_col, hl_col = st.columns(2)
        with ad_col:
            st.markdown("##### Línea avance/descenso")
            st.line_chart(breadth["advance_decline"])
        with hl_col:
            st.markdown("##### Nuevos máximos y mínimos")
            st.line_chart(breadth["highs_lows"])
            st.caption(f"Última lectura: {new_highs} máximos y {new_lows} mínimos de 52 semanas.")
    else:
        st.warning(f"Amplitud por componentes no disponible: {breadth_error}")

    st.markdown("#### S&P 500 tradicional frente al equiponderado")
    if equal_weight is not None:
        st.line_chart(equal_weight, y_label="Base 100")
        st.caption(f"RSP relativo a SPY desde el inicio del periodo: {rsp_relative:.1f}. Por debajo de 100 indica liderazgo más concentrado.")
    else:
        st.warning(f"Comparación SPY/RSP no disponible: {equal_error}")

    for name, error in errors.items():
        st.warning(f"{name}: {error}")

    _draw_indices_report({
        "global_breadth_score": global_score,
        "breadth_label": label,
        "index_score": index_score,
        "breadth_20": float(breadth_values["Sobre EMA 20"]),
        "breadth_50": float(breadth_values["Sobre EMA 50"]),
        "breadth_200": float(breadth_values["Sobre EMA 200"]),
        "rsp_relative": rsp_relative,
        "new_highs": new_highs,
        "new_lows": new_lows,
        "divergence_count": len(divergences),
    })


@st.cache_data(ttl=1800)
def load_report_context(years: int) -> dict:
    start = pd.Timestamp(date.today()) - pd.DateOffset(years=max(years, 3))
    gli = load_global_liquidity(years)["GLI"].dropna()
    sp500 = load_asset_history(years, "S&P 500").dropna()
    vix = get_market_history(VIX, start).dropna()
    dxy = get_market_history(DXY, start).dropna()
    fed_rate = load_fed_rate_history(years).dropna()
    sentiment = load_sp500_fear_greed(years).dropna()
    market = calculate_market_regime(gli, sp500, vix, dxy, fed_rate, sentiment)
    macro = load_macro_credit_risk(years)
    macro_data = macro["data"].iloc[-1]
    trends = build_gli_trends(gli)
    sp500_weekly = sp500.resample("W-FRI").last().ffill().dropna()
    dxy_weekly = dxy.resample("W-FRI").last().ffill().dropna()
    fed_position = max(0, len(fed_rate) - 14)
    return {
        "market_score": market["score"],
        "market_regime": market["regime"],
        "macro_score": macro["score"],
        "macro_regime": macro["classification"],
        "gli_latest": float(gli.iloc[-1]),
        "gli_change4": float(gli.pct_change(4).iloc[-1] * 100),
        "gli_above_ema200": bool(trends["GLI"].iloc[-1] >= trends["EMA 200"].iloc[-1]),
        "sp500_above_ema200": bool(sp500_weekly.iloc[-1] >= sp500_weekly.ewm(span=200, adjust=False).mean().iloc[-1]),
        "vix": float(vix.iloc[-1]),
        "dxy_change12": float(dxy_weekly.pct_change(12).iloc[-1] * 100),
        "fed_rate": float(fed_rate.iloc[-1]),
        "fed_rate_change": float(fed_rate.iloc[-1] - fed_rate.iloc[fed_position]),
        "sentiment": float(sentiment.iloc[-1]),
        "inflation": float(macro_data["Inflación interanual"]),
        "unemployment": float(macro_data["Desempleo"]),
        "hy_spread": float(macro_data["Spread high yield"]),
        "nfci": float(macro_data["NFCI"]),
        "yield_curve": float(macro_data["Curva 10Y–2Y"]),
    }


def draw_generated_report(section: str, period: str) -> None:
    st.markdown('<div class="section-title">Informe automático y escenarios</div>', unsafe_allow_html=True)
    try:
        report = build_section_report(section, load_report_context(PERIOD_YEARS[period]))
        st.info(report["situation"])
        with st.expander("Señales que sustentan el diagnóstico", expanded=True):
            for signal in report["signals"]:
                st.write(f"• {signal}")
        scenarios = pd.DataFrame(report["scenarios"])
        scenarios["Probabilidad"] = scenarios["Probabilidad"].map(lambda value: f"{value}%")
        st.dataframe(scenarios, hide_index=True, use_container_width=True)
        st.caption(
            "Las probabilidades suman 100% y son estimaciones heurísticas derivadas de los indicadores actuales; "
            "no son probabilidades estadísticas calibradas ni recomendaciones de inversión."
        )
        slug = {
            "Resumen": "resumen", "Liquidez global": "liquidez-global",
            "Política monetaria": "politica-monetaria", "Mercados": "mercados",
            "Riesgo macro y crédito": "riesgo-macro-credito",
        }[section]
        pdf_data = build_report_pdf(section, report)
        st.download_button(
            "Descargar informe PDF",
            data=pdf_data,
            file_name=f"informe-{slug}-{date.today().isoformat()}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"download_report_pdf_{slug}",
        )
    except Exception as error:
        st.warning(f"No se pudo generar el informe: {type(error).__name__}: {error}")


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


@st.cache_data(ttl=3600)
def load_us_macro_history() -> pd.DataFrame:
    return build_us_macro_history(12)


@st.cache_data(ttl=21600)
def load_bls_calendar() -> pd.DataFrame:
    return get_bls_release_calendar(14)


def draw_us_economy() -> None:
    st.markdown('<div class="section-title">Últimos datos de la economía de EEUU</div>', unsafe_allow_html=True)
    st.caption("Inflación, empleo, salarios y actividad. Histórico oficial de los últimos 12 meses.")
    try:
        history = load_us_macro_history()
        latest = latest_us_macro(history)
        labels = [
            "IPC interanual", "IPC subyacente interanual", "Nóminas no agrícolas",
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
                column.caption(f'Dato de {item["date"].strftime("%m/%Y")} · variación frente al dato anterior')

        st.markdown('<div class="section-title">Tabla de los últimos 12 meses</div>', unsafe_allow_html=True)
        table = history.reset_index().rename(columns={"index": "Mes"})
        table["Mes"] = table["Mes"].dt.strftime("%m/%Y")
        st.dataframe(
            table.style.format({
                "IPC interanual": "{:.1f}%",
                "IPC subyacente interanual": "{:.1f}%",
                "Nóminas no agrícolas": "{:+,.0f} mil",
                "Desempleo": "{:.1f}%",
                "Salarios interanual": "{:.1f}%",
                "Ventas minoristas mensual": "{:+.1f}%",
                "Producción industrial interanual": "{:+.1f}%",
            }, na_rep="—"),
            hide_index=True, use_container_width=True,
        )

        chart_data = history[["IPC interanual", "IPC subyacente interanual", "Salarios interanual"]]
        st.line_chart(chart_data, y_label="Variación interanual (%)")
        source_status = history.attrs.get("source_status", {})
        cached = [name for name, status in source_status.items() if status != "live"]
        st.caption("Fuente: FRED / BLS." + (f" En caché: {', '.join(cached)}." if cached else " Datos en directo."))
    except Exception as error:
        st.warning(f"No se pudieron cargar los datos económicos: {type(error).__name__}: {error}")

    st.markdown('<div class="section-title">Próximas publicaciones</div>', unsafe_allow_html=True)
    try:
        calendar = load_bls_calendar()
        if calendar.empty:
            st.info("El calendario oficial no ha publicado todavía nuevas fechas.")
        else:
            shown = calendar.copy()
            shown["Fecha"] = shown["Fecha"].dt.strftime("%d/%m/%Y · %H:%M ET")
            st.dataframe(shown, hide_index=True, use_container_width=True)
            next_release = calendar.iloc[0]
            days = max(0, (next_release["Fecha"].normalize() - pd.Timestamp.today().normalize()).days)
            st.info(f'Próxima: {next_release["Publicación"]} · {next_release["Fecha"].strftime("%d/%m/%Y")} · dentro de {days} días.')
        st.link_button("Abrir calendario oficial del BLS", "https://www.bls.gov/schedule/", use_container_width=True)
        st.caption("Horas expresadas en horario del Este de EEUU (ET). El BLS puede revisar las fechas.")
    except Exception as error:
        st.warning(f"No se pudo consultar el calendario BLS: {type(error).__name__}: {error}")
        st.link_button("Consultar calendario oficial", "https://www.bls.gov/schedule/", use_container_width=True)


def draw_dashboard(section: str = "Resumen", period: str = "3 años") -> None:
    draw_header(section)
    market_data = load_market_data() if section in {"Resumen", "Mercados", "Datos y diagnóstico"} else {}
    fred_data = load_fred_data() if section in {"Resumen", "Liquidez global", "Datos y diagnóstico"} else {}
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
    elif section == "Política monetaria":
        draw_policy_comparison(period)
    elif section == "Mercados":
        draw_market_cards(market_data)
        draw_market_regime(period)
        draw_fear_greed(period)
        draw_market_comparison(period)
        draw_gli_intelligence(period)
    elif section == "Riesgo macro y crédito":
        draw_macro_credit_risk(period)
    elif section == "Índices y amplitud":
        draw_indices_breadth(period)
    elif section == "Economía EEUU":
        draw_us_economy()
    elif section == "Datos y diagnóstico":
        draw_connection_status(market_data, fred_data, net)
        st.caption("Registro local: cache/diagnostics.log")

    if section in {
        "Resumen", "Liquidez global", "Política monetaria", "Mercados",
        "Riesgo macro y crédito",
    }:
        draw_generated_report(section, period)

    st.caption(
        "US Net Liquidity = Balance FED − TGA − Reverse Repo. "
        "Aproximación orientativa; no es una medida oficial de la Reserva Federal."
    )
