from pathlib import Path

import streamlit as st

# Debe ejecutarse antes de importar los clientes FRED/Yahoo.
from utils.network import remove_dead_local_proxy

remove_dead_local_proxy()

from modules.dashboard import draw_dashboard
from modules.sidebar import draw_sidebar


def load_css(file_path: str) -> None:
    css_path = Path(file_path)

    if not css_path.exists():
        st.warning(f"No se encontró el archivo CSS: {file_path}")
        return

    css = css_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Global Liquidity Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css("assets/style.css")


def _render_section(section: str) -> None:
    period = draw_sidebar()
    draw_dashboard(section=section, period=period)


def page_summary() -> None:
    _render_section("Resumen")


def page_liquidity() -> None:
    _render_section("Liquidez global")


def page_policy() -> None:
    _render_section("Política monetaria")


def page_markets() -> None:
    _render_section("Mercados")


def page_macro_risk() -> None:
    _render_section("Riesgo macro y crédito")


def page_indices() -> None:
    _render_section("Índices y amplitud")


def page_diagnostics() -> None:
    _render_section("Datos y diagnóstico")


navigation = st.navigation([
    st.Page(page_summary, title="Resumen", icon="🏠", default=True),
    st.Page(page_liquidity, title="Liquidez global", icon="🌊", url_path="liquidez-global"),
    st.Page(page_policy, title="Política monetaria", icon="🏦", url_path="politica-monetaria"),
    st.Page(page_markets, title="Mercados", icon="📈", url_path="mercados"),
    st.Page(page_macro_risk, title="Riesgo macro y crédito", icon="⚠️", url_path="riesgo-macro-credito"),
    st.Page(page_indices, title="Índices y amplitud", icon="🌐", url_path="indices-amplitud"),
    st.Page(page_diagnostics, title="Datos y diagnóstico", icon="🔧", url_path="datos-diagnostico"),
], expanded=True)
navigation.run()
