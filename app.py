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

section, period = draw_sidebar()
draw_dashboard(section=section, period=period)
