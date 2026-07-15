import streamlit as st


def draw_sidebar() -> tuple[str, str]:

    st.sidebar.title("🌍 Global Liquidity Monitor")

    st.sidebar.markdown("---")

    st.sidebar.subheader("Navegación")

    section = st.sidebar.radio(
        "Selecciona una sección",
        [
            "Resumen",
            "Liquidez global",
            "Política monetaria",
            "Mercados",
            "Riesgo macro y crédito",
            "Datos y diagnóstico",
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.info("Versión 0.4 · Informes y escenarios")

    period = st.sidebar.selectbox(
        "Periodo histórico",
        ["1 año", "3 años", "5 años", "10 años"],
        index=1,
    )

    if st.sidebar.button("Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    return section, period
