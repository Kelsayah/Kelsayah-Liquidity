import streamlit as st


def draw_sidebar() -> str:

    st.sidebar.title("🌍 Global Liquidity Monitor")

    st.sidebar.markdown("---")

    st.sidebar.info("Versión 0.6 · Índices y amplitud")

    period = st.sidebar.selectbox(
        "Periodo histórico",
        ["1 año", "3 años", "5 años", "10 años"],
        index=1,
    )

    if st.sidebar.button("Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    return period
