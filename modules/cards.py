from html import escape

import streamlit as st


def format_number(
    value: float | None,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    if value is None:
        return "Sin datos"

    return f"{value:,.{decimals}f}{suffix}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "Sin variación"

    arrow = "▲" if value > 0 else "▼" if value < 0 else "●"
    return f"{arrow} {value:+.2f}%"


def get_delta_class(
    value: float | None,
    inverse: bool = False,
) -> str:
    if value is None or value == 0:
        return "metric-delta-neutral"

    positive = value > 0

    if inverse:
        positive = not positive

    return (
        "metric-delta-positive"
        if positive
        else "metric-delta-negative"
    )


def draw_market_card(
    label: str,
    data: dict,
    suffix: str = "",
    decimals: int = 2,
    inverse_delta: bool = False,
) -> None:
    price = data.get("price")
    change_pct = data.get("change_pct")
    error = data.get("error")

    if error or price is None:
        value_text = "Sin datos"
        delta_text = "Conexión no disponible"
        delta_class = "metric-delta-neutral"
    else:
        value_text = format_number(
            price,
            decimals=decimals,
            suffix=suffix,
        )
        delta_text = format_delta(change_pct)
        if data.get("data_source") == "cache":
            delta_text += " · dato en caché"
        delta_class = get_delta_class(
            change_pct,
            inverse=inverse_delta,
        )

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value_text)}</div>
        <div class="{delta_class}">{escape(delta_text)}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def draw_pending_card(
    label: str,
    value: str,
    status: str,
) -> None:
    html = f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-delta-neutral">{escape(status)}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def draw_fred_card(
    label: str,
    data: dict,
    divisor: float = 1.0,
    suffix: str = "",
    decimals: int = 2,
    inverse_delta: bool = False,
) -> None:
    value = data.get("value")
    change_pct = data.get("change_pct")
    error = data.get("error")
    date = data.get("date")

    if error or value is None:
        value_text = "Sin datos"
        delta_text = "Conexión no disponible"
        delta_class = "metric-delta-neutral"
        date_text = ""
    else:
        adjusted_value = value / divisor

        value_text = format_number(
            adjusted_value,
            decimals=decimals,
            suffix=suffix,
        )

        delta_text = format_delta(change_pct)
        if data.get("data_source") == "cache":
            delta_text += " · dato en caché"

        delta_class = get_delta_class(
            change_pct,
            inverse=inverse_delta,
        )

        date_text = (
            date.strftime("%d/%m/%Y")
            if date is not None
            else ""
        )

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value_text)}</div>
        <div class="{delta_class}">{escape(delta_text)}</div>
        <div style="
            margin-top: 8px;
            color: #64748b;
            font-size: 12px;
        ">
            {escape(date_text)}
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def draw_liquidity_card(
    label: str,
    data: dict,
    suffix: str = " B USD",
    decimals: int = 1,
) -> None:
    value = data.get("value")
    change = data.get("change")
    change_pct = data.get("change_pct")
    error = data.get("error")

    if error or value is None:
        value_text = "Sin datos"
        delta_text = "Cálculo no disponible"
        delta_class = "metric-delta-neutral"
    else:
        value_text = format_number(
            value,
            decimals=decimals,
            suffix=suffix,
        )

        if change is None:
            delta_text = "Sin variación"
        else:
            delta_text = (
                f"{change:+,.1f} B USD · "
                f"{change_pct:+.2f}%"
            )

        delta_class = get_delta_class(change_pct)

    html = f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value_text)}</div>
        <div class="{delta_class}">{escape(delta_text)}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
