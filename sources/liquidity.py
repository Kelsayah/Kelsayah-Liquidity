import pandas as pd


def calculate_us_net_liquidity(
    fed_balance: dict,
    tga: dict,
    reverse_repo: dict,
) -> dict:
    """
    Calcula la liquidez neta aproximada de EE. UU.

    Fórmula:
    Balance FED - TGA - Reverse Repo

    Todas las cifras se convierten a miles de millones de USD.
    """

    errors = []

    if fed_balance.get("error"):
        errors.append(f"Balance FED: {fed_balance['error']}")

    if tga.get("error"):
        errors.append(f"TGA: {tga['error']}")

    if reverse_repo.get("error"):
        errors.append(f"Reverse Repo: {reverse_repo['error']}")

    if errors:
        return {
            "value": None,
            "previous": None,
            "change": None,
            "change_pct": None,
            "error": " | ".join(errors),
        }

    try:
        # WALCL y WTREGEN vienen en millones de USD.
        fed_current = fed_balance["value"] / 1000
        fed_previous = fed_balance["previous"] / 1000

        tga_current = tga["value"] / 1000
        tga_previous = tga["previous"] / 1000

        # RRPONTSYD ya viene en miles de millones de USD.
        rrp_current = reverse_repo["value"]
        rrp_previous = reverse_repo["previous"]

        current = fed_current - tga_current - rrp_current
        previous = fed_previous - tga_previous - rrp_previous

        change = current - previous

        if previous == 0:
            change_pct = 0.0
        else:
            change_pct = (change / previous) * 100

        return {
            "value": current,
            "previous": previous,
            "change": change,
            "change_pct": change_pct,
            "error": None,
        }

    except Exception as error:
        return {
            "value": None,
            "previous": None,
            "change": None,
            "change_pct": None,
            "error": f"{type(error).__name__}: {error}",
        }


# Alias compatible con el nombre utilizado por versiones anteriores del dashboard.
calculate_net_liquidity = calculate_us_net_liquidity


def build_us_net_liquidity_history(
    fed_balance: pd.Series,
    tga: pd.Series,
    reverse_repo: pd.Series,
) -> pd.DataFrame:
    """Alinea las fuentes a cierre semanal antes de calcular liquidez neta."""
    frame = pd.concat(
        {
            "Balance FED": fed_balance / 1000,
            "TGA": tga / 1000,
            "Reverse Repo": reverse_repo,
        },
        axis=1,
        sort=False,
    ).sort_index()

    # Las publicaciones tienen calendarios distintos. El cierre semanal con
    # forward-fill evita mezclar arbitrariamente el último punto de cada serie.
    weekly = frame.resample("W-FRI").last().ffill().dropna()
    weekly["US Net Liquidity"] = (
        weekly["Balance FED"] - weekly["TGA"] - weekly["Reverse Repo"]
    )
    return weekly
