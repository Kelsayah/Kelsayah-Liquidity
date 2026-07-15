import pandas as pd


def get_china_lpr_history() -> pd.DataFrame:
    """Histórico de cambios LPR desde 2023, convertido a frecuencia diaria escalonada."""
    changes = pd.DataFrame(
        [
            ("2023-06-20", 3.55, 4.20),
            ("2023-08-21", 3.45, 4.20),
            ("2024-02-20", 3.45, 3.95),
            ("2024-07-22", 3.35, 3.85),
            ("2024-10-21", 3.10, 3.60),
            ("2025-05-20", 3.00, 3.50),
            ("2026-06-22", 3.00, 3.50),
        ],
        columns=["date", "China LPR 1 año", "China LPR 5 años"],
    ).set_index("date")
    changes.index = pd.to_datetime(changes.index)
    return changes


def build_policy_comparison(series: dict[str, pd.Series]) -> pd.DataFrame:
    frame = pd.concat(series, axis=1, sort=False).sort_index()
    return frame.resample("W-FRI").last().ffill().dropna()


def rate_change(series: pd.Series, months: int) -> float:
    target = series.index[-1] - pd.DateOffset(months=months)
    previous = series.loc[series.index <= target]
    if previous.empty:
        return float("nan")
    return float(series.iloc[-1] - previous.iloc[-1])


def classify_policy(series: pd.Series) -> str:
    change = rate_change(series, 6)
    if pd.isna(change):
        return "Sin histórico"
    if change >= 0.10:
        return "Endureciendo"
    if change <= -0.10:
        return "Relajando"
    return "Estable"
