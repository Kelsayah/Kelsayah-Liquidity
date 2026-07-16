from html.parser import HTMLParser
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from utils.yfinance_cache import configure_yfinance_cache


configure_yfinance_cache()


EMA_PERIODS = (10, 20, 34, 50, 200)
SP500_LIST_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class _ConstituentParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.row = []
        self.cell = []
        self.symbols = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "constituents":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.in_row, self.row = True, []
        elif self.in_row and tag == "td":
            self.in_cell, self.cell = True, []

    def handle_data(self, data):
        if self.in_cell:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.in_cell and tag == "td":
            self.row.append("".join(self.cell).strip())
            self.in_cell = False
        elif self.in_row and tag == "tr":
            if self.row:
                self.symbols.append(self.row[0].replace(".", "-"))
            self.in_row = False
        elif self.in_table and tag == "table":
            self.in_table = False


def get_sp500_symbols() -> list[str]:
    request = Request(SP500_LIST_URL, headers={"User-Agent": "GlobalLiquidityMonitor/1.0"})
    with urlopen(request, timeout=25) as response:
        parser = _ConstituentParser()
        parser.feed(response.read().decode("utf-8", errors="ignore"))
    symbols = list(dict.fromkeys(parser.symbols))
    if len(symbols) < 450:
        raise ValueError(f"Solo se encontraron {len(symbols)} componentes del S&P 500")
    return symbols


def calculate_sp500_breadth(close: pd.DataFrame) -> dict:
    close = close.sort_index().dropna(axis=1, how="all")
    valid = close.count() >= 200
    close = close.loc[:, valid]
    if close.shape[1] < 10:
        raise ValueError("No hay suficientes componentes para calcular amplitud")
    breadth = pd.DataFrame(index=close.index)
    for period in (20, 50, 200):
        ema = close.ewm(span=period, adjust=False).mean()
        breadth[f"Sobre EMA {period}"] = close.gt(ema).sum(axis=1).divide(close.notna().sum(axis=1)).multiply(100)
    returns = close.pct_change(fill_method=None)
    advances = returns.gt(0).sum(axis=1)
    declines = returns.lt(0).sum(axis=1)
    ad_line = advances.subtract(declines).cumsum().rename("Línea avance/descenso")
    rolling_high = close.rolling(252, min_periods=126).max()
    rolling_low = close.rolling(252, min_periods=126).min()
    highs_lows = pd.DataFrame({
        "Nuevos máximos": close.eq(rolling_high).sum(axis=1),
        "Nuevos mínimos": close.eq(rolling_low).sum(axis=1),
    })
    return {
        "breadth": breadth.dropna(how="all"),
        "advance_decline": ad_line,
        "highs_lows": highs_lows,
        "coverage": int(close.shape[1]),
    }


def download_sp500_breadth() -> dict:
    symbols = get_sp500_symbols()
    data = yf.download(
        symbols, period="18mo", auto_adjust=True, progress=False,
        threads=True, group_by="column", timeout=45,
    )
    if data is None or data.empty:
        raise ValueError("Yahoo Finance no devolvió componentes del S&P 500")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    return calculate_sp500_breadth(close)


def download_index_histories(symbols: dict[str, str], start) -> dict[str, pd.Series]:
    data = yf.download(
        list(symbols.values()), start=start, auto_adjust=True, progress=False,
        threads=True, group_by="column", timeout=50,
    )
    if data is None or data.empty:
        raise ValueError("Yahoo Finance no devolvió los índices globales")
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    histories = {}
    for name, symbol in symbols.items():
        if isinstance(close, pd.DataFrame) and symbol in close:
            series = close[symbol].dropna().astype(float)
        elif len(symbols) == 1:
            series = close.squeeze().dropna().astype(float)
        else:
            continue
        if not series.empty:
            series.name = symbol
            histories[name] = series
    if len(histories) < max(1, len(symbols) - 1):
        raise ValueError(f"Solo se descargaron {len(histories)} de {len(symbols)} índices")
    return histories


def build_index_technical(series: pd.Series) -> dict:
    daily = series.dropna().astype(float).sort_index().rename("Precio").to_frame()
    if len(daily) < 50:
        raise ValueError("Histórico insuficiente para calcular las EMAs")
    for period in EMA_PERIODS:
        daily[f"EMA {period}"] = daily["Precio"].ewm(span=period, adjust=False).mean()
    weekly = daily[["Precio"]].resample("W-FRI").last().ffill().dropna()
    today = pd.Timestamp.today().normalize()
    if weekly.index[-1] > today:
        weekly.index = pd.DatetimeIndex([*weekly.index[:-1], today])
    for period in EMA_PERIODS:
        weekly[f"EMA {period}"] = weekly["Precio"].ewm(span=period, adjust=False).mean()
    monthly = daily[["Precio"]].resample("ME").last().ffill().dropna()
    if monthly.index[-1] > today:
        monthly.index = pd.DatetimeIndex([*monthly.index[:-1], today])
    for period in EMA_PERIODS:
        monthly[f"EMA {period}"] = monthly["Precio"].ewm(span=period, adjust=False).mean()
    daily_score = sum(daily["Precio"].iloc[-1] >= daily[f"EMA {period}"].iloc[-1] for period in EMA_PERIODS) * 20
    weekly_score = sum(weekly["Precio"].iloc[-1] >= weekly[f"EMA {period}"].iloc[-1] for period in EMA_PERIODS) * 20
    monthly_score = sum(monthly["Precio"].iloc[-1] >= monthly[f"EMA {period}"].iloc[-1] for period in EMA_PERIODS) * 20
    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "daily_score": int(daily_score),
        "weekly_score": int(weekly_score),
        "monthly_score": int(monthly_score),
        "score": int(round((daily_score + weekly_score + monthly_score) / 3)),
    }
