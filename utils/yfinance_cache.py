from pathlib import Path

import yfinance as yf


def configure_yfinance_cache() -> Path:
    cache_dir = Path(__file__).resolve().parents[1] / "cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    return cache_dir
