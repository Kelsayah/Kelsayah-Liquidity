import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "series"
LOG_FILE = Path(__file__).resolve().parents[1] / "cache" / "diagnostics.log"


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_").lower()


def _paths(name: str) -> tuple[Path, Path]:
    base = CACHE_DIR / _safe_name(name)
    return base.with_suffix(".csv"), base.with_suffix(".json")


def save_series(name: str, series: pd.Series, provider: str) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        csv_path, meta_path = _paths(name)
        clean = series.dropna().astype(float).sort_index()
        clean.rename("value").to_csv(csv_path, index_label="date")
        meta_path.write_text(
            json.dumps({
                "provider": provider,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "latest_observation": pd.Timestamp(clean.index[-1]).isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        # La persistencia es una protección; nunca debe invalidar un dato en directo.
        return


def load_series(name: str) -> tuple[pd.Series, dict]:
    csv_path, meta_path = _paths(name)
    if not csv_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No existe caché local para {name}")
    frame = pd.read_csv(csv_path, parse_dates=["date"])
    series = frame.set_index("date")["value"].astype(float).sort_index()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return series, meta


def mark_series(series: pd.Series, provider: str, source: str, error: str | None = None) -> pd.Series:
    series.attrs["data_status"] = {
        "provider": provider,
        "source": source,
        "error": error,
        "latest_observation": pd.Timestamp(series.index[-1]),
    }
    return series


def log_data_error(provider: str, key: str, error: Exception) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{provider}\t{key}\t{type(error).__name__}: {error}\n")
    except OSError:
        return
