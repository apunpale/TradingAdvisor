import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from Src.config import DATA_PATH, load_tickers
from Src.config import TICKERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder where individual ticker CSVs are stored
TICKER_FOLDER = os.path.join(DATA_PATH, "Tickers")


def ensure_ticker_folder():
    """Ensure the ticker folder exists."""
    os.makedirs(TICKER_FOLDER, exist_ok=True)


def backfill_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: Optional[str] = None,
    folder_override: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Download historical data for a ticker and save it as CSV.
    Supports overriding the output folder (used for benchmarks).
    """

    # Decide where to save
    if folder_override:
        save_folder = folder_override
    else:
        save_folder = TICKER_FOLDER

    os.makedirs(save_folder, exist_ok=True)

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    ticker = ticker.strip().upper()

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return None

    # Fix multi-index columns (MU,MU,MU issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    keep_cols = ["Date", "Close", "High", "Low", "Open", "Volume"]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"Missing columns for {ticker}: {missing}")
        return None

    df = df[keep_cols]

    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    path = os.path.join(save_folder, f"{ticker}.csv")
    df.to_csv(path, index=False)

    return df




def refresh_all_tickers(
    start: str = "2018-01-01",
    end: Optional[str] = None
) -> None:
    """
    Re-download data for all tickers in tickers.txt.
    """
    tickers = load_tickers()
    for t in tickers:
        backfill_ticker(t, start=start, end=end)


def load_ticker_timeseries(ticker: str) -> Optional[pd.DataFrame]:
    """
    Load a single ticker's CSV as a DataFrame.
    """
    ensure_ticker_folder()

    ticker = ticker.strip().upper()
    path = os.path.join(TICKER_FOLDER, f"{ticker}.csv")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df


def load_panel_from_tickers(tickers: list) -> Optional[pd.DataFrame]:
    """
    Load all tickers (trading + benchmarks) into a single multi-index DataFrame.
    Structure:
        Close | AAPL
        Close | AMZN
        Close | ^GSPC
        ...
    """

    ensure_ticker_folder()

    series = []

    for t in tickers:
        # Decide folder based on whether ticker is a benchmark
        if t.startswith("^"):
            folder = os.path.join(ROOT, "Data", "Benchmarks")
        else:
            folder = TICKER_FOLDER

        path = os.path.join(folder, f"{t}.csv")
        if not os.path.exists(path):
            print(f"Missing CSV for {t}: {path}")
            continue

        df = pd.read_csv(path, parse_dates=["Date"])
        if df is None or df.empty:
            continue

        df = df.set_index("Date")

        # Keep only OHLCV
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # MultiIndex columns: (ColumnName, Ticker)
        df.columns = pd.MultiIndex.from_product([df.columns, [t]])

        series.append(df)

    if not series:
        return None

    panel = pd.concat(series, axis=1).sort_index()
    return panel


# --- Compatibility wrappers expected by older tests/scripts ---
def download_prices() -> pd.DataFrame:
    """Compatibility: return a panel DataFrame for available tickers.
    If no tickers are configured, returns an empty DataFrame.
    """
    tickers = load_tickers() if TICKERS == [] else TICKERS
    if not tickers:
        return pd.DataFrame()
    panel = load_panel_from_tickers(tickers)
    return panel if panel is not None else pd.DataFrame()


def save_to_csv(df: pd.DataFrame, path: Optional[str] = None) -> None:
    if path is None:
        path = os.path.join(DATA_PATH, "prices.csv")
    # If given a panel (multi-index), write as-is; otherwise try normal DataFrame
    df.to_csv(path, index=False)


def load_from_csv(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(DATA_PATH, "prices.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["Date"]) 
    except Exception:
        return pd.read_csv(path)


def export_ticker_timeseries(df: pd.DataFrame) -> None:
    """Export per-ticker CSVs into the ticker folder when possible.
    Works with panels where second-level columns are tickers.
    """
    ensure_ticker_folder()
    if df is None or df.empty:
        return
    # If MultiIndex columns with tickers at level 1
    cols = df.columns
    if isinstance(cols, pd.MultiIndex) and cols.nlevels >= 2:
        tickers = cols.get_level_values(1).unique()
        for t in tickers:
            try:
                sub = df.xs(t, level=1, axis=1)
                path = os.path.join(TICKER_FOLDER, f"{t}.csv")
                sub.reset_index().to_csv(path, index=False)
            except Exception:
                continue
    else:
        # Nothing to export
        return

