import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from Src.config import DATA_PATH, load_tickers


# Folder where individual ticker CSVs are stored
TICKER_FOLDER = os.path.join(DATA_PATH, "Tickers")


def ensure_ticker_folder():
    """Ensure the ticker folder exists."""
    os.makedirs(TICKER_FOLDER, exist_ok=True)


def backfill_ticker(
    ticker: str,
    start: str = "2018-01-01",
    end: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Download historical data for a ticker and save it as CSV
    with only: Date, Close, High, Low, Open, Volume.
    """

    os.makedirs(TICKER_FOLDER, exist_ok=True)

    ticker = ticker.strip().upper()

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"No data returned for {ticker}")
        return None

# Fix multi-index columns created by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index so Date becomes a column
    df = df.reset_index()

    # Select only the required columns
    keep_cols = ["Date", "Close", "High", "Low", "Open", "Volume"]
    df = df[keep_cols]

    path = os.path.join(TICKER_FOLDER, f"{ticker}.csv")
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
    Load all tickers into a single multi-index DataFrame.
    Structure:
        Close | AAPL
        Close | AMZN
        ...
    """
    ensure_ticker_folder()

    series = []

    for t in tickers:
        df = load_ticker_timeseries(t)
        if df is None or df.empty:
            continue

        # Keep only OHLCV
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # MultiIndex columns: (ColumnName, Ticker)
        df.columns = pd.MultiIndex.from_product([df.columns, [t]])

        series.append(df)

    if not series:
        return None

    panel = pd.concat(series, axis=1).sort_index()
    return panel
