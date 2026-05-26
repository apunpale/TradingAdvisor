import os

# Base data folder
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
TICKERS_FILE = os.path.join(DATA_PATH, "tickers.txt")
RESTRICTED_FILE = os.path.join(DATA_PATH, "restricted_list.txt")


def ensure_data_path():
    os.makedirs(DATA_PATH, exist_ok=True)


def load_tickers(path: str = TICKERS_FILE) -> list[str]:
    ensure_data_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip().upper() for line in f if line.strip()]


def add_ticker(ticker: str, path: str = TICKERS_FILE) -> None:
    ensure_data_path()
    ticker = ticker.strip().upper()
    tickers = set(load_tickers(path))
    if ticker not in tickers:
        with open(path, "a") as f:
            f.write(ticker + "\n")


def load_restricted_list(path: str = RESTRICTED_FILE) -> set[str]:
    ensure_data_path()
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip().upper() for line in f if line.strip()}
