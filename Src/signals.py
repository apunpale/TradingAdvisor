from __future__ import annotations

from typing import Dict, List

import pandas as pd


def compute_ma20_signals_for_day(
    panel: pd.DataFrame,
    date,
    tickers: List[str],
) -> Dict[str, dict]:
    """
    For each ticker, compute:
      - price_today
      - ma20_today
      - ma20_yesterday
      - buy_signal (bool)
      - sell_signal (bool)
      - momentum_strength (float)
    """
    signals: Dict[str, dict] = {}

    if date not in panel.index:
        return signals

    # restrict to data up to 'date'
    panel_to_date = panel.loc[:date]

    for t in tickers:
        if ("Close", t) not in panel_to_date.columns:
            continue

        close = panel_to_date["Close"][t]
        if len(close) < 21:
            continue

        ma20 = close.rolling(20).mean()
        price_today = close.iloc[-1]
        price_yesterday = close.iloc[-2]
        ma20_today = ma20.iloc[-1]
        ma20_yesterday = ma20.iloc[-2]

        buy_signal = price_today > ma20_today and price_yesterday <= ma20_yesterday
        sell_signal = price_today < ma20_today and price_yesterday >= ma20_yesterday

        momentum_strength = 0.0
        if ma20_today > 0:
            momentum_strength = (price_today - ma20_today) / ma20_today

        signals[t] = {
            "price": float(price_today),
            "ma20_today": float(ma20_today),
            "ma20_yesterday": float(ma20_yesterday),
            "buy": bool(buy_signal),
            "sell": bool(sell_signal),
            "momentum_strength": float(momentum_strength),
        }

    return signals
