from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from Src.config import load_restricted_list
from Src.portfolio import Portfolio
from Src.signals import compute_ma20_signals_for_day


def run_backtest(
    panel: pd.DataFrame,
    tickers: list[str],
    initial_cash: float,
    start_date: datetime,
    end_date: datetime,
    max_capital: float = 5_000.0,
    min_holding_days: int = 30,
) -> Portfolio:
    restricted = load_restricted_list()
    tradable = [t for t in tickers if t not in restricted]

    portfolio = Portfolio(initial_cash=initial_cash)

    dates = panel.index[(panel.index >= start_date) & (panel.index <= end_date)]

    for date in dates:
        # compute signals for this day
        day_signals = compute_ma20_signals_for_day(panel, date, tradable)

        # prices for valuation
        prices_today = {
            t: s["price"] for t, s in day_signals.items()
        }

        # SELL first (respect 30-day holding)
        for ticker in list(portfolio.holdings.keys()):
            if ticker not in day_signals:
                continue
            sig = day_signals[ticker]
            if sig["sell"] and portfolio.can_sell(ticker, date, min_holding_days):
                portfolio.sell(ticker, sig["price"], date)

        # BUY next (no limit on number of buys)
        # candidates: tickers with buy signal and not currently held
        buy_candidates = []
        for t, sig in day_signals.items():
            if not sig["buy"]:
                continue
            if t in portfolio.holdings:
                continue
            buy_candidates.append((t, sig))

        # sort by strongest momentum first
        buy_candidates.sort(
            key=lambda x: x[1]["momentum_strength"], reverse=True
        )

        for ticker, sig in buy_candidates:
            prices_today[ticker] = sig["price"]
            # current exposure
            exposure = portfolio.total_exposure(prices_today)
            remaining_capacity = max_capital - exposure
            if remaining_capacity <= 0:
                break
            if portfolio.cash <= 0:
                break

            amount = min(remaining_capacity, portfolio.cash)
            if amount <= 0:
                continue

            portfolio.buy(ticker, sig["price"], amount, date)

        # log day
        # for signals, we store momentum_strength per ticker
        signal_scores = {
            t: s["momentum_strength"] for t, s in day_signals.items()
        }
        portfolio.log_day(date, prices_today, signal_scores)

    return portfolio


def equity_curve(portfolio: Portfolio) -> pd.Series:
    dates = [h.date for h in portfolio.history]
    values = [h.portfolio_value for h in portfolio.history]
    return pd.Series(values, index=pd.to_datetime(dates), name="Equity")


def drawdown_series(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return equity
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    dd.name = "Drawdown"
    return dd


def total_return(portfolio: Portfolio) -> float:
    if not portfolio.history:
        return 0.0
    start = portfolio.history[0].portfolio_value
    end = portfolio.history[-1].portfolio_value
    if start <= 0:
        return 0.0
    return (end - start) / start


def cagr(portfolio: Portfolio) -> float:
    if not portfolio.history:
        return 0.0
    start_date = portfolio.history[0].date
    end_date = portfolio.history[-1].date
    days = (end_date - start_date).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    start = portfolio.history[0].portfolio_value
    end = portfolio.history[-1].portfolio_value
    if start <= 0 or end <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    dd = drawdown_series(equity)
    return float(dd.min())


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate / 252.0
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def best_trade(portfolio: Portfolio):
    if not portfolio.trades:
        return None
    df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    # approximate PnL per trade: SELL positive, BUY negative
    df["pnl"] = df["shares"] * df["price"] * df["side"].map({"BUY": -1, "SELL": 1})
    return df.loc[df["pnl"].idxmax()]


def worst_trade(portfolio: Portfolio):
    if not portfolio.trades:
        return None
    df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    df["pnl"] = df["shares"] * df["price"] * df["side"].map({"BUY": -1, "SELL": 1})
    return df.loc[df["pnl"].idxmin()]


def win_rate(portfolio: Portfolio) -> float:
    if not portfolio.trades:
        return 0.0
    df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    # pair BUY/SELL by ticker and time order is non-trivial; for now, count SELLs with positive value as wins
    sells = df[df["side"] == "SELL"].copy()
    if sells.empty:
        return 0.0
    # here pnl is just value; in a more detailed engine we'd track trade-level PnL
    # for now, treat all sells as "wins" if they exist
    return 1.0
