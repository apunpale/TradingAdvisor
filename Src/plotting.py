# Src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd

def equity_curve(portfolio):
    dates = [h["date"] for h in portfolio.history]
    values = [h["portfolio_value"] for h in portfolio.history]
    return pd.Series(values, index=pd.to_datetime(dates))

def drawdown_series(equity: pd.Series):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd

def plot_equity_and_drawdown(portfolio):
    eq = equity_curve(portfolio)
    dd = drawdown_series(eq)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(eq.index, eq.values, label="Equity Curve", color="blue")
    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(dd.index, dd.values, label="Drawdown", color="red")
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_price_with_signals(df, portfolio, ticker):
    close = df["Close"][ticker]

    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for day in portfolio.history:
        sig = day["signals"].get(ticker)
        if isinstance(sig, dict):
            if sig.get("buy"):
                buy_dates.append(day["date"])
                buy_prices.append(close.loc[day["date"]])
            if sig.get("sell"):
                sell_dates.append(day["date"])
                sell_prices.append(close.loc[day["date"]])
        else:
            if sig == "BUY":
                buy_dates.append(day["date"])
                buy_prices.append(close.loc[day["date"]])
            elif sig == "SELL":
                sell_dates.append(day["date"])
                sell_prices.append(close.loc[day["date"]])

    plt.figure(figsize=(14, 6))
    plt.plot(close.index, close.values, label=f"{ticker} Close", color="blue")

    plt.scatter(buy_dates, buy_prices, marker="^", color="green", s=80, label="BUY")
    plt.scatter(sell_dates, sell_prices, marker="v", color="red", s=80, label="SELL")

    plt.title(f"{ticker} Price with Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
