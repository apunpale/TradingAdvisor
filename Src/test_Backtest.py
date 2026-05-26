# Src/test_backtest.py

from Src.data_loader import download_prices, save_to_csv, load_from_csv, export_ticker_timeseries
from Src.backtest import run_backtest, export_signals, export_trades
from Src.plotting import plot_price_with_signals, plot_equity_and_drawdown
from Src.config import TICKERS

def main():
    df = download_prices()
    save_to_csv(df)

    df2 = load_from_csv()

    export_ticker_timeseries(df2)

    portfolio = run_backtest(df2)

    export_signals(portfolio)
    export_trades(portfolio)

    plot_equity_and_drawdown(portfolio)

    for ticker in TICKERS:
        plot_price_with_signals(df2, portfolio, ticker)

    last = portfolio.history[-1]
    print("Final date:", last["date"])
    print("Final portfolio value:", last["portfolio_value"])
    print("Final cash:", last["cash"])
    print("Final holdings:", last["holdings"])

if __name__ == "__main__":
    main()
