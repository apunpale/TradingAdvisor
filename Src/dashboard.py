import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Src.config import load_tickers, add_ticker
from Src.data_loader import backfill_ticker, load_panel_from_tickers, refresh_all_tickers
from Src.backtest import (
    run_backtest,
    equity_curve,
    drawdown_series,
    total_return,
    cagr,
    max_drawdown,
    sharpe_ratio,
    best_trade,
    worst_trade,
)


def main():
    st.title("Multi‑Ticker MA20 Crossover Backtest")

    # --- Sidebar: Ticker management ---
    st.sidebar.header("Tickers")

    tickers = load_tickers()
    st.sidebar.write("Current tickers:", ", ".join(tickers) if tickers else "None")

    new_ticker = st.sidebar.text_input("Add new ticker")
    if st.sidebar.button("Backfill new ticker from 2018"):
        if new_ticker:
            df_new = backfill_ticker(new_ticker)
            if df_new is not None:
                add_ticker(new_ticker)
                st.sidebar.success(f"Backfilled and added {new_ticker.upper()}")
            else:
                st.sidebar.error("No data for ticker")
        else:
            st.sidebar.warning("Enter a ticker symbol first")

    if st.sidebar.button("Refresh all tickers (2018‑today)"):
        refresh_all_tickers()
        st.sidebar.success("Refreshed all tickers")

    # --- Load panel ---
    panel = load_panel_from_tickers(load_tickers())
    if panel is None or panel.empty:
        st.error("No data available. Add and backfill tickers first.")
        return

    available_tickers = sorted({col[1] for col in panel.columns if col[0] == "Close"})

    # --- Backtest configuration ---
    st.header("Backtest configuration")

    selected_tickers = st.multiselect(
        "Select tickers for backtesting",
        options=available_tickers,
        default=available_tickers[:2] if len(available_tickers) >= 2 else available_tickers,
    )

    initial_cash = st.number_input("Initial investment (£)", value=5000, step=500)
    start_date = st.date_input("Start date", value=datetime(2018, 1, 2))
    end_date = st.date_input("End date", value=datetime.today())

    if not selected_tickers:
        st.warning("Select at least one ticker to run a backtest.")
        return

    if st.button("Run backtest"):
        run_and_display(panel, selected_tickers, initial_cash, start_date, end_date)


def run_and_display(panel, tickers, initial_cash, start_date, end_date):
    st.subheader("Backtest results")

    portfolio = run_backtest(
        panel,
        tickers=tickers,
        initial_cash=initial_cash,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
    )

    eq = equity_curve(portfolio)
    dd = drawdown_series(eq)

    tr = total_return(portfolio)
    cg = cagr(portfolio)
    mdd = max_drawdown(eq)
    sh = sharpe_ratio(eq)
    bt = best_trade(portfolio)
    wt = worst_trade(portfolio)

    col1, col2, col3 = st.columns(3)
    col1.metric("Final portfolio value", f"£{eq.iloc[-1]:,.0f}" if not eq.empty else "£0")
    col2.metric("Total return", f"{tr*100:.1f}%" if tr is not None else "0.0%")
    col3.metric("CAGR", f"{cg*100:.1f}%" if cg is not None else "0.0%")

    col4, col5 = st.columns(2)
    col4.metric("Max drawdown", f"{mdd*100:.1f}%")
    col5.metric("Sharpe ratio", f"{sh:.2f}")

    if bt is not None:
        st.markdown("**Best trade**")
        st.write(
            f"{bt['ticker']} | {bt['side']} | {bt['date']} | "
            f"Price: {bt['price']:.2f} | Value: £{bt['value']:.2f}"
        )

    if wt is not None:
        st.markdown("**Worst trade**")
        st.write(
            f"{wt['ticker']} | {wt['side']} | {wt['date']} | "
            f"Price: {wt['price']:.2f} | Value: £{wt['value']:.2f}"
        )

    st.subheader("Equity curve")
    if not eq.empty:
        st.line_chart(eq)

    st.subheader("Drawdown")
    if not dd.empty:
        st.area_chart(dd)

    # Per‑ticker price view
    st.subheader("Per‑ticker time series")
    ticker = st.selectbox("Select ticker", tickers)
    if ticker:
        close = panel["Close"][ticker]
        st.line_chart(close)

        with st.expander("Show daily momentum scores for this ticker"):
            rows = []
            for h in portfolio.history:
                rows.append(
                    {
                        "Date": h.date,
                        "Portfolio value": h.portfolio_value,
                        "Momentum score": h.signals.get(ticker, 0.0),
                    }
                )
            df_signals = pd.DataFrame(rows).set_index("Date")
            st.dataframe(df_signals)


if __name__ == "__main__":
    main()
