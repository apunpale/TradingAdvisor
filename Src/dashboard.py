import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Src.config import load_tickers, add_ticker
from Src.data_loader import (
    backfill_ticker,
    load_panel_from_tickers,
    refresh_all_tickers,
)
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

# -----------------------------
# Benchmark storage locations
# -----------------------------
BENCHMARK_FILE = os.path.join(ROOT, "Data", "benchmarks.txt")
BENCHMARK_FOLDER = os.path.join(ROOT, "Data", "Benchmarks")
os.makedirs(BENCHMARK_FOLDER, exist_ok=True)


# -----------------------------
# Benchmark helpers
# -----------------------------
def load_benchmarks():
    if not os.path.exists(BENCHMARK_FILE):
        return []
    with open(BENCHMARK_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def add_benchmark(ticker: str):
    ticker = ticker.upper().strip()
    benchmarks = load_benchmarks()
    if ticker not in benchmarks:
        with open(BENCHMARK_FILE, "a") as f:
            f.write(ticker + "\n")


def backfill_benchmark(ticker: str):
    ticker = ticker.upper().strip()
    df = backfill_ticker(
        ticker,
        start="2018-01-01",
        end=datetime.today().strftime("%Y-%m-%d"),
        folder_override=BENCHMARK_FOLDER,
    )
    return df


def refresh_all_benchmarks():
    for t in load_benchmarks():
        backfill_benchmark(t)


def benchmark_return(panel, start_date, end_date, ticker):
    """
    Compute benchmark return using the Close price series.
    """
    try:
        series = panel["Close"][ticker]
    except KeyError:
        return None

    # Restrict to date range
    series = series.loc[(series.index >= start_date) & (series.index <= end_date)]

    if series.empty:
        return None

    start_price = series.iloc[0]
    end_price = series.iloc[-1]

    return (end_price - start_price) / start_price



def rolling_alpha_beta(strategy, benchmark, window=60):
    df = pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).dropna()
    returns = df.pct_change().dropna()

    cov = returns["strategy"].rolling(window).cov(returns["benchmark"])
    var = returns["benchmark"].rolling(window).var()

    beta = cov / var
    alpha = (
        returns["strategy"].rolling(window).mean()
        - beta * returns["benchmark"].rolling(window).mean()
    )

    return alpha, beta


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="MA20 Multi‑Ticker Backtest", layout="wide")
    st.title("📈 Multi‑Ticker MA20 Crossover Backtest")

    # -----------------------------
    # Sidebar: Tickers
    # -----------------------------
    st.sidebar.header("📊 Trading Tickers")

    tickers = load_tickers()
    st.sidebar.write("Current tickers:", ", ".join(tickers) if tickers else "None")

    new_ticker = st.sidebar.text_input("Add new ticker")
    if st.sidebar.button("Backfill new ticker"):
        if new_ticker:
            df_new = backfill_ticker(new_ticker)
            if df_new is not None:
                add_ticker(new_ticker)
                st.sidebar.success(f"Added & backfilled {new_ticker.upper()}")
            else:
                st.sidebar.error("No data for ticker")
        else:
            st.sidebar.warning("Enter a ticker symbol first")

    if st.sidebar.button("Refresh all tickers"):
        refresh_all_tickers()
        st.sidebar.success("Refreshed all tickers")

    # -----------------------------
    # Sidebar: Benchmarks
    # -----------------------------
    st.sidebar.header("📌 Benchmarks")

    benchmarks = load_benchmarks()
    st.sidebar.write("Current benchmarks:", ", ".join(benchmarks) if benchmarks else "None")

    new_bench = st.sidebar.text_input("Add benchmark ticker")
    if st.sidebar.button("Backfill benchmark"):
        if new_bench:
            df_b = backfill_benchmark(new_bench)
            if df_b is not None:
                add_benchmark(new_bench)
                st.sidebar.success(f"Added & backfilled benchmark {new_bench.upper()}")
            else:
                st.sidebar.error("No data for benchmark")
        else:
            st.sidebar.warning("Enter a benchmark ticker")

    if st.sidebar.button("Refresh all benchmarks"):
        refresh_all_benchmarks()
        st.sidebar.success("Refreshed all benchmarks")

    # -----------------------------
    # Load panel (tickers + benchmarks)
    # -----------------------------
    all_tickers = load_tickers()
    all_benchmarks = load_benchmarks()

    panel = load_panel_from_tickers(all_tickers + all_benchmarks)
    if panel is None or panel.empty:
        st.error("No data available. Add and backfill tickers first.")
        return

    # -----------------------------
    # Backtest configuration
    # -----------------------------
    st.header("⚙️ Backtest Configuration")

    selected_tickers = st.multiselect(
        "Select tickers for backtesting",
        options=all_tickers,
        default=all_tickers[:2] if len(all_tickers) >= 2 else all_tickers,
    )

    benchmark_choice = st.selectbox(
        "Select benchmark",
        options=all_benchmarks,
        index=0 if all_benchmarks else None,
    )

    initial_cash = st.number_input("Initial investment (£)", value=5000, step=500)
    start_date = st.date_input("Start date", value=datetime(2018, 1, 2))
    end_date = st.date_input("End date", value=datetime.today())

    if st.button("Run backtest"):
        run_and_display(
            panel,
            selected_tickers,
            benchmark_choice,
            initial_cash,
            start_date,
            end_date,
        )


# -----------------------------
# Backtest + Display
# -----------------------------
def run_and_display(panel, tickers, benchmark_ticker, initial_cash, start_date, end_date):
    st.subheader("📊 Backtest Results")

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

    bench = benchmark_return(panel, pd.to_datetime(start_date), pd.to_datetime(end_date), benchmark_ticker)
    beat = (bench is not None and tr is not None and tr > bench)

    # -----------------------------
    # Metrics
    # -----------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final value", f"£{eq.iloc[-1]:,.0f}")
    col2.metric("Total return", f"{tr*100:.1f}%")
    col3.metric("CAGR", f"{cg*100:.1f}%")
    col4.metric("Max drawdown", f"{mdd*100:.1f}%")

    col5, col6 = st.columns(2)
    col5.metric("Sharpe ratio", f"{sh:.2f}")
    if bench is not None:
        col6.metric("Benchmark return", f"{bench*100:.1f}%", "Beat" if beat else "Lagged")
    else:
        col6.metric("Benchmark return", "N/A")

    # -----------------------------
    # Relative performance chart
    # -----------------------------
    st.subheader("📈 Relative Performance (Normalised)")

    bench_series = panel["Close"][benchmark_ticker].loc[eq.index]
    bench_norm = bench_series / bench_series.iloc[0]
    eq_norm = eq / eq.iloc[0]

    rel_df = pd.DataFrame({"Strategy": eq_norm, benchmark_ticker: bench_norm})
    st.line_chart(rel_df)

    # -----------------------------
    # Rolling alpha/beta
    # -----------------------------
    st.subheader("📉 Rolling Alpha & Beta (60‑day)")

    alpha, beta = rolling_alpha_beta(eq, bench_series)

    st.write("**Rolling Alpha**")
    st.line_chart(alpha)

    st.write("**Rolling Beta**")
    st.line_chart(beta)

    # -----------------------------
    # Per‑ticker vs benchmark table
    # -----------------------------
    rows = []
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    for t in tickers:
        prices = panel["Close"][t]

        # Align dates safely
        try:
            start_p = prices.loc[prices.index >= start_ts].iloc[0]
            end_p = prices.loc[prices.index <= end_ts].iloc[-1]
        except Exception:
            continue  # skip tickers with missing data

        ret = (end_p - start_p) / start_p

        rows.append({
            "Ticker": t,
            "Return %": ret * 100,
            "Beat Benchmark": ret > bench if bench is not None else None,
        })

    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare)


    # -----------------------------
    # Equity curve
    # -----------------------------
    st.subheader("📈 Equity Curve")
    st.line_chart(eq)

    # -----------------------------
    # Drawdown
    # -----------------------------
    st.subheader("📉 Drawdown")
    st.area_chart(dd)

    # -----------------------------
    # Per‑ticker time series
    # -----------------------------
    st.subheader("📊 Per‑Ticker Time Series")
    ticker = st.selectbox("Select ticker", tickers)
    if ticker:
        close = panel["Close"][ticker]
        st.line_chart(close)

        with st.expander("Show daily momentum scores"):
            rows = []
            for h in portfolio.history:
                rows.append({
                    "Date": h.date,
                    "Portfolio value": h.portfolio_value,
                    "Momentum score": h.signals.get(ticker, 0.0),
                })
            df_signals = pd.DataFrame(rows).set_index("Date")
            st.dataframe(df_signals)


if __name__ == "__main__":
    main()
