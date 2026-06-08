from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from Src.config import load_restricted_list
from Src.portfolio import Portfolio
from Src.signals import compute_ma20_signals_for_day


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


def run_backtest(
    panel: pd.DataFrame,
    tickers: list[str] | None = None,
    initial_cash: float | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    monthly_contribution: float = 0.0,
    max_capital: float = 5_000.0,
    min_holding_days: int = 30,
    broker_fee_per_transaction: float = 0.0,
    # new knobs for more pragmatic behaviour
    momentum_scale: float = 0.10,          # how strong momentum must be to reach full per‑ticker allocation
    loss_cut_threshold: float = -0.08,     # strong negative momentum → treat as “loss‑cut” condition
    trend_reversal_threshold: float = -0.02,  # mild negative momentum + sell signal → “trend reversal” exit
) -> Portfolio:
    """Run MA20 backtest with momentum‑scaled position sizing and more nuanced sell logic.

    Backwards compatible: older callers may pass only `panel` — in that
    case we infer tickers, cash and date range from available data.
    """
    restricted = load_restricted_list()

    # Backwards-compatible behaviour: allow older callers to pass only `panel`.
    if tickers is None:
        try:
            tickers = list(panel.columns.get_level_values(1).unique())
        except Exception:
            from Src.config import load_tickers

            tickers = load_tickers()

    if initial_cash is None:
        initial_cash = 5_000.0

    # limit dates to panel bounds when not provided
    if start_date is None:
        try:
            start_date = panel.index.min()
        except Exception:
            start_date = datetime(2018, 1, 2)
    if end_date is None:
        try:
            end_date = panel.index.max()
        except Exception:
            end_date = datetime.today()

    tradable = [t for t in tickers if t not in restricted]

    portfolio = Portfolio(initial_cash=initial_cash)

    dates = panel.index[
        (panel.index >= pd.to_datetime(start_date))
        & (panel.index <= pd.to_datetime(end_date))
    ]

    # per‑ticker capital cap (used for scaling in)
    per_ticker_cap = max_capital / max(len(tradable), 1)

    seen_months = set()
    for date in dates:
        # monthly contribution: add on the first trading day of each calendar month
        month_key = (int(date.year), int(date.month))
        if month_key not in seen_months:
            if monthly_contribution and monthly_contribution > 0:
                portfolio.contribute(monthly_contribution, date)
            seen_months.add(month_key)

        # compute signals for this day
        day_signals = compute_ma20_signals_for_day(panel, date, tradable)

        # prices for valuation
        prices_today = {t: s["price"] for t, s in day_signals.items()}

        # -------------------------
        # SELL logic (full exits)
        # -------------------------
        # We still sell full positions (Portfolio API is full‑sell),
        # but we distinguish between:
        #   - strong negative momentum (loss‑cut)
        #   - mild negative momentum + MA20 sell signal (trend reversal)
        for ticker in list(portfolio.holdings.keys()):
            if ticker not in day_signals:
                continue
            sig = day_signals[ticker]
            ms = sig.get("momentum_strength", 0.0)

            if not portfolio.can_sell(ticker, date, min_holding_days):
                continue

            # Loss‑cut: momentum very negative
            if ms <= loss_cut_threshold:
                portfolio.sell(ticker, sig["price"], date, fee=broker_fee_per_transaction)
                continue

            # Trend reversal: MA20 sell signal + mildly negative momentum
            if sig.get("sell") and ms <= trend_reversal_threshold:
                portfolio.sell(ticker, sig["price"], date, fee=broker_fee_per_transaction)
                continue

        # -------------------------
        # BUY logic (scaled by momentum)
        # -------------------------
        # Candidates: tickers with buy signal (we allow re‑entry after sells)
        buy_candidates = []
        for t, sig in day_signals.items():
            if not sig.get("buy"):
                continue
            buy_candidates.append((t, sig))

        # sort by strongest momentum first
        buy_candidates.sort(
            key=lambda x: x[1].get("momentum_strength", 0.0), reverse=True
        )

        for ticker, sig in buy_candidates:
            price = sig["price"]
            prices_today[ticker] = price

            # current exposure across all tickers
            exposure = portfolio.total_exposure(prices_today)
            remaining_capacity = max_capital - exposure
            if remaining_capacity <= 0:
                break
            if portfolio.cash <= 0:
                break

            # scale target allocation by momentum strength
            ms = max(sig.get("momentum_strength", 0.0), 0.0)
            if momentum_scale <= 0:
                scale = 1.0
            else:
                scale = min(ms / momentum_scale, 1.0)

            target_value = per_ticker_cap * scale

            # if we already hold some, compute current value
            pos = portfolio.holdings.get(ticker)
            current_shares = pos.shares if pos else 0.0
            current_value = current_shares * price

            buy_amount = target_value - current_value
            if buy_amount <= 0:
                continue

            if portfolio.cash <= broker_fee_per_transaction:
                continue

            available_cash = portfolio.cash - broker_fee_per_transaction
            amount = min(buy_amount, remaining_capacity, available_cash)
            if amount <= 0:
                continue

            portfolio.buy(ticker, price, amount, date, fee=broker_fee_per_transaction)

        # log day
        portfolio.log_day(date, prices_today, day_signals)

    return portfolio


def run_backtest_with_benchmark(
    panel,
    tickers,
    initial_cash,
    start_date,
    end_date,
    monthly_contribution: float = 0.0,
    max_capital=5_000.0,
    min_holding_days=30,
):
    portfolio = run_backtest(
        panel=panel,
        tickers=tickers,
        initial_cash=initial_cash,
        start_date=start_date,
        end_date=end_date,
        monthly_contribution=monthly_contribution,
        max_capital=max_capital,
        min_holding_days=min_holding_days,
        broker_fee_per_transaction=0.0,
    )

    # Strategy return
    strategy_return = (portfolio.final_value - initial_cash) / initial_cash

    # Benchmark return
    benchmark_return = compute_benchmark_return(panel, start_date, end_date)

    beat_benchmark = (
        benchmark_return is not None
        and strategy_return > benchmark_return
    )

    return {
        "portfolio": portfolio,
        "strategy_return": strategy_return,
        "benchmark_return": benchmark_return,
        "beat_benchmark": beat_benchmark,
    }


def compute_benchmark_return(panel: pd.DataFrame, start_date, end_date) -> float | None:
    """
    Computes S&P 500 (^GSPC) return over the backtest window.
    Returns decimal return (e.g., 0.12 for +12%).
    """
    if "^GSPC" not in panel.columns.get_level_values(0):
        return None  # benchmark not available

    spx = panel["^GSPC"].reset_index()

    mask = (spx["Date"] >= start_date) & (spx["Date"] <= end_date)
    spx_period = spx.loc[mask]

    if spx_period.empty:
        return None

    start_price = spx_period.iloc[0]["Close"]
    end_price = spx_period.iloc[-1]["Close"]

    return (end_price - start_price) / start_price


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
    # Compute total invested capital (initial cash + all external contributions)
    total_contrib = sum(a for (_d, a) in getattr(portfolio, "contributions", []))
    invested = float(getattr(portfolio, "initial_cash", 0.0)) + float(total_contrib)
    end = float(portfolio.history[-1].portfolio_value)
    if invested <= 0:
        return 0.0
    return (end - invested) / invested


def cagr(portfolio: Portfolio) -> float:
    if not portfolio.history:
        return 0.0
    start_date = portfolio.history[0].date
    end_date = portfolio.history[-1].date
    days = (end_date - start_date).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    # Use total invested capital (initial cash + contributions) as the starting base
    total_contrib = sum(a for (_d, a) in getattr(portfolio, "contributions", []))
    invested = float(getattr(portfolio, "initial_cash", 0.0)) + float(total_contrib)
    end = float(portfolio.history[-1].portfolio_value)
    if invested <= 0 or end <= 0:
        return 0.0
    return (end / invested) ** (1 / years) - 1


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
    df["pnl"] = df["shares"] * df["price"] * df["side"].map({"BUY": -1, "SELL": 1}) - df["fee"]
    return df.loc[df["pnl"].idxmax()]


def worst_trade(portfolio: Portfolio):
    if not portfolio.trades:
        return None
    df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    df["pnl"] = df["shares"] * df["price"] * df["side"].map({"BUY": -1, "SELL": 1}) - df["fee"]
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


def export_signals(portfolio: Portfolio, path: str | None = None) -> None:
    """Export per-day signal snapshots to CSV for compatibility with old scripts."""
    import os

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "signals.csv"
        )

    if not portfolio.history:
        # write empty file
        pd.DataFrame().to_csv(path, index=False)
        return

    rows = []
    for s in portfolio.history:
        row = {
            "date": s.date,
            "cash": s.cash,
            "portfolio_value": s.portfolio_value,
        }
        for ticker, sig in s.signals.items():
            if isinstance(sig, dict):
                row.update(
                    {
                        f"{ticker}_price": sig.get("price"),
                        f"{ticker}_ma20_today": sig.get("ma20_today"),
                        f"{ticker}_ma20_yesterday": sig.get("ma20_yesterday"),
                        f"{ticker}_buy": sig.get("buy"),
                        f"{ticker}_sell": sig.get("sell"),
                        f"{ticker}_momentum_strength": sig.get(
                            "momentum_strength"
                        ),
                        f"{ticker}_holding_qty": s.holdings.get(ticker, 0.0),
                    }
                )
            else:
                row[f"{ticker}_momentum_strength"] = sig
                row[f"{ticker}_holding_qty"] = s.holdings.get(ticker, 0.0)
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


def export_trades(portfolio: Portfolio, path: str | None = None) -> None:
    """Export trade list to CSV for compatibility with old scripts."""
    import os

    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "Data", "trades.csv"
        )

    if not portfolio.trades:
        pd.DataFrame().to_csv(path, index=False)
        return

    df = pd.DataFrame([t.__dict__ for t in portfolio.trades])
    df.to_csv(path, index=False)
