from datetime import datetime

from Src.portfolio import Portfolio, PortfolioSnapshot
from Src.backtest import total_return, cagr


def test_total_return_and_cagr_with_contributions():
    # Setup portfolio with initial cash 1000 and one contribution 500
    p = Portfolio(initial_cash=1000)
    contrib_date = datetime(2020, 2, 1)
    p.contributions.append((contrib_date, 500.0))

    # Create history: start date and end date one year later
    start = datetime(2020, 1, 1)
    end = datetime(2021, 1, 1)

    start_snapshot = PortfolioSnapshot(
        date=start,
        cash=1000.0,
        holdings={},
        portfolio_value=1000.0,
        signals={},
    )

    # final portfolio value after contributions and returns
    final_value = 2000.0
    end_snapshot = PortfolioSnapshot(
        date=end,
        cash=0.0,
        holdings={},
        portfolio_value=final_value,
        signals={},
    )

    p.history = [start_snapshot, end_snapshot]

    tr = total_return(p)
    cg = cagr(p)

    # invested = 1000 + 500 = 1500
    invested = 1500.0
    expected_tr = (final_value - invested) / invested
    assert abs(tr - expected_tr) < 1e-8

    # expected CAGR using same day-count (accounts for leap year)
    days = (end - start).days
    years = days / 365.25
    expected_cg = (final_value / invested) ** (1 / years) - 1
    assert abs(cg - expected_cg) < 1e-8
