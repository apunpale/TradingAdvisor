from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple


@dataclass
class Position:
    shares: float
    entry_date: datetime


@dataclass
class Trade:
    date: datetime
    ticker: str
    side: str  # "BUY" or "SELL"
    price: float
    value: float
    shares: float
    fee: float = 0.0


@dataclass
class PortfolioSnapshot:
    date: datetime
    cash: float
    holdings: Dict[str, float]
    portfolio_value: float
    signals: Dict[str, Any]


@dataclass
class Portfolio:
    initial_cash: float = 5_000.0
    cash: float = field(init=False)
    holdings: Dict[str, Position] = field(default_factory=dict)
    trades: List[Trade] = field(default_factory=list)
    history: List[PortfolioSnapshot] = field(default_factory=list)
    contributions: List[Tuple[datetime, float]] = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_cash

    def value(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for ticker, pos in self.holdings.items():
            price = prices.get(ticker, 0.0)
            total += pos.shares * price
        return total

    def total_exposure(self, prices: Dict[str, float]) -> float:
        exposure = 0.0
        for ticker, pos in self.holdings.items():
            price = prices.get(ticker, 0.0)
            exposure += pos.shares * price
        return exposure

    def can_sell(self, ticker: str, date: datetime, min_holding_days: int = 30) -> bool:
        pos = self.holdings.get(ticker)
        if not pos:
            return False
        holding_days = (date - pos.entry_date).days
        return holding_days >= min_holding_days

    def buy(self, ticker: str, price: float, amount: float, date: datetime, fee: float = 0.0) -> None:
        total_cost = amount + fee
        if price <= 0 or amount <= 0 or total_cost > self.cash:
            return

        shares = amount / price
        self.cash -= total_cost

        if ticker in self.holdings:
            # keep earliest entry date
            pos = self.holdings[ticker]
            pos.shares += shares
        else:
            self.holdings[ticker] = Position(shares=shares, entry_date=date)

        self.trades.append(
            Trade(
                date=date,
                ticker=ticker,
                side="BUY",
                price=price,
                value=amount,
                shares=shares,
                fee=fee,
            )
        )

    def sell(self, ticker: str, price: float, date: datetime, fee: float = 0.0) -> None:
        if price <= 0 or ticker not in self.holdings:
            return
        pos = self.holdings[ticker]
        shares_to_sell = pos.shares
        if shares_to_sell <= 0:
            return

        value = shares_to_sell * price
        self.cash += value - fee
        del self.holdings[ticker]

        self.trades.append(
            Trade(
                date=date,
                ticker=ticker,
                side="SELL",
                price=price,
                value=value,
                shares=shares_to_sell,
                fee=fee,
            )
        )

    def log_day(self, date: datetime, prices: Dict[str, float], signals: Dict[str, Any]) -> None:
        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            holdings={t: p.shares for t, p in self.holdings.items()},
            portfolio_value=self.value(prices),
            signals={t: v.copy() if isinstance(v, dict) else v for t, v in signals.items()},
        )
        self.history.append(snapshot)

    def contribute(self, amount: float, date: datetime) -> None:
        """Add external contribution to the portfolio cash and record it."""
        if amount is None or amount <= 0:
            return
        self.cash += amount
        self.contributions.append((date, amount))
