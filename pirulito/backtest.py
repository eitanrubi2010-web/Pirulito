"""Walk-forward backtest.

The point of this module is to keep the engine honest. At every rebalance date
the series are truncated to that date before scoring, so the decision only ever
sees data that existed at the time — no lookahead. The portfolio then holds the
top-ranked names until the next rebalance, and the result is compared against
equal-weight buy-and-hold over the same universe.

A backtest that beats its benchmark on one universe over one period is weak
evidence. Treat it as a sanity check that the scoring is not actively harmful,
not as a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import indicators as ind
from .config import Config
from .data.base import PriceSeries
from .features import from_series
from .scoring import Verdict, evaluate


@dataclass
class Rebalance:
    """One decision point in the simulation."""

    date: Date
    picks: List[str]
    scores: Dict[str, float]
    equity: float


@dataclass
class BacktestResult:
    """Equity curve and summary statistics for a strategy and its benchmark."""

    dates: List[Date]
    equity: List[float]
    benchmark: List[float]
    rebalances: List[Rebalance] = field(default_factory=list)
    initial_capital: float = 10_000.0

    @property
    def total_return(self) -> float:
        return self._total(self.equity)

    @property
    def benchmark_total_return(self) -> float:
        return self._total(self.benchmark)

    def _total(self, curve: List[float]) -> float:
        if len(curve) < 2 or curve[0] <= 0:
            return 0.0
        return curve[-1] / curve[0] - 1.0

    @property
    def years(self) -> float:
        if len(self.dates) < 2:
            return 0.0
        return (self.dates[-1] - self.dates[0]).days / 365.25

    def cagr(self, curve: Optional[List[float]] = None) -> float:
        curve = curve if curve is not None else self.equity
        if len(curve) < 2 or curve[0] <= 0 or self.years <= 0:
            return 0.0
        return (curve[-1] / curve[0]) ** (1.0 / self.years) - 1.0

    def stats(self) -> Dict[str, Any]:
        rets = ind.pct_returns(self.equity)
        bench_rets = ind.pct_returns(self.benchmark)
        wins = sum(1 for r in rets if r > 0)
        return {
            "años": round(self.years, 2),
            "capital_inicial": round(self.initial_capital, 2),
            "capital_final": round(self.equity[-1], 2) if self.equity else 0.0,
            "retorno_total": round(self.total_return, 4),
            "cagr": round(self.cagr(), 4),
            "volatilidad": round(ind.annualized_volatility(rets), 4),
            "sharpe": round(ind.sharpe_ratio(rets), 3),
            "max_drawdown": round(ind.max_drawdown(self.equity), 4),
            "dias_positivos": round(wins / len(rets), 4) if rets else 0.0,
            "rebalanceos": len(self.rebalances),
            "benchmark_retorno_total": round(self.benchmark_total_return, 4),
            "benchmark_cagr": round(self.cagr(self.benchmark), 4),
            "benchmark_max_drawdown": round(ind.max_drawdown(self.benchmark), 4),
            "benchmark_sharpe": round(ind.sharpe_ratio(bench_rets), 3),
            "exceso_cagr": round(self.cagr() - self.cagr(self.benchmark), 4),
        }


def _price_lookup(series: PriceSeries) -> Dict[Date, float]:
    return {bar.date: bar.close for bar in series.bars}


def _price_on_or_before(
    lookup: Dict[Date, float], sorted_dates: List[Date], target: Date
) -> Optional[float]:
    """Last known close at or before ``target``; ``None`` before the series starts."""
    if target in lookup:
        return lookup[target]
    lo, hi = 0, len(sorted_dates) - 1
    best: Optional[Date] = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_dates[mid] <= target:
            best = sorted_dates[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return lookup[best] if best is not None else None


def run_backtest(
    histories: Mapping[str, PriceSeries],
    cfg: Config,
    top_n: int = 5,
    rebalance_days: int = 21,
    commission_bps: float = 5.0,
    initial_capital: Optional[float] = None,
) -> BacktestResult:
    """Simulate the strategy over the supplied histories.

    ``rebalance_days`` counts trading days between decisions, and
    ``commission_bps`` charges a round-trip cost on the traded notional so the
    result is not flattered by free turnover.
    """
    usable = {s: series for s, series in histories.items() if len(series) > 0}
    if not usable:
        raise ValueError("No hay historicos para simular")
    if top_n < 1:
        raise ValueError("top_n debe ser al menos 1")

    capital = initial_capital if initial_capital is not None else cfg.capital

    all_dates = sorted({bar.date for series in usable.values() for bar in series.bars})
    warmup = cfg.required_bars
    if len(all_dates) <= warmup + rebalance_days:
        raise ValueError(
            f"Historico insuficiente para backtest: {len(all_dates)} dias, "
            f"se necesitan mas de {warmup + rebalance_days}"
        )

    lookups = {s: _price_lookup(series) for s, series in usable.items()}
    sorted_dates = {s: sorted(lookup) for s, lookup in lookups.items()}

    timeline = all_dates[warmup:]
    rebalance_indices = set(range(0, len(timeline), rebalance_days))

    cash = capital
    shares: Dict[str, float] = {}
    equity_curve: List[float] = []
    rebalances: List[Rebalance] = []

    # Benchmark: equal-weight, buy once on day one, hold to the end. Only
    # symbols that were actually trading on day one can be bought, and the
    # capital is split among those — dividing by the full universe would leave
    # the benchmark partly uninvested and flatter the strategy by comparison.
    first_day = timeline[0]
    opening_prices = {
        symbol: price
        for symbol in usable
        if (price := _price_on_or_before(lookups[symbol], sorted_dates[symbol], first_day))
        and price > 0
    }
    if not opening_prices:
        raise ValueError("Ningun simbolo tiene precio en la fecha de inicio")
    per_symbol = capital / len(opening_prices)
    bench_shares: Dict[str, float] = {
        symbol: per_symbol / price for symbol, price in opening_prices.items()
    }
    bench_curve: List[float] = []

    for step, day in enumerate(timeline):
        prices: Dict[str, float] = {}
        for symbol in usable:
            price = _price_on_or_before(lookups[symbol], sorted_dates[symbol], day)
            if price is not None:
                prices[symbol] = price

        if step in rebalance_indices:
            equity = cash + sum(
                count * prices.get(symbol, 0.0) for symbol, count in shares.items()
            )
            picks, scores = _select(usable, cfg, day, top_n)

            traded_notional = _turnover(shares, picks, equity, prices)
            cash = equity - traded_notional * (commission_bps / 10_000.0)

            shares = {}
            if picks:
                allocation = cash / len(picks)
                for symbol in picks:
                    price = prices.get(symbol)
                    if price and price > 0:
                        shares[symbol] = allocation / price
                        cash -= allocation
            rebalances.append(Rebalance(day, picks, scores, equity))

        equity = cash + sum(
            count * prices.get(symbol, 0.0) for symbol, count in shares.items()
        )
        equity_curve.append(equity)

        bench_value = sum(
            count * prices.get(symbol, 0.0) for symbol, count in bench_shares.items()
        )
        bench_curve.append(bench_value if bench_value > 0 else capital)

    return BacktestResult(
        dates=list(timeline),
        equity=equity_curve,
        benchmark=bench_curve,
        rebalances=rebalances,
        initial_capital=capital,
    )


def _select(
    histories: Mapping[str, PriceSeries], cfg: Config, day: Date, top_n: int
) -> Tuple[List[str], Dict[str, float]]:
    """Score every symbol using only data available on ``day``."""
    ranked: List[Tuple[float, str]] = []
    scores: Dict[str, float] = {}

    for symbol, series in histories.items():
        visible = series.truncate(day)
        if len(visible) < cfg.min_bars:
            continue
        try:
            evaluation = evaluate(from_series(visible, cfg), cfg)
        except (ValueError, ZeroDivisionError):
            continue
        scores[symbol] = evaluation.score
        if evaluation.verdict in (Verdict.STRONG_BUY, Verdict.BUY):
            ranked.append((evaluation.score, symbol))

    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked[:top_n]], scores


def _turnover(
    current: Dict[str, float],
    picks: List[str],
    equity: float,
    prices: Dict[str, float],
) -> float:
    """Notional that changes hands moving from ``current`` to ``picks``."""
    if equity <= 0:
        return 0.0
    target_value = equity / len(picks) if picks else 0.0
    symbols = set(current) | set(picks)
    traded = 0.0
    for symbol in symbols:
        held = current.get(symbol, 0.0) * prices.get(symbol, 0.0)
        wanted = target_value if symbol in picks else 0.0
        traded += abs(wanted - held)
    return traded
