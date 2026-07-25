"""Orchestration: fetch, score, rank.

Two entry points mirror the two data shapes. :func:`scan_market` sweeps a whole
market through TradingView's screener in one request. :func:`analyze_symbols`
walks a specific watchlist, and can use either the snapshot endpoint or full
OHLCV history depending on how much depth is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .config import Config
from .data.base import DataProvider
from .data.tradingview import TradingViewScanner
from .features import Features, from_scanner_row, from_series
from .scoring import Evaluation, Verdict, evaluate


@dataclass
class ScanReport:
    """The results of one scan, plus what went wrong along the way."""

    evaluations: List[Evaluation]
    errors: Dict[str, str]
    source: str

    def __len__(self) -> int:
        return len(self.evaluations)

    def top(self, count: int = 10) -> List[Evaluation]:
        return self.evaluations[:count]

    def buys(self) -> List[Evaluation]:
        return [
            e
            for e in self.evaluations
            if e.verdict in (Verdict.STRONG_BUY, Verdict.BUY)
        ]

    def filter_verdicts(self, verdicts: Sequence[Verdict]) -> List[Evaluation]:
        wanted = set(verdicts)
        return [e for e in self.evaluations if e.verdict in wanted]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "count": len(self.evaluations),
            "errors": dict(self.errors),
            "results": [e.to_dict() for e in self.evaluations],
        }


def _rank(evaluations: List[Evaluation]) -> List[Evaluation]:
    """Best first. Confidence breaks ties so a well-supported score wins."""
    return sorted(
        evaluations, key=lambda e: (e.score, e.confidence), reverse=True
    )


def scan_market(
    cfg: Config,
    scanner: Optional[TradingViewScanner] = None,
    limit: int = 200,
    sort_by: str = "market_cap_basic",
    min_score: Optional[float] = None,
) -> ScanReport:
    """Score the top ``limit`` symbols of a TradingView market.

    One HTTP request covers the whole universe, so scanning several hundred
    tickers costs about the same as scanning one.
    """
    scanner = scanner or TradingViewScanner()
    rows = scanner.scan(limit=limit, sort_by=sort_by)
    return _score_rows(rows, cfg, scanner.name, min_score)


def analyze_symbols(
    symbols: Iterable[str],
    cfg: Config,
    scanner: Optional[TradingViewScanner] = None,
    min_score: Optional[float] = None,
) -> ScanReport:
    """Score a specific watchlist using TradingView's snapshot data."""
    wanted = [s.strip().upper() for s in symbols if s and s.strip()]
    if not wanted:
        return ScanReport([], {}, "tradingview-scanner")

    scanner = scanner or TradingViewScanner()
    rows = scanner.quotes(wanted)

    report = _score_rows(rows, cfg, scanner.name, min_score)
    returned = {e.symbol.upper() for e in report.evaluations}
    for symbol in wanted:
        bare = symbol.split(":")[-1]
        if bare not in returned and bare not in report.errors:
            report.errors[bare] = "TradingView no devolvio datos para este simbolo"
    return report


def analyze_with_history(
    symbols: Iterable[str],
    provider: DataProvider,
    cfg: Config,
    lookback_days: int = 400,
    min_score: Optional[float] = None,
    on_progress: Optional[Callable[[str, int, int], None]] = None,
) -> ScanReport:
    """Score a watchlist from full OHLCV history.

    Slower than the snapshot path — one request per symbol — but it computes
    every indicator locally, which unlocks the MACD slope and the regression
    trend-quality signal, and it is the same code path the backtest uses.
    """
    wanted = [s.strip().upper() for s in symbols if s and s.strip()]
    evaluations: List[Evaluation] = []
    errors: Dict[str, str] = {}

    for index, symbol in enumerate(wanted, start=1):
        if on_progress:
            on_progress(symbol, index, len(wanted))
        try:
            series = provider.fetch(symbol, lookback_days)
        except Exception as exc:
            errors[symbol] = f"Error al descargar: {exc}"
            continue

        if series is None or not series:
            errors[symbol] = "Sin datos"
            continue
        if len(series) < cfg.min_bars:
            errors[symbol] = (
                f"Historico insuficiente: {len(series)} barras, "
                f"se necesitan {cfg.min_bars}"
            )
            continue

        try:
            features = from_series(series, cfg)
            evaluations.append(evaluate(features, cfg))
        except Exception as exc:
            errors[symbol] = f"Error al evaluar: {exc}"

    if min_score is not None:
        evaluations = [e for e in evaluations if e.score >= min_score]

    return ScanReport(_rank(evaluations), errors, getattr(provider, "name", "history"))


def _score_rows(
    rows: List[Dict[str, Any]],
    cfg: Config,
    source: str,
    min_score: Optional[float],
) -> ScanReport:
    evaluations: List[Evaluation] = []
    errors: Dict[str, str] = {}

    for row in rows:
        symbol = row.get("name") or row.get("ticker") or "?"
        try:
            features: Features = from_scanner_row(row, cfg)
            evaluations.append(evaluate(features, cfg))
        except Exception as exc:
            errors[str(symbol)] = str(exc)

    if min_score is not None:
        evaluations = [e for e in evaluations if e.score >= min_score]

    return ScanReport(_rank(evaluations), errors, source)
