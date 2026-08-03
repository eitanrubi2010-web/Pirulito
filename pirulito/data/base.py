"""Price containers and the data-provider interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from typing import Iterable, List, Optional, Protocol, Sequence


@dataclass(frozen=True)
class PriceBar:
    """One period of OHLCV data."""

    date: Date
    open: float
    high: float
    low: float
    close: float
    volume: float

    def validate(self) -> None:
        if self.low > self.high:
            raise ValueError(f"{self.date}: low {self.low} above high {self.high}")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError(f"{self.date}: non-positive price")
        if self.volume < 0:
            raise ValueError(f"{self.date}: negative volume")


@dataclass
class PriceSeries:
    """An ordered, de-duplicated OHLCV history for a single symbol."""

    symbol: str
    bars: List[PriceBar] = field(default_factory=list)

    def __post_init__(self) -> None:
        by_date = {bar.date: bar for bar in self.bars}
        self.bars = [by_date[d] for d in sorted(by_date)]

    def __len__(self) -> int:
        return len(self.bars)

    def __bool__(self) -> bool:
        return bool(self.bars)

    @property
    def dates(self) -> List[Date]:
        return [b.date for b in self.bars]

    @property
    def opens(self) -> List[float]:
        return [b.open for b in self.bars]

    @property
    def highs(self) -> List[float]:
        return [b.high for b in self.bars]

    @property
    def lows(self) -> List[float]:
        return [b.low for b in self.bars]

    @property
    def closes(self) -> List[float]:
        return [b.close for b in self.bars]

    @property
    def volumes(self) -> List[float]:
        return [b.volume for b in self.bars]

    @property
    def last(self) -> PriceBar:
        if not self.bars:
            raise ValueError(f"{self.symbol}: empty series")
        return self.bars[-1]

    def tail(self, count: int) -> "PriceSeries":
        """The most recent ``count`` bars."""
        if count <= 0:
            return PriceSeries(self.symbol, [])
        return PriceSeries(self.symbol, self.bars[-count:])

    def truncate(self, as_of: Date) -> "PriceSeries":
        """Every bar up to and including ``as_of`` — the backtest's time machine."""
        return PriceSeries(self.symbol, [b for b in self.bars if b.date <= as_of])

    def validate(self) -> None:
        for bar in self.bars:
            bar.validate()


class DataProvider(Protocol):
    """Anything that can hand back a :class:`PriceSeries` for a symbol."""

    name: str

    def fetch(self, symbol: str, lookback_days: int) -> Optional[PriceSeries]:
        ...


def bars_from_rows(rows: Iterable[Sequence], symbol: str) -> PriceSeries:
    """Build a series from ``(date, open, high, low, close, volume)`` rows."""
    bars = []
    for row in rows:
        d, o, h, low, c, v = row
        bars.append(
            PriceBar(
                date=d,
                open=float(o),
                high=float(h),
                low=float(low),
                close=float(c),
                volume=float(v),
            )
        )
    return PriceSeries(symbol=symbol, bars=bars)
