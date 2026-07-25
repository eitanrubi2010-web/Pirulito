"""Load price history from local CSV files.

Useful for offline work and for feeding the engine data exported from
TradingView itself ("Export chart data..." in the chart's menu). Column names
are matched case-insensitively and several common spellings are accepted, so
exports from most tools load without editing.
"""

from __future__ import annotations

import csv
from datetime import date as Date
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .base import PriceBar, PriceSeries

_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "time", "datetime", "timestamp", "fecha"],
    "open": ["open", "o", "apertura"],
    "high": ["high", "h", "maximo", "máximo"],
    "low": ["low", "l", "minimo", "mínimo"],
    "close": ["close", "c", "adj close", "adj_close", "cierre", "price"],
    "volume": ["volume", "vol", "v", "volumen"],
}

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
]


class CsvProvider:
    """Reads ``<directory>/<SYMBOL>.csv`` files."""

    name = "csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def fetch(self, symbol: str, lookback_days: int = 400) -> Optional[PriceSeries]:
        path = self._resolve(symbol)
        if path is None:
            return None
        series = load_csv(path, symbol.upper())
        if series is None:
            return None
        return series.tail(lookback_days) if lookback_days > 0 else series

    def _resolve(self, symbol: str) -> Optional[Path]:
        clean = symbol.strip().upper().replace(":", "_")
        for candidate in (clean, clean.lower(), symbol.strip()):
            path = self.directory / f"{candidate}.csv"
            if path.exists():
                return path
        return None


def load_csv(path: str | Path, symbol: Optional[str] = None) -> Optional[PriceSeries]:
    """Parse a single OHLCV CSV file into a :class:`PriceSeries`."""
    path = Path(path)
    ticker = symbol or path.stem.upper()

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        mapping = _map_columns(reader.fieldnames)
        missing = [key for key in ("date", "close") if key not in mapping]
        if missing:
            raise ValueError(
                f"{path}: faltan columnas obligatorias: {', '.join(missing)}"
            )

        bars: List[PriceBar] = []
        for row in reader:
            bar = _parse_row(row, mapping)
            if bar is not None:
                bars.append(bar)

    return PriceSeries(symbol=ticker, bars=bars) if bars else None


def _map_columns(fieldnames: List[str]) -> Dict[str, str]:
    lowered = {name.strip().lower(): name for name in fieldnames if name}
    mapping: Dict[str, str] = {}
    for canonical, options in _ALIASES.items():
        for option in options:
            if option in lowered:
                mapping[canonical] = lowered[option]
                break
    return mapping


def _parse_row(row: Dict[str, str], mapping: Dict[str, str]) -> Optional[PriceBar]:
    raw_date = row.get(mapping["date"], "")
    parsed_date = _parse_date(raw_date)
    if parsed_date is None:
        return None

    close = _to_float(row.get(mapping["close"]))
    if close is None or close <= 0:
        return None

    open_price = _to_float(row.get(mapping.get("open", ""))) or close
    high = _to_float(row.get(mapping.get("high", ""))) or max(open_price, close)
    low = _to_float(row.get(mapping.get("low", ""))) or min(open_price, close)
    volume = _to_float(row.get(mapping.get("volume", ""))) or 0.0

    # Tolerate exports where high/low do not bracket open/close.
    high = max(high, open_price, close)
    low = min(low, open_price, close)
    if low <= 0:
        return None

    return PriceBar(
        date=parsed_date,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=max(0.0, volume),
    )


def _parse_date(value: str) -> Optional[Date]:
    text = (value or "").strip()
    if not text:
        return None
    # Unix timestamps, which TradingView exports use.
    if text.isdigit() and len(text) >= 9:
        try:
            return datetime.utcfromtimestamp(int(text)).date()
        except (ValueError, OverflowError, OSError):
            return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text[: len(fmt) + 4], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "null", "n/a", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None
