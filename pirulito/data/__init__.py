"""Data sources for the engine.

Two shapes of provider exist, and they feed different parts of the pipeline:

* **Snapshot** (:class:`~pirulito.data.tradingview.TradingViewScanner`) returns
  precomputed indicator values for many symbols at once. Fast to scan, but has
  no history, so it cannot be backtested.
* **History** (``TradingViewHistory``, ``CsvProvider``, ``SyntheticProvider``)
  returns OHLCV bars, from which the engine computes indicators itself.
"""

from __future__ import annotations

from typing import Any

from .base import DataProvider, PriceBar, PriceSeries, bars_from_rows
from .csv_file import CsvProvider, load_csv
from .synthetic import SyntheticProvider
from .tradingview import (
    SCANNER_COLUMNS,
    TradingViewError,
    TradingViewHistory,
    TradingViewScanner,
)

__all__ = [
    "DataProvider",
    "PriceBar",
    "PriceSeries",
    "bars_from_rows",
    "CsvProvider",
    "load_csv",
    "SyntheticProvider",
    "TradingViewScanner",
    "TradingViewHistory",
    "TradingViewError",
    "SCANNER_COLUMNS",
    "get_history_provider",
]

HISTORY_PROVIDERS = {
    "tradingview": TradingViewHistory,
    "csv": CsvProvider,
    "synthetic": SyntheticProvider,
}


def get_history_provider(name: str, **kwargs: Any) -> DataProvider:
    """Build a history provider by name.

    ``csv`` requires ``directory``; the others take optional credentials or a
    seed. Raises ``ValueError`` listing the valid names on a typo.
    """
    key = name.strip().lower()
    if key not in HISTORY_PROVIDERS:
        valid = ", ".join(sorted(HISTORY_PROVIDERS))
        raise ValueError(f"Proveedor desconocido '{name}'. Opciones: {valid}")
    return HISTORY_PROVIDERS[key](**kwargs)
