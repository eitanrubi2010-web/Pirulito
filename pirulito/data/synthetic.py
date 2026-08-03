"""Deterministic fake market data.

Exists so the engine can be exercised — tests, demos, a machine behind a
firewall — without touching a live feed. Prices follow a geometric random walk
whose drift and volatility are derived from the symbol name, so the same ticker
always produces the same history and results stay reproducible.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import date as Date
from datetime import timedelta
from typing import List, Optional

from .base import PriceBar, PriceSeries


class SyntheticProvider:
    """Generates repeatable pseudo-market histories."""

    name = "synthetic"

    def __init__(self, end_date: Optional[Date] = None, seed_salt: str = "") -> None:
        self.end_date = end_date or Date.today()
        self.seed_salt = seed_salt

    def _params(self, symbol: str) -> tuple[float, float, float]:
        """Derive (start_price, annual_drift, annual_vol) from the symbol."""
        digest = hashlib.sha256((symbol + self.seed_salt).encode()).digest()
        start_price = 20.0 + (digest[0] / 255.0) * 380.0
        drift = -0.25 + (digest[1] / 255.0) * 0.70
        vol = 0.15 + (digest[2] / 255.0) * 0.55
        return start_price, drift, vol

    def fetch(self, symbol: str, lookback_days: int = 400) -> Optional[PriceSeries]:
        start_price, drift, vol = self._params(symbol)
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256((symbol + self.seed_salt).encode()).digest()[:8], "big"
            )
        )

        daily_drift = drift / 252.0
        daily_vol = vol / math.sqrt(252.0)

        bars: List[PriceBar] = []
        price = start_price
        # Walk back over calendar days, skipping weekends, so the series ends
        # on end_date with roughly lookback_days trading sessions.
        days: List[Date] = []
        cursor = self.end_date
        while len(days) < lookback_days:
            if cursor.weekday() < 5:
                days.append(cursor)
            cursor -= timedelta(days=1)
        days.reverse()

        for day in days:
            shock = rng.gauss(0.0, 1.0)
            price *= math.exp(daily_drift - 0.5 * daily_vol**2 + daily_vol * shock)
            price = max(price, 0.5)

            intraday = abs(rng.gauss(0.0, daily_vol)) * price
            open_price = price * (1.0 + rng.gauss(0.0, daily_vol * 0.3))
            high = max(open_price, price) + intraday * 0.6
            low = min(open_price, price) - intraday * 0.6
            low = max(low, 0.1)
            volume = max(1000.0, rng.gauss(2_000_000, 500_000))

            bars.append(
                PriceBar(
                    date=day,
                    open=round(open_price, 4),
                    high=round(high, 4),
                    low=round(low, 4),
                    close=round(price, 4),
                    volume=round(volume, 0),
                )
            )

        return PriceSeries(symbol=symbol.upper(), bars=bars)
