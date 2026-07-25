"""TradingView data access.

TradingView publishes no official public market-data API. Two practical routes
exist, and this module wraps both:

``TradingViewScanner``
    Posts to the same ``scanner.tradingview.com`` endpoint the website's stock
    screener uses. One request returns hundreds of symbols with SMA50/200, RSI,
    MACD, ATR, ADX, 52-week range, performance and their own technical rating
    already computed server-side. This is the fast path for scanning a market.

``TradingViewHistory``
    Pulls real OHLCV candles through ``tvdatafeed``, an unofficial websocket
    client. Needed for backtesting, which the snapshot endpoint cannot support.
    Install it from GitHub — it is not on PyPI:

        pip install git+https://github.com/rongardF/tvdatafeed.git

Both are unofficial. TradingView's terms of service restrict scraping and
redistribution of their data, so treat this as personal research tooling, keep
request rates low, and use a paid data feed if you ever put money behind it.
"""

from __future__ import annotations

import time
from datetime import date as Date
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .base import PriceBar, PriceSeries

SCANNER_URL = "https://scanner.tradingview.com/{market}/scan"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# Screener columns the feature layer knows how to read. Unsuffixed names are
# daily-timeframe values.
SCANNER_COLUMNS: List[str] = [
    "name",
    "description",
    "exchange",
    "sector",
    "market_cap_basic",
    "close",
    "volume",
    "change",
    "SMA50",
    "SMA200",
    "RSI",
    "MACD.macd",
    "MACD.signal",
    "ATR",
    "ADX",
    "ADX+DI",
    "ADX-DI",
    "Volatility.D",
    "Perf.1M",
    "Perf.3M",
    "Perf.6M",
    "Perf.Y",
    "price_52_week_high",
    "price_52_week_low",
    "average_volume_10d_calc",
    "average_volume_60d_calc",
    "Recommend.All",
]


class TradingViewError(RuntimeError):
    """Raised when TradingView returns something unusable."""


class TradingViewScanner:
    """Snapshot access to TradingView's stock screener.

    Parameters mirror the website's screener: a ``market`` (``america``,
    ``argentina``, ``spain``, ``crypto``, ...) and optional exchange/price
    filters that narrow the universe before ranking.
    """

    name = "tradingview-scanner"

    def __init__(
        self,
        market: str = "america",
        exchanges: Optional[Iterable[str]] = None,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000.0,
        timeout: float = 20.0,
        session: Any = None,
    ) -> None:
        self.market = market
        self.exchanges = list(exchanges) if exchanges else None
        self.min_price = min_price
        self.min_dollar_volume = min_dollar_volume
        self.timeout = timeout
        self._session = session

    def _http(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - requests is a hard dep
            raise TradingViewError(
                "Falta la libreria 'requests'. Instalala con: pip install requests"
            ) from exc
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": DEFAULT_USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Origin": "https://www.tradingview.com",
                "Referer": "https://www.tradingview.com/",
            }
        )
        return self._session

    def _build_filters(self) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "is_primary", "operation": "equal", "right": True},
        ]
        if self.min_price > 0:
            filters.append(
                {"left": "close", "operation": "greater", "right": self.min_price}
            )
        if self.min_dollar_volume > 0:
            filters.append(
                {
                    "left": "average_volume_10d_calc|close",
                    "operation": "greater",
                    "right": self.min_dollar_volume,
                }
            )
        if self.exchanges:
            filters.append(
                {"left": "exchange", "operation": "in_range", "right": self.exchanges}
            )
        return filters

    def scan(
        self,
        limit: int = 200,
        sort_by: str = "market_cap_basic",
        columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top ``limit`` symbols of the market as column dicts."""
        cols = columns or SCANNER_COLUMNS
        payload = {
            "filter": self._build_filters(),
            "options": {"lang": "en"},
            "markets": [self.market],
            "symbols": {"query": {"types": []}, "tickers": []},
            "columns": cols,
            "sort": {"sortBy": sort_by, "sortOrder": "desc"},
            "range": [0, max(1, limit)],
        }
        return self._post(payload, cols)

    def quotes(
        self, symbols: Iterable[str], columns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Fetch specific tickers, e.g. ``["NASDAQ:AAPL", "NYSE:KO"]``.

        Bare symbols are accepted and prefixed with the market's usual exchange
        guess, but an explicit ``EXCHANGE:SYMBOL`` is always more reliable.
        """
        tickers = [self._qualify(s) for s in symbols]
        if not tickers:
            return []
        cols = columns or SCANNER_COLUMNS
        payload = {
            "symbols": {"tickers": tickers, "query": {"types": []}},
            "columns": cols,
        }
        return self._post(payload, cols)

    def _qualify(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        if ":" in symbol:
            return symbol
        default_exchange = (self.exchanges or ["NASDAQ"])[0]
        return f"{default_exchange}:{symbol}"

    def _post(self, payload: Dict[str, Any], columns: List[str]) -> List[Dict[str, Any]]:
        session = self._http()
        url = SCANNER_URL.format(market=self.market)
        last_error: Optional[Exception] = None

        for attempt in range(3):
            try:
                response = session.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                body = response.json()
                break
            except Exception as exc:  # network, JSON or HTTP failure
                last_error = exc
                if attempt == 2:
                    raise TradingViewError(
                        f"No se pudo consultar el screener de TradingView: {exc}"
                    ) from exc
                time.sleep(2 ** attempt)
        else:  # pragma: no cover - loop always breaks or raises
            raise TradingViewError(f"Screener de TradingView no disponible: {last_error}")

        rows = body.get("data") or []
        out: List[Dict[str, Any]] = []
        for entry in rows:
            values = entry.get("d") or []
            record: Dict[str, Any] = dict(zip(columns, values))
            ticker = entry.get("s", "")
            record["ticker"] = ticker
            if ticker and ":" in ticker:
                record.setdefault("exchange", ticker.split(":", 1)[0])
                record["name"] = record.get("name") or ticker.split(":", 1)[1]
            out.append(record)
        return out


class TradingViewHistory:
    """OHLCV candles via ``tvdatafeed``.

    Only needed for the backtest and for local indicator computation; scanning
    does not require it. Import failures are reported with the install command
    rather than a bare ``ModuleNotFoundError``.
    """

    name = "tradingview-history"

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        default_exchange: str = "NASDAQ",
    ) -> None:
        self.default_exchange = default_exchange
        self._username = username
        self._password = password
        self._client = None

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from tvDatafeed import Interval, TvDatafeed
        except ImportError as exc:
            raise TradingViewError(
                "Falta 'tvdatafeed', que no esta en PyPI. Instalalo con:\n"
                "  pip install git+https://github.com/rongardF/tvdatafeed.git"
            ) from exc
        self._interval = Interval
        if self._username and self._password:
            self._client = TvDatafeed(self._username, self._password)
        else:
            self._client = TvDatafeed()
        return self._client

    def fetch(self, symbol: str, lookback_days: int = 400) -> Optional[PriceSeries]:
        """Daily bars for one symbol, as a :class:`PriceSeries`."""
        client = self._connect()
        exchange = self.default_exchange
        ticker = symbol.strip().upper()
        if ":" in ticker:
            exchange, ticker = ticker.split(":", 1)

        frame = client.get_hist(
            symbol=ticker,
            exchange=exchange,
            interval=self._interval.in_daily,
            n_bars=max(lookback_days, 60),
        )
        if frame is None or len(frame) == 0:
            return None

        bars: List[PriceBar] = []
        for stamp, row in frame.iterrows():
            bar_date = stamp.date() if hasattr(stamp, "date") else _to_date(stamp)
            try:
                bar = PriceBar(
                    date=bar_date,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0) or 0.0),
                )
                bar.validate()
            except (ValueError, KeyError, TypeError):
                continue
            bars.append(bar)

        return PriceSeries(symbol=ticker, bars=bars) if bars else None


def _to_date(value: Any) -> Date:
    if isinstance(value, Date):
        return value
    return datetime.fromisoformat(str(value)[:19]).date()
