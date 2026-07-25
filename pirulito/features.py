"""The measurement layer that sits between raw data and the scoring engine.

Every input the algorithm needs is collected into a single :class:`Features`
object. Two things can build one:

* :func:`from_series` computes the indicators locally from OHLCV history.
* :func:`from_scanner_row` maps the values TradingView's screener already
  computed server-side.

Anything a given source cannot supply stays ``None``, and the scorer drops the
affected signal and renormalizes the remaining weights. That is what lets one
scoring engine serve both a full-history backtest and a one-request scan of
several thousand tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from typing import Any, Dict, Optional

from . import indicators as ind
from .config import Config
from .data.base import PriceSeries


@dataclass
class Features:
    """Normalized indicator readings for one symbol at one point in time."""

    symbol: str
    price: float
    source: str

    as_of: Optional[Date] = None
    description: Optional[str] = None
    exchange: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None

    # trend
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None

    # oscillators
    rsi: Optional[float] = None
    macd_hist: Optional[float] = None
    macd_hist_slope: Optional[float] = None

    # momentum
    momentum_return: Optional[float] = None
    annual_return: Optional[float] = None

    # risk
    annual_volatility: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    current_drawdown: Optional[float] = None
    atr: Optional[float] = None

    # 52-week range
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    # volume
    rel_volume: Optional[float] = None
    price_change_recent: Optional[float] = None
    avg_dollar_volume: Optional[float] = None

    # trend quality — regression from history, ADX from the scanner
    trend_slope_annual: Optional[float] = None
    trend_r2: Optional[float] = None
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None

    # TradingView's own aggregated technical rating, already in [-1, 1]
    tv_recommendation: Optional[float] = None

    bars_available: Optional[int] = None

    @property
    def atr_pct(self) -> Optional[float]:
        if self.atr is None or self.price <= 0:
            return None
        return self.atr / self.price

    @property
    def label(self) -> str:
        return self.description or self.symbol


def from_series(series: PriceSeries, cfg: Config) -> Features:
    """Compute every indicator locally from an OHLCV history."""
    closes = series.closes
    if not closes:
        raise ValueError(f"{series.symbol}: empty price series")
    price = closes[-1]

    feats = Features(
        symbol=series.symbol,
        price=price,
        source="history",
        as_of=series.last.date,
        bars_available=len(closes),
    )

    if len(closes) >= cfg.sma_fast:
        feats.sma_fast = _last(ind.sma(closes, cfg.sma_fast))
    if len(closes) >= cfg.sma_slow:
        feats.sma_slow = _last(ind.sma(closes, cfg.sma_slow))

    if len(closes) >= cfg.rsi_period + 2:
        feats.rsi = _last(ind.rsi(closes, cfg.rsi_period))

    if len(closes) >= cfg.macd_slow + cfg.macd_signal:
        _, _, hist = ind.macd(closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        dense = [v for v in hist if v is not None]
        if dense:
            feats.macd_hist = dense[-1]
            if len(dense) >= 6:
                feats.macd_hist_slope = (dense[-1] - dense[-6]) / 5.0

    # 12-1 momentum: one-year return that stops a month short, since the most
    # recent month tends to mean-revert rather than persist.
    if len(closes) >= cfg.momentum_window + 1:
        end = len(closes) - 1 - cfg.momentum_skip
        start = len(closes) - 1 - cfg.momentum_window
        if start >= 0 and end > start and closes[start] > 0:
            feats.momentum_return = closes[end] / closes[start] - 1.0

    window = closes[-cfg.volatility_window :]
    rets = ind.pct_returns(window)
    if len(rets) >= 20:
        feats.annual_return = ind.annualized_return(window)
        feats.annual_volatility = ind.annualized_volatility(rets)
        feats.sharpe = ind.sharpe_ratio(rets, cfg.risk_free_rate)
        feats.max_drawdown = ind.max_drawdown(window)
        feats.current_drawdown = ind.current_drawdown(window)

    high_window = closes[-cfg.high_window :]
    if len(high_window) >= 60:
        feats.high_52w = max(high_window)
        feats.low_52w = min(high_window)

    atr_series = ind.atr(series.highs, series.lows, closes, cfg.atr_period)
    feats.atr = _last(atr_series)

    volumes = series.volumes
    if len(volumes) >= cfg.volume_baseline and len(closes) > cfg.volume_window:
        recent = sum(volumes[-cfg.volume_window :]) / cfg.volume_window
        baseline = sum(volumes[-cfg.volume_baseline :]) / cfg.volume_baseline
        if baseline > 0:
            feats.rel_volume = recent / baseline
        past = closes[-cfg.volume_window - 1]
        if past > 0:
            feats.price_change_recent = closes[-1] / past - 1.0

    vol_window = min(len(series.bars), cfg.volume_window)
    if vol_window > 0:
        recent_bars = series.bars[-vol_window:]
        feats.avg_dollar_volume = (
            sum(b.close * b.volume for b in recent_bars) / vol_window
        )

    quality_window = closes[-cfg.quality_window :]
    if len(quality_window) >= 40:
        slope, r2 = ind.linear_trend(quality_window)
        feats.trend_slope_annual = slope * ind.TRADING_DAYS
        feats.trend_r2 = r2

    return feats


def from_scanner_row(row: Dict[str, Any], cfg: Config) -> Features:
    """Map one TradingView screener row onto :class:`Features`.

    The screener reports percentages as whole numbers (``2.5`` meaning 2.5%),
    so everything is divided by 100 to match the fractions used everywhere
    else. Daily volatility is annualized with the usual square-root-of-time
    scaling.
    """
    symbol = row.get("name") or row.get("symbol") or "?"
    price = _num(row.get("close"))
    if price is None or price <= 0:
        raise ValueError(f"{symbol}: scanner row has no usable close price")

    feats = Features(
        symbol=symbol,
        price=price,
        source="scanner",
        description=row.get("description"),
        exchange=row.get("exchange"),
        sector=row.get("sector"),
        market_cap=_num(row.get("market_cap_basic")),
        sma_fast=_num(row.get("SMA50")),
        sma_slow=_num(row.get("SMA200")),
        rsi=_num(row.get("RSI")),
        atr=_num(row.get("ATR")),
        high_52w=_num(row.get("price_52_week_high")),
        low_52w=_num(row.get("price_52_week_low")),
        adx=_num(row.get("ADX")),
        di_plus=_num(row.get("ADX+DI")),
        di_minus=_num(row.get("ADX-DI")),
        tv_recommendation=_num(row.get("Recommend.All")),
    )

    macd_line = _num(row.get("MACD.macd"))
    macd_sig = _num(row.get("MACD.signal"))
    if macd_line is not None and macd_sig is not None:
        feats.macd_hist = macd_line - macd_sig
    # The screener is a snapshot, so there is no previous bar to derive a
    # histogram slope from; the MACD signal falls back to level only.

    perf_year = _pct(row.get("Perf.Y"))
    perf_month = _pct(row.get("Perf.1M"))
    if perf_year is not None:
        feats.annual_return = perf_year
        if perf_month is not None and perf_month > -1.0:
            feats.momentum_return = (1.0 + perf_year) / (1.0 + perf_month) - 1.0
        else:
            feats.momentum_return = perf_year
    if perf_month is not None:
        feats.price_change_recent = perf_month

    daily_vol = _pct(row.get("Volatility.D"))
    if daily_vol is not None and daily_vol > 0:
        feats.annual_volatility = daily_vol * (ind.TRADING_DAYS ** 0.5)
        if feats.annual_return is not None:
            feats.sharpe = (
                feats.annual_return - cfg.risk_free_rate
            ) / feats.annual_volatility

    if feats.high_52w and feats.high_52w > 0:
        feats.current_drawdown = max(0.0, (feats.high_52w - price) / feats.high_52w)

    vol_10d = _num(row.get("average_volume_10d_calc"))
    vol_60d = _num(row.get("average_volume_60d_calc"))
    if vol_10d is not None:
        feats.avg_dollar_volume = price * vol_10d
        if vol_60d and vol_60d > 0:
            feats.rel_volume = vol_10d / vol_60d

    return feats


def _last(values) -> Optional[float]:
    for v in reversed(values):
        if v is not None:
            return v
    return None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # drop NaN


def _pct(value: Any) -> Optional[float]:
    """Convert a percentage reported as a whole number into a fraction."""
    number = _num(value)
    return None if number is None else number / 100.0
