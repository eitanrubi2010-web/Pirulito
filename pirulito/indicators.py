"""Technical indicators implemented on plain Python sequences.

Every function returns a list aligned to the input length, using ``None`` for
the warm-up positions where the indicator is not yet defined. Keeping the
alignment explicit means callers can always index an indicator with the same
index they use for prices.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Number = float
OptList = List[Optional[float]]

TRADING_DAYS = 252


def sma(values: Sequence[float], period: int) -> OptList:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: OptList = [None] * len(values)
    if len(values) < period:
        return out
    total = float(sum(values[:period]))
    out[period - 1] = total / period
    for i in range(period, len(values)):
        total += values[i] - values[i - period]
        out[i] = total / period
    return out


def ema(values: Sequence[float], period: int) -> OptList:
    """Exponential moving average, seeded with the first SMA."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: OptList = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    prev = float(sum(values[:period])) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rsi(values: Sequence[float], period: int = 14) -> OptList:
    """Relative Strength Index using Wilder's smoothing."""
    n = len(values)
    out: OptList = [None] * n
    if n < period + 1:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, n):
        change = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[OptList, OptList, OptList]:
    """Return ``(macd_line, signal_line, histogram)``."""
    if fast >= slow:
        raise ValueError("fast period must be shorter than slow period")
    n = len(values)
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    line: OptList = [None] * n
    for i in range(n):
        f, s = ema_fast[i], ema_slow[i]
        if f is not None and s is not None:
            line[i] = f - s

    sig: OptList = [None] * n
    hist: OptList = [None] * n
    start = next((i for i, v in enumerate(line) if v is not None), None)
    if start is None:
        return line, sig, hist

    dense = [v for v in line[start:] if v is not None]
    smoothed = ema(dense, signal)
    for offset, value in enumerate(smoothed):
        sig[start + offset] = value
    for i in range(n):
        if line[i] is not None and sig[i] is not None:
            hist[i] = line[i] - sig[i]
    return line, sig, hist


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> OptList:
    """Wilder's true range."""
    n = len(closes)
    if not (len(highs) == len(lows) == n):
        raise ValueError("highs, lows and closes must have the same length")
    out: OptList = [None] * n
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> OptList:
    """Average True Range using Wilder's smoothing."""
    n = len(closes)
    out: OptList = [None] * n
    if n < period:
        return out
    tr = true_range(highs, lows, closes)
    prev = sum(float(v) for v in tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + float(tr[i])) / period
        out[i] = prev
    return out


def bollinger(
    values: Sequence[float], period: int = 20, num_std: float = 2.0
) -> Tuple[OptList, OptList, OptList]:
    """Return ``(lower_band, middle_band, upper_band)``."""
    mid = sma(values, period)
    lower: OptList = [None] * len(values)
    upper: OptList = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = mid[i]
        if mean is None:
            continue
        variance = sum((v - mean) ** 2 for v in window) / period
        sd = math.sqrt(variance)
        lower[i] = mean - num_std * sd
        upper[i] = mean + num_std * sd
    return lower, mid, upper


def rolling_max(values: Sequence[float], period: int) -> OptList:
    out: OptList = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = max(values[i - period + 1 : i + 1])
    return out


def rolling_min(values: Sequence[float], period: int) -> OptList:
    out: OptList = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = min(values[i - period + 1 : i + 1])
    return out


def pct_returns(values: Sequence[float]) -> List[float]:
    """Simple period-over-period returns; length is ``len(values) - 1``."""
    out: List[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        out.append((values[i] - prev) / prev if prev else 0.0)
    return out


def log_returns(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        out.append(math.log(cur / prev) if prev > 0 and cur > 0 else 0.0)
    return out


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two points."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def annualized_volatility(
    returns: Sequence[float], periods_per_year: int = TRADING_DAYS
) -> float:
    return stdev(returns) * math.sqrt(periods_per_year)


def annualized_return(
    values: Sequence[float], periods_per_year: int = TRADING_DAYS
) -> float:
    """Compound annual growth rate implied by the first and last price."""
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return 0.0
    periods = len(values) - 1
    years = periods / periods_per_year
    if years <= 0:
        return 0.0
    return (values[-1] / values[0]) ** (1.0 / years) - 1.0


def max_drawdown(values: Sequence[float]) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    peak = float("-inf")
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak - v) / peak)
    return worst


def current_drawdown(values: Sequence[float]) -> float:
    """Decline from the running peak to the last observation."""
    if not values:
        return 0.0
    peak = max(values)
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - values[-1]) / peak)


def linear_trend(values: Sequence[float]) -> Tuple[float, float]:
    """Least-squares fit over ``log(values)``.

    Returns ``(slope_per_period, r_squared)``. The slope is a log return per
    period, so multiplying by 252 annualizes it. R-squared measures how
    cleanly the price follows that trend rather than how steep it is.
    """
    points = [math.log(v) for v in values if v > 0]
    n = len(points)
    if n < 3:
        return 0.0, 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(points) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0
    sxy = sum((xs[i] - mean_x) * (points[i] - mean_y) for i in range(n))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in points)
    ss_res = sum((points[i] - (intercept + slope * xs[i])) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, max(0.0, min(1.0, r2))


def beta(asset_returns: Sequence[float], market_returns: Sequence[float]) -> Optional[float]:
    """Covariance of asset vs market divided by market variance."""
    n = min(len(asset_returns), len(market_returns))
    if n < 20:
        return None
    a = list(asset_returns[-n:])
    m = list(market_returns[-n:])
    mean_a = sum(a) / n
    mean_m = sum(m) / n
    var_m = sum((x - mean_m) ** 2 for x in m)
    if var_m == 0:
        return None
    cov = sum((a[i] - mean_a) * (m[i] - mean_m) for i in range(n))
    return cov / var_m


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualized Sharpe ratio from periodic returns."""
    vol = annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    mean_annual = (sum(returns) / len(returns)) * periods_per_year if returns else 0.0
    return (mean_annual - risk_free_rate) / vol


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def squash(value: float, scale: float) -> float:
    """Map an unbounded value into ``[-1, 1]``.

    ``scale`` is the input magnitude that lands at roughly 0.76, so it sets
    what counts as a "strong" reading for that particular measurement.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    return math.tanh(value / scale)
