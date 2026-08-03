"""Individual buy/sell signals.

Each signal reduces one aspect of a stock to a number in ``[-1, 1]``, where +1
is maximally bullish and -1 maximally bearish. They are deliberately
continuous rather than boolean: a stock 1% above its 200-day average should not
score the same as one 30% above it, and a signal that flips between two values
makes the composite score jitter whenever price hovers near a threshold.

A signal returns ``None`` when its inputs are missing, which happens routinely
because the TradingView screener and a local OHLCV history expose different
fields. The scorer then renormalizes the weights over whatever did compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from . import indicators as ind
from .config import Config
from .features import Features


@dataclass
class Signal:
    """One scored component of the final recommendation."""

    name: str
    score: float
    weight: float
    rationale: str

    def __post_init__(self) -> None:
        self.score = ind.clamp(self.score)

    @property
    def contribution(self) -> float:
        return self.score * self.weight


def trend_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Where price sits relative to its fast and slow moving averages.

    Blends the gap between the two averages — the golden/death-cross idea, but
    measured as a distance so it degrades smoothly instead of firing once — with
    how far price has extended above each of them.
    """
    if f.sma_fast is None or f.sma_slow is None:
        return None
    if f.sma_fast <= 0 or f.sma_slow <= 0:
        return None

    spread = (f.sma_fast - f.sma_slow) / f.sma_slow
    score = (
        0.45 * ind.squash(spread, 0.06)
        + 0.35 * ind.squash((f.price - f.sma_slow) / f.sma_slow, 0.10)
        + 0.20 * ind.squash((f.price - f.sma_fast) / f.sma_fast, 0.05)
    )

    stance = "alcista" if score > 0.1 else "bajista" if score < -0.1 else "lateral"
    rationale = (
        f"Precio {f.price:,.2f} vs SMA{cfg.sma_fast} {f.sma_fast:,.2f} "
        f"({(f.price / f.sma_fast - 1) * 100:+.1f}%) y SMA{cfg.sma_slow} "
        f"{f.sma_slow:,.2f} ({(f.price / f.sma_slow - 1) * 100:+.1f}%); "
        f"tendencia {stance}"
    )
    return Signal("trend", score, cfg.weights.trend, rationale)


def momentum_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Twelve-month return excluding the most recent month.

    Skipping the last month avoids the short-term reversal effect, where the
    very latest move tends to partially undo itself before the longer trend
    reasserts.
    """
    if f.momentum_return is None:
        return None
    score = ind.squash(f.momentum_return, 0.25)
    rationale = f"Momentum 12-1: {f.momentum_return * 100:+.1f}%"
    return Signal("momentum", score, cfg.weights.momentum, rationale)


def macd_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """MACD histogram level, plus its slope when history is available.

    The level says which way momentum leans; the slope says whether that lean
    is building or fading, which is the difference between an early entry and a
    late one. Snapshot data has no previous bar, so it scores on level alone.
    """
    if f.macd_hist is None or f.price <= 0:
        return None

    level = ind.squash(f.macd_hist / f.price, 0.01)
    if f.macd_hist_slope is None:
        score = level
        trend_note = "sin historico para la pendiente"
    else:
        slope = ind.squash(f.macd_hist_slope / f.price, 0.002)
        score = 0.6 * level + 0.4 * slope
        if slope > 0.1:
            trend_note = "acelerando"
        elif slope < -0.1:
            trend_note = "desacelerando"
        else:
            trend_note = "estable"

    rationale = (
        f"Histograma MACD {f.macd_hist:+.3f} "
        f"({f.macd_hist / f.price * 100:+.2f}% del precio), {trend_note}"
    )
    return Signal("macd", score, cfg.weights.macd, rationale)


def rsi_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Reward healthy strength, penalize exhaustion at both extremes.

    Peaks at RSI 60 — strong but not stretched — and falls linearly to -1 at 20
    (capitulation) and at 100 (blow-off top). Direction is already covered by
    the trend and momentum signals, so this one only judges sustainability.
    """
    if f.rsi is None:
        return None

    score = ind.clamp(1.0 - abs(f.rsi - 60.0) / 20.0)
    if f.rsi >= 70:
        state = "sobrecomprado"
    elif f.rsi <= 30:
        state = "sobrevendido"
    else:
        state = "zona neutral"
    rationale = f"RSI({cfg.rsi_period}) = {f.rsi:.1f} ({state})"
    return Signal("rsi", score, cfg.weights.rsi, rationale)


def risk_adjusted_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Sharpe ratio over the volatility window.

    Return on its own rewards whatever moved most; dividing by volatility asks
    how much of that move the holder actually had to sit through.
    """
    if f.sharpe is None:
        return None
    score = ind.squash(f.sharpe, 1.0)
    vol_note = (
        f", vol anualizada {f.annual_volatility * 100:.1f}%"
        if f.annual_volatility is not None
        else ""
    )
    rationale = f"Sharpe {f.sharpe:+.2f}{vol_note}"
    return Signal("risk_adjusted", score, cfg.weights.risk_adjusted, rationale)


def near_high_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Position within the 52-week range, and distance below the high.

    Stocks near their 52-week high have historically kept outperforming, while
    a deep discount from the high more often means something broke than that
    something is cheap.
    """
    if f.high_52w is None or f.low_52w is None:
        return None
    if f.high_52w <= 0 or f.high_52w == f.low_52w:
        return None

    position = (f.price - f.low_52w) / (f.high_52w - f.low_52w)
    drop = (f.high_52w - f.price) / f.high_52w
    score = 0.5 * (2.0 * position - 1.0) + 0.5 * ind.clamp(1.0 - drop / 0.20)

    rationale = (
        f"A {drop * 100:.1f}% del maximo de 52 semanas ({f.high_52w:,.2f}); "
        f"percentil {ind.clamp(position, 0.0, 1.0) * 100:.0f} del rango anual"
    )
    return Signal("near_high", score, cfg.weights.near_high, rationale)


def volume_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """Whether volume is confirming the recent price move.

    A volume surge is only bullish if price rose with it; the same surge into a
    falling price is distribution, so the direction term flips the sign rather
    than merely scaling it.
    """
    if f.rel_volume is None or f.price_change_recent is None:
        return None

    magnitude = ind.squash(f.rel_volume - 1.0, 0.4)
    direction = ind.clamp(f.price_change_recent / 0.05)
    score = magnitude * direction

    rationale = (
        f"Volumen reciente {f.rel_volume:.2f}x su linea base, "
        f"con precio {f.price_change_recent * 100:+.1f}% en el periodo"
    )
    return Signal("volume", score, cfg.weights.volume, rationale)


def trend_quality_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """How cleanly price follows its own trend, rather than how far it went.

    With history this is the R-squared of a regression on log price, which
    separates a steady climb from an equally profitable but violent one. From
    the screener it is ADX, which measures the same idea: strength of trend,
    with direction taken from the +DI/-DI spread.
    """
    if f.trend_slope_annual is not None and f.trend_r2 is not None:
        score = ind.squash(f.trend_slope_annual, 0.20) * f.trend_r2
        rationale = (
            f"Regresion log-precio: pendiente anualizada "
            f"{f.trend_slope_annual * 100:+.1f}%, R2 {f.trend_r2:.2f}"
        )
        return Signal("trend_quality", score, cfg.weights.trend_quality, rationale)

    if f.adx is not None and f.di_plus is not None and f.di_minus is not None:
        strength = ind.clamp((f.adx - 20.0) / 25.0, 0.0, 1.0)
        direction = ind.squash(f.di_plus - f.di_minus, 15.0)
        score = strength * direction
        rationale = (
            f"ADX {f.adx:.1f} (+DI {f.di_plus:.1f} / -DI {f.di_minus:.1f}); "
            f"tendencia {'fuerte' if f.adx >= 25 else 'debil'}"
        )
        return Signal("trend_quality", score, cfg.weights.trend_quality, rationale)

    return None


def tv_rating_signal(f: Features, cfg: Config) -> Optional[Signal]:
    """TradingView's own aggregated technical rating.

    Already normalized to ``[-1, 1]`` across their oscillator and moving-average
    panels, so it enters the blend directly. It overlaps with the other signals
    by construction, which is why it carries a small weight — it is a second
    opinion, not an independent factor.
    """
    if f.tv_recommendation is None:
        return None

    value = ind.clamp(f.tv_recommendation)
    if value >= 0.5:
        label = "compra fuerte"
    elif value >= 0.1:
        label = "compra"
    elif value <= -0.5:
        label = "venta fuerte"
    elif value <= -0.1:
        label = "venta"
    else:
        label = "neutral"
    rationale = f"Rating tecnico de TradingView: {value:+.2f} ({label})"
    return Signal("tv_rating", value, cfg.weights.tv_rating, rationale)


SignalFn = Callable[[Features, Config], Optional[Signal]]

ALL_SIGNALS: List[SignalFn] = [
    trend_signal,
    momentum_signal,
    macd_signal,
    rsi_signal,
    risk_adjusted_signal,
    near_high_signal,
    volume_signal,
    trend_quality_signal,
    tv_rating_signal,
]


def compute_signals(features: Features, cfg: Config) -> List[Signal]:
    """Run every signal, keeping the ones this data source can support."""
    out: List[Signal] = []
    for fn in ALL_SIGNALS:
        signal = fn(features, cfg)
        if signal is not None and signal.weight > 0:
            out.append(signal)
    return out
