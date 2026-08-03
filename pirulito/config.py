"""Tunable parameters for the scoring engine.

Everything the algorithm treats as a judgement call lives here: how much each
signal counts, what makes a position too risky to recommend, and where the
score thresholds for each verdict sit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class Weights:
    """Relative importance of each signal. Values are normalized on use."""

    trend: float = 0.20
    momentum: float = 0.18
    macd: float = 0.11
    rsi: float = 0.09
    risk_adjusted: float = 0.14
    near_high: float = 0.10
    volume: float = 0.05
    trend_quality: float = 0.06
    tv_rating: float = 0.07

    def __post_init__(self) -> None:
        for f in fields(self):
            if getattr(self, f.name) < 0:
                raise ValueError(f"weight {f.name} must not be negative")
        if self.total() <= 0:
            raise ValueError("at least one weight must be positive")

    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> Dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Config:
    """All knobs for indicator windows, risk gates, verdicts and sizing."""

    weights: Weights = field(default_factory=Weights)

    # --- indicator windows (in trading days) ---
    sma_fast: int = 50
    sma_slow: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    momentum_window: int = 252
    momentum_skip: int = 21
    volatility_window: int = 252
    high_window: int = 252
    volume_window: int = 20
    volume_baseline: int = 60
    quality_window: int = 120

    # A symbol with less history than this cannot be scored at all.
    min_bars: int = 220

    # --- risk gates: a passing score still gets downgraded if these trip ---
    max_annual_volatility: float = 0.75
    max_current_drawdown: float = 0.35
    min_avg_dollar_volume: float = 1_000_000.0

    # --- verdict thresholds on the composite score, which lives in [-1, 1] ---
    strong_buy_above: float = 0.35
    buy_above: float = 0.15
    reduce_below: float = -0.15
    sell_below: float = -0.35

    # --- position sizing ---
    capital: float = 10_000.0
    risk_per_trade: float = 0.01
    atr_stop_multiple: float = 2.5
    reward_risk_target: float = 2.0
    max_position_pct: float = 0.20

    risk_free_rate: float = 0.04

    def __post_init__(self) -> None:
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be shorter than sma_slow")
        if self.momentum_skip >= self.momentum_window:
            raise ValueError("momentum_skip must be shorter than momentum_window")
        if not (self.sell_below < self.reduce_below < self.buy_above < self.strong_buy_above):
            raise ValueError("verdict thresholds must be strictly increasing")
        if not 0 < self.risk_per_trade <= 0.5:
            raise ValueError("risk_per_trade must be within (0, 0.5]")
        if not 0 < self.max_position_pct <= 1.0:
            raise ValueError("max_position_pct must be within (0, 1]")
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if self.atr_stop_multiple <= 0:
            raise ValueError("atr_stop_multiple must be positive")

    @property
    def required_bars(self) -> int:
        """Bars needed before every signal can be computed."""
        return max(
            self.min_bars,
            self.sma_slow,
            self.momentum_window + 1,
            self.volatility_window,
            self.high_window,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["weights"] = self.weights.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        payload = dict(data)
        weights = payload.pop("weights", None)
        known = {f.name for f in fields(cls)} - {"weights"}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        kwargs: Dict[str, Any] = {k: v for k, v in payload.items() if k in known}
        if weights is not None:
            weight_names = {f.name for f in fields(Weights)}
            bad = set(weights) - weight_names
            if bad:
                raise ValueError(f"unknown weight keys: {', '.join(sorted(bad))}")
            kwargs["weights"] = Weights(**weights)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
