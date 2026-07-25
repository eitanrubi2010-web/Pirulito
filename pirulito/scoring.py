"""Composite scoring: turn signals into a verdict.

The blend is a weighted average over the signals that could be computed, so a
missing input costs coverage rather than silently scoring as neutral. Risk
gates then act as a veto: a stock can earn a high score and still be knocked
down for being too volatile, too far below its highs, or too thin to exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as Date
from enum import Enum
from typing import Any, Dict, List, Optional

from . import indicators as ind
from .config import Config
from .features import Features
from .risk import PositionPlan, check_risk_gates, plan_position
from .signals import Signal, compute_signals


class Verdict(str, Enum):
    """Ordered from most bullish to most bearish."""

    STRONG_BUY = "COMPRA FUERTE"
    BUY = "COMPRA"
    HOLD = "MANTENER"
    REDUCE = "REDUCIR"
    SELL = "VENTA"

    @property
    def rank(self) -> int:
        return _VERDICT_ORDER.index(self)


_VERDICT_ORDER: List[Verdict] = [
    Verdict.STRONG_BUY,
    Verdict.BUY,
    Verdict.HOLD,
    Verdict.REDUCE,
    Verdict.SELL,
]


def downgrade(verdict: Verdict, notches: int = 1) -> Verdict:
    """Move a verdict toward the bearish end, stopping at SELL."""
    index = min(verdict.rank + max(0, notches), len(_VERDICT_ORDER) - 1)
    return _VERDICT_ORDER[index]


@dataclass
class Evaluation:
    """Everything the engine concluded about one symbol."""

    symbol: str
    price: float
    score: float
    verdict: Verdict
    confidence: float
    signals: List[Signal] = field(default_factory=list)
    features: Optional[Features] = None
    plan: Optional[PositionPlan] = None
    warnings: List[str] = field(default_factory=list)
    raw_verdict: Optional[Verdict] = None
    as_of: Optional[Date] = None
    coverage: float = 0.0
    """Share of total configured weight that actually had data behind it."""

    @property
    def score_100(self) -> float:
        """The score rescaled to 0-100, which reads more naturally."""
        return (self.score + 1.0) * 50.0

    @property
    def was_downgraded(self) -> bool:
        return self.raw_verdict is not None and self.raw_verdict != self.verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "price": round(self.price, 4),
            "score": round(self.score, 4),
            "score_100": round(self.score_100, 2),
            "verdict": self.verdict.value,
            "raw_verdict": self.raw_verdict.value if self.raw_verdict else None,
            "confidence": round(self.confidence, 3),
            "coverage": round(self.coverage, 3),
            "signals": [
                {
                    "name": s.name,
                    "score": round(s.score, 4),
                    "weight": round(s.weight, 4),
                    "contribution": round(s.contribution, 4),
                    "rationale": s.rationale,
                }
                for s in self.signals
            ],
            "plan": (
                {
                    "shares": self.plan.shares,
                    "entry": round(self.plan.entry_price, 4),
                    "stop": round(self.plan.stop_price, 4),
                    "target": round(self.plan.target_price, 4),
                    "notional": round(self.plan.notional, 2),
                    "pct_of_capital": round(self.plan.pct_of_capital, 4),
                    "risk_amount": round(self.plan.risk_amount, 2),
                    "reward_risk": round(self.plan.reward_risk, 2),
                }
                if self.plan
                else None
            ),
            "warnings": list(self.warnings),
        }


def classify(score: float, cfg: Config) -> Verdict:
    """Map a composite score onto a verdict band."""
    if score >= cfg.strong_buy_above:
        return Verdict.STRONG_BUY
    if score >= cfg.buy_above:
        return Verdict.BUY
    if score <= cfg.sell_below:
        return Verdict.SELL
    if score <= cfg.reduce_below:
        return Verdict.REDUCE
    return Verdict.HOLD


def _confidence(signals: List[Signal], coverage: float) -> float:
    """How much the signals agree with each other, scaled by data coverage.

    Signals pulling in opposite directions produce a score near zero that looks
    like a calm 'hold' but is really an argument. Reporting the dispersion
    separately keeps that distinction visible.
    """
    if not signals:
        return 0.0
    scores = [s.score for s in signals]
    dispersion = ind.stdev(scores) if len(scores) > 1 else 0.0
    # Dispersion of a set of values in [-1, 1] rarely exceeds ~1.0, so treat
    # that as full disagreement.
    agreement = ind.clamp(1.0 - dispersion, 0.0, 1.0)
    return round(agreement * coverage, 4)


def evaluate(features: Features, cfg: Config) -> Evaluation:
    """Score one symbol and produce its recommendation."""
    signals = compute_signals(features, cfg)

    total_weight = sum(s.weight for s in signals)
    if total_weight <= 0:
        return Evaluation(
            symbol=features.symbol,
            price=features.price,
            score=0.0,
            verdict=Verdict.HOLD,
            confidence=0.0,
            features=features,
            warnings=["Datos insuficientes para calcular ninguna senal"],
            as_of=features.as_of,
            coverage=0.0,
        )

    score = sum(s.contribution for s in signals) / total_weight
    score = ind.clamp(score)
    coverage = total_weight / cfg.weights.total()

    raw_verdict = classify(score, cfg)
    warnings = check_risk_gates(features, cfg)
    verdict = downgrade(raw_verdict, len(warnings)) if warnings else raw_verdict

    plan = None
    if verdict in (Verdict.STRONG_BUY, Verdict.BUY):
        plan = plan_position(features, cfg)

    return Evaluation(
        symbol=features.symbol,
        price=features.price,
        score=score,
        verdict=verdict,
        confidence=_confidence(signals, coverage),
        signals=signals,
        features=features,
        plan=plan,
        warnings=warnings,
        raw_verdict=raw_verdict,
        as_of=features.as_of,
        coverage=coverage,
    )
