"""Risk gates and position sizing.

The score decides what to buy; this module decides how much, and where to admit
the idea was wrong. Sizing is risk-based rather than capital-based: every
position is scaled so that being stopped out costs the same fraction of the
account, whatever the stock's volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import Config
from .features import Features


@dataclass
class PositionPlan:
    """A concrete order sketch derived from ATR and the risk budget."""

    shares: int
    entry_price: float
    stop_price: float
    target_price: float
    notional: float
    pct_of_capital: float
    risk_amount: float
    limited_by_cap: bool

    @property
    def reward_risk(self) -> float:
        risk_per_share = self.entry_price - self.stop_price
        if risk_per_share <= 0:
            return 0.0
        return (self.target_price - self.entry_price) / risk_per_share


def check_risk_gates(features: Features, cfg: Config) -> List[str]:
    """Reasons a symbol is too risky to recommend regardless of its score.

    Only checks that have data to work with can fire, so a scan without a
    given field simply skips that gate rather than failing the symbol.
    """
    warnings: List[str] = []

    if (
        features.annual_volatility is not None
        and features.annual_volatility > cfg.max_annual_volatility
    ):
        warnings.append(
            f"Volatilidad anualizada {features.annual_volatility * 100:.0f}% supera "
            f"el limite de {cfg.max_annual_volatility * 100:.0f}%"
        )

    if (
        features.current_drawdown is not None
        and features.current_drawdown > cfg.max_current_drawdown
    ):
        warnings.append(
            f"Caida actual desde maximos {features.current_drawdown * 100:.0f}% "
            f"supera el limite de {cfg.max_current_drawdown * 100:.0f}%"
        )

    if (
        features.avg_dollar_volume is not None
        and features.avg_dollar_volume < cfg.min_avg_dollar_volume
    ):
        warnings.append(
            f"Liquidez baja: volumen medio {features.avg_dollar_volume:,.0f} por dia, "
            f"minimo {cfg.min_avg_dollar_volume:,.0f}"
        )

    return warnings


def plan_position(features: Features, cfg: Config) -> Optional[PositionPlan]:
    """Size a position so a stop-out costs ``risk_per_trade`` of capital.

    The stop sits ``atr_stop_multiple`` ATRs below entry, so a volatile name
    gets a wider stop and correspondingly fewer shares. A separate cap on
    position size keeps a very quiet stock from swallowing the account just
    because its stop happens to be tight.
    """
    price = features.price
    atr = features.atr
    if price <= 0 or atr is None or atr <= 0:
        return None

    stop_distance = cfg.atr_stop_multiple * atr
    stop_price = price - stop_distance
    if stop_price <= 0:
        return None

    risk_budget = cfg.capital * cfg.risk_per_trade
    shares_by_risk = int(risk_budget // stop_distance)
    shares_by_cap = int((cfg.capital * cfg.max_position_pct) // price)

    limited_by_cap = shares_by_risk > shares_by_cap
    shares = min(shares_by_risk, shares_by_cap)
    if shares <= 0:
        return None

    notional = shares * price
    return PositionPlan(
        shares=shares,
        entry_price=price,
        stop_price=stop_price,
        target_price=price + cfg.reward_risk_target * stop_distance,
        notional=notional,
        pct_of_capital=notional / cfg.capital,
        risk_amount=shares * stop_distance,
        limited_by_cap=limited_by_cap,
    )
