"""Pirulito — seguimiento y puntuacion de acciones con datos de TradingView."""

from .config import Config, Weights
from .features import Features, from_scanner_row, from_series
from .scoring import Evaluation, Verdict, classify, evaluate
from .screener import ScanReport, analyze_symbols, analyze_with_history, scan_market
from .signals import Signal, compute_signals

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Weights",
    "Features",
    "from_series",
    "from_scanner_row",
    "Signal",
    "compute_signals",
    "Evaluation",
    "Verdict",
    "evaluate",
    "classify",
    "ScanReport",
    "scan_market",
    "analyze_symbols",
    "analyze_with_history",
]
