"""Terminal and file output.

Plain ASCII tables with no rendering dependencies, so output stays readable
when piped to a file or a log. Colour is applied only when stdout is a TTY.
"""

from __future__ import annotations

import csv
import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .backtest import BacktestResult
from .scoring import Evaluation, Verdict
from .screener import ScanReport

_COLORS = {
    Verdict.STRONG_BUY: "\033[1;32m",
    Verdict.BUY: "\033[32m",
    Verdict.HOLD: "\033[33m",
    Verdict.REDUCE: "\033[35m",
    Verdict.SELL: "\033[31m",
}
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"


def _tty() -> bool:
    return sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _tty() else text


def _verdict_text(verdict: Verdict) -> str:
    return _paint(verdict.value, _COLORS.get(verdict, ""))


def _visible_len(text: str) -> int:
    """Length ignoring ANSI escape sequences, so columns stay aligned."""
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            while i < len(text) and text[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
) -> str:
    """Render an ASCII table, padding around any ANSI colour codes."""
    if not rows:
        return "(sin resultados)"
    aligns = aligns or ["<"] * len(headers)
    widths = [_visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _visible_len(str(cell)))

    def line(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            text = str(cell)
            pad = widths[i] - _visible_len(text)
            parts.append(" " * pad + text if aligns[i] == ">" else text + " " * pad)
        return "  ".join(parts).rstrip()

    separator = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), separator] + [line(r) for r in rows])


def format_scan(report: ScanReport, limit: int = 20, show_errors: bool = True) -> str:
    """Ranked table of scan results."""
    rows: List[List[str]] = []
    for rank, ev in enumerate(report.evaluations[:limit], start=1):
        flag = " *" if ev.was_downgraded else ""
        rows.append(
            [
                str(rank),
                ev.symbol,
                f"{ev.price:,.2f}",
                f"{ev.score_100:.1f}",
                _verdict_text(ev.verdict) + flag,
                f"{ev.confidence * 100:.0f}%",
                f"{ev.coverage * 100:.0f}%",
            ]
        )

    out = [
        _paint(f"Resultados ({len(report.evaluations)} simbolos, fuente: {report.source})", _BOLD),
        "",
        render_table(
            ["#", "Simbolo", "Precio", "Puntaje", "Veredicto", "Confianza", "Cobertura"],
            rows,
            [">", "<", ">", ">", "<", ">", ">"],
        ),
    ]

    if any(e.was_downgraded for e in report.evaluations[:limit]):
        out.append("")
        out.append(_paint("  * veredicto rebajado por controles de riesgo", _DIM))

    if show_errors and report.errors:
        out.append("")
        out.append(_paint(f"Omitidos ({len(report.errors)}):", _DIM))
        for symbol, reason in list(report.errors.items())[:10]:
            out.append(_paint(f"  {symbol}: {reason}", _DIM))
        if len(report.errors) > 10:
            out.append(_paint(f"  ... y {len(report.errors) - 10} mas", _DIM))

    return "\n".join(out)


def format_detail(ev: Evaluation) -> str:
    """Full breakdown for a single symbol, signal by signal."""
    header = f"{ev.symbol}  —  {ev.price:,.2f}"
    if ev.features and ev.features.description:
        header = f"{ev.symbol}  —  {ev.features.description}  —  {ev.price:,.2f}"

    out = [
        _paint(header, _BOLD),
        f"Fecha: {ev.as_of or 'n/d'}    Fuente: {ev.features.source if ev.features else 'n/d'}",
        "",
        f"Puntaje: {ev.score_100:.1f}/100  ({ev.score:+.3f})",
        f"Veredicto: {_verdict_text(ev.verdict)}"
        + (
            f"   (antes de riesgo: {ev.raw_verdict.value})"
            if ev.was_downgraded and ev.raw_verdict
            else ""
        ),
        f"Confianza: {ev.confidence * 100:.0f}%    Cobertura de senales: {ev.coverage * 100:.0f}%",
        "",
        _paint("Desglose de senales", _BOLD),
    ]

    rows = []
    for signal in sorted(ev.signals, key=lambda s: abs(s.contribution), reverse=True):
        rows.append(
            [
                signal.name,
                f"{signal.score:+.2f}",
                _bar(signal.score),
                f"{signal.weight:.2f}",
                f"{signal.contribution:+.3f}",
                signal.rationale,
            ]
        )
    out.append(
        render_table(
            ["Senal", "Valor", "", "Peso", "Aporte", "Detalle"],
            rows,
            ["<", ">", "<", ">", ">", "<"],
        )
    )

    if ev.warnings:
        out.append("")
        out.append(_paint("Advertencias de riesgo", _BOLD))
        for warning in ev.warnings:
            out.append(f"  ! {warning}")

    if ev.plan:
        plan = ev.plan
        out.append("")
        out.append(_paint("Plan de posicion", _BOLD))
        out.append(
            f"  {plan.shares} acciones a {plan.entry_price:,.2f} "
            f"= {plan.notional:,.2f} ({plan.pct_of_capital * 100:.1f}% del capital)"
        )
        out.append(
            f"  Stop {plan.stop_price:,.2f}   Objetivo {plan.target_price:,.2f}   "
            f"Ratio beneficio/riesgo {plan.reward_risk:.1f}:1"
        )
        out.append(f"  Riesgo maximo de la operacion: {plan.risk_amount:,.2f}")
        if plan.limited_by_cap:
            out.append(
                _paint("  (tamano limitado por el tope de posicion, no por el riesgo)", _DIM)
            )

    return "\n".join(out)


def _bar(score: float, width: int = 11) -> str:
    """A small centred bar: left of centre is bearish, right is bullish."""
    half = width // 2
    filled = int(round(abs(score) * half))
    if score >= 0:
        return " " * half + "|" + "#" * filled + " " * (half - filled)
    return " " * (half - filled) + "#" * filled + "|" + " " * half


def format_backtest(result: BacktestResult) -> str:
    """Summary statistics comparing the strategy to buy-and-hold."""
    stats = result.stats()
    out = [
        _paint("Resultado del backtest", _BOLD),
        f"Periodo: {result.dates[0]} a {result.dates[-1]}  ({stats['años']} años)",
        "",
    ]

    rows = [
        [
            "Retorno total",
            f"{stats['retorno_total'] * 100:+.1f}%",
            f"{stats['benchmark_retorno_total'] * 100:+.1f}%",
        ],
        ["CAGR", f"{stats['cagr'] * 100:+.1f}%", f"{stats['benchmark_cagr'] * 100:+.1f}%"],
        ["Sharpe", f"{stats['sharpe']:.2f}", f"{stats['benchmark_sharpe']:.2f}"],
        [
            "Max drawdown",
            f"-{stats['max_drawdown'] * 100:.1f}%",
            f"-{stats['benchmark_max_drawdown'] * 100:.1f}%",
        ],
        ["Volatilidad", f"{stats['volatilidad'] * 100:.1f}%", "—"],
    ]
    out.append(
        render_table(["Metrica", "Estrategia", "Comprar y mantener"], rows, ["<", ">", ">"])
    )

    out.append("")
    out.append(
        f"Capital: {stats['capital_inicial']:,.2f} -> {stats['capital_final']:,.2f}   "
        f"Rebalanceos: {stats['rebalanceos']}"
    )
    excess = stats["exceso_cagr"]
    verdict = "supera" if excess > 0 else "no supera"
    out.append(f"La estrategia {verdict} al benchmark por {excess * 100:+.1f}% anual.")
    return "\n".join(out)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)


def write_csv(path: str, evaluations: Iterable[Evaluation]) -> None:
    """Flat CSV of the ranking, one row per symbol."""
    fields = [
        "symbol",
        "as_of",
        "price",
        "score",
        "score_100",
        "verdict",
        "confidence",
        "coverage",
        "shares",
        "stop",
        "target",
        "warnings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ev in evaluations:
            writer.writerow(
                {
                    "symbol": ev.symbol,
                    "as_of": ev.as_of or "",
                    "price": round(ev.price, 4),
                    "score": round(ev.score, 4),
                    "score_100": round(ev.score_100, 2),
                    "verdict": ev.verdict.value,
                    "confidence": round(ev.confidence, 3),
                    "coverage": round(ev.coverage, 3),
                    "shares": ev.plan.shares if ev.plan else "",
                    "stop": round(ev.plan.stop_price, 4) if ev.plan else "",
                    "target": round(ev.plan.target_price, 4) if ev.plan else "",
                    "warnings": " | ".join(ev.warnings),
                }
            )
