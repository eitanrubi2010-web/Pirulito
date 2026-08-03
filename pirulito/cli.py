"""Command line interface.

    python -m pirulito scan --market america --limit 200
    python -m pirulito analyze AAPL MSFT NVDA
    python -m pirulito analyze AAPL --history --provider tradingview
    python -m pirulito backtest AAPL MSFT KO --provider synthetic --years 3
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .backtest import run_backtest
from .config import Config
from .data import CsvProvider, SyntheticProvider, get_history_provider
from .data.tradingview import TradingViewError, TradingViewScanner
from .report import (
    format_backtest,
    format_detail,
    format_scan,
    write_csv,
    write_json,
)
from .screener import analyze_symbols, analyze_with_history, scan_market

DISCLAIMER = (
    "Esto es una herramienta de analisis, no asesoramiento financiero. "
    "El rendimiento pasado no predice el futuro."
)


def _common_options() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    They are attached to the main parser and to every subparser. The defaults
    are ``SUPPRESS`` so that an unused occurrence leaves the attribute alone
    instead of overwriting a value given on the other side of the subcommand —
    argparse would otherwise reset ``pirulito --csv out.csv scan`` back to None.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", default=argparse.SUPPRESS, help="Ruta a un JSON de configuracion"
    )
    common.add_argument(
        "--capital",
        type=float,
        default=argparse.SUPPRESS,
        help="Capital total para dimensionar posiciones",
    )
    common.add_argument(
        "--risk",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraccion del capital a arriesgar por operacion (ej. 0.01)",
    )
    common.add_argument(
        "--json",
        dest="json_out",
        default=argparse.SUPPRESS,
        help="Guardar resultados en JSON",
    )
    common.add_argument(
        "--csv",
        dest="csv_out",
        default=argparse.SUPPRESS,
        help="Guardar el ranking en CSV",
    )
    return common


def _opt(args: argparse.Namespace, name: str, fallback=None):
    return getattr(args, name, fallback)


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="pirulito",
        description="Rastrea acciones con datos de TradingView y las puntua.",
        epilog=DISCLAIMER,
        parents=[common],
    )

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan", parents=[common], help="Escanear un mercado entero via TradingView"
    )
    scan.add_argument("--market", default="america", help="america, argentina, spain...")
    scan.add_argument("--exchange", action="append", help="Filtrar por bolsa (repetible)")
    scan.add_argument("--limit", type=int, default=200, help="Cuantos simbolos traer")
    scan.add_argument("--top", type=int, default=20, help="Cuantos mostrar")
    scan.add_argument("--min-score", type=float, help="Filtrar por puntaje minimo (-1 a 1)")
    scan.add_argument("--only-buys", action="store_true", help="Solo COMPRA y COMPRA FUERTE")
    scan.add_argument("--detail", type=int, default=0, help="Desglose de los N mejores")

    analyze = sub.add_parser(
        "analyze", parents=[common], help="Analizar simbolos concretos"
    )
    analyze.add_argument("symbols", nargs="+", help="Ej: AAPL NASDAQ:MSFT")
    analyze.add_argument("--market", default="america")
    analyze.add_argument(
        "--history",
        action="store_true",
        help="Usar historico OHLCV en vez del snapshot del screener",
    )
    analyze.add_argument(
        "--provider",
        default="tradingview",
        choices=["tradingview", "csv", "synthetic"],
        help="Fuente del historico (solo con --history)",
    )
    analyze.add_argument("--csv-dir", help="Carpeta de CSVs (con --provider csv)")
    analyze.add_argument("--lookback", type=int, default=400, help="Barras a descargar")
    analyze.add_argument("--no-detail", action="store_true", help="Solo la tabla")

    backtest = sub.add_parser(
        "backtest", parents=[common], help="Simular la estrategia sobre historicos"
    )
    backtest.add_argument("symbols", nargs="+")
    backtest.add_argument(
        "--provider", default="synthetic", choices=["tradingview", "csv", "synthetic"]
    )
    backtest.add_argument("--csv-dir")
    backtest.add_argument("--years", type=float, default=3.0, help="Años a simular")
    backtest.add_argument("--top-n", type=int, default=5, help="Posiciones simultaneas")
    backtest.add_argument(
        "--rebalance", type=int, default=21, help="Dias habiles entre rebalanceos"
    )
    backtest.add_argument(
        "--commission", type=float, default=5.0, help="Comision en puntos basicos"
    )

    sub.add_parser(
        "config", parents=[common], help="Imprimir la configuracion efectiva en JSON"
    )
    return parser


def _load_config(args: argparse.Namespace) -> Config:
    config_path = _opt(args, "config")
    cfg = Config.load(config_path) if config_path else Config()
    capital = _opt(args, "capital")
    if capital is not None:
        cfg.capital = capital
    risk = _opt(args, "risk")
    if risk is not None:
        cfg.risk_per_trade = risk
    cfg.__post_init__()
    return cfg


def _history_provider(args: argparse.Namespace):
    name = getattr(args, "provider", "synthetic")
    if name == "csv":
        directory = getattr(args, "csv_dir", None)
        if not directory:
            raise SystemExit("--csv-dir es obligatorio cuando --provider csv")
        return CsvProvider(directory)
    if name == "synthetic":
        return SyntheticProvider()
    return get_history_provider(name)


def _cmd_scan(args: argparse.Namespace, cfg: Config) -> int:
    scanner = TradingViewScanner(market=args.market, exchanges=args.exchange)
    report = scan_market(cfg, scanner=scanner, limit=args.limit, min_score=args.min_score)

    if args.only_buys:
        report.evaluations = report.buys()

    print(format_scan(report, limit=args.top))
    for evaluation in report.evaluations[: args.detail]:
        print()
        print(format_detail(evaluation))

    _write_outputs(args, report)
    return 0


def _cmd_analyze(args: argparse.Namespace, cfg: Config) -> int:
    if args.history:
        provider = _history_provider(args)
        report = analyze_with_history(
            args.symbols, provider, cfg, lookback_days=args.lookback
        )
    else:
        scanner = TradingViewScanner(market=args.market)
        report = analyze_symbols(args.symbols, cfg, scanner=scanner)

    print(format_scan(report, limit=len(args.symbols)))
    if not args.no_detail:
        for evaluation in report.evaluations:
            print()
            print(format_detail(evaluation))

    _write_outputs(args, report)
    return 0


def _cmd_backtest(args: argparse.Namespace, cfg: Config) -> int:
    provider = _history_provider(args)
    lookback = int(args.years * 252) + cfg.required_bars + 30

    histories = {}
    failures = {}
    for symbol in args.symbols:
        try:
            series = provider.fetch(symbol, lookback)
        except Exception as exc:
            failures[symbol] = str(exc)
            continue
        if series and len(series) > cfg.required_bars:
            histories[symbol] = series
        else:
            got = len(series) if series else 0
            failures[symbol] = f"solo {got} barras, se necesitan {cfg.required_bars}"

    for symbol, reason in failures.items():
        print(f"Omitido {symbol}: {reason}", file=sys.stderr)
    if not histories:
        print("No hay historicos suficientes para simular.", file=sys.stderr)
        return 1

    result = run_backtest(
        histories,
        cfg,
        top_n=args.top_n,
        rebalance_days=args.rebalance,
        commission_bps=args.commission,
    )
    print(format_backtest(result))

    json_out = _opt(args, "json_out")
    if json_out:
        write_json(
            json_out,
            {
                "stats": result.stats(),
                "rebalances": [
                    {"date": r.date, "picks": r.picks, "equity": round(r.equity, 2)}
                    for r in result.rebalances
                ],
            },
        )
        print(f"\nJSON guardado en {json_out}")
    return 0


def _write_outputs(args: argparse.Namespace, report) -> None:
    json_out = _opt(args, "json_out")
    if json_out:
        write_json(json_out, report.to_dict())
        print(f"\nJSON guardado en {json_out}")
    csv_out = _opt(args, "csv_out")
    if csv_out:
        write_csv(csv_out, report.evaluations)
        print(f"CSV guardado en {csv_out}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = _load_config(args)
    except (ValueError, OSError) as exc:
        print(f"Configuracion invalida: {exc}", file=sys.stderr)
        return 2

    if args.command == "config":
        import json

        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return 0

    handlers = {
        "scan": _cmd_scan,
        "analyze": _cmd_analyze,
        "backtest": _cmd_backtest,
    }
    try:
        code = handlers[args.command](args, cfg)
    except TradingViewError as exc:
        print(f"\nError de TradingView: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
        return 130

    if code == 0:
        print(f"\n{DISCLAIMER}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
