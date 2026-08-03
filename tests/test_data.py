import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from pirulito.config import Config
from pirulito.data import get_history_provider
from pirulito.data.base import PriceBar, PriceSeries, bars_from_rows
from pirulito.data.csv_file import CsvProvider, load_csv
from pirulito.data.synthetic import SyntheticProvider
from pirulito.data.tradingview import (
    SCANNER_COLUMNS,
    TradingViewError,
    TradingViewScanner,
)
from pirulito.screener import analyze_symbols


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for ``requests.Session`` so the scanner can be tested offline."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "body": json, "timeout": timeout})
        return FakeResponse(self.payload, self.status_code)


def scanner_payload(*symbols):
    """Build a response shaped like the real screener's column-array format."""
    rows = []
    for offset, symbol in enumerate(symbols):
        values = []
        for column in SCANNER_COLUMNS:
            if column == "name":
                values.append(symbol)
            elif column == "description":
                values.append(f"{symbol} Inc.")
            elif column == "exchange":
                values.append("NASDAQ")
            elif column == "sector":
                values.append("Technology")
            elif column == "close":
                values.append(100.0 + offset)
            elif column == "SMA50":
                values.append(95.0)
            elif column == "SMA200":
                values.append(85.0)
            elif column == "RSI":
                values.append(60.0)
            elif column == "MACD.macd":
                values.append(1.5)
            elif column == "MACD.signal":
                values.append(1.0)
            elif column == "ATR":
                values.append(3.0)
            elif column == "ADX":
                values.append(28.0)
            elif column == "ADX+DI":
                values.append(25.0)
            elif column == "ADX-DI":
                values.append(12.0)
            elif column == "Volatility.D":
                values.append(1.4)
            elif column.startswith("Perf."):
                values.append(20.0)
            elif column == "price_52_week_high":
                values.append(110.0)
            elif column == "price_52_week_low":
                values.append(70.0)
            elif column == "average_volume_10d_calc":
                values.append(30_000_000)
            elif column == "average_volume_60d_calc":
                values.append(25_000_000)
            elif column == "Recommend.All":
                values.append(0.5)
            elif column == "market_cap_basic":
                values.append(1e12)
            else:
                values.append(0.0)
        rows.append({"s": f"NASDAQ:{symbol}", "d": values})
    return {"totalCount": len(rows), "data": rows}


class TestScannerRequests(unittest.TestCase):
    def test_scan_maps_columns_onto_names(self):
        session = FakeSession(scanner_payload("AAPL", "MSFT"))
        scanner = TradingViewScanner(session=session)
        rows = scanner.scan(limit=2)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["name"], "AAPL")
        self.assertEqual(rows[0]["close"], 100.0)
        self.assertEqual(rows[0]["ticker"], "NASDAQ:AAPL")

    def test_scan_posts_to_the_right_market(self):
        session = FakeSession(scanner_payload("AAPL"))
        TradingViewScanner(market="argentina", session=session).scan(limit=1)
        self.assertIn("/argentina/scan", session.calls[0]["url"])

    def test_filters_include_price_and_liquidity(self):
        session = FakeSession(scanner_payload("AAPL"))
        TradingViewScanner(session=session, min_price=7.0).scan()
        filters = session.calls[0]["body"]["filter"]
        self.assertTrue(any(f["left"] == "close" and f["right"] == 7.0 for f in filters))

    def test_exchange_filter_is_passed_through(self):
        session = FakeSession(scanner_payload("AAPL"))
        TradingViewScanner(session=session, exchanges=["NYSE"]).scan()
        filters = session.calls[0]["body"]["filter"]
        self.assertTrue(any(f["left"] == "exchange" for f in filters))

    def test_quotes_qualify_bare_symbols(self):
        session = FakeSession(scanner_payload("AAPL"))
        scanner = TradingViewScanner(session=session, exchanges=["NYSE"])
        scanner.quotes(["aapl", "NASDAQ:MSFT"])
        tickers = session.calls[0]["body"]["symbols"]["tickers"]
        self.assertEqual(tickers, ["NYSE:AAPL", "NASDAQ:MSFT"])

    def test_quotes_with_no_symbols_makes_no_request(self):
        session = FakeSession(scanner_payload())
        self.assertEqual(TradingViewScanner(session=session).quotes([]), [])
        self.assertEqual(session.calls, [])

    def test_http_failure_becomes_a_tradingview_error(self):
        session = FakeSession({}, status_code=500)
        with self.assertRaises(TradingViewError):
            TradingViewScanner(session=session).scan()

    def test_empty_response_yields_no_rows(self):
        session = FakeSession({"totalCount": 0, "data": []})
        self.assertEqual(TradingViewScanner(session=session).scan(), [])


class TestScannerScoring(unittest.TestCase):
    def test_analyze_symbols_scores_the_response(self):
        session = FakeSession(scanner_payload("AAPL", "MSFT"))
        scanner = TradingViewScanner(session=session)
        report = analyze_symbols(["AAPL", "MSFT"], Config(), scanner=scanner)

        self.assertEqual(len(report), 2)
        self.assertTrue(all(e.score > 0 for e in report.evaluations))
        # The scanner supplies the TradingView rating that history cannot.
        names = {s.name for s in report.evaluations[0].signals}
        self.assertIn("tv_rating", names)

    def test_results_are_ranked_best_first(self):
        session = FakeSession(scanner_payload("AAPL", "MSFT", "NVDA"))
        report = analyze_symbols(
            ["AAPL", "MSFT", "NVDA"], Config(), scanner=TradingViewScanner(session=session)
        )
        scores = [e.score for e in report.evaluations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_missing_symbol_is_reported_as_an_error(self):
        session = FakeSession(scanner_payload("AAPL"))
        report = analyze_symbols(
            ["AAPL", "GHOST"], Config(), scanner=TradingViewScanner(session=session)
        )
        self.assertIn("GHOST", report.errors)

    def test_report_serializes(self):
        session = FakeSession(scanner_payload("AAPL"))
        report = analyze_symbols(["AAPL"], Config(), scanner=TradingViewScanner(session=session))
        payload = json.loads(json.dumps(report.to_dict(), default=str))
        self.assertEqual(payload["count"], 1)


class TestPriceSeries(unittest.TestCase):
    def make(self):
        return bars_from_rows(
            [
                (date(2024, 1, 3), 10, 11, 9, 10.5, 1000),
                (date(2024, 1, 1), 9, 10, 8, 9.5, 900),
                (date(2024, 1, 2), 9.5, 10.5, 9, 10.0, 950),
            ],
            "TEST",
        )

    def test_bars_are_sorted_by_date(self):
        self.assertEqual(self.make().dates, [date(2024, 1, i) for i in (1, 2, 3)])

    def test_duplicate_dates_keep_one_bar(self):
        bars = [
            PriceBar(date(2024, 1, 1), 1, 2, 0.5, 1.5, 10),
            PriceBar(date(2024, 1, 1), 1, 2, 0.5, 9.9, 10),
        ]
        series = PriceSeries("X", bars)
        self.assertEqual(len(series), 1)

    def test_truncate_hides_the_future(self):
        series = self.make()
        self.assertEqual(len(series.truncate(date(2024, 1, 2))), 2)

    def test_truncate_before_start_is_empty(self):
        self.assertEqual(len(self.make().truncate(date(2023, 1, 1))), 0)

    def test_tail_takes_the_most_recent(self):
        self.assertEqual(self.make().tail(2).dates[0], date(2024, 1, 2))

    def test_last_on_empty_series_raises(self):
        with self.assertRaises(ValueError):
            PriceSeries("X", []).last

    def test_validate_rejects_low_above_high(self):
        with self.assertRaises(ValueError):
            PriceBar(date(2024, 1, 1), 10, 9, 11, 10, 100).validate()

    def test_validate_rejects_negative_price(self):
        with self.assertRaises(ValueError):
            PriceBar(date(2024, 1, 1), -1, 9, 1, 5, 100).validate()


class TestSyntheticProvider(unittest.TestCase):
    def test_is_deterministic(self):
        a = SyntheticProvider(end_date=date(2025, 1, 1)).fetch("AAPL", 300)
        b = SyntheticProvider(end_date=date(2025, 1, 1)).fetch("AAPL", 300)
        self.assertEqual(a.closes, b.closes)

    def test_different_symbols_differ(self):
        provider = SyntheticProvider(end_date=date(2025, 1, 1))
        self.assertNotEqual(
            provider.fetch("AAPL", 300).closes, provider.fetch("MSFT", 300).closes
        )

    def test_produces_requested_number_of_bars(self):
        self.assertEqual(len(SyntheticProvider().fetch("X", 250)), 250)

    def test_bars_are_internally_consistent(self):
        SyntheticProvider().fetch("X", 300).validate()

    def test_skips_weekends(self):
        for bar in SyntheticProvider().fetch("X", 100).bars:
            self.assertLess(bar.date.weekday(), 5)


class TestCsvProvider(unittest.TestCase):
    def write(self, text, name="AAPL.csv"):
        directory = Path(tempfile.mkdtemp())
        (directory / name).write_text(text, encoding="utf-8")
        return directory

    def test_reads_standard_ohlcv(self):
        directory = self.write(
            "Date,Open,High,Low,Close,Volume\n"
            "2024-01-01,10,11,9,10.5,1000\n"
            "2024-01-02,10.5,12,10,11.5,1200\n"
        )
        series = CsvProvider(directory).fetch("AAPL", 10)
        self.assertEqual(len(series), 2)
        self.assertAlmostEqual(series.closes[-1], 11.5)

    def test_accepts_spanish_headers(self):
        directory = self.write(
            "fecha,apertura,maximo,minimo,cierre,volumen\n2024-01-01,10,11,9,10.5,1000\n"
        )
        self.assertEqual(len(CsvProvider(directory).fetch("AAPL", 10)), 1)

    def test_close_only_file_synthesizes_the_other_fields(self):
        directory = self.write("Date,Close\n2024-01-01,10.5\n2024-01-02,11.0\n")
        series = CsvProvider(directory).fetch("AAPL", 10)
        self.assertAlmostEqual(series.bars[0].open, 10.5)
        self.assertAlmostEqual(series.bars[0].high, 10.5)

    def test_missing_close_column_raises(self):
        directory = self.write("Date,Open\n2024-01-01,10\n")
        with self.assertRaises(ValueError):
            CsvProvider(directory).fetch("AAPL", 10)

    def test_unknown_symbol_returns_none(self):
        directory = self.write("Date,Close\n2024-01-01,10\n")
        self.assertIsNone(CsvProvider(directory).fetch("NOPE", 10))

    def test_unparseable_rows_are_skipped(self):
        directory = self.write(
            "Date,Close\n2024-01-01,10\nnot-a-date,11\n2024-01-03,n/a\n2024-01-04,12\n"
        )
        self.assertEqual(len(CsvProvider(directory).fetch("AAPL", 10)), 2)

    def test_unix_timestamps_parse(self):
        directory = self.write("time,close\n1704067200,10.5\n")
        series = load_csv(directory / "AAPL.csv")
        self.assertEqual(series.bars[0].date.year, 2024)


class TestProviderRegistry(unittest.TestCase):
    def test_builds_known_providers(self):
        self.assertIsInstance(get_history_provider("synthetic"), SyntheticProvider)

    def test_unknown_provider_lists_the_options(self):
        with self.assertRaises(ValueError) as ctx:
            get_history_provider("bloomberg")
        self.assertIn("synthetic", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
