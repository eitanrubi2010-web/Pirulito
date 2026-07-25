import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from pirulito.cli import build_parser, main


class TestOptionPlacement(unittest.TestCase):
    """Shared options must work on either side of the subcommand.

    Argparse resets a parent-parser option when the subparser reparses it, so
    without SUPPRESS defaults `pirulito --csv out.csv scan` silently drops the
    flag. These tests pin both orderings.
    """

    def parse(self, argv):
        return build_parser().parse_args(argv)

    def test_capital_before_subcommand(self):
        args = self.parse(["--capital", "50000", "analyze", "AAPL"])
        self.assertEqual(args.capital, 50000.0)

    def test_capital_after_subcommand(self):
        args = self.parse(["analyze", "AAPL", "--capital", "50000"])
        self.assertEqual(args.capital, 50000.0)

    def test_csv_after_subcommand(self):
        args = self.parse(["scan", "--csv", "out.csv"])
        self.assertEqual(args.csv_out, "out.csv")

    def test_json_before_subcommand(self):
        args = self.parse(["--json", "out.json", "scan"])
        self.assertEqual(args.json_out, "out.json")

    def test_unused_options_are_absent_not_none_clobbered(self):
        args = self.parse(["--capital", "1000", "scan"])
        self.assertEqual(args.capital, 1000.0)
        self.assertIsNone(getattr(args, "csv_out", None))

    def test_csv_dir_does_not_collide_with_csv(self):
        args = self.parse(
            ["analyze", "AAPL", "--csv-dir", "datos", "--csv", "salida.csv"]
        )
        self.assertEqual(args.csv_dir, "datos")
        self.assertEqual(args.csv_out, "salida.csv")

    def test_subcommand_is_required(self):
        with self.assertRaises(SystemExit):
            self.parse([])


class TestEndToEnd(unittest.TestCase):
    """Full CLI runs against the offline synthetic provider."""

    def run_cli(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_analyze_prints_a_ranking(self):
        code, output = self.run_cli(
            ["analyze", "AAPL", "MSFT", "--history", "--provider", "synthetic", "--no-detail"]
        )
        self.assertEqual(code, 0)
        self.assertIn("AAPL", output)
        self.assertIn("Veredicto", output)

    def test_analyze_writes_both_export_formats(self):
        directory = Path(tempfile.mkdtemp())
        csv_path = directory / "r.csv"
        json_path = directory / "r.json"
        code, _ = self.run_cli(
            [
                "analyze", "MSFT", "--history", "--provider", "synthetic",
                "--no-detail", "--csv", str(csv_path), "--json", str(json_path),
            ]
        )
        self.assertEqual(code, 0)
        self.assertTrue(csv_path.exists())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["count"], 1)

    def test_capital_changes_the_position_size(self):
        _, small = self.run_cli(
            ["analyze", "MSFT", "--history", "--provider", "synthetic", "--capital", "10000"]
        )
        _, large = self.run_cli(
            ["analyze", "MSFT", "--history", "--provider", "synthetic", "--capital", "100000"]
        )
        self.assertIn("Plan de posicion", small)
        self.assertNotEqual(small, large)

    def test_config_command_emits_valid_json(self):
        code, output = self.run_cli(["config"])
        self.assertEqual(code, 0)
        self.assertIn("weights", json.loads(output))

    def test_config_reflects_capital_override(self):
        _, output = self.run_cli(["--capital", "77000", "config"])
        self.assertEqual(json.loads(output)["capital"], 77000.0)

    def test_backtest_runs_and_reports(self):
        code, output = self.run_cli(
            [
                "backtest", "AAPL", "MSFT", "NVDA", "KO",
                "--provider", "synthetic", "--years", "2", "--top-n", "2",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("Comprar y mantener", output)
        self.assertIn("CAGR", output)

    def test_invalid_config_file_is_reported(self):
        directory = Path(tempfile.mkdtemp())
        bad = directory / "bad.json"
        bad.write_text('{"sma_fast": 300, "sma_slow": 50}', encoding="utf-8")
        code, _ = self.run_cli(["--config", str(bad), "config"])
        self.assertEqual(code, 2)

    def test_csv_provider_without_directory_fails_clearly(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["analyze", "AAPL", "--history", "--provider", "csv"])


if __name__ == "__main__":
    unittest.main()
