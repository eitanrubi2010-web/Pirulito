import unittest
from datetime import date, timedelta

from pirulito.backtest import _select, _turnover, run_backtest
from pirulito.config import Config
from pirulito.data.base import PriceBar, PriceSeries


def ramp(symbol, n=500, start=100.0, growth=0.001, crash_at=None, crash_to=0.4):
    """A smooth series, optionally collapsing to ``crash_to`` after ``crash_at``."""
    bars = []
    day = date(2023, 1, 2)
    price = start
    for i in range(n):
        if crash_at is not None and i == crash_at:
            price *= crash_to
        else:
            price *= 1 + growth
        bars.append(
            PriceBar(
                date=day + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=5_000_000,
            )
        )
    return PriceSeries(symbol, bars)


class TestNoLookahead(unittest.TestCase):
    """The decision at a date must not change when future bars are appended."""

    def setUp(self):
        self.cfg = Config()

    def test_picks_ignore_bars_after_the_decision_date(self):
        decision_day = ramp("A", n=400).bars[-1].date

        # Same first 400 bars in both worlds; one of them then crashes hard.
        short = {"A": ramp("A", n=400), "B": ramp("B", n=400, growth=0.0005)}
        long = {
            "A": ramp("A", n=500, crash_at=401, crash_to=0.2),
            "B": ramp("B", n=500, growth=0.0005),
        }

        picks_short, scores_short = _select(short, self.cfg, decision_day, top_n=2)
        picks_long, scores_long = _select(long, self.cfg, decision_day, top_n=2)

        self.assertEqual(picks_short, picks_long)
        for symbol in scores_short:
            self.assertAlmostEqual(scores_short[symbol], scores_long[symbol], places=12)

    def test_truncation_is_what_hides_the_future(self):
        series = ramp("A", n=500, crash_at=401, crash_to=0.2)
        cutoff = series.bars[399].date
        visible = series.truncate(cutoff)
        self.assertEqual(len(visible), 400)
        self.assertGreater(visible.closes[-1], series.closes[-1])


class TestRunBacktest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.histories = {
            "UP": ramp("UP", n=600, growth=0.0015),
            "FLAT": ramp("FLAT", n=600, growth=0.0),
            "DOWN": ramp("DOWN", n=600, growth=-0.001),
        }

    def test_curves_align_with_the_timeline(self):
        result = run_backtest(self.histories, self.cfg, top_n=1, rebalance_days=21)
        self.assertEqual(len(result.equity), len(result.dates))
        self.assertEqual(len(result.benchmark), len(result.dates))
        self.assertGreater(len(result.rebalances), 0)

    def test_prefers_the_rising_stock(self):
        result = run_backtest(self.histories, self.cfg, top_n=1, rebalance_days=21)
        chosen = [s for r in result.rebalances for s in r.picks]
        self.assertTrue(chosen, "the strategy should hold something")
        self.assertNotIn("DOWN", chosen)

    def test_beats_equal_weight_when_only_one_stock_rises(self):
        result = run_backtest(self.histories, self.cfg, top_n=1, rebalance_days=21)
        self.assertGreater(result.total_return, result.benchmark_total_return)

    def test_holdings_never_exceed_top_n(self):
        result = run_backtest(self.histories, self.cfg, top_n=2, rebalance_days=21)
        for rebalance in result.rebalances:
            self.assertLessEqual(len(rebalance.picks), 2)

    def test_commission_reduces_the_result(self):
        free = run_backtest(self.histories, self.cfg, top_n=1, commission_bps=0.0)
        costly = run_backtest(self.histories, self.cfg, top_n=1, commission_bps=100.0)
        self.assertGreater(free.total_return, costly.total_return)

    def test_starts_at_the_configured_capital(self):
        result = run_backtest(self.histories, self.cfg, top_n=1, initial_capital=50_000)
        self.assertAlmostEqual(result.equity[0], 50_000, delta=50_000 * 0.02)

    def test_equity_never_goes_negative(self):
        result = run_backtest(self.histories, self.cfg, top_n=2)
        self.assertTrue(all(v >= 0 for v in result.equity))

    def test_stats_expose_both_sides_of_the_comparison(self):
        stats = run_backtest(self.histories, self.cfg, top_n=1).stats()
        for key in ("cagr", "sharpe", "max_drawdown", "benchmark_cagr", "exceso_cagr"):
            self.assertIn(key, stats)

    def test_all_cash_when_nothing_qualifies(self):
        # Every symbol falling means no BUY verdicts, so the portfolio sits out.
        bearish = {
            "A": ramp("A", n=600, growth=-0.002),
            "B": ramp("B", n=600, growth=-0.0025),
        }
        result = run_backtest(bearish, self.cfg, top_n=2, rebalance_days=21)
        self.assertGreater(result.total_return, result.benchmark_total_return)

    def test_rejects_empty_universe(self):
        with self.assertRaises(ValueError):
            run_backtest({}, self.cfg)

    def test_rejects_too_little_history(self):
        with self.assertRaises(ValueError):
            run_backtest({"A": ramp("A", n=100)}, self.cfg)

    def test_rejects_non_positive_top_n(self):
        with self.assertRaises(ValueError):
            run_backtest(self.histories, self.cfg, top_n=0)


class TestTurnover(unittest.TestCase):
    def test_first_entry_trades_the_full_allocation(self):
        traded = _turnover({}, ["A", "B"], 10_000.0, {"A": 100.0, "B": 50.0})
        self.assertAlmostEqual(traded, 10_000.0)

    def test_holding_the_same_position_trades_nothing(self):
        current = {"A": 100.0}  # 100 shares at 100 = 10,000
        traded = _turnover(current, ["A"], 10_000.0, {"A": 100.0})
        self.assertAlmostEqual(traded, 0.0)

    def test_swapping_positions_trades_both_sides(self):
        current = {"A": 100.0}
        traded = _turnover(current, ["B"], 10_000.0, {"A": 100.0, "B": 50.0})
        self.assertAlmostEqual(traded, 20_000.0)

    def test_no_equity_means_no_turnover(self):
        self.assertEqual(_turnover({"A": 1.0}, ["A"], 0.0, {"A": 10.0}), 0.0)


if __name__ == "__main__":
    unittest.main()
