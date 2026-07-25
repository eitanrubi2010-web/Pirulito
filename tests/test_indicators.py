import math
import unittest

from pirulito import indicators as ind


class TestMovingAverages(unittest.TestCase):
    def test_sma_warmup_is_none_then_correct(self):
        values = [1, 2, 3, 4, 5]
        result = ind.sma(values, 3)
        self.assertEqual(result[:2], [None, None])
        self.assertAlmostEqual(result[2], 2.0)
        self.assertAlmostEqual(result[4], 4.0)

    def test_sma_matches_length_of_input(self):
        self.assertEqual(len(ind.sma([1, 2, 3], 2)), 3)

    def test_sma_too_short_is_all_none(self):
        self.assertEqual(ind.sma([1, 2], 5), [None, None])

    def test_sma_rolling_sum_does_not_drift(self):
        # The incremental update could accumulate float error; compare against
        # a direct recomputation over a long noisy series.
        values = [100 + math.sin(i) * 50 for i in range(500)]
        rolling = ind.sma(values, 20)
        for i in (100, 250, 499):
            direct = sum(values[i - 19 : i + 1]) / 20
            self.assertAlmostEqual(rolling[i], direct, places=6)

    def test_ema_seeds_with_sma(self):
        values = [1, 2, 3, 4, 5, 6]
        result = ind.ema(values, 3)
        self.assertAlmostEqual(result[2], 2.0)
        self.assertGreater(result[5], result[2])

    def test_ema_reacts_faster_than_sma(self):
        values = [10.0] * 30 + [20.0] * 5
        fast = ind.ema(values, 10)[-1]
        slow = ind.sma(values, 10)[-1]
        self.assertGreater(fast, slow)

    def test_zero_period_rejected(self):
        with self.assertRaises(ValueError):
            ind.sma([1, 2, 3], 0)


class TestRsi(unittest.TestCase):
    def test_monotonic_rise_pins_at_100(self):
        values = list(range(1, 40))
        self.assertAlmostEqual(ind.rsi(values, 14)[-1], 100.0)

    def test_monotonic_fall_pins_near_zero(self):
        values = list(range(40, 1, -1))
        self.assertAlmostEqual(ind.rsi(values, 14)[-1], 0.0)

    def test_stays_within_bounds(self):
        values = [100 + math.sin(i / 3) * 10 for i in range(200)]
        for value in ind.rsi(values, 14):
            if value is not None:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

    def test_flat_series_is_neutral(self):
        self.assertAlmostEqual(ind.rsi([50.0] * 40, 14)[-1], 50.0)


class TestMacd(unittest.TestCase):
    def test_histogram_is_line_minus_signal(self):
        values = [100 + i * 0.5 for i in range(120)]
        line, signal, hist = ind.macd(values)
        for i in range(len(values)):
            if hist[i] is not None:
                self.assertAlmostEqual(hist[i], line[i] - signal[i], places=9)

    def test_uptrend_gives_positive_macd(self):
        values = [100 * (1.01**i) for i in range(120)]
        line, _, _ = ind.macd(values)
        self.assertGreater(line[-1], 0)

    def test_fast_must_be_shorter_than_slow(self):
        with self.assertRaises(ValueError):
            ind.macd([1.0] * 100, fast=26, slow=12)


class TestAtr(unittest.TestCase):
    def test_atr_of_constant_range_equals_range(self):
        highs = [11.0] * 40
        lows = [9.0] * 40
        closes = [10.0] * 40
        self.assertAlmostEqual(ind.atr(highs, lows, closes, 14)[-1], 2.0, places=6)

    def test_true_range_accounts_for_gaps(self):
        highs = [10.0, 20.0]
        lows = [9.0, 19.0]
        closes = [9.5, 19.5]
        # Gap up: the true range spans from the prior close, not just the bar.
        self.assertAlmostEqual(ind.true_range(highs, lows, closes)[1], 10.5)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            ind.true_range([1.0, 2.0], [1.0], [1.0, 2.0])


class TestRiskStats(unittest.TestCase):
    def test_max_drawdown_of_rising_series_is_zero(self):
        self.assertAlmostEqual(ind.max_drawdown([1, 2, 3, 4]), 0.0)

    def test_max_drawdown_measures_peak_to_trough(self):
        self.assertAlmostEqual(ind.max_drawdown([100, 120, 60, 90]), 0.5)

    def test_current_drawdown_uses_last_point(self):
        self.assertAlmostEqual(ind.current_drawdown([100, 120, 60, 90]), 0.25)

    def test_current_drawdown_at_high_is_zero(self):
        self.assertAlmostEqual(ind.current_drawdown([100, 120, 60, 130]), 0.0)

    def test_annualized_return_of_doubling_in_one_year(self):
        values = [100 * (2 ** (i / 252)) for i in range(253)]
        self.assertAlmostEqual(ind.annualized_return(values), 1.0, places=2)

    def test_volatility_of_flat_series_is_zero(self):
        self.assertAlmostEqual(ind.annualized_volatility([0.0] * 50), 0.0)

    def test_sharpe_is_zero_when_volatility_is_zero(self):
        self.assertEqual(ind.sharpe_ratio([0.0] * 50), 0.0)


class TestTrendFit(unittest.TestCase):
    def test_perfect_exponential_has_r2_of_one(self):
        values = [100 * (1.001**i) for i in range(200)]
        slope, r2 = ind.linear_trend(values)
        self.assertGreater(slope, 0)
        self.assertAlmostEqual(r2, 1.0, places=6)

    def test_noise_lowers_r2(self):
        clean = [100 * (1.001**i) for i in range(200)]
        noisy = [v * (1 + 0.15 * math.sin(i * 2.3)) for i, v in enumerate(clean)]
        self.assertGreater(ind.linear_trend(clean)[1], ind.linear_trend(noisy)[1])

    def test_short_series_returns_zeros(self):
        self.assertEqual(ind.linear_trend([1.0, 2.0]), (0.0, 0.0))


class TestBeta(unittest.TestCase):
    def test_identical_series_has_beta_one(self):
        market = [0.01 * math.sin(i) for i in range(100)]
        self.assertAlmostEqual(ind.beta(market, market), 1.0, places=9)

    def test_double_amplitude_has_beta_two(self):
        market = [0.01 * math.sin(i) for i in range(100)]
        asset = [2 * r for r in market]
        self.assertAlmostEqual(ind.beta(asset, market), 2.0, places=9)

    def test_too_few_points_returns_none(self):
        self.assertIsNone(ind.beta([0.1] * 5, [0.1] * 5))


class TestScaling(unittest.TestCase):
    def test_squash_is_bounded(self):
        for value in (-1e6, -10, 0, 10, 1e6):
            self.assertGreaterEqual(ind.squash(value, 0.1), -1.0)
            self.assertLessEqual(ind.squash(value, 0.1), 1.0)

    def test_squash_is_monotonic(self):
        values = [ind.squash(x / 10, 0.25) for x in range(-30, 31)]
        self.assertEqual(values, sorted(values))

    def test_squash_scale_sets_the_reference_point(self):
        self.assertAlmostEqual(ind.squash(0.25, 0.25), math.tanh(1.0))

    def test_squash_rejects_non_positive_scale(self):
        with self.assertRaises(ValueError):
            ind.squash(1.0, 0.0)

    def test_clamp_bounds(self):
        self.assertEqual(ind.clamp(5.0), 1.0)
        self.assertEqual(ind.clamp(-5.0), -1.0)
        self.assertEqual(ind.clamp(0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
