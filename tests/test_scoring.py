import unittest
from datetime import date, timedelta

from pirulito.config import Config, Weights
from pirulito.data.base import PriceBar, PriceSeries
from pirulito.features import Features, from_scanner_row, from_series
from pirulito.risk import check_risk_gates, plan_position
from pirulito.scoring import Verdict, classify, downgrade, evaluate
from pirulito.signals import compute_signals, rsi_signal, trend_signal


def make_series(symbol="TEST", n=400, start=100.0, daily_growth=0.001, vol=1_000_000):
    """A smooth, steadily rising history — deliberately boring and predictable."""
    bars = []
    day = date(2024, 1, 1)
    price = start
    for i in range(n):
        price *= 1 + daily_growth
        bars.append(
            PriceBar(
                date=day + timedelta(days=i),
                open=price * 0.995,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=vol,
            )
        )
    return PriceSeries(symbol, bars)


class TestFeaturesFromSeries(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_uptrend_populates_every_field(self):
        f = from_series(make_series(), self.cfg)
        self.assertEqual(f.source, "history")
        for field in (
            "sma_fast",
            "sma_slow",
            "rsi",
            "macd_hist",
            "momentum_return",
            "sharpe",
            "high_52w",
            "atr",
            "trend_r2",
        ):
            self.assertIsNotNone(getattr(f, field), f"{field} should be computed")

    def test_short_history_leaves_long_windows_empty(self):
        f = from_series(make_series(n=60), self.cfg)
        self.assertIsNotNone(f.rsi)
        self.assertIsNone(f.sma_slow)
        self.assertIsNone(f.momentum_return)

    def test_empty_series_rejected(self):
        with self.assertRaises(ValueError):
            from_series(PriceSeries("X", []), self.cfg)

    def test_price_is_the_last_close(self):
        series = make_series()
        self.assertAlmostEqual(from_series(series, self.cfg).price, series.closes[-1])


class TestFeaturesFromScanner(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.row = {
            "name": "AAPL",
            "description": "Apple Inc.",
            "close": 200.0,
            "SMA50": 190.0,
            "SMA200": 170.0,
            "RSI": 58.0,
            "MACD.macd": 2.5,
            "MACD.signal": 1.5,
            "ATR": 4.0,
            "ADX": 30.0,
            "ADX+DI": 28.0,
            "ADX-DI": 15.0,
            "Volatility.D": 1.5,
            "Perf.Y": 30.0,
            "Perf.1M": 5.0,
            "price_52_week_high": 210.0,
            "price_52_week_low": 140.0,
            "average_volume_10d_calc": 50_000_000,
            "average_volume_60d_calc": 45_000_000,
            "Recommend.All": 0.6,
        }

    def test_percentages_become_fractions(self):
        f = from_scanner_row(self.row, self.cfg)
        self.assertAlmostEqual(f.annual_return, 0.30)
        self.assertAlmostEqual(f.price_change_recent, 0.05)

    def test_daily_volatility_is_annualized(self):
        f = from_scanner_row(self.row, self.cfg)
        self.assertAlmostEqual(f.annual_volatility, 0.015 * (252**0.5), places=6)

    def test_macd_histogram_is_line_minus_signal(self):
        self.assertAlmostEqual(from_scanner_row(self.row, self.cfg).macd_hist, 1.0)

    def test_momentum_excludes_the_last_month(self):
        f = from_scanner_row(self.row, self.cfg)
        self.assertAlmostEqual(f.momentum_return, 1.30 / 1.05 - 1.0, places=9)

    def test_snapshot_has_no_macd_slope(self):
        self.assertIsNone(from_scanner_row(self.row, self.cfg).macd_hist_slope)

    def test_drawdown_measured_from_52_week_high(self):
        f = from_scanner_row(self.row, self.cfg)
        self.assertAlmostEqual(f.current_drawdown, 10.0 / 210.0, places=9)

    def test_missing_close_rejected(self):
        with self.assertRaises(ValueError):
            from_scanner_row({"name": "X", "close": None}, self.cfg)

    def test_missing_optional_fields_stay_none(self):
        f = from_scanner_row({"name": "X", "close": 50.0}, self.cfg)
        self.assertIsNone(f.rsi)
        self.assertIsNone(f.sharpe)


class TestSignals(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_every_signal_stays_within_bounds(self):
        for series in (
            make_series(daily_growth=0.002),
            make_series(daily_growth=-0.002),
            make_series(daily_growth=0.0),
        ):
            for signal in compute_signals(from_series(series, self.cfg), self.cfg):
                self.assertGreaterEqual(signal.score, -1.0, signal.name)
                self.assertLessEqual(signal.score, 1.0, signal.name)

    def test_trend_signal_sign_follows_direction(self):
        up = trend_signal(from_series(make_series(daily_growth=0.002), self.cfg), self.cfg)
        down = trend_signal(from_series(make_series(daily_growth=-0.002), self.cfg), self.cfg)
        self.assertGreater(up.score, 0.5)
        self.assertLess(down.score, -0.5)

    def test_trend_signal_needs_both_averages(self):
        f = Features(symbol="X", price=10.0, source="test", sma_fast=9.0)
        self.assertIsNone(trend_signal(f, self.cfg))

    def test_rsi_peaks_at_sixty(self):
        def score_at(value):
            f = Features(symbol="X", price=10.0, source="test", rsi=value)
            return rsi_signal(f, self.cfg).score

        self.assertAlmostEqual(score_at(60), 1.0)
        self.assertAlmostEqual(score_at(40), 0.0)
        self.assertAlmostEqual(score_at(80), 0.0)
        self.assertLess(score_at(95), 0.0)
        self.assertLess(score_at(15), 0.0)

    def test_volume_signal_flips_sign_with_price_direction(self):
        rising = Features(
            symbol="X", price=10.0, source="test", rel_volume=2.0, price_change_recent=0.10
        )
        falling = Features(
            symbol="X", price=10.0, source="test", rel_volume=2.0, price_change_recent=-0.10
        )
        from pirulito.signals import volume_signal

        self.assertGreater(volume_signal(rising, self.cfg).score, 0)
        self.assertLess(volume_signal(falling, self.cfg).score, 0)

    def test_trend_quality_falls_back_to_adx(self):
        from pirulito.signals import trend_quality_signal

        f = Features(
            symbol="X", price=10.0, source="scanner", adx=35.0, di_plus=30.0, di_minus=10.0
        )
        signal = trend_quality_signal(f, self.cfg)
        self.assertIsNotNone(signal)
        self.assertGreater(signal.score, 0)
        self.assertIn("ADX", signal.rationale)

    def test_zero_weight_signals_are_dropped(self):
        cfg = Config(weights=Weights(trend=0.0, momentum=1.0))
        names = {s.name for s in compute_signals(from_series(make_series(), cfg), cfg)}
        self.assertNotIn("trend", names)
        self.assertIn("momentum", names)


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_bands_map_to_expected_verdicts(self):
        self.assertEqual(classify(0.9, self.cfg), Verdict.STRONG_BUY)
        self.assertEqual(classify(0.20, self.cfg), Verdict.BUY)
        self.assertEqual(classify(0.0, self.cfg), Verdict.HOLD)
        self.assertEqual(classify(-0.20, self.cfg), Verdict.REDUCE)
        self.assertEqual(classify(-0.9, self.cfg), Verdict.SELL)

    def test_thresholds_are_inclusive_at_the_boundary(self):
        self.assertEqual(classify(self.cfg.buy_above, self.cfg), Verdict.BUY)
        self.assertEqual(classify(self.cfg.strong_buy_above, self.cfg), Verdict.STRONG_BUY)

    def test_downgrade_moves_toward_sell_and_stops(self):
        self.assertEqual(downgrade(Verdict.STRONG_BUY), Verdict.BUY)
        self.assertEqual(downgrade(Verdict.STRONG_BUY, 2), Verdict.HOLD)
        self.assertEqual(downgrade(Verdict.SELL, 5), Verdict.SELL)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_steady_uptrend_is_a_buy(self):
        ev = evaluate(from_series(make_series(daily_growth=0.0015), self.cfg), self.cfg)
        self.assertIn(ev.verdict, (Verdict.BUY, Verdict.STRONG_BUY))
        self.assertGreater(ev.score, self.cfg.buy_above)

    def test_steady_downtrend_is_not_a_buy(self):
        ev = evaluate(from_series(make_series(daily_growth=-0.0015), self.cfg), self.cfg)
        self.assertIn(ev.verdict, (Verdict.SELL, Verdict.REDUCE, Verdict.HOLD))
        self.assertLess(ev.score, 0)

    def test_score_100_is_a_rescaling_of_score(self):
        ev = evaluate(from_series(make_series(), self.cfg), self.cfg)
        self.assertAlmostEqual(ev.score_100, (ev.score + 1) * 50)

    def test_coverage_is_full_with_complete_history(self):
        ev = evaluate(from_series(make_series(), self.cfg), self.cfg)
        # Only the TradingView rating is unavailable from local history.
        expected = 1 - self.cfg.weights.tv_rating / self.cfg.weights.total()
        self.assertAlmostEqual(ev.coverage, expected, places=6)

    def test_illiquid_stock_is_downgraded(self):
        thin = make_series(daily_growth=0.0015, vol=10)
        ev = evaluate(from_series(thin, self.cfg), self.cfg)
        self.assertTrue(ev.warnings)
        self.assertTrue(ev.was_downgraded)
        # Higher rank means more bearish, so the gate moved it down the scale.
        self.assertGreater(ev.verdict.rank, ev.raw_verdict.rank)

    def test_no_usable_signals_yields_hold(self):
        f = Features(symbol="X", price=10.0, source="test")
        ev = evaluate(f, self.cfg)
        self.assertEqual(ev.verdict, Verdict.HOLD)
        self.assertEqual(ev.score, 0.0)
        self.assertEqual(ev.coverage, 0.0)
        self.assertTrue(ev.warnings)

    def test_plan_only_attached_to_buys(self):
        buy = evaluate(from_series(make_series(daily_growth=0.0015), self.cfg), self.cfg)
        sell = evaluate(from_series(make_series(daily_growth=-0.002), self.cfg), self.cfg)
        self.assertIsNotNone(buy.plan)
        self.assertIsNone(sell.plan)

    def test_serializes_to_json_safe_dict(self):
        import json

        ev = evaluate(from_series(make_series(), self.cfg), self.cfg)
        payload = json.loads(json.dumps(ev.to_dict(), default=str))
        self.assertEqual(payload["symbol"], "TEST")
        self.assertIn("signals", payload)


class TestRisk(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(capital=10_000.0, risk_per_trade=0.01, atr_stop_multiple=2.0)

    def test_position_risk_matches_the_budget(self):
        f = Features(symbol="X", price=100.0, source="test", atr=5.0)
        plan = plan_position(f, self.cfg)
        # Stop is 2 x ATR = 10 below entry, budget is 100, so 10 shares.
        self.assertEqual(plan.shares, 10)
        self.assertAlmostEqual(plan.stop_price, 90.0)
        self.assertAlmostEqual(plan.risk_amount, 100.0)

    def test_more_volatile_stock_gets_fewer_shares(self):
        calm = plan_position(Features("X", 100.0, "test", atr=2.0), self.cfg)
        wild = plan_position(Features("Y", 100.0, "test", atr=8.0), self.cfg)
        self.assertGreater(calm.shares, wild.shares)

    def test_position_cap_limits_quiet_stocks(self):
        f = Features(symbol="X", price=10.0, source="test", atr=0.05)
        plan = plan_position(f, self.cfg)
        self.assertTrue(plan.limited_by_cap)
        self.assertLessEqual(plan.pct_of_capital, self.cfg.max_position_pct + 1e-9)

    def test_reward_risk_matches_configured_target(self):
        plan = plan_position(Features("X", 100.0, "test", atr=5.0), self.cfg)
        self.assertAlmostEqual(plan.reward_risk, self.cfg.reward_risk_target)

    def test_missing_atr_yields_no_plan(self):
        self.assertIsNone(plan_position(Features("X", 100.0, "test"), self.cfg))

    def test_stop_below_zero_yields_no_plan(self):
        self.assertIsNone(plan_position(Features("X", 10.0, "test", atr=100.0), self.cfg))

    def test_gates_fire_independently(self):
        f = Features(
            symbol="X",
            price=10.0,
            source="test",
            annual_volatility=2.0,
            current_drawdown=0.9,
            avg_dollar_volume=1.0,
        )
        self.assertEqual(len(check_risk_gates(f, self.cfg)), 3)

    def test_gates_skip_missing_data(self):
        self.assertEqual(check_risk_gates(Features("X", 10.0, "test"), self.cfg), [])


class TestConfig(unittest.TestCase):
    def test_round_trips_through_dict(self):
        cfg = Config(capital=50_000.0, weights=Weights(trend=0.5, momentum=0.5))
        restored = Config.from_dict(cfg.to_dict())
        self.assertEqual(restored.capital, 50_000.0)
        self.assertAlmostEqual(restored.weights.trend, 0.5)

    def test_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"nonsense": 1})

    def test_rejects_out_of_order_thresholds(self):
        with self.assertRaises(ValueError):
            Config(buy_above=-0.5, reduce_below=0.5)

    def test_rejects_inverted_moving_averages(self):
        with self.assertRaises(ValueError):
            Config(sma_fast=200, sma_slow=50)

    def test_rejects_negative_weights(self):
        with self.assertRaises(ValueError):
            Weights(trend=-1.0)

    def test_rejects_all_zero_weights(self):
        with self.assertRaises(ValueError):
            Weights(
                trend=0, momentum=0, macd=0, rsi=0, risk_adjusted=0,
                near_high=0, volume=0, trend_quality=0, tv_rating=0,
            )


if __name__ == "__main__":
    unittest.main()
