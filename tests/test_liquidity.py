import unittest
import os

import pandas as pd

from sources.analytics import compare_liquidity_with_asset
from sources.global_liquidity import (
    build_custom_gli, build_global_liquidity_index, build_gli_trends,
    build_tradingview_view,
)
from sources.liquidity import build_us_net_liquidity_history, calculate_us_net_liquidity
from sources.market_regime import calculate_market_regime, classify_market_regime
from sources.macro_credit_risk import calculate_macro_credit_risk, classify_macro_risk
from sources.section_reports import build_section_report
from sources.signals import build_markdown_report, compare_gli_with_asset, interpret_liquidity
from sources.policy_rates import classify_policy, get_china_lpr_history, rate_change
from sources.sentiment import (
    classify_fear_greed, parse_cnn_fear_greed, parse_market_fear_greed,
)
from utils.network import remove_dead_local_proxy
from utils.persistence import mark_series


class LiquidityTests(unittest.TestCase):
    def test_section_reports_generate_three_scenarios_totalling_100(self):
        context = {
            "market_score": 70, "market_regime": "Risk-on moderado",
            "macro_score": 30, "macro_regime": "Riesgo moderado",
            "gli_change4": 1.2, "inflation": 2.8, "fed_rate_change": -0.25,
            "vix": 17.0, "sp500_above_ema200": True, "gli_above_ema200": True,
            "gli_latest": 70.0, "dxy_change12": -1.5, "fed_rate": 4.0,
            "yield_curve": 0.4, "sentiment": 62, "hy_spread": 3.1,
            "nfci": -0.4, "unemployment": 4.1,
        }
        sections = [
            "Resumen", "Liquidez global", "Política monetaria", "Mercados",
            "Riesgo macro y crédito",
        ]
        for section in sections:
            report = build_section_report(section, context)
            self.assertEqual(len(report["scenarios"]), 3)
            self.assertEqual(sum(row["Probabilidad"] for row in report["scenarios"]), 100)
            self.assertIn("# Informe", report["markdown"])

    def test_macro_credit_risk_builds_history_and_low_risk_reading(self):
        dates = pd.date_range("2022-01-01", periods=48, freq="MS")
        growth = pd.Series([100 * (1.002 ** i) for i in range(48)], index=dates)
        production = pd.Series([100 * (1.003 ** i) for i in range(48)], index=dates)
        constant = lambda value: pd.Series([value] * 48, index=dates, dtype=float)
        result = calculate_macro_credit_risk({
            "Curva 10Y–2Y": constant(1.2),
            "Spread high yield": constant(2.5),
            "NFCI": constant(-0.6),
            "IPC": growth,
            "Desempleo": pd.Series([4.5 - i * 0.01 for i in range(48)], index=dates),
            "Producción industrial": production,
        })
        self.assertLess(result["score"], 25)
        self.assertEqual(result["classification"], "Riesgo bajo")
        self.assertEqual(len(result["details"]), 6)
        self.assertFalse(result["history"].empty)
        self.assertEqual(classify_macro_risk(85), "Riesgo extremo")

    def test_market_regime_detects_risk_on_environment(self):
        dates = pd.date_range("2022-01-07", periods=210, freq="W-FRI")
        rising = pd.Series(range(100, 310), index=dates, dtype=float)
        falling_dxy = pd.Series(range(310, 100, -1), index=dates, dtype=float)
        result = calculate_market_regime(
            rising, rising * 20, pd.Series([14.0] * 210, index=dates),
            falling_dxy, pd.Series([5.0] * 190 + [4.5] * 20, index=dates),
            pd.Series([70.0] * 210, index=dates),
        )
        self.assertGreaterEqual(result["score"], 75)
        self.assertEqual(result["regime"], "Risk-on fuerte")
        self.assertEqual(len(result["details"]), 6)
        self.assertEqual(classify_market_regime(20), "Crisis de liquidez")

    def test_current_calculation_and_units(self):
        result = calculate_us_net_liquidity(
            {"value": 8_000_000, "previous": 7_900_000, "error": None},
            {"value": 500_000, "previous": 450_000, "error": None},
            {"value": 100, "previous": 120, "error": None},
        )
        self.assertAlmostEqual(result["value"], 7400)
        self.assertAlmostEqual(result["change"], 70)

    def test_history_aligns_different_publication_dates(self):
        fed = pd.Series([8_000_000, 8_100_000], index=pd.to_datetime(["2026-01-07", "2026-01-14"]))
        tga = pd.Series([500_000, 600_000], index=pd.to_datetime(["2026-01-05", "2026-01-12"]))
        rrp = pd.Series([100, 90], index=pd.to_datetime(["2026-01-02", "2026-01-09"]))
        result = build_us_net_liquidity_history(fed, tga, rrp)
        self.assertTrue(all(result.index.weekday == 4))
        self.assertAlmostEqual(result.iloc[-1]["US Net Liquidity"], 7410)

    def test_source_error_is_propagated(self):
        result = calculate_us_net_liquidity(
            {"error": "fallo"}, {"error": None}, {"error": None}
        )
        self.assertIn("Balance FED", result["error"])

    def test_comparison_is_base_100_and_has_emas(self):
        dates = pd.date_range("2020-01-03", periods=210, freq="W-FRI")
        liquidity = pd.Series(range(100, 310), index=dates, dtype=float)
        asset = pd.Series(range(200, 410), index=dates, dtype=float)
        normalized, trends = compare_liquidity_with_asset(liquidity, asset)
        self.assertTrue((normalized.iloc[0] == 100).all())
        self.assertEqual(
            list(trends.columns),
            ["Activo", "EMA 10", "EMA 20", "EMA 50", "EMA 200"],
        )
        self.assertGreater(trends.iloc[-1]["Activo"], trends.iloc[-1]["EMA 200"])

    def test_global_liquidity_converts_currencies_to_trillions(self):
        dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
        constant = lambda value: pd.Series([value] * 3, index=dates, dtype=float)
        result = build_global_liquidity_index(
            constant(8_000_000),  # 8 T USD
            constant(7_000_000),  # 7 T EUR
            constant(7_500_000),  # 750 T JPY
            constant(1.2),
            constant(150),
        )
        self.assertAlmostEqual(result.iloc[-1]["FED"], 8.0)
        self.assertAlmostEqual(result.iloc[-1]["BCE"], 8.4)
        self.assertAlmostEqual(result.iloc[-1]["BoJ"], 5.0)
        self.assertAlmostEqual(result.iloc[-1]["GLI"], 21.4)

    def test_gli_trends_include_requested_emas(self):
        dates = pd.date_range("2020-01-03", periods=210, freq="W-FRI")
        trends = build_gli_trends(pd.Series(range(100, 310), index=dates, dtype=float))
        self.assertEqual(
            list(trends.columns),
            ["GLI", "EMA 10", "EMA 20", "EMA 50", "EMA 200"],
        )

    def test_tradingview_view_supports_change_smoothing_and_offset(self):
        dates = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
        gli = pd.Series(range(100, 160), index=dates, dtype=float)
        raw = build_tradingview_view(gli, "Variación mensual", 1, 0)
        shifted = build_tradingview_view(gli, "Variación mensual", 4, 60)
        self.assertAlmostEqual(raw.iloc[0], 4.0)
        self.assertEqual(shifted.index[0], raw.index[0] + pd.Timedelta(days=60))
        self.assertIn("mensual", shifted.name)

    def test_custom_gli_adds_and_subtracts_selected_components(self):
        dates = pd.date_range("2026-01-02", periods=2, freq="W-FRI")
        history = pd.DataFrame({
            "FED": [8.0, 8.1], "TGA": [0.5, 0.6],
            "Reverse Repo": [0.1, 0.1], "BCE": [7.0, 7.1],
        }, index=dates)
        result = build_custom_gli(
            history, ["Balance FED", "Cuenta del Tesoro (TGA)", "Reverse Repo", "Balance BCE"]
        )
        self.assertAlmostEqual(result.iloc[-1], 14.5)

    def test_global_liquidity_adds_china_proxy_separately(self):
        dates = pd.date_range("2025-01-03", periods=4, freq="W-FRI")
        constant = lambda value: pd.Series([value] * 4, index=dates, dtype=float)
        result = build_global_liquidity_index(
            constant(8_000_000), constant(7_000_000), constant(7_500_000),
            constant(1.2), constant(150), constant(3_500_000), constant(7),
        )
        self.assertAlmostEqual(result.iloc[-1]["China M2"], 50.0)
        self.assertAlmostEqual(result.iloc[-1]["GLI Bancos Centrales"], 21.4)
        self.assertAlmostEqual(result.iloc[-1]["GLI"], 71.4)

    def test_gli_asset_comparison_lags_and_interprets(self):
        dates = pd.date_range("2020-01-03", periods=210, freq="W-FRI")
        gli = pd.Series(range(100, 310), index=dates, dtype=float)
        asset = pd.Series(range(200, 410), index=dates, dtype=float)
        comparison, correlation = compare_gli_with_asset(gli, asset, 4)
        self.assertTrue((comparison.iloc[0] == 100).all())
        self.assertGreater(correlation, 0.9)
        analysis = interpret_liquidity(gli, asset)
        self.assertEqual(analysis["regime"], "Expansivo")

    def test_report_contains_methodology(self):
        dates = pd.date_range("2026-01-02", periods=2, freq="W-FRI")
        frame = pd.DataFrame({
            "GLI": [70.0, 71.0], "GLI Bancos Centrales": [21.0, 21.2],
            "China M2": [49.0, 49.8],
        }, index=dates)
        report = build_markdown_report(
            frame, {"regime": "Neutral", "messages": ["Lectura de prueba."]}, "Bitcoin"
        )
        self.assertIn("Metodología", report)
        self.assertIn("Bitcoin", report)

    def test_only_dead_local_proxy_is_removed(self):
        previous_http = os.environ.get("HTTP_PROXY")
        previous_https = os.environ.get("HTTPS_PROXY")
        previous_lower = os.environ.get("http_proxy")
        try:
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
            os.environ["http_proxy"] = "http://127.0.0.1:9"
            os.environ["HTTPS_PROXY"] = "http://proxy.empresa:8080"
            removed = remove_dead_local_proxy()
            self.assertIn("HTTP_PROXY", removed)
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("http_proxy", os.environ)
            self.assertEqual(os.environ["HTTPS_PROXY"], "http://proxy.empresa:8080")
        finally:
            if previous_http is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = previous_http
            if previous_https is None:
                os.environ.pop("HTTPS_PROXY", None)
            else:
                os.environ["HTTPS_PROXY"] = previous_https
            if previous_lower is None:
                os.environ.pop("http_proxy", None)
            else:
                os.environ["http_proxy"] = previous_lower

    def test_series_status_metadata(self):
        dates = pd.date_range("2026-01-02", periods=3, freq="W-FRI")
        original = pd.Series([1.0, 2.0, 3.0], index=dates)
        marked = mark_series(original, "Proveedor", "cache", "fallo simulado")
        self.assertEqual(marked.attrs["data_status"]["source"], "cache")
        self.assertEqual(marked.attrs["data_status"]["provider"], "Proveedor")

    def test_policy_rate_changes_and_classification(self):
        dates = pd.date_range("2025-01-03", periods=60, freq="W-FRI")
        rising = pd.Series([1 + i / 100 for i in range(60)], index=dates)
        self.assertGreater(rate_change(rising, 6), 0)
        self.assertEqual(classify_policy(rising), "Endureciendo")

    def test_china_lpr_latest_values(self):
        latest = get_china_lpr_history().iloc[-1]
        self.assertAlmostEqual(latest["China LPR 1 año"], 3.0)
        self.assertAlmostEqual(latest["China LPR 5 años"], 3.5)

    def test_fear_greed_classification(self):
        self.assertEqual(classify_fear_greed(10), "Miedo extremo")
        self.assertEqual(classify_fear_greed(50), "Neutral")
        self.assertEqual(classify_fear_greed(90), "Codicia extrema")

    def test_cnn_fear_greed_parser(self):
        payload = {
            "fear_and_greed_historical": {"data": [{"x": 1_767_225_600_000, "y": 42}]},
            "fear_and_greed": {"score": 55, "timestamp": "2026-01-02T12:00:00Z"},
        }
        series = parse_cnn_fear_greed(payload)
        self.assertEqual(series.name, "S&P 500 Fear & Greed")
        self.assertEqual(series.iloc[-1], 55)

    def test_market_fear_greed_fallback_parser(self):
        series = parse_market_fear_greed({
            "score": {"score": 63},
            "recent": [{"date": "2026-07-14", "score": 60},
                       {"date": "2026-07-15", "score": 62}],
        })
        self.assertEqual(len(series), 2)
        self.assertEqual(series.iloc[-1], 62)


if __name__ == "__main__":
    unittest.main()
