from types import SimpleNamespace
import os
import importlib
import sys


def test_fetch_metrics_smoke(monkeypatch):
	# Ensure comps_lib is importable when running from this test directory
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	if base_dir not in sys.path:
		sys.path.insert(0, base_dir)
	# Arrange: fake yfinance.Ticker
	class FakeTicker:
		def __init__(self, ticker):
			self.fast_info = {"lastPrice": 100.0}
			self.info = {
				"regularMarketPrice": 100.0,
				"marketCap": 2_000_000_000,
				"trailingPE": 25.5,
				"forwardPE": 22.0,
				"priceToBook": 7.5,
				"enterpriseToEbitda": 18.0,
				"profitMargins": 0.22,
				"revenueGrowth": 0.10,
				"trailingEps": 4.0,
				"beta": 1.1,
				"debtToEquity": 1.2,
				"currentRatio": 0.9,
			}

	# Build a fake yfinance module
	fake_mod = SimpleNamespace(Ticker=FakeTicker)
	monkeypatch.setitem(sys.modules, "yfinance", fake_mod)

	# Import the metrics module under test
	metrics = importlib.import_module("comps_lib.metrics")
	row = metrics.fetch_metrics_for_ticker("FAKE")

	assert row["Ticker"] == "FAKE"
	for key in [
		"Price",
		"MarketCap",
		"P/E (TTM)",
		"P/E (Fwd)",
		"P/B",
		"EV/EBITDA",
		"ProfitMargin",
		"RevenueGrowth",
		"EPS (TTM)",
		"Beta",
		"DebtToEquity",
		"CurrentRatio",
	]:
		assert key in row


def test_empty_on_missing_data(monkeypatch):
	class FakeTicker:
		def __init__(self, ticker):
			self.fast_info = {}
			self.info = {}

	fake_mod = SimpleNamespace(Ticker=FakeTicker)
	monkeypatch.setitem(sys.modules, "yfinance", fake_mod)

	metrics = importlib.import_module("comps_lib.metrics")
	row = metrics.fetch_metrics_for_ticker("EMPTY")

	# Either ERR_NO_DATA markers or None values are acceptable for this smoke test
	assert row["Ticker"] == "EMPTY"
	assert set(metrics.COLUMNS) == set(row.keys())

