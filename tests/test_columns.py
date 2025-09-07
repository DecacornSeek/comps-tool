import os
import sys


def test_columns_exact_order():
	# Ensure comps_lib is importable when running from this test directory
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	if base_dir not in sys.path:
		sys.path.insert(0, base_dir)

	from comps_lib.metrics import COLUMNS

	expected = [
		"Ticker",
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
	]
	assert COLUMNS == expected

