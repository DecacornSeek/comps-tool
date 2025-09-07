import logging
from typing import Any, Dict, List, Optional

import yfinance as yf


# Erweiterte Spaltenreihenfolge mit zusätzlichen KPIs
COLUMNS: List[str] = [
	"Ticker",
	"Price",
	"MarketCap",
	"Enterprise Value",
	"P/E (TTM)",
	"P/E (Fwd)",
	"P/B",
	"P/S",
	"EV/EBITDA",
	"EV/Revenue",
	"ProfitMargin",
	"OperatingMargin",
	"GrossMargin",
	"RevenueGrowth",
	"EarningsGrowth",
	"EPS (TTM)",
	"EPS (Fwd)",
	"ROE",
	"ROA",
	"ROIC",
	"Free Cash Flow",
	"FCF Yield",
	"Beta",
	"DebtToEquity",
	"CurrentRatio",
	"QuickRatio",
	"InterestCoverage",
	"DividendYield",
	"PayoutRatio",
	"BookValuePerShare",
	"TangibleBookValue",
	"52W High",
	"52W Low",
	"52W Range %",
	"Average Volume",
	"Float Shares",
	"Shares Outstanding",
	"Insider Ownership",
	"Institutional Ownership",
	"Short Ratio",
	"PEG Ratio",
	"Industry",
	"Sector",
	"Country",
	"Employees",
]


def _safe_get(mapping: Optional[dict], key: str) -> Optional[Any]:
	"""Return mapping[key] if available and non-None; else None."""
	if not mapping:
		return None
	value = mapping.get(key)
	return value if value is not None else None


def _first_of(mapping: Optional[dict], keys: List[str]) -> Optional[Any]:
	"""Return first non-None value for any of the keys in mapping."""
	if not mapping:
		return None
	for key in keys:
		if key in mapping and mapping[key] is not None:
			return mapping[key]
	return None


def _coerce_float(value: Any) -> Optional[float]:
	"""Try to convert input to float; return None if not possible."""
	if value is None:
		return None
	try:
		return float(value)
	except Exception:
		return None


def _empty_row(ticker: str) -> Dict[str, Any]:
	return {col: (ticker if col == "Ticker" else None) for col in COLUMNS}


def fetch_metrics_for_ticker(ticker: str) -> Dict[str, Any]:
	"""Fetch core metrics for one ticker via yfinance.

	On error, returns a row with all metrics marked as 'ERR_FETCH'.
	If no data is available but no exception occurs, marks metrics as 'ERR_NO_DATA'.
	"""
	logger = logging.getLogger(__name__)
	row: Dict[str, Any] = _empty_row(ticker)

	try:
		yt = yf.Ticker(ticker)
		# yfinance provides both fast_info and info.
		fi = getattr(yt, "fast_info", None)
		fi = dict(fi) if fi is not None else None
		info = getattr(yt, "info", None)
		info = dict(info) if info is not None else None

		# Price: prefer fast_info
		price = _first_of(
			fi,
			[
				"lastPrice",
				"last_price",
				"regularMarketPrice",
				"last_trade_price",
			],
		)
		if price is None:
			price = _safe_get(info, "regularMarketPrice")

		# Berechne abgeleitete Metriken
		market_cap = _coerce_float(_safe_get(info, "marketCap"))
		enterprise_value = _coerce_float(_safe_get(info, "enterpriseValue"))
		total_revenue = _coerce_float(_safe_get(info, "totalRevenue"))
		free_cash_flow = _coerce_float(_safe_get(info, "freeCashflow"))
		shares_outstanding = _coerce_float(_safe_get(info, "sharesOutstanding"))
		fifty_two_week_high = _coerce_float(_safe_get(info, "fiftyTwoWeekHigh"))
		fifty_two_week_low = _coerce_float(_safe_get(info, "fiftyTwoWeekLow"))
		
		# FCF Yield berechnen
		fcf_yield = None
		if free_cash_flow and market_cap:
			fcf_yield = free_cash_flow / market_cap
			
		# 52W Range % berechnen
		range_percent = None
		if fifty_two_week_high and fifty_two_week_low and price:
			range_percent = (price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low)
		
		# EV/Revenue berechnen
		ev_revenue = None
		if enterprise_value and total_revenue:
			ev_revenue = enterprise_value / total_revenue

		row.update(
			{
				"Price": _coerce_float(price),
				"MarketCap": market_cap,
				"Enterprise Value": enterprise_value,
				"P/E (TTM)": _coerce_float(_safe_get(info, "trailingPE")),
				"P/E (Fwd)": _coerce_float(_safe_get(info, "forwardPE")),
				"P/B": _coerce_float(_safe_get(info, "priceToBook")),
				"P/S": _coerce_float(_safe_get(info, "priceToSalesTrailing12Months")),
				"EV/EBITDA": _coerce_float(_safe_get(info, "enterpriseToEbitda")),
				"EV/Revenue": ev_revenue,
				"ProfitMargin": _coerce_float(_safe_get(info, "profitMargins")),
				"OperatingMargin": _coerce_float(_safe_get(info, "operatingMargins")),
				"GrossMargin": _coerce_float(_safe_get(info, "grossMargins")),
				"RevenueGrowth": _coerce_float(_safe_get(info, "revenueGrowth")),
				"EarningsGrowth": _coerce_float(_safe_get(info, "earningsGrowth")),
				"EPS (TTM)": _coerce_float(_safe_get(info, "trailingEps")),
				"EPS (Fwd)": _coerce_float(_safe_get(info, "forwardEps")),
				"ROE": _coerce_float(_safe_get(info, "returnOnEquity")),
				"ROA": _coerce_float(_safe_get(info, "returnOnAssets")),
				"ROIC": _coerce_float(_safe_get(info, "returnOnInvestedCapital")),
				"Free Cash Flow": free_cash_flow,
				"FCF Yield": fcf_yield,
				"Beta": _coerce_float(_safe_get(info, "beta")),
				"DebtToEquity": _coerce_float(_safe_get(info, "debtToEquity")),
				"CurrentRatio": _coerce_float(_safe_get(info, "currentRatio")),
				"QuickRatio": _coerce_float(_safe_get(info, "quickRatio")),
				"InterestCoverage": _coerce_float(_safe_get(info, "interestCoverage")),
				"DividendYield": _coerce_float(_safe_get(info, "dividendYield")),
				"PayoutRatio": _coerce_float(_safe_get(info, "payoutRatio")),
				"BookValuePerShare": _coerce_float(_safe_get(info, "bookValue")),
				"TangibleBookValue": _coerce_float(_safe_get(info, "tangibleBookValue")),
				"52W High": fifty_two_week_high,
				"52W Low": fifty_two_week_low,
				"52W Range %": range_percent,
				"Average Volume": _coerce_float(_safe_get(info, "averageVolume")),
				"Float Shares": _coerce_float(_safe_get(info, "floatShares")),
				"Shares Outstanding": shares_outstanding,
				"Insider Ownership": _coerce_float(_safe_get(info, "heldPercentInsiders")),
				"Institutional Ownership": _coerce_float(_safe_get(info, "heldPercentInstitutions")),
				"Short Ratio": _coerce_float(_safe_get(info, "shortRatio")),
				"PEG Ratio": _coerce_float(_safe_get(info, "pegRatio")),
				"Industry": _safe_get(info, "industry"),
				"Sector": _safe_get(info, "sector"),
				"Country": _safe_get(info, "country"),
				"Employees": _coerce_float(_safe_get(info, "fullTimeEmployees")),
			}
		)

		# If we got literally nothing useful, mark as no data
		metrics_only = [row[c] for c in COLUMNS if c != "Ticker"]
		if all(v is None for v in metrics_only):
			for c in COLUMNS:
				if c == "Ticker":
					continue
				row[c] = "ERR_NO_DATA"
			logger.warning("No data for ticker %s", ticker)

	except Exception as exc:  # noqa: BLE001 — we want to catch and continue per requirements
		logger.warning("Failed to fetch metrics for %s: %s", ticker, exc)
		for c in COLUMNS:
			if c == "Ticker":
				continue
			row[c] = "ERR_FETCH"

	return row

