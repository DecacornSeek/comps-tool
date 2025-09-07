import logging
from typing import Any, Dict, List, Optional

import yfinance as yf


# Feste Spaltenreihenfolge gemäß Spezifikation
COLUMNS: List[str] = [
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

		row.update(
			{
				"Price": _coerce_float(price),
				"MarketCap": _coerce_float(_safe_get(info, "marketCap")),
				"P/E (TTM)": _coerce_float(_safe_get(info, "trailingPE")),
				"P/E (Fwd)": _coerce_float(_safe_get(info, "forwardPE")),
				"P/B": _coerce_float(_safe_get(info, "priceToBook")),
				"EV/EBITDA": _coerce_float(_safe_get(info, "enterpriseToEbitda")),
				"ProfitMargin": _coerce_float(_safe_get(info, "profitMargins")),
				"RevenueGrowth": _coerce_float(_safe_get(info, "revenueGrowth")),
				"EPS (TTM)": _coerce_float(_safe_get(info, "trailingEps")),
				"Beta": _coerce_float(_safe_get(info, "beta")),
				"DebtToEquity": _coerce_float(_safe_get(info, "debtToEquity")),
				"CurrentRatio": _coerce_float(_safe_get(info, "currentRatio")),
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

