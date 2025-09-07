from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def write_outputs(df: pd.DataFrame, excel_path: str) -> Dict[str, str]:
	"""Write DataFrame to CSV and Excel with basic formatting.

	Returns a dict with paths for {"excel": path, "csv": path}.
	"""
	# CSV
	csv_path = os.path.splitext(excel_path)[0] + ".csv"
	df.to_csv(csv_path, index=False)

	# Excel with formatting
	with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
		df.to_excel(writer, index=False, sheet_name="Comps")
		ws = writer.sheets["Comps"]

		# Column widths
		widths = {
			"Ticker": 12,
			"Price": 12,
			"MarketCap": 16,
			"Enterprise Value": 16,
			"P/E (TTM)": 12,
			"P/E (Fwd)": 12,
			"P/B": 10,
			"P/S": 10,
			"EV/EBITDA": 12,
			"EV/Revenue": 12,
			"ProfitMargin": 14,
			"OperatingMargin": 16,
			"GrossMargin": 14,
			"RevenueGrowth": 16,
			"EarningsGrowth": 16,
			"EPS (TTM)": 12,
			"EPS (Fwd)": 12,
			"ROE": 10,
			"ROA": 10,
			"ROIC": 10,
			"Free Cash Flow": 16,
			"FCF Yield": 12,
			"Beta": 10,
			"DebtToEquity": 14,
			"CurrentRatio": 12,
			"QuickRatio": 12,
			"InterestCoverage": 16,
			"DividendYield": 14,
			"PayoutRatio": 12,
			"BookValuePerShare": 18,
			"TangibleBookValue": 18,
			"52W High": 12,
			"52W Low": 12,
			"52W Range %": 14,
			"Average Volume": 16,
			"Float Shares": 16,
			"Shares Outstanding": 18,
			"Insider Ownership": 18,
			"Institutional Ownership": 20,
			"Short Ratio": 12,
			"PEG Ratio": 12,
			"Industry": 20,
			"Sector": 16,
			"Country": 12,
			"Employees": 12,
		}
		for idx, col in enumerate(df.columns, start=1):
			ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = widths.get(col, 12)

		# Number formats per column
		num_formats = {
			"Price": "0.00",
			"MarketCap": "#,##0",
			"Enterprise Value": "#,##0",
			"P/E (TTM)": "0.00",
			"P/E (Fwd)": "0.00",
			"P/B": "0.00",
			"P/S": "0.00",
			"EV/EBITDA": "0.00",
			"EV/Revenue": "0.00",
			"ProfitMargin": "0.0%",
			"OperatingMargin": "0.0%",
			"GrossMargin": "0.0%",
			"RevenueGrowth": "0.0%",
			"EarningsGrowth": "0.0%",
			"EPS (TTM)": "0.00",
			"EPS (Fwd)": "0.00",
			"ROE": "0.0%",
			"ROA": "0.0%",
			"ROIC": "0.0%",
			"Free Cash Flow": "#,##0",
			"FCF Yield": "0.0%",
			"Beta": "0.00",
			"DebtToEquity": "0.00",
			"CurrentRatio": "0.00",
			"QuickRatio": "0.00",
			"InterestCoverage": "0.00",
			"DividendYield": "0.0%",
			"PayoutRatio": "0.0%",
			"BookValuePerShare": "0.00",
			"TangibleBookValue": "#,##0",
			"52W High": "0.00",
			"52W Low": "0.00",
			"52W Range %": "0.0%",
			"Average Volume": "#,##0",
			"Float Shares": "#,##0",
			"Shares Outstanding": "#,##0",
			"Insider Ownership": "0.0%",
			"Institutional Ownership": "0.0%",
			"Short Ratio": "0.00",
			"PEG Ratio": "0.00",
			"Employees": "#,##0",
		}
		max_row = ws.max_row
		max_col = ws.max_column
		for j in range(1, max_col + 1):
			col_name = df.columns[j - 1]
			fmt = num_formats.get(col_name)
			if not fmt:
				continue
			for i in range(2, max_row + 1):  # skip header
				cell = ws.cell(row=i, column=j)
				cell.number_format = fmt

	return {"excel": excel_path, "csv": csv_path}

