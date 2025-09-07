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
			"P/E (TTM)": 12,
			"P/E (Fwd)": 12,
			"P/B": 10,
			"EV/EBITDA": 12,
			"ProfitMargin": 14,
			"RevenueGrowth": 16,
			"EPS (TTM)": 12,
			"Beta": 10,
			"DebtToEquity": 14,
			"CurrentRatio": 12,
		}
		for idx, col in enumerate(df.columns, start=1):
			ws.column_dimensions[ws.cell(row=1, column=idx).column_letter].width = widths.get(col, 12)

		# Number formats per column
		num_formats = {
			"Price": "0.00",
			"MarketCap": "#,##0",
			"P/E (TTM)": "0.00",
			"P/E (Fwd)": "0.00",
			"P/B": "0.00",
			"EV/EBITDA": "0.00",
			"ProfitMargin": "0.0%",
			"RevenueGrowth": "0.0%",
			"EPS (TTM)": "0.00",
			"Beta": "0.00",
			"DebtToEquity": "0.00",
			"CurrentRatio": "0.00",
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

