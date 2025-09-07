import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List

import pandas as pd

from comps_lib.metrics import COLUMNS, fetch_metrics_for_ticker
from comps_lib.io import write_outputs


def _read_tickers_from_file(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		lines = [line.strip() for line in f.readlines()]
		return [ln for ln in lines if ln and not ln.startswith("#")]  # allow comments


def _unique_preserve_order(items: Iterable[str]) -> List[str]:
	seen = set()
	result: List[str] = []
	for it in items:
		u = it.strip()
		if not u or u in seen:
			continue
		seen.add(u)
		result.append(u)
	return result


def main() -> int:
	parser = argparse.ArgumentParser(description="Build comps table from Yahoo Finance.")
	parser.add_argument("--tickers", nargs="*", help="Ticker Symbole, z. B. AAPL MSFT GOOGL")
	parser.add_argument("--file", dest="file", help="Datei mit Tickern (eine pro Zeile)")
	parser.add_argument("--out", dest="out", default="comps.xlsx", help="Pfad der Excel-Ausgabedatei")
	parser.add_argument("--retry", dest="retry", type=int, default=0, help="Reserviert, hat aktuell keine Funktion")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	logger = logging.getLogger("comps")

	input_tickers: List[str] = []
	if args.tickers:
		input_tickers.extend(args.tickers)
	if args.file:
		input_tickers.extend(_read_tickers_from_file(args.file))

	tickers = _unique_preserve_order(input_tickers)
	if not tickers:
		logger.error("Keine Ticker angegeben. Verwende --tickers oder --file.")
		return 2

	logger.info("Lade Kennzahlen für %d Ticker ...", len(tickers))

	# Parallel fetch (I/O-bound)
	rows = []
	with ThreadPoolExecutor(max_workers=min(8, max(1, len(tickers)))) as pool:
		futures = {pool.submit(fetch_metrics_for_ticker, t): t for t in tickers}
		# Gather results in input order
		for t in tickers:
			for fut, _t in futures.items():
				if _t == t:
					rows.append(fut.result())
					break

	df = pd.DataFrame(rows, columns=COLUMNS)
	write_outputs(df, args.out)

	logger.info("Fertig. Ausgaben: %s und %s", args.out, args.out.replace(".xlsx", ".csv"))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

