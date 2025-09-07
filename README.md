## Comps-Tool (Python CLI)

Kleines, robustes CLI-Tool, das für eine Liste von Ticker-Symbolen Kernkennzahlen aus Yahoo Finance (via `yfinance`) lädt und als `comps.xlsx` (formatiert) und `comps.csv` exportiert.

### Features
- **Quelle**: Yahoo Finance über `yfinance` (keine HTML-Scrapes)
- **Fehlertolerant**: Ungültige Ticker stoppen den Lauf nicht; die Zeile wird mit `ERR_*`-Werten markiert
- **Reproduzierbar**: `requirements.txt` mit Minimalversionen
- **Schnellstart**: CLI mit `--tickers` oder Dateiinput `--file`
- **Output**: CSV + hübsch formatiertes Excel

### Installation (≤ 60 Sekunden)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Nutzung
```bash
python comps.py --tickers AAPL MSFT GOOGL
# oder
python comps.py --file tickers.txt --out comps.xlsx
```

Erzeugt im Arbeitsverzeichnis `comps.xlsx` und `comps.csv`.

### Spalten (in fixer Reihenfolge)
- **Ticker**
- **Price**
- **MarketCap**
- **P/E (TTM)**
- **P/E (Fwd)**
- **P/B**
- **EV/EBITDA**
- **ProfitMargin**
- **RevenueGrowth**
- **EPS (TTM)**
- **Beta**
- **DebtToEquity** (falls verfügbar; sonst leer)
- **CurrentRatio** (falls verfügbar; sonst leer)

Fehlende Metriken bleiben leer. Bei Fehlern pro Ticker werden die Metriken mit `ERR_*` markiert und es wird ein Warn-Log ausgegeben.

### Tests
```bash
pytest -q
```
Die Tests mocken `yfinance`, laufen also auch ohne Internet und ohne installierte Abhängigkeiten.

### Hinweise
- Währung und Werte (z. B. Market Cap, Price) werden “as is” von Yahoo übernommen.
- Kein Multi-Currency-Handling im MVP.
- Optionales `--retry`-Flag ist vorbereitet, hat derzeit aber keine Funktion.
