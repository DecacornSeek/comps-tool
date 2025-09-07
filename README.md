# 📊 Advanced Stock Analysis Suite

Eine umfassende Lösung für Aktienanalysen mit erweiterten KPIs, Branchenvergleichen, interaktivem Dashboard und automatisierten Berichten.

## 🚀 Features

### 📈 Erweiterte Finanzkennzahlen (45+ KPIs)
- **Bewertung**: P/E, P/B, P/S, EV/EBITDA, EV/Revenue, PEG Ratio
- **Profitabilität**: ROE, ROA, ROIC, Profit/Operating/Gross Margins  
- **Wachstum**: Revenue Growth, Earnings Growth
- **Liquidität**: Current Ratio, Quick Ratio, Interest Coverage
- **Verschuldung**: Debt-to-Equity, Free Cash Flow, FCF Yield
- **Marktmetriken**: Beta, 52W Range, Dividend Yield, Market Cap
- **Eigentümerstruktur**: Insider/Institutional Ownership, Float Shares
- **Unternehmensdaten**: Sektor, Branche, Land, Mitarbeiteranzahl

### 🔍 Intelligente Analysen
- **Branchenvergleich**: Automatische Peer-Gruppierung nach Industry/Sektor
- **Perzentil-Rankings**: Relative Performance-Bewertung
- **Composite Scores**: Valuation, Quality, Growth Scores
- **Financial Health Score**: Kombinierter Gesundheitsindex
- **Korrelationsanalyse**: Zusammenhänge zwischen Metriken

### 📊 Interaktives Web-Dashboard  
- **Live-Datenanalyse**: Ticker-Input mit sofortiger Analyse
- **Erweiterte Charts**: Risk-Return Matrix, Value-Quality Frontier
- **Peer-Vergleiche**: Radar-Charts und Branchenvergleiche
- **Financial Health**: Visuelle Gesundheitsbewertungen
- **Responsive Design**: Bootstrap-basiertes Interface

### 📋 Automatisierte Reports
- **PDF-Generation**: Vollständige Analysereports mit Charts
- **Excel-Export**: Formatierte Tabellen mit Styling
- **JSON-Analytics**: Strukturierte Analysedaten
- **Ranking-Reports**: Top-Performer Listen

### 🌐 Multiple Interfaces
- **Command Line**: Flexible CLI für Batch-Processing
- **Web Dashboard**: Interaktive Analyse-Oberfläche  
- **Unified Suite**: Ein Tool für alle Funktionen
- **API-Ready**: Modular aufgebaut für Integration

## 🛠 Installation

```bash
# Repository klonen
git clone <repository-url>
cd webapp

# Dependencies installieren
pip install -r requirements.txt

# Sofort einsatzbereit!
```

## 📖 Verwendung

### 🔥 Quick Start - Unified Suite

```bash
# Kompletter Workflow: Daten holen + Analysieren + Report
python stock_analysis_suite.py workflow --tickers AAPL MSFT GOOGL AMZN --output my_analysis

# Nur Daten holen
python stock_analysis_suite.py fetch --tickers AAPL MSFT GOOGL --output stocks.xlsx

# Bestehende Daten analysieren
python stock_analysis_suite.py analyze --csv stocks.csv --output analysis

# PDF-Report generieren
python stock_analysis_suite.py report --csv stocks.csv --output report.pdf

# Peer-Analyse für spezifischen Ticker
python stock_analysis_suite.py peer --csv stocks.csv --ticker AAPL

# Web-Dashboard starten
python stock_analysis_suite.py dashboard --port 8050
```

### 💻 Einzelne Tools

```bash
# Originales Tool (erweiterte KPIs)
python comps.py --tickers AAPL MSFT GOOGL --out enhanced_analysis.xlsx

# Erweiterte Analyse bestehender Daten
python analyze.py --csv data.csv --output analysis --ticker AAPL

# Nur PDF-Report generieren
python generate_report.py --csv data.csv --output report.pdf

# Dashboard einzeln starten
python dashboard.py
```

### 🌐 Web-Dashboard

```bash
# Dashboard starten
python dashboard.py
# Oder via Suite:
python stock_analysis_suite.py dashboard --port 8050

# Dashboard URL: http://localhost:8050
```

**Dashboard Features:**
- 🎯 **Live Ticker Analysis**: Ticker eingeben → Sofortige Analyse
- 📊 **Multi-Chart Views**: Sektor, Bewertung, Performance, Korrelationen  
- 🔍 **Peer Comparisons**: Dropdown-Auswahl für detaillierte Vergleiche
- 📈 **Advanced Analytics**: Risk-Return, Value-Quality Matrix
- 📋 **Interactive Tables**: Sortierbar, filterbar, paginiert

## 📊 Erweiterte KPIs im Detail

| Kategorie | Metriken |
|-----------|----------|
| **Bewertung** | P/E (TTM/Fwd), P/B, P/S, EV/EBITDA, EV/Revenue, PEG Ratio |
| **Profitabilität** | ROE, ROA, ROIC, Profit Margin, Operating Margin, Gross Margin |
| **Wachstum** | Revenue Growth, Earnings Growth |  
| **Liquidität** | Current Ratio, Quick Ratio, Interest Coverage |
| **Verschuldung** | Debt-to-Equity, Free Cash Flow, FCF Yield |
| **Risiko** | Beta, 52W High/Low/Range %, Short Ratio |
| **Dividenden** | Dividend Yield, Payout Ratio |
| **Bewertung** | Book Value/Share, Tangible Book Value |
| **Markt** | Market Cap, Enterprise Value, Average Volume |
| **Ownership** | Insider %, Institutional %, Float Shares |
| **Stammdaten** | Sektor, Branche, Land, Mitarbeiterzahl |

## 🔍 Analysefunktionen

### Branchenvergleich
```python
# Automatische Peer-Gruppierung nach Industry
peer_data = calculate_peer_comparison(df, 'AAPL', 'Industry')

# Zeigt Percentile Rankings, vs-Median Performance
# Verfügbar für alle numerischen KPIs
```

### Composite Scores
- **Valuation Score**: Niedrigere P/E, P/B, P/S = höherer Score
- **Quality Score**: ROE, ROA, Profitability = höherer Score  
- **Growth Score**: Revenue/Earnings Growth = höherer Score
- **Health Score**: Liquidität, Verschuldung, Profitabilität

### Visualisierungen
- **Risk-Return Matrix**: Beta vs ROE mit Marktkapitalisierung
- **Value-Quality Frontier**: Valuation Score vs Quality Score
- **Financial Health Dashboard**: Kombinierte Gesundheitsbewertung
- **Correlation Heatmap**: Zusammenhänge zwischen Metriken
- **Peer Radar Charts**: Multi-dimensionale Vergleiche

## 📁 Output-Dateien

```
my_analysis_data.xlsx          # Rohdaten mit allen KPIs  
my_analysis_data.csv           # CSV-Version
my_analysis_enhanced.xlsx      # + Composite Scores + Percentiles
my_analysis_analysis.json      # Sektor/Industry Statistiken  
my_analysis_peer_comparisons.json  # Peer-Vergleichsdaten
my_analysis_rankings.xlsx      # Top-Performer Listen
my_analysis_report.pdf         # Vollständiger Report mit Charts
```

## 🎯 Use Cases

### Portfolio-Manager
```bash
# Monatliche Portfolio-Analyse
python stock_analysis_suite.py workflow --file portfolio_tickers.txt --output monthly_$(date +%Y%m)
```

### Research-Analysten  
```bash
# Branchenanalyse mit Peer-Vergleichen
python stock_analysis_suite.py fetch --tickers AAPL MSFT GOOGL AMZN META --output tech_giants.xlsx
python stock_analysis_suite.py peer --csv tech_giants.csv --ticker AAPL
```

### Investment-Teams
```bash
# Dashboard für Team-Meetings
python stock_analysis_suite.py dashboard --port 8050
# → Interaktive Analyse in Meetings
```

## 🔧 Technische Details

### Architektur
```
comps_lib/
├── metrics.py          # Kerndatensammlung (Yahoo Finance)
├── analysis.py         # Erweiterte Analysefunktionen
├── visualizations.py   # Plotly Charts & Grafiken
├── pdf_reports.py      # ReportLab PDF-Generation
└── io.py              # Excel/CSV Export mit Formatierung
```

### Abhängigkeiten
- **Daten**: yfinance, pandas, numpy
- **Visualisierung**: plotly, matplotlib, seaborn  
- **Web**: dash, dash-bootstrap-components, flask
- **Reports**: reportlab, kaleido
- **Export**: openpyxl

### Performance
- **Parallel Fetching**: ThreadPoolExecutor für mehrere Ticker
- **Caching**: Wiederverwendung von berechneten Metriken
- **Memory Efficient**: Stream-Processing für große Datasets
- **Error Resilient**: Fehlerhafte Ticker stoppen Pipeline nicht

## 🚀 Deployment

### Produktions-Setup
```bash
# PM2 für Dashboard (bereits konfiguriert)
pm2 start ecosystem.config.js

# Oder Gunicorn
gunicorn -w 4 -b 0.0.0.0:8050 dashboard:server
```

### Docker
```dockerfile
FROM python:3.12-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app
EXPOSE 8050
CMD ["python", "dashboard.py"]
```

## 📈 Roadmap

- [ ] **Real-time Data**: WebSocket-Integration für Live-Updates
- [ ] **More Data Sources**: Alpha Vantage, Polygon, Quandl Integration  
- [ ] **ML Features**: Anomalie-Detection, Clustering
- [ ] **API Endpoints**: REST API für externe Integration
- [ ] **User Management**: Multi-User Dashboard
- [ ] **Alerts**: E-Mail/Slack Notifications bei Schwellenwerten

## 🤝 Contributing

Das Tool ist modular aufgebaut - neue Features können einfach hinzugefügt werden:

1. **Neue KPIs**: Erweitern Sie `metrics.py` 
2. **Neue Analysen**: Fügen Sie Funktionen in `analysis.py` hinzu
3. **Neue Charts**: Erstellen Sie Visualisierungen in `visualizations.py`
4. **Dashboard Features**: Erweitern Sie `dashboard.py`

## 🔍 Troubleshooting

### Häufige Probleme
- **Kaleido Errors**: `pip install plotly==5.17.0 kaleido==0.2.1`
- **yfinance Issues**: Ticker-Symbole überprüfen, Yahoo Finance Status
- **Memory Issues**: Weniger Ticker parallel verarbeiten
- **Dashboard Port**: `--port` Parameter ändern bei Konflikten

### Support
- Logs prüfen: Dashboard logs unter `logs/`
- Verbose Mode: `--debug` Flag für detaillierte Ausgabe
- Test mit einzelnen Tickern bei Problemen

---

**🎉 Happy Analyzing! Erstellen Sie professionelle Aktienanalysen in Minuten statt Stunden.**