"""
Extended analysis script with peer comparison and sector analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from comps_lib.analysis import (
    calculate_sector_stats,
    calculate_industry_stats,
    add_percentile_rankings,
    calculate_peer_comparison,
    calculate_valuation_metrics,
    generate_summary_stats
)
from comps_lib.io import write_outputs


def _read_tickers_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
        return [ln for ln in lines if ln and not ln.startswith("#")]


def analyze_existing_data(csv_path: str, output_prefix: str = "analysis") -> None:
    """Analysiert bestehende CSV-Daten und erstellt erweiterte Berichte."""
    logger = logging.getLogger("analyzer")
    
    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Lade Daten
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} companies from {csv_path}")
    
    # Analysiere Daten
    logger.info("Calculating valuation metrics and scores...")
    enhanced_df = calculate_valuation_metrics(df)
    
    # Füge Perzentil-Rankings hinzu
    key_metrics = [
        'P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA', 'ROE', 'ROA', 'ROIC',
        'ProfitMargin', 'OperatingMargin', 'RevenueGrowth', 'FCF Yield', 'Beta'
    ]
    enhanced_df = add_percentile_rankings(enhanced_df, key_metrics)
    
    # Speichere erweiterte Daten
    enhanced_output = f"{output_prefix}_enhanced.xlsx"
    write_outputs(enhanced_df, enhanced_output)
    logger.info(f"Enhanced data saved to {enhanced_output}")
    
    # Erstelle Branchenanalyse
    logger.info("Calculating sector and industry statistics...")
    sector_stats = calculate_sector_stats(df, key_metrics)
    industry_stats = calculate_industry_stats(df, key_metrics)
    
    # Erstelle Summary
    summary = generate_summary_stats(df)
    
    # Speichere Analyseergebnisse als JSON
    analysis_results = {
        'summary': summary,
        'sector_stats': sector_stats,
        'industry_stats': industry_stats,
        'metadata': {
            'total_metrics': len(key_metrics),
            'analysis_date': pd.Timestamp.now().isoformat(),
            'source_file': csv_path
        }
    }
    
    analysis_json_path = f"{output_prefix}_analysis.json"
    with open(analysis_json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Analysis results saved to {analysis_json_path}")
    
    # Peer-Vergleich für jeden Ticker
    peer_comparisons = {}
    for ticker in df['Ticker']:
        try:
            peer_data = calculate_peer_comparison(df, ticker, 'Industry')
            if peer_data and 'metrics' in peer_data:
                peer_comparisons[ticker] = peer_data
        except Exception as e:
            logger.warning(f"Failed to calculate peer comparison for {ticker}: {e}")
    
    if peer_comparisons:
        peer_json_path = f"{output_prefix}_peer_comparisons.json"
        with open(peer_json_path, 'w', encoding='utf-8') as f:
            json.dump(peer_comparisons, f, indent=2, ensure_ascii=False)
        logger.info(f"Peer comparisons saved to {peer_json_path}")
    
    # Erstelle Ranking-Report
    create_ranking_report(enhanced_df, f"{output_prefix}_rankings.xlsx")
    
    logger.info("Analysis completed successfully!")


def create_ranking_report(df: pd.DataFrame, output_path: str) -> None:
    """Erstellt einen Ranking-Report mit Top-Performern in verschiedenen Kategorien."""
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Top Valuation Scores
        if 'Valuation_Score' in df.columns:
            valuation_top = df.nlargest(10, 'Valuation_Score')[
                ['Ticker', 'Sector', 'Industry', 'Valuation_Score', 'P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA']
            ].round(2)
            valuation_top.to_excel(writer, sheet_name='Top_Valuation', index=False)
        
        # Top Quality Scores
        if 'Quality_Score' in df.columns:
            quality_top = df.nlargest(10, 'Quality_Score')[
                ['Ticker', 'Sector', 'Industry', 'Quality_Score', 'ROE', 'ROA', 'ROIC', 'ProfitMargin']
            ].round(2)
            quality_top.to_excel(writer, sheet_name='Top_Quality', index=False)
        
        # Top Growth Scores
        if 'Growth_Score' in df.columns:
            growth_top = df.nlargest(10, 'Growth_Score')[
                ['Ticker', 'Sector', 'Industry', 'Growth_Score', 'RevenueGrowth', 'EarningsGrowth']
            ].round(2)
            growth_top.to_excel(writer, sheet_name='Top_Growth', index=False)
        
        # Largest Market Caps
        if 'MarketCap' in df.columns:
            largest_caps = df.nlargest(10, 'MarketCap')[
                ['Ticker', 'Sector', 'Industry', 'MarketCap', 'Price', 'P/E (TTM)', 'ROE']
            ].round(2)
            largest_caps.to_excel(writer, sheet_name='Largest_Companies', index=False)
        
        # High Dividend Yields
        if 'DividendYield' in df.columns:
            high_dividends = df.nlargest(10, 'DividendYield')[
                ['Ticker', 'Sector', 'Industry', 'DividendYield', 'PayoutRatio', 'P/E (TTM)', 'DebtToEquity']
            ].round(2)
            high_dividends.to_excel(writer, sheet_name='High_Dividends', index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze stock data with peer comparison and rankings")
    parser.add_argument("--csv", dest="csv_path", required=True, help="Path to CSV file with stock data")
    parser.add_argument("--output", dest="output_prefix", default="analysis", help="Output file prefix")
    parser.add_argument("--ticker", dest="ticker", help="Specific ticker for peer comparison")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("analyzer")
    
    try:
        # Führe Hauptanalyse durch
        analyze_existing_data(args.csv_path, args.output_prefix)
        
        # Spezifische Ticker-Analyse falls gewünscht
        if args.ticker:
            df = pd.read_csv(args.csv_path)
            peer_data = calculate_peer_comparison(df, args.ticker.upper(), 'Industry')
            
            if peer_data and 'metrics' in peer_data:
                logger.info(f"Peer analysis for {args.ticker}:")
                logger.info(f"  Industry: {peer_data['comparison_value']}")
                logger.info(f"  Peers found: {peer_data['peer_count']}")
                
                # Zeige Top-Metriken
                for metric in ['P/E (TTM)', 'ROE', 'ProfitMargin', 'RevenueGrowth']:
                    if metric in peer_data['metrics']:
                        data = peer_data['metrics'][metric]
                        logger.info(f"  {metric}: {data['ticker_value']:.2f} "
                                   f"(vs peer median: {data['peer_median']:.2f}, "
                                   f"percentile: {data['percentile_rank']:.0f}%)")
            else:
                logger.warning(f"No peer data found for {args.ticker}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())