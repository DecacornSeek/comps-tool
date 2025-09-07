"""
Complete Stock Analysis Suite - Unified Command Line Interface
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Import all modules
from comps_lib.metrics import fetch_metrics_for_ticker, COLUMNS
from comps_lib.analysis import (
    calculate_peer_comparison,
    calculate_valuation_metrics, 
    generate_summary_stats,
    calculate_sector_stats,
    calculate_industry_stats,
    add_percentile_rankings
)
from comps_lib.io import write_outputs
from comps_lib.pdf_reports import generate_pdf_report


def fetch_stock_data(tickers: List[str], output_file: str = "stock_data.xlsx") -> str:
    """Fetch stock data for given tickers."""
    logger = logging.getLogger("stock_analyzer")
    logger.info(f"Fetching data for {len(tickers)} tickers...")
    
    rows = []
    for ticker in tickers:
        try:
            row = fetch_metrics_for_ticker(ticker)
            rows.append(row)
            logger.info(f"✓ Fetched data for {ticker}")
        except Exception as e:
            logger.error(f"✗ Failed to fetch {ticker}: {e}")
            # Add error row
            error_row = {col: (ticker if col == "Ticker" else "ERROR") for col in COLUMNS}
            rows.append(error_row)
    
    df = pd.DataFrame(rows, columns=COLUMNS)
    
    # Save data
    paths = write_outputs(df, output_file)
    logger.info(f"Data saved to: {paths['excel']} and {paths['csv']}")
    
    return paths['csv']


def analyze_data(csv_path: str, output_prefix: str = "analysis") -> dict:
    """Perform comprehensive analysis on existing data."""
    logger = logging.getLogger("stock_analyzer")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    logger.info(f"Analyzing data from {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} companies from {csv_path}")
    
    # Enhance with calculated metrics
    logger.info("Calculating valuation metrics and scores...")
    enhanced_df = calculate_valuation_metrics(df)
    
    # Add percentile rankings
    key_metrics = [
        'P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA', 'ROE', 'ROA', 'ROIC',
        'ProfitMargin', 'OperatingMargin', 'RevenueGrowth', 'FCF Yield', 'Beta'
    ]
    enhanced_df = add_percentile_rankings(enhanced_df, key_metrics)
    
    # Save enhanced data
    enhanced_output = f"{output_prefix}_enhanced.xlsx"
    write_outputs(enhanced_df, enhanced_output)
    logger.info(f"Enhanced data saved to {enhanced_output}")
    
    # Calculate statistics
    logger.info("Calculating sector and industry statistics...")
    sector_stats = calculate_sector_stats(df, key_metrics)
    industry_stats = calculate_industry_stats(df, key_metrics)
    summary = generate_summary_stats(df)
    
    # Save analysis results
    import json
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
    
    # Peer comparisons
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
    
    logger.info("Analysis completed successfully!")
    
    return {
        'enhanced_data': enhanced_output,
        'analysis_json': analysis_json_path,
        'peer_comparisons': peer_json_path if peer_comparisons else None,
        'rankings': None  # Could be implemented
    }


def generate_report(csv_path: str, output_path: Optional[str] = None) -> str:
    """Generate comprehensive PDF report."""
    logger = logging.getLogger("stock_analyzer")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not output_path:
        csv_stem = Path(csv_path).stem
        output_path = f"{csv_stem}_report.pdf"
    
    logger.info(f"Generating PDF report: {output_path}")
    
    df = pd.read_csv(csv_path)
    result_path = generate_pdf_report(df, output_path)
    
    logger.info(f"PDF report generated: {result_path}")
    return result_path


def start_dashboard(port: int = 8050, debug: bool = False):
    """Start the interactive dashboard."""
    logger = logging.getLogger("stock_analyzer")
    logger.info(f"Starting dashboard on port {port}")
    
    try:
        from dashboard import app
        app.run_server(debug=debug, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise


def peer_analysis(csv_path: str, ticker: str, industry_field: str = 'Industry') -> dict:
    """Perform peer analysis for a specific ticker."""
    logger = logging.getLogger("stock_analyzer")
    
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    logger.info(f"Performing peer analysis for {ticker}")
    peer_data = calculate_peer_comparison(df, ticker.upper(), industry_field)
    
    if peer_data and 'metrics' in peer_data:
        logger.info(f"✓ Found {peer_data['peer_count']} peers in {peer_data['comparison_value']}")
        
        # Display key metrics
        print(f"\n📊 Peer Analysis Results for {ticker}:")
        print(f"Industry: {peer_data['comparison_value']}")
        print(f"Peer Companies: {peer_data['peer_count']}")
        print("\nKey Metrics Comparison:")
        
        for metric in ['P/E (TTM)', 'ROE', 'ProfitMargin', 'RevenueGrowth', 'DebtToEquity']:
            if metric in peer_data['metrics']:
                data = peer_data['metrics'][metric]
                print(f"  {metric}:")
                print(f"    {ticker}: {data['ticker_value']:.2f}")
                print(f"    Peer Median: {data['peer_median']:.2f}")
                print(f"    Percentile Rank: {data['percentile_rank']:.0f}%")
                print(f"    vs Median: {data['vs_median_percent']:+.1f}%")
                print()
    else:
        logger.warning(f"No peer data found for {ticker}")
    
    return peer_data


def main():
    parser = argparse.ArgumentParser(
        description="Complete Stock Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for specific tickers
  python stock_analysis_suite.py fetch --tickers AAPL MSFT GOOGL --output my_stocks.xlsx
  
  # Fetch data from file
  python stock_analysis_suite.py fetch --file tickers.txt --output portfolio.xlsx
  
  # Analyze existing data
  python stock_analysis_suite.py analyze --csv portfolio.csv --output my_analysis
  
  # Generate PDF report
  python stock_analysis_suite.py report --csv portfolio.csv --output portfolio_report.pdf
  
  # Peer analysis for specific stock
  python stock_analysis_suite.py peer --csv portfolio.csv --ticker AAPL
  
  # Start interactive dashboard
  python stock_analysis_suite.py dashboard --port 8050
  
  # Complete workflow (fetch + analyze + report)
  python stock_analysis_suite.py workflow --tickers AAPL MSFT GOOGL AMZN --output complete_analysis
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch stock data')
    fetch_parser.add_argument('--tickers', nargs='*', help='Ticker symbols')
    fetch_parser.add_argument('--file', help='File with ticker symbols')
    fetch_parser.add_argument('--output', default='stock_data.xlsx', help='Output file')
    
    # Analyze command  
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing data')
    analyze_parser.add_argument('--csv', required=True, help='CSV file with stock data')
    analyze_parser.add_argument('--output', default='analysis', help='Output prefix')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate PDF report')
    report_parser.add_argument('--csv', required=True, help='CSV file with stock data')
    report_parser.add_argument('--output', help='Output PDF file')
    
    # Peer command
    peer_parser = subparsers.add_parser('peer', help='Peer analysis for specific ticker')
    peer_parser.add_argument('--csv', required=True, help='CSV file with stock data')
    peer_parser.add_argument('--ticker', required=True, help='Ticker symbol to analyze')
    peer_parser.add_argument('--field', default='Industry', help='Comparison field (Industry/Sector)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start interactive dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8050, help='Port number')
    dashboard_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Workflow command
    workflow_parser = subparsers.add_parser('workflow', help='Complete analysis workflow')
    workflow_parser.add_argument('--tickers', nargs='*', help='Ticker symbols')
    workflow_parser.add_argument('--file', help='File with ticker symbols')
    workflow_parser.add_argument('--output', default='complete_analysis', help='Output prefix')
    workflow_parser.add_argument('--skip-report', action='store_true', help='Skip PDF report generation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s"
    )
    logger = logging.getLogger("stock_analyzer")
    
    try:
        if args.command == 'fetch':
            tickers = []
            if args.tickers:
                tickers.extend(args.tickers)
            if args.file:
                with open(args.file, 'r') as f:
                    file_tickers = [line.strip() for line in f.readlines()]
                    tickers.extend([t for t in file_tickers if t and not t.startswith('#')])
            
            if not tickers:
                logger.error("No tickers specified. Use --tickers or --file")
                return 1
            
            csv_path = fetch_stock_data(tickers, args.output)
            logger.info(f"✅ Data fetching completed. CSV: {csv_path}")
            
        elif args.command == 'analyze':
            results = analyze_data(args.csv, args.output)
            logger.info("✅ Analysis completed. Generated files:")
            for key, path in results.items():
                logger.info(f"  {key}: {path}")
            
        elif args.command == 'report':
            pdf_path = generate_report(args.csv, args.output)
            logger.info(f"✅ PDF report generated: {pdf_path}")
            
        elif args.command == 'peer':
            peer_data = peer_analysis(args.csv, args.ticker, args.field)
            
        elif args.command == 'dashboard':
            start_dashboard(args.port, args.debug)
            
        elif args.command == 'workflow':
            logger.info("🚀 Starting complete analysis workflow")
            
            # Step 1: Fetch data
            tickers = []
            if args.tickers:
                tickers.extend(args.tickers)
            if args.file:
                with open(args.file, 'r') as f:
                    file_tickers = [line.strip() for line in f.readlines()]
                    tickers.extend([t for t in file_tickers if t and not t.startswith('#')])
            
            if not tickers:
                logger.error("No tickers specified for workflow. Use --tickers or --file")
                return 1
            
            # Fetch
            csv_path = fetch_stock_data(tickers, f"{args.output}_data.xlsx")
            logger.info("✅ Step 1: Data fetching completed")
            
            # Analyze 
            analysis_results = analyze_data(csv_path, args.output)
            logger.info("✅ Step 2: Analysis completed")
            
            # Generate Report (unless skipped)
            if not args.skip_report:
                pdf_path = generate_report(csv_path, f"{args.output}_report.pdf")
                logger.info("✅ Step 3: PDF report generated")
            else:
                logger.info("⏭️  Step 3: PDF report skipped")
            
            # Summary
            logger.info("\n🎉 Complete workflow finished!")
            logger.info("Generated files:")
            logger.info(f"  📊 Data: {csv_path}")
            for key, path in analysis_results.items():
                logger.info(f"  📈 {key}: {path}")
            if not args.skip_report:
                logger.info(f"  📋 Report: {pdf_path}")
            
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.command == 'dashboard':
            logger.info("Use Ctrl+C to stop the dashboard")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)