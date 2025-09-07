"""
Portfolio Manager - Manage and extend your stock portfolio analysis
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set
import pandas as pd

from comps_lib.metrics import fetch_metrics_for_ticker, COLUMNS
from comps_lib.analysis import calculate_valuation_metrics
from comps_lib.io import write_outputs
from comps_lib.pdf_reports import generate_pdf_report


class PortfolioManager:
    """Manages a persistent stock portfolio with incremental updates."""
    
    def __init__(self, portfolio_name: str = "my_portfolio"):
        self.portfolio_name = portfolio_name
        self.data_file = f"{portfolio_name}.xlsx"
        self.csv_file = f"{portfolio_name}.csv"
        self.logger = logging.getLogger("portfolio_manager")
        
    def load_existing_portfolio(self) -> pd.DataFrame:
        """Load existing portfolio if it exists."""
        if Path(self.csv_file).exists():
            df = pd.read_csv(self.csv_file)
            self.logger.info(f"📂 Loaded existing portfolio: {len(df)} stocks")
            return df
        else:
            self.logger.info("📝 Creating new portfolio")
            return pd.DataFrame(columns=COLUMNS)
    
    def get_existing_tickers(self) -> Set[str]:
        """Get set of tickers already in portfolio."""
        df = self.load_existing_portfolio()
        if 'Ticker' in df.columns:
            return set(df['Ticker'].str.upper())
        return set()
    
    def add_tickers(self, tickers: List[str], update_existing: bool = False) -> pd.DataFrame:
        """Add new tickers to portfolio or update existing ones."""
        tickers = [t.strip().upper() for t in tickers]
        existing_df = self.load_existing_portfolio()
        existing_tickers = self.get_existing_tickers()
        
        # Determine which tickers to fetch
        if update_existing:
            tickers_to_fetch = tickers
            self.logger.info(f"🔄 Updating {len(tickers)} tickers (including existing)")
        else:
            tickers_to_fetch = [t for t in tickers if t not in existing_tickers]
            already_exist = [t for t in tickers if t in existing_tickers]
            
            if already_exist:
                self.logger.info(f"⏭️  Skipping existing tickers: {', '.join(already_exist)}")
            
            if not tickers_to_fetch:
                self.logger.info("ℹ️  All tickers already in portfolio. Use --update to refresh data.")
                return existing_df
        
        # Fetch new data
        if tickers_to_fetch:
            self.logger.info(f"📥 Fetching data for {len(tickers_to_fetch)} tickers...")
            new_rows = []
            
            for ticker in tickers_to_fetch:
                try:
                    row = fetch_metrics_for_ticker(ticker)
                    new_rows.append(row)
                    self.logger.info(f"  ✅ {ticker}")
                except Exception as e:
                    self.logger.error(f"  ❌ {ticker}: {e}")
                    # Add error row
                    error_row = {col: (ticker if col == "Ticker" else "ERROR") for col in COLUMNS}
                    new_rows.append(error_row)
            
            new_df = pd.DataFrame(new_rows, columns=COLUMNS)
            
            if update_existing:
                # Replace existing data completely
                # Remove old entries for updated tickers
                existing_df = existing_df[~existing_df['Ticker'].str.upper().isin([t.upper() for t in tickers_to_fetch])]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Add to existing data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = existing_df
        
        # Sort by Ticker for consistency
        combined_df = combined_df.sort_values('Ticker').reset_index(drop=True)
        
        # Save updated portfolio
        self.save_portfolio(combined_df)
        
        return combined_df
    
    def remove_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Remove tickers from portfolio."""
        tickers = [t.strip().upper() for t in tickers]
        df = self.load_existing_portfolio()
        
        if df.empty:
            self.logger.warning("Portfolio is empty")
            return df
        
        initial_count = len(df)
        df = df[~df['Ticker'].str.upper().isin(tickers)]
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            self.logger.info(f"🗑️  Removed {removed_count} tickers from portfolio")
            self.save_portfolio(df)
        else:
            self.logger.info("No tickers were removed (not found in portfolio)")
        
        return df
    
    def list_portfolio(self) -> pd.DataFrame:
        """List current portfolio contents."""
        df = self.load_existing_portfolio()
        
        if df.empty:
            print("📝 Portfolio is empty")
            return df
        
        # Display summary
        print(f"\n📊 Portfolio: {self.portfolio_name}")
        print(f"📈 Total Stocks: {len(df)}")
        
        if 'Sector' in df.columns:
            sectors = df['Sector'].value_counts()
            print(f"🏭 Sectors: {sectors.to_dict()}")
        
        if 'MarketCap' in df.columns:
            total_mcap = pd.to_numeric(df['MarketCap'], errors='coerce').sum()
            if not pd.isna(total_mcap):
                print(f"💰 Total Market Cap: ${total_mcap/1e12:.2f}T")
        
        # Show ticker list with key metrics
        display_cols = ['Ticker', 'Price', 'MarketCap', 'P/E (TTM)', 'ROE', 'Sector']
        available_cols = [col for col in display_cols if col in df.columns]
        
        if available_cols:
            display_df = df[available_cols].copy()
            
            # Format for display
            if 'MarketCap' in display_df.columns:
                display_df['MarketCap'] = display_df['MarketCap'].apply(
                    lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A"
                )
            
            for col in ['P/E (TTM)', 'ROE']:
                if col in display_df.columns:
                    if col == 'ROE':
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                        )
                    else:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                        )
            
            print(f"\n{display_df.to_string(index=False)}")
        
        return df
    
    def save_portfolio(self, df: pd.DataFrame):
        """Save portfolio to files."""
        paths = write_outputs(df, self.data_file)
        self.logger.info(f"💾 Portfolio saved: {paths['excel']} ({len(df)} stocks)")
    
    def analyze_portfolio(self, include_report: bool = True):
        """Analyze current portfolio."""
        df = self.load_existing_portfolio()
        
        if df.empty:
            self.logger.error("Portfolio is empty. Add some tickers first.")
            return
        
        self.logger.info(f"📊 Analyzing portfolio with {len(df)} stocks...")
        
        # Enhanced analysis
        enhanced_df = calculate_valuation_metrics(df)
        enhanced_path = f"{self.portfolio_name}_analysis.xlsx"
        write_outputs(enhanced_df, enhanced_path)
        self.logger.info(f"📈 Enhanced analysis: {enhanced_path}")
        
        # Generate report if requested
        if include_report:
            report_path = f"{self.portfolio_name}_report.pdf"
            generate_pdf_report(df, report_path)
            self.logger.info(f"📋 Report generated: {report_path}")
        
        # Portfolio summary
        self.show_portfolio_summary(enhanced_df)
    
    def show_portfolio_summary(self, df: pd.DataFrame):
        """Show portfolio analysis summary."""
        print(f"\n📊 Portfolio Analysis Summary")
        print("=" * 50)
        
        # Portfolio composition
        if 'Sector' in df.columns:
            sectors = df['Sector'].value_counts()
            print(f"🏭 Sector Allocation:")
            for sector, count in sectors.head(5).items():
                pct = count / len(df) * 100
                print(f"  {sector}: {count} stocks ({pct:.1f}%)")
        
        # Valuation summary
        if 'P/E (TTM)' in df.columns:
            pe_values = pd.to_numeric(df['P/E (TTM)'], errors='coerce').dropna()
            if len(pe_values) > 0:
                print(f"\n📈 Valuation Metrics:")
                print(f"  Average P/E: {pe_values.mean():.1f}")
                print(f"  Median P/E: {pe_values.median():.1f}")
        
        # Performance metrics
        if 'ROE' in df.columns:
            roe_values = pd.to_numeric(df['ROE'], errors='coerce').dropna()
            if len(roe_values) > 0:
                high_roe = (roe_values > 0.15).sum()
                print(f"\n💪 Performance:")
                print(f"  Average ROE: {roe_values.mean():.1%}")
                print(f"  High ROE (>15%): {high_roe} stocks")
        
        # Top performers
        if 'Valuation_Score' in df.columns:
            top_value = df.nlargest(3, 'Valuation_Score')[['Ticker', 'Valuation_Score']]
            print(f"\n🏆 Top Value Picks:")
            for _, row in top_value.iterrows():
                print(f"  {row['Ticker']}: {row['Valuation_Score']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Manager - Manage your stock analysis portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create/add to portfolio
  python portfolio_manager.py add --tickers AAPL MSFT GOOGL --portfolio tech_stocks
  
  # Add ASML to existing portfolio
  python portfolio_manager.py add --tickers ASML.AS --portfolio tech_stocks
  
  # Update existing stocks with fresh data
  python portfolio_manager.py add --tickers AAPL MSFT --update --portfolio tech_stocks
  
  # List current portfolio
  python portfolio_manager.py list --portfolio tech_stocks
  
  # Analyze portfolio (with report)
  python portfolio_manager.py analyze --portfolio tech_stocks
  
  # Remove stocks
  python portfolio_manager.py remove --tickers AAPL --portfolio tech_stocks
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add tickers to portfolio')
    add_parser.add_argument('--tickers', nargs='+', required=True, help='Ticker symbols to add')
    add_parser.add_argument('--portfolio', default='my_portfolio', help='Portfolio name')
    add_parser.add_argument('--update', action='store_true', help='Update existing tickers')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove tickers from portfolio')
    remove_parser.add_argument('--tickers', nargs='+', required=True, help='Ticker symbols to remove')
    remove_parser.add_argument('--portfolio', default='my_portfolio', help='Portfolio name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List portfolio contents')
    list_parser.add_argument('--portfolio', default='my_portfolio', help='Portfolio name')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze portfolio')
    analyze_parser.add_argument('--portfolio', default='my_portfolio', help='Portfolio name')
    analyze_parser.add_argument('--no-report', action='store_true', help='Skip PDF report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s"
    )
    
    try:
        portfolio = PortfolioManager(args.portfolio)
        
        if args.command == 'add':
            portfolio.add_tickers(args.tickers, update_existing=args.update)
            
        elif args.command == 'remove':
            portfolio.remove_tickers(args.tickers)
            
        elif args.command == 'list':
            portfolio.list_portfolio()
            
        elif args.command == 'analyze':
            portfolio.analyze_portfolio(include_report=not args.no_report)
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())