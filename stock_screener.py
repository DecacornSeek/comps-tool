"""
Stock Screener - Master Watchlist + Individual Analysis
Perfect for building a comprehensive stock database and analyzing individual opportunities
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


class StockScreener:
    """Manages a master watchlist and provides individual stock analysis."""
    
    def __init__(self, watchlist_name: str = "master_watchlist"):
        self.watchlist_name = watchlist_name
        self.data_file = f"{watchlist_name}.xlsx"
        self.csv_file = f"{watchlist_name}.csv"
        self.logger = logging.getLogger("stock_screener")
        
    def load_watchlist(self) -> pd.DataFrame:
        """Load existing watchlist."""
        if Path(self.csv_file).exists():
            df = pd.read_csv(self.csv_file)
            self.logger.info(f"📂 Master Watchlist loaded: {len(df)} stocks")
            return df
        else:
            self.logger.info("📝 Creating new master watchlist")
            return pd.DataFrame(columns=COLUMNS)
    
    def add_to_watchlist(self, tickers: List[str], update_existing: bool = False) -> pd.DataFrame:
        """Add new tickers to master watchlist."""
        tickers = [t.strip().upper() for t in tickers]
        existing_df = self.load_watchlist()
        existing_tickers = set()
        
        if 'Ticker' in existing_df.columns:
            existing_tickers = set(existing_df['Ticker'].str.upper())
        
        # Determine which tickers to fetch
        if update_existing:
            tickers_to_fetch = tickers
            self.logger.info(f"🔄 Updating {len(tickers)} tickers (including existing)")
        else:
            tickers_to_fetch = [t for t in tickers if t not in existing_tickers]
            already_exist = [t for t in tickers if t in existing_tickers]
            
            if already_exist:
                self.logger.info(f"⏭️  Already in watchlist: {', '.join(already_exist)}")
            
            if not tickers_to_fetch:
                self.logger.info("ℹ️  All tickers already in watchlist. Use --update to refresh data.")
                return existing_df
        
        # Fetch new data
        if tickers_to_fetch:
            self.logger.info(f"📥 Adding {len(tickers_to_fetch)} stocks to master watchlist...")
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
                existing_df = existing_df[~existing_df['Ticker'].str.upper().isin([t.upper() for t in tickers_to_fetch])]
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                # Add to existing data
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = existing_df
        
        # Sort by Ticker for consistency
        combined_df = combined_df.sort_values('Ticker').reset_index(drop=True)
        
        # Save updated watchlist
        self.save_watchlist(combined_df)
        
        return combined_df
    
    def save_watchlist(self, df: pd.DataFrame):
        """Save watchlist to files."""
        paths = write_outputs(df, self.data_file)
        self.logger.info(f"💾 Master watchlist saved: {len(df)} stocks")
    
    def screen_watchlist(self, criteria: dict = None) -> pd.DataFrame:
        """Screen watchlist based on criteria."""
        df = self.load_watchlist()
        
        if df.empty:
            self.logger.warning("Watchlist is empty")
            return df
        
        # Default screening criteria
        if criteria is None:
            criteria = {
                'min_market_cap': 1e9,  # > $1B
                'max_pe': 50,           # P/E < 50
                'min_roe': 0.1,         # ROE > 10%
            }
        
        # Apply filters
        filtered_df = df.copy()
        
        if 'min_market_cap' in criteria:
            market_cap_filter = pd.to_numeric(filtered_df['MarketCap'], errors='coerce') >= criteria['min_market_cap']
            filtered_df = filtered_df[market_cap_filter]
        
        if 'max_pe' in criteria:
            pe_filter = pd.to_numeric(filtered_df['P/E (TTM)'], errors='coerce') <= criteria['max_pe']
            filtered_df = filtered_df[pe_filter | pd.isna(pd.to_numeric(filtered_df['P/E (TTM)'], errors='coerce'))]
        
        if 'min_roe' in criteria:
            roe_filter = pd.to_numeric(filtered_df['ROE'], errors='coerce') >= criteria['min_roe']
            filtered_df = filtered_df[roe_filter | pd.isna(pd.to_numeric(filtered_df['ROE'], errors='coerce'))]
        
        self.logger.info(f"📊 Screening results: {len(filtered_df)} of {len(df)} stocks match criteria")
        
        return filtered_df
    
    def analyze_stock(self, ticker: str, peer_tickers: List[str] = None) -> dict:
        """Deep analysis of individual stock with peer comparison."""
        ticker = ticker.upper()
        df = self.load_watchlist()
        
        if df.empty:
            self.logger.error("Watchlist is empty. Add some stocks first.")
            return {}
        
        # Check if ticker is in watchlist
        if ticker not in df['Ticker'].str.upper().values:
            self.logger.warning(f"{ticker} not in watchlist. Adding it now...")
            self.add_to_watchlist([ticker])
            df = self.load_watchlist()
        
        # Get stock data
        stock_data = df[df['Ticker'].str.upper() == ticker].iloc[0]
        
        print(f'\n🔍 INDIVIDUAL STOCK ANALYSIS: {ticker}')
        print('=' * 60)
        
        # Basic info
        print(f'💰 Price: ${stock_data["Price"]:.2f}')
        print(f'📊 Market Cap: ${stock_data["MarketCap"]/1e9:.1f}B')
        print(f'🏭 Industry: {stock_data.get("Industry", "N/A")}')
        print(f'🌍 Sector: {stock_data.get("Sector", "N/A")}')
        
        # Key metrics
        print(f'\n📈 KEY METRICS:')
        print(f'   P/E (TTM): {stock_data["P/E (TTM)"]:.1f}' if pd.notna(stock_data["P/E (TTM)"]) else '   P/E (TTM): N/A')
        print(f'   ROE: {stock_data["ROE"]:.1%}' if pd.notna(stock_data["ROE"]) else '   ROE: N/A')
        print(f'   Revenue Growth: {stock_data["RevenueGrowth"]:.1%}' if pd.notna(stock_data["RevenueGrowth"]) else '   Revenue Growth: N/A')
        print(f'   Profit Margin: {stock_data["ProfitMargin"]:.1%}' if pd.notna(stock_data["ProfitMargin"]) else '   Profit Margin: N/A')
        
        # Auto peer selection if not provided
        if peer_tickers is None:
            # Find peers in same industry/sector from watchlist
            same_industry = df[df['Industry'] == stock_data.get('Industry')]
            same_sector = df[df['Sector'] == stock_data.get('Sector')]
            
            # Prefer same industry, fallback to sector
            if len(same_industry) > 1:
                peers_df = same_industry[same_industry['Ticker'] != ticker]
                comparison_basis = f"Industry: {stock_data.get('Industry')}"
            elif len(same_sector) > 1:
                peers_df = same_sector[same_sector['Ticker'] != ticker]
                comparison_basis = f"Sector: {stock_data.get('Sector')}"
            else:
                peers_df = pd.DataFrame()
                comparison_basis = "No peers found in watchlist"
        else:
            # Use provided peer tickers
            peer_tickers_upper = [p.upper() for p in peer_tickers]
            peers_df = df[df['Ticker'].str.upper().isin(peer_tickers_upper)]
            comparison_basis = f"Custom peer group ({len(peer_tickers)} stocks)"
        
        # Peer comparison
        if not peers_df.empty:
            print(f'\n🔍 PEER COMPARISON ({comparison_basis}):')
            
            # Create comparison table
            comparison_data = [stock_data]  # Add our stock first
            comparison_data.extend([peers_df.iloc[i] for i in range(len(peers_df))])
            
            comp_df = pd.DataFrame(comparison_data)
            
            print(f'   {"Ticker":8s} {"P/E":>6s} {"ROE":>8s} {"Growth":>8s} {"Margin":>8s}')
            print(f'   {"-"*50}')
            
            for _, row in comp_df.iterrows():
                pe_str = f"{row['P/E (TTM)']:.1f}" if pd.notna(row['P/E (TTM)']) else "N/A"
                roe_str = f"{row['ROE']:.1%}" if pd.notna(row['ROE']) else "N/A"
                growth_str = f"{row['RevenueGrowth']:.1%}" if pd.notna(row['RevenueGrowth']) else "N/A"
                margin_str = f"{row['ProfitMargin']:.1%}" if pd.notna(row['ProfitMargin']) else "N/A"
                highlight = " ←" if row['Ticker'].upper() == ticker else ""
                
                print(f'   {row["Ticker"]:8s} {pe_str:>6s} {roe_str:>8s} {growth_str:>8s} {margin_str:>8s}{highlight}')
            
            # Peer analysis
            numeric_peers = peers_df.select_dtypes(include=[float, int])
            if len(numeric_peers) > 0:
                print(f'\n📊 RELATIVE POSITIONING:')
                
                # P/E comparison
                if 'P/E (TTM)' in numeric_peers.columns:
                    peer_pe_values = pd.to_numeric(peers_df['P/E (TTM)'], errors='coerce').dropna()
                    if len(peer_pe_values) > 0 and pd.notna(stock_data['P/E (TTM)']):
                        peer_median_pe = peer_pe_values.median()
                        pe_vs_peers = (stock_data['P/E (TTM)'] - peer_median_pe) / peer_median_pe * 100
                        pe_assessment = "CHEAPER" if pe_vs_peers < -10 else "EXPENSIVE" if pe_vs_peers > 10 else "FAIRLY VALUED"
                        print(f'   P/E vs Peers: {pe_vs_peers:+.1f}% ({pe_assessment})')
                
                # ROE comparison
                if 'ROE' in numeric_peers.columns:
                    peer_roe_values = pd.to_numeric(peers_df['ROE'], errors='coerce').dropna()
                    if len(peer_roe_values) > 0 and pd.notna(stock_data['ROE']):
                        peer_median_roe = peer_roe_values.median()
                        roe_vs_peers = (stock_data['ROE'] - peer_median_roe) / peer_median_roe * 100
                        roe_assessment = "SUPERIOR" if roe_vs_peers > 20 else "INFERIOR" if roe_vs_peers < -20 else "COMPARABLE"
                        print(f'   ROE vs Peers: {roe_vs_peers:+.1f}% ({roe_assessment})')
        
        else:
            print(f'\n⚠️  No peers found for comparison')
            print(f'   Add more stocks in {stock_data.get("Industry", "same industry")} to enable peer analysis')
        
        # Investment recommendation
        print(f'\n🎯 QUICK ASSESSMENT:')
        
        factors = []
        
        # Valuation
        pe = stock_data.get('P/E (TTM)')
        if pd.notna(pe):
            if pe < 15:
                factors.append("✅ Attractive valuation")
            elif pe < 25:
                factors.append("🟡 Reasonable valuation")  
            else:
                factors.append("⚠️ High valuation")
        
        # Profitability
        roe = stock_data.get('ROE')
        if pd.notna(roe):
            if roe > 0.20:
                factors.append("✅ Excellent profitability")
            elif roe > 0.10:
                factors.append("🟡 Good profitability")
            else:
                factors.append("⚠️ Low profitability")
        
        # Growth
        growth = stock_data.get('RevenueGrowth')
        if pd.notna(growth):
            if growth > 0.15:
                factors.append("✅ Strong growth")
            elif growth > 0.05:
                factors.append("🟡 Moderate growth")
            else:
                factors.append("⚠️ Slow growth")
        
        for factor in factors:
            print(f'   {factor}')
        
        return {
            'ticker': ticker,
            'stock_data': stock_data,
            'peers': peers_df,
            'assessment_factors': factors
        }
    
    def show_watchlist_summary(self):
        """Show summary of current watchlist."""
        df = self.load_watchlist()
        
        if df.empty:
            print("📝 Watchlist is empty")
            return
        
        print(f'\n📊 MASTER WATCHLIST SUMMARY')
        print(f'=' * 50)
        print(f'📈 Total Stocks: {len(df)}')
        
        # Sector breakdown
        if 'Sector' in df.columns:
            sectors = df['Sector'].value_counts().head(5)
            print(f'\n🏭 Top Sectors:')
            for sector, count in sectors.items():
                pct = count / len(df) * 100
                print(f'   {sector}: {count} stocks ({pct:.1f}%)')
        
        # Market cap distribution
        if 'MarketCap' in df.columns:
            market_caps = pd.to_numeric(df['MarketCap'], errors='coerce').dropna()
            if len(market_caps) > 0:
                print(f'\n💰 Market Cap Distribution:')
                large_cap = (market_caps >= 10e9).sum()
                mid_cap = ((market_caps >= 2e9) & (market_caps < 10e9)).sum()
                small_cap = (market_caps < 2e9).sum()
                
                print(f'   Large Cap (>$10B): {large_cap} stocks')
                print(f'   Mid Cap ($2-10B): {mid_cap} stocks')
                print(f'   Small Cap (<$2B): {small_cap} stocks')
        
        # Recent additions (top 10)
        print(f'\n📋 Recent Additions (showing 10):')
        display_cols = ['Ticker', 'Sector', 'Industry', 'MarketCap', 'P/E (TTM)', 'ROE']
        available_cols = [col for col in display_cols if col in df.columns]
        
        if available_cols:
            display_df = df[available_cols].tail(10).copy()
            
            # Format for display
            if 'MarketCap' in display_df.columns:
                display_df['MarketCap'] = display_df['MarketCap'].apply(
                    lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A"
                )
            
            if 'ROE' in display_df.columns:
                display_df['ROE'] = display_df['ROE'].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                )
            
            if 'P/E (TTM)' in display_df.columns:
                display_df['P/E (TTM)'] = display_df['P/E (TTM)'].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
            
            print(display_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Stock Screener - Master Watchlist Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add stocks to master watchlist
  python stock_screener.py add --tickers AAPL MSFT GOOGL ASML.AS SAP.DE
  
  # Show current watchlist
  python stock_screener.py list
  
  # Analyze specific stock with auto peer detection
  python stock_screener.py analyze --ticker ASML.AS
  
  # Analyze with custom peer group
  python stock_screener.py analyze --ticker ASML.AS --peers NVDA TSM QCOM
  
  # Screen watchlist for opportunities  
  python stock_screener.py screen --min-roe 0.15 --max-pe 25
  
  # Update existing data
  python stock_screener.py add --tickers AAPL MSFT --update
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add stocks to watchlist')
    add_parser.add_argument('--tickers', nargs='+', required=True, help='Ticker symbols to add')
    add_parser.add_argument('--update', action='store_true', help='Update existing tickers')
    
    # List command
    list_parser = subparsers.add_parser('list', help='Show watchlist summary')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze individual stock')
    analyze_parser.add_argument('--ticker', required=True, help='Ticker to analyze')
    analyze_parser.add_argument('--peers', nargs='*', help='Custom peer tickers for comparison')
    
    # Screen command
    screen_parser = subparsers.add_parser('screen', help='Screen watchlist for opportunities')
    screen_parser.add_argument('--min-roe', type=float, default=0.10, help='Minimum ROE (default: 0.10)')
    screen_parser.add_argument('--max-pe', type=float, default=50, help='Maximum P/E (default: 50)')
    screen_parser.add_argument('--min-mcap', type=float, default=1e9, help='Minimum Market Cap (default: 1B)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    
    try:
        screener = StockScreener()
        
        if args.command == 'add':
            screener.add_to_watchlist(args.tickers, update_existing=args.update)
            
        elif args.command == 'list':
            screener.show_watchlist_summary()
            
        elif args.command == 'analyze':
            screener.analyze_stock(args.ticker, peer_tickers=args.peers)
            
        elif args.command == 'screen':
            criteria = {
                'min_roe': args.min_roe,
                'max_pe': args.max_pe, 
                'min_market_cap': args.min_mcap
            }
            filtered_df = screener.screen_watchlist(criteria)
            
            if not filtered_df.empty:
                print(f'\n📊 SCREENING RESULTS ({len(filtered_df)} stocks):')
                print('=' * 50)
                
                # Show filtered results
                display_cols = ['Ticker', 'Sector', 'P/E (TTM)', 'ROE', 'RevenueGrowth', 'MarketCap']
                available_cols = [col for col in display_cols if col in filtered_df.columns]
                
                if available_cols:
                    display_df = filtered_df[available_cols].copy()
                    
                    # Format for display
                    for col in ['ROE', 'RevenueGrowth']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                            )
                    
                    if 'MarketCap' in display_df.columns:
                        display_df['MarketCap'] = display_df['MarketCap'].apply(
                            lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A"
                        )
                    
                    if 'P/E (TTM)' in display_df.columns:
                        display_df['P/E (TTM)'] = display_df['P/E (TTM)'].apply(
                            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                        )
                    
                    print(display_df.to_string(index=False))
            else:
                print(f'\n❌ No stocks match the screening criteria')
        
        return 0
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())