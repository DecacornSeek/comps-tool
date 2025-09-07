#!/usr/bin/env python3
"""
Test if P/E and EPS historical data is available
"""
import yfinance as yf
import pandas as pd

def test_pe_eps_data(ticker):
    """Test what P/E and EPS data we can get"""
    print(f"\n=== Testing {ticker} for P/E and EPS Data ===")
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials
        
        print(f"Quarterly Financials available: {'Yes' if not quarterly_financials.empty else 'No'}")
        
        if not quarterly_financials.empty:
            print(f"Available financial metrics:")
            
            # Check for EPS metrics
            eps_metrics = ['Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares']
            for metric in eps_metrics:
                if metric in quarterly_financials.index:
                    print(f"  ✅ {metric}")
                    # Get last quarter value
                    last_quarter = quarterly_financials.columns[0]
                    value = quarterly_financials.loc[metric, last_quarter]
                    if 'EPS' in metric:
                        print(f"     Latest quarter: ${value:.3f}")
                    else:
                        print(f"     Latest quarter: {value:,.0f} shares")
                else:
                    print(f"  ❌ {metric} - NOT FOUND")
        
        # Get historical market data for P/E calculation
        print(f"\nTrying to get historical market data for P/E calculation...")
        
        # Get quarterly price data (at end of each quarter)
        hist = stock.history(period="2y", interval="1d")
        if not hist.empty:
            print(f"  ✅ Historical price data available: {len(hist)} days")
            
            # Get quarterly end dates and prices
            quarterly_prices = {}
            
            # Define quarter end dates for last 2 years
            from datetime import datetime, date
            import pandas as pd
            
            quarter_ends = [
                date(2025, 6, 30), date(2025, 3, 31), 
                date(2024, 12, 31), date(2024, 9, 30),
                date(2024, 6, 30), date(2024, 3, 31),
                date(2023, 12, 31), date(2023, 9, 30)
            ]
            
            for quarter_end in quarter_ends:
                # Find closest trading day to quarter end
                try:
                    quarter_end_str = quarter_end.strftime('%Y-%m-%d')
                    
                    # Look for exact date or closest before
                    available_dates = hist.index.date
                    closest_date = None
                    
                    for available_date in reversed(available_dates):
                        if available_date <= quarter_end:
                            closest_date = available_date
                            break
                    
                    if closest_date:
                        price = hist.loc[hist.index.date == closest_date, 'Close'].iloc[0]
                        quarterly_prices[quarter_end] = price
                        print(f"     {quarter_end}: ${price:.2f}")
                    
                except Exception as e:
                    continue
            
            print(f"  ✅ Got quarterly prices for {len(quarterly_prices)} quarters")
            
            # Now try to calculate P/E ratios
            if not quarterly_financials.empty and 'Basic EPS' in quarterly_financials.index:
                print(f"\nCalculating historical P/E ratios:")
                
                eps_data = quarterly_financials.loc['Basic EPS']
                
                for quarter_date, eps in eps_data.items():
                    quarter_end_date = quarter_date.date()
                    
                    if quarter_end_date in quarterly_prices and eps > 0:
                        price = quarterly_prices[quarter_end_date]
                        pe_ratio = price / eps
                        quarter_num = (quarter_date.month-1)//3 + 1
                        print(f"     {quarter_date.year} Q{quarter_num}: P/E = {pe_ratio:.1f} (Price: ${price:.2f}, EPS: ${eps:.3f})")
                    elif eps <= 0:
                        quarter_num = (quarter_date.month-1)//3 + 1
                        print(f"     {quarter_date.year} Q{quarter_num}: P/E = N/A (Negative EPS: ${eps:.3f})")
                
                return True
        else:
            print(f"  ❌ No historical price data available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Test with a few different tickers
    tickers = ['ZETA', 'TTD', 'GOOGL']
    
    for ticker in tickers:
        success = test_pe_eps_data(ticker)
        if not success:
            break
        print("\n" + "="*60)