#!/usr/bin/env python3
"""
Test historical data fetching to see what's available
"""
import yfinance as yf
import pandas as pd

def test_historical_data(ticker_symbol):
    """Test what historical data we can get for a ticker"""
    print(f"\n=== Testing {ticker_symbol} ===")
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get financial statements
        quarterly_financials = ticker.quarterly_financials
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        print(f"Quarterly Financials shape: {quarterly_financials.shape if not quarterly_financials.empty else 'EMPTY'}")
        print(f"Quarterly Balance Sheet shape: {quarterly_balance.shape if not quarterly_balance.empty else 'EMPTY'}")
        print(f"Quarterly Cash Flow shape: {quarterly_cashflow.shape if not quarterly_cashflow.empty else 'EMPTY'}")
        
        if not quarterly_financials.empty:
            print(f"\nFinancials columns (first 5): {quarterly_financials.columns[:5].tolist()}")
            print(f"Available metrics in financials:")
            for metric in ['Total Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Basic EPS', 'Diluted EPS']:
                if metric in quarterly_financials.index:
                    print(f"  ✅ {metric}")
                    # Get last quarter value
                    last_quarter = quarterly_financials.columns[0]
                    value = quarterly_financials.loc[metric, last_quarter]
                    quarter_num = (last_quarter.month-1)//3 + 1
                    print(f"     Latest quarter ({last_quarter.year} Q{quarter_num}): {value:,.0f}")
                else:
                    print(f"  ❌ {metric} - NOT FOUND")
        
        if not quarterly_cashflow.empty:
            print(f"\nCash Flow metrics:")
            for metric in ['Free Cash Flow', 'Operating Cash Flow', 'Capital Expenditure']:
                if metric in quarterly_cashflow.index:
                    print(f"  ✅ {metric}")
                    last_quarter = quarterly_cashflow.columns[0]
                    value = quarterly_cashflow.loc[metric, last_quarter]
                    print(f"     Latest quarter: {value:,.0f}")
                else:
                    print(f"  ❌ {metric} - NOT FOUND")
                    
        return True
        
    except Exception as e:
        print(f"❌ Error for {ticker_symbol}: {e}")
        return False

if __name__ == "__main__":
    # Test ZETA and a few other tickers
    tickers = ['ZETA', 'TTD', 'GOOGL']
    
    for ticker in tickers:
        success = test_historical_data(ticker)
        if not success:
            break