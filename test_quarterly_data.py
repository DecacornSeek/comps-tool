#!/usr/bin/env python3
"""
Test script to check quarterly financial data availability
for building the comprehensive chart like in the provided image
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def check_quarterly_data_availability(ticker):
    """Check what quarterly financial data is available for the last 3 years"""
    
    print(f"\n=== Checking Quarterly Data for {ticker} ===")
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials
        quarterly_balance_sheet = stock.quarterly_balance_sheet
        quarterly_cash_flow = stock.quarterly_cashflow
        
        print(f"\n📊 Available Quarterly Periods:")
        if not quarterly_financials.empty:
            periods = quarterly_financials.columns
            print(f"Quarterly Financials: {len(periods)} quarters")
            for i, period in enumerate(periods[:8]):  # Show last 8 quarters (2 years)
                print(f"  {i+1}. {period.strftime('%Y-Q%q (%b %Y)')}")
        
        # Check key metrics availability
        key_metrics = {
            'Revenue': ['Total Revenue', 'Revenue'],
            'EBIT': ['Operating Income', 'EBIT'], 
            'Net Income': ['Net Income', 'Net Income Common Stockholders'],
            'Gross Profit': ['Gross Profit'],
            'Operating Expenses': ['Operating Expense', 'Total Operating Expenses']
        }
        
        print(f"\n📈 Available Metrics:")
        available_data = {}
        
        for metric_name, possible_keys in key_metrics.items():
            found_key = None
            data_points = []
            
            for key in possible_keys:
                if key in quarterly_financials.index:
                    found_key = key
                    data_series = quarterly_financials.loc[key].dropna()
                    data_points = data_series.values[:8]  # Last 8 quarters
                    break
            
            if found_key and len(data_points) > 0:
                print(f"  ✅ {metric_name}: {found_key} ({len(data_points)} quarters)")
                available_data[metric_name] = {
                    'key': found_key,
                    'data': data_points,
                    'periods': quarterly_financials.columns[:len(data_points)]
                }
                
                # Show sample values (in billions)
                recent_values = data_points[:4] / 1e9  # Convert to billions
                print(f"     Recent quarters (B): {[f'{val:.2f}' for val in recent_values]}")
            else:
                print(f"  ❌ {metric_name}: Not available")
        
        # Calculate margins if we have the data
        if 'Revenue' in available_data and 'Net Income' in available_data:
            revenue_data = available_data['Revenue']['data']
            net_income_data = available_data['Net Income']['data']
            
            # Calculate net profit margins
            margins = []
            for i in range(min(len(revenue_data), len(net_income_data))):
                if revenue_data[i] != 0:
                    margin = (net_income_data[i] / revenue_data[i]) * 100
                    margins.append(margin)
            
            print(f"\n💹 Calculated Net Profit Margins:")
            print(f"     Last 4 quarters (%): {[f'{m:.1f}' for m in margins[:4]]}")
            available_data['Net Profit Margin'] = {
                'key': 'Calculated',
                'data': margins,
                'periods': available_data['Revenue']['periods'][:len(margins)]
            }
        
        return available_data
        
    except Exception as e:
        print(f"❌ Error checking data for {ticker}: {e}")
        return {}

def test_chart_data_structure(available_data):
    """Test if we can build the chart like in the image"""
    
    print(f"\n🔧 Chart Data Structure Test:")
    
    required_metrics = ['Revenue', 'EBIT', 'Net Income', 'Net Profit Margin']
    can_build_chart = True
    
    for metric in required_metrics:
        if metric in available_data:
            data_points = len(available_data[metric]['data'])
            print(f"  ✅ {metric}: {data_points} data points")
        else:
            print(f"  ❌ {metric}: Missing")
            can_build_chart = False
    
    if can_build_chart:
        print(f"\n🎯 Chart Building: ✅ POSSIBLE")
        print(f"   Can create bars for Revenue, EBIT, Net Income")
        print(f"   Can create line for Net Profit Margin")
        print(f"   Time period: Last {len(available_data['Revenue']['data'])} quarters")
    else:
        print(f"\n🎯 Chart Building: ❌ MISSING DATA")
    
    return can_build_chart

def check_additional_metrics(ticker):
    """Check availability of Market Cap, P/E, EPS for header"""
    
    print(f"\n📊 Additional Metrics for Header:")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        metrics = {
            'Market Cap': ['marketCap', 'Market Capitalization'],
            'P/E Ratio': ['trailingPE', 'forwardPE'],
            'EPS': ['trailingEps', 'forwardEps']
        }
        
        available_header_data = {}
        
        for metric_name, keys in metrics.items():
            found = False
            for key in keys:
                if key in info and info[key] is not None:
                    value = info[key]
                    print(f"  ✅ {metric_name}: {value}")
                    available_header_data[metric_name] = value
                    found = True
                    break
            
            if not found:
                print(f"  ❌ {metric_name}: Not available")
        
        return available_header_data
        
    except Exception as e:
        print(f"❌ Error checking header metrics: {e}")
        return {}

if __name__ == "__main__":
    # Test with multiple tickers
    test_tickers = ['ZETA', 'NVDA', 'AAPL']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        
        # Check quarterly data
        quarterly_data = check_quarterly_data_availability(ticker)
        
        # Test chart feasibility
        can_build = test_chart_data_structure(quarterly_data)
        
        # Check header metrics
        header_data = check_additional_metrics(ticker)
        
        print(f"\n📋 Summary for {ticker}:")
        print(f"   Quarterly Chart: {'✅ Possible' if can_build else '❌ Limited'}")
        print(f"   Header Metrics: {len(header_data)}/3 available")
        
        if can_build:
            print(f"   🎯 {ticker} is READY for comprehensive chart implementation")
            break
    
    print(f"\n{'='*60}")
    print("Data availability check complete!")