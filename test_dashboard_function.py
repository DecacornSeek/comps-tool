#!/usr/bin/env python3
"""
Test the actual dashboard function to see what it returns
"""
import sys
sys.path.append('/home/user/webapp')

# Import the actual function from peer_dashboard
from peer_dashboard import get_historical_metrics, load_peer_data
import pandas as pd

def test_dashboard_functions():
    print("=== Testing Dashboard Functions ===\n")
    
    # Test 1: Load peer data
    print("1. Testing load_peer_data():")
    df = load_peer_data()
    if df.empty:
        print("   ❌ No peer data loaded")
        return False
    else:
        print(f"   ✅ Loaded {len(df)} tickers: {df['Ticker'].tolist()}")
        tickers = df['Ticker'].tolist()[:3]  # First 3 tickers
    
    # Test 2: Test historical metrics for first ticker
    print(f"\n2. Testing get_historical_metrics() for {tickers[0]}:")
    historical_data = get_historical_metrics(tickers[0])
    
    if historical_data is None:
        print("   ❌ No historical data returned")
        return False
    
    print(f"   ✅ Got historical data structure")
    print(f"   Keys: {list(historical_data.keys())}")
    
    if 'quarterly_data' in historical_data:
        quarterly_data = historical_data['quarterly_data']
        print(f"   Quarterly data metrics: {list(quarterly_data.keys())}")
        
        # Check each metric
        for metric in ['Revenue', 'EBITDA', 'Net Income', 'Free Cash Flow', 'Basic EPS']:
            if metric in quarterly_data:
                quarters = quarterly_data[metric]
                print(f"   ✅ {metric}: {len(quarters)} quarters")
                if quarters:
                    # Show first quarter data
                    first_quarter = list(quarters.keys())[0]
                    value = quarters[first_quarter]
                    print(f"      Latest: {first_quarter} = {value:,.0f}")
            else:
                print(f"   ❌ {metric}: Missing")
    
    # Test 3: Test create_chart_data_table function
    print(f"\n3. Testing create_chart_data_table() with tickers: {tickers}")
    try:
        from peer_dashboard import create_chart_data_table
        table_result = create_chart_data_table(tickers)
        print(f"   ✅ Table creation successful")
        print(f"   Table type: {type(table_result)}")
    except Exception as e:
        print(f"   ❌ Table creation failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_dashboard_functions()
    if success:
        print("\n🎯 All tests passed - dashboard should work!")
    else:
        print("\n❌ Tests failed - dashboard has issues")