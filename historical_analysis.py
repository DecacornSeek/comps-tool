"""
Historical Financial Analysis Tool
Analyzes trends over last 4 quarters and 2 years
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse

def get_historical_metrics(ticker_symbol):
    """
    Get historical financial metrics for a ticker
    Returns quarterly and yearly trends
    """
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        # Get financial statements
        quarterly_financials = ticker.quarterly_financials
        yearly_financials = ticker.financials
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        # Prepare results dictionary
        results = {
            'ticker': ticker_symbol,
            'quarterly_data': {},
            'yearly_data': {},
            'trends': {},
            'available_quarters': quarterly_financials.columns.tolist() if not quarterly_financials.empty else [],
            'available_years': yearly_financials.columns.tolist() if not yearly_financials.empty else []
        }
        
        # Key metrics to track
        quarterly_metrics = {
            'Revenue': 'Total Revenue',
            'Net Income': 'Net Income',
            'EBITDA': 'EBITDA',
            'Operating Income': 'Total Operating Income As Reported',
            'Gross Profit': 'Gross Profit'
        }
        
        balance_metrics = {
            'Total Debt': 'Total Debt',
            'Cash': 'Cash And Cash Equivalents',
            'Working Capital': 'Working Capital',
            'Stockholders Equity': 'Stockholders Equity',
            'Total Assets': 'Total Assets'
        }
        
        cashflow_metrics = {
            'Free Cash Flow': 'Free Cash Flow',
            'Operating Cash Flow': 'Operating Cash Flow',
            'Capital Expenditure': 'Capital Expenditure'
        }
        
        # Extract quarterly data
        print(f"🔍 Analyzing {ticker_symbol} historical trends...")
        
        # Financials
        for display_name, metric_name in quarterly_metrics.items():
            if metric_name in quarterly_financials.index:
                data = quarterly_financials.loc[metric_name].dropna()
                results['quarterly_data'][display_name] = data.to_dict()
        
        # Balance sheet
        for display_name, metric_name in balance_metrics.items():
            if metric_name in quarterly_balance.index:
                data = quarterly_balance.loc[metric_name].dropna()
                results['quarterly_data'][display_name] = data.to_dict()
        
        # Cash flow
        for display_name, metric_name in cashflow_metrics.items():
            if metric_name in quarterly_cashflow.index:
                data = quarterly_cashflow.loc[metric_name].dropna()
                results['quarterly_data'][display_name] = data.to_dict()
        
        # Calculate trends (QoQ and YoY growth rates)
        results['trends'] = calculate_growth_trends(results['quarterly_data'])
        
        return results
        
    except Exception as e:
        print(f"❌ Error fetching data for {ticker_symbol}: {e}")
        return None

def calculate_growth_trends(quarterly_data):
    """Calculate growth trends from quarterly data"""
    trends = {}
    
    for metric_name, data_dict in quarterly_data.items():
        if len(data_dict) >= 2:
            # Convert to pandas Series and sort by date
            series = pd.Series(data_dict).sort_index()
            
            # Calculate QoQ growth (most recent vs previous quarter)
            if len(series) >= 2:
                recent_qoq = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100
                trends[f"{metric_name}_QoQ"] = recent_qoq
            
            # Calculate YoY growth if we have enough data
            if len(series) >= 4:
                recent_yoy = ((series.iloc[-1] / series.iloc[-4]) - 1) * 100
                trends[f"{metric_name}_YoY"] = recent_yoy
            
            # Calculate 4-quarter average growth
            if len(series) >= 4:
                growth_rates = []
                for i in range(1, min(5, len(series))):
                    if series.iloc[-i-1] != 0:  # Avoid division by zero
                        qoq_growth = ((series.iloc[-i] / series.iloc[-i-1]) - 1) * 100
                        growth_rates.append(qoq_growth)
                if growth_rates:
                    trends[f"{metric_name}_Avg_QoQ"] = np.mean(growth_rates)
    
    return trends

def create_trends_visualization(historical_data):
    """Create interactive charts for historical trends"""
    if not historical_data or not historical_data['quarterly_data']:
        return None
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend', 'Profitability Trend', 'Cash Flow Trend', 'Balance Sheet Health'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Revenue trend
    if 'Revenue' in historical_data['quarterly_data']:
        revenue_data = historical_data['quarterly_data']['Revenue']
        dates = list(revenue_data.keys())
        values = list(revenue_data.values())
        
        fig.add_trace(
            go.Scatter(x=dates, y=values, name='Revenue', 
                      line=dict(color='blue', width=3),
                      mode='lines+markers'),
            row=1, col=1
        )
    
    # Profitability (Net Income vs EBITDA)
    if 'Net Income' in historical_data['quarterly_data']:
        ni_data = historical_data['quarterly_data']['Net Income']
        dates = list(ni_data.keys())
        values = list(ni_data.values())
        
        fig.add_trace(
            go.Scatter(x=dates, y=values, name='Net Income',
                      line=dict(color='green', width=2),
                      mode='lines+markers'),
            row=1, col=2
        )
    
    if 'EBITDA' in historical_data['quarterly_data']:
        ebitda_data = historical_data['quarterly_data']['EBITDA']
        dates = list(ebitda_data.keys())
        values = list(ebitda_data.values())
        
        fig.add_trace(
            go.Scatter(x=dates, y=values, name='EBITDA',
                      line=dict(color='orange', width=2),
                      mode='lines+markers'),
            row=1, col=2
        )
    
    # Cash Flow
    if 'Free Cash Flow' in historical_data['quarterly_data']:
        fcf_data = historical_data['quarterly_data']['Free Cash Flow']
        dates = list(fcf_data.keys())
        values = list(fcf_data.values())
        
        fig.add_trace(
            go.Scatter(x=dates, y=values, name='Free Cash Flow',
                      line=dict(color='purple', width=3),
                      mode='lines+markers'),
            row=2, col=1
        )
    
    # Balance Sheet (Debt vs Cash)
    if 'Total Debt' in historical_data['quarterly_data']:
        debt_data = historical_data['quarterly_data']['Total Debt']
        dates = list(debt_data.keys())
        values = list(debt_data.values())
        
        fig.add_trace(
            go.Scatter(x=dates, y=values, name='Total Debt',
                      line=dict(color='red', width=2),
                      mode='lines+markers'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text=f"📊 {historical_data['ticker']} - Historical Financial Trends",
        showlegend=True
    )
    
    return fig

def analyze_peer_historical_trends(tickers):
    """Analyze historical trends for multiple tickers"""
    print(f"🚀 Analyzing historical trends for {len(tickers)} tickers...")
    
    results = {}
    for ticker in tickers:
        print(f"  📈 Processing {ticker}...")
        historical_data = get_historical_metrics(ticker)
        if historical_data:
            results[ticker] = historical_data
    
    return results

def create_peer_trends_comparison(peer_historical_data):
    """Create comparison charts for peer group trends"""
    if not peer_historical_data:
        return None
    
    # Revenue Growth Comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trends', 'Free Cash Flow Trends', 
                       'QoQ Revenue Growth', 'YoY Revenue Growth'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    # Revenue trends for all peers
    for i, (ticker, data) in enumerate(peer_historical_data.items()):
        if 'Revenue' in data['quarterly_data']:
            revenue_data = data['quarterly_data']['Revenue']
            dates = list(revenue_data.keys())
            values = [v/1e6 for v in revenue_data.values()]  # Convert to millions
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, name=f'{ticker} Revenue',
                          line=dict(color=colors[i % len(colors)], width=2),
                          mode='lines+markers'),
                row=1, col=1
            )
    
    # FCF trends
    for i, (ticker, data) in enumerate(peer_historical_data.items()):
        if 'Free Cash Flow' in data['quarterly_data']:
            fcf_data = data['quarterly_data']['Free Cash Flow']
            dates = list(fcf_data.keys())
            values = [v/1e6 for v in fcf_data.values()]  # Convert to millions
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, name=f'{ticker} FCF',
                          line=dict(color=colors[i % len(colors)], width=2),
                          mode='lines+markers'),
                row=1, col=2
            )
    
    # QoQ Growth rates
    qoq_tickers = []
    qoq_values = []
    for ticker, data in peer_historical_data.items():
        if 'Revenue_QoQ' in data['trends']:
            qoq_tickers.append(ticker)
            qoq_values.append(data['trends']['Revenue_QoQ'])
    
    if qoq_tickers:
        fig.add_trace(
            go.Bar(x=qoq_tickers, y=qoq_values, name='QoQ Growth %',
                   marker_color='lightblue'),
            row=2, col=1
        )
    
    # YoY Growth rates
    yoy_tickers = []
    yoy_values = []
    for ticker, data in peer_historical_data.items():
        if 'Revenue_YoY' in data['trends']:
            yoy_tickers.append(ticker)
            yoy_values.append(data['trends']['Revenue_YoY'])
    
    if yoy_tickers:
        fig.add_trace(
            go.Bar(x=yoy_tickers, y=yoy_values, name='YoY Growth %',
                   marker_color='lightgreen'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="📊 Peer Group Historical Trends Comparison",
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Revenue ($M)", row=1, col=1)
    fig.update_yaxes(title_text="FCF ($M)", row=1, col=2)
    fig.update_yaxes(title_text="Growth %", row=2, col=1)
    fig.update_yaxes(title_text="Growth %", row=2, col=2)
    
    return fig

def print_trends_summary(historical_data):
    """Print a summary of trends for a ticker"""
    if not historical_data:
        return
    
    ticker = historical_data['ticker']
    trends = historical_data['trends']
    
    print(f"\n🎯 {ticker} - TRENDS SUMMARY:")
    print("="*50)
    
    # Revenue trends
    if 'Revenue_QoQ' in trends:
        qoq = trends['Revenue_QoQ']
        print(f"📈 Revenue QoQ Growth: {qoq:+.1f}%")
    
    if 'Revenue_YoY' in trends:
        yoy = trends['Revenue_YoY']
        print(f"📈 Revenue YoY Growth: {yoy:+.1f}%")
    
    if 'Revenue_Avg_QoQ' in trends:
        avg = trends['Revenue_Avg_QoQ']
        print(f"📈 Revenue Avg QoQ (4Q): {avg:+.1f}%")
    
    # Profitability trends
    if 'Net Income_QoQ' in trends:
        ni_qoq = trends['Net Income_QoQ']
        print(f"💰 Net Income QoQ: {ni_qoq:+.1f}%")
    
    # Cash flow trends
    if 'Free Cash Flow_QoQ' in trends:
        fcf_qoq = trends['Free Cash Flow_QoQ']
        print(f"💵 FCF QoQ Growth: {fcf_qoq:+.1f}%")
    
    if 'Free Cash Flow_YoY' in trends:
        fcf_yoy = trends['Free Cash Flow_YoY']
        print(f"💵 FCF YoY Growth: {fcf_yoy:+.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Historical Financial Trends Analysis')
    parser.add_argument('--ticker', type=str, help='Single ticker to analyze')
    parser.add_argument('--tickers', nargs='+', help='Multiple tickers for peer analysis')
    parser.add_argument('--output', type=str, default='historical_analysis', help='Output filename prefix')
    
    args = parser.parse_args()
    
    if args.ticker:
        # Single ticker analysis
        historical_data = get_historical_metrics(args.ticker)
        if historical_data:
            print_trends_summary(historical_data)
            
            # Create visualization
            fig = create_trends_visualization(historical_data)
            if fig:
                fig.write_html(f"{args.output}_{args.ticker}.html")
                print(f"📊 Chart saved as {args.output}_{args.ticker}.html")
    
    elif args.tickers:
        # Peer group analysis
        peer_data = analyze_peer_historical_trends(args.tickers)
        
        # Print summaries for all
        for ticker, data in peer_data.items():
            print_trends_summary(data)
        
        # Create peer comparison chart
        fig = create_peer_trends_comparison(peer_data)
        if fig:
            fig.write_html(f"{args.output}_peer_comparison.html")
            print(f"📊 Peer comparison chart saved as {args.output}_peer_comparison.html")
    
    else:
        print("Please specify either --ticker or --tickers")

if __name__ == "__main__":
    main()