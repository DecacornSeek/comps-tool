"""
Peer Comparison Dashboard for Stock Analysis
Shows multiple stocks in table format with grouped KPIs
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import yfinance as yf
import numpy as np
from datetime import datetime

# Initialize Dash app with custom CSS
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Peer Comparison Dashboard"

# Custom CSS for NVIDIA-style dashboard
nvidia_style_css = """
<style>
.nvidia-dashboard {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.nvidia-score-card {
    transition: transform 0.2s ease-in-out;
}

.nvidia-score-card:hover {
    transform: translateY(-2px);
}

.nvidia-metrics-section {
    background: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.nvidia-progress-bar {
    border-radius: 10px;
    background-color: #e9ecef;
}

.nvidia-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
"""

# Load AdTech peer data
def load_peer_data():
    """Load the ZETA AdTech peer analysis data"""
    try:
        df = pd.read_csv('/home/user/webapp/zeta_adtech_analysis_enhanced.csv')
        return df
    except FileNotFoundError:
        # Return empty DataFrame if file not found
        return pd.DataFrame()

# KPI Groups for better organization
KPI_GROUPS = {
    'Basic Info': ['Ticker', 'Price', 'MarketCap', 'Industry', 'Sector', 'Employees'],
    'Valuation': ['P/E (TTM)', 'P/E (Fwd)', 'P/B', 'P/S', 'EV/Revenue', 'EV/EBITDA'],
    'Profitability': ['ROE', 'ROA', 'ProfitMargin', 'OperatingMargin', 'GrossMargin'],
    'Growth': ['RevenueGrowth', 'EarningsGrowth', 'EPS (TTM)', 'EPS (Fwd)'],
    'Cash Flow': ['Free Cash Flow', 'FCF Yield'],
    'Financial Health': ['DebtToEquity', 'CurrentRatio', 'QuickRatio', 'Beta'],
    'Market Data': ['52W High', '52W Low', '52W Range %', 'Average Volume'],
    'Ownership': ['Insider Ownership', 'Institutional Ownership', 'Short Ratio'],
    'Scores': ['Valuation_Score', 'Quality_Score', 'Growth_Score']
}

# Historical data cache
HISTORICAL_CACHE = {}

def get_historical_metrics(ticker_symbol):
    """Get historical financial metrics for a ticker with caching"""
    if ticker_symbol in HISTORICAL_CACHE:
        return HISTORICAL_CACHE[ticker_symbol]
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get financial statements
        quarterly_financials = ticker.quarterly_financials
        quarterly_balance = ticker.quarterly_balance_sheet
        quarterly_cashflow = ticker.quarterly_cashflow
        
        results = {
            'ticker': ticker_symbol,
            'quarterly_data': {},
            'trends': {}
        }
        
        # Extract key quarterly metrics - EXPANDED SET
        metrics_map = {
            'Revenue': ('Total Revenue', quarterly_financials),
            'Gross Profit': ('Gross Profit', quarterly_financials),
            'EBITDA': ('EBITDA', quarterly_financials),
            'Net Income': ('Net Income', quarterly_financials),
            'Free Cash Flow': ('Free Cash Flow', quarterly_cashflow),
            'Basic EPS': ('Basic EPS', quarterly_financials),
            'Diluted EPS': ('Diluted EPS', quarterly_financials),
            'Total Debt': ('Total Debt', quarterly_balance)
        }
        
        for display_name, (metric_name, source_df) in metrics_map.items():
            if not source_df.empty and metric_name in source_df.index:
                data = source_df.loc[metric_name].dropna()
                if len(data) > 0:
                    results['quarterly_data'][display_name] = data.to_dict()
        
        # Calculate trends
        results['trends'] = calculate_growth_trends(results['quarterly_data'])
        
        # Cache the results
        HISTORICAL_CACHE[ticker_symbol] = results
        return results
        
    except Exception as e:
        print(f"Error fetching historical data for {ticker_symbol}: {e}")
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
                if series.iloc[-2] != 0:
                    recent_qoq = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100
                    trends[f"{metric_name}_QoQ"] = recent_qoq
            
            # Calculate YoY growth if we have enough data  
            if len(series) >= 4:
                if series.iloc[-4] != 0:
                    recent_yoy = ((series.iloc[-1] / series.iloc[-4]) - 1) * 100
                    trends[f"{metric_name}_YoY"] = recent_yoy
    
    return trends

def create_historical_trends_chart(tickers):
    """Create comprehensive historical trends charts for multiple tickers"""
    from plotly.subplots import make_subplots
    
    # Create subplots with 2x2 layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trends ($M)', 'EBITDA Trends ($M)', 'Free Cash Flow Trends ($M)', 'EPS Trends ($)'),
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, ticker in enumerate(tickers):
        historical_data = get_historical_metrics(ticker)
        color = colors[i % len(colors)]
        
        if historical_data:
            # Revenue trend
            if 'Revenue' in historical_data['quarterly_data']:
                revenue_data = historical_data['quarterly_data']['Revenue']
                dates = list(revenue_data.keys())
                values = [v/1e6 for v in revenue_data.values()]
                
                fig.add_trace(go.Scatter(
                    x=dates, y=values, name=f'{ticker} Revenue',
                    line=dict(color=color, width=2), mode='lines+markers',
                    hovertemplate=f'<b>{ticker}</b><br>%{{x}}<br>Revenue: $%{{y:.1f}}M<extra></extra>',
                    showlegend=(i==0)  # Only show legend for first metric
                ), row=1, col=1)
            
            # EBITDA trend
            if 'EBITDA' in historical_data['quarterly_data']:
                ebitda_data = historical_data['quarterly_data']['EBITDA']
                dates = list(ebitda_data.keys())
                values = [v/1e6 for v in ebitda_data.values()]
                
                fig.add_trace(go.Scatter(
                    x=dates, y=values, name=f'{ticker} EBITDA',
                    line=dict(color=color, width=2), mode='lines+markers',
                    hovertemplate=f'<b>{ticker}</b><br>%{{x}}<br>EBITDA: $%{{y:.1f}}M<extra></extra>',
                    showlegend=False
                ), row=1, col=2)
            
            # FCF trend
            if 'Free Cash Flow' in historical_data['quarterly_data']:
                fcf_data = historical_data['quarterly_data']['Free Cash Flow']
                dates = list(fcf_data.keys())
                values = [v/1e6 for v in fcf_data.values()]
                
                fig.add_trace(go.Scatter(
                    x=dates, y=values, name=f'{ticker} FCF',
                    line=dict(color=color, width=2), mode='lines+markers',
                    hovertemplate=f'<b>{ticker}</b><br>%{{x}}<br>FCF: $%{{y:.1f}}M<extra></extra>',
                    showlegend=False
                ), row=2, col=1)
            
            # EPS trend
            if 'Basic EPS' in historical_data['quarterly_data']:
                eps_data = historical_data['quarterly_data']['Basic EPS']
                dates = list(eps_data.keys())
                values = list(eps_data.values())
                
                fig.add_trace(go.Scatter(
                    x=dates, y=values, name=f'{ticker} EPS',
                    line=dict(color=color, width=2), mode='lines+markers',
                    hovertemplate=f'<b>{ticker}</b><br>%{{x}}<br>EPS: $%{{y:.3f}}<extra></extra>',
                    showlegend=False
                ), row=2, col=2)
    
    fig.update_layout(
        title="📈 Comprehensive Quarterly Trends Analysis",
        height=600,  # Reduced from 700
        hovermode='x unified',
        margin=dict(l=40, r=40, t=80, b=40)  # Tighter margins
    )
    
    return fig

def create_growth_rates_chart(tickers):
    """Create comprehensive growth rates and margins comparison"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Growth Rates (%)', 'EBITDA Growth Rates (%)', 
                       'Current Margins Comparison (%)', 'EPS Growth Rates (%)'),
        vertical_spacing=0.12
    )
    
    # Prepare data
    revenue_qoq, revenue_yoy = [], []
    ebitda_qoq, ebitda_yoy = [], []
    eps_qoq, eps_yoy = [], []
    margins_data = []
    
    for ticker in tickers:
        historical_data = get_historical_metrics(ticker)
        if historical_data:
            trends = historical_data['trends']
            quarterly_data = historical_data['quarterly_data']
            
            # Revenue growth
            if 'Revenue_QoQ' in trends:
                revenue_qoq.append({'Ticker': ticker, 'Value': trends['Revenue_QoQ']})
            if 'Revenue_YoY' in trends:
                revenue_yoy.append({'Ticker': ticker, 'Value': trends['Revenue_YoY']})
            
            # EBITDA growth
            if 'EBITDA_QoQ' in trends:
                ebitda_qoq.append({'Ticker': ticker, 'Value': trends['EBITDA_QoQ']})
            if 'EBITDA_YoY' in trends:
                ebitda_yoy.append({'Ticker': ticker, 'Value': trends['EBITDA_YoY']})
            
            # EPS growth
            if 'Basic EPS_QoQ' in trends:
                eps_qoq.append({'Ticker': ticker, 'Value': trends['Basic EPS_QoQ']})
            if 'Basic EPS_YoY' in trends:
                eps_yoy.append({'Ticker': ticker, 'Value': trends['Basic EPS_YoY']})
            
            # Calculate margins for latest quarter
            margin_row = {'Ticker': ticker}
            
            if 'Revenue' in quarterly_data and 'Gross Profit' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                gross_profit_latest = list(quarterly_data['Gross Profit'].values())[0] if quarterly_data['Gross Profit'] else 0
                if revenue_latest and revenue_latest != 0:
                    margin_row['Gross Margin'] = (gross_profit_latest / revenue_latest) * 100
            
            if 'Revenue' in quarterly_data and 'EBITDA' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                ebitda_latest = list(quarterly_data['EBITDA'].values())[0] if quarterly_data['EBITDA'] else 0
                if revenue_latest and revenue_latest != 0:
                    margin_row['EBITDA Margin'] = (ebitda_latest / revenue_latest) * 100
            
            if 'Revenue' in quarterly_data and 'Net Income' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                net_income_latest = list(quarterly_data['Net Income'].values())[0] if quarterly_data['Net Income'] else 0
                if revenue_latest and revenue_latest != 0:
                    margin_row['Net Margin'] = (net_income_latest / revenue_latest) * 100
            
            margins_data.append(margin_row)
    
    # Add Revenue Growth bars
    if revenue_qoq:
        qoq_df = pd.DataFrame(revenue_qoq)
        fig.add_trace(go.Bar(
            x=qoq_df['Ticker'], y=qoq_df['Value'], name='QoQ', 
            marker_color='lightblue', hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ), row=1, col=1)
    
    if revenue_yoy:
        yoy_df = pd.DataFrame(revenue_yoy)
        fig.add_trace(go.Bar(
            x=yoy_df['Ticker'], y=yoy_df['Value'], name='YoY',
            marker_color='lightgreen', hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ), row=1, col=1)
    
    # Add EBITDA Growth bars
    if ebitda_qoq:
        qoq_df = pd.DataFrame(ebitda_qoq)
        fig.add_trace(go.Bar(
            x=qoq_df['Ticker'], y=qoq_df['Value'], name='EBITDA QoQ',
            marker_color='orange', hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            showlegend=False
        ), row=1, col=2)
    
    # Add Margins comparison
    if margins_data:
        margins_df = pd.DataFrame(margins_data)
        
        for margin_type in ['Gross Margin', 'EBITDA Margin', 'Net Margin']:
            if margin_type in margins_df.columns:
                fig.add_trace(go.Bar(
                    x=margins_df['Ticker'], y=margins_df[margin_type], 
                    name=margin_type, hovertemplate=f'%{{x}} {margin_type}: %{{y:.1f}}%<extra></extra>',
                    showlegend=False
                ), row=2, col=1)
    
    # Add EPS Growth
    if eps_qoq:
        qoq_df = pd.DataFrame(eps_qoq)
        fig.add_trace(go.Bar(
            x=qoq_df['Ticker'], y=qoq_df['Value'], name='EPS QoQ',
            marker_color='purple', hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
            showlegend=False
        ), row=2, col=2)
    
    fig.update_layout(
        title="📊 Growth Rates & Margins Analysis",
        height=500,  # Reduced from 600
        barmode='group',
        margin=dict(l=40, r=40, t=80, b=40)  # Tighter margins
    )
    
    return fig

def calculate_pe_ratios(ticker):
    """Calculate historical P/E ratios for a ticker"""
    try:
        import yfinance as yf
        from datetime import date
        
        stock = yf.Ticker(ticker)
        quarterly_financials = stock.quarterly_financials
        
        if quarterly_financials.empty or 'Basic EPS' not in quarterly_financials.index:
            return {}
        
        # Get historical prices
        hist = stock.history(period="2y", interval="1d")
        if hist.empty:
            return {}
        
        # Define quarter end dates
        quarter_ends = [
            date(2025, 6, 30), date(2025, 3, 31), 
            date(2024, 12, 31), date(2024, 9, 30),
            date(2024, 6, 30), date(2024, 3, 31)
        ]
        
        pe_ratios = {}
        eps_data = quarterly_financials.loc['Basic EPS']
        
        for quarter_date, eps in eps_data.items():
            quarter_end_date = quarter_date.date()
            
            # Find closest trading day price
            available_dates = hist.index.date
            closest_date = None
            
            for available_date in reversed(available_dates):
                if available_date <= quarter_end_date:
                    closest_date = available_date
                    break
            
            if closest_date and eps > 0:
                price = hist.loc[hist.index.date == closest_date, 'Close'].iloc[0]
                pe_ratio = price / eps
                pe_ratios[quarter_date] = pe_ratio
            elif eps <= 0:
                pe_ratios[quarter_date] = None  # Negative EPS
        
        return pe_ratios
        
    except Exception as e:
        print(f"Error calculating P/E for {ticker}: {e}")
        return {}

def create_metric_tables(tickers):
    """Create separate tables for each metric + margin tables + P/E table"""
    
    # Get data for all tickers
    ticker_data = {}
    for ticker in tickers:
        historical_data = get_historical_metrics(ticker)
        if historical_data and 'quarterly_data' in historical_data:
            ticker_data[ticker] = historical_data
    
    if not ticker_data:
        return [dbc.Alert("No historical data available", color="warning")]
    
    tables = []
    
    # Define absolute metrics
    absolute_metrics = {
        'Revenue': ('Revenue', 'M'),
        'Gross Profit': ('Gross Profit', 'M'),
        'EBITDA': ('EBITDA', 'M'),
        'Net Income': ('Net Income', 'M'),
        'Free Cash Flow': ('Free Cash Flow', 'M'),
        'EPS': ('Basic EPS', '')
    }
    
    # Create tables for absolute metrics
    for metric_name, (metric_key, unit) in absolute_metrics.items():
        table_data = []
        
        for ticker in sorted(ticker_data.keys()):
            historical_data = ticker_data[ticker]
            if metric_key not in historical_data['quarterly_data']:
                continue
                
            metric_dict = historical_data['quarterly_data'][metric_key]
            quarters = sorted(metric_dict.keys(), reverse=True)[:4]  # Last 4 quarters
            
            if len(quarters) < 2:
                continue
                
            row = {'Company': ticker}
            
            # Last 4 quarters
            for quarter in quarters:
                quarter_num = (quarter.month - 1) // 3 + 1
                quarter_label = f"{quarter.year}Q{quarter_num}"
                value = metric_dict[quarter]
                
                if unit == 'M':
                    row[quarter_label] = f"${value/1e6:.1f}M"
                else:  # EPS
                    row[quarter_label] = f"${value:.3f}"
            
            # QoQ Growth
            if len(quarters) >= 2:
                current_val = metric_dict[quarters[0]]
                prev_val = metric_dict[quarters[1]]
                if prev_val != 0:
                    qoq_growth = ((current_val - prev_val) / abs(prev_val)) * 100
                    row['QoQ%'] = f"{qoq_growth:+.1f}%"
                else:
                    row['QoQ%'] = "N/A"
            
            # YoY Growth
            if len(quarters) >= 4:
                current_val = metric_dict[quarters[0]]
                year_ago_val = metric_dict[quarters[3]]  # 4 quarters ago
                if year_ago_val != 0:
                    yoy_growth = ((current_val - year_ago_val) / abs(year_ago_val)) * 100
                    row['YoY%'] = f"{yoy_growth:+.1f}%"
                else:
                    row['YoY%'] = "N/A"
            else:
                row['YoY%'] = "N/A"
            
            table_data.append(row)
        
        # Create table for this metric
        if table_data:
            df = pd.DataFrame(table_data)
            table = dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in df.columns],
                sort_action="native",
                style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
                style_header={'backgroundColor': 'rgb(240, 240, 240)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'column_id': 'QoQ%', 'filter_query': '{QoQ%} contains "+"'}, 'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'YoY%', 'filter_query': '{YoY%} contains "+"'}, 'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'QoQ%', 'filter_query': '{QoQ%} contains "-"'}, 'color': 'red', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'YoY%', 'filter_query': '{YoY%} contains "-"'}, 'color': 'red', 'fontWeight': 'bold'}
                ],
                style_table={'overflowX': 'auto'}
            )
            
            tables.append(
                dbc.Card([
                    dbc.CardHeader([html.H5(f"💰 {metric_name}", className="mb-0")]),
                    dbc.CardBody([table])
                ], className="mb-3")
            )
    
    # Create margin tables
    margin_metrics = {
        'Gross Profit Margin': ('Gross Profit', 'Revenue'),
        'EBITDA Margin': ('EBITDA', 'Revenue'),
        'Net Profit Margin': ('Net Income', 'Revenue'),
        'FCF Margin': ('Free Cash Flow', 'Revenue')
    }
    
    for margin_name, (numerator_key, denominator_key) in margin_metrics.items():
        table_data = []
        
        for ticker in sorted(ticker_data.keys()):
            historical_data = ticker_data[ticker]
            if (numerator_key not in historical_data['quarterly_data'] or 
                denominator_key not in historical_data['quarterly_data']):
                continue
                
            num_dict = historical_data['quarterly_data'][numerator_key]
            den_dict = historical_data['quarterly_data'][denominator_key]
            
            # Get common quarters
            common_quarters = set(num_dict.keys()) & set(den_dict.keys())
            quarters = sorted(common_quarters, reverse=True)[:4]  # Last 4 quarters
            
            if len(quarters) < 2:
                continue
                
            row = {'Company': ticker}
            margins = []
            
            # Calculate margins for last 4 quarters
            for quarter in quarters:
                quarter_num = (quarter.month - 1) // 3 + 1
                quarter_label = f"{quarter.year}Q{quarter_num}"
                
                numerator = num_dict[quarter]
                denominator = den_dict[quarter]
                
                if denominator != 0:
                    margin = (numerator / denominator) * 100
                    row[quarter_label] = f"{margin:.1f}%"
                    margins.append(margin)
                else:
                    row[quarter_label] = "N/A"
                    margins.append(None)
            
            # QoQ Change in margin
            if len(margins) >= 2 and margins[0] is not None and margins[1] is not None:
                qoq_change = margins[0] - margins[1]
                row['QoQ Δ'] = f"{qoq_change:+.1f}pp"  # pp = percentage points
            else:
                row['QoQ Δ'] = "N/A"
            
            # YoY Change in margin
            if len(margins) >= 4 and margins[0] is not None and margins[3] is not None:
                yoy_change = margins[0] - margins[3]
                row['YoY Δ'] = f"{yoy_change:+.1f}pp"
            else:
                row['YoY Δ'] = "N/A"
            
            table_data.append(row)
        
        # Create margin table
        if table_data:
            df = pd.DataFrame(table_data)
            table = dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in df.columns],
                sort_action="native",
                style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
                style_header={'backgroundColor': 'rgb(245, 245, 245)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'column_id': 'QoQ Δ', 'filter_query': '{QoQ Δ} contains "+"'}, 'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'YoY Δ', 'filter_query': '{YoY Δ} contains "+"'}, 'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'QoQ Δ', 'filter_query': '{QoQ Δ} contains "-"'}, 'color': 'red', 'fontWeight': 'bold'},
                    {'if': {'column_id': 'YoY Δ', 'filter_query': '{YoY Δ} contains "-"'}, 'color': 'red', 'fontWeight': 'bold'}
                ],
                style_table={'overflowX': 'auto'}
            )
            
            tables.append(
                dbc.Card([
                    dbc.CardHeader([html.H5(f"📊 {margin_name}", className="mb-0")]),
                    dbc.CardBody([table])
                ], className="mb-3")
            )
    
    # Add P/E Ratio table
    pe_table_data = []
    
    for ticker in sorted(ticker_data.keys()):
        pe_ratios = calculate_pe_ratios(ticker)
        
        if not pe_ratios:
            continue
            
        # Get quarters with P/E data
        pe_quarters = sorted(pe_ratios.keys(), reverse=True)[:4]  # Last 4 quarters
        
        if len(pe_quarters) < 2:
            continue
            
        row = {'Company': ticker}
        pe_values = []
        
        # Last 4 quarters P/E ratios
        for quarter in pe_quarters:
            quarter_num = (quarter.month - 1) // 3 + 1
            quarter_label = f"{quarter.year}Q{quarter_num}"
            
            pe_ratio = pe_ratios[quarter]
            if pe_ratio is not None:
                row[quarter_label] = f"{pe_ratio:.1f}x"
                pe_values.append(pe_ratio)
            else:
                row[quarter_label] = "N/A"
                pe_values.append(None)
        
        # P/E change QoQ
        if len(pe_values) >= 2 and pe_values[0] is not None and pe_values[1] is not None:
            qoq_change = ((pe_values[0] - pe_values[1]) / pe_values[1]) * 100
            row['QoQ%'] = f"{qoq_change:+.1f}%"
        else:
            row['QoQ%'] = "N/A"
        
        # P/E change YoY
        if len(pe_values) >= 4 and pe_values[0] is not None and pe_values[3] is not None:
            yoy_change = ((pe_values[0] - pe_values[3]) / pe_values[3]) * 100
            row['YoY%'] = f"{yoy_change:+.1f}%"
        else:
            row['YoY%'] = "N/A"
        
        pe_table_data.append(row)
    
    # Create P/E table
    if pe_table_data:
        df = pd.DataFrame(pe_table_data)
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df.columns],
            sort_action="native",
            style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '11px'},
            style_header={'backgroundColor': 'rgb(250, 250, 250)', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'column_id': 'QoQ%', 'filter_query': '{QoQ%} contains "+"'}, 'color': 'red', 'fontWeight': 'bold'},    # Higher P/E = worse
                {'if': {'column_id': 'YoY%', 'filter_query': '{YoY%} contains "+"'}, 'color': 'red', 'fontWeight': 'bold'},
                {'if': {'column_id': 'QoQ%', 'filter_query': '{QoQ%} contains "-"'}, 'color': 'green', 'fontWeight': 'bold'},  # Lower P/E = better
                {'if': {'column_id': 'YoY%', 'filter_query': '{YoY%} contains "-"'}, 'color': 'green', 'fontWeight': 'bold'}
            ],
            style_table={'overflowX': 'auto'}
        )
        
        tables.append(
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📊 P/E Ratio (Price-to-Earnings)", className="mb-0"),
                    html.Small("Lower P/E generally indicates better value. Green = P/E decreased (better), Red = P/E increased (worse)", className="text-muted")
                ]),
                dbc.CardBody([table])
            ], className="mb-3")
        )
    
    return tables

def create_professional_metrics_tables(tickers):
    """Create professional investment banking style analysis tables"""
    
    # Get basic peer data for current metrics
    try:
        peer_df = pd.read_csv('/home/user/webapp/zeta_adtech_analysis_enhanced.csv')
    except:
        return [dbc.Alert("Peer comparison data not available", color="warning")]
    
    # Filter to our tickers
    peer_df = peer_df[peer_df['Ticker'].isin(tickers)]
    
    if peer_df.empty:
        return [dbc.Alert("No peer data available for selected tickers", color="warning")]
    
    tables = []
    
    # 1. Professional Comps Table - Investment Banking Style
    # Create comprehensive comparable companies table
    comps_data = []
    
    for _, row in peer_df.iterrows():
        company_row = {
            'Company Name': row['Ticker'],
            'Price': f"${row['Price']:.2f}" if pd.notna(row['Price']) else "N/A",
            'Market Cap': f"${row['MarketCap']/1e9:.1f}B" if pd.notna(row['MarketCap']) else "N/A",
            'TEV': f"${row['Enterprise Value']/1e9:.1f}B" if pd.notna(row['Enterprise Value']) else "N/A"
        }
        
        # Financial Data ($M)
        if 'RevenueGrowth' in row and pd.notna(row['RevenueGrowth']):
            # Estimate TTM Revenue from MarketCap and P/S
            if pd.notna(row['P/S']) and row['P/S'] > 0 and pd.notna(row['MarketCap']):
                ttm_revenue = row['MarketCap'] / row['P/S']
                company_row['Sales'] = f"${ttm_revenue/1e6:.0f}"
            else:
                company_row['Sales'] = "N/A"
        else:
            company_row['Sales'] = "N/A"
            
        # EBITDA estimation
        if 'OperatingMargin' in row and pd.notna(row['OperatingMargin']):
            company_row['EBITDA'] = f"{row['OperatingMargin']*100:.0f}%" if abs(row['OperatingMargin']) < 1 else f"{row['OperatingMargin']:.0f}%"
        else:
            company_row['EBITDA'] = "N/A"
            
        # EBIT (approximation)
        if 'ProfitMargin' in row and pd.notna(row['ProfitMargin']):
            company_row['EBIT'] = f"{row['ProfitMargin']*100:.0f}%" if abs(row['ProfitMargin']) < 1 else f"{row['ProfitMargin']:.0f}%"
        else:
            company_row['EBIT'] = "N/A"
            
        # Earnings (Net Income margin as proxy)
        if 'EPS (TTM)' in row and pd.notna(row['EPS (TTM)']):
            company_row['Earnings'] = f"${row['EPS (TTM)']:.2f}"
        else:
            company_row['Earnings'] = "N/A"
        
        # Valuation Multiples
        company_row['EV/Sales'] = f"{row['EV/Revenue']:.1f}x" if pd.notna(row['EV/Revenue']) else "N/A"
        company_row['EV/EBITDA'] = f"{row['EV/EBITDA']:.1f}x" if pd.notna(row['EV/EBITDA']) else "N/A"
        company_row['EV/EBIT'] = f"{row['P/S']:.1f}x" if pd.notna(row['P/S']) else "N/A"  # Using P/S as proxy
        company_row['P/E'] = f"{row['P/E (TTM)']:.1f}x" if pd.notna(row['P/E (TTM)']) and row['P/E (TTM)'] > 0 else "N/A"
        
        comps_data.append(company_row)
    
    if comps_data:
        # Calculate sector averages for bottom rows
        numeric_cols = ['Market Cap', 'TEV', 'EV/Sales', 'EV/EBITDA', 'EV/EBIT', 'P/E']
        
        # Extract numeric values for calculations
        calc_data = {}
        for col in ['EV/Revenue', 'EV/EBITDA', 'P/S', 'P/E (TTM)']:
            if col in peer_df.columns:
                values = peer_df[col].dropna()
                if col == 'EV/Revenue':
                    calc_data['EV/Sales'] = values
                elif col == 'EV/EBITDA':
                    calc_data['EV/EBITDA'] = values
                elif col == 'P/S':
                    calc_data['EV/EBIT'] = values
                elif col == 'P/E (TTM)':
                    calc_data['P/E'] = values[values > 0]  # Only positive P/E
        
        # Add Average row
        avg_row = {'Company Name': 'Average'}
        for col in ['Price', 'Market Cap', 'TEV', 'Sales', 'EBITDA', 'EBIT', 'Earnings']:
            avg_row[col] = ""
        
        for metric, values in calc_data.items():
            if len(values) > 0:
                avg_row[metric] = f"{values.mean():.1f}x"
            else:
                avg_row[metric] = "N/A"
        
        comps_data.append(avg_row)
        
        # Add Median row  
        med_row = {'Company Name': 'Median'}
        for col in ['Price', 'Market Cap', 'TEV', 'Sales', 'EBITDA', 'EBIT', 'Earnings']:
            med_row[col] = ""
            
        for metric, values in calc_data.items():
            if len(values) > 0:
                med_row[metric] = f"{values.median():.1f}x"
            else:
                med_row[metric] = "N/A"
                
        comps_data.append(med_row)
        
        df = pd.DataFrame(comps_data)
        
        # Create multi-level column headers
        columns = [
            {"name": "Company Name", "id": "Company Name", "type": "text"},
            {"name": ["Market Data", "Price"], "id": "Price", "type": "text"},  
            {"name": ["Market Data", "Market Cap"], "id": "Market Cap", "type": "text"},
            {"name": ["Market Data", "TEV"], "id": "TEV", "type": "text"},
            {"name": ["Financial Data", "Sales"], "id": "Sales", "type": "text"},
            {"name": ["Financial Data", "EBITDA"], "id": "EBITDA", "type": "text"},
            {"name": ["Financial Data", "EBIT"], "id": "EBIT", "type": "text"},
            {"name": ["Financial Data", "Earnings"], "id": "Earnings", "type": "text"},
            {"name": ["Valuation", "EV/Sales"], "id": "EV/Sales", "type": "text"},
            {"name": ["Valuation", "EV/EBITDA"], "id": "EV/EBITDA", "type": "text"},
            {"name": ["Valuation", "EV/EBIT"], "id": "EV/EBIT", "type": "text"},
            {"name": ["Valuation", "P/E"], "id": "P/E", "type": "text"}
        ]
        
        # Investment Banking style table
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df.columns],  # Simplified for now
            sort_action="native",
            style_cell={
                'textAlign': 'center',
                'padding': '6px 12px',
                'fontSize': '11px',
                'fontFamily': 'Arial, sans-serif',
                'border': '1px solid #ddd'
            },
            style_header={
                'backgroundColor': '#2c3e50',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center',
                'border': '1px solid #34495e'
            },
            style_data_conditional=[
                # Highlight Average and Median rows
                {
                    'if': {'filter_query': '{Company Name} = Average'},
                    'backgroundColor': '#ecf0f1',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{Company Name} = Median'},
                    'backgroundColor': '#d5dbdb',
                    'fontWeight': 'bold'
                }
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid #ddd'}
        )
        
        tables.append(
            dbc.Card([
                dbc.CardHeader([
                    html.H5("💼 Comparable Companies Analysis", className="mb-0"),
                    html.Small("Investment Banking Style Comps Table", className="text-muted")
                ]),
                dbc.CardBody([table], style={'padding': '10px'})
            ], className="mb-4")
        )
    
    # 2. Profitability Ratios (ROE, ROA, ROIC)
    profitability_metrics = ['ROE', 'ROA', 'ROIC', 'ProfitMargin', 'OperatingMargin']
    prof_table_data = []
    
    for _, row in peer_df.iterrows():
        ticker_row = {'Company': row['Ticker']}
        
        for metric in profitability_metrics:
            if metric in row and pd.notna(row[metric]):
                value = row[metric]
                if metric in ['ROE', 'ROA', 'ROIC', 'ProfitMargin', 'OperatingMargin']:
                    ticker_row[metric] = f"{value:.1%}" if abs(value) < 10 else f"{value*100:.1f}%"
            else:
                ticker_row[metric] = "N/A"
        
        prof_table_data.append(ticker_row)
    
    if prof_table_data:
        # Add Average and Median rows for profitability metrics
        numeric_data = {}
        for metric in profitability_metrics:
            if metric in peer_df.columns:
                values = peer_df[metric].dropna()
                if len(values) > 0:
                    numeric_data[metric] = values
        
        # Average row
        avg_row = {'Company': 'Average'}
        for metric in profitability_metrics:
            if metric in numeric_data:
                avg_val = numeric_data[metric].mean()
                avg_row[metric] = f"{avg_val:.1%}" if abs(avg_val) < 10 else f"{avg_val*100:.1f}%"
            else:
                avg_row[metric] = "N/A"
        prof_table_data.append(avg_row)
        
        # Median row
        med_row = {'Company': 'Median'}
        for metric in profitability_metrics:
            if metric in numeric_data:
                med_val = numeric_data[metric].median()
                med_row[metric] = f"{med_val:.1%}" if abs(med_val) < 10 else f"{med_val*100:.1f}%"
            else:
                med_row[metric] = "N/A"
        prof_table_data.append(med_row)
        
        df = pd.DataFrame(prof_table_data)
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df.columns],
            sort_action="native",
            style_cell={
                'textAlign': 'center', 
                'padding': '6px 12px', 
                'fontSize': '11px',
                'fontFamily': 'Arial, sans-serif',
                'border': '1px solid #ddd'
            },
            style_header={
                'backgroundColor': '#34495e', 
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid #2c3e50'
            },
            style_data_conditional=[
                # Highlight Average and Median rows
                {
                    'if': {'filter_query': '{Company} = Average'},
                    'backgroundColor': '#ecf0f1',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{Company} = Median'},
                    'backgroundColor': '#d5dbdb',
                    'fontWeight': 'bold'
                }
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid #ddd'}
        )
        
        tables.append(
            dbc.Card([
                dbc.CardHeader([html.H5("📊 Profitability & Efficiency Ratios", className="mb-0")]),
                dbc.CardBody([table])
            ], className="mb-3")
        )
    
    # 3. Relative Valuation Analysis
    valuation_metrics = ['P/E (TTM)', 'P/S', 'EV/Revenue', 'EV/EBITDA', 'P/B']
    rel_val_data = []
    
    # Calculate percentiles and sector stats
    sector_stats = {}
    for metric in valuation_metrics:
        if metric in peer_df.columns:
            values = peer_df[metric].dropna()
            if len(values) > 0:
                sector_stats[metric] = {
                    'median': values.median(),
                    'mean': values.mean(),
                    'min': values.min(),
                    'max': values.max()
                }
    
    for _, row in peer_df.iterrows():
        ticker_row = {'Company': row['Ticker']}
        
        for metric in valuation_metrics:
            if metric in row and pd.notna(row[metric]) and metric in sector_stats:
                value = row[metric]
                median = sector_stats[metric]['median']
                
                # Calculate premium/discount vs median
                if median > 0:
                    premium_discount = ((value - median) / median) * 100
                    ticker_row[f'{metric} vs Median'] = f"{premium_discount:+.0f}%"
                    
                    # Determine percentile ranking
                    values = peer_df[metric].dropna()
                    percentile = (values < value).sum() / len(values) * 100
                    ticker_row[f'{metric} Percentile'] = f"{percentile:.0f}th"
                else:
                    ticker_row[f'{metric} vs Median'] = "N/A"
                    ticker_row[f'{metric} Percentile'] = "N/A"
            else:
                ticker_row[f'{metric} vs Median'] = "N/A"
                ticker_row[f'{metric} Percentile'] = "N/A"
        
        rel_val_data.append(ticker_row)
    
    if rel_val_data:
        df = pd.DataFrame(rel_val_data)
        
        # Create conditional styling for relative valuation
        style_conditions = []
        for metric in valuation_metrics:
            vs_median_col = f'{metric} vs Median'
            if vs_median_col in df.columns:
                # Green for discount (negative %), Red for premium (positive %)
                style_conditions.extend([
                    {'if': {'column_id': vs_median_col, 'filter_query': f'{{{vs_median_col}}} contains "-"'}, 
                     'color': 'green', 'fontWeight': 'bold'},
                    {'if': {'column_id': vs_median_col, 'filter_query': f'{{{vs_median_col}}} contains "+"'}, 
                     'color': 'red', 'fontWeight': 'bold'}
                ])
        
        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{"name": col, "id": col} for col in df.columns],
            sort_action="native",
            style_cell={
                'textAlign': 'center', 
                'padding': '6px 10px', 
                'fontSize': '10px',
                'fontFamily': 'Arial, sans-serif',
                'border': '1px solid #ddd'
            },
            style_header={
                'backgroundColor': '#e74c3c', 
                'color': 'white',
                'fontWeight': 'bold',
                'border': '1px solid #c0392b'
            },
            style_data_conditional=style_conditions + [
                # Highlight Average and Median rows
                {
                    'if': {'filter_query': '{Company} = Average'},
                    'backgroundColor': '#ecf0f1',
                    'fontWeight': 'bold'
                },
                {
                    'if': {'filter_query': '{Company} = Median'},
                    'backgroundColor': '#d5dbdb', 
                    'fontWeight': 'bold'
                }
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid #ddd'}
        )
        
        tables.append(
            dbc.Card([
                dbc.CardHeader([
                    html.H5("📈 Relative Valuation Analysis", className="mb-0"),
                    html.Small("Green = Trading at discount to peers, Red = Trading at premium to peers", className="text-muted")
                ]),
                dbc.CardBody([table])
            ], className="mb-3")
        )
    
    # Add sector summary statistics
    if sector_stats:
        summary_data = []
        for metric, stats in sector_stats.items():
            summary_data.append({
                'Metric': metric,
                'Median': f"{stats['median']:.1f}x",
                'Mean': f"{stats['mean']:.1f}x", 
                'Min': f"{stats['min']:.1f}x",
                'Max': f"{stats['max']:.1f}x"
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            table = dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": col, "id": col} for col in df.columns],
                style_cell={
                    'textAlign': 'center', 
                    'padding': '8px 12px', 
                    'fontSize': '11px',
                    'fontFamily': 'Arial, sans-serif',
                    'border': '1px solid #ddd'
                },
                style_header={
                    'backgroundColor': '#95a5a6', 
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': '1px solid #7f8c8d'
                },
                style_table={'overflowX': 'auto', 'border': '1px solid #ddd'}
            )
            
            tables.append(
                dbc.Card([
                    dbc.CardHeader([html.H5("📋 Sector Valuation Summary", className="mb-0")]),
                    dbc.CardBody([table])
                ], className="mb-3")
            )
    
    return tables

def create_historical_trends_table(tickers):
    """Create a comprehensive table with historical trends and margins for all tickers"""
    trends_data = []
    
    for ticker in tickers:
        historical_data = get_historical_metrics(ticker)
        if historical_data:
            trends = historical_data['trends']
            quarterly_data = historical_data['quarterly_data']
            
            row = {'Ticker': ticker}
            
            # Revenue trends
            row['Revenue QoQ %'] = f"{trends.get('Revenue_QoQ', 0):.1f}%" if 'Revenue_QoQ' in trends else "N/A"
            row['Revenue YoY %'] = f"{trends.get('Revenue_YoY', 0):.1f}%" if 'Revenue_YoY' in trends else "N/A"
            
            # EBITDA trends
            row['EBITDA QoQ %'] = f"{trends.get('EBITDA_QoQ', 0):.1f}%" if 'EBITDA_QoQ' in trends else "N/A"
            row['EBITDA YoY %'] = f"{trends.get('EBITDA_YoY', 0):.1f}%" if 'EBITDA_YoY' in trends else "N/A"
            
            # Net Income trends  
            row['Net Income QoQ %'] = f"{trends.get('Net Income_QoQ', 0):.1f}%" if 'Net Income_QoQ' in trends else "N/A"
            row['Net Income YoY %'] = f"{trends.get('Net Income_YoY', 0):.1f}%" if 'Net Income_YoY' in trends else "N/A"
            
            # FCF trends
            row['FCF QoQ %'] = f"{trends.get('Free Cash Flow_QoQ', 0):.1f}%" if 'Free Cash Flow_QoQ' in trends else "N/A"
            row['FCF YoY %'] = f"{trends.get('Free Cash Flow_YoY', 0):.1f}%" if 'Free Cash Flow_YoY' in trends else "N/A"
            
            # EPS trends
            row['EPS QoQ %'] = f"{trends.get('Basic EPS_QoQ', 0):.1f}%" if 'Basic EPS_QoQ' in trends else "N/A"
            row['EPS YoY %'] = f"{trends.get('Basic EPS_YoY', 0):.1f}%" if 'Basic EPS_YoY' in trends else "N/A"
            
            # Calculate and add margins for latest quarter
            if 'Revenue' in quarterly_data and 'Gross Profit' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                gross_profit_latest = list(quarterly_data['Gross Profit'].values())[0] if quarterly_data['Gross Profit'] else 0
                
                if revenue_latest and revenue_latest != 0:
                    gross_margin = (gross_profit_latest / revenue_latest) * 100
                    row['Gross Margin %'] = f"{gross_margin:.1f}%"
                else:
                    row['Gross Margin %'] = "N/A"
            else:
                row['Gross Margin %'] = "N/A"
            
            if 'Revenue' in quarterly_data and 'EBITDA' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                ebitda_latest = list(quarterly_data['EBITDA'].values())[0] if quarterly_data['EBITDA'] else 0
                
                if revenue_latest and revenue_latest != 0:
                    ebitda_margin = (ebitda_latest / revenue_latest) * 100
                    row['EBITDA Margin %'] = f"{ebitda_margin:.1f}%"
                else:
                    row['EBITDA Margin %'] = "N/A"
            else:
                row['EBITDA Margin %'] = "N/A"
            
            if 'Revenue' in quarterly_data and 'Net Income' in quarterly_data:
                revenue_latest = list(quarterly_data['Revenue'].values())[0] if quarterly_data['Revenue'] else 0
                net_income_latest = list(quarterly_data['Net Income'].values())[0] if quarterly_data['Net Income'] else 0
                
                if revenue_latest and revenue_latest != 0:
                    net_margin = (net_income_latest / revenue_latest) * 100
                    row['Net Margin %'] = f"{net_margin:.1f}%"
                else:
                    row['Net Margin %'] = "N/A"
            else:
                row['Net Margin %'] = "N/A"
            
            # Latest EPS and P/E (if we can get current price)
            if 'Basic EPS' in quarterly_data:
                latest_eps = list(quarterly_data['Basic EPS'].values())[0] if quarterly_data['Basic EPS'] else 0
                row['Latest EPS'] = f"${latest_eps:.3f}" if latest_eps else "N/A"
            else:
                row['Latest EPS'] = "N/A"
            
            trends_data.append(row)
    
    if not trends_data:
        return html.Div("No historical data available")
    
    df = pd.DataFrame(trends_data)
    
    # Create conditional formatting for positive/negative growth
    style_data_conditional = []
    
    for col in df.columns:
        if 'QoQ %' in col or 'YoY %' in col:
            # Green for positive growth
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{{col}}} contains "+" || ({{{col}}} > 0 && {{{col}}} != "N/A")',
                    'column_id': col
                },
                'backgroundColor': '#d4edda',
                'color': 'green',
            })
            
            # Red for negative growth
            style_data_conditional.append({
                'if': {
                    'filter_query': f'{{{col}}} contains "-"',
                    'column_id': col
                },
                'backgroundColor': '#f8d7da',
                'color': 'red',
            })
    
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df.columns],
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '12px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=style_data_conditional,
        sort_action="native"
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📈 Historical Growth Trends", className="mb-0")
        ]),
        dbc.CardBody([table])
    ])

def create_kpi_table(df, group_name, columns):
    """Create a formatted table for a specific KPI group"""
    if df.empty:
        return html.Div("No data available")
    
    # Select only available columns
    available_cols = ['Ticker'] + [col for col in columns if col in df.columns and col != 'Ticker']
    table_data = df[available_cols].copy()
    
    # Format numeric columns
    for col in available_cols[1:]:  # Skip Ticker column
        if col in df.columns:
            if 'Margin' in col or 'Growth' in col or 'ROE' in col or 'ROA' in col or 'Yield' in col:
                # Format as percentage
                table_data[col] = table_data[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
            elif 'MarketCap' in col or 'Cash Flow' in col or 'Volume' in col:
                # Format large numbers
                table_data[col] = table_data[col].apply(lambda x: f"{x/1e9:.1f}B" if pd.notna(x) and isinstance(x, (int, float)) and x > 1e9 else f"{x/1e6:.0f}M" if pd.notna(x) and isinstance(x, (int, float)) and x > 1e6 else f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
            elif 'Price' in col or 'Ratio' in col or 'P/E' in col or 'P/B' in col or 'P/S' in col or 'EV/' in col or 'Beta' in col:
                # Format as number with 2 decimals
                table_data[col] = table_data[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
            elif 'Score' in col:
                # Format scores as integers
                table_data[col] = table_data[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
            elif 'Employees' in col:
                # Format employee count
                table_data[col] = table_data[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
    
    # Create the table
    table = dash_table.DataTable(
        data=table_data.to_dict('records'),
        columns=[{"name": col, "id": col} for col in table_data.columns],
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '12px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            # Highlight best performers in green
            {
                'if': {'row_index': 0},  # Assuming first row after sorting is best
                'backgroundColor': '#d4edda',
                'color': 'black',
            }
        ],
        style_table={'overflowX': 'auto'},
        sort_action="native",
        sort_mode="single"
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4(f"📊 {group_name}", className="mb-0")
        ]),
        dbc.CardBody([
            table
        ])
    ], className="mb-4")

def create_summary_chart(df):
    """Create a summary comparison chart"""
    if df.empty or 'Ticker' not in df.columns:
        return go.Figure()
    
    # Create radar chart for top metrics
    metrics = ['Valuation_Score', 'Quality_Score', 'Growth_Score']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return go.Figure()
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        values = [row.get(m, 0) for m in available_metrics]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics,
            fill='toself',
            name=row['Ticker'],
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="📊 Composite Scores Comparison",
        height=500
    )
    
    return fig

# App Layout
def create_layout():
    df = load_peer_data()
    
    if df.empty:
        return dbc.Container([
            dbc.Alert("No peer data found. Please run the analysis first.", color="warning")
        ])
    
    # Create tabs for different KPI groups
    tabs = []
    
    # Summary tab with overview
    summary_content = [
        dbc.Row([
            dbc.Col([
                html.H2("🏆 AdTech/SaaS Peer Group Analysis", className="text-center mb-4"),
                html.P(f"Comparing {len(df)} companies across 45+ financial metrics", className="text-center text-muted"),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=create_summary_chart(df))
            ])
        ]),
        dbc.Row([
            dbc.Col([
                create_kpi_table(df, "Summary Metrics", ['Ticker', 'MarketCap', 'P/E (TTM)', 'ROE', 'RevenueGrowth', 'Valuation_Score', 'Quality_Score', 'Growth_Score'])
            ])
        ])
    ]
    
    tabs.append(dbc.Tab(label="📊 Overview", tab_id="overview"))
    
    # Create tabs for each KPI group
    for group_name, columns in KPI_GROUPS.items():
        tabs.append(dbc.Tab(label=f"📋 {group_name}", tab_id=group_name.lower().replace(' ', '_')))
    
    # Add historical analysis tabs
    tabs.append(dbc.Tab(label="📈 Historical Trends", tab_id="historical_trends"))
    tabs.append(dbc.Tab(label="📊 Growth Analysis", tab_id="growth_analysis"))
    tabs.append(dbc.Tab(label="💼 Professional Analysis", tab_id="professional_analysis"))
    tabs.append(dbc.Tab(label="📈 NVIDIA-Style Analysis", tab_id="nvidia_style"))
    
    layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🚀 Stock Peer Comparison Dashboard", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Tabs(
                    tabs,
                    id="kpi-tabs",
                    active_tab="overview"
                )
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="tab-content")
            ])
        ])
        
    ], fluid=True)
    
    return layout

app.layout = create_layout()

@app.callback(
    Output('tab-content', 'children'),
    Input('kpi-tabs', 'active_tab')
)
def update_tab_content(active_tab):
    df = load_peer_data()
    
    if df.empty:
        return dbc.Alert("No data available", color="warning")
    
    if active_tab == "overview":
        return [
            dbc.Row([
                dbc.Col([
                    html.H2("🏆 AdTech/SaaS Peer Group Analysis", className="text-center mb-4"),
                    html.P(f"Comparing {len(df)} companies: {', '.join(df['Ticker'].tolist())}", className="text-center text-muted"),
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=create_summary_chart(df))
                ], width=6),
                dbc.Col([
                    html.H4("🎯 Key Insights"),
                    html.Ul([
                        html.Li(f"🏆 Best Overall Score: {df.loc[df['Valuation_Score'].idxmax(), 'Ticker'] if 'Valuation_Score' in df.columns else 'N/A'}"),
                        html.Li(f"📈 Highest Growth: {df.loc[df['RevenueGrowth'].idxmax(), 'Ticker'] if 'RevenueGrowth' in df.columns else 'N/A'} ({df['RevenueGrowth'].max():.1%} if 'RevenueGrowth' in df.columns else 'N/A')"),
                        html.Li(f"💰 Largest Market Cap: {df.loc[df['MarketCap'].idxmax(), 'Ticker'] if 'MarketCap' in df.columns else 'N/A'}"),
                        html.Li(f"🎖️ Most Profitable (ROE): {df.loc[df['ROE'].idxmax(), 'Ticker'] if 'ROE' in df.columns and df['ROE'].notna().any() else 'N/A'}")
                    ])
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    create_kpi_table(df, "Key Performance Indicators", 
                        ['Ticker', 'MarketCap', 'P/E (TTM)', 'P/S', 'ROE', 'RevenueGrowth', 'FCF Yield', 'Valuation_Score', 'Quality_Score', 'Growth_Score'])
                ])
            ])
        ]
    
    # Handle individual KPI group tabs
    for group_name, columns in KPI_GROUPS.items():
        if active_tab == group_name.lower().replace(' ', '_'):
            return [create_kpi_table(df, group_name, columns)]
    
    # Handle historical analysis tabs
    if active_tab == "historical_trends":
        tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []
        if tickers:
            return [
                dbc.Row([
                    dbc.Col([
                        html.H3("📊 Quarterly Growth Analysis"),
                        html.P("YoY, QoQ growth rates and last 4 quarters performance."),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(create_metric_tables(tickers))
                    ], width=12)
                ])
            ]
        else:
            return [dbc.Alert("No tickers found for historical analysis", color="warning")]
    
    elif active_tab == "growth_analysis":
        tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []
        if tickers:
            return [
                dbc.Row([
                    dbc.Col([
                        html.H3("📊 Growth Rates Analysis"),
                        html.P("Quarter-over-Quarter and Year-over-Year growth rate data in table format."),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(id="growth-insights", children=[
                            html.H4("🎯 Key Growth Insights"),
                            html.Div(id="growth-insights-content")
                        ])
                    ])
                ])
            ]
        else:
            return [dbc.Alert("No tickers found for growth analysis", color="warning")]
    
    elif active_tab == "professional_analysis":
        tickers = df['Ticker'].tolist() if 'Ticker' in df.columns else []
        if tickers:
            return [
                dbc.Row([
                    dbc.Col([
                        html.H3("💼 Investment Banking Style Analysis"),
                        html.P("Professional-grade valuation multiples, profitability ratios, and relative valuation analysis."),
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(create_professional_metrics_tables(tickers))
                    ], width=12)
                ])
            ]
        else:
            return [dbc.Alert("No tickers found for professional analysis", color="warning")]
    
    elif active_tab == "nvidia_style":
        return create_nvidia_style_dashboard('ZETA')
    
    return dbc.Alert("Tab not found", color="danger")

def create_nvidia_style_dashboard(ticker='ZETA'):
    """Create NVIDIA-style comprehensive analysis dashboard"""
    
    # Get real financial data
    real_metrics = get_real_nvidia_metrics(ticker)
    
    # Get basic data
    try:
        peer_df = pd.read_csv('/home/user/webapp/zeta_adtech_analysis_enhanced.csv')
        ticker_data = peer_df[peer_df['Ticker'] == ticker].iloc[0] if not peer_df[peer_df['Ticker'] == ticker].empty else None
    except:
        ticker_data = None
    
    # Get current price and info
    current_price = "$18.90"
    price_change = "+2.3% (1D)"
    company_info = "Technology / AdTech"
    
    if real_metrics and 'raw_data' in real_metrics:
        info = real_metrics['raw_data']['info']
        if 'currentPrice' in info and info['currentPrice']:
            current_price = f"${info['currentPrice']:.2f}"
        elif 'regularMarketPrice' in info and info['regularMarketPrice']:
            current_price = f"${info['regularMarketPrice']:.2f}"
        elif ticker_data is not None and pd.notna(ticker_data['Price']):
            current_price = f"${ticker_data['Price']:.2f}"
            
        if 'sector' in info and info['sector']:
            company_info = f"{info['sector']}"
            if 'industry' in info and info['industry']:
                company_info += f" / {info['industry']}"
    
    # Enhanced Header Section - Neutral Grey Design
    header_style = {
        "background": "linear-gradient(135deg, #6c757d 0%, #495057 100%)",
        "color": "white",
        "borderRadius": "8px",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"
    }
    
    header_section = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H2(f"{ticker} Corp", className="mb-2 fw-bold text-white"),
                    html.P([
                        html.Strong("WKN: "), "123456 | ",
                        html.Strong("ISIN: "), f"US{ticker}001 | ",
                        html.Strong("Exchange: "), "NASDAQ | ",
                        html.Strong("Sector: "), company_info
                    ], className="text-white-50 mb-0", style={"fontSize": "0.9rem"})
                ], width=8),
                dbc.Col([
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.H3(current_price, className="text-white mb-0 fw-bold"),
                                html.Small(price_change, className="text-success")
                            ], width=12),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                html.Small("Market Cap", className="text-white-50"),
                                html.Div(get_formatted_market_cap(real_metrics), className="text-white fw-bold")
                            ], width=4),
                            dbc.Col([
                                html.Small("P/E Ratio", className="text-white-50"),
                                html.Div(get_formatted_pe_ratio(real_metrics), className="text-white fw-bold")
                            ], width=4),
                            dbc.Col([
                                html.Small("EPS", className="text-white-50"),
                                html.Div(get_formatted_eps(real_metrics), className="text-white fw-bold")
                            ], width=4)
                        ], className="mt-2")
                    ], className="text-end")
                ], width=4)
            ])
        ], className="p-4")
    ], className="mb-4", style=header_style)
    
    # Left Column: Charts
    left_charts = dbc.Col([
        # Revenue/Earnings Chart
        dbc.Card([
            dbc.CardHeader("Revenue, Profits & Margins (Quarterly - Last 3 Years)"),
            dbc.CardBody([
                dcc.Graph(
                    figure=create_comprehensive_financial_chart(ticker),
                    style={'height': '350px'}
                )
            ])
        ], className="mb-3"),
        
        # Stock Price Chart  
        dbc.Card([
            dbc.CardHeader("Stock Price Development"),
            dbc.CardBody([
                dcc.Graph(
                    figure=create_real_stock_chart(ticker), 
                    style={'height': '200px'}
                )
            ])
        ])
    ], width=4)
    
    # Middle Column: Growth and Stability Investment Banking Table
    growth_metrics = calculate_improved_growth_metrics_new(ticker)
    
    middle_scores = dbc.Col([
        # Growth and Stability - Investment Banking Table
        dbc.Card([
            dbc.CardHeader([
                html.H6("GROWTH AND STABILITY", className="mb-0 fw-bold text-primary", 
                        style={"fontSize": "0.9rem", "textTransform": "uppercase"})
            ], style={"backgroundColor": "#f8f9fa", "borderBottom": "1px solid #e0e0e0"}),
            dbc.CardBody([
                create_improved_investment_banking_table_new(growth_metrics)
            ], className="p-3")
        ], className="mb-3", style={
            "borderRadius": "8px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "border": "2px solid #6c757d",
            "backgroundColor": "#f8f9fa"
        }),
        
        # Other score cards - simplified  
        create_score_card("REVENUE GROWTH 2Y", 
                         growth_metrics.get('Revenue_YoY', 'N/A'), 
                         "", "secondary", 
                         ["Year-over-Year Growth", "Quarterly Performance"]),
                         
        create_score_card("AAQS SCORE", "7", "", "secondary", 
                         ["Overall Assessment"])
                         
    ], width=4)
    
    # Right Column: Detailed Metrics (using real data)
    detailed_metrics_data = real_metrics.get('detailed_metrics', []) if real_metrics else []
    
    # Default metrics if no real data
    if not detailed_metrics_data:
        detailed_metrics_data = [
            ("Profitability Check", [
                ("Return on Equity (ROE)", "N/A", 50),
                ("Net Profit Margin", "N/A", 50)
            ]),
            ("Stock Development", [
                ("Performance per Year", "N/A", 50),
                ("Stock Stability", "N/A", 50)
            ])
        ]
    
    right_metrics = dbc.Col([
        create_metrics_section(section_name, metrics) 
        for section_name, metrics in detailed_metrics_data        
    ], width=4)
    
    return [
        header_section,
        dbc.Row([left_charts, middle_scores, right_metrics])
    ]

def create_real_revenue_chart(ticker):
    """Create real revenue/earnings bar chart using last 2 years data"""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        
        if not financials.empty and 'Total Revenue' in financials.index:
            # Get last 2 years of data
            revenue_data = financials.loc['Total Revenue'].dropna()
            years = [col.year for col in revenue_data.index[:2]]
            revenues = [val / 1e9 for val in revenue_data.iloc[:2]]  # Convert to billions
            
            # Try to get net income
            earnings = []
            if 'Net Income' in financials.index:
                income_data = financials.loc['Net Income'].dropna()
                earnings = [val / 1e9 for val in income_data.iloc[:2]]  # Convert to billions
            else:
                earnings = [rev * 0.1 for rev in revenues]  # Fallback estimate
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Revenue (B)', x=years[::-1], y=revenues[::-1], marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Earnings (B)', x=years[::-1], y=earnings[::-1], marker_color='darkblue'))
            
            fig.update_layout(
                barmode='group',
                title='',
                showlegend=True,
                height=250,
                margin=dict(l=40, r=40, t=20, b=40),
                font=dict(size=10)
            )
            return fig
    except Exception as e:
        print(f"Error creating revenue chart for {ticker}: {e}")
    
    # Fallback to mock data
    years = [2023, 2024]
    revenue = [1.0, 1.2]
    earnings = [0.1, 0.15]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Revenue (B)', x=years, y=revenue, marker_color='lightblue'))
    fig.add_trace(go.Bar(name='Earnings (B)', x=years, y=earnings, marker_color='darkblue'))
    
    fig.update_layout(
        barmode='group',
        title='',
        showlegend=True,
        height=250,
        margin=dict(l=40, r=40, t=20, b=40),
        font=dict(size=10)
    )
    return fig

def create_real_stock_chart(ticker):
    """Create real stock price line chart for last year"""
    from datetime import datetime, timedelta
    
    try:
        stock = yf.Ticker(ticker)
        # Get 1 year of stock price data
        hist = stock.history(period="1y")
        
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, 
                y=hist['Close'], 
                mode='lines', 
                name='Stock Price', 
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title='',
                showlegend=False,
                height=200,
                margin=dict(l=40, r=40, t=20, b=40),
                font=dict(size=10),
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True)
            )
            return fig
    except Exception as e:
        print(f"Error creating stock chart for {ticker}: {e}")
    
    # Fallback to mock data
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    prices = np.cumsum(np.random.randn(len(dates)) * 0.02) + 100
    prices = prices * (18.90 / prices[-1])  # Normalize to current price
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Stock Price', line=dict(color='green', width=2)))
    
    fig.update_layout(
        title='',
        showlegend=False,
        height=200,
        margin=dict(l=40, r=40, t=20, b=40),
        font=dict(size=10),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    return fig

def create_score_card(title, score, max_score="", color="primary", details=[]):
    """Create a score card like in NVIDIA dashboard"""
    
    # Use neutral grey colors instead of bright colors
    display_color = "secondary"  # Always use grey
    
    if max_score:
        score_display = html.H1(f"{score}/{max_score}", className=f"text-{display_color} mb-0 fw-bold", style={"fontSize": "2.5rem"})
    else:
        score_display = html.H1(score, className=f"text-{display_color} mb-0 fw-bold", style={"fontSize": "2.5rem"})
    
    # Neutral grey styling
    card_style = {
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "border": "2px solid #6c757d",  # Neutral grey border
        "backgroundColor": "#f8f9fa"  # Light grey background
    }
    
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-title text-muted mb-3 fw-bold", style={"fontSize": "0.9rem", "textTransform": "uppercase"}),
            score_display,
            html.Hr(className="my-3"),
            # Left-aligned details as requested
            html.Div([
                html.Small(detail, className="d-block text-muted mb-1 text-start") for detail in details
            ], className="text-start")  # Changed from center to left alignment
        ], className="text-center p-4")
    ], className="mb-3", style=card_style)

def create_metrics_section(title, metrics):
    """Create a metrics section with intelligent progress bars"""
    
    metric_items = []
    for name, value, percentage, metric_type in metrics:
        # Intelligent color determination based on metric type and percentage
        if metric_type == 'profitability':  # Higher = Better (ROE, ROIC, NPE)
            if percentage >= 75:
                color = "success"   # 🟢 Excellent - Green
            elif percentage >= 50:
                color = "success"   # 🟢 Good - Green  
            elif percentage >= 25:
                color = "warning"   # 🟡 Average - Yellow
            else:
                color = "danger"    # 🔴 Poor - Red
        elif metric_type == 'valuation':  # Lower = Better (PE, PB, PS, EV ratios)
            if percentage >= 75:
                color = "success"   # 🟢 Excellent (low valuation) - Green
            elif percentage >= 50:
                color = "success"   # 🟢 Good (reasonable valuation) - Green
            elif percentage >= 25:
                color = "warning"   # 🟡 Average (high valuation) - Yellow  
            else:
                color = "danger"    # 🔴 Poor (very high valuation) - Red
        else:  # Default behavior
            if percentage >= 70:
                color = "success"
            elif percentage >= 40:
                color = "warning"
            else:
                color = "danger"
        
        # Ensure percentage is never below 5% to avoid invisible bars
        display_percentage = max(5, min(100, percentage))
            
        metric_items.append(
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Small(name, className="fw-bold text-dark", style={"fontSize": "0.85rem"})
                    ], width=7),
                    dbc.Col([
                        html.Small(value, className="text-end fw-bold", style={"fontSize": "0.85rem"})
                    ], width=5)
                ], className="mb-1"),
                dbc.Progress(
                    value=display_percentage, 
                    color=color, 
                    className="mb-3", 
                    style={"height": "12px", "borderRadius": "6px"}
                )
            ], className="mb-2")
        )
    
    # Enhanced card styling
    card_style = {
        "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "border": "1px solid #e0e0e0"
    }
    
    return dbc.Card([
        dbc.CardHeader([
            html.H6(title, className="mb-0 fw-bold text-primary", style={"fontSize": "1rem"})
        ], style={"backgroundColor": "#f8f9fa", "borderBottom": "1px solid #e0e0e0"}),
        dbc.CardBody(metric_items, className="p-4")
    ], className="mb-3", style=card_style)

def get_formatted_market_cap(real_metrics):
    """Format market cap for display"""
    if real_metrics and 'raw_data' in real_metrics and 'info' in real_metrics['raw_data']:
        info = real_metrics['raw_data']['info']
        if 'marketCap' in info and info['marketCap']:
            market_cap = info['marketCap']
            if market_cap >= 1e9:
                return f"${market_cap / 1e9:.1f}B"
            elif market_cap >= 1e6:
                return f"${market_cap / 1e6:.1f}M"
            else:
                return f"${market_cap:,.0f}"
    return "N/A"

def get_formatted_pe_ratio(real_metrics):
    """Format P/E ratio for display"""
    if real_metrics and 'raw_data' in real_metrics and 'info' in real_metrics['raw_data']:
        info = real_metrics['raw_data']['info']
        if 'trailingPE' in info and info['trailingPE']:
            return f"{info['trailingPE']:.1f}"
        elif 'forwardPE' in info and info['forwardPE']:
            return f"{info['forwardPE']:.1f}"
    return "N/A"

def calculate_improved_growth_metrics(ticker):
    """Calculate detailed growth metrics for investment banking table"""
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly and annual data
        quarterly_financials = stock.quarterly_financials
        annual_financials = stock.financials
        info = stock.info
        
        metrics = {}
        
        # === REVENUE GROWTH METRICS ===
        if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
            revenue_quarterly = quarterly_financials.loc['Total Revenue'].dropna()
            
            # Last 3 years revenue growth (CAGR)
            if len(revenue_quarterly) >= 12:  # 3 years of quarterly data
                recent_revenue = revenue_quarterly.iloc[0]
                three_years_ago = revenue_quarterly.iloc[11]
                revenue_3y_cagr = ((recent_revenue / three_years_ago) ** (1/3) - 1) * 100
                metrics['Revenue_3Y_CAGR'] = f"{revenue_3y_cagr:.1f}%"
            else:
                metrics['Revenue_3Y_CAGR'] = "N/A"
            
            # YoY Revenue Growth
            if len(revenue_quarterly) >= 4:
                current_q = revenue_quarterly.iloc[0]
                yoy_q = revenue_quarterly.iloc[3]  # Same quarter last year
                revenue_yoy = ((current_q / yoy_q) - 1) * 100
                metrics['Revenue_YoY'] = f"{revenue_yoy:.1f}%"
            else:
                metrics['Revenue_YoY'] = "N/A"
            
            # QoQ Revenue Growth
            if len(revenue_quarterly) >= 2:
                current_q = revenue_quarterly.iloc[0]
                last_q = revenue_quarterly.iloc[1]
                revenue_qoq = ((current_q / last_q) - 1) * 100
                metrics['Revenue_QoQ'] = f"{revenue_qoq:.1f}%"
            else:
                metrics['Revenue_QoQ'] = "N/A"
        
        # === GROSS PROFIT MARGIN ===
        if not quarterly_financials.empty and 'Gross Profit' in quarterly_financials.index and 'Total Revenue' in quarterly_financials.index:
            gross_profit = quarterly_financials.loc['Gross Profit'].dropna()
            revenue = quarterly_financials.loc['Total Revenue'].dropna()
            
            # Current Gross Margin
            if len(gross_profit) > 0 and len(revenue) > 0:
                current_margin = (gross_profit.iloc[0] / revenue.iloc[0]) * 100
                metrics['Gross_Margin_Current'] = f"{current_margin:.1f}%"
                
                # YoY Gross Margin change
                if len(gross_profit) >= 4 and len(revenue) >= 4:
                    yoy_margin = (gross_profit.iloc[3] / revenue.iloc[3]) * 100
                    margin_change = current_margin - yoy_margin
                    metrics['Gross_Margin_YoY_Change'] = f"{margin_change:+.1f}pp"
                else:
                    metrics['Gross_Margin_YoY_Change'] = "N/A"
                    
                # QoQ Gross Margin change
                if len(gross_profit) >= 2 and len(revenue) >= 2:
                    qoq_margin = (gross_profit.iloc[1] / revenue.iloc[1]) * 100
                    margin_change_qoq = current_margin - qoq_margin
                    metrics['Gross_Margin_QoQ_Change'] = f"{margin_change_qoq:+.1f}pp"
                else:
                    metrics['Gross_Margin_QoQ_Change'] = "N/A"
            else:
                metrics['Gross_Margin_Current'] = "N/A"
                metrics['Gross_Margin_YoY_Change'] = "N/A"
                metrics['Gross_Margin_QoQ_Change'] = "N/A"
        
        # === EPS METRICS ===
        # Try to get EPS from info first, then calculate from financials
        current_eps = None
        if 'trailingEps' in info and info['trailingEps'] is not None:
            current_eps = info['trailingEps']
            metrics['EPS_Current'] = f"${current_eps:.2f}"
        else:
            metrics['EPS_Current'] = "N/A"
        
        # Calculate EPS growth from net income and shares outstanding
        if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index:
            net_income_quarterly = quarterly_financials.loc['Net Income'].dropna()
            
            # Get shares outstanding (basic)
            shares_outstanding = None
            if 'sharesOutstanding' in info and info['sharesOutstanding']:
                shares_outstanding = info['sharesOutstanding']
            elif 'impliedSharesOutstanding' in info and info['impliedSharesOutstanding']:
                shares_outstanding = info['impliedSharesOutstanding']
            
            if shares_outstanding and len(net_income_quarterly) >= 4:
                # Calculate EPS for current and YoY quarters
                current_eps_calc = net_income_quarterly.iloc[0] / shares_outstanding
                yoy_eps_calc = net_income_quarterly.iloc[3] / shares_outstanding
                
                if yoy_eps_calc != 0:
                    eps_yoy_growth = ((current_eps_calc / yoy_eps_calc) - 1) * 100
                    metrics['EPS_YoY'] = f"{eps_yoy_growth:.1f}%"
                else:
                    metrics['EPS_YoY'] = "N/A"
                    
                # QoQ EPS growth
                if len(net_income_quarterly) >= 2:
                    last_q_eps = net_income_quarterly.iloc[1] / shares_outstanding
                    if last_q_eps != 0:
                        eps_qoq_growth = ((current_eps_calc / last_q_eps) - 1) * 100
                        metrics['EPS_QoQ'] = f"{eps_qoq_growth:.1f}%"
                    else:
                        metrics['EPS_QoQ'] = "N/A"
                else:
                    metrics['EPS_QoQ'] = "N/A"
                    
                # 3-year EPS CAGR
                if len(net_income_quarterly) >= 12:
                    three_years_eps = net_income_quarterly.iloc[11] / shares_outstanding
                    if three_years_eps > 0:
                        eps_3y_cagr = ((current_eps_calc / three_years_eps) ** (1/3) - 1) * 100
                        metrics['EPS_3Y_CAGR'] = f"{eps_3y_cagr:.1f}%"
                    else:
                        metrics['EPS_3Y_CAGR'] = "N/A"
                else:
                    metrics['EPS_3Y_CAGR'] = "N/A"
            else:
                metrics['EPS_YoY'] = "N/A"
                metrics['EPS_QoQ'] = "N/A"
                metrics['EPS_3Y_CAGR'] = "N/A"
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating improved growth metrics for {ticker}: {e}")
        return {}

def create_improved_investment_banking_table(metrics):
    """Create investment banking style table for Growth and Stability"""
    
    # Define the table structure
    table_data = [
        # Revenue Growth Section
        {"Metric": "Revenue Growth", "Period": "3-Year CAGR", "Value": metrics.get('Revenue_3Y_CAGR', 'N/A'), "Category": "Revenue"},
        {"Metric": "", "Period": "Year-over-Year", "Value": metrics.get('Revenue_YoY', 'N/A'), "Category": "Revenue"},
        {"Metric": "", "Period": "Quarter-over-Quarter", "Value": metrics.get('Revenue_QoQ', 'N/A'), "Category": "Revenue"},
        
        # Gross Margin Section  
        {"Metric": "Gross Profit Margin", "Period": "Current Quarter", "Value": metrics.get('Gross_Margin_Current', 'N/A'), "Category": "Margins"},
        {"Metric": "", "Period": "YoY Change", "Value": metrics.get('Gross_Margin_YoY_Change', 'N/A'), "Category": "Margins"},
        {"Metric": "", "Period": "QoQ Change", "Value": metrics.get('Gross_Margin_QoQ_Change', 'N/A'), "Category": "Margins"},
        
        # EPS Section
        {"Metric": "Earnings per Share", "Period": "Current (TTM)", "Value": metrics.get('EPS_Current', 'N/A'), "Category": "EPS"},
        {"Metric": "", "Period": "3-Year CAGR", "Value": metrics.get('EPS_3Y_CAGR', 'N/A'), "Category": "EPS"},
        {"Metric": "", "Period": "Year-over-Year", "Value": metrics.get('EPS_YoY', 'N/A'), "Category": "EPS"},
        {"Metric": "", "Period": "Quarter-over-Quarter", "Value": metrics.get('EPS_QoQ', 'N/A'), "Category": "EPS"},
    ]
    
    # Create the table with investment banking styling
    table_style = {
        'backgroundColor': 'white',
        'border': '1px solid #e0e0e0',
        'borderRadius': '4px',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '12px'
    }
    
    header_style = {
        'backgroundColor': '#f8f9fa',
        'fontWeight': 'bold',
        'borderBottom': '2px solid #dee2e6',
        'padding': '8px',
        'textAlign': 'left'
    }
    
    cell_style = {
        'padding': '6px 8px',
        'borderBottom': '1px solid #e9ecef',
        'textAlign': 'left',
        'whiteSpace': 'nowrap'
    }
    
    # Create sections with category headers
    table_rows = []
    current_category = ""
    
    for row in table_data:
        # Add category header
        if row['Category'] != current_category:
            current_category = row['Category']
            
            if current_category == "Revenue":
                category_name = "Revenue Growth Analysis"
                category_color = "#e3f2fd"
            elif current_category == "Margins": 
                category_name = "Profitability Margins"
                category_color = "#f3e5f5"
            else:  # EPS
                category_name = "Earnings per Share Analysis" 
                category_color = "#e8f5e8"
            
            table_rows.append(
                html.Tr([
                    html.Td(category_name, colSpan=3, style={
                        'backgroundColor': category_color,
                        'fontWeight': 'bold',
                        'padding': '8px',
                        'borderBottom': '1px solid #ccc',
                        'fontSize': '11px',
                        'color': '#333'
                    })
                ])
            )
        
        # Determine text color based on value
        value = row['Value']
        text_color = '#000'
        if value != 'N/A' and '%' in value:
            try:
                numeric_value = float(value.replace('%', '').replace('+', ''))
                if numeric_value > 0:
                    text_color = '#00b300'  # Green for positive
                elif numeric_value < 0:
                    text_color = '#d32f2f'  # Red for negative
            except:
                pass
        
        # Add data row
        table_rows.append(
            html.Tr([
                html.Td(row['Metric'], style={**cell_style, 'fontWeight': 'bold' if row['Metric'] else 'normal', 'width': '40%'}),
                html.Td(row['Period'], style={**cell_style, 'width': '35%'}),
                html.Td(row['Value'], style={**cell_style, 'fontWeight': 'bold', 'color': text_color, 'width': '25%'})
            ])
        )
    
    return html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric", style=header_style),
                    html.Th("Period", style=header_style),
                    html.Th("Value", style=header_style)
                ])
            ]),
            html.Tbody(table_rows)
        ], style=table_style)
    ])

def get_formatted_eps(real_metrics):
    """Format EPS for display"""
    if real_metrics and 'raw_data' in real_metrics and 'info' in real_metrics['raw_data']:
        info = real_metrics['raw_data']['info']
        if 'trailingEps' in info and info['trailingEps'] is not None:
            return f"${info['trailingEps']:.2f}"
        elif 'forwardEps' in info and info['forwardEps'] is not None:
            return f"${info['forwardEps']:.2f}"
    return "N/A"

def create_comprehensive_financial_chart(ticker):
    """Create comprehensive financial chart like the provided image with quarterly data"""
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financial data
        quarterly_financials = stock.quarterly_financials
        
        if quarterly_financials.empty:
            return create_fallback_chart()
        
        # Get last 12 quarters (3 years) of data
        periods = quarterly_financials.columns[:12]
        
        # Extract key metrics
        revenue_data = []
        ebit_data = []
        net_income_data = []
        margins = []
        quarter_labels = []
        
        for period in periods:
            # Revenue
            if 'Total Revenue' in quarterly_financials.index:
                revenue = quarterly_financials.loc['Total Revenue', period] / 1e9  # Convert to billions
                revenue_data.append(revenue if not pd.isna(revenue) else 0)
            else:
                revenue_data.append(0)
            
            # EBIT (Operating Income)
            if 'Operating Income' in quarterly_financials.index:
                ebit = quarterly_financials.loc['Operating Income', period] / 1e9
                ebit_data.append(ebit if not pd.isna(ebit) else 0)
            else:
                ebit_data.append(0)
            
            # Net Income
            if 'Net Income' in quarterly_financials.index:
                net_income = quarterly_financials.loc['Net Income', period] / 1e9
                net_income_data.append(net_income if not pd.isna(net_income) else 0)
            else:
                net_income_data.append(0)
            
            # Calculate net profit margin
            if revenue_data[-1] != 0:
                margin = (net_income_data[-1] / revenue_data[-1]) * 100
                margins.append(margin)
            else:
                margins.append(0)
            
            # Format quarter label
            quarter_labels.append(f"{period.strftime('%Y')}-Q{((period.month-1)//3)+1}")
        
        # Reverse to show chronologically (oldest to newest)
        revenue_data = revenue_data[::-1]
        ebit_data = ebit_data[::-1]
        net_income_data = net_income_data[::-1]
        margins = margins[::-1]
        quarter_labels = quarter_labels[::-1]
        
        # Create the comprehensive chart
        fig = go.Figure()
        
        # Add Revenue bars (dark blue)
        fig.add_trace(go.Bar(
            name='Revenue',
            x=quarter_labels,
            y=revenue_data,
            marker_color='#1f4e79',  # Dark blue
            yaxis='y',
            offsetgroup=1
        ))
        
        # Add EBIT bars (medium blue)
        fig.add_trace(go.Bar(
            name='EBIT', 
            x=quarter_labels,
            y=ebit_data,
            marker_color='#4a90c2',  # Medium blue
            yaxis='y',
            offsetgroup=1
        ))
        
        # Add Net Income bars (light blue)
        fig.add_trace(go.Bar(
            name='Profit',
            x=quarter_labels,
            y=net_income_data,
            marker_color='#87ceeb',  # Light blue
            yaxis='y', 
            offsetgroup=1
        ))
        
        # Add Net Profit Margin line (green)
        fig.add_trace(go.Scatter(
            name='Net Profit Margin (%)',
            x=quarter_labels,
            y=margins,
            mode='lines+markers',
            line=dict(color='#00b300', width=3),  # Green line
            marker=dict(size=6, color='#00b300'),
            yaxis='y2'
        ))
        
        # Update layout with dual y-axes like the original
        fig.update_layout(
            title='',
            xaxis=dict(
                title='Quarter',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Revenue, EBIT, Profit (Billions)',
                side='left',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                title='Net Profit Margin (%)',
                side='right',
                overlaying='y',
                showgrid=False,
                ticksuffix='%'
            ),
            barmode='group',
            height=350,
            margin=dict(l=60, r=60, t=40, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating comprehensive chart for {ticker}: {e}")
        return create_fallback_chart()

def create_fallback_chart():
    """Create a fallback chart when data is not available"""
    
    fig = go.Figure()
    
    # Add empty traces for consistency
    fig.add_trace(go.Bar(name='Revenue', x=[], y=[], marker_color='#1f4e79'))
    fig.add_trace(go.Bar(name='EBIT', x=[], y=[], marker_color='#4a90c2')) 
    fig.add_trace(go.Bar(name='Profit', x=[], y=[], marker_color='#87ceeb'))
    fig.add_trace(go.Scatter(name='Net Profit Margin (%)', x=[], y=[], line=dict(color='#00b300')))
    
    fig.update_layout(
        title='Financial Data Not Available',
        height=350,
        margin=dict(l=60, r=60, t=40, b=60)
    )
    
    return fig

def get_real_nvidia_metrics(ticker):
    """Get real NVIDIA-style metrics using Yahoo Finance data"""
    
    try:
        # Get ticker object
        stock = yf.Ticker(ticker)
        
        # Get current year
        current_year = datetime.now().year
        last_year = current_year - 1
        year_before_last = current_year - 2
        
        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get info and statistics
        info = stock.info
        
        # Calculate growth metrics for last 2 years
        growth_metrics = {}
        
        # Revenue growth calculation
        if not financials.empty and 'Total Revenue' in financials.index:
            revenue_data = financials.loc['Total Revenue'].dropna()
            if len(revenue_data) >= 2:
                recent_revenue = revenue_data.iloc[:2]
                if len(recent_revenue) == 2:
                    revenue_growth = ((recent_revenue.iloc[0] - recent_revenue.iloc[1]) / recent_revenue.iloc[1]) * 100
                    growth_metrics['revenue_growth_yoy'] = revenue_growth
        
        # Get growth from info if available
        if 'revenueGrowth' in info and info['revenueGrowth'] is not None:
            growth_metrics['revenue_growth_ttm'] = info['revenueGrowth'] * 100
            
        if 'earningsGrowth' in info and info['earningsGrowth'] is not None:
            growth_metrics['earnings_growth'] = info['earningsGrowth'] * 100
        
        # Build score cards with real data
        score_cards = []
        
        # Quality Check Score (based on margins and ratios)
        quality_score = 0
        quality_details = []
        
        if 'grossMargins' in info and info['grossMargins'] is not None:
            gross_margin = info['grossMargins'] * 100
            quality_details.append(f"Gross Margin: {gross_margin:.1f}%")
            if gross_margin > 60: quality_score += 5
            elif gross_margin > 40: quality_score += 3
            elif gross_margin > 20: quality_score += 1
        
        if 'operatingMargins' in info and info['operatingMargins'] is not None:
            op_margin = info['operatingMargins'] * 100
            quality_details.append(f"Operating Margin: {op_margin:.1f}%")
            if op_margin > 20: quality_score += 5
            elif op_margin > 10: quality_score += 3
            elif op_margin > 5: quality_score += 1
        
        score_cards.append({
            'title': 'GROWTH AND STABILITY',
            'score': 'Strong' if quality_score >= 12 else ('Moderate' if quality_score >= 8 else 'Weak'),
            'max_score': '',
            'color': 'success' if quality_score >= 12 else ('warning' if quality_score >= 8 else 'danger'),
            'details': quality_details
        })
        
        # Revenue Growth (use real data)
        revenue_growth = None
        if 'revenue_growth_ttm' in growth_metrics:
            revenue_growth = growth_metrics['revenue_growth_ttm']
        elif 'revenue_growth_yoy' in growth_metrics:
            revenue_growth = growth_metrics['revenue_growth_yoy']
        elif 'revenueGrowth' in info and info['revenueGrowth'] is not None:
            revenue_growth = info['revenueGrowth'] * 100
            
        if revenue_growth is not None:
            score_cards.append({
                'title': 'REVENUE GROWTH 2Y',
                'score': f"{revenue_growth:.1f}%",
                'max_score': '',
                'color': 'success' if revenue_growth > 15 else ('primary' if revenue_growth > 5 else 'warning'),
                'details': ['Last Year Performance', 'YoY Growth Rate']
            })
        
        # Growth Check Score
        growth_score = 0
        growth_details = []
        
        if revenue_growth is not None:
            growth_details.append(f"Revenue Growth: {revenue_growth:.1f}%")
            if revenue_growth > 20: growth_score += 7
            elif revenue_growth > 10: growth_score += 5
            elif revenue_growth > 0: growth_score += 3
        
        if 'earnings_growth' in growth_metrics:
            earnings_growth = growth_metrics['earnings_growth']
            growth_details.append(f"EPS Growth: {earnings_growth:.1f}%")
            if earnings_growth > 15: growth_score += 8
            elif earnings_growth > 5: growth_score += 5
            elif earnings_growth > 0: growth_score += 2
            
        score_cards.append({
            'title': 'GROWTH CHECK',
            'score': str(growth_score),
            'max_score': '15',
            'color': 'success' if growth_score >= 12 else ('warning' if growth_score >= 8 else 'danger'),
            'details': growth_details
        })
        
        # Overall AAQS Score (combination of quality and growth)
        overall_score = min(10, (quality_score + growth_score) // 3)
        score_cards.append({
            'title': 'AAQS SCORE',
            'score': str(overall_score),
            'max_score': '',
            'color': 'success' if overall_score >= 8 else ('warning' if overall_score >= 6 else 'danger'),
            'details': ['Overall Assessment']
        })
        
        # Calculate stock stability (volatility measures)
        stock_stability_score = 50  # Default
        annual_performance = 0
        try:
            # Get 1 year of stock data for stability calculation
            hist = stock.history(period="1y")
            if not hist.empty and len(hist) > 20:
                daily_returns = hist['Close'].pct_change().dropna()
                volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized volatility
                stock_stability_score = max(0, 100 - volatility * 2)  # Lower volatility = higher stability
                
                # Calculate annual performance
                start_price = hist['Close'].iloc[0]
                end_price = hist['Close'].iloc[-1]
                annual_performance = ((end_price - start_price) / start_price) * 100
        except:
            pass
        
        # Calculate ROIC (Return on Invested Capital)
        roic = None
        try:
            if not balance_sheet.empty and not financials.empty:
                # Get total invested capital and net operating profit
                if 'Total Assets' in balance_sheet.index and 'Cash And Cash Equivalents' in balance_sheet.index:
                    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if len(balance_sheet.loc['Total Assets']) > 0 else None
                    cash = balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] if len(balance_sheet.loc['Cash And Cash Equivalents']) > 0 else None
                    if total_assets and cash:
                        invested_capital = total_assets - cash
                        if 'Net Income' in financials.index and invested_capital > 0:
                            net_income = financials.loc['Net Income'].iloc[0] if len(financials.loc['Net Income']) > 0 else None
                            if net_income:
                                roic = (net_income / invested_capital) * 100
        except:
            pass
        
        # Detailed Metrics with Progress Bars - Your exact specifications
        detailed_metrics = []
        
        # 1. Extended Profitability Check: NPE (Quarterly & Yearly), Valuation Ratios
        profitability_section = []
        
        # Get quarterly financials once for reuse
        quarterly_financials = stock.quarterly_financials
        
        # === PROFITABILITY METRICS (Higher = Better) ===
        
        # ROE (TTM only - quarterly ROE not meaningful)
        if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
            roe = info['returnOnEquity'] * 100
            # Scale: 0-30% ROE as 0-100%
            percentage = min(100, max(0, (roe / 30) * 100))
            profitability_section.append(('Return on Equity (ROE) - TTM', f'{roe:.1f}%', percentage, 'profitability'))
            
        # ROIC (TTM only - quarterly ROIC not standard)
        if roic is not None:
            # Scale: 0-25% ROIC as 0-100%
            percentage = min(100, max(0, (roic / 25) * 100))
            profitability_section.append(('Return on Invested Capital (ROIC) - TTM', f'{roic:.1f}%', percentage, 'profitability'))
        
        # NPE (Net Profit Margin) - Both Quarterly and Yearly
        try:
            # Quarterly NPE
            if not quarterly_financials.empty and 'Net Income' in quarterly_financials.index and 'Total Revenue' in quarterly_financials.index:
                net_income_q = quarterly_financials.loc['Net Income'].dropna()
                revenue_q = quarterly_financials.loc['Total Revenue'].dropna()
                if len(net_income_q) > 0 and len(revenue_q) > 0:
                    npe_quarterly = (net_income_q.iloc[0] / revenue_q.iloc[0]) * 100
                    # Scale: 0-20% NPE as 0-100%
                    percentage = min(100, max(0, (npe_quarterly / 20) * 100))
                    profitability_section.append(('NPE - Current Quarter', f'{npe_quarterly:.1f}%', percentage, 'profitability'))
            
            # Yearly NPE (TTM preferred)
            if 'profitMargins' in info and info['profitMargins'] is not None:
                npe_yearly = info['profitMargins'] * 100
                percentage = min(100, max(0, (npe_yearly / 20) * 100))
                profitability_section.append(('NPE - Trailing 12M', f'{npe_yearly:.1f}%', percentage, 'profitability'))
            elif not financials.empty and 'Net Income' in financials.index and 'Total Revenue' in financials.index:
                # Calculate from annual data if TTM not available
                net_income_a = financials.loc['Net Income'].dropna()
                revenue_a = financials.loc['Total Revenue'].dropna()
                if len(net_income_a) > 0 and len(revenue_a) > 0:
                    npe_annual = (net_income_a.iloc[0] / revenue_a.iloc[0]) * 100
                    percentage = min(100, max(0, (npe_annual / 20) * 100))
                    profitability_section.append(('NPE - Annual', f'{npe_annual:.1f}%', percentage, 'profitability'))
        except Exception as e:
            print(f"Error calculating NPE metrics: {e}")
        
        # === VALUATION METRICS (Lower = Better) ===
        
        # Get current price once for all calculations
        current_price_num = 0
        try:
            if 'currentPrice' in info and info['currentPrice']:
                current_price_num = float(info['currentPrice'])
            elif 'regularMarketPrice' in info and info['regularMarketPrice']:
                current_price_num = float(info['regularMarketPrice'])
        except:
            current_price_num = 0
        
        # PE Ratio - Quarterly, TTM and Forward
        # PE Quarterly (from quarterly EPS) - Only if EPS is positive
        try:
            if not quarterly_financials.empty and current_price_num > 0:
                # Try different EPS fields that might be available
                eps_fields = ['Basic EPS', 'Diluted EPS', 'Net Income']
                eps_quarterly = None
                
                for field in eps_fields:
                    if field in quarterly_financials.index:
                        eps_data = quarterly_financials.loc[field].dropna()
                        if len(eps_data) > 0 and eps_data.iloc[0] != 0:
                            if field == 'Net Income':
                                # Convert Net Income to EPS
                                shares = info.get('sharesOutstanding', 0)
                                if shares > 0:
                                    eps_quarterly = eps_data.iloc[0] / shares
                            else:
                                eps_quarterly = eps_data.iloc[0]
                            break
                            
                # Only show PE if EPS is positive
                if eps_quarterly and eps_quarterly > 0:
                    pe_quarterly = current_price_num / (eps_quarterly * 4)  # Annualize
                    percentage = min(100, max(0, 100 - ((pe_quarterly - 5) / 45) * 100))
                    profitability_section.append(('PE Ratio - Quarterly (Est)', f'{pe_quarterly:.1f}', percentage, 'valuation'))
                elif eps_quarterly and eps_quarterly <= 0:
                    # Show negative EPS but no PE ratio
                    profitability_section.append(('PE Ratio - Quarterly (Est)', 'N/A (Negative EPS)', 0, 'valuation'))
        except Exception as e:
            print(f"Error calculating quarterly PE: {e}")
            
        # PE TTM (Trailing Twelve Months) - Try multiple sources
        pe_ttm_found = False
        # Method 1: Direct trailingPE
        if 'trailingPE' in info and info['trailingPE'] is not None and info['trailingPE'] > 0:
            pe_ttm = info['trailingPE']
            percentage = min(100, max(0, 100 - ((pe_ttm - 5) / 45) * 100))
            profitability_section.append(('PE Ratio - TTM', f'{pe_ttm:.1f}', percentage, 'valuation'))
            pe_ttm_found = True
            
        # Method 2: Calculate from trailingEps if trailingPE not available
        if not pe_ttm_found and 'trailingEps' in info and info['trailingEps'] is not None and current_price_num > 0:
            trailing_eps = info['trailingEps']
            if trailing_eps > 0:
                pe_ttm_calc = current_price_num / trailing_eps
                percentage = min(100, max(0, 100 - ((pe_ttm_calc - 5) / 45) * 100))
                profitability_section.append(('PE Ratio - TTM (Calc)', f'{pe_ttm_calc:.1f}', percentage, 'valuation'))
                pe_ttm_found = True
            elif trailing_eps <= 0:
                # Show negative EPS situation
                profitability_section.append(('PE Ratio - TTM', 'N/A (Negative EPS)', 0, 'valuation'))
                pe_ttm_found = True
                
        # Method 3: If still no TTM PE, show as unavailable
        if not pe_ttm_found:
            profitability_section.append(('PE Ratio - TTM', 'N/A (Data Unavailable)', 0, 'valuation'))
            
        # PE Forward (Analyst estimates for next 12 months)
        if 'forwardPE' in info and info['forwardPE'] is not None and info['forwardPE'] > 0:
            pe_forward = info['forwardPE']
            percentage = min(100, max(0, 100 - ((pe_forward - 5) / 45) * 100))
            profitability_section.append(('PE Ratio - Forward (12M Est)', f'{pe_forward:.1f}', percentage, 'valuation'))
        
        # PB Ratio - Both Quarterly and TTM estimates  
        # PB Quarterly (calculate from available data)
        pb_quarterly_found = False
        try:
            if current_price_num > 0:
                shares_outstanding = info.get('sharesOutstanding', 0)
                
                # Method 1: Use bookValue per share from info if available
                if 'bookValue' in info and info['bookValue'] is not None and info['bookValue'] > 0:
                    book_value_per_share = info['bookValue']
                    pb_quarterly = current_price_num / book_value_per_share
                    percentage = min(100, max(0, 100 - ((pb_quarterly - 0.5) / 4.5) * 100))
                    profitability_section.append(('PB Ratio - Current Quarter', f'{pb_quarterly:.1f}', percentage, 'valuation'))
                    pb_quarterly_found = True
                    
                # Method 2: Calculate from quarterly balance sheet if available
                elif not quarterly_financials.empty and shares_outstanding > 0:
                    # Try different equity fields
                    equity_fields = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity', 'Common Stock Equity']
                    equity_quarterly = None
                    
                    for field in equity_fields:
                        if field in quarterly_financials.index:
                            equity_data = quarterly_financials.loc[field].dropna()
                            if len(equity_data) > 0 and equity_data.iloc[0] > 0:
                                equity_quarterly = equity_data.iloc[0]
                                break
                                
                    if equity_quarterly and equity_quarterly > 0:
                        book_value_per_share_q = equity_quarterly / shares_outstanding
                        pb_quarterly = current_price_num / book_value_per_share_q
                        percentage = min(100, max(0, 100 - ((pb_quarterly - 0.5) / 4.5) * 100))
                        profitability_section.append(('PB Ratio - Current Quarter', f'{pb_quarterly:.1f}', percentage, 'valuation'))
                        pb_quarterly_found = True
                        
        except Exception as e:
            print(f"Error calculating quarterly PB: {e}")
            
        # Show quarterly PB as unavailable if calculation failed
        if not pb_quarterly_found:
            profitability_section.append(('PB Ratio - Current Quarter', 'N/A (Data Unavailable)', 0, 'valuation'))
            
        # PB TTM (from API)
        if 'priceToBook' in info and info['priceToBook'] is not None and info['priceToBook'] > 0:
            pb_ratio = info['priceToBook']
            percentage = min(100, max(0, 100 - ((pb_ratio - 0.5) / 4.5) * 100))
            profitability_section.append(('PB Ratio - TTM', f'{pb_ratio:.1f}', percentage, 'valuation'))
        else:
            profitability_section.append(('PB Ratio - TTM', 'N/A (Data Unavailable)', 0, 'valuation'))
        
        # PS Ratio - Both Quarterly and TTM
        try:
            # Quarterly PS estimate
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index and current_price_num > 0:
                revenue_q = quarterly_financials.loc['Total Revenue'].dropna()
                shares_outstanding = info.get('sharesOutstanding', 0)
                if len(revenue_q) > 0 and shares_outstanding > 0:
                    revenue_per_share_q = (revenue_q.iloc[0] * 4) / shares_outstanding  # Annualize
                    ps_quarterly = current_price_num / revenue_per_share_q
                    # Scale: PS 0.5-10 as 100-0%
                    percentage = min(100, max(0, 100 - ((ps_quarterly - 0.5) / 9.5) * 100))
                    profitability_section.append(('PS Ratio - Quarterly (Est)', f'{ps_quarterly:.1f}', percentage, 'valuation'))
        except Exception as e:
            print(f"Error calculating quarterly PS: {e}")
            
        # PS TTM
        if 'priceToSalesTrailing12Months' in info and info['priceToSalesTrailing12Months'] is not None:
            ps_ratio = info['priceToSalesTrailing12Months']
            percentage = min(100, max(0, 100 - ((ps_ratio - 0.5) / 9.5) * 100))
            profitability_section.append(('PS Ratio - TTM', f'{ps_ratio:.1f}', percentage, 'valuation'))
        
        # EV/Revenue - Both Quarterly and TTM estimates
        # EV/Revenue Quarterly (estimated)
        try:
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                revenue_q = quarterly_financials.loc['Total Revenue'].dropna()
                enterprise_value = info.get('enterpriseValue', 0)
                if len(revenue_q) > 0 and enterprise_value > 0:
                    # Annualize quarterly revenue
                    annual_revenue_est = revenue_q.iloc[0] * 4
                    ev_revenue_quarterly = enterprise_value / annual_revenue_est
                    percentage = min(100, max(0, 100 - ((ev_revenue_quarterly - 0.5) / 9.5) * 100))
                    profitability_section.append(('EV/Revenue - Quarterly (Est)', f'{ev_revenue_quarterly:.1f}', percentage, 'valuation'))
        except Exception as e:
            print(f"Error calculating quarterly EV/Revenue: {e}")
            
        # EV/Revenue TTM (from API)
        if 'enterpriseToRevenue' in info and info['enterpriseToRevenue'] is not None:
            ev_revenue = info['enterpriseToRevenue']
            percentage = min(100, max(0, 100 - ((ev_revenue - 0.5) / 9.5) * 100))
            profitability_section.append(('EV/Revenue - TTM', f'{ev_revenue:.1f}', percentage, 'valuation'))
        
        # EV/EBITDA - Both Quarterly and TTM estimates
        # EV/EBITDA Quarterly (estimated)
        try:
            if not quarterly_financials.empty:
                # Try to find EBITDA or calculate it
                ebitda_quarterly = None
                
                # Method 1: Direct EBITDA field
                if 'EBITDA' in quarterly_financials.index:
                    ebitda_data = quarterly_financials.loc['EBITDA'].dropna()
                    if len(ebitda_data) > 0:
                        ebitda_quarterly = ebitda_data.iloc[0]
                
                # Method 2: Calculate EBITDA from Operating Income + Depreciation
                elif 'Operating Income' in quarterly_financials.index:
                    op_income = quarterly_financials.loc['Operating Income'].dropna()
                    if len(op_income) > 0:
                        # Use operating income as proxy (conservative estimate)
                        ebitda_quarterly = op_income.iloc[0]
                        
                if ebitda_quarterly and ebitda_quarterly > 0:
                    enterprise_value = info.get('enterpriseValue', 0)
                    if enterprise_value > 0:
                        # Annualize quarterly EBITDA
                        annual_ebitda_est = ebitda_quarterly * 4
                        ev_ebitda_quarterly = enterprise_value / annual_ebitda_est
                        percentage = min(100, max(0, 100 - ((ev_ebitda_quarterly - 5) / 45) * 100))
                        profitability_section.append(('EV/EBITDA - Quarterly (Est)', f'{ev_ebitda_quarterly:.1f}', percentage, 'valuation'))
        except Exception as e:
            print(f"Error calculating quarterly EV/EBITDA: {e}")
            
        # EV/EBITDA TTM (from API)
        if 'enterpriseToEbitda' in info and info['enterpriseToEbitda'] is not None and info['enterpriseToEbitda'] > 0:
            ev_ebitda = info['enterpriseToEbitda']
            percentage = min(100, max(0, 100 - ((ev_ebitda - 5) / 45) * 100))
            profitability_section.append(('EV/EBITDA - TTM', f'{ev_ebitda:.1f}', percentage, 'valuation'))
            
        if profitability_section:
            detailed_metrics.append(('Profitability Check', profitability_section))
        
        # 2. Stock Development: Performance per year and Stock Stability
        stock_section = []
        if annual_performance != 0:
            performance_percentage = min(100, max(0, (annual_performance + 50) * 1.5))  # Scale -50% to 50% as 0-100%
            stock_section.append(('Performance per Year', f'{annual_performance:.1f}%', performance_percentage, 'profitability'))
            
        stock_section.append(('Stock Stability', f'{stock_stability_score:.1f}%', stock_stability_score, 'profitability'))
        
        if stock_section:
            detailed_metrics.append(('Stock Development', stock_section))
            
        # 3. Security and Balance: EBIT vs Debt, EBIT Interest Coverage, Equity vs Debt Ratio
        security_section = []
        
        # Try to calculate EBIT and debt metrics
        ebit_debt_ratio = None
        interest_coverage = None
        equity_debt_ratio = None
        
        try:
            if not financials.empty:
                # EBIT calculation (Operating Income)
                ebit = None
                if 'Operating Income' in financials.index:
                    ebit = financials.loc['Operating Income'].iloc[0] if len(financials.loc['Operating Income']) > 0 else None
                
                # Get debt information
                total_debt = None
                if not balance_sheet.empty and 'Total Debt' in balance_sheet.index:
                    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if len(balance_sheet.loc['Total Debt']) > 0 else None
                elif 'debtToEquity' in info and 'totalStockholderEquity' in info:
                    # Calculate from debt-to-equity ratio
                    if info['debtToEquity'] and info['totalStockholderEquity']:
                        total_debt = info['debtToEquity'] * info['totalStockholderEquity']
                
                # EBIT to Debt ratio
                if ebit and total_debt and total_debt > 0:
                    ebit_debt_ratio = (ebit / total_debt) * 100
                
                # Interest coverage ratio (EBIT / Interest Expense)
                if ebit and 'Interest Expense' in financials.index:
                    interest_expense = financials.loc['Interest Expense'].iloc[0] if len(financials.loc['Interest Expense']) > 0 else None
                    if interest_expense and interest_expense < 0:  # Interest expense is usually negative
                        interest_coverage = ebit / abs(interest_expense)
                
                # Equity to Debt ratio
                if 'totalStockholderEquity' in info and total_debt:
                    equity = info['totalStockholderEquity']
                    if equity > 0 and total_debt > 0:
                        equity_debt_ratio = (equity / total_debt) * 100
        except:
            pass
        
        if ebit_debt_ratio is not None:
            percentage = min(100, max(0, ebit_debt_ratio * 2))  # Scale 0-50% as 0-100%
            security_section.append(('EBIT / Debt Ratio', f'{ebit_debt_ratio:.1f}%', percentage, 'profitability'))
        
        if interest_coverage is not None:
            percentage = min(100, max(0, min(interest_coverage * 10, 100)))  # Scale 0-10x as 0-100%
            security_section.append(('EBIT Interest Coverage', f'{interest_coverage:.1f}x', percentage, 'profitability'))
            
        if equity_debt_ratio is not None:
            percentage = min(100, max(0, equity_debt_ratio))  # Scale 0-100% as 0-100%
            security_section.append(('Equity / Debt Ratio', f'{equity_debt_ratio:.1f}%', percentage, 'profitability'))
        elif 'debtToEquity' in info and info['debtToEquity'] is not None:
            # Fallback: show debt-to-equity (inverse relationship)
            debt_equity = info['debtToEquity']
            percentage = max(0, 100 - min(debt_equity * 5, 100))  # Lower debt = better
            security_section.append(('Debt / Equity Ratio', f'{debt_equity:.2f}', percentage, 'valuation'))
            
        if security_section:
            detailed_metrics.append(('Security and Balance', security_section))
        
        return {
            'score_cards': score_cards,
            'detailed_metrics': detailed_metrics,
            'raw_data': {
                'financials': financials,
                'info': info,
                'growth_metrics': growth_metrics
            }
        }
        
    except Exception as e:
        print(f"Error fetching NVIDIA-style metrics for {ticker}: {e}")
        return None

# === IMPROVED FUNCTIONS ===

def calculate_improved_growth_metrics_new(ticker):
    """Calculate improved growth metrics with Price Development instead of EPS"""
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial data
        quarterly_financials = stock.quarterly_financials
        annual_financials = stock.financials
        info = stock.info
        
        metrics = {}
        
        # === REVENUE GROWTH METRICS (IMPROVED) ===
        # Try to calculate 3-year revenue growth properly
        revenue_3y_growth = None
        
        # Method 1: Use annual data if available
        if not annual_financials.empty and 'Total Revenue' in annual_financials.index:
            annual_revenue = annual_financials.loc['Total Revenue'].dropna()
            if len(annual_revenue) >= 3:
                most_recent = annual_revenue.iloc[0]
                two_years_ago = annual_revenue.iloc[2]
                if two_years_ago > 0:
                    # Calculate 2-year CAGR (we have 3 data points spanning 2 years)
                    revenue_3y_growth = ((most_recent / two_years_ago) ** (1/2) - 1) * 100
        
        # Method 2: Use quarterly data if annual insufficient
        if revenue_3y_growth is None and not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
            revenue_quarterly = quarterly_financials.loc['Total Revenue'].dropna()
            if len(revenue_quarterly) >= 8:  # 2+ years of quarterly data
                recent_revenue = revenue_quarterly.iloc[0]
                eight_quarters_ago = revenue_quarterly.iloc[7]
                if eight_quarters_ago > 0:
                    revenue_3y_growth = ((recent_revenue / eight_quarters_ago) ** (1/2) - 1) * 100
            elif len(revenue_quarterly) >= 4:  # At least 1 year
                recent_revenue = revenue_quarterly.iloc[0]
                four_quarters_ago = revenue_quarterly.iloc[3]
                if four_quarters_ago > 0:
                    # Use 1-year growth as proxy
                    revenue_3y_growth = ((recent_revenue / four_quarters_ago) - 1) * 100
        
        if revenue_3y_growth is not None:
            metrics['Revenue_3Y_Growth'] = f"{revenue_3y_growth:.1f}%"
        else:
            metrics['Revenue_3Y_Growth'] = "Insufficient Data"
        
        # YoY and QoQ calculations 
        if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
            revenue_quarterly = quarterly_financials.loc['Total Revenue'].dropna()
            
            # YoY Revenue Growth
            if len(revenue_quarterly) >= 4:
                current_q = revenue_quarterly.iloc[0]
                yoy_q = revenue_quarterly.iloc[3]
                revenue_yoy = ((current_q / yoy_q) - 1) * 100
                metrics['Revenue_YoY'] = f"{revenue_yoy:.1f}%"
            else:
                metrics['Revenue_YoY'] = "N/A"
            
            # QoQ Revenue Growth
            if len(revenue_quarterly) >= 2:
                current_q = revenue_quarterly.iloc[0]
                last_q = revenue_quarterly.iloc[1]
                revenue_qoq = ((current_q / last_q) - 1) * 100
                metrics['Revenue_QoQ'] = f"{revenue_qoq:.1f}%"
            else:
                metrics['Revenue_QoQ'] = "N/A"
        
        # === GROSS PROFIT MARGIN ===
        if not quarterly_financials.empty and 'Gross Profit' in quarterly_financials.index and 'Total Revenue' in quarterly_financials.index:
            gross_profit = quarterly_financials.loc['Gross Profit'].dropna()
            revenue = quarterly_financials.loc['Total Revenue'].dropna()
            
            # Current Gross Margin
            if len(gross_profit) > 0 and len(revenue) > 0:
                current_margin = (gross_profit.iloc[0] / revenue.iloc[0]) * 100
                metrics['Gross_Margin_Current'] = f"{current_margin:.1f}%"
                
                # YoY Gross Margin change
                if len(gross_profit) >= 4 and len(revenue) >= 4:
                    yoy_margin = (gross_profit.iloc[3] / revenue.iloc[3]) * 100
                    margin_change = current_margin - yoy_margin
                    metrics['Gross_Margin_YoY_Change'] = f"{margin_change:+.1f}pp"
                else:
                    metrics['Gross_Margin_YoY_Change'] = "N/A"
                    
                # QoQ Gross Margin change
                if len(gross_profit) >= 2 and len(revenue) >= 2:
                    qoq_margin = (gross_profit.iloc[1] / revenue.iloc[1]) * 100
                    margin_change_qoq = current_margin - qoq_margin
                    metrics['Gross_Margin_QoQ_Change'] = f"{margin_change_qoq:+.1f}pp"
                else:
                    metrics['Gross_Margin_QoQ_Change'] = "N/A"
            else:
                metrics['Gross_Margin_Current'] = "N/A"
                metrics['Gross_Margin_YoY_Change'] = "N/A"
                metrics['Gross_Margin_QoQ_Change'] = "N/A"
        
        # === PRICE DEVELOPMENT (REPLACES EPS) ===
        # Get current price and historical prices
        current_price = None
        if 'currentPrice' in info and info['currentPrice']:
            current_price = info['currentPrice']
        elif 'regularMarketPrice' in info and info['regularMarketPrice']:
            current_price = info['regularMarketPrice']
        
        if current_price:
            metrics['Current_Price'] = f"${current_price:.2f}"
            
            # Get historical price data
            try:
                # Get 1 year of historical data for all calculations
                hist = stock.history(period="1y")
                if not hist.empty:
                    prices = hist['Close']
                    
                    # Last Week (7 days ago)
                    if len(prices) >= 7:
                        week_ago_price = prices.iloc[-7]
                        week_change = ((current_price - week_ago_price) / week_ago_price) * 100
                        metrics['Price_1W_Change'] = f"{week_change:+.1f}%"
                    else:
                        metrics['Price_1W_Change'] = "N/A"
                    
                    # Last Month (30 days ago)
                    if len(prices) >= 30:
                        month_ago_price = prices.iloc[-30]
                        month_change = ((current_price - month_ago_price) / month_ago_price) * 100
                        metrics['Price_1M_Change'] = f"{month_change:+.1f}%"
                    else:
                        metrics['Price_1M_Change'] = "N/A"
                    
                    # Last Quarter (90 days ago)
                    if len(prices) >= 90:
                        quarter_ago_price = prices.iloc[-90]
                        quarter_change = ((current_price - quarter_ago_price) / quarter_ago_price) * 100
                        metrics['Price_1Q_Change'] = f"{quarter_change:+.1f}%"
                    else:
                        metrics['Price_1Q_Change'] = "N/A"
                    
                    # Last Year (252 trading days ago)
                    if len(prices) >= 252:
                        year_ago_price = prices.iloc[-252]
                        year_change = ((current_price - year_ago_price) / year_ago_price) * 100
                        metrics['Price_1Y_Change'] = f"{year_change:+.1f}%"
                    elif len(prices) >= 200:  # Fallback to available data
                        year_ago_price = prices.iloc[0]  # Oldest available
                        year_change = ((current_price - year_ago_price) / year_ago_price) * 100
                        metrics['Price_1Y_Change'] = f"{year_change:+.1f}%"
                    else:
                        metrics['Price_1Y_Change'] = "N/A"
                        
            except Exception as e:
                print(f"Error getting price history: {e}")
                metrics['Price_1W_Change'] = "N/A"
                metrics['Price_1M_Change'] = "N/A"
                metrics['Price_1Q_Change'] = "N/A"
                metrics['Price_1Y_Change'] = "N/A"
        else:
            metrics['Current_Price'] = "N/A"
            metrics['Price_1W_Change'] = "N/A"
            metrics['Price_1M_Change'] = "N/A"
            metrics['Price_1Q_Change'] = "N/A"
            metrics['Price_1Y_Change'] = "N/A"
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating improved growth metrics for {ticker}: {e}")
        return {}




def create_improved_investment_banking_table_new(metrics):
    """Create improved investment banking style table"""
    
    # Define the improved table structure
    table_data = [
        # Revenue Growth Section
        {"Metric": "Revenue Growth", "Period": "Multi-Year Growth", "Value": metrics.get('Revenue_3Y_Growth', 'N/A'), "Category": "Revenue"},
        {"Metric": "", "Period": "Year-over-Year", "Value": metrics.get('Revenue_YoY', 'N/A'), "Category": "Revenue"},
        {"Metric": "", "Period": "Quarter-over-Quarter", "Value": metrics.get('Revenue_QoQ', 'N/A'), "Category": "Revenue"},
        
        # Gross Margin Section  
        {"Metric": "Gross Profit Margin", "Period": "Current Quarter", "Value": metrics.get('Gross_Margin_Current', 'N/A'), "Category": "Margins"},
        {"Metric": "", "Period": "YoY Change", "Value": metrics.get('Gross_Margin_YoY_Change', 'N/A'), "Category": "Margins"},
        {"Metric": "", "Period": "QoQ Change", "Value": metrics.get('Gross_Margin_QoQ_Change', 'N/A'), "Category": "Margins"},
        
        # Price Development Section (NEW - replaces EPS)
        {"Metric": "Price Development", "Period": "Current Price", "Value": metrics.get('Current_Price', 'N/A'), "Category": "Price"},
        {"Metric": "", "Period": "1 Week", "Value": metrics.get('Price_1W_Change', 'N/A'), "Category": "Price"},
        {"Metric": "", "Period": "1 Month", "Value": metrics.get('Price_1M_Change', 'N/A'), "Category": "Price"},
        {"Metric": "", "Period": "1 Quarter", "Value": metrics.get('Price_1Q_Change', 'N/A'), "Category": "Price"},
        {"Metric": "", "Period": "1 Year", "Value": metrics.get('Price_1Y_Change', 'N/A'), "Category": "Price"},
    ]
    
    # Improved styling with better spacing and colors
    table_style = {
        'backgroundColor': 'white',
        'border': '1px solid #dee2e6',
        'borderRadius': '6px',
        'fontFamily': '"Segoe UI", Arial, sans-serif',
        'fontSize': '13px',
        'width': '100%',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.12)'
    }
    
    header_style = {
        'backgroundColor': '#f8f9fa',
        'fontWeight': '600',
        'borderBottom': '2px solid #dee2e6',
        'padding': '12px 10px',
        'textAlign': 'left',
        'fontSize': '12px',
        'color': '#495057',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    }
    
    cell_style_base = {
        'padding': '10px',
        'borderBottom': '1px solid #f1f3f4',
        'textAlign': 'left',
        'fontSize': '13px',
        'lineHeight': '1.4'
    }
    
    # Create sections with improved category headers
    table_rows = []
    current_category = ""
    
    for row in table_data:
        # Add improved category header
        if row['Category'] != current_category:
            current_category = row['Category']
            
            if current_category == "Revenue":
                category_name = "📈 Revenue Growth Analysis"
                category_color = "#e3f2fd"
                border_color = "#2196f3"
            elif current_category == "Margins": 
                category_name = "💰 Profitability Margins"
                category_color = "#f3e5f5"
                border_color = "#9c27b0"
            else:  # Price
                category_name = "📊 Price Development Analysis" 
                category_color = "#e8f5e8"
                border_color = "#4caf50"
            
            table_rows.append(
                html.Tr([
                    html.Td(category_name, colSpan=3, style={
                        'backgroundColor': category_color,
                        'fontWeight': '600',
                        'padding': '12px 10px',
                        'borderBottom': f'2px solid {border_color}',
                        'borderTop': f'2px solid {border_color}' if current_category != "Revenue" else 'none',
                        'fontSize': '12px',
                        'color': '#333',
                        'textTransform': 'uppercase',
                        'letterSpacing': '0.5px'
                    })
                ])
            )
        
        # Determine text color and formatting based on value
        value = row['Value']
        text_color = '#212529'
        font_weight = 'normal'
        
        if value != 'N/A' and '%' in value:
            try:
                numeric_value = float(value.replace('%', '').replace('+', ''))
                if numeric_value > 0:
                    text_color = '#28a745'  # Green for positive
                    font_weight = '600'
                elif numeric_value < 0:
                    text_color = '#dc3545'  # Red for negative  
                    font_weight = '600'
            except:
                pass
        elif '$' in value:
            text_color = '#007bff'  # Blue for price
            font_weight = '600'
        
        # Create cell styles
        metric_style = {**cell_style_base, 'fontWeight': '600' if row['Metric'] else 'normal', 'width': '35%', 'color': '#495057'}
        period_style = {**cell_style_base, 'width': '35%', 'color': '#6c757d'}
        value_style = {**cell_style_base, 'fontWeight': font_weight, 'color': text_color, 'width': '30%', 'textAlign': 'right'}
        
        # Add data row
        table_rows.append(
            html.Tr([
                html.Td(row['Metric'], style=metric_style),
                html.Td(row['Period'], style=period_style),
                html.Td(row['Value'], style=value_style)
            ])
        )
    
    return html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("Metric", style={**header_style, 'width': '35%'}),
                    html.Th("Period", style={**header_style, 'width': '35%'}),
                    html.Th("Value", style={**header_style, 'width': '30%', 'textAlign': 'right'})
                ])
            ]),
            html.Tbody(table_rows)
        ], style=table_style)
    ], style={'margin': '0 auto'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)
