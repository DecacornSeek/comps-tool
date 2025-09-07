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

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Peer Comparison Dashboard"

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

def create_growth_analysis_table(tickers):
    """Create simple table with YoY, QoQ growth and last 4 quarters"""
    
    all_data = []
    
    for ticker in tickers:
        historical_data = get_historical_metrics(ticker)
        if not historical_data or 'quarterly_data' not in historical_data:
            continue
            
        quarterly_data = historical_data['quarterly_data']
        
        # Key metrics to analyze
        metrics = {
            'Revenue': 'Revenue', 
            'Gross Profit': 'Gross Profit',
            'EBITDA': 'EBITDA',
            'Net Income': 'Net Income',
            'Free Cash Flow': 'Free Cash Flow'
        }
        
        for metric_name, metric_key in metrics.items():
            if metric_key not in quarterly_data:
                continue
                
            metric_dict = quarterly_data[metric_key]
            quarters = sorted(metric_dict.keys(), reverse=True)  # Most recent first
            
            if len(quarters) < 2:
                continue  # Need at least 2 quarters for QoQ
                
            # Get current quarter (Q0) and previous quarters
            current_q = quarters[0] if len(quarters) > 0 else None
            prev_q = quarters[1] if len(quarters) > 1 else None  # Q-1 for QoQ
            year_ago_q = quarters[4] if len(quarters) > 4 else None  # Q-4 for YoY
            
            # Create row
            row = {
                'Company': ticker,
                'Metric': metric_name
            }
            
            # Last 4 quarters
            for i, quarter in enumerate(quarters[:4]):
                quarter_num = (quarter.month - 1) // 3 + 1
                quarter_label = f"{quarter.year}Q{quarter_num}"
                value = metric_dict[quarter]
                
                if metric_name in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Free Cash Flow']:
                    row[quarter_label] = f"${value/1e6:.1f}M"
                else:
                    row[quarter_label] = f"${value:.2f}"
            
            # QoQ Growth (current vs previous quarter)
            if current_q and prev_q and prev_q in metric_dict and metric_dict[prev_q] != 0:
                current_val = metric_dict[current_q]
                prev_val = metric_dict[prev_q]
                qoq_growth = ((current_val - prev_val) / abs(prev_val)) * 100
                row['QoQ Growth %'] = f"{qoq_growth:+.1f}%"
            else:
                row['QoQ Growth %'] = "N/A"
            
            # YoY Growth (current vs same quarter last year)
            if current_q and year_ago_q and year_ago_q in metric_dict and metric_dict[year_ago_q] != 0:
                current_val = metric_dict[current_q]
                year_ago_val = metric_dict[year_ago_q]
                yoy_growth = ((current_val - year_ago_val) / abs(year_ago_val)) * 100
                row['YoY Growth %'] = f"{yoy_growth:+.1f}%"
            else:
                row['YoY Growth %'] = "N/A"
            
            all_data.append(row)
    
    if not all_data:
        return dbc.Alert("No historical data available for growth analysis", color="warning")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create table
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in df.columns],
        sort_action="native",
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '11px',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center'
        },
        style_data_conditional=[
            # Green for positive growth
            {
                'if': {'column_id': 'QoQ Growth %', 'filter_query': '{QoQ Growth %} contains "+"'},
                'color': 'green',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'YoY Growth %', 'filter_query': '{YoY Growth %} contains "+"'},
                'color': 'green', 
                'fontWeight': 'bold'
            },
            # Red for negative growth
            {
                'if': {'column_id': 'QoQ Growth %', 'filter_query': '{QoQ Growth %} contains "-"'},
                'color': 'red',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'YoY Growth %', 'filter_query': '{YoY Growth %} contains "-"'},
                'color': 'red',
                'fontWeight': 'bold'
            }
        ],
        style_table={'overflowX': 'auto'}
    )
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4("📊 Growth Analysis: YoY, QoQ & Last 4 Quarters", className="mb-0"),
            html.P("Quarter-over-Quarter, Year-over-Year growth rates and recent quarterly performance", className="text-muted mb-0 mt-1")
        ]),
        dbc.CardBody([table])
    ])

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
                        create_growth_analysis_table(tickers)
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
    
    return dbc.Alert("Tab not found", color="danger")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)