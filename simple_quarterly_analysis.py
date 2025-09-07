"""
Simple Quarterly Analysis Dashboard
Shows QoQ and YoY data for key metrics in clean tables
"""

import pandas as pd
import yfinance as yf
from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import numpy as np

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Quarterly Financial Analysis"

def get_quarterly_data(ticker):
    """Get quarterly financial data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get quarterly financials
        quarterly_financials = stock.quarterly_financials
        quarterly_cashflow = stock.quarterly_cashflow
        
        if quarterly_financials.empty or quarterly_cashflow.empty:
            return None
            
        # Get last 5 quarters of data
        quarters = quarterly_financials.columns[:5]
        
        data = []
        for quarter in quarters:
            quarter_data = {'Quarter': quarter.strftime('%Y Q%s' % ((quarter.month-1)//3 + 1))}
            
            # Revenue
            revenue = quarterly_financials.loc['Total Revenue', quarter] if 'Total Revenue' in quarterly_financials.index else 0
            quarter_data['Revenue'] = revenue / 1e6  # Convert to millions
            
            # Gross Profit
            gross_profit = quarterly_financials.loc['Gross Profit', quarter] if 'Gross Profit' in quarterly_financials.index else 0
            quarter_data['Gross Profit'] = gross_profit / 1e6
            quarter_data['Gross Profit Margin'] = (gross_profit / revenue * 100) if revenue != 0 else 0
            
            # EBITDA (approximation)
            operating_income = quarterly_financials.loc['Operating Income', quarter] if 'Operating Income' in quarterly_financials.index else 0
            quarter_data['EBITDA'] = operating_income / 1e6  # Simplified
            quarter_data['EBITDA Margin'] = (operating_income / revenue * 100) if revenue != 0 else 0
            
            # Net Income
            net_income = quarterly_financials.loc['Net Income', quarter] if 'Net Income' in quarterly_financials.index else 0
            quarter_data['Net Income'] = net_income / 1e6
            quarter_data['Net Profit Margin'] = (net_income / revenue * 100) if revenue != 0 else 0
            
            # Free Cash Flow
            operating_cf = quarterly_cashflow.loc['Operating Cash Flow', quarter] if 'Operating Cash Flow' in quarterly_cashflow.index else 0
            capex = quarterly_cashflow.loc['Capital Expenditure', quarter] if 'Capital Expenditure' in quarterly_cashflow.index else 0
            fcf = operating_cf + capex  # CapEx is usually negative
            quarter_data['FCF'] = fcf / 1e6
            quarter_data['FCF Margin'] = (fcf / revenue * 100) if revenue != 0 else 0
            
            data.append(quarter_data)
            
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_growth_rates(df):
    """Calculate QoQ and YoY growth rates"""
    if df is None or len(df) < 2:
        return None
        
    growth_data = []
    
    for i in range(len(df)):
        quarter_data = {'Quarter': df.iloc[i]['Quarter']}
        
        # Current quarter values
        current = df.iloc[i]
        
        # QoQ Growth (compared to previous quarter)
        if i < len(df) - 1:
            previous = df.iloc[i + 1]
            for metric in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'FCF']:
                if previous[metric] != 0:
                    qoq_growth = ((current[metric] - previous[metric]) / abs(previous[metric])) * 100
                    quarter_data[f'{metric} QoQ%'] = round(qoq_growth, 1)
                else:
                    quarter_data[f'{metric} QoQ%'] = 'N/A'
        else:
            for metric in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'FCF']:
                quarter_data[f'{metric} QoQ%'] = 'N/A'
        
        # YoY Growth (compared to same quarter last year)
        if i < len(df) - 4:  # 4 quarters ago
            year_ago = df.iloc[i + 4]
            for metric in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'FCF']:
                if year_ago[metric] != 0:
                    yoy_growth = ((current[metric] - year_ago[metric]) / abs(year_ago[metric])) * 100
                    quarter_data[f'{metric} YoY%'] = round(yoy_growth, 1)
                else:
                    quarter_data[f'{metric} YoY%'] = 'N/A'
        else:
            for metric in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'FCF']:
                quarter_data[f'{metric} YoY%'] = 'N/A'
                
        growth_data.append(quarter_data)
    
    return pd.DataFrame(growth_data)

# Load AdTech peer data for tickers
def load_tickers():
    """Load tickers from the peer analysis file"""
    try:
        df = pd.read_csv('/home/user/webapp/zeta_adtech_analysis_enhanced.csv')
        return df['Ticker'].tolist()[:5]  # First 5 tickers
    except:
        return ['ZETA', 'TTD', 'NVDA', 'ASML', 'GOOGL']  # Default tickers

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("📊 Quarterly Financial Analysis", className="text-center mb-4"),
            html.P("QoQ and YoY growth analysis for key financial metrics", className="text-center text-muted"),
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id="ticker-input", placeholder="Enter ticker (e.g., ZETA)", value="ZETA"),
                dbc.Button("Analyze", id="analyze-btn", color="primary", n_clicks=0)
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="results-container")
        ])
    ])
], fluid=True)

@app.callback(
    Output("results-container", "children"),
    [Input("analyze-btn", "n_clicks")],
    [State("ticker-input", "value")]
)
def update_analysis(n_clicks, ticker):
    if not ticker or n_clicks == 0:
        return dbc.Alert("Enter a ticker and click Analyze", color="info")
    
    # Get quarterly data
    quarterly_df = get_quarterly_data(ticker.upper())
    if quarterly_df is None:
        return dbc.Alert(f"Could not fetch data for {ticker}", color="danger")
    
    # Calculate growth rates
    growth_df = calculate_growth_rates(quarterly_df)
    
    # Format the quarterly data for display
    display_df = quarterly_df.copy()
    for col in ['Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'FCF']:
        display_df[f'{col} ($M)'] = display_df[col].round(1)
        display_df = display_df.drop(columns=[col])
    
    # Format margins
    for col in ['Gross Profit Margin', 'EBITDA Margin', 'Net Profit Margin', 'FCF Margin']:
        display_df[f'{col} (%)'] = display_df[col].round(1)
        display_df = display_df.drop(columns=[col])
    
    return [
        dbc.Card([
            dbc.CardHeader([
                html.H4(f"📈 {ticker.upper()} - Quarterly Financial Data", className="mb-0")
            ]),
            dbc.CardBody([
                dash_table.DataTable(
                    data=display_df.to_dict('records'),
                    columns=[{"name": col, "id": col} for col in display_df.columns],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_table={'overflowX': 'auto'}
                )
            ])
        ], className="mb-4"),
        
        dbc.Card([
            dbc.CardHeader([
                html.H4(f"📊 {ticker.upper()} - Growth Rates (QoQ & YoY)", className="mb-0")
            ]),
            dbc.CardBody([
                dash_table.DataTable(
                    data=growth_df.to_dict('records') if growth_df is not None else [],
                    columns=[{"name": col, "id": col} for col in growth_df.columns] if growth_df is not None else [],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                    style_table={'overflowX': 'auto'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{' + col + '} > 0', 'column_id': col},
                            'color': 'green'
                        } for col in growth_df.columns if 'QoQ%' in col or 'YoY%' in col
                    ] + [
                        {
                            'if': {'filter_query': '{' + col + '} < 0', 'column_id': col},
                            'color': 'red'
                        } for col in growth_df.columns if 'QoQ%' in col or 'YoY%' in col
                    ] if growth_df is not None else []
                )
            ])
        ])
    ]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)