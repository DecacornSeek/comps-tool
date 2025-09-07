"""
Interactive Dashboard for Stock Analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table, callback
import dash_bootstrap_components as dbc

from comps_lib.metrics import fetch_metrics_for_ticker, COLUMNS
from comps_lib.analysis import (
    calculate_peer_comparison, 
    calculate_valuation_metrics, 
    generate_summary_stats
)
from comps_lib.io import write_outputs
from comps_lib.visualizations import (
    create_correlation_heatmap,
    create_valuation_radar_chart,
    create_risk_return_chart,
    create_efficiency_frontier,
    create_financial_health_score,
    create_peer_comparison_chart
)

# Konfiguration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dash App Initialisierung
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Stock Analysis Dashboard"

# Global Data Storage
GLOBAL_DATA = {
    'df': pd.DataFrame(),
    'analysis_data': {},
    'last_updated': None
}

# Helper Functions
def load_sample_data() -> pd.DataFrame:
    """Lädt Beispieldaten falls verfügbar."""
    csv_files = list(Path("/home/user/webapp").glob("*comps.csv"))
    if csv_files:
        try:
            return pd.read_csv(csv_files[-1])  # Neueste Datei
        except Exception as e:
            logger.warning(f"Failed to load {csv_files[-1]}: {e}")
    return pd.DataFrame()

def create_metrics_cards(df: pd.DataFrame) -> List[dbc.Col]:
    """Erstellt Metrik-Karten für das Dashboard."""
    if df.empty:
        return []
    
    cards = []
    
    # Gesamtanzahl Unternehmen
    total_companies = len(df)
    cards.append(
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{total_companies}", className="text-primary"),
                    html.P("Companies", className="card-text")
                ])
            ], className="text-center")
        ], width=3)
    )
    
    # Durchschnittliche Marktkapitalisierung
    if 'MarketCap' in df.columns:
        avg_mcap = pd.to_numeric(df['MarketCap'], errors='coerce').mean()
        if not pd.isna(avg_mcap):
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"${avg_mcap/1e9:.1f}B", className="text-success"),
                            html.P("Avg Market Cap", className="card-text")
                        ])
                    ], className="text-center")
                ], width=3)
            )
    
    # Anzahl Sektoren
    if 'Sector' in df.columns:
        sectors = df['Sector'].nunique()
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{sectors}", className="text-info"),
                        html.P("Sectors", className="card-text")
                    ])
                ], className="text-center")
            ], width=3)
        )
    
    # Durchschnittliches P/E Ratio
    if 'P/E (TTM)' in df.columns:
        avg_pe = pd.to_numeric(df['P/E (TTM)'], errors='coerce').mean()
        if not pd.isna(avg_pe):
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{avg_pe:.1f}", className="text-warning"),
                            html.P("Avg P/E Ratio", className="card-text")
                        ])
                    ], className="text-center")
                ], width=3)
            )
    
    return cards


def create_sector_chart(df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Sektorverteilungsdiagramm."""
    if df.empty or 'Sector' not in df.columns:
        return go.Figure()
    
    sector_counts = df['Sector'].value_counts()
    
    fig = px.pie(
        values=sector_counts.values,
        names=sector_counts.index,
        title="Sector Distribution"
    )
    
    fig.update_layout(
        height=400,
        showlegend=True
    )
    
    return fig


def create_valuation_scatter(df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Streudiagramm für Bewertungsmetriken."""
    if df.empty or 'P/E (TTM)' not in df.columns or 'P/B' not in df.columns:
        return go.Figure()
    
    # Filter out extreme values for better visualization
    df_clean = df.copy()
    pe_values = pd.to_numeric(df_clean['P/E (TTM)'], errors='coerce')
    pb_values = pd.to_numeric(df_clean['P/B'], errors='coerce')
    
    # Filter reasonable ranges
    df_clean = df_clean[
        (pe_values > 0) & (pe_values < 100) & 
        (pb_values > 0) & (pb_values < 50)
    ]
    
    if len(df_clean) == 0:
        return go.Figure()
    
    color_col = 'Sector' if 'Sector' in df_clean.columns else None
    
    fig = px.scatter(
        df_clean, 
        x='P/E (TTM)', 
        y='P/B',
        color=color_col,
        hover_data=['Ticker', 'MarketCap'] if 'MarketCap' in df_clean.columns else ['Ticker'],
        title="P/E vs P/B Ratio"
    )
    
    fig.update_layout(
        height=500,
        xaxis_title="P/E Ratio (TTM)",
        yaxis_title="Price-to-Book Ratio"
    )
    
    return fig


def create_performance_chart(df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Performancediagramm."""
    if df.empty:
        return go.Figure()
    
    # Verwende verfügbare Metriken
    metrics = []
    for col in ['ROE', 'ROA', 'ROIC', 'ProfitMargin']:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 0:
                metrics.append(col)
    
    if not metrics:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=metrics,
        specs=[[{"secondary_y": False} for _ in range(len(metrics))]]
    )
    
    for i, metric in enumerate(metrics, 1):
        values = pd.to_numeric(df[metric], errors='coerce')
        tickers = df['Ticker'][values.notna()]
        clean_values = values.dropna()
        
        fig.add_trace(
            go.Bar(
                x=tickers,
                y=clean_values,
                name=metric,
                showlegend=False
            ),
            row=1, col=i
        )
        
        # Update y-axis title
        fig.update_yaxes(title_text=metric, row=1, col=i)
        fig.update_xaxes(tickangle=45, row=1, col=i)
    
    fig.update_layout(
        height=400,
        title_text="Key Performance Metrics",
        showlegend=False
    )
    
    return fig

# Layout des Dashboards
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📊 Stock Analysis Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Ticker Input Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Add New Stocks"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id="ticker-input",
                                type="text",
                                placeholder="Enter ticker symbols (e.g., AAPL MSFT GOOGL)",
                                className="form-control",
                                style={"width": "100%"}
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button("Analyze", id="analyze-button", color="primary", className="w-100")
                        ], width=4)
                    ]),
                    html.Div(id="loading-status", className="mt-2")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Metrics Cards
    html.Div(id="metrics-cards-container"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="sector-chart")
        ], width=6),
        dbc.Col([
            dcc.Graph(id="valuation-chart")
        ], width=6)
    ], className="mb-4"),
    
    # Performance Chart
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="performance-chart")
        ])
    ], className="mb-4"),
    
    # Advanced Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Advanced Analysis"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(label="Risk vs Return", tab_id="risk-return"),
                        dbc.Tab(label="Value vs Quality", tab_id="value-quality"),
                        dbc.Tab(label="Financial Health", tab_id="health"),
                        dbc.Tab(label="Correlations", tab_id="correlations"),
                        dbc.Tab(label="Peer Analysis", tab_id="peer-analysis"),
                    ], id="advanced-tabs", active_tab="risk-return"),
                    html.Div(id="advanced-chart-content")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Ticker Selection for Peer Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select Ticker for Detailed Analysis:"),
                            dcc.Dropdown(
                                id="ticker-dropdown",
                                placeholder="Select a ticker...",
                                value=None
                            )
                        ], width=6),
                        dbc.Col([
                            html.Div(id="ticker-details")
                        ], width=6)
                    ])
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Data Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Stock Data"),
                dbc.CardBody([
                    html.Div(id="data-table-container")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Store for data
    dcc.Store(id='stock-data-store', data={}),
    
], fluid=True)


# Callbacks
@app.callback(
    [Output('stock-data-store', 'data'),
     Output('loading-status', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [dash.dependencies.State('ticker-input', 'value')]
)
def analyze_stocks(n_clicks, ticker_input):
    if n_clicks is None or not ticker_input:
        # Load existing data if available
        df = load_sample_data()
        if not df.empty:
            GLOBAL_DATA['df'] = df
            return df.to_dict('records'), ""
        return {}, ""
    
    # Parse ticker input
    tickers = [t.strip().upper() for t in ticker_input.replace(',', ' ').split() if t.strip()]
    
    if not tickers:
        return {}, dbc.Alert("Please enter valid ticker symbols", color="warning")
    
    try:
        # Fetch data for tickers
        loading_msg = dbc.Alert(f"Fetching data for {len(tickers)} stocks...", color="info")
        
        rows = []
        for ticker in tickers:
            try:
                row = fetch_metrics_for_ticker(ticker)
                rows.append(row)
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
                # Add error row
                error_row = {col: (ticker if col == "Ticker" else "ERROR") for col in COLUMNS}
                rows.append(error_row)
        
        df = pd.DataFrame(rows, columns=COLUMNS)
        
        # Store globally
        GLOBAL_DATA['df'] = df
        
        success_msg = dbc.Alert(f"Successfully loaded {len(df)} stocks!", color="success")
        
        return df.to_dict('records'), success_msg
        
    except Exception as e:
        error_msg = dbc.Alert(f"Error: {str(e)}", color="danger")
        return {}, error_msg


@app.callback(
    Output('metrics-cards-container', 'children'),
    [Input('stock-data-store', 'data')]
)
def update_metrics_cards(data):
    if not data:
        return html.Div()
    
    df = pd.DataFrame(data)
    cards = create_metrics_cards(df)
    
    if cards:
        return dbc.Row(cards, className="mb-4")
    else:
        return html.Div()


@app.callback(
    Output('sector-chart', 'figure'),
    [Input('stock-data-store', 'data')]
)
def update_sector_chart(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_sector_chart(df)


@app.callback(
    Output('valuation-chart', 'figure'),
    [Input('stock-data-store', 'data')]
)
def update_valuation_chart(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_valuation_scatter(df)


@app.callback(
    Output('performance-chart', 'figure'),
    [Input('stock-data-store', 'data')]
)
def update_performance_chart(data):
    if not data:
        return go.Figure()
    
    df = pd.DataFrame(data)
    return create_performance_chart(df)


@app.callback(
    Output('data-table-container', 'children'),
    [Input('stock-data-store', 'data')]
)
def update_data_table(data):
    if not data:
        return html.P("No data available. Please analyze some stocks first.")
    
    df = pd.DataFrame(data)
    
    # Select key columns for display
    display_cols = [
        'Ticker', 'Price', 'MarketCap', 'P/E (TTM)', 'P/B', 'ROE', 'ProfitMargin', 'Sector'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    if not display_cols:
        return html.P("No displayable data found.")
    
    display_df = df[display_cols].copy()
    
    # Format numeric columns
    numeric_cols = display_df.select_dtypes(include=[float, int]).columns
    for col in numeric_cols:
        if col == 'MarketCap':
            display_df[col] = display_df[col].apply(
                lambda x: f"${x/1e9:.1f}B" if pd.notna(x) and x > 0 else "N/A"
            )
        elif col in ['ROE', 'ProfitMargin']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
        else:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )
    
    return dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in display_df.columns],
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        page_size=10,
        sort_action="native",
        filter_action="native"
    )


@app.callback(
    Output('ticker-dropdown', 'options'),
    [Input('stock-data-store', 'data')]
)
def update_ticker_dropdown(data):
    if not data:
        return []
    
    df = pd.DataFrame(data)
    return [{'label': ticker, 'value': ticker} for ticker in sorted(df['Ticker'].unique())]


@app.callback(
    Output('advanced-chart-content', 'children'),
    [Input('advanced-tabs', 'active_tab'),
     Input('stock-data-store', 'data'),
     Input('ticker-dropdown', 'value')]
)
def update_advanced_charts(active_tab, data, selected_ticker):
    if not data:
        return html.P("No data available for advanced analysis.")
    
    df = pd.DataFrame(data)
    
    # Add calculated scores if not present
    if 'Valuation_Score' not in df.columns:
        df = calculate_valuation_metrics(df)
    
    if active_tab == "risk-return":
        fig = create_risk_return_chart(df)
        return dcc.Graph(figure=fig)
    
    elif active_tab == "value-quality":
        fig = create_efficiency_frontier(df)
        return dcc.Graph(figure=fig)
    
    elif active_tab == "health":
        fig = create_financial_health_score(df)
        return dcc.Graph(figure=fig)
    
    elif active_tab == "correlations":
        key_metrics = ['P/E (TTM)', 'P/B', 'ROE', 'ROA', 'ProfitMargin', 'RevenueGrowth', 'Beta']
        available_metrics = [m for m in key_metrics if m in df.columns]
        if available_metrics:
            fig = create_correlation_heatmap(df, available_metrics)
            return dcc.Graph(figure=fig)
        else:
            return html.P("Not enough numeric data for correlation analysis.")
    
    elif active_tab == "peer-analysis":
        if selected_ticker:
            peer_fig = create_peer_comparison_chart(df, selected_ticker)
            radar_fig = create_valuation_radar_chart(df, selected_ticker)
            
            return html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H5(f"Peer Comparison: {selected_ticker}"),
                        dcc.Graph(figure=peer_fig)
                    ], width=6),
                    dbc.Col([
                        html.H5(f"Performance Radar: {selected_ticker}"),
                        dcc.Graph(figure=radar_fig)
                    ], width=6)
                ])
            ])
        else:
            return html.P("Please select a ticker from the dropdown above for peer analysis.")
    
    return html.P("Select a tab to view analysis.")


@app.callback(
    Output('ticker-details', 'children'),
    [Input('ticker-dropdown', 'value'),
     Input('stock-data-store', 'data')]
)
def update_ticker_details(selected_ticker, data):
    if not selected_ticker or not data:
        return ""
    
    df = pd.DataFrame(data)
    
    if selected_ticker not in df['Ticker'].values:
        return ""
    
    ticker_data = df[df['Ticker'] == selected_ticker].iloc[0]
    
    # Key metrics to display
    key_info = []
    
    sector = ticker_data.get('Sector', 'N/A')
    industry = ticker_data.get('Industry', 'N/A')
    price = ticker_data.get('Price')
    market_cap = ticker_data.get('MarketCap')
    pe_ratio = ticker_data.get('P/E (TTM)')
    
    key_info.append(html.H6(f"{selected_ticker}", className="text-primary"))
    key_info.append(html.P(f"Sector: {sector}"))
    key_info.append(html.P(f"Industry: {industry}"))
    
    if pd.notna(price):
        key_info.append(html.P(f"Price: ${price:.2f}"))
    
    if pd.notna(market_cap):
        key_info.append(html.P(f"Market Cap: ${market_cap/1e9:.1f}B"))
    
    if pd.notna(pe_ratio):
        key_info.append(html.P(f"P/E Ratio: {pe_ratio:.2f}"))
    
    return key_info


if __name__ == '__main__':
    # Load initial data
    initial_df = load_sample_data()
    if not initial_df.empty:
        GLOBAL_DATA['df'] = initial_df
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)