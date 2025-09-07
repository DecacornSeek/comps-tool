#!/usr/bin/env python3
"""
Improved metrics calculation with Price Development instead of EPS
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dash import html

def calculate_improved_growth_metrics(ticker):
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
        
        # YoY and QoQ calculations (existing logic)
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

def create_improved_investment_banking_table(metrics):
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

if __name__ == "__main__":
    # Test the improved metrics
    metrics = calculate_improved_growth_metrics('ZETA')
    print("Improved Growth Metrics:", metrics)
    
    table = create_improved_investment_banking_table(metrics)
    print("Table created successfully!")