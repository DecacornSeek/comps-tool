#!/usr/bin/env python3
"""
Test script to explore Yahoo Finance data for NVIDIA-style dashboard
Focus on last year and year before last data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

def get_last_two_years_data(ticker):
    """Get financial data for last year and year before last"""
    
    # Get ticker object
    stock = yf.Ticker(ticker)
    
    # Get current year
    current_year = datetime.now().year
    last_year = current_year - 1
    year_before_last = current_year - 2
    
    print(f"\n=== {ticker} Financial Data Analysis ===")
    print(f"Current Year: {current_year}")
    print(f"Last Year: {last_year}")
    print(f"Year Before Last: {year_before_last}")
    
    # Get financial statements
    try:
        # Annual financials
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        print("\n--- Available Annual Data Years ---")
        if not financials.empty:
            print(f"Financials columns: {list(financials.columns)}")
        if not balance_sheet.empty:
            print(f"Balance Sheet columns: {list(balance_sheet.columns)}")
        if not cash_flow.empty:
            print(f"Cash Flow columns: {list(cash_flow.columns)}")
            
        # Get quarterly data
        quarterly_financials = stock.quarterly_financials
        quarterly_balance_sheet = stock.quarterly_balance_sheet
        quarterly_cash_flow = stock.quarterly_cashflow
        
        print("\n--- Available Quarterly Data ---")
        if not quarterly_financials.empty:
            print(f"Quarterly Financials columns: {list(quarterly_financials.columns)}")
            
        # Get info and statistics
        info = stock.info
        print(f"\n--- Key Info Available ---")
        key_metrics = [
            'totalRevenue', 'revenueGrowth', 'grossMargins', 'operatingMargins', 
            'profitMargins', 'returnOnEquity', 'returnOnAssets', 'currentRatio',
            'debtToEquity', 'freeCashflow', 'earningsGrowth', 'revenueQuarterlyGrowth',
            'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
            'marketCap', 'enterpriseValue', 'beta'
        ]
        
        available_metrics = {}
        for metric in key_metrics:
            if metric in info:
                available_metrics[metric] = info[metric]
                print(f"{metric}: {info[metric]}")
                
        return {
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'quarterly_financials': quarterly_financials,
            'info': available_metrics,
            'last_year': last_year,
            'year_before_last': year_before_last
        }
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_growth_metrics(data):
    """Calculate growth metrics for last 2 years"""
    if not data:
        return {}
        
    metrics = {}
    
    # Revenue growth calculation
    financials = data['financials']
    if not financials.empty and 'Total Revenue' in financials.index:
        revenue_data = financials.loc['Total Revenue'].dropna()
        if len(revenue_data) >= 2:
            # Get most recent 2 years
            recent_revenue = revenue_data.iloc[:2]
            if len(recent_revenue) == 2:
                revenue_growth = ((recent_revenue.iloc[0] - recent_revenue.iloc[1]) / recent_revenue.iloc[1]) * 100
                metrics['revenue_growth_yoy'] = revenue_growth
                print(f"Revenue Growth YoY: {revenue_growth:.1f}%")
    
    # Get growth from info if available
    info = data['info']
    if 'revenueGrowth' in info and info['revenueGrowth'] is not None:
        metrics['revenue_growth_ttm'] = info['revenueGrowth'] * 100
        print(f"Revenue Growth TTM: {info['revenueGrowth'] * 100:.1f}%")
        
    if 'earningsGrowth' in info and info['earningsGrowth'] is not None:
        metrics['earnings_growth'] = info['earningsGrowth'] * 100  
        print(f"Earnings Growth: {info['earningsGrowth'] * 100:.1f}%")
        
    return metrics

def create_nvidia_style_metrics(ticker):
    """Create NVIDIA-style metrics using real data"""
    
    data = get_last_two_years_data(ticker)
    if not data:
        return None
        
    growth_metrics = calculate_growth_metrics(data)
    info = data['info']
    
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
        
    if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
        roe = info['returnOnEquity'] * 100
        quality_details.append(f"ROE: {roe:.1f}%")
        if roe > 20: quality_score += 5
        elif roe > 15: quality_score += 3
        elif roe > 10: quality_score += 1
    
    score_cards.append({
        'title': 'QUALITÄTS-CHECK',
        'score': str(quality_score),
        'max_score': '15',
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
            'title': 'Umsatz-Wachs. 2J',
            'score': f"{revenue_growth:.1f}%",
            'max_score': '',
            'color': 'success' if revenue_growth > 15 else ('primary' if revenue_growth > 5 else 'warning'),
            'details': ['Performance letztes Jahr', 'YoY Growth Rate']
        })
    
    # Growth Check Score
    growth_score = 0
    growth_details = []
    
    if revenue_growth is not None:
        growth_details.append(f"Umsatzwachstum: {revenue_growth:.1f}%")
        if revenue_growth > 20: growth_score += 7
        elif revenue_growth > 10: growth_score += 5
        elif revenue_growth > 0: growth_score += 3
    
    if 'earningsGrowth' in growth_metrics:
        earnings_growth = growth_metrics['earnings_growth']
        growth_details.append(f"EPS-Wachstum: {earnings_growth:.1f}%")
        if earnings_growth > 15: growth_score += 8
        elif earnings_growth > 5: growth_score += 5
        elif earnings_growth > 0: growth_score += 2
        
    score_cards.append({
        'title': 'WACHSTUMS-CHECK',
        'score': str(growth_score),
        'max_score': '15',
        'color': 'success' if growth_score >= 12 else ('warning' if growth_score >= 8 else 'danger'),
        'details': growth_details
    })
    
    # Overall AAQS Score (combination of quality and growth)
    overall_score = min(10, (quality_score + growth_score) // 3)
    score_cards.append({
        'title': 'AAQS Score',
        'score': str(overall_score),
        'max_score': '',
        'color': 'success' if overall_score >= 8 else ('warning' if overall_score >= 6 else 'danger'),
        'details': ['Gesamtbewertung']
    })
    
    # Detailed Metrics with Progress Bars
    detailed_metrics = []
    
    # Growth and Stability Section
    growth_section = []
    if revenue_growth is not None:
        percentage = min(100, max(0, (revenue_growth + 20) * 2))  # Scale -10 to 40% as 0 to 100%
        growth_section.append(('Umsatzwachstum 2 Jahre', f'{revenue_growth:.1f}%', percentage))
        
    if 'earningsGrowth' in growth_metrics:
        earnings_growth = growth_metrics['earningsGrowth']
        percentage = min(100, max(0, (earnings_growth + 20) * 2))
        growth_section.append(('EPS-Wachstum 2 Jahre', f'{earnings_growth:.1f}%', percentage))
    
    if growth_section:
        detailed_metrics.append(('Wachstum und Stabilität', growth_section))
    
    # Profitability Section  
    profitability_section = []
    if 'grossMargins' in info and info['grossMargins'] is not None:
        gross_margin = info['grossMargins'] * 100
        percentage = min(100, gross_margin)
        profitability_section.append(('Bruttomarge', f'{gross_margin:.1f}%', percentage))
        
    if 'operatingMargins' in info and info['operatingMargins'] is not None:
        op_margin = info['operatingMargins'] * 100
        percentage = min(100, max(0, op_margin * 2))  # Scale 0-50% as 0-100%
        profitability_section.append(('Betriebsmarge', f'{op_margin:.1f}%', percentage))
        
    if 'returnOnEquity' in info and info['returnOnEquity'] is not None:
        roe = info['returnOnEquity'] * 100
        percentage = min(100, max(0, roe * 3))  # Scale 0-33% as 0-100%
        profitability_section.append(('Eigenkapitalrendite', f'{roe:.1f}%', percentage))
        
    if profitability_section:
        detailed_metrics.append(('Rentabilität und Effizienz', profitability_section))
        
    # Financial Health Section
    financial_section = []
    if 'currentRatio' in info and info['currentRatio'] is not None:
        current_ratio = info['currentRatio']
        percentage = min(100, max(0, (current_ratio - 0.5) * 50))  # Scale 0.5-2.5 as 0-100%
        financial_section.append(('Liquiditätsgrad', f'{current_ratio:.2f}', percentage))
        
    if 'debtToEquity' in info and info['debtToEquity'] is not None:
        debt_equity = info['debtToEquity']
        percentage = max(0, 100 - debt_equity * 20)  # Lower debt is better
        financial_section.append(('Verschuldungsgrad', f'{debt_equity:.2f}', percentage))
        
    if financial_section:
        detailed_metrics.append(('Sicherheit und Bilanz', financial_section))
    
    print(f"\n=== Generated Metrics for {ticker} ===")
    print("Score Cards:")
    for card in score_cards:
        print(f"  {card['title']}: {card['score']}{('/' + card['max_score']) if card['max_score'] else ''}")
        
    print("\nDetailed Metrics:")
    for section_name, metrics in detailed_metrics:
        print(f"  {section_name}:")
        for name, value, percentage in metrics:
            print(f"    {name}: {value} ({percentage:.0f}%)")
    
    return {
        'score_cards': score_cards,
        'detailed_metrics': detailed_metrics,
        'raw_data': data
    }

if __name__ == "__main__":
    # Test with ZETA and NVDA
    for ticker in ['ZETA', 'NVDA']:
        print(f"\n{'='*50}")
        result = create_nvidia_style_metrics(ticker)
        if result:
            print(f"Successfully created NVIDIA-style metrics for {ticker}")
        else:
            print(f"Failed to get data for {ticker}")