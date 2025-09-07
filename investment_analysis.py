"""
Investment Manager Analysis Tool
Provides comprehensive analysis for investment decision making
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_investment_opportunity(ticker, peers=None):
    """
    Comprehensive investment analysis for professional decision making
    """
    
    print(f'📊 INVESTMENT ANALYSIS: {ticker}')
    print('=' * 60)
    print(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    
    # Get stock data
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Basic Info
    print(f'\n🏢 COMPANY OVERVIEW:')
    print(f'   Company: {info.get("longName", "N/A")}')
    print(f'   Sector: {info.get("sector", "N/A")}')
    print(f'   Industry: {info.get("industry", "N/A")}')
    print(f'   Market Cap: ${info.get("marketCap", 0)/1e9:.1f}B')
    print(f'   Employees: {info.get("fullTimeEmployees", "N/A"):,}')
    
    # Current Valuation
    print(f'\n📈 CURRENT VALUATION:')
    current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
    print(f'   Current Price: ${current_price:.2f}')
    print(f'   P/E (TTM): {info.get("trailingPE", "N/A"):.1f}' if info.get("trailingPE") else '   P/E (TTM): N/A')
    print(f'   P/E (Forward): {info.get("forwardPE", "N/A"):.1f}' if info.get("forwardPE") else '   P/E (Forward): N/A')
    print(f'   P/B Ratio: {info.get("priceToBook", "N/A"):.1f}' if info.get("priceToBook") else '   P/B Ratio: N/A')
    print(f'   EV/EBITDA: {info.get("enterpriseToEbitda", "N/A"):.1f}' if info.get("enterpriseToEbitda") else '   EV/EBITDA: N/A')
    
    # Profitability
    print(f'\n💰 PROFITABILITY:')
    print(f'   ROE: {info.get("returnOnEquity", 0)*100:.1f}%' if info.get("returnOnEquity") else '   ROE: N/A')
    print(f'   ROA: {info.get("returnOnAssets", 0)*100:.1f}%' if info.get("returnOnAssets") else '   ROA: N/A')
    print(f'   Profit Margin: {info.get("profitMargins", 0)*100:.1f}%' if info.get("profitMargins") else '   Profit Margin: N/A')
    print(f'   Operating Margin: {info.get("operatingMargins", 0)*100:.1f}%' if info.get("operatingMargins") else '   Operating Margin: N/A')
    
    # Growth
    print(f'\n📊 GROWTH METRICS:')
    print(f'   Revenue Growth (TTM): {info.get("revenueGrowth", 0)*100:.1f}%' if info.get("revenueGrowth") else '   Revenue Growth: N/A')
    print(f'   Earnings Growth (TTM): {info.get("earningsGrowth", 0)*100:.1f}%' if info.get("earningsGrowth") else '   Earnings Growth: N/A')
    print(f'   Revenue (TTM): ${info.get("totalRevenue", 0)/1e9:.1f}B' if info.get("totalRevenue") else '   Revenue: N/A')
    
    # Financial Health
    print(f'\n🏥 FINANCIAL HEALTH:')
    print(f'   Current Ratio: {info.get("currentRatio", "N/A"):.2f}' if info.get("currentRatio") else '   Current Ratio: N/A')
    print(f'   Debt/Equity: {info.get("debtToEquity", "N/A"):.1f}' if info.get("debtToEquity") else '   Debt/Equity: N/A')
    print(f'   Free Cash Flow: ${info.get("freeCashflow", 0)/1e9:.1f}B' if info.get("freeCashflow") else '   Free Cash Flow: N/A')
    print(f'   Cash per Share: ${info.get("totalCashPerShare", "N/A"):.2f}' if info.get("totalCashPerShare") else '   Cash per Share: N/A')
    
    # Historical Performance
    print(f'\n📈 PRICE PERFORMANCE:')
    try:
        hist = stock.history(period='2y')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
            price_1y = hist['Close'].iloc[-252] if len(hist) >= 252 else hist['Close'].iloc[0]
            price_6m = hist['Close'].iloc[-126] if len(hist) >= 126 else hist['Close'].iloc[0]
            price_3m = hist['Close'].iloc[-63] if len(hist) >= 63 else hist['Close'].iloc[0]
            
            print(f'   1Y Performance: {(current_price/price_1y - 1)*100:+.1f}%')
            print(f'   6M Performance: {(current_price/price_6m - 1)*100:+.1f}%')
            print(f'   3M Performance: {(current_price/price_3m - 1)*100:+.1f}%')
            
            # 52-week range
            high_52w = info.get('fiftyTwoWeekHigh', hist['High'].max())
            low_52w = info.get('fiftyTwoWeekLow', hist['Low'].min())
            print(f'   52W High: ${high_52w:.2f}')
            print(f'   52W Low: ${low_52w:.2f}')
            print(f'   Distance from High: {(current_price/high_52w - 1)*100:.1f}%')
    except:
        print('   Historical data not available')
    
    # Quarterly Trends
    print(f'\n📊 QUARTERLY TRENDS:')
    try:
        quarterly_financials = stock.quarterly_financials
        quarterly_earnings = stock.quarterly_earnings
        
        if not quarterly_financials.empty:
            revenue_cols = quarterly_financials.columns[:4]  # Last 4 quarters
            if 'Total Revenue' in quarterly_financials.index:
                revenues = quarterly_financials.loc['Total Revenue', revenue_cols]
                revenues = revenues.dropna()
                if len(revenues) >= 2:
                    latest_growth = (revenues.iloc[0] / revenues.iloc[1] - 1) * 100
                    print(f'   Latest Quarter Revenue Growth: {latest_growth:.1f}%')
                    
                    # YoY growth if 4+ quarters available
                    if len(revenues) >= 4:
                        yoy_growth = (revenues.iloc[0] / revenues.iloc[3] - 1) * 100
                        print(f'   YoY Revenue Growth: {yoy_growth:.1f}%')
        
        if not quarterly_earnings.empty and len(quarterly_earnings.columns) >= 4:
            earnings_cols = quarterly_earnings.columns[:4]
            eps_data = quarterly_earnings.loc['Earnings', earnings_cols] if 'Earnings' in quarterly_earnings.index else None
            if eps_data is not None:
                eps_data = eps_data.dropna()
                if len(eps_data) >= 4:
                    eps_trend = "Improving" if eps_data.iloc[0] > eps_data.iloc[3] else "Declining"
                    print(f'   EPS Trend (YoY): {eps_trend}')
                    
    except Exception as e:
        print('   Quarterly data analysis failed')
    
    # Peer Comparison
    if peers:
        print(f'\n🔍 PEER COMPARISON:')
        peer_data = []
        
        for peer_ticker in peers:
            try:
                peer_stock = yf.Ticker(peer_ticker)
                peer_info = peer_stock.info
                peer_data.append({
                    'Ticker': peer_ticker,
                    'P/E': peer_info.get('trailingPE'),
                    'ROE': peer_info.get('returnOnEquity', 0) * 100 if peer_info.get('returnOnEquity') else None,
                    'Revenue Growth': peer_info.get('revenueGrowth', 0) * 100 if peer_info.get('revenueGrowth') else None,
                    'Market Cap': peer_info.get('marketCap', 0) / 1e9
                })
            except:
                continue
        
        if peer_data:
            peer_df = pd.DataFrame(peer_data)
            
            # Add our stock for comparison
            our_data = {
                'Ticker': ticker,
                'P/E': info.get('trailingPE'),
                'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else None,
                'Revenue Growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else None,
                'Market Cap': info.get('marketCap', 0) / 1e9
            }
            
            all_data = pd.concat([pd.DataFrame([our_data]), peer_df], ignore_index=True)
            
            print(f'   {"Ticker":8s} {"P/E":>6s} {"ROE":>8s} {"Growth":>8s} {"MCap($B)":>10s}')
            print(f'   {"-"*45}')
            
            for _, row in all_data.iterrows():
                pe_str = f"{row['P/E']:.1f}" if pd.notna(row['P/E']) else "N/A"
                roe_str = f"{row['ROE']:.1f}%" if pd.notna(row['ROE']) else "N/A"
                growth_str = f"{row['Revenue Growth']:.1f}%" if pd.notna(row['Revenue Growth']) else "N/A"
                mcap_str = f"{row['Market Cap']:.1f}" if pd.notna(row['Market Cap']) else "N/A"
                highlight = " ←" if row['Ticker'] == ticker else ""
                
                print(f'   {row["Ticker"]:8s} {pe_str:>6s} {roe_str:>8s} {growth_str:>8s} {mcap_str:>10s}{highlight}')
            
            # Peer analysis
            peer_metrics = peer_df.dropna(subset=['P/E'])
            if len(peer_metrics) > 0 and pd.notna(our_data['P/E']):
                peer_avg_pe = peer_metrics['P/E'].mean()
                pe_vs_peers = (our_data['P/E'] - peer_avg_pe) / peer_avg_pe * 100
                valuation_assessment = "UNDERVALUED" if pe_vs_peers < -15 else "OVERVALUED" if pe_vs_peers > 15 else "FAIRLY VALUED"
                print(f'\n   📊 Valuation vs Peers: {pe_vs_peers:+.1f}% ({valuation_assessment})')
    
    # Investment Thesis
    print(f'\n🎯 INVESTMENT THESIS:')
    
    # Scoring system
    score = 0
    max_score = 0
    factors = []
    
    # Valuation Score
    pe_ratio = info.get('trailingPE')
    if pe_ratio:
        max_score += 1
        if pe_ratio < 15:
            score += 1
            factors.append("✅ Attractive valuation (P/E < 15)")
        elif pe_ratio < 25:
            score += 0.5
            factors.append("🟡 Reasonable valuation (P/E 15-25)")
        else:
            factors.append("⚠️ High valuation (P/E > 25)")
    
    # Profitability Score
    roe = info.get('returnOnEquity')
    if roe:
        max_score += 1
        if roe > 0.20:  # > 20%
            score += 1
            factors.append("✅ Excellent profitability (ROE > 20%)")
        elif roe > 0.10:  # > 10%
            score += 0.5
            factors.append("🟡 Good profitability (ROE > 10%)")
        else:
            factors.append("⚠️ Low profitability (ROE < 10%)")
    
    # Growth Score
    revenue_growth = info.get('revenueGrowth')
    if revenue_growth:
        max_score += 1
        if revenue_growth > 0.15:  # > 15%
            score += 1
            factors.append("✅ Strong growth (Revenue > 15%)")
        elif revenue_growth > 0.05:  # > 5%
            score += 0.5
            factors.append("🟡 Moderate growth (Revenue > 5%)")
        else:
            factors.append("⚠️ Slow growth (Revenue < 5%)")
    
    # Financial Health Score
    current_ratio = info.get('currentRatio')
    debt_equity = info.get('debtToEquity')
    if current_ratio or debt_equity:
        max_score += 1
        healthy = True
        if current_ratio and current_ratio < 1.0:
            healthy = False
        if debt_equity and debt_equity > 100:  # >100% debt/equity
            healthy = False
        
        if healthy:
            score += 1
            factors.append("✅ Strong financial health")
        else:
            factors.append("⚠️ Concerns about financial health")
    
    # Print factors
    for factor in factors:
        print(f'   {factor}')
    
    # Overall recommendation
    if max_score > 0:
        score_percentage = score / max_score
        if score_percentage >= 0.8:
            recommendation = "STRONG BUY 🚀"
        elif score_percentage >= 0.6:
            recommendation = "BUY ✅"
        elif score_percentage >= 0.4:
            recommendation = "HOLD 📊"
        else:
            recommendation = "AVOID ⚠️"
        
        print(f'\n   📊 Investment Score: {score:.1f}/{max_score} ({score_percentage:.0%})')
        print(f'   🎯 Recommendation: {recommendation}')
    
    # Key Risks
    print(f'\n⚠️ KEY RISKS TO MONITOR:')
    print(f'   • Sector cyclicality and market conditions')
    print(f'   • Competition from industry peers')
    print(f'   • Regulatory and geopolitical factors')
    print(f'   • Execution of growth strategy')
    
    print(f'\n' + '='*60)
    print(f'📝 This analysis is for informational purposes only.')
    print(f'   Consult financial advisors before making investment decisions.')

# Example usage
if __name__ == "__main__":
    # Analyze ASML with semiconductor peers
    analyze_investment_opportunity('ASML.AS', peers=['NVDA', 'TSM', 'QCOM', 'AMD', 'AVGO'])