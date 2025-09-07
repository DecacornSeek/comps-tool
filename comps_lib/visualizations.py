"""
Advanced visualization functions for stock analysis
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional, Tuple
import math


def create_correlation_heatmap(df: pd.DataFrame, metrics: List[str]) -> go.Figure:
    """Erstellt eine Korrelations-Heatmap für numerische Metriken."""
    # Filter numeric columns and remove rows with all NaN
    numeric_df = df[metrics].select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    fig = px.imshow(
        correlation_matrix,
        title="Correlation Matrix of Financial Metrics",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    # Add correlation values as text
    fig.update_traces(
        text=correlation_matrix.round(2),
        texttemplate="%{text}",
        textfont={"size": 10}
    )
    
    fig.update_layout(
        height=600,
        width=800
    )
    
    return fig


def create_valuation_radar_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Erstellt ein Radar-Chart für Bewertungsmetriken eines spezifischen Tickers."""
    if ticker not in df['Ticker'].values:
        return go.Figure()
    
    ticker_data = df[df['Ticker'] == ticker].iloc[0]
    
    # Bewertungsmetriken (niedrigere Werte sind besser)
    valuation_metrics = ['P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA']
    
    # Profitabilitätsmetriken (höhere Werte sind besser)  
    profitability_metrics = ['ROE', 'ROA', 'ROIC', 'ProfitMargin', 'OperatingMargin']
    
    # Wachstumsmetriken (höhere Werte sind besser)
    growth_metrics = ['RevenueGrowth', 'EarningsGrowth']
    
    all_metrics = valuation_metrics + profitability_metrics + growth_metrics
    available_metrics = [m for m in all_metrics if m in df.columns]
    
    if not available_metrics:
        return go.Figure()
    
    values = []
    labels = []
    
    for metric in available_metrics:
        ticker_value = pd.to_numeric(ticker_data.get(metric), errors='coerce')
        
        if pd.notna(ticker_value):
            # Berechne Perzentil im Vergleich zu allen anderen Aktien
            series_values = pd.to_numeric(df[metric], errors='coerce').dropna()
            
            if len(series_values) > 1:
                if metric in valuation_metrics:
                    # Für Bewertungsmetriken: niedrigere Werte = besseres Perzentil
                    percentile = (series_values > ticker_value).sum() / len(series_values) * 100
                else:
                    # Für andere Metriken: höhere Werte = besseres Perzentil
                    percentile = (series_values < ticker_value).sum() / len(series_values) * 100
                
                values.append(percentile)
                labels.append(metric)
    
    if not values:
        return go.Figure()
    
    # Schließe den Kreis
    values.append(values[0])
    labels.append(labels[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=ticker,
        line=dict(color='blue'),
        fillcolor='rgba(0,0,255,0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title=f"{ticker} - Performance Radar (Percentile Rankings)",
        height=500
    )
    
    return fig


def create_scatter_matrix(df: pd.DataFrame, metrics: List[str], color_by: str = 'Sector') -> go.Figure:
    """Erstellt eine Streudiagramm-Matrix für mehrere Metriken.""" 
    # Filter metrics that exist in dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) < 2:
        return go.Figure()
    
    # Limit to first 4 metrics for readability
    display_metrics = available_metrics[:4]
    
    # Clean data
    df_clean = df[display_metrics + [color_by, 'Ticker']].copy()
    
    # Remove rows with too many NaNs
    df_clean = df_clean.dropna(thresh=len(display_metrics)//2 + 1)
    
    if len(df_clean) == 0:
        return go.Figure()
    
    fig = px.scatter_matrix(
        df_clean,
        dimensions=display_metrics,
        color=color_by,
        hover_data=['Ticker'],
        title="Multi-Metric Scatter Matrix"
    )
    
    fig.update_layout(
        height=700,
        width=800
    )
    
    return fig


def create_risk_return_chart(df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Risiko-Rendite-Diagramm."""
    if 'Beta' not in df.columns:
        return go.Figure()
    
    # Verwende verschiedene Rendite-Metriken
    return_metrics = ['ROE', 'ROA', 'ROIC']
    return_metric = None
    
    for metric in return_metrics:
        if metric in df.columns:
            return_metric = metric
            break
    
    if not return_metric:
        return go.Figure()
    
    # Clean data
    df_clean = df[['Ticker', 'Beta', return_metric, 'Sector', 'MarketCap']].copy()
    df_clean = df_clean.dropna()
    
    if len(df_clean) == 0:
        return go.Figure()
    
    # Create bubble chart
    fig = px.scatter(
        df_clean,
        x='Beta',
        y=return_metric,
        size='MarketCap',
        color='Sector',
        hover_data=['Ticker'],
        title=f"Risk vs Return: Beta vs {return_metric}",
        labels={
            'Beta': 'Risk (Beta)',
            return_metric: f'Return ({return_metric})'
        }
    )
    
    # Add quadrant lines
    mean_beta = df_clean['Beta'].mean()
    mean_return = df_clean[return_metric].mean()
    
    fig.add_hline(y=mean_return, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=mean_beta, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add quadrant labels
    fig.add_annotation(
        x=mean_beta + (df_clean['Beta'].max() - mean_beta) * 0.8,
        y=mean_return + (df_clean[return_metric].max() - mean_return) * 0.8,
        text="High Risk<br>High Return",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.add_annotation(
        x=mean_beta - (mean_beta - df_clean['Beta'].min()) * 0.8,
        y=mean_return + (df_clean[return_metric].max() - mean_return) * 0.8,
        text="Low Risk<br>High Return",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_efficiency_frontier(df: pd.DataFrame) -> go.Figure:
    """Erstellt eine vereinfachte Effizienzgrenze basierend auf Bewertung vs. Qualität."""
    if 'Valuation_Score' not in df.columns or 'Quality_Score' not in df.columns:
        return go.Figure()
    
    df_clean = df[['Ticker', 'Valuation_Score', 'Quality_Score', 'Sector', 'MarketCap']].dropna()
    
    if len(df_clean) == 0:
        return go.Figure()
    
    fig = px.scatter(
        df_clean,
        x='Valuation_Score',
        y='Quality_Score', 
        size='MarketCap',
        color='Sector',
        hover_data=['Ticker'],
        title="Efficiency Frontier: Value vs Quality",
        labels={
            'Valuation_Score': 'Valuation Score (Higher = Better Value)',
            'Quality_Score': 'Quality Score (Higher = Better Quality)'
        }
    )
    
    # Add diagonal line for balanced value-quality
    min_score = min(df_clean['Valuation_Score'].min(), df_clean['Quality_Score'].min())
    max_score = max(df_clean['Valuation_Score'].max(), df_clean['Quality_Score'].max())
    
    fig.add_shape(
        type="line",
        x0=min_score, y0=min_score,
        x1=max_score, y1=max_score,
        line=dict(dash="dash", color="gray", width=2),
    )
    
    # Add quadrant labels
    mid_val = (df_clean['Valuation_Score'].max() + df_clean['Valuation_Score'].min()) / 2
    mid_qual = (df_clean['Quality_Score'].max() + df_clean['Quality_Score'].min()) / 2
    
    fig.add_annotation(
        x=df_clean['Valuation_Score'].max() * 0.9,
        y=df_clean['Quality_Score'].max() * 0.9,
        text="High Value<br>High Quality<br>💎",
        showarrow=False,
        bgcolor="rgba(144,238,144,0.8)"
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_financial_health_score(df: pd.DataFrame) -> go.Figure:
    """Erstellt ein Financial Health Score Dashboard."""
    # Definiere Komponenten des Financial Health Scores
    liquidity_metrics = ['CurrentRatio', 'QuickRatio']
    leverage_metrics = ['DebtToEquity'] 
    profitability_metrics = ['ROE', 'ROA', 'ProfitMargin']
    
    scores = []
    tickers = []
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        health_components = []
        
        # Liquidity Score (höher ist besser)
        liquidity_values = []
        for metric in liquidity_metrics:
            if metric in df.columns:
                value = pd.to_numeric(row.get(metric), errors='coerce')
                if pd.notna(value) and value > 0:
                    # Normalisiere: 1.0+ ist gut, cap bei 3.0
                    norm_value = min(value, 3.0) / 3.0
                    liquidity_values.append(norm_value)
        
        if liquidity_values:
            health_components.append(np.mean(liquidity_values))
        
        # Leverage Score (niedriger ist besser für D/E)
        for metric in leverage_metrics:
            if metric in df.columns:
                value = pd.to_numeric(row.get(metric), errors='coerce')
                if pd.notna(value) and value >= 0:
                    # Invertiere: niedrigere D/E ist besser, cap bei 2.0
                    norm_value = max(0, 1 - min(value, 2.0) / 2.0)
                    health_components.append(norm_value)
        
        # Profitability Score (höher ist besser)
        prof_values = []
        for metric in profitability_metrics:
            if metric in df.columns:
                value = pd.to_numeric(row.get(metric), errors='coerce')
                if pd.notna(value):
                    if metric in ['ROE', 'ROA']:
                        # ROE/ROA als Dezimal (0.15 = 15%)
                        norm_value = min(max(value, 0), 0.3) / 0.3
                    else:  # ProfitMargin
                        norm_value = min(max(value, 0), 0.4) / 0.4
                    prof_values.append(norm_value)
        
        if prof_values:
            health_components.append(np.mean(prof_values))
        
        if health_components:
            overall_score = np.mean(health_components) * 100
            scores.append(overall_score)
            tickers.append(ticker)
    
    if not scores:
        return go.Figure()
    
    # Erstelle horizontales Balkendiagramm
    fig = go.Figure()
    
    # Farbkodierung basierend auf Score
    colors = []
    for score in scores:
        if score >= 80:
            colors.append('green')
        elif score >= 60:
            colors.append('yellow')
        elif score >= 40:
            colors.append('orange')
        else:
            colors.append('red')
    
    fig.add_trace(go.Bar(
        y=tickers,
        x=scores,
        orientation='h',
        marker_color=colors,
        text=[f"{score:.0f}" for score in scores],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Financial Health Score by Company",
        xaxis_title="Health Score (0-100)",
        yaxis_title="Companies",
        height=max(400, len(tickers) * 25),
        showlegend=False
    )
    
    return fig


def create_peer_comparison_chart(df: pd.DataFrame, ticker: str, industry_field: str = 'Industry') -> go.Figure:
    """Erstellt ein Peer-Vergleichsdiagramm für einen spezifischen Ticker."""
    if ticker not in df['Ticker'].values:
        return go.Figure()
    
    ticker_row = df[df['Ticker'] == ticker].iloc[0]
    industry = ticker_row.get(industry_field)
    
    if pd.isna(industry):
        return go.Figure()
    
    # Finde Peers
    peers = df[df[industry_field] == industry]
    
    if len(peers) <= 1:
        return go.Figure()
    
    # Metriken für Vergleich
    comparison_metrics = ['P/E (TTM)', 'ROE', 'ProfitMargin', 'RevenueGrowth', 'DebtToEquity']
    available_metrics = [m for m in comparison_metrics if m in df.columns]
    
    if not available_metrics:
        return go.Figure()
    
    # Erstelle Subplot für jede Metrik
    fig = make_subplots(
        rows=1, cols=len(available_metrics),
        subplot_titles=available_metrics,
        shared_yaxes=True
    )
    
    for i, metric in enumerate(available_metrics, 1):
        peer_values = pd.to_numeric(peers[metric], errors='coerce').dropna()
        peer_tickers = peers['Ticker'][pd.to_numeric(peers[metric], errors='coerce').notna()]
        
        if len(peer_values) == 0:
            continue
        
        # Farbkodierung: Ziel-Ticker hervorheben
        colors = ['red' if t == ticker else 'lightblue' for t in peer_tickers]
        
        fig.add_trace(
            go.Bar(
                x=peer_tickers,
                y=peer_values,
                name=metric,
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=i
        )
        
        # Füge Durchschnittslinie hinzu
        mean_value = peer_values.mean()
        fig.add_hline(
            y=mean_value, 
            line_dash="dash", 
            line_color="gray",
            row=1, col=i
        )
        
        fig.update_xaxes(tickangle=45, row=1, col=i)
    
    fig.update_layout(
        title=f"Peer Comparison: {ticker} vs {industry} Industry",
        height=400,
        showlegend=False
    )
    
    return fig