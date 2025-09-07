"""
Analysefunktionen für Branchenvergleich und Peer-Analysen
"""

import logging
import statistics
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def calculate_sector_stats(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Berechnet Branchenstatistiken für gegebene Metriken."""
    sector_stats = {}
    
    # Gruppiere nach Sektor
    if 'Sector' not in df.columns:
        return sector_stats
        
    for sector in df['Sector'].dropna().unique():
        sector_data = df[df['Sector'] == sector]
        sector_stats[sector] = {}
        
        for metric in metrics:
            if metric in df.columns:
                values = pd.to_numeric(sector_data[metric], errors='coerce').dropna()
                if len(values) > 0:
                    sector_stats[sector][metric] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()) if len(values) > 1 else 0.0,
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': len(values)
                    }
    
    return sector_stats


def calculate_industry_stats(df: pd.DataFrame, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Berechnet Industriestatistiken für gegebene Metriken."""
    industry_stats = {}
    
    # Gruppiere nach Industry
    if 'Industry' not in df.columns:
        return industry_stats
        
    for industry in df['Industry'].dropna().unique():
        industry_data = df[df['Industry'] == industry]
        industry_stats[industry] = {}
        
        for metric in metrics:
            if metric in df.columns:
                values = pd.to_numeric(industry_data[metric], errors='coerce').dropna()
                if len(values) > 0:
                    industry_stats[industry][metric] = {
                        'mean': float(values.mean()),
                        'median': float(values.median()),
                        'std': float(values.std()) if len(values) > 1 else 0.0,
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': len(values)
                    }
    
    return industry_stats


def add_percentile_rankings(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Fügt Perzentil-Rankings für jede Metrik hinzu."""
    result_df = df.copy()
    
    for metric in metrics:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            # Berechne Perzentile (0-100)
            percentiles = values.rank(pct=True) * 100
            result_df[f'{metric}_Percentile'] = percentiles.round(1)
    
    return result_df


def calculate_peer_comparison(df: pd.DataFrame, ticker: str, comparison_field: str = 'Industry') -> Dict[str, Any]:
    """Vergleicht einen Ticker mit seinen Peers basierend auf Industry/Sector."""
    if ticker not in df['Ticker'].values:
        return {}
    
    ticker_row = df[df['Ticker'] == ticker].iloc[0]
    comparison_value = ticker_row.get(comparison_field)
    
    if pd.isna(comparison_value):
        return {}
    
    # Finde Peers in derselben Gruppe
    peers = df[df[comparison_field] == comparison_value]
    
    if len(peers) <= 1:
        return {'message': f'No peers found in {comparison_field}: {comparison_value}'}
    
    # Relevante Metriken für Vergleich
    key_metrics = [
        'P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA', 'ROE', 'ROA', 'ROIC',
        'ProfitMargin', 'OperatingMargin', 'RevenueGrowth', 'EarningsGrowth',
        'DebtToEquity', 'CurrentRatio', 'FCF Yield', 'Beta'
    ]
    
    comparison_data = {
        'ticker': ticker,
        'comparison_field': comparison_field,
        'comparison_value': comparison_value,
        'peer_count': len(peers) - 1,  # Excluding the ticker itself
        'metrics': {}
    }
    
    for metric in key_metrics:
        if metric in df.columns:
            peer_values = pd.to_numeric(peers[metric], errors='coerce').dropna()
            ticker_value = pd.to_numeric(ticker_row.get(metric), errors='coerce')
            
            if len(peer_values) > 1 and not pd.isna(ticker_value):
                peer_mean = peer_values.mean()
                peer_median = peer_values.median()
                peer_std = peer_values.std()
                
                # Berechne relative Position
                percentile = (peer_values < ticker_value).sum() / len(peer_values) * 100
                
                comparison_data['metrics'][metric] = {
                    'ticker_value': float(ticker_value),
                    'peer_mean': float(peer_mean),
                    'peer_median': float(peer_median),
                    'peer_std': float(peer_std),
                    'percentile_rank': float(percentile),
                    'vs_mean_percent': float((ticker_value - peer_mean) / peer_mean * 100) if peer_mean != 0 else 0,
                    'vs_median_percent': float((ticker_value - peer_median) / peer_median * 100) if peer_median != 0 else 0
                }
    
    return comparison_data


def calculate_valuation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet zusätzliche Bewertungsmetriken und Scores."""
    result_df = df.copy()
    
    # Simple Valuation Score (niedrigere Werte = günstiger)
    valuation_metrics = ['P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA']
    valuation_scores = []
    
    for _, row in df.iterrows():
        scores = []
        for metric in valuation_metrics:
            value = pd.to_numeric(row.get(metric), errors='coerce')
            if not pd.isna(value) and value > 0:
                # Invertiere und normalisiere (niedriger = besser)
                series_values = pd.to_numeric(df[metric], errors='coerce')
                series_values = series_values[series_values > 0]  # Nur positive Werte
                if len(series_values) > 1:
                    percentile = (series_values < value).sum() / len(series_values)
                    # Invertiere: niedrige Bewertung = hoher Score
                    scores.append(1 - percentile)
        
        if scores:
            valuation_scores.append(statistics.mean(scores) * 100)
        else:
            valuation_scores.append(np.nan)
    
    result_df['Valuation_Score'] = valuation_scores
    
    # Quality Score basierend auf Profitabilität und Stabilität
    quality_metrics = ['ROE', 'ROA', 'ROIC', 'ProfitMargin', 'OperatingMargin']
    quality_scores = []
    
    for _, row in df.iterrows():
        scores = []
        for metric in quality_metrics:
            value = pd.to_numeric(row.get(metric), errors='coerce')
            if not pd.isna(value):
                # Normalisiere basierend auf Verteilung
                series_values = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(series_values) > 1:
                    percentile = (series_values < value).sum() / len(series_values)
                    scores.append(percentile)
        
        if scores:
            quality_scores.append(statistics.mean(scores) * 100)
        else:
            quality_scores.append(np.nan)
    
    result_df['Quality_Score'] = quality_scores
    
    # Growth Score
    growth_metrics = ['RevenueGrowth', 'EarningsGrowth']
    growth_scores = []
    
    for _, row in df.iterrows():
        scores = []
        for metric in growth_metrics:
            value = pd.to_numeric(row.get(metric), errors='coerce')
            if not pd.isna(value):
                series_values = pd.to_numeric(df[metric], errors='coerce').dropna()
                if len(series_values) > 1:
                    percentile = (series_values < value).sum() / len(series_values)
                    scores.append(percentile)
        
        if scores:
            growth_scores.append(statistics.mean(scores) * 100)
        else:
            growth_scores.append(np.nan)
    
    result_df['Growth_Score'] = growth_scores
    
    return result_df


def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generiert zusammenfassende Statistiken für den gesamten Datensatz."""
    summary = {
        'total_companies': len(df),
        'sectors': df['Sector'].nunique() if 'Sector' in df.columns else 0,
        'industries': df['Industry'].nunique() if 'Industry' in df.columns else 0,
        'countries': df['Country'].nunique() if 'Country' in df.columns else 0,
    }
    
    # Top Sektoren nach Anzahl Unternehmen
    if 'Sector' in df.columns:
        sector_counts = df['Sector'].value_counts().head(5)
        summary['top_sectors'] = sector_counts.to_dict()
    
    # Marktkapitalisierungsverteilung
    if 'MarketCap' in df.columns:
        market_caps = pd.to_numeric(df['MarketCap'], errors='coerce').dropna()
        if len(market_caps) > 0:
            summary['market_cap_stats'] = {
                'total': float(market_caps.sum()),
                'median': float(market_caps.median()),
                'largest': float(market_caps.max()),
                'smallest': float(market_caps.min())
            }
    
    return summary