"""
PDF Report Generation for Stock Analysis
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.colors import HexColor

from .analysis import calculate_valuation_metrics, generate_summary_stats, calculate_peer_comparison
from .visualizations import (
    create_correlation_heatmap, create_valuation_radar_chart, 
    create_risk_return_chart, create_efficiency_frontier,
    create_financial_health_score
)

logger = logging.getLogger(__name__)


class StockAnalysisReportGenerator:
    """Klasse zum Generieren von PDF-Berichten für Aktienanalysen."""
    
    def __init__(self, df: pd.DataFrame, output_path: str):
        self.df = df
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Erstellt angepasste Stile für den Report."""
        # Title Style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Heading Style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue
        )
        
        # Subheading Style
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.black
        )
    
    def create_summary_table(self, summary_stats: Dict[str, Any]) -> Table:
        """Erstellt eine Zusammenfassungstabelle."""
        data = [
            ['Metric', 'Value'],
            ['Total Companies', str(summary_stats.get('total_companies', 'N/A'))],
            ['Sectors', str(summary_stats.get('sectors', 'N/A'))],
            ['Industries', str(summary_stats.get('industries', 'N/A'))],
            ['Countries', str(summary_stats.get('countries', 'N/A'))]
        ]
        
        # Market Cap Stats hinzufügen wenn verfügbar
        if 'market_cap_stats' in summary_stats:
            mcap_stats = summary_stats['market_cap_stats']
            data.extend([
                ['Total Market Cap', f"${mcap_stats.get('total', 0)/1e12:.2f}T"],
                ['Median Market Cap', f"${mcap_stats.get('median', 0)/1e9:.1f}B"],
                ['Largest Company', f"${mcap_stats.get('largest', 0)/1e9:.1f}B"],
                ['Smallest Company', f"${mcap_stats.get('smallest', 0)/1e9:.1f}B"]
            ])
        
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def create_top_performers_table(self, df: pd.DataFrame, metric: str, n: int = 10) -> Optional[Table]:
        """Erstellt eine Tabelle mit Top-Performern für eine spezifische Metrik."""
        if metric not in df.columns:
            return None
        
        # Sortiere nach Metrik (absteigende Reihenfolge für die meisten Metriken)
        is_valuation_metric = metric in ['P/E (TTM)', 'P/B', 'P/S', 'EV/EBITDA', 'DebtToEquity']
        ascending = is_valuation_metric  # Für Bewertungsmetriken: niedrigere Werte sind besser
        
        sorted_df = df.sort_values(metric, ascending=ascending).head(n)
        
        # Erstelle Tabellendaten
        data = [['Rank', 'Ticker', 'Sector', metric]]
        
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            ticker = row.get('Ticker', 'N/A')
            sector = row.get('Sector', 'N/A')
            value = row.get(metric)
            
            if pd.notna(value):
                if metric in ['ROE', 'ROA', 'ROIC', 'ProfitMargin', 'OperatingMargin', 'RevenueGrowth', 'EarningsGrowth', 'DividendYield']:
                    formatted_value = f"{value:.1%}"
                elif metric == 'MarketCap':
                    formatted_value = f"${value/1e9:.1f}B"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = 'N/A'
            
            data.append([str(i), ticker, sector, formatted_value])
        
        if len(data) <= 1:  # Nur Header
            return None
        
        table = Table(data, colWidths=[0.5*inch, 1*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        return table
    
    def save_plotly_chart_as_image(self, fig: go.Figure, filename: str) -> Optional[str]:
        """Speichert ein Plotly-Chart als Bild und gibt den Pfad zurück."""
        try:
            # Erstelle temporären Pfad für Chart-Images
            charts_dir = Path(self.output_path).parent / "temp_charts"
            charts_dir.mkdir(exist_ok=True)
            
            image_path = charts_dir / f"{filename}.png"
            
            # Exportiere Chart als PNG
            fig.write_image(str(image_path), width=600, height=400, engine="kaleido")
            
            return str(image_path)
        except Exception as e:
            logger.error(f"Failed to save chart {filename}: {e}")
            return None
    
    def generate_report(self) -> str:
        """Generiert den vollständigen PDF-Report."""
        try:
            # Erstelle PDF-Dokument
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Story-Elemente sammeln
            story = []
            
            # Title Page
            story.append(Paragraph("Stock Analysis Report", self.title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['Normal']))
            story.append(Spacer(1, 24))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", self.heading_style))
            
            # Berechne erweiterte Metriken
            enhanced_df = calculate_valuation_metrics(self.df)
            summary_stats = generate_summary_stats(self.df)
            
            # Summary Table
            summary_table = self.create_summary_table(summary_stats)
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Key Insights
            story.append(Paragraph("Key Insights", self.subheading_style))
            insights = self.generate_key_insights(summary_stats, enhanced_df)
            for insight in insights:
                story.append(Paragraph(f"• {insight}", self.styles['Normal']))
            story.append(Spacer(1, 24))
            
            # Sector Analysis
            if 'top_sectors' in summary_stats:
                story.append(Paragraph("Sector Distribution", self.heading_style))
                sector_data = []
                for sector, count in summary_stats['top_sectors'].items():
                    sector_data.append([sector, str(count), f"{count/summary_stats['total_companies']*100:.1f}%"])
                
                if sector_data:
                    sector_table_data = [['Sector', 'Companies', 'Percentage']] + sector_data
                    sector_table = Table(sector_table_data, colWidths=[2*inch, 1*inch, 1*inch])
                    sector_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(sector_table)
                    story.append(Spacer(1, 24))
            
            # Top Performers Sections
            top_performers_metrics = {
                'ROE': 'Top ROE Performers',
                'ProfitMargin': 'Top Profit Margin Leaders',
                'RevenueGrowth': 'Top Revenue Growth',
                'P/E (TTM)': 'Best P/E Ratios (Value)',
                'MarketCap': 'Largest Companies'
            }
            
            for metric, title in top_performers_metrics.items():
                if metric in enhanced_df.columns:
                    story.append(Paragraph(title, self.heading_style))
                    top_table = self.create_top_performers_table(enhanced_df, metric, 10)
                    if top_table:
                        story.append(top_table)
                        story.append(Spacer(1, 12))
            
            # Charts Section (Neue Seite)
            story.append(PageBreak())
            story.append(Paragraph("Visual Analysis", self.heading_style))
            
            # Generiere und füge Charts hinzu
            charts_to_include = [
                ('sector_pie', 'Sector Distribution', self.create_sector_pie_chart()),
                ('risk_return', 'Risk vs Return Analysis', create_risk_return_chart(enhanced_df)),
                ('efficiency_frontier', 'Value vs Quality Matrix', create_efficiency_frontier(enhanced_df)),
                ('financial_health', 'Financial Health Scores', create_financial_health_score(enhanced_df))
            ]
            
            for chart_id, chart_title, fig in charts_to_include:
                if fig and fig.data:  # Check if chart has data
                    story.append(Paragraph(chart_title, self.subheading_style))
                    chart_path = self.save_plotly_chart_as_image(fig, chart_id)
                    if chart_path and Path(chart_path).exists():
                        try:
                            img = Image(chart_path, width=5*inch, height=3.33*inch)
                            story.append(img)
                            story.append(Spacer(1, 12))
                        except Exception as e:
                            logger.warning(f"Failed to add chart {chart_title}: {e}")
            
            # Data Appendix (Neue Seite)
            story.append(PageBreak())
            story.append(Paragraph("Data Appendix", self.heading_style))
            story.append(Paragraph("Complete dataset used for analysis:", self.subheading_style))
            
            # Full data table (ausgewählte Spalten)
            display_cols = ['Ticker', 'Sector', 'Price', 'MarketCap', 'P/E (TTM)', 'P/B', 'ROE', 'ProfitMargin']
            available_cols = [col for col in display_cols if col in self.df.columns]
            
            if available_cols:
                # Erstelle Datentabelle
                table_data = [available_cols]  # Header
                
                for _, row in self.df.iterrows():
                    row_data = []
                    for col in available_cols:
                        value = row.get(col)
                        if pd.notna(value):
                            if col in ['ROE', 'ProfitMargin'] and isinstance(value, (int, float)):
                                formatted_value = f"{value:.1%}"
                            elif col == 'MarketCap' and isinstance(value, (int, float)):
                                formatted_value = f"${value/1e9:.1f}B"
                            elif isinstance(value, (int, float)):
                                formatted_value = f"{value:.2f}"
                            else:
                                formatted_value = str(value)
                        else:
                            formatted_value = 'N/A'
                        row_data.append(formatted_value)
                    table_data.append(row_data)
                
                # Berechne Spaltenbreiten dynamisch
                col_widths = [A4[0] / len(available_cols) - 10] * len(available_cols)
                
                data_table = Table(table_data, colWidths=col_widths)
                data_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                
                story.append(data_table)
            
            # Build PDF
            doc.build(story)
            
            # Cleanup temporary chart images
            charts_dir = Path(self.output_path).parent / "temp_charts"
            if charts_dir.exists():
                for chart_file in charts_dir.glob("*.png"):
                    try:
                        chart_file.unlink()
                    except Exception:
                        pass
                try:
                    charts_dir.rmdir()
                except Exception:
                    pass
            
            logger.info(f"PDF report generated successfully: {self.output_path}")
            return self.output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise
    
    def generate_key_insights(self, summary_stats: Dict[str, Any], enhanced_df: pd.DataFrame) -> List[str]:
        """Generiert Key Insights basierend auf den Daten."""
        insights = []
        
        # Portfolio Size Insight
        total_companies = summary_stats.get('total_companies', 0)
        if total_companies > 0:
            insights.append(f"Analysis covers {total_companies} companies across {summary_stats.get('sectors', 0)} sectors")
        
        # Market Cap Insight
        if 'market_cap_stats' in summary_stats:
            mcap_stats = summary_stats['market_cap_stats']
            total_mcap = mcap_stats.get('total', 0)
            if total_mcap > 0:
                insights.append(f"Total market capitalization: ${total_mcap/1e12:.2f} trillion")
        
        # Sector Concentration
        if 'top_sectors' in summary_stats and summary_stats['top_sectors']:
            top_sector = list(summary_stats['top_sectors'].keys())[0]
            top_sector_count = summary_stats['top_sectors'][top_sector]
            concentration = top_sector_count / total_companies * 100
            insights.append(f"{top_sector} dominates the portfolio with {concentration:.1f}% of companies")
        
        # Valuation Insights
        if 'P/E (TTM)' in enhanced_df.columns:
            pe_values = pd.to_numeric(enhanced_df['P/E (TTM)'], errors='coerce').dropna()
            if len(pe_values) > 0:
                avg_pe = pe_values.mean()
                insights.append(f"Average P/E ratio: {avg_pe:.1f}")
        
        # Profitability Insights
        if 'ROE' in enhanced_df.columns:
            roe_values = pd.to_numeric(enhanced_df['ROE'], errors='coerce').dropna()
            if len(roe_values) > 0:
                avg_roe = roe_values.mean()
                high_roe_count = (roe_values > 0.15).sum()  # 15%+ ROE
                insights.append(f"Average ROE: {avg_roe:.1%}, {high_roe_count} companies with ROE > 15%")
        
        # Growth Insights
        if 'RevenueGrowth' in enhanced_df.columns:
            growth_values = pd.to_numeric(enhanced_df['RevenueGrowth'], errors='coerce').dropna()
            if len(growth_values) > 0:
                avg_growth = growth_values.mean()
                insights.append(f"Average revenue growth: {avg_growth:.1%}")
        
        return insights
    
    def create_sector_pie_chart(self) -> Optional[go.Figure]:
        """Erstellt ein Kreisdiagramm für die Sektorverteilung."""
        if 'Sector' not in self.df.columns:
            return None
        
        sector_counts = self.df['Sector'].value_counts()
        
        if len(sector_counts) == 0:
            return None
        
        fig = go.Figure(data=[go.Pie(
            labels=sector_counts.index,
            values=sector_counts.values,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Sector Distribution",
            showlegend=True,
            width=600,
            height=400
        )
        
        return fig


def generate_pdf_report(df: pd.DataFrame, output_path: str) -> str:
    """Convenience-Funktion zum Generieren eines PDF-Reports."""
    generator = StockAnalysisReportGenerator(df, output_path)
    return generator.generate_report()