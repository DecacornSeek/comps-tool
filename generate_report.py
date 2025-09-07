"""
Script to generate comprehensive PDF reports from stock analysis data
"""

import argparse
import logging
from pathlib import Path
import pandas as pd

from comps_lib.pdf_reports import generate_pdf_report
from comps_lib.analysis import calculate_valuation_metrics


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report from stock analysis data")
    parser.add_argument("--csv", dest="csv_path", required=True, help="Path to CSV file with stock data")
    parser.add_argument("--output", dest="output_path", help="Output PDF path (default: auto-generated)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("report_generator")
    
    try:
        # Lade Daten
        if not Path(args.csv_path).exists():
            logger.error(f"CSV file not found: {args.csv_path}")
            return 1
        
        df = pd.read_csv(args.csv_path)
        logger.info(f"Loaded {len(df)} companies from {args.csv_path}")
        
        # Bestimme Output-Pfad
        if args.output_path:
            output_path = args.output_path
        else:
            csv_stem = Path(args.csv_path).stem
            output_path = f"{csv_stem}_report.pdf"
        
        # Generiere Report
        logger.info(f"Generating PDF report: {output_path}")
        result_path = generate_pdf_report(df, output_path)
        
        logger.info(f"✅ PDF report generated successfully: {result_path}")
        logger.info(f"📊 Report includes analysis of {len(df)} companies")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to generate report: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())