#!/usr/bin/env python3
"""
Evaluate Covered Calls - Analyze covered call spread opportunities from CSV data.

This script reads CSV data containing covered call spread information and provides
detailed analysis including risk metrics, profitability metrics, and recommendations.

Usage:
    python scripts/evaluate_covered_calls.py --file results.csv
    python scripts/evaluate_covered_calls.py < results.csv
    cat results.csv | python scripts/evaluate_covered_calls.py
    python scripts/evaluate_covered_calls.py --file results.csv --html --output-dir results_html
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Load CSV data from file or stdin.
    
    Args:
        file_path: Path to CSV file, or None to read from stdin
        
    Returns:
        DataFrame with loaded and cleaned data
    """
    if file_path is None or file_path == '-':
        # Read from stdin
        df = pd.read_csv(sys.stdin)
    else:
        # Read from file
        df = pd.read_csv(file_path)
    
    # Clean up the data - remove duplicate header rows
    df = df[df['ticker'] != 'ticker']
    
    # Convert numeric columns
    numeric_cols = [
        'pe_ratio', 'market_cap_b', 'current_price', 'strike_price', 'price_above_current',
        'option_premium', 'option_premium_percentage', 'delta', 'theta', 'days_to_expiry',
        'short_premium_total', 'short_daily_premium', 'long_strike_price', 'long_option_premium',
        'long_delta', 'long_theta', 'long_days_to_expiry', 'long_premium_total', 'premium_diff',
        'net_premium', 'net_daily_premium', 'volume', 'num_contracts'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def print_top_20_analysis(df: pd.DataFrame) -> None:
    """Print analysis for top 20 performers by net daily premium."""
    top_20 = df.nlargest(20, 'net_daily_premium')
    
    print("=" * 120)
    print("DEEP DIVE: TOP 20 COVERED CALL SPREADS BY NET DAILY PREMIUM")
    print("=" * 120)
    
    # Create comprehensive analysis
    analysis = top_20[['ticker', 'current_price', 'strike_price', 'net_daily_premium', 
                       'volume', 'pe_ratio', 'market_cap_b']].copy()
    
    # Calculate key risk metrics
    analysis['moneyness'] = ((top_20['strike_price'] - top_20['current_price']) / top_20['current_price'] * 100).round(2)
    analysis['short_delta'] = top_20['delta']
    analysis['long_delta'] = top_20['long_delta']
    analysis['delta_diff'] = (top_20['long_delta'] - top_20['delta']).round(3)
    analysis['theta_decay_daily'] = top_20['theta']
    analysis['premium_capture_pct'] = top_20['option_premium_percentage']
    analysis['spread_width'] = top_20['long_strike_price'] - top_20['strike_price']
    analysis['net_premium'] = top_20['net_premium']
    analysis['roi_on_spread'] = ((top_20['net_premium'] / 100000) * 100).round(2)
    
    print("\n### RISK METRICS OVERVIEW ###\n")
    print(analysis[['ticker', 'moneyness', 'short_delta', 'long_delta', 'delta_diff', 
                    'theta_decay_daily', 'volume']].to_string(index=False))
    
    print("\n\n### PROFITABILITY METRICS ###\n")
    print(analysis[['ticker', 'net_daily_premium', 'net_premium', 'roi_on_spread', 
                    'premium_capture_pct', 'spread_width']].to_string(index=False))
    
    print("\n\n### FUNDAMENTAL QUALITY ###\n")
    print(analysis[['ticker', 'pe_ratio', 'market_cap_b', 'current_price']].to_string(index=False))


def print_detailed_analysis(df: pd.DataFrame) -> None:
    """Print detailed analysis for top 10 picks."""
    print("\n\n" + "=" * 120)
    print("DETAILED ANALYSIS: TOP 10 PICKS WITH OPTION TICKERS")
    print("=" * 120)
    
    top_10 = df.nlargest(10, 'net_daily_premium')
    
    for idx, row in top_10.iterrows():
        ticker = row['ticker']
        print(f"\n{'=' * 110}")
        print(f"#{top_10.index.get_loc(idx) + 1}: {ticker}")
        print(f"{'=' * 110}")
        
        # Position Details
        print(f"\n📊 POSITION STRUCTURE:")
        print(f"   Current Price: ${row['current_price']:.2f}")
        print(f"   Short Strike: ${row['strike_price']:.2f} ({((row['strike_price']-row['current_price'])/row['current_price']*100):.2f}% OTM)")
        print(f"   Long Strike: ${row['long_strike_price']:.2f}")
        print(f"   Spread Width: ${row['long_strike_price'] - row['strike_price']:.2f}")
        
        # Option Tickers - HIGHLIGHTED
        print(f"\n🎯 OPTION TICKERS:")
        print(f"   ┌─ SHORT (SELL): {row['option_ticker']}")
        print(f"   │  Strike: ${row['strike_price']:.2f} | Expiry: {row['expiration_date'][:10]} ({int(row['days_to_expiry'])} DTE)")
        print(f"   │  Premium: ${row['option_premium']:.2f} per contract")
        print(f"   │  Total Credit: ${row['short_premium_total']:,.0f} ({int(row['num_contracts'])} contracts)")
        print(f"   │")
        print(f"   └─ LONG (BUY):  {row['long_option_ticker']}")
        print(f"      Strike: ${row['long_strike_price']:.2f} | Expiry: {row['long_expiration_date'][:10]} ({int(row['long_days_to_expiry'])} DTE)")
        print(f"      Premium: ${row['long_option_premium']:.2f} per contract")
        print(f"      Total Debit: ${row['long_premium_total']:,.0f} ({int(row['num_contracts'])} contracts)")
        
        # Premium Analysis
        print(f"\n💰 PREMIUM BREAKDOWN:")
        print(f"   Short Premium: ${row['short_premium_total']:,.0f} (${row['short_daily_premium']:,.0f}/day)")
        print(f"   Long Premium: ${row['long_premium_total']:,.0f}")
        print(f"   Net Credit: ${row['net_premium']:,.0f}")
        print(f"   Daily Income: ${row['net_daily_premium']:,.2f}")
        print(f"   ROI on $100k: {(row['net_premium']/100000*100):.2f}%")
        
        # Greeks & Risk
        print(f"\n📈 GREEKS & RISK:")
        print(f"   Short Delta: {row['delta']:.2f} | Long Delta: {row['long_delta']:.2f} | Net: {row['long_delta']-row['delta']:.3f}")
        print(f"   Short Theta: {row['theta']:.2f} | Long Theta: {row['long_theta']:.2f}")
        print(f"   Assignment Risk: {'LOW' if row['delta'] < 0.35 else 'MODERATE' if row['delta'] < 0.50 else 'HIGH'}")
        
        # Liquidity & Fundamentals
        print(f"\n🔄 LIQUIDITY & FUNDAMENTALS:")
        print(f"   Volume: {row['volume']:,.0f} contracts")
        print(f"   Num Contracts: {row['num_contracts']:.0f}")
        print(f"   P/E Ratio: {row['pe_ratio']:.2f}")
        print(f"   Market Cap: ${row['market_cap_b']:.2f}B")
        
        # Risk Assessment
        print(f"\n⚠️  RISK ASSESSMENT:")
        
        # Assignment risk
        if row['delta'] < 0.35:
            assignment_risk = "LOW - Strike is well OTM"
        elif row['delta'] < 0.50:
            assignment_risk = "MODERATE - Near ATM, watch closely"
        else:
            assignment_risk = "HIGH - ITM or very close, likely assignment"
        print(f"   Assignment Risk: {assignment_risk}")
        
        # Liquidity risk
        if row['volume'] > 1000:
            liquidity = "EXCELLENT - Very liquid"
        elif row['volume'] > 300:
            liquidity = "GOOD - Adequate liquidity"
        elif row['volume'] > 100:
            liquidity = "FAIR - May have wider spreads"
        else:
            liquidity = "POOR - Low liquidity, watch bid-ask"
        print(f"   Liquidity: {liquidity}")
        
        # Valuation
        if row['pe_ratio'] < 15:
            valuation = "ATTRACTIVE - Trading at discount"
        elif row['pe_ratio'] < 25:
            valuation = "FAIR - Reasonably valued"
        elif row['pe_ratio'] < 50:
            valuation = "ELEVATED - Premium valuation"
        else:
            valuation = "EXPENSIVE - Very high P/E"
        print(f"   Valuation: {valuation}")
        
        # Overall score
        score = 0
        if row['net_daily_premium'] > 10000: score += 3
        elif row['net_daily_premium'] > 7000: score += 2
        else: score += 1
        
        if row['volume'] > 1000: score += 3
        elif row['volume'] > 300: score += 2
        elif row['volume'] > 100: score += 1
        
        if row['delta'] < 0.35: score += 3
        elif row['delta'] < 0.50: score += 2
        else: score += 1
        
        if row['pe_ratio'] < 25: score += 2
        elif row['pe_ratio'] < 50: score += 1
        
        print(f"\n⭐ OVERALL SCORE: {score}/11")
        
        if score >= 9:
            recommendation = "STRONG BUY - Excellent risk/reward"
        elif score >= 7:
            recommendation = "BUY - Good opportunity"
        elif score >= 5:
            recommendation = "HOLD - Acceptable but monitor"
        else:
            recommendation = "PASS - Better opportunities available"
        
        print(f"   Recommendation: {recommendation}")


def generate_html_output(df: pd.DataFrame, output_dir: str) -> None:
    """Generate HTML output with sortable table.
    
    Args:
        df: DataFrame with the results
        output_dir: Directory path where to create the HTML output
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for HTML table
    df_display = df.copy()
    
    # Format numeric columns for better display
    numeric_cols = [
        'pe_ratio', 'market_cap_b', 'current_price', 'strike_price', 'price_above_current',
        'option_premium', 'option_premium_percentage', 'delta', 'theta', 'days_to_expiry',
        'short_premium_total', 'short_daily_premium', 'long_strike_price', 'long_option_premium',
        'long_delta', 'long_theta', 'long_days_to_expiry', 'long_premium_total', 'premium_diff',
        'net_premium', 'net_daily_premium', 'volume', 'num_contracts'
    ]
    
    def format_numeric_value(x, col_name):
        """Format numeric value based on column type."""
        if pd.isna(x) or x == '' or x is None:
            return ''
        try:
            val = float(x)
            if 'premium' in col_name.lower() or 'price' in col_name.lower() or 'cap' in col_name.lower():
                return f"${val:,.2f}"
            elif 'ratio' in col_name.lower() or 'delta' in col_name.lower() or 'theta' in col_name.lower():
                return f"{val:.2f}"
            elif 'days' in col_name.lower() or 'volume' in col_name.lower() or 'contracts' in col_name.lower():
                return f"{int(val):,}"
            elif 'percentage' in col_name.lower():
                return f"{val:.2f}%"
            else:
                return f"{val:.2f}"
        except (ValueError, TypeError):
            return str(x) if x != '' else ''
    
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: format_numeric_value(x, col))
    
    # Replace remaining NaN with empty strings for display
    df_display = df_display.fillna('')
    
    # Get current timestamp for display
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate HTML - build it piece by piece to avoid f-string issues
    html_parts = []
    
    # HTML head and styles
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Covered Calls Analysis Results</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .table-wrapper {
            overflow-x: auto;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }
        
        thead {
            background: #f8f9fa;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        th {
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
            position: relative;
        }
        
        th:hover {
            background: #e9ecef;
        }
        
        th.sortable::after {
            content: ' ↕';
            opacity: 0.5;
            font-size: 0.8em;
        }
        
        th.sort-asc::after {
            content: ' ↑';
            opacity: 1;
            color: #667eea;
        }
        
        th.sort-desc::after {
            content: ' ↓';
            opacity: 1;
            color: #667eea;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
            color: #495057;
        }
        
        tbody tr:hover {
            background: #f8f9fa;
        }
        
        tbody tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        tbody tr:nth-child(even):hover {
            background: #e9ecef;
        }
        
        .stats {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .stat-item {
            text-align: center;
            padding: 10px;
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            table {
                font-size: 0.8em;
            }
            
            th, td {
                padding: 8px 6px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Covered Calls Analysis Results</h1>
            <p>Generated: """ + timestamp + """</p>
            <p>Click column headers to sort • """ + str(len(df)) + """ total results</p>
        </div>
        
        <div class="table-wrapper">
            <table id="resultsTable">
                <thead>
                    <tr>
""")
    
    # Generate table headers
    for col in df_display.columns:
        col_index = df_display.columns.get_loc(col)
        col_title = col.replace("_", " ").title()
        html_parts.append(f'                        <th class="sortable" onclick="sortTable({col_index})">{col_title}</th>\n')
    
    html_parts.append("""                    </tr>
                </thead>
                <tbody>
""")
    
    # Generate table rows
    for _, row in df_display.iterrows():
        html_parts.append("                    <tr>\n")
        for col in df_display.columns:
            cell_value = str(row[col]) if pd.notna(row[col]) else ''
            # Escape HTML special characters
            cell_value = cell_value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_parts.append(f'                        <td>{cell_value}</td>\n')
        html_parts.append("                    </tr>\n")
    
    html_parts.append("""                </tbody>
            </table>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value" id="totalCount">""" + str(len(df)) + """</div>
                <div class="stat-label">Total Results</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="visibleCount">""" + str(len(df)) + """</div>
                <div class="stat-label">Visible Rows</div>
            </div>
        </div>
    </div>
    
    <script>
        let sortDirection = {};
        let currentSortColumn = -1;
        
        function sortTable(columnIndex) {
            const table = document.getElementById('resultsTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            
            // Remove sort classes from all headers
            headers.forEach(h => {
                h.classList.remove('sort-asc', 'sort-desc');
            });
            
            // Determine sort direction
            if (currentSortColumn === columnIndex) {
                sortDirection[columnIndex] = sortDirection[columnIndex] === 'asc' ? 'desc' : 'asc';
            } else {
                sortDirection[columnIndex] = 'asc';
                currentSortColumn = columnIndex;
            }
            
            // Add sort class to current header
            headers[columnIndex].classList.add(sortDirection[columnIndex] === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Sort rows
            rows.sort((a, b) => {
                const aText = a.cells[columnIndex].textContent.trim();
                const bText = b.cells[columnIndex].textContent.trim();
                
                // Try to parse as number (handles currency, percentages, etc.)
                const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ''));
                const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ''));
                
                let comparison = 0;
                
                if (!isNaN(aNum) && !isNaN(bNum)) {
                    // Both are numbers
                    comparison = aNum - bNum;
                } else {
                    // String comparison
                    comparison = aText.localeCompare(bText);
                }
                
                return sortDirection[columnIndex] === 'asc' ? comparison : -comparison;
            });
            
            // Re-append sorted rows
            rows.forEach(row => tbody.appendChild(row));
            
            // Update visible count
            document.getElementById('visibleCount').textContent = rows.length;
        }
        
        // Initialize - sort by net_daily_premium if available
        document.addEventListener('DOMContentLoaded', function() {
            const headers = document.querySelectorAll('th');
            headers.forEach((header, index) => {
                if (header.textContent.toLowerCase().includes('net daily premium')) {
                    sortTable(index);
                }
            });
        });
    </script>
</body>
</html>
""")
    
    html_content = ''.join(html_parts)
    
    # Write HTML file
    html_file = output_path / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML output generated successfully!", file=sys.stderr)
    print(f"Output directory: {output_path.absolute()}", file=sys.stderr)
    print(f"Open: {html_file.absolute()}", file=sys.stderr)


def print_summary_rankings(df: pd.DataFrame) -> None:
    """Print summary rankings and top recommendations."""
    print("\n\n" + "=" * 120)
    print("SUMMARY RANKINGS")
    print("=" * 120)
    
    top_10 = df.nlargest(10, 'net_daily_premium')
    
    # Create final ranking
    ranking = top_10[['ticker', 'net_daily_premium', 'volume', 'delta', 'pe_ratio', 
                      'option_ticker', 'long_option_ticker']].copy()
    ranking['liquidity_score'] = ranking['volume'].apply(lambda x: 3 if x > 1000 else 2 if x > 300 else 1)
    ranking['delta_score'] = ranking['delta'].apply(lambda x: 3 if x < 0.35 else 2 if x < 0.50 else 1)
    ranking['premium_score'] = ranking['net_daily_premium'].apply(lambda x: 3 if x > 10000 else 2 if x > 7000 else 1)
    ranking['pe_score'] = ranking['pe_ratio'].apply(lambda x: 2 if x < 25 else 1 if x < 50 else 0)
    ranking['total_score'] = ranking['liquidity_score'] + ranking['delta_score'] + ranking['premium_score'] + ranking['pe_score']
    
    ranking = ranking.sort_values('total_score', ascending=False)
    
    print("\nFINAL RANKINGS (by composite score):\n")
    print(ranking[['ticker', 'net_daily_premium', 'volume', 'delta', 'total_score']].to_string(index=False))
    
    print("\n\n" + "=" * 120)
    print("🏆 TOP 3 RECOMMENDATIONS WITH TRADE DETAILS")
    print("=" * 120)
    
    top_3 = ranking.head(3)
    for i, (idx, row) in enumerate(top_3.iterrows(), 1):
        ticker_data = df[df['ticker'] == row['ticker']].iloc[0]
        
        print(f"\n{'─' * 110}")
        print(f"#{i}. {row['ticker']} - Score: {row['total_score']}/11 | Daily Premium: ${row['net_daily_premium']:,.2f} | Volume: {row['volume']:,.0f}")
        print(f"{'─' * 110}")
        print(f"\n   📍 CURRENT PRICE: ${ticker_data['current_price']:.2f}")
        print(f"\n   🔴 SELL (SHORT):  {row['option_ticker']}")
        print(f"      Strike: ${ticker_data['strike_price']:.2f} | Exp: {ticker_data['expiration_date'][:10]} | Premium: ${ticker_data['option_premium']:.2f}")
        print(f"      Contracts: {int(ticker_data['num_contracts'])} | Total Credit: ${ticker_data['short_premium_total']:,.0f}")
        print(f"\n   🟢 BUY (LONG):    {row['long_option_ticker']}")
        print(f"      Strike: ${ticker_data['long_strike_price']:.2f} | Exp: {ticker_data['long_expiration_date'][:10]} | Premium: ${ticker_data['long_option_premium']:.2f}")
        print(f"      Contracts: {int(ticker_data['num_contracts'])} | Total Debit: ${ticker_data['long_premium_total']:,.0f}")
        print(f"\n   💵 NET POSITION: ${ticker_data['net_premium']:,.0f} credit ({(ticker_data['net_premium']/100000*100):.1f}% ROI)")
        print(f"   📊 RISK: Delta {ticker_data['delta']:.2f} | Volume {ticker_data['volume']:,.0f} | P/E {ticker_data['pe_ratio']:.1f}")
    
    print("\n\n" + "=" * 120)
    print("📋 QUICK REFERENCE: ALL TOP 10 OPTION TICKERS")
    print("=" * 120)
    
    print("\n{:<8} {:<30} {:<30} {:<12}".format("TICKER", "SHORT (SELL)", "LONG (BUY)", "DAILY $"))
    print("─" * 110)
    
    for idx, row in ranking.iterrows():
        ticker_data = df[df['ticker'] == row['ticker']].iloc[0]
        print("{:<8} {:<30} {:<30} ${:>10,.2f}".format(
            row['ticker'],
            row['option_ticker'],
            row['long_option_ticker'],
            row['net_daily_premium']
        ))
    
    print("\n" + "=" * 120)


def main():
    """Main function to run the covered calls evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate covered call spread opportunities from CSV data",
        epilog="""
Examples:
  python scripts/evaluate_covered_calls.py --file results.csv
  python scripts/evaluate_covered_calls.py < results.csv
  cat results.csv | python scripts/evaluate_covered_calls.py
  python scripts/evaluate_covered_calls.py --file -
  python scripts/evaluate_covered_calls.py --file results.csv --html --output-dir results_html
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help="Path to CSV file to process. Use '-' or omit to read from stdin (default: stdin)"
    )
    
    parser.add_argument(
        '--html',
        action='store_true',
        help="Generate HTML output with sortable table"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='html_output',
        help="Directory path for HTML output (default: html_output)"
    )
    
    args = parser.parse_args()
    
    # Determine input source
    file_path = args.file
    
    try:
        # Load data
        df = load_data(file_path)
        
        if df.empty:
            print("Error: No data loaded. Check your input file or stdin.", file=sys.stderr)
            sys.exit(1)
        
        # Generate HTML output if requested
        if args.html:
            generate_html_output(df, args.output_dir)
        else:
            # Print analyses to stdout
            print_top_20_analysis(df)
            print_detailed_analysis(df)
            print_summary_rankings(df)
        
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Input data is empty.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Required column missing: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
