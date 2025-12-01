"""
Comprehensive analysis HTML generation.
"""

import pandas as pd
import numpy as np
import re
import html as html_escape
from .data_processor import extract_numeric_value


def generate_summary_statistics_html(df: pd.DataFrame) -> str:
    """Generate HTML for summary statistics section.
    
    Args:
        df: DataFrame with the results
        
    Returns:
        String containing HTML for summary statistics
    """
    # Check if required columns exist
    if 'spread_impact_pct' not in df.columns:
        return ''
    
    html_parts = []
    html_parts.append('            <div class="analysis-item">\n')
    html_parts.append('                <h3>📊 SUMMARY STATISTICS</h3>\n')
    
    # Safely convert spread_impact_pct to numeric, handling malformed strings
    def safe_extract_number(val):
        if pd.isna(val):
            return np.nan
        try:
            val_str = str(val)
            try:
                return float(val_str)
            except (ValueError, TypeError):
                match = re.search(r'-?\d+\.?\d*', val_str)
                if match:
                    return float(match.group())
                return np.nan
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    # Ensure spread_impact_pct is numeric before calculations
    spread_impact_numeric = df['spread_impact_pct'].apply(safe_extract_number)
    spread_impact_numeric = pd.to_numeric(spread_impact_numeric, errors='coerce')
    
    # Spread Impact Statistics
    spread_impact_mean = spread_impact_numeric.mean()
    spread_impact_median = spread_impact_numeric.median()
    low_impact_count = (spread_impact_numeric < 5).sum()
    high_impact_count = (spread_impact_numeric > 10).sum()
    
    html_parts.append('                <div class="analysis-section">\n')
    html_parts.append('                    <h4>Spread Impact Statistics</h4>\n')
    html_parts.append(f'                    <p><span class="label">Average spread impact:</span> {spread_impact_mean:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Median spread impact:</span> {spread_impact_median:.2f}%</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &lt;5% impact:</span> {low_impact_count}/{len(df)}</p>\n')
    html_parts.append(f'                    <p><span class="label">Trades with &gt;10% impact:</span> <span class="risk-badge risk-high">{high_impact_count}/{len(df)} ⚠️</span></p>\n')
    html_parts.append('                </div>\n')
    
    # Liquidity Distribution
    if 'liquidity_score' in df.columns:
        # Safely convert liquidity_score to numeric
        liquidity_score_numeric = df['liquidity_score'].apply(safe_extract_number)
        liquidity_score_numeric = pd.to_numeric(liquidity_score_numeric, errors='coerce')
        
        high_liquidity = (liquidity_score_numeric >= 7).sum()
        medium_liquidity = ((liquidity_score_numeric >= 4) & (liquidity_score_numeric < 7)).sum()
        low_liquidity = (liquidity_score_numeric < 4).sum()
        
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>Liquidity Distribution</h4>\n')
        html_parts.append(f'                    <p><span class="label">High liquidity (7-10):</span> {high_liquidity} trades</p>\n')
        html_parts.append(f'                    <p><span class="label">Medium liquidity (4-6):</span> {medium_liquidity} trades</p>\n')
        html_parts.append(f'                    <p><span class="label">Low liquidity (&lt;4):</span> <span class="risk-badge risk-high">{low_liquidity} trades ⚠️</span></p>\n')
        html_parts.append('                </div>\n')
    
    html_parts.append('            </div>\n')
    
    return ''.join(html_parts)


def generate_detailed_analysis_html(df: pd.DataFrame) -> str:
    """Generate HTML for unified comprehensive analysis section.
    
    Args:
        df: DataFrame with the results
        
    Returns:
        String containing HTML for comprehensive analysis
    """
    # Helper function to safely get numeric value from row
    def safe_get_numeric(row, key, default=0):
        """Safely get and convert a numeric value from a row."""
        val = row.get(key, default)
        if pd.isna(val):
            return default
        try:
            # Handle malformed strings
            if isinstance(val, str):
                match = re.search(r'-?\d+\.?\d*', val)
                if match:
                    return float(match.group())
                return default
            return float(val)
        except (ValueError, TypeError):
            return default
    
    # Convert net_daily_premi to numeric if it exists, handling object dtype
    if 'net_daily_premi' in df.columns:
        df = df.copy()
        df['_net_daily_premi_numeric'] = df['net_daily_premi'].apply(lambda x: safe_get_numeric({'val': x}, 'val', np.nan))
        df_valid = df[df['_net_daily_premi_numeric'].notna()]
        if len(df_valid) > 0:
            top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
        else:
            top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
    elif 'net_daily_premium' in df.columns:
        df = df.copy()
        df['_net_daily_premi_numeric'] = df['net_daily_premium'].apply(lambda x: safe_get_numeric({'val': x}, 'val', np.nan))
        df_valid = df[df['_net_daily_premi_numeric'].notna()]
        if len(df_valid) > 0:
            top_10 = df_valid.nlargest(10, '_net_daily_premi_numeric').drop(columns=['_net_daily_premi_numeric'])
        else:
            top_10 = df.head(10).drop(columns=['_net_daily_premi_numeric'], errors='ignore')
    else:
        # Fallback: use first 10 rows if column doesn't exist
        top_10 = df.head(10)
    
    html_parts = []
    html_parts.append('        <div class="detailed-analysis">\n')
    html_parts.append('            <h2>📊 COMPREHENSIVE ANALYSIS: TOP 10 PICKS WITH OPTION TICKERS</h2>\n')
    
    # Add summary statistics at the top
    html_parts.append(generate_summary_statistics_html(df))
    
    for idx, (row_idx, row) in enumerate(top_10.iterrows(), 1):
        ticker = row['ticker']
        
        # Get option type for this row
        option_type = str(row.get('option_type', 'call')).lower() if pd.notna(row.get('option_type')) else 'call'
        is_put = (option_type == 'put')
        
        # Calculate values
        curr_price = safe_get_numeric(row, 'curr_price') or safe_get_numeric(row, 'current_price', 0)
        strike_price = safe_get_numeric(row, 'strike_price', 0)
        
        # Calculate moneyness
        if is_put:
            moneyness = ((curr_price - strike_price) / curr_price * 100) if curr_price != 0 and strike_price != 0 else 0
            moneyness_label = "OTM" if moneyness > 0 else ("ITM" if moneyness < 0 else "ATM")
        else:
            moneyness = ((strike_price - curr_price) / curr_price * 100) if curr_price != 0 and strike_price != 0 else 0
            moneyness_label = "OTM" if moneyness > 0 else ("ITM" if moneyness < 0 else "ATM")
        
        l_strike = safe_get_numeric(row, 'l_strike', 0)
        spread_width = l_strike - strike_price if l_strike != 0 and strike_price != 0 else 0
        
        l_delta = safe_get_numeric(row, 'l_delta', 0)
        delta = safe_get_numeric(row, 'delta', 0)
        delta_diff = l_delta - delta
        
        net_premium = safe_get_numeric(row, 'net_premium', 0)
        roi = (net_premium / 100000 * 100) if net_premium != 0 else 0
        
        # Calculate score
        score = 0
        net_daily_val = safe_get_numeric(row, 'net_daily_premi', 0)
        if net_daily_val > 10000: score += 3
        elif net_daily_val > 7000: score += 2
        elif net_daily_val > 0: score += 1
        
        volume = safe_get_numeric(row, 'volume', 0)
        if volume > 1000: score += 3
        elif volume > 300: score += 2
        elif volume > 100: score += 1
        
        delta_score = safe_get_numeric(row, 'delta', 0)
        delta_abs = abs(delta_score) if delta_score else 0
        if delta_abs > 0:
            if delta_abs < 0.35: score += 3
            elif delta_abs < 0.50: score += 2
            else: score += 1
        
        pe_ratio_score = safe_get_numeric(row, 'pe_ratio', 0)
        if pe_ratio_score > 0:
            if pe_ratio_score < 25: score += 2
            elif pe_ratio_score < 50: score += 1
        
        # Determine recommendation
        if score >= 9:
            recommendation = "STRONG BUY - Excellent risk/reward"
        elif score >= 7:
            recommendation = "BUY - Good opportunity"
        elif score >= 5:
            recommendation = "HOLD - Acceptable but monitor"
        else:
            recommendation = "PASS - Better opportunities available"
        
        # Assignment risk
        delta_abs = abs(delta_score) if delta_score else 0
        if delta_abs > 0:
            if delta_abs < 0.35:
                if is_put:
                    assignment_risk = "LOW - Strike is well OTM (below current price)"
                else:
                    assignment_risk = "LOW - Strike is well OTM (above current price)"
                risk_class = "risk-low"
            elif delta_abs < 0.50:
                assignment_risk = "MODERATE - Near ATM, watch closely"
                risk_class = "risk-moderate"
            else:
                if is_put:
                    assignment_risk = "HIGH - ITM or very close (strike above current), likely assignment"
                else:
                    assignment_risk = "HIGH - ITM or very close (strike below current), likely assignment"
                risk_class = "risk-high"
        else:
            assignment_risk = "UNKNOWN"
            risk_class = "risk-moderate"
        
        # Liquidity
        if volume > 0:
            if volume > 1000:
                liquidity = "EXCELLENT - Very liquid"
            elif volume > 300:
                liquidity = "GOOD - Adequate liquidity"
            elif volume > 100:
                liquidity = "FAIR - May have wider spreads"
            else:
                liquidity = "POOR - Low liquidity, watch bid-ask"
        else:
            liquidity = "UNKNOWN"
        
        # Valuation
        if pe_ratio_score > 0:
            if pe_ratio_score < 15:
                valuation = "ATTRACTIVE - Trading at discount"
            elif pe_ratio_score < 25:
                valuation = "FAIR - Reasonably valued"
            elif pe_ratio_score < 50:
                valuation = "ELEVATED - Premium valuation"
            else:
                valuation = "EXPENSIVE - Very high P/E"
        else:
            valuation = "UNKNOWN"
        
        # Format dates
        exp_date = str(row['expiration_date'])[:10] if pd.notna(row.get('expiration_date')) else 'N/A'
        l_exp_date = str(row['l_expiration_date'])[:10] if pd.notna(row.get('l_expiration_date')) else 'N/A'
        
        html_parts.append(f'            <div class="analysis-item">\n')
        # Make ticker a link
        if ticker and ticker != 'N/A':
            ticker_link = f'/stock_info/{html_escape.escape(ticker)}'
            html_parts.append(f'                <h3>#{idx}: <a href="{ticker_link}" target="_blank" style="color: #667eea; text-decoration: none;">{html_escape.escape(ticker)}</a></h3>\n')
        else:
            html_parts.append(f'                <h3>#{idx}: {html_escape.escape(ticker)}</h3>\n')
        
        # Position Structure
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>📊 POSITION STRUCTURE</h4>\n')
        curr_price_val = safe_get_numeric(row, 'curr_price', 0) or safe_get_numeric(row, 'current_price', 0)
        option_type_display = option_type.upper() if option_type else "CALL"
        html_parts.append(f'                    <p><span class="label">Option Type:</span> {option_type_display}</p>\n')
        html_parts.append(f'                    <p><span class="label">Current Price:</span> ${curr_price_val:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Short Strike:</span> ${strike_price:.2f} ({abs(moneyness):.2f}% {moneyness_label})</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Strike:</span> ${l_strike:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Spread Width:</span> ${spread_width:.2f}</p>\n')
        html_parts.append('                </div>\n')
        
        # Option Tickers
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>🎯 OPTION TICKERS</h4>\n')
        html_parts.append('                    <div class="option-tickers">\n')
        option_ticker_short = html_escape.escape(str(row.get("option_ticker", "N/A"))) if pd.notna(row.get("option_ticker")) else "N/A"
        option_ticker_long = html_escape.escape(str(row.get("l_option_ticker", "N/A"))) if pd.notna(row.get("l_option_ticker")) else "N/A"
        bid_ask_short = html_escape.escape(str(row.get("bid:ask", "N/A:N/A")))
        bid_ask_long = html_escape.escape(str(row.get("l_bid:ask", "N/A:N/A")))
        html_parts.append(f'                        <p><span class="short">┌─ SHORT (SELL):</span> {option_ticker_short}</p>\n')
        
        days_to_expiry = safe_get_numeric(row, 'days_to_expiry', 0)
        opt_prem = safe_get_numeric(row, 'opt_prem.', 0)
        s_prem_tot = safe_get_numeric(row, 's_prem_tot', 0)
        num_contracts_display = safe_get_numeric(row, 'num_contracts', 0)
        l_days_to_expiry = safe_get_numeric(row, 'l_days_to_expiry', 0)
        l_prem = safe_get_numeric(row, 'l_prem', 0) or safe_get_numeric(row, 'l_opt_prem', 0)
        buy_cost_val = safe_get_numeric(row, "buy_cost", 0) or safe_get_numeric(row, "l_prem_tot", 0)
        
        html_parts.append(f'                        <p>│  Strike: ${strike_price:.2f} | Expiry: {html_escape.escape(exp_date)} ({int(days_to_expiry)} DTE)</p>\n')
        html_parts.append(f'                        <p>│  Premium: ${opt_prem:.2f} per contract | Bid:Ask: {bid_ask_short}</p>\n')
        html_parts.append(f'                        <p>│  Total Credit: ${s_prem_tot:,.0f} ({int(num_contracts_display)} contracts)</p>\n')
        html_parts.append('                        <p>│</p>\n')
        html_parts.append(f'                        <p><span class="long">└─ LONG (BUY):</span> {option_ticker_long}</p>\n')
        html_parts.append(f'                        <p>   Strike: ${l_strike:.2f} | Expiry: {html_escape.escape(l_exp_date)} ({int(l_days_to_expiry)} DTE)</p>\n')
        html_parts.append(f'                        <p>   Premium: ${l_prem:.2f} per contract | Bid:Ask: {bid_ask_long}</p>\n')
        html_parts.append(f'                        <p>   Total Debit: ${buy_cost_val:,.0f} ({int(num_contracts_display)} contracts)</p>\n')
        html_parts.append('                    </div>\n')
        html_parts.append('                </div>\n')
        
        # Premium Breakdown
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>💰 PREMIUM BREAKDOWN</h4>\n')
        html_parts.append(f'                    <p><span class="label">Short Premium:</span> ${s_prem_tot:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Long Premium:</span> ${buy_cost_val:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Net Credit:</span> ${net_premium:,.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Daily Income:</span> ${net_daily_val:,.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">ROI on $100k:</span> {roi:.2f}%</p>\n')
        
        # Bid/Ask & Spread Analysis
        if 'spread_impact_pct' in row and pd.notna(row.get('spread_impact_pct')):
            spread_slippage = safe_get_numeric(row, 'spread_slippage', 0)
            net_after_spread = safe_get_numeric(row, 'net_premium_after_spread', net_premium)
            net_daily_after = safe_get_numeric(row, 'net_daily_premium_after_spread', net_daily_val)
            spread_impact = safe_get_numeric(row, 'spread_impact_pct', 0)
            
            if 'liquidity_score' in row and pd.notna(row.get('liquidity_score')):
                liquidity_score = safe_get_numeric(row, 'liquidity_score', 0)
                assignment_risk_score = safe_get_numeric(row, 'assignment_risk', 0)
                trade_quality = safe_get_numeric(row, 'trade_quality', 0)
                html_parts.append('                </div>\n')
                html_parts.append('                <div class="analysis-section">\n')
                html_parts.append('                    <h4>💱 SPREAD & LIQUIDITY ANALYSIS</h4>\n')
                html_parts.append(f'                    <p><span class="label">Spread Slippage:</span> ${spread_slippage:,.0f}</p>\n')
                html_parts.append(f'                    <p><span class="label">Net Premium After Spread:</span> ${net_after_spread:,.0f}</p>\n')
                html_parts.append(f'                    <p><span class="label">Daily Income After Spread:</span> ${net_daily_after:,.2f}</p>\n')
                html_parts.append(f'                    <p><span class="label">Spread Impact:</span> {spread_impact:.2f}%</p>\n')
                html_parts.append(f'                    <p><span class="label">Liquidity Score:</span> {liquidity_score:.0f}/10</p>\n')
                html_parts.append(f'                    <p><span class="label">Assignment Risk:</span> {assignment_risk_score:.0f}/6</p>\n')
                html_parts.append(f'                    <p><span class="label">Trade Quality Score:</span> {trade_quality:.1f}</p>\n')
        
        html_parts.append('                </div>\n')
        
        # Greeks & Risk
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>📈 GREEKS & RISK</h4>\n')
        html_parts.append(f'                    <p><span class="label">Short Delta:</span> {delta:.2f} | Long Delta: {l_delta:.2f} | Net: {delta_diff:.3f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Short Theta:</span> {safe_get_numeric(row, "theta", 0):.2f} | Long Theta: {safe_get_numeric(row, "l_theta", 0):.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Assignment Risk:</span> {assignment_risk}</p>\n')
        html_parts.append('                </div>\n')
        
        # Liquidity & Fundamentals
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>🔄 LIQUIDITY & FUNDAMENTALS</h4>\n')
        html_parts.append(f'                    <p><span class="label">Volume:</span> {volume:,.0f} contracts</p>\n')
        html_parts.append(f'                    <p><span class="label">Num Contracts:</span> {num_contracts_display:.0f}</p>\n')
        html_parts.append(f'                    <p><span class="label">P/E Ratio:</span> {pe_ratio_score:.2f}</p>\n')
        html_parts.append(f'                    <p><span class="label">Market Cap:</span> ${safe_get_numeric(row, "market_cap_b", 0):.2f}B</p>\n')
        html_parts.append('                </div>\n')
        
        # Risk Assessment
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>⚠️  RISK ASSESSMENT</h4>\n')
        html_parts.append(f'                    <p><span class="label">Assignment Risk:</span> <span class="risk-badge {risk_class}">{assignment_risk}</span></p>\n')
        html_parts.append(f'                    <p><span class="label">Liquidity:</span> {liquidity}</p>\n')
        html_parts.append(f'                    <p><span class="label">Valuation:</span> {valuation}</p>\n')
        html_parts.append('                </div>\n')
        
        # Overall Score
        html_parts.append('                <div class="analysis-section">\n')
        html_parts.append('                    <h4>⭐ OVERALL SCORE & RECOMMENDATION</h4>\n')
        html_parts.append(f'                    <p><span class="label">Score:</span> {score}/11</p>\n')
        html_parts.append(f'                    <p><span class="label">Recommendation:</span> <strong>{recommendation}</strong></p>\n')
        html_parts.append('                </div>\n')
        
        html_parts.append('            </div>\n')
    
    html_parts.append('        </div>\n')
    
    return ''.join(html_parts)

