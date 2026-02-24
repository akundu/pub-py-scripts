#!/usr/bin/env python3
"""
Web Dashboard for Continuous Mode

Simple Flask-based dashboard to monitor regime, opportunities, and positions.
"""

import sys
import json
import time as _time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.continuous.config import CONFIG
from scripts.continuous.market_data import get_current_market_context, MarketContext
from scripts.continuous.opportunity_scanner import scan_opportunities, TradeOpportunity
from scripts.continuous.position_tracker import PositionTracker


class DashboardData:
    """Manages dashboard data."""

    def __init__(self):
        """Initialize dashboard data manager."""
        self.tracker = PositionTracker()

    def get_current_data(self) -> Dict:
        """
        Get current dashboard data.

        Returns:
            Dictionary with all dashboard data
        """
        # Get market context
        try:
            market_context = get_current_market_context(CONFIG.default_ticker, CONFIG.default_trend)
        except Exception as e:
            print(f"Error fetching market context: {e}")
            market_context = None

        # Get opportunities (with today's date for option chain auto-loading)
        opportunities = []
        if market_context:
            try:
                from datetime import date
                opportunities = scan_opportunities(
                    market_context, top_n=CONFIG.regime_top_n_configs,
                    target_date=date.today(),
                )
            except Exception as e:
                print(f"Error scanning opportunities: {e}")

        # Get positions
        open_positions = self.tracker.get_open_positions()
        summary = self.tracker.get_summary()

        # Format data
        data = {
            'timestamp': datetime.now().isoformat(),
            'market': market_context.to_dict() if market_context else {},
            'opportunities': [opp.to_dict() for opp in opportunities],  # All opportunities
            'positions': {
                'open': [pos.to_dict() for pos in open_positions],
                'summary': summary,
            },
            'alerts': self._get_recent_alerts(limit=20),
            'data_sources': {
                'grid_file': str(CONFIG.grid_file),
                'chain_dirs': [
                    'options_csv_output_full/{ticker}_options_{date}.csv',
                    'options_csv_output/{ticker}_options_{date}.csv',
                    'csv_exports/options/{ticker}/{date}.csv',
                ],
                'config_file': 'scripts/continuous/config.py',
            },
        }

        return data

    def _get_recent_alerts(self, limit: int = 20) -> List[str]:
        """Get recent alerts from log file."""
        if not CONFIG.alerts_log.exists():
            return []

        try:
            with open(CONFIG.alerts_log, 'r') as f:
                lines = f.readlines()
                return lines[-limit:] if lines else []
        except Exception as e:
            print(f"Error reading alerts: {e}")
            return []

    def save_to_file(self, data: Dict = None):
        """Save dashboard data to JSON file.

        Args:
            data: Pre-built dashboard data dict.  When *None* (legacy
                  behaviour) the method will call get_current_data() which
                  triggers a full re-scan.  Prefer passing data directly
                  from the continuous_mode cycle so the dashboard matches
                  the console output.
        """
        if data is None:
            data = self.get_current_data()

        try:
            CONFIG.dashboard_data.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG.dashboard_data, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving dashboard data: {e}")


def generate_html_dashboard(data: Dict) -> str:
    """
    Generate HTML dashboard.

    Args:
        data: Dashboard data dictionary

    Returns:
        HTML string
    """
    market = data.get('market', {})
    opportunities = data.get('opportunities', [])
    positions = data.get('positions', {})
    alerts = data.get('alerts', [])
    data_sources = data.get('data_sources', {})

    # Market context section
    vix_level = market.get('vix_level', 0)
    vix_regime = market.get('vix_regime', 'unknown').replace('_', ' ').upper()
    price = market.get('current_price', 0)
    price_change = market.get('price_change_pct', 0)
    volume_ratio = market.get('volume_ratio', 1.0)
    is_market_hours = market.get('is_market_hours', False)

    # Position summary
    pos_summary = positions.get('summary', {})
    open_count = pos_summary.get('open_positions', 0)
    total_risk = pos_summary.get('total_risk', 0)
    unrealized_pnl = pos_summary.get('unrealized_pnl', 0)
    realized_pnl = pos_summary.get('realized_pnl', 0)

    # Opportunities table
    opp_rows = ''
    total_opps = len(opportunities)
    for i, opp in enumerate(opportunities, 1):
        in_window = '✓' if opp.get('is_in_entry_window') else ''
        quality = '✓' if opp.get('meets_quality_threshold') else ''

        # Build trade instruction display
        trade_instruction = opp.get('trade_instruction', '')
        resolved = opp.get('resolved_trade')
        trade_legs_html = ''

        if resolved and resolved.get('legs'):
            legs = resolved['legs']
            data_src = resolved.get('data_source', 'estimated')
            src_badge = (
                '<span style="background:#22c55e;color:white;padding:1px 6px;border-radius:3px;font-size:10px;">LIVE</span>'
                if data_src == 'chain' else
                '<span style="background:#6b7280;color:white;padding:1px 6px;border-radius:3px;font-size:10px;">EST</span>'
            )

            leg_lines = []
            for leg in legs:
                action = 'Sell' if leg['side'] == 'sell' else 'Buy'
                color = '#ef4444' if leg['side'] == 'sell' else '#22c55e'
                price_str = f"${leg['price']:.2f}" if leg.get('price') else ''
                leg_lines.append(
                    f'<span style="color:{color};font-weight:600;">{action}</span> '
                    f'{leg["strike"]:.0f} {leg["option_type"].title()} '
                    f'<span style="color:#666;">{price_str}</span>'
                )

            credit_total = resolved.get('total_credit', 0)
            max_risk_r = resolved.get('max_risk', 0)
            exp = resolved.get('expiration', '')

            trade_legs_html = (
                f'<div style="font-size:12px;line-height:1.6;margin-top:4px;">'
                f'{src_badge} '
                + ' <span style="color:#999;">/</span> '.join(leg_lines)
                + f'<br><strong>Credit: ${credit_total:,.0f}</strong>'
                  f' | Risk: ${max_risk_r:,.0f}'
                  f' | Exp: {exp}'
                f'</div>'
            )
        elif trade_instruction:
            trade_legs_html = f'<div style="font-size:12px;margin-top:4px;">{trade_instruction}</div>'

        # Rows beyond #5 get a class for hide/show toggling
        extra_class = ' class="extra-opp"' if i > 5 else ''

        opp_rows += f"""
        <tr{extra_class}>
            <td>{i}</td>
            <td>{opp.get('dte')}D</td>
            <td>{opp.get('band')}</td>
            <td>{opp.get('spread_type', '').replace('_', ' ').title()}</td>
            <td>{opp.get('entry_time_pst')}</td>
            <td>{opp.get('expected_win_pct', 0):.1f}%</td>
            <td>{opp.get('expected_roi_pct', 0):.1f}%</td>
            <td>{opp.get('sharpe', 0):.2f}</td>
            <td>{in_window}</td>
            <td>{quality}</td>
        </tr>
        <tr{extra_class}>
            <td colspan="10" style="padding:2px 10px 10px 30px; border-bottom: 2px solid #e5e7eb;">
                {trade_legs_html or '<span style="color:#999;font-size:12px;">No strike data available</span>'}
            </td>
        </tr>
        """

    # Add expand/collapse button if there are more than 5 opportunities
    extra_count = max(0, total_opps - 5)
    expand_button = ''
    if extra_count > 0:
        expand_button = f"""
        <tr id="expand-row">
            <td colspan="10" style="text-align:center; padding:12px;">
                <button id="expand-btn" onclick="toggleExtra()"
                    style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           color:white; border:none; padding:8px 24px; border-radius:6px;
                           cursor:pointer; font-size:14px; font-weight:600;">
                    Show {extra_count} More Opportunities ▼
                </button>
            </td>
        </tr>
        """

    # Positions table
    pos_rows = ''
    for pos in positions.get('open', []):
        pos_id = pos.get('position_id', '')
        dte = pos.get('dte', 0)
        band = pos.get('band', '')
        spread = pos.get('spread_type', '').replace('_', ' ').title()
        credit = pos.get('credit_received', 0)
        risk = pos.get('max_risk', 0)
        pnl = pos.get('current_pnl', 0)
        pnl_pct = (pnl / credit * 100) if credit > 0 else 0

        pos_rows += f"""
        <tr>
            <td>{pos_id}</td>
            <td>{dte}D</td>
            <td>{band}</td>
            <td>{spread}</td>
            <td>${credit:.2f}</td>
            <td>${risk:.2f}</td>
            <td class="{'pnl-positive' if pnl >= 0 else 'pnl-negative'}">${pnl:.2f} ({pnl_pct:+.1f}%)</td>
        </tr>
        """

    if not pos_rows:
        pos_rows = '<tr><td colspan="7" style="text-align:center;">No open positions</td></tr>'

    # Alerts section
    alert_lines = ''.join([f'<div class="alert-line">{line.strip()}</div>' for line in alerts[-10:]])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Options Trading Dashboard</title>
        <meta http-equiv="refresh" content="{CONFIG.dashboard_refresh_seconds}">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }}
            .card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .card h2 {{
                margin-top: 0;
                color: #333;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }}
            .stat {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #667eea;
            }}
            .stat-label {{
                font-size: 12px;
                color: #666;
                text-transform: uppercase;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-top: 5px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th {{
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .pnl-positive {{
                color: #22c55e;
                font-weight: bold;
            }}
            .pnl-negative {{
                color: #ef4444;
                font-weight: bold;
            }}
            .regime {{
                display: inline-block;
                padding: 5px 15px;
                border-radius: 20px;
                font-weight: bold;
                color: white;
            }}
            .regime-low {{ background-color: #22c55e; }}
            .regime-medium {{ background-color: #f59e0b; }}
            .regime-high {{ background-color: #ef4444; }}
            .regime-extreme {{ background-color: #991b1b; }}
            .alert-line {{
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 4px 0;
                border-bottom: 1px solid #eee;
            }}
            .status {{
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }}
            .status-open {{ background-color: #22c55e; }}
            .status-closed {{ background-color: #6b7280; }}
            .timestamp {{
                color: #666;
                font-size: 14px;
            }}
            .extra-opp {{
                display: none;
            }}
            .extra-opp.visible {{
                display: table-row;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Options Trading Dashboard - Continuous Mode (Alert-Only)</h1>
                <p class="timestamp">Last Updated: {data.get('timestamp', 'N/A')}</p>
            </div>

            <div class="card">
                <h2>Market Context</h2>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Ticker</div>
                        <div class="stat-value">{market.get('ticker', 'N/A')}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Price</div>
                        <div class="stat-value">${price:,.2f}</div>
                        <div class="{'pnl-positive' if price_change >= 0 else 'pnl-negative'}">{price_change:+.2f}%</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">VIX</div>
                        <div class="stat-value">{vix_level:.2f}</div>
                        <span class="regime regime-{vix_regime.split()[0].lower()}">{vix_regime}</span>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Volume</div>
                        <div class="stat-value">{volume_ratio:.2f}x</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Market Status</div>
                        <div class="stat-value">
                            <span class="status {'status-open' if is_market_hours else 'status-closed'}"></span>
                            {'OPEN' if is_market_hours else 'CLOSED'}
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Top Opportunities (Regime-Filtered)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>DTE</th>
                            <th>Band</th>
                            <th>Spread</th>
                            <th>Entry Time</th>
                            <th>Win%</th>
                            <th>ROI%</th>
                            <th>Sharpe</th>
                            <th>Window</th>
                            <th>Quality</th>
                        </tr>
                    </thead>
                    <tbody>
                        {opp_rows or '<tr><td colspan="10" style="text-align:center;">No opportunities found</td></tr>'}
                        {expand_button}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Portfolio Summary</h2>
                <div class="stats">
                    <div class="stat">
                        <div class="stat-label">Open Positions</div>
                        <div class="stat-value">{open_count}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Total Risk</div>
                        <div class="stat-value">${total_risk:,.2f}</div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Unrealized P&L</div>
                        <div class="stat-value {'pnl-positive' if unrealized_pnl >= 0 else 'pnl-negative'}">
                            ${unrealized_pnl:,.2f}
                        </div>
                    </div>
                    <div class="stat">
                        <div class="stat-label">Realized P&L</div>
                        <div class="stat-value {'pnl-positive' if realized_pnl >= 0 else 'pnl-negative'}">
                            ${realized_pnl:,.2f}
                        </div>
                    </div>
                </div>

                <h3>Open Positions</h3>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>DTE</th>
                            <th>Band</th>
                            <th>Spread</th>
                            <th>Credit</th>
                            <th>Risk</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pos_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Recent Alerts (Last 10)</h2>
                <div style="max-height: 300px; overflow-y: auto;">
                    {alert_lines or '<div style="text-align:center; color:#666;">No alerts</div>'}
                </div>
            </div>

            <div class="card" style="background:#f8f9fa; border:1px solid #e5e7eb;">
                <h2 style="font-size:16px; color:#666;">Data Sources</h2>
                <div style="font-family:'Courier New',monospace; font-size:13px; color:#555; line-height:1.8;">
                    <div><strong>Grid file:</strong> {data_sources.get('grid_file', 'N/A')}</div>
                    <div><strong>Option chains (priority order):</strong></div>
                    {''.join(f'<div style="padding-left:20px;">{i+1}. {d}</div>' for i, d in enumerate(data_sources.get('chain_dirs', [])))}
                    <div><strong>Config:</strong> {data_sources.get('config_file', 'N/A')}</div>
                </div>
            </div>
        </div>
        <script>
        function toggleExtra() {{
            var rows = document.querySelectorAll('.extra-opp');
            var btn = document.getElementById('expand-btn');
            var showing = rows.length > 0 && rows[0].classList.contains('visible');
            rows.forEach(function(r) {{
                if (showing) {{
                    r.classList.remove('visible');
                }} else {{
                    r.classList.add('visible');
                }}
            }});
            if (showing) {{
                btn.innerHTML = 'Show {extra_count} More Opportunities &#9660;';
            }} else {{
                btn.innerHTML = 'Hide Extra Opportunities &#9650;';
            }}
        }}
        </script>
    </body>
    </html>
    """

    return html


def generate_simulation_banner(sim_info: Dict) -> str:
    """Generate HTML banner for simulation mode."""
    pct = sim_info.get('pct_complete', 0)
    step = sim_info.get('step', 0)
    total = sim_info.get('total_steps', 0)
    speed = sim_info.get('speed', 30)
    sim_date = sim_info.get('date', '')
    sim_time = sim_info.get('sim_time', '')

    return f"""
    <div style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
                color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px;
                display: flex; justify-content: space-between; align-items: center;">
        <div>
            <strong style="font-size: 18px;">SIMULATION MODE</strong>
            <span style="margin-left: 15px;">Date: {sim_date}</span>
            <span style="margin-left: 15px;">Speed: {speed}x</span>
        </div>
        <div style="text-align: right;">
            <strong style="font-size: 22px;">{sim_time}</strong>
            <div style="margin-top: 4px;">Step {step}/{total} ({pct:.0f}%)</div>
        </div>
    </div>
    <div style="background: #e5e7eb; border-radius: 4px; height: 8px; margin-bottom: 20px;">
        <div style="background: linear-gradient(90deg, #f59e0b, #ef4444); height: 100%;
                    border-radius: 4px; width: {pct:.1f}%; transition: width 0.5s;"></div>
    </div>
    """


def start_dashboard_server():
    """Start Flask dashboard server."""
    try:
        from flask import Flask, render_template_string
    except ImportError:
        print("Error: Flask not installed. Install with: pip install flask")
        return

    app = Flask(__name__)
    dashboard_data = DashboardData()

    @app.route('/')
    def index():
        """Dashboard home page.

        Reads from dashboard_data.json written by continuous_mode.py so
        that the dashboard shows the SAME data logged to console without
        triggering a duplicate scan.  Falls back to a live scan when the
        JSON file is missing (standalone dashboard mode).
        """
        data = None

        # Try reading the JSON file that continuous_mode writes every cycle
        if CONFIG.dashboard_data.exists():
            try:
                import os
                age = _time.time() - os.path.getmtime(str(CONFIG.dashboard_data))
                # Only use cached file if it's less than 5 minutes old
                if age < 300:
                    with open(CONFIG.dashboard_data, 'r') as f:
                        data = json.load(f)
            except (json.JSONDecodeError, Exception):
                data = None

        # Fallback: live scan (standalone mode or stale/missing file)
        if data is None:
            data = dashboard_data.get_current_data()

        html = generate_html_dashboard(data)
        return render_template_string(html)

    @app.route('/sim')
    def simulation():
        """Simulation dashboard - reads from JSON file, refreshes every 5 seconds."""
        try:
            if CONFIG.dashboard_data.exists():
                with open(CONFIG.dashboard_data, 'r') as f:
                    data = json.load(f)
            else:
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'market': {},
                    'opportunities': [],
                    'positions': {'open': [], 'summary': {}},
                    'alerts': [],
                }
        except (json.JSONDecodeError, Exception) as e:
            data = {
                'timestamp': datetime.now().isoformat(),
                'market': {},
                'opportunities': [],
                'positions': {'open': [], 'summary': {}},
                'alerts': [],
                'error': str(e),
            }

        # Check if simulation is active
        sim_info = data.get('simulation', {})
        is_sim = sim_info.get('active', False)

        # Generate the standard dashboard HTML
        html = generate_html_dashboard(data)

        if is_sim:
            # Inject simulation banner and faster refresh
            sim_banner = generate_simulation_banner(sim_info)
            html = html.replace(
                f'<meta http-equiv="refresh" content="{CONFIG.dashboard_refresh_seconds}">',
                '<meta http-equiv="refresh" content="5">'
            )
            html = html.replace(
                '<div class="header">',
                sim_banner + '\n            <div class="header">'
            )
            # Update header title
            html = html.replace(
                'Options Trading Dashboard - Continuous Mode (Alert-Only)',
                f'Options Trading Dashboard - SIMULATION {sim_info.get("date", "")}'
            )
        else:
            # No simulation running - show waiting message
            html = html.replace(
                f'<meta http-equiv="refresh" content="{CONFIG.dashboard_refresh_seconds}">',
                '<meta http-equiv="refresh" content="3">'
            )
            html = html.replace(
                '<div class="header">',
                """<div style="background: #6b7280; color: white; padding: 15px 20px;
                            border-radius: 8px; margin-bottom: 20px; text-align: center;">
                    <strong>Waiting for simulation...</strong>
                    <div style="margin-top: 5px;">Start with:
                    <code>python scripts/continuous/simulate_day.py --date 2026-02-20 --speed 30 --auto-start</code></div>
                </div>
                <div class="header">"""
            )

        return render_template_string(html)

    print(f"Starting dashboard server on http://localhost:{CONFIG.dashboard_port}")
    print(f"  Live dashboard:       http://localhost:{CONFIG.dashboard_port}/")
    print(f"  Simulation dashboard: http://localhost:{CONFIG.dashboard_port}/sim")
    print(f"Dashboard will auto-refresh every {CONFIG.dashboard_refresh_seconds} seconds (live) / 5 seconds (sim)")
    app.run(host='0.0.0.0', port=CONFIG.dashboard_port, debug=False)


if __name__ == '__main__':
    """Start dashboard."""
    start_dashboard_server()
