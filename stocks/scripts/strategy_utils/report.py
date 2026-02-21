"""
Report generation for strategy recommendations.

Generates console output with ANSI colors and JSON export.
"""

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    # Background
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    prefix = Colors.BOLD if bold else ''
    return f"{prefix}{color}{text}{Colors.RESET}"


def confidence_color(level: str) -> str:
    """Get color for confidence level."""
    colors = {
        'HIGH': Colors.GREEN,
        'MEDIUM-HIGH': Colors.GREEN,
        'MEDIUM': Colors.YELLOW,
        'LOW': Colors.RED,
        'NONE': Colors.RED,
    }
    return colors.get(level, Colors.WHITE)


def regime_color(regime: str) -> str:
    """Get color for VIX regime."""
    colors = {
        'LOW': Colors.GREEN,
        'STABLE': Colors.GREEN,
        'MEDIUM': Colors.YELLOW,
        'HIGH': Colors.RED,
        'EXTREME': Colors.MAGENTA,
    }
    return colors.get(regime, Colors.WHITE)


def format_header(title: str, width: int = 65) -> str:
    """Format a section header."""
    border = '=' * width
    return f"\n{colorize(border, Colors.CYAN, bold=True)}\n {colorize(title, Colors.WHITE, bold=True)}\n{colorize(border, Colors.CYAN, bold=True)}\n"


def format_subheader(title: str) -> str:
    """Format a subsection header."""
    return f"\n{colorize(title.upper(), Colors.CYAN, bold=True)}\n"


def format_market_regime(vix_regime: Dict[str, Any]) -> str:
    """Format the market regime section."""
    lines = [format_subheader("Market Regime")]

    regime = vix_regime.get('regime', 'UNKNOWN')
    color = regime_color(regime)

    if vix_regime.get('vix_current'):
        lines.append(f"  VIX Current: {colorize(f'{vix_regime['vix_current']:.1f}', color, bold=True)}")
    if vix_regime.get('vix_10d_avg'):
        lines.append(f"  VIX 10-day Avg: {vix_regime['vix_10d_avg']:.1f}")
    if vix_regime.get('vix1d_current'):
        lines.append(f"  VIX1D Current: {vix_regime['vix1d_current']:.1f}")

    lines.append(f"  Regime: {colorize(regime, color, bold=True)} ({vix_regime.get('regime_description', '')})")

    if vix_regime.get('mock_data'):
        lines.append(f"  {colorize('(Using estimated values - VIX data unavailable)', Colors.DIM)}")

    return '\n'.join(lines)


def format_convergence_table(
    convergence: Dict[str, Dict],
    windows: List[str]
) -> str:
    """Format the parameter convergence table."""
    lines = [format_subheader("Parameter Convergence Analysis")]

    # Table header
    col_widths = {
        'param': 22,
        'window': 8,
        'confidence': 12,
    }

    # Build header row
    header = f"{'Parameter':<{col_widths['param']}}"
    for w in windows:
        header += f" {w:^{col_widths['window']}}"
    header += f" {'Confidence':^{col_widths['confidence']}}"

    border = '-' * len(header)

    lines.append(colorize(header, Colors.WHITE, bold=True))
    lines.append(border)

    # Data rows
    param_display_names = {
        'percent_beyond_put': 'pb (put threshold)',
        'percent_beyond_call': 'pc (call threshold)',
        'max_spread_width_put': 'msw_put',
        'max_spread_width_call': 'msw_call',
        'min_trading_hour': 'min_hour',
        'max_trading_hour': 'max_hour',
    }

    for param, data in convergence.items():
        display_name = param_display_names.get(param, param)
        row = f"{display_name:<{col_widths['param']}}"

        for window in windows:
            value = data['values'].get(window, '-')
            if isinstance(value, float):
                value_str = f"{value:.4f}" if value < 0.1 else f"{value:.2f}"
            else:
                value_str = str(value) if value != '-' else '-'
            row += f" {value_str:^{col_widths['window']}}"

        # Confidence with color
        conf_level = data['confidence_level']
        conf_score = data['confidence_score'] * 100
        conf_str = f"{conf_level} ({conf_score:.0f}%)"
        color = confidence_color(conf_level)
        row += f" {colorize(conf_str, color):^{col_widths['confidence'] + len(color) + len(Colors.RESET)}}"

        lines.append(row)

    return '\n'.join(lines)


def format_recommendation(
    recommendation: Dict[str, Any],
    ticker: str,
    vix_regime: Dict[str, Any],
    convergence: Dict[str, Dict],
    max_live_capital: float,
    risk_cap: float,
    output_timezone: str = 'PDT'
) -> str:
    """Format the recommended strategy section."""
    lines = [format_subheader(f"Recommended Strategy for Tomorrow")]

    # Trading window
    min_hour = recommendation.get('min_trading_hour', 6)
    max_hour = recommendation.get('max_trading_hour', 9)
    lines.append(f"  Entry Window: {colorize(f'{min_hour}:00 AM - {max_hour}:00 AM {output_timezone}', Colors.GREEN, bold=True)}")
    lines.append(f"  Exit Strategy: Close by {max_hour}:00 AM or at profit target")
    lines.append("")

    # Parameters
    lines.append("  Parameters:")

    params = [
        ('percent_beyond_put', 'percent_beyond_put'),
        ('percent_beyond_call', 'percent_beyond_call'),
        ('max_spread_width_put', 'max_spread_width_put'),
        ('max_spread_width_call', 'max_spread_width_call'),
        ('profit_target_pct', 'profit_target_pct'),
    ]

    for key, display_key in params:
        if key in recommendation:
            value = recommendation[key]
            conf = convergence.get(key, {}).get('confidence_level', '')

            if isinstance(value, float) and value < 0.1:
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)

            if conf:
                color = confidence_color(conf)
                lines.append(f"    {display_key}: {value_str} {colorize(f'({conf} confidence)', color)}")
            else:
                lines.append(f"    {display_key}: {value_str}")

    lines.append("")

    # Capital allocation
    lines.append("  Capital Allocation:")
    lines.append(f"    Single trade risk: ${risk_cap:,.0f}")
    lines.append(f"    Max concurrent capital: ${max_live_capital:,.0f}")

    # Estimate number of trades
    if risk_cap > 0:
        expected_trades = int(max_live_capital / risk_cap)
        lines.append(f"    Expected positions: {expected_trades} concurrent")

    return '\n'.join(lines)


def format_confidence_summary(summary: Dict[str, Any]) -> str:
    """Format the overall confidence summary."""
    lines = []

    level = summary['overall_level']
    score = summary['average_confidence']
    color = confidence_color(level)

    lines.append(f"\n{colorize(f'OVERALL CONFIDENCE: {score:.0f}% ({level})', color, bold=True)}")

    # Breakdown
    high = summary['high_confidence_count']
    med = summary['medium_confidence_count']
    low = summary['low_confidence_count']

    if high > 0:
        lines.append(f"  {colorize('✓', Colors.GREEN)} {high} parameter(s) with HIGH confidence")
    if med > 0:
        lines.append(f"  {colorize('~', Colors.YELLOW)} {med} parameter(s) with MEDIUM confidence")
    if low > 0:
        lines.append(f"  {colorize('!', Colors.RED)} {low} parameter(s) with LOW confidence")

    return '\n'.join(lines)


def format_risk_factors(vix_regime: Dict[str, Any], convergence: Dict[str, Dict]) -> str:
    """Format risk factors and warnings."""
    lines = [format_subheader("Risk Factors")]

    regime = vix_regime.get('regime', 'MEDIUM')

    # VIX-based warnings
    if regime in ('HIGH', 'EXTREME'):
        lines.append(f"  {colorize('!', Colors.RED)} VIX elevated - consider reducing position sizes")
    elif regime in ('LOW', 'STABLE'):
        lines.append(f"  {colorize('✓', Colors.GREEN)} Low volatility - parameters should be reliable")

    # Check for diverging parameters
    for param, data in convergence.items():
        if data['confidence_level'] in ('LOW', 'NONE'):
            lines.append(f"  {colorize('⚠', Colors.YELLOW)} {param} showing divergence across timeframes")

    # General warnings
    if (vix_regime.get('vix_current') or 0) > 0:
        lines.append(f"  - If VIX spikes >25: Consider widening thresholds")

    lines.append(f"  - Monitor early session for unusual activity")

    return '\n'.join(lines)


def generate_console_report(
    ticker: str,
    vix_regime: Dict[str, Any],
    convergence: Dict[str, Dict],
    recommendation: Dict[str, Any],
    summary: Dict[str, Any],
    max_live_capital: float,
    risk_cap: float,
    output_timezone: str = 'PDT',
    windows: Optional[List[str]] = None
) -> str:
    """
    Generate the full console report.

    Args:
        ticker: Underlying ticker
        vix_regime: VIX regime data
        convergence: Convergence analysis results
        recommendation: Recommended parameters
        summary: Confidence summary
        max_live_capital: Maximum live capital
        risk_cap: Risk cap per trade
        output_timezone: Display timezone
        windows: Time windows used (for table headers)

    Returns:
        Formatted report string
    """
    if windows is None:
        windows = ['1yr', '6mo', '3mo', '1mo', '1wk']

    today = date.today()
    tomorrow = today + timedelta(days=1)

    sections = [
        format_header(f"{ticker} STRATEGY RECOMMENDATION - {tomorrow}"),
        format_market_regime(vix_regime),
        format_convergence_table(convergence, windows),
        format_recommendation(
            recommendation, ticker, vix_regime, convergence,
            max_live_capital, risk_cap, output_timezone
        ),
        format_confidence_summary(summary),
        format_risk_factors(vix_regime, convergence),
    ]

    return '\n'.join(sections)


def generate_json_report(
    ticker: str,
    vix_regime: Dict[str, Any],
    convergence: Dict[str, Dict],
    recommendation: Dict[str, Any],
    summary: Dict[str, Any],
    max_live_capital: float,
    risk_cap: float,
    output_timezone: str = 'PDT',
    windows: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate JSON report for programmatic use.

    Returns:
        Dict suitable for JSON serialization
    """
    if windows is None:
        windows = ['1yr', '6mo', '3mo', '1mo', '1wk']

    today = date.today()
    tomorrow = today + timedelta(days=1)

    return {
        'ticker': ticker,
        'generated_at': datetime.now().isoformat(),
        'for_date': str(tomorrow),
        'market_regime': {
            'vix_current': vix_regime.get('vix_current'),
            'vix_10d_avg': vix_regime.get('vix_10d_avg'),
            'vix1d_current': vix_regime.get('vix1d_current'),
            'regime': vix_regime.get('regime'),
            'description': vix_regime.get('regime_description'),
        },
        'convergence': {
            param: {
                'values': data['values'],
                'confidence_score': data['confidence_score'],
                'confidence_level': data['confidence_level'],
                'recommended': data['recommended'],
            }
            for param, data in convergence.items()
        },
        'recommendation': recommendation,
        'confidence': {
            'overall_score': summary['average_confidence'],
            'overall_level': summary['overall_level'],
            'high_count': summary['high_confidence_count'],
            'medium_count': summary['medium_confidence_count'],
            'low_count': summary['low_confidence_count'],
        },
        'capital': {
            'max_live_capital': max_live_capital,
            'risk_cap': risk_cap,
        },
        'settings': {
            'output_timezone': output_timezone,
            'windows_analyzed': windows,
        },
    }


def save_json_report(report: Dict[str, Any], output_path: str) -> None:
    """Save JSON report to file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
