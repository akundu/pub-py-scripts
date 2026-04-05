#!/usr/bin/env python3
"""Backtest V4 — Multi-Ticker, IC Fallback, Advanced Rolling.

Runs the v4 strategy across NDX, SPX, and RUT with iron condor fallback,
chain-aware profit exits, and advanced rolling. Optionally compares against
v3 baseline and IV regime condor strategies.

Usage:
  python run_backtest_v4.py                        # v4 multi-ticker
  python run_backtest_v4.py --compare              # v4 vs v3 vs IC
  python run_backtest_v4.py --tickers NDX,SPX      # override tickers
  python run_backtest_v4.py --report-only          # regenerate report
  python run_backtest_v4.py --help                 # show help
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results" / "backtest_v4"
CHART_DIR = OUTPUT_DIR / "charts"

sys.path.insert(0, str(BASE_DIR))

NUM_WORKERS = min(6, cpu_count())


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_single_strategy(args):
    """Run a single strategy in a subprocess."""
    label, config_source = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    import scripts.backtesting.providers.csv_equity_provider    # noqa
    import scripts.backtesting.providers.csv_options_provider    # noqa
    import scripts.backtesting.instruments.credit_spread         # noqa
    import scripts.backtesting.instruments.iron_condor           # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa
    import scripts.backtesting.strategies.credit_spread.iv_regime_condor  # noqa
    import scripts.backtesting.strategies.credit_spread.backtest_v4       # noqa
    import scripts.backtesting.strategies.credit_spread.backtest_v5       # noqa

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    try:
        if isinstance(config_source, dict):
            config = BacktestConfig.from_dict(config_source)
        else:
            config = BacktestConfig.load(str(BASE_DIR / config_source))

        engine = BacktestEngine(config)
        results = engine.run(dry_run=False)

        return (label, {
            "trades": results.get("trades", []),
            "metrics": results.get("metrics", {}),
        })
    except Exception as e:
        import traceback
        return (label, {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "trades": [],
            "metrics": {},
        })


def _build_v3_baseline_config(ticker: str = "NDX", start_date: str = "2025-03-13",
                              end_date: str = "2026-03-13") -> dict:
    """Build a v3 baseline config for comparison."""
    # RUT uses smaller spread width
    spread_width = 10 if ticker == "RUT" else 50
    return {
        "infra": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "lookback_days": 250,
            "output_dir": f"results/backtest_v4/v3_{ticker.lower()}",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output_full", "dte_buckets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": {
                "dte": 0, "percentile": 95, "lookback": 120,
                "option_types": ["put", "call"], "spread_width": spread_width,
                "interval_minutes": 10, "entry_start_utc": "13:00",
                "entry_end_utc": "17:00", "num_contracts": 1,
                "max_loss_estimate": 10000, "min_credit": 0.30,
                "use_mid": True, "directional_entry": "pursuit",
                "roll_enabled": True, "max_rolls": 2,
                "roll_check_start_utc": "18:00", "max_move_cap": 150,
            },
        },
        "constraints": {
            "budget": {"max_spend_per_transaction": 20000, "daily_budget": 200000},
            "trading_hours": {"entry_start": "13:00", "entry_end": "17:00"},
            "exit_rules": {"profit_target_pct": 0.50, "stop_loss_pct": 2.0, "mode": "first_triggered"},
        },
        "report": {"formats": ["csv"], "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"]},
    }


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades: list) -> dict:
    """Compute standard backtest metrics."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_pnl": 0, "avg_pnl": 0, "roi": 0, "sharpe": 0,
            "max_drawdown": 0, "profit_factor": 0,
        }

    pnl_values = np.array([float(t.get("pnl", 0)) for t in trades])
    credits = np.array([float(t.get("credit", t.get("initial_credit", 0)) * 100) for t in trades])
    risks = np.array([abs(float(t.get("max_loss", 0))) for t in trades])

    total = len(pnl_values)
    wins = int((pnl_values > 0).sum())
    losses = total - wins

    total_gains = float(pnl_values[pnl_values > 0].sum()) if wins > 0 else 0
    total_losses_val = float(abs(pnl_values[pnl_values < 0].sum())) if losses > 0 else 0

    cumulative = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0

    sharpe = 0
    if total > 1 and pnl_values.std(ddof=1) > 0:
        sharpe = float((pnl_values.mean() / pnl_values.std(ddof=1)) * np.sqrt(252))

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "net_pnl": round(float(pnl_values.sum()), 2),
        "avg_pnl": round(float(pnl_values.mean()), 2),
        "roi": round(float(credits.sum()) / max(float(risks.sum()), 1) * 100, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_dd, 2),
        "profit_factor": round(total_gains / max(total_losses_val, 0.01), 2),
    }


# ---------------------------------------------------------------------------
# Multi-ticker analytics
# ---------------------------------------------------------------------------

def compute_ticker_analytics(trades: list) -> dict:
    """Compute per-ticker metrics."""
    by_ticker = {}
    for t in trades:
        ticker = t.get("ticker", t.get("metadata", {}).get("ticker", "Unknown"))
        by_ticker.setdefault(ticker, []).append(t)

    result = {}
    for ticker, ticker_trades in sorted(by_ticker.items()):
        result[ticker] = compute_metrics(ticker_trades)
        result[ticker]["trade_count"] = len(ticker_trades)
    return result


def compute_roll_analytics(trades: list) -> dict:
    """Compute roll analytics from trade results."""
    roll_types = {"breach_roll": [], "expiry_roll": [], "p95_roll": []}

    for t in trades:
        reason = t.get("exit_reason", "")
        chain_pnl = t.get("metadata", {}).get("total_chain_pnl", t.get("total_chain_pnl", None))
        roll_count = t.get("metadata", {}).get("roll_count", t.get("roll_count", 0))

        if not reason.startswith("roll_trigger"):
            continue

        if "breach" in reason:
            cat = "breach_roll"
        elif "expiry" in reason:
            cat = "expiry_roll"
        else:
            cat = "p95_roll"

        entry = {"reason": reason, "pnl": t.get("pnl", 0), "roll_count": roll_count}
        if chain_pnl is not None:
            entry["chain_pnl"] = chain_pnl
        roll_types[cat].append(entry)

    summary = {}
    for cat, rolls in roll_types.items():
        if not rolls:
            summary[cat] = {"count": 0, "avg_pnl": 0, "success_rate": 0}
            continue
        pnls = [r.get("chain_pnl", r["pnl"]) for r in rolls]
        summary[cat] = {
            "count": len(rolls),
            "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
            "success_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100, 1),
        }
    return summary


def compute_liquidity_analytics(trades: list) -> dict:
    """Compute liquidity-related analytics from trades with liquidity metadata."""
    liq_trades = []
    for t in trades:
        meta = t.get("metadata", {})
        liq = meta.get("liquidity", t.get("liquidity"))
        if liq and isinstance(liq, dict):
            liq_trades.append({
                "ticker": t.get("ticker", meta.get("ticker", "Unknown")),
                "liquidity_score": liq.get("liquidity_score", 0),
                "avg_bid_ask_pct": liq.get("avg_bid_ask_pct", 0),
                "avg_volume": liq.get("avg_volume", 0),
                "avg_iv": liq.get("avg_iv", 0),
                "pnl": float(t.get("pnl", 0)),
            })

    if not liq_trades:
        return {"has_data": False}

    by_ticker = {}
    for lt in liq_trades:
        ticker = lt["ticker"]
        by_ticker.setdefault(ticker, []).append(lt)

    ticker_liq = {}
    for ticker, entries in sorted(by_ticker.items()):
        scores = [e["liquidity_score"] for e in entries]
        ba_pcts = [e["avg_bid_ask_pct"] for e in entries]
        volumes = [e["avg_volume"] for e in entries]
        pnls = [e["pnl"] for e in entries]
        ticker_liq[ticker] = {
            "trades": len(entries),
            "avg_liq_score": round(np.mean(scores), 3) if scores else 0,
            "avg_bid_ask_pct": round(np.mean(ba_pcts) * 100, 2) if ba_pcts else 0,
            "avg_volume": round(np.mean(volumes), 1) if volumes else 0,
            "avg_pnl": round(np.mean(pnls), 2) if pnls else 0,
        }

    return {"has_data": True, "by_ticker": ticker_liq}


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def generate_html_report(
    all_results: dict,
    ticker_analytics: dict,
    roll_analytics: dict,
    output_dir: Path,
    chart_dir: Path,
    liquidity_analytics: dict = None,
):
    """Generate comprehensive HTML report with roll and multi-ticker analytics."""
    os.makedirs(chart_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    # Generate charts
    _generate_charts(all_results, ticker_analytics, roll_analytics, chart_dir)

    # Build HTML
    strategies_html = ""
    for label, data in all_results.items():
        m = data.get("metrics", {})
        if isinstance(m, dict) and m.get("total_trades", 0) > 0:
            strategies_html += f"""
            <tr>
                <td>{label}</td>
                <td>{m.get('total_trades', 0)}</td>
                <td>{m.get('win_rate', 0)}%</td>
                <td>${m.get('net_pnl', 0):,.0f}</td>
                <td>${m.get('avg_pnl', 0):,.0f}</td>
                <td>{m.get('sharpe', 0)}</td>
                <td>${m.get('max_drawdown', 0):,.0f}</td>
                <td>{m.get('profit_factor', 0)}</td>
            </tr>"""

    # Ticker allocation table
    ticker_html = ""
    for ticker, m in ticker_analytics.items():
        ticker_html += f"""
        <tr>
            <td>{ticker}</td>
            <td>{m.get('trade_count', 0)}</td>
            <td>{m.get('win_rate', 0)}%</td>
            <td>${m.get('net_pnl', 0):,.0f}</td>
            <td>${m.get('avg_pnl', 0):,.0f}</td>
            <td>{m.get('sharpe', 0)}</td>
        </tr>"""

    # Roll analytics table
    roll_html = ""
    for cat, data in roll_analytics.items():
        roll_html += f"""
        <tr>
            <td>{cat.replace('_', ' ').title()}</td>
            <td>{data.get('count', 0)}</td>
            <td>{data.get('success_rate', 0)}%</td>
            <td>${data.get('avg_pnl', 0):,.0f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest V4 — Multi-Ticker Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; font-size: 14px; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
.hero {{ background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); border: 1px solid #30363d; border-radius: 12px; padding: 40px; margin-bottom: 30px; text-align: center; }}
.hero h1 {{ color: #58a6ff; font-size: 28px; margin-bottom: 8px; }}
.hero .subtitle {{ color: #8b949e; font-size: 16px; }}
.kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 30px; }}
.kpi-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; text-align: center; }}
.kpi-card .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
.kpi-card .label {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
.section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 24px; margin-bottom: 24px; }}
.section h2 {{ color: #58a6ff; margin-bottom: 16px; font-size: 20px; }}
.section h3 {{ color: #c9d1d9; margin: 16px 0 8px; }}
.narrative {{ color: #8b949e; margin: 12px 0; font-size: 14px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #58a6ff; font-weight: 600; background: #0d1117; }}
td {{ color: #c9d1d9; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
img {{ max-width: 100%; border-radius: 8px; margin: 12px 0; }}
.chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
.footer {{ text-align: center; padding: 20px; color: #484f58; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">

<div class="hero">
    <h1>Backtest V4 — Multi-Ticker, IC Fallback, Advanced Rolling</h1>
    <div class="subtitle">NDX / SPX / RUT | Chain-Aware Exits | Report generated {today}</div>
</div>

<div class="kpi-strip">
    {"".join(f'''<div class="kpi-card"><div class="value">{v}</div><div class="label">{k}</div></div>'''
        for k, v in _get_kpi_values(all_results).items())}
</div>

<div class="section">
    <h2>Strategy Comparison</h2>
    <p class="narrative">Performance metrics across all strategy variants. V4 evaluates NDX, SPX, and RUT simultaneously at each interval, picking the best credit/risk opportunities.</p>
    <table>
        <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>Avg P&L</th><th>Sharpe</th><th>Max DD</th><th>PF</th></tr>
        {strategies_html}
    </table>
</div>

<div class="section">
    <h2>Multi-Ticker Allocation</h2>
    <p class="narrative">How trades were distributed across tickers. Higher credit/risk ratios drive allocation.</p>
    <div class="chart-grid">
        <div><img src="charts/ticker_allocation_pie.png" alt="Ticker Allocation"></div>
        <div>
            <table>
                <tr><th>Ticker</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th><th>Avg P&L</th><th>Sharpe</th></tr>
                {ticker_html}
            </table>
        </div>
    </div>
</div>

<div class="section">
    <h2>Roll Analytics</h2>
    <p class="narrative">Breakdown of roll triggers by category. Breach rolls fire when price penetrates the short strike with low DTE. Expiry rolls fire on expiration day based on loss/credit ratio. P95 rolls use the historical remaining move distribution.</p>
    <table>
        <tr><th>Roll Type</th><th>Count</th><th>Success Rate</th><th>Avg Chain P&L</th></tr>
        {roll_html}
    </table>
    <div class="chart-grid">
        <div><img src="charts/roll_frequency.png" alt="Roll Frequency"></div>
        <div><img src="charts/cumulative_pnl.png" alt="Cumulative P&L"></div>
    </div>
</div>

<div class="section">
    <h2>Liquidity Analytics</h2>
    <p class="narrative">Liquidity scoring accounts for bid-ask spread width (40%), volume (30%), quote availability (20%), and IV presence (10%). Higher scores indicate better fills and more reliable pricing.</p>
    {_build_liquidity_html(liquidity_analytics)}
</div>

<div class="section">
    <h2>Cumulative P&L</h2>
    <p class="narrative">Shows how P&L accumulates over the backtest period for each strategy variant.</p>
    <img src="charts/cumulative_pnl.png" alt="Cumulative P&L">
</div>

<div class="section">
    <h2>Monthly Breakdown</h2>
    <img src="charts/monthly_pnl.png" alt="Monthly P&L">
</div>

<div class="footer">
    Generated by run_backtest_v4.py | Backtest V4 Multi-Ticker Strategy
</div>

</div>
</body>
</html>"""

    report_path = output_dir / f"report_backtest_v4_{today}.html"
    report_path.write_text(html)

    # Create index.html symlink
    index_path = output_dir / "index.html"
    if index_path.exists() or index_path.is_symlink():
        index_path.unlink()
    index_path.symlink_to(report_path.name)

    print(f"Report: {report_path}")
    return report_path


def _build_liquidity_html(liquidity_analytics: dict) -> str:
    """Build HTML for liquidity analytics section."""
    if not liquidity_analytics or not liquidity_analytics.get("has_data"):
        return '<p class="narrative">No liquidity data available (trades may not have liquidity metadata).</p>'

    rows = ""
    for ticker, m in liquidity_analytics.get("by_ticker", {}).items():
        score = m.get("avg_liq_score", 0)
        score_class = "positive" if score >= 0.6 else ("negative" if score < 0.3 else "")
        rows += f"""
        <tr>
            <td>{ticker}</td>
            <td>{m.get('trades', 0)}</td>
            <td class="{score_class}">{score:.3f}</td>
            <td>{m.get('avg_bid_ask_pct', 0):.1f}%</td>
            <td>{m.get('avg_volume', 0):.0f}</td>
            <td>${m.get('avg_pnl', 0):,.0f}</td>
        </tr>"""

    return f"""
    <table>
        <tr><th>Ticker</th><th>Trades</th><th>Avg Liquidity Score</th><th>Avg Bid-Ask %</th><th>Avg Volume</th><th>Avg P&L</th></tr>
        {rows}
    </table>"""


def _get_kpi_values(all_results: dict) -> dict:
    """Extract KPI values from the primary strategy."""
    primary = None
    for label, data in all_results.items():
        if "v4" in label.lower():
            primary = data.get("metrics", {})
            break
    if primary is None:
        primary = next(iter(all_results.values()), {}).get("metrics", {})

    return {
        "Total Trades": primary.get("total_trades", 0),
        "Win Rate": f"{primary.get('win_rate', 0)}%",
        "Net P&L": f"${primary.get('net_pnl', 0):,.0f}",
        "Sharpe": primary.get("sharpe", 0),
        "Max Drawdown": f"${primary.get('max_drawdown', 0):,.0f}",
        "Profit Factor": primary.get("profit_factor", 0),
    }


def _generate_charts(all_results, ticker_analytics, roll_analytics, chart_dir):
    """Generate all charts."""
    plt.style.use("dark_background")

    # Ticker allocation pie chart
    if ticker_analytics:
        fig, ax = plt.subplots(figsize=(8, 6))
        tickers = list(ticker_analytics.keys())
        counts = [ticker_analytics[t].get("trade_count", 0) for t in tickers]
        colors = ["#58a6ff", "#3fb950", "#f0883e", "#f85149", "#a371f7"]
        if sum(counts) > 0:
            ax.pie(counts, labels=tickers, autopct="%1.0f%%", colors=colors[:len(tickers)])
            ax.set_title("Trade Allocation by Ticker", color="#c9d1d9")
        fig.savefig(chart_dir / "ticker_allocation_pie.png", dpi=100, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)

    # Roll frequency bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(roll_analytics.keys())
    counts = [roll_analytics[c].get("count", 0) for c in cats]
    colors = ["#f85149", "#f0883e", "#58a6ff"]
    ax.bar([c.replace("_", " ").title() for c in cats], counts, color=colors[:len(cats)])
    ax.set_title("Roll Trigger Frequency by Type", color="#c9d1d9")
    ax.set_ylabel("Count")
    fig.savefig(chart_dir / "roll_frequency.png", dpi=100, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)

    # Cumulative P&L per strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_cycle = ["#58a6ff", "#3fb950", "#f0883e", "#f85149", "#a371f7"]
    for i, (label, data) in enumerate(all_results.items()):
        trades = data.get("trades", [])
        if not trades:
            continue
        pnls = [float(t.get("pnl", 0)) for t in trades]
        cum_pnl = np.cumsum(pnls)
        ax.plot(range(len(cum_pnl)), cum_pnl, label=label,
                color=colors_cycle[i % len(colors_cycle)], linewidth=1.5)
    ax.set_title("Cumulative P&L by Strategy", color="#c9d1d9")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.1)
    fig.savefig(chart_dir / "cumulative_pnl.png", dpi=100, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)

    # Monthly P&L
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, data) in enumerate(all_results.items()):
        trades = data.get("trades", [])
        if not trades:
            continue
        monthly = {}
        for t in trades:
            td = t.get("trading_date", t.get("entry_date", ""))
            if isinstance(td, str) and len(td) >= 7:
                month = td[:7]
            elif hasattr(td, "strftime"):
                month = td.strftime("%Y-%m")
            else:
                continue
            monthly[month] = monthly.get(month, 0) + float(t.get("pnl", 0))
        if monthly:
            months = sorted(monthly.keys())
            values = [monthly[m] for m in months]
            ax.bar([m for m in months], values, alpha=0.7,
                   label=label, color=colors_cycle[i % len(colors_cycle)])
    ax.set_title("Monthly P&L", color="#c9d1d9")
    ax.set_ylabel("P&L ($)")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.1)
    fig.savefig(chart_dir / "monthly_pnl.png", dpi=100, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="""
Backtest V4 — Multi-Ticker, IC Fallback, Advanced Rolling.

Evaluates NDX, SPX, and RUT simultaneously at each interval, picks best
credit/risk opportunities, with iron condor fallback in low-IV and
chain-aware profit exits.
        """,
        epilog="""
Examples:
  %(prog)s                                  Run v4 multi-ticker backtest
  %(prog)s --compare                        Compare v4 vs v3 vs IC
  %(prog)s --tickers NDX,SPX                Override tickers
  %(prog)s --report-only                    Regenerate report from existing results
  %(prog)s --help                           Show this help message
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison: v4 vs v3 vs IC")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Override tickers (comma-separated, e.g. NDX,SPX)")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate report from existing CSVs")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Override end date (YYYY-MM-DD)")

    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    if args.report_only:
        print("Report-only mode: loading existing results...")
        all_results = _load_existing_results(OUTPUT_DIR)
        if not all_results:
            print("No existing results found. Run backtest first.")
            return
    else:
        all_results = _run_backtests(args)

    if not all_results:
        print("No results to report.")
        return

    # Compute analytics
    primary_trades = []
    for label, data in all_results.items():
        if "v4" in label.lower():
            primary_trades = data.get("trades", [])
            break
    if not primary_trades:
        primary_trades = next(iter(all_results.values()), {}).get("trades", [])

    ticker_analytics = compute_ticker_analytics(primary_trades)
    roll_analytics_data = compute_roll_analytics(primary_trades)
    liquidity_analytics_data = compute_liquidity_analytics(primary_trades)

    # Generate report
    generate_html_report(
        all_results, ticker_analytics, roll_analytics_data,
        OUTPUT_DIR, CHART_DIR,
        liquidity_analytics=liquidity_analytics_data,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST V4 RESULTS")
    print("=" * 70)
    for label, data in all_results.items():
        m = data.get("metrics", {})
        if "error" in data:
            print(f"\n{label}: ERROR - {data['error']}")
            continue
        print(f"\n{label}:")
        print(f"  Trades: {m.get('total_trades', 0)}")
        print(f"  Win Rate: {m.get('win_rate', 0)}%")
        print(f"  Net P&L: ${m.get('net_pnl', 0):,.0f}")
        print(f"  Sharpe: {m.get('sharpe', 0)}")
        print(f"  Max DD: ${m.get('max_drawdown', 0):,.0f}")
        print(f"  Profit Factor: {m.get('profit_factor', 0)}")

    if ticker_analytics:
        print(f"\nTicker Allocation:")
        for ticker, m in ticker_analytics.items():
            print(f"  {ticker}: {m.get('trade_count', 0)} trades, "
                  f"{m.get('win_rate', 0)}% WR, ${m.get('net_pnl', 0):,.0f} P&L")

    if roll_analytics_data:
        print(f"\nRoll Analytics:")
        for cat, data in roll_analytics_data.items():
            if data.get("count", 0) > 0:
                print(f"  {cat}: {data['count']} rolls, "
                      f"{data.get('success_rate', 0)}% success, "
                      f"${data.get('avg_pnl', 0):,.0f} avg chain P&L")

    if liquidity_analytics_data and liquidity_analytics_data.get("has_data"):
        print(f"\nLiquidity Analytics:")
        for ticker, m in liquidity_analytics_data.get("by_ticker", {}).items():
            print(f"  {ticker}: score={m.get('avg_liq_score', 0):.3f}, "
                  f"bid-ask={m.get('avg_bid_ask_pct', 0):.1f}%, "
                  f"vol={m.get('avg_volume', 0):.0f}, "
                  f"P&L=${m.get('avg_pnl', 0):,.0f}")


def _run_backtests(args) -> dict:
    """Run backtests (v4 and optionally comparison strategies)."""
    configs = []

    tickers = args.tickers.split(",") if args.tickers else ["NDX"]
    start_date = args.start_date or "2025-03-13"
    end_date = args.end_date or "2026-03-13"

    for ticker in tickers:
        t_lower = ticker.lower()
        v4_config = f"scripts/backtesting/configs/backtest_v4_{t_lower}.yaml"
        if not (BASE_DIR / v4_config).exists():
            # Use NDX config as fallback
            v4_config = "scripts/backtesting/configs/backtest_v4_ndx.yaml"
        configs.append((f"V4 ({ticker})", v4_config))

        if args.compare:
            configs.append((f"V3 ({ticker})", _build_v3_baseline_config(
                ticker=ticker, start_date=start_date, end_date=end_date,
            )))

    print(f"Running {len(configs)} strategy backtests with {NUM_WORKERS} workers...")

    if len(configs) == 1:
        results = [run_single_strategy(configs[0])]
    else:
        with Pool(processes=min(len(configs), NUM_WORKERS)) as pool:
            results = pool.map(run_single_strategy, configs)

    all_results = {}
    for label, data in results:
        if "error" in data:
            print(f"WARNING: {label} failed: {data['error']}")
        trades = data.get("trades", [])
        metrics = compute_metrics(trades) if trades else data.get("metrics", {})
        all_results[label] = {"trades": trades, "metrics": metrics}

        # Save trades CSV
        if trades:
            _save_trades_csv(trades, label, OUTPUT_DIR)

    return all_results


def _save_trades_csv(trades: list, label: str, output_dir: Path):
    """Save trades to CSV."""
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    csv_path = output_dir / f"trades_{safe_label}.csv"
    df = pd.DataFrame(trades)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(trades)} trades to {csv_path}")


def _load_existing_results(output_dir: Path) -> dict:
    """Load existing trade CSVs for report regeneration."""
    results = {}
    for csv_path in output_dir.glob("trades_*.csv"):
        label = csv_path.stem.replace("trades_", "").replace("_", " ").title()
        try:
            df = pd.read_csv(csv_path)
            trades = df.to_dict("records")
            metrics = compute_metrics(trades)
            results[label] = {"trades": trades, "metrics": metrics}
        except Exception as e:
            print(f"Warning: Failed to load {csv_path}: {e}")
    return results


if __name__ == "__main__":
    main()
