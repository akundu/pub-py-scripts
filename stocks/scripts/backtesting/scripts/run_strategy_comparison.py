#!/usr/bin/env python3
"""Strategy Comparison — New Strategies vs v3 Baseline.

Runs each new strategy (IV Regime Condor, Weekly Iron Condor, Tail Hedged)
on NDX and compares against the v3 cross-ticker portfolio baseline.

Also runs a baseline percentile_entry_credit_spread (single-ticker NDX) to
provide a fair apples-to-apples comparison on the same date range.

Usage:
  python run_strategy_comparison.py                 # Full run
  python run_strategy_comparison.py --analyze       # Re-analyze existing results
  python run_strategy_comparison.py --strategies iv_regime_condor tail_hedged
      Run only specific strategies
  python run_strategy_comparison.py --help
      Show this help message
"""

import argparse
import os
import sys
import warnings
import yaml
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
OUTPUT_DIR = BASE_DIR / "results" / "strategy_comparison"
CHART_DIR = OUTPUT_DIR / "charts"

sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------------
# Strategy configurations
# ---------------------------------------------------------------------------

STRATEGIES = {
    "baseline_ndx": {
        "label": "Baseline (P95 Credit Spread, NDX)",
        "config_path": None,  # Built programmatically
        "color": "#8b949e",
    },
    "iv_regime_condor": {
        "label": "IV Regime Iron Condor",
        "config_path": "scripts/backtesting/configs/iv_regime_condor_ndx.yaml",
        "color": "#3498db",
    },
    "weekly_iron_condor": {
        "label": "Weekly Iron Condor (5-7 DTE)",
        "config_path": "scripts/backtesting/configs/weekly_iron_condor_ndx.yaml",
        "color": "#2ecc71",
    },
    "tail_hedged": {
        "label": "Tail Hedged Credit Spread",
        "config_path": "scripts/backtesting/configs/tail_hedged_ndx.yaml",
        "color": "#e74c3c",
    },
}


def _build_baseline_config() -> dict:
    """Build a baseline P95 credit spread config matching v3's NDX settings."""
    return {
        "infra": {
            "ticker": "NDX",
            "lookback_days": 250,
            "output_dir": "results/strategy_comparison/baseline_ndx",
        },
        "providers": [
            {"name": "csv_equity", "role": "equity", "params": {"csv_dir": "equities_output"}},
            {"name": "csv_options", "role": "options", "params": {
                "csv_dir": "options_csv_output",
                "dte_buckets": [0],
            }},
        ],
        "strategy": {
            "name": "percentile_entry_credit_spread",
            "params": {
                "dte": 0,
                "percentile": 95,
                "lookback": 120,
                "option_types": ["put", "call"],
                "spread_width": 50,
                "interval_minutes": 10,
                "entry_start_utc": "13:00",
                "entry_end_utc": "17:00",
                "num_contracts": 1,
                "max_loss_estimate": 10000,
                "min_credit": 0.30,
                "use_mid": True,
                "directional_entry": "both",
            },
        },
        "constraints": {
            "budget": {
                "max_spend_per_transaction": 20000,
                "daily_budget": 100000,
            },
            "trading_hours": {
                "entry_start": "13:00",
                "entry_end": "17:00",
                "forced_exit_time": "20:45",
            },
            "exit_rules": {
                "profit_target_pct": 0.50,
                "stop_loss_pct": 2.0,
                "time_exit": "20:30",
                "mode": "first_triggered",
            },
        },
        "report": {
            "formats": ["csv"],
            "metrics": ["win_rate", "roi", "sharpe", "max_drawdown", "profit_factor"],
        },
    }


# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------

def run_single_strategy(args):
    """Run a single strategy backtest in a subprocess."""
    strategy_name, config_source = args

    sys.path.insert(0, str(BASE_DIR))
    os.chdir(str(BASE_DIR))

    import scripts.backtesting.providers.csv_equity_provider    # noqa
    import scripts.backtesting.providers.csv_options_provider    # noqa
    import scripts.backtesting.instruments.credit_spread         # noqa
    import scripts.backtesting.instruments.iron_condor           # noqa
    import scripts.backtesting.strategies.credit_spread.percentile_entry  # noqa
    import scripts.backtesting.strategies.credit_spread.iv_regime_condor  # noqa
    import scripts.backtesting.strategies.credit_spread.weekly_iron_condor  # noqa
    import scripts.backtesting.strategies.credit_spread.tail_hedged  # noqa

    from scripts.backtesting.config import BacktestConfig
    from scripts.backtesting.engine import BacktestEngine

    import logging
    logging.basicConfig(level=logging.WARNING)

    try:
        if isinstance(config_source, dict):
            # Programmatic config (baseline)
            config_path = BASE_DIR / f"_tmp_{strategy_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config_source, f)
            config = BacktestConfig.from_yaml(str(config_path))
            config_path.unlink(missing_ok=True)
        else:
            config = BacktestConfig.load(str(BASE_DIR / config_source))

        engine = BacktestEngine(config)
        results = engine.run(dry_run=False)

        # Extract trades from collector
        trades = results.get("trades", [])
        if not trades and hasattr(engine, "collector") and engine.collector:
            trades = engine.collector._results

        metrics = results.get("metrics", {})

        return (strategy_name, {
            "trades": trades,
            "metrics": metrics,
            "num_dates": results.get("dates", 0),
        })
    except Exception as e:
        import traceback
        return (strategy_name, {
            "trades": [],
            "metrics": {},
            "error": f"{e}\n{traceback.format_exc()}",
        })


def compute_metrics(trades: list) -> dict:
    """Compute standard metrics from a list of trade dicts."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "net_pnl": 0, "avg_pnl": 0, "total_credit": 0, "total_risk": 0,
            "roi": 0, "sharpe": 0, "max_drawdown": 0, "profit_factor": 0,
        }

    pnl_values = np.array([float(t.get("pnl", 0)) for t in trades])
    credits = np.array([float(t.get("credit", 0)) for t in trades])
    risks = np.array([abs(float(t.get("max_loss", 0))) for t in trades])

    total = len(pnl_values)
    wins = int((pnl_values > 0).sum())
    losses = total - wins
    net_pnl = float(pnl_values.sum())
    total_credit = float(credits.sum())
    total_risk = float(risks.sum())

    roi = (total_credit / total_risk * 100) if total_risk > 0 else 0

    sharpe = 0.0
    if total > 1 and pnl_values.std() > 0:
        sharpe = float((pnl_values.mean() / pnl_values.std(ddof=1)) * np.sqrt(252))

    cum_pnl = np.cumsum(pnl_values)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((peak - cum_pnl).max())

    total_gains = float(pnl_values[pnl_values > 0].sum())
    total_losses_val = float(np.abs(pnl_values[pnl_values <= 0]).sum())
    pf = total_gains / total_losses_val if total_losses_val > 0 else float("inf")

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total * 100 if total > 0 else 0,
        "net_pnl": net_pnl,
        "avg_pnl": net_pnl / total if total > 0 else 0,
        "total_credit": total_credit,
        "total_risk": total_risk,
        "roi": roi,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
    }


# ---------------------------------------------------------------------------
# Analysis & Comparison
# ---------------------------------------------------------------------------

def load_v3_baseline() -> dict:
    """Load v3 portfolio metrics from existing results."""
    v3_trades_path = BASE_DIR / "results" / "tiered_portfolio_v3" / "portfolio_trades.csv"
    if not v3_trades_path.exists():
        print("  WARNING: v3 results not found. Run run_tiered_backtest_v3.py first.")
        return {
            "total_trades": 5157, "win_rate": 96.7, "net_pnl": 32120000,
            "roi": 46.1, "sharpe": 5.52, "max_drawdown": 809690,
            "profit_factor": 10.39,
            "note": "hardcoded from last v3 run (2026-03-10)",
        }

    df = pd.read_csv(v3_trades_path)
    pnl_col = "adjusted_pnl" if "adjusted_pnl" in df.columns else "pnl"
    pnl = df[pnl_col].values
    total = len(pnl)
    wins = int((pnl > 0).sum())

    credit_col = "adjusted_credit" if "adjusted_credit" in df.columns else "credit"
    risk_col = "adjusted_max_loss" if "adjusted_max_loss" in df.columns else "max_loss"
    total_credit = float(df[credit_col].sum()) if credit_col in df.columns else 0
    total_risk = float(df[risk_col].abs().sum()) if risk_col in df.columns else 0

    roi = (total_credit / total_risk * 100) if total_risk > 0 else 0
    sharpe = 0
    if total > 1 and pnl.std() > 0:
        sharpe = float((pnl.mean() / pnl.std(ddof=1)) * np.sqrt(252))
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float((peak - cum_pnl).max())
    total_gains = float(pnl[pnl > 0].sum())
    total_losses_val = float(np.abs(pnl[pnl <= 0]).sum())
    pf = total_gains / total_losses_val if total_losses_val > 0 else float("inf")

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total * 100 if total > 0 else 0,
        "net_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "roi": roi,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "note": "from portfolio_trades.csv (3 tickers, volume-adjusted)",
    }


def print_comparison(all_metrics: dict):
    """Print side-by-side comparison table."""
    print()
    print("=" * 120)
    print("  STRATEGY COMPARISON")
    print("=" * 120)

    # Header
    names = list(all_metrics.keys())
    col_width = max(20, max(len(STRATEGIES.get(n, {}).get("label", n)) for n in names) + 2)

    print(f"\n  {'Metric':<22}", end="")
    for name in names:
        label = STRATEGIES.get(name, {}).get("label", name)
        if name == "v3_baseline":
            label = "v3 Cross-Ticker"
        print(f" {label:>{col_width}}", end="")
    print()
    print(f"  {'─' * 22}", end="")
    for _ in names:
        print(f" {'─' * col_width}", end="")
    print()

    # Rows
    metrics_to_show = [
        ("Total Trades", "total_trades", "{:,.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Net P&L", "net_pnl", "${:,.0f}"),
        ("Avg P&L/Trade", "avg_pnl", "${:,.0f}"),
        ("ROI", "roi", "{:.1f}%"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "${:,.0f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
    ]

    for label, key, fmt in metrics_to_show:
        print(f"  {label:<22}", end="")
        for name in names:
            m = all_metrics[name]
            val = m.get(key, 0)
            if key == "profit_factor" and val > 999:
                formatted = "∞"
            else:
                formatted = fmt.format(val)
            print(f" {formatted:>{col_width}}", end="")
        print()

    # Notes
    for name in names:
        note = all_metrics[name].get("note")
        if note:
            print(f"\n  [{name}] {note}")

    print("\n" + "=" * 120)


def generate_comparison_charts(all_results: dict, all_metrics: dict):
    """Generate comparison charts."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Metrics comparison bar chart
    _chart_metrics_bars(all_metrics)

    # 2. Cumulative P&L comparison
    _chart_cumulative_pnl(all_results)

    print(f"\nCharts saved to {CHART_DIR}/")


def _chart_metrics_bars(all_metrics: dict):
    """Bar chart comparing key metrics across strategies."""
    names = [n for n in all_metrics if n != "v3_baseline"]
    labels = [STRATEGIES.get(n, {}).get("label", n) for n in names]
    colors = [STRATEGIES.get(n, {}).get("color", "#888") for n in names]

    metrics = ["win_rate", "roi", "sharpe", "profit_factor"]
    metric_labels = ["Win Rate (%)", "ROI (%)", "Sharpe", "Profit Factor"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        values = []
        for name in names:
            val = all_metrics[name].get(metric, 0)
            if metric == "profit_factor" and val > 100:
                val = 100  # Cap for display
            values.append(val)

        x = np.arange(len(names))
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(mlabel, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        # Add v3 baseline line if available
        v3_val = all_metrics.get("v3_baseline", {}).get(metric, 0)
        if v3_val > 0:
            if metric == "profit_factor" and v3_val > 100:
                v3_val = 100
            ax.axhline(y=v3_val, color="#f39c12", linestyle="--", linewidth=2,
                       label=f"v3 baseline: {v3_val:.1f}")
            ax.legend(fontsize=7)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)


def _chart_cumulative_pnl(all_results: dict):
    """Cumulative P&L for each strategy."""
    fig, ax = plt.subplots(figsize=(16, 8))

    for name, result in all_results.items():
        trades = result.get("trades", [])
        if not trades:
            continue
        pnl_values = [float(t.get("pnl", 0)) for t in trades]
        cum_pnl = np.cumsum(pnl_values)
        color = STRATEGIES.get(name, {}).get("color", "#888")
        label = STRATEGIES.get(name, {}).get("label", name)
        ax.plot(range(len(cum_pnl)), cum_pnl, label=label, color=color,
                linewidth=2, alpha=0.85)

    ax.set_title("Cumulative P&L Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "cumulative_pnl_comparison.png", dpi=150)
    plt.close(fig)


def generate_html_report(all_metrics: dict, all_results: dict):
    """Generate comparison HTML report."""
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Build comparison table rows
    names = list(all_metrics.keys())
    metrics_rows = [
        ("Total Trades", "total_trades", "{:,.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Net P&L", "net_pnl", "${:,.0f}"),
        ("Avg P&L/Trade", "avg_pnl", "${:,.0f}"),
        ("ROI", "roi", "{:.1f}%"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "${:,.0f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
    ]

    header_cells = "".join(
        f'<th>{STRATEGIES.get(n, {}).get("label", n) if n != "v3_baseline" else "v3 Cross-Ticker"}</th>'
        for n in names
    )

    body_rows = ""
    for label, key, fmt in metrics_rows:
        cells = ""
        for n in names:
            val = all_metrics[n].get(key, 0)
            if key == "profit_factor" and val > 999:
                formatted = "&infin;"
            else:
                formatted = fmt.format(val)
            # Color code: green if better than v3 baseline
            v3_val = all_metrics.get("v3_baseline", {}).get(key, 0)
            css = ""
            if n != "v3_baseline" and v3_val > 0:
                if key in ("win_rate", "roi", "sharpe", "profit_factor", "net_pnl", "avg_pnl"):
                    css = ' class="positive"' if val > v3_val else ' class="negative"' if val < v3_val * 0.8 else ''
                elif key == "max_drawdown":
                    css = ' class="positive"' if val < v3_val else ' class="negative"' if val > v3_val * 1.5 else ''
            cells += f"<td{css}>{formatted}</td>"
        body_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strategy Comparison — {today_str}</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
    --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; }}
  .hero {{ background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border); padding: 60px 40px; text-align: center; }}
  .hero h1 {{ font-size: 2.4em; font-weight: 700; background: linear-gradient(90deg, #3498db, #2ecc71);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero .subtitle {{ font-size: 1.1em; color: var(--muted); max-width: 900px; margin: 10px auto; }}
  .hero .date {{ margin-top: 12px; font-size: 0.9em; color: var(--muted); }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 24px; }}
  .section {{ margin-bottom: 48px; }}
  .section h2 {{ font-size: 1.5em; border-bottom: 2px solid var(--accent); padding-bottom: 8px; display: inline-block; margin-bottom: 16px; }}
  .narrative {{ color: var(--muted); margin: 12px 0 24px; max-width: 950px; line-height: 1.7; }}
  .narrative strong {{ color: var(--text); }}
  .chart-container {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
  .chart-container img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; background: var(--card); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}
  th {{ background: #1c2333; color: var(--accent); padding: 10px 12px; text-align: right; font-weight: 600; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  .positive {{ color: var(--green); }}
  .negative {{ color: var(--red); }}
  .callout {{ background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.2); border-left: 4px solid var(--accent); border-radius: 8px; padding: 16px 20px; margin: 16px 0; color: var(--muted); }}
  .callout strong {{ color: var(--text); }}
  .footer {{ text-align: center; padding: 32px; color: var(--muted); font-size: 0.85em; border-top: 1px solid var(--border); margin-top: 48px; }}
</style>
</head>
<body>
<div class="hero">
  <h1>Strategy Diversification — Comparison Report</h1>
  <div class="subtitle">
    Comparing new strategies (IV Regime Condor, Weekly Iron Condor, Tail Hedged)
    against the v3 cross-ticker baseline on NDX.
  </div>
  <div class="date">Generated: {today_str}</div>
</div>
<div class="container">

<div class="callout">
  <strong>Note:</strong> The v3 baseline uses 3 tickers (NDX, SPX, RUT) with cross-ticker selection
  and volume-adjusted sizing ($500K unified budget). New strategies run single-ticker NDX with
  $100K daily budget, so raw P&amp;L is not directly comparable. Focus on <strong>win rate, ROI,
  Sharpe, and profit factor</strong> for apples-to-apples comparison.
</div>

<div class="section">
  <h2>Metrics Comparison</h2>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>Key Metrics</h2>
  <div class="chart-container"><img src="charts/metrics_comparison.png" alt="Metrics Comparison"></div>
</div>

<div class="section">
  <h2>Cumulative P&amp;L</h2>
  <div class="narrative">
    Cumulative P&amp;L by trade number (not time-aligned). Different strategies enter at different
    frequencies, so compare the trajectory shape and drawdown depth rather than absolute values.
  </div>
  <div class="chart-container"><img src="charts/cumulative_pnl_comparison.png" alt="Cumulative P&L"></div>
</div>

</div>
<div class="footer">Strategy Comparison &mdash; Generated {today_str}</div>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"report_strategy_comparison_{today_str}.html"
    with open(report_path, "w") as f:
        f.write(html)

    index_path = OUTPUT_DIR / "index.html"
    if index_path.exists() or index_path.is_symlink():
        index_path.unlink()
    index_path.symlink_to(report_path.name)

    print(f"\nHTML report saved to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''
Strategy Comparison — New Strategies vs v3 Baseline.

Runs each new strategy on NDX and compares against the v3 cross-ticker
portfolio baseline. Produces a side-by-side metrics table, charts, and
an HTML report.
        ''',
        epilog='''
Examples:
  %(prog)s
      Full run: baseline + all 3 new strategies + comparison

  %(prog)s --analyze
      Skip backtests, re-analyze existing results

  %(prog)s --strategies iv_regime_condor tail_hedged
      Run only specific new strategies

  %(prog)s --help
      Show this help message
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Skip backtests, only re-analyze")
    parser.add_argument("--strategies", nargs="+",
                        choices=["baseline_ndx", "iv_regime_condor",
                                 "weekly_iron_condor", "tail_hedged"],
                        help="Run only these strategies (default: all)")

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    strategy_names = args.strategies or list(STRATEGIES.keys())

    # Phase 1: Run backtests
    all_results = {}
    if not args.analyze:
        configs = []
        for name in strategy_names:
            if name == "baseline_ndx":
                configs.append((name, _build_baseline_config()))
            else:
                configs.append((name, STRATEGIES[name]["config_path"]))

        print(f"Running {len(configs)} strategy backtests in parallel...")
        num_workers = min(len(configs), cpu_count())

        with Pool(processes=num_workers) as pool:
            results = pool.map(run_single_strategy, configs)

        for name, result in results:
            all_results[name] = result
            error = result.get("error")
            if error:
                print(f"  [{name}] ERROR: {error}")
            else:
                n_trades = len(result.get("trades", []))
                print(f"  [{name}] {n_trades} trades")
    else:
        # Load from CSVs if available
        for name in strategy_names:
            output_dir = OUTPUT_DIR / name
            csv_path = output_dir / "trades.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                all_results[name] = {"trades": df.to_dict("records")}
                print(f"  [{name}] Loaded {len(df)} trades from CSV")
            else:
                print(f"  [{name}] No cached results, skipping")

    # Phase 2: Compute metrics
    all_metrics = {}

    # Load v3 baseline
    print("\nLoading v3 baseline metrics...")
    all_metrics["v3_baseline"] = load_v3_baseline()

    for name in strategy_names:
        if name in all_results:
            trades = all_results[name].get("trades", [])
            all_metrics[name] = compute_metrics(trades)

    # Phase 3: Print comparison
    print_comparison(all_metrics)

    # Phase 4: Charts
    print("\nGenerating comparison charts...")
    generate_comparison_charts(all_results, all_metrics)

    # Phase 5: HTML report
    report_path = generate_html_report(all_metrics, all_results)

    # Open report
    if report_path and report_path.exists():
        os.system(f'open "{report_path}"')

    print("\nDone!")


if __name__ == "__main__":
    main()
