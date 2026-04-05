"""Compare adaptive vs flat (decaying) budget allocation.

Runs the same orchestration config twice — once with decaying mode and once with
adaptive mode — then produces a side-by-side comparison report with metrics,
adaptive budget log CSV, and HTML report with charts.

Examples:
  python -m scripts.backtesting.scripts.compare_adaptive_vs_flat \\
      --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml \\
      --output results/adaptive_vs_flat/

  python -m scripts.backtesting.scripts.compare_adaptive_vs_flat \\
      --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml \\
      --output results/adaptive_vs_flat/ --skip-flat

  python -m scripts.backtesting.scripts.compare_adaptive_vs_flat --help
"""

import argparse
import copy
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.backtesting.orchestration.manifest import OrchestrationManifest
from scripts.backtesting.orchestration.engine import OrchestratorEngine


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("adaptive_comparison")


def run_orchestration(manifest: OrchestrationManifest, logger_: logging.Logger):
    """Run a single orchestration and return results."""
    engine = OrchestratorEngine(manifest, logger_)
    results = engine.run()
    engine.save_results(results)
    return results, engine


def print_comparison_table(flat_metrics: dict, adaptive_metrics: dict, logger_):
    """Print a side-by-side metrics comparison."""
    metrics_keys = [
        ("total_trades", "Total Trades", "d"),
        ("win_rate", "Win Rate (%)", ".1f"),
        ("net_pnl", "Net P&L ($)", ",.0f"),
        ("roi", "ROI (%)", ".1f"),
        ("sharpe", "Sharpe Ratio", ".2f"),
        ("max_drawdown", "Max Drawdown ($)", ",.0f"),
        ("profit_factor", "Profit Factor", ".2f"),
        ("avg_credit", "Avg Credit ($)", ",.2f"),
        ("avg_max_loss", "Avg Max Loss ($)", ",.2f"),
    ]

    logger_.info("\n" + "=" * 70)
    logger_.info("ADAPTIVE vs FLAT BUDGET COMPARISON")
    logger_.info("=" * 70)
    logger_.info(f"{'Metric':<25} {'Flat (Decaying)':>18} {'Adaptive':>18} {'Delta':>12}")
    logger_.info("-" * 70)

    for key, label, fmt in metrics_keys:
        flat_val = flat_metrics.get(key, 0)
        adap_val = adaptive_metrics.get(key, 0)
        if isinstance(flat_val, (int, float)) and isinstance(adap_val, (int, float)):
            delta = adap_val - flat_val
            logger_.info(
                f"{label:<25} {flat_val:>{18}{fmt}} {adap_val:>{18}{fmt}} "
                f"{delta:>+{12}{fmt}}"
            )
        else:
            logger_.info(f"{label:<25} {str(flat_val):>18} {str(adap_val):>18}")

    logger_.info("=" * 70)


def generate_html_report(
    flat_results: dict,
    adaptive_results: dict,
    output_dir: str,
    adaptive_engine: OrchestratorEngine,
):
    """Generate an HTML comparison report with charts."""
    today = date.today().isoformat()
    flat_m = flat_results.get("combined_metrics", {})
    adap_m = adaptive_results.get("combined_metrics", {})

    # Build metrics rows
    metrics_html = ""
    for key, label in [
        ("total_trades", "Total Trades"),
        ("win_rate", "Win Rate (%)"),
        ("net_pnl", "Net P&L ($)"),
        ("roi", "ROI (%)"),
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown", "Max Drawdown ($)"),
        ("profit_factor", "Profit Factor"),
    ]:
        fv = flat_m.get(key, 0)
        av = adap_m.get(key, 0)
        delta = av - fv if isinstance(fv, (int, float)) and isinstance(av, (int, float)) else ""
        color = "#3fb950" if isinstance(delta, (int, float)) and delta > 0 else "#f85149"
        if key in ("max_drawdown",):
            # Lower is better for drawdown
            color = "#3fb950" if isinstance(delta, (int, float)) and delta < 0 else "#f85149"
        delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else ""
        metrics_html += f"""
        <tr>
            <td>{label}</td>
            <td>{fv if isinstance(fv, str) else f'{fv:,.2f}'}</td>
            <td>{av if isinstance(av, str) else f'{av:,.2f}'}</td>
            <td style="color: {color}; font-weight: bold;">{delta_str}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Adaptive vs Flat Budget Comparison — {today}</title>
<style>
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, 'Segoe UI', monospace; margin: 2rem; }}
h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
h2 {{ color: #8b949e; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #161b22; color: #58a6ff; padding: 0.75rem; text-align: left; border-bottom: 2px solid #30363d; }}
td {{ padding: 0.6rem 0.75rem; border-bottom: 1px solid #21262d; }}
tr:hover {{ background: #161b22; }}
.kpi-strip {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
.kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1rem 1.5rem; min-width: 150px; }}
.kpi-label {{ color: #8b949e; font-size: 0.85rem; }}
.kpi-value {{ color: #f0f6fc; font-size: 1.4rem; font-weight: bold; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
.section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
</style>
</head>
<body>
<h1>Adaptive vs Flat Budget Comparison</h1>
<p>Generated: {today}</p>

<div class="kpi-strip">
  <div class="kpi">
    <div class="kpi-label">Flat Win Rate</div>
    <div class="kpi-value">{flat_m.get('win_rate', 0):.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Adaptive Win Rate</div>
    <div class="kpi-value">{adap_m.get('win_rate', 0):.1f}%</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Flat Net P&L</div>
    <div class="kpi-value">${flat_m.get('net_pnl', 0):,.0f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Adaptive Net P&L</div>
    <div class="kpi-value">${adap_m.get('net_pnl', 0):,.0f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Flat Sharpe</div>
    <div class="kpi-value">{flat_m.get('sharpe', 0):.2f}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Adaptive Sharpe</div>
    <div class="kpi-value">{adap_m.get('sharpe', 0):.2f}</div>
  </div>
</div>

<div class="section">
<h2>Side-by-Side Metrics</h2>
<table>
<tr><th>Metric</th><th>Flat (Decaying)</th><th>Adaptive</th><th>Delta</th></tr>
{metrics_html}
</table>
</div>

<div class="section">
<h2>Methodology</h2>
<p>Both runs use the same orchestration config (same algos, triggers, exit rules).
The only difference is the interval budget allocation mode:</p>
<ul>
<li><strong>Flat (Decaying)</strong>: Each interval gets <code>remaining / intervals_left</code></li>
<li><strong>Adaptive</strong>: Five composable mechanisms (reserve floor, opportunity scaling,
momentum boost, time-weight curve, contract scaling) adjust the per-interval budget
based on market conditions and opportunity quality.</li>
</ul>
<p>The adaptive budget log CSV (<code>adaptive_budget_log.csv</code>) contains per-interval
analytics for detailed analysis of how each mechanism contributed.</p>
</div>

</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"report_adaptive_vs_flat_{today}.html")
    with open(report_path, "w") as f:
        f.write(html)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="""
Compare adaptive vs flat (decaying) budget allocation for orchestration.

Runs the same config twice — once with decaying and once with adaptive mode —
then produces side-by-side metrics, budget log CSV, and HTML report.
        """,
        epilog="""
Examples:
  %(prog)s --config scripts/backtesting/configs/orchestration_adaptive_budget.yaml
      Run full comparison with default output dir

  %(prog)s --config configs/orchestration_adaptive_budget.yaml --output results/test/
      Custom output directory

  %(prog)s --config configs/orchestration_adaptive_budget.yaml --skip-flat
      Only run adaptive mode (if flat results already exist)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to orchestration YAML config")
    parser.add_argument("--output", default="results/adaptive_vs_flat/",
                        help="Output directory for results (default: results/adaptive_vs_flat/)")
    parser.add_argument("--skip-flat", action="store_true",
                        help="Skip the flat run (use existing results)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()
    logger_ = setup_logging(args.log_level)

    os.makedirs(args.output, exist_ok=True)

    flat_results = {}
    adaptive_results = {}
    adaptive_engine = None

    # --- Run 1: Flat (decaying) ---
    if not args.skip_flat:
        logger_.info("=" * 60)
        logger_.info("RUN 1: Flat (decaying) budget mode")
        logger_.info("=" * 60)
        manifest = OrchestrationManifest.load(args.config)
        manifest.config.interval_budget_mode = "decaying"
        manifest.config.output_dir = os.path.join(args.output, "flat")
        flat_results, _ = run_orchestration(manifest, logger_)
    else:
        logger_.info("Skipping flat run (--skip-flat)")

    # --- Run 2: Adaptive ---
    logger_.info("\n" + "=" * 60)
    logger_.info("RUN 2: Adaptive budget mode")
    logger_.info("=" * 60)
    manifest = OrchestrationManifest.load(args.config)
    manifest.config.interval_budget_mode = "adaptive"
    manifest.config.output_dir = os.path.join(args.output, "adaptive")
    adaptive_results, adaptive_engine = run_orchestration(manifest, logger_)

    # --- Comparison ---
    flat_m = flat_results.get("combined_metrics", {})
    adap_m = adaptive_results.get("combined_metrics", {})

    if flat_m and adap_m:
        print_comparison_table(flat_m, adap_m, logger_)

    # Save comparison CSV
    if flat_m or adap_m:
        rows = []
        if flat_m:
            rows.append({"mode": "flat_decaying", **flat_m})
        rows.append({"mode": "adaptive", **adap_m})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(args.output, "comparison_metrics.csv"), index=False)

    # Generate HTML report
    if flat_m and adap_m:
        report_path = generate_html_report(
            flat_results, adaptive_results, args.output, adaptive_engine
        )
        logger_.info(f"\nHTML report: {report_path}")

    logger_.info(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
