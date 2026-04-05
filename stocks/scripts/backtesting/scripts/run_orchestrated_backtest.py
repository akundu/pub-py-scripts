"""CLI entry point for orchestrated multi-algorithm backtesting.

Runs multiple strategy instances in parallel, then applies trigger-based
selection to cherry-pick the best algo per trading slot. Supports both
daily and 5-minute interval modes with position tracking and exit rules.

Examples:
  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx.yaml

  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx.yaml --dry-run

  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx_interval.yaml --interval

  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx.yaml \\
      --instance v3:NDX:p95_dte0

  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx.yaml \\
      --group index_0dte

  python run_orchestrated_backtest.py \\
      --config scripts/backtesting/configs/orchestration_ndx.yaml \\
      --compare-v3 --baselines
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.backtesting.orchestration.manifest import OrchestrationManifest
from scripts.backtesting.orchestration.engine import OrchestratorEngine


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("orchestration")


def load_v3_baseline(results_dir: str = "results/tiered_portfolio_v3") -> dict:
    """Load v3 cross-ticker baseline results for comparison."""
    from scripts.backtesting.orchestration.baseline import load_v3_baseline as _load
    return _load(results_dir)


def print_comparison(orchestrated: dict, v3: dict, per_instance: dict,
                     baselines: dict, logger):
    """Print side-by-side comparison table."""
    orch_metrics = orchestrated.get("combined_metrics", {})
    v3_metrics = v3.get("metrics", {})

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Orchestrated vs Baselines vs Individual Algos")
    logger.info("=" * 80)

    header = f"{'System':<30} {'Trades':>7} {'WR%':>7} {'ROI%':>8} {'Sharpe':>8} {'Net P&L':>12} {'MaxDD':>10} {'PF':>7}"
    logger.info(header)
    logger.info("-" * 80)

    def _row(name, m, trades=None):
        t = trades if trades is not None else m.get("total_trades", 0)
        return (
            f"{name:<30} {t:>7} "
            f"{m.get('win_rate', 0):>6.1f}% "
            f"{m.get('roi', 0):>7.1f}% "
            f"{m.get('sharpe', 0):>8.2f} "
            f"${m.get('net_pnl', 0):>10,.0f} "
            f"${m.get('max_drawdown', 0):>8,.0f} "
            f"{m.get('profit_factor', 0):>6.2f}"
        )

    logger.info(_row("ORCHESTRATED (combined)", orch_metrics,
                      orchestrated.get("total_accepted", 0)))

    if v3_metrics:
        logger.info(_row("v3 Cross-Ticker Baseline", v3_metrics,
                          v3.get("total_trades", 0)))

    # Baselines
    for name, data in baselines.items():
        if "error" not in data:
            logger.info(_row(f"Baseline: {name}", data.get("metrics", {}),
                              data.get("total_trades", 0)))

    logger.info("-" * 80)

    # Per-instance
    for iid, data in per_instance.items():
        m = data.get("metrics", {})
        logger.info(_row(iid, m, data.get("total_trades", 0)))


def generate_html_report(summary: dict, v3_baseline: dict, baselines: dict,
                         output_dir: str):
    """Generate HTML report for orchestrated results."""
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    today = date.today().isoformat()
    report_path = os.path.join(output_dir, f"report_orchestrated_{today}.html")

    orch_metrics = summary.get("combined_metrics", {})
    attribution = summary.get("per_algo_attribution", {})
    overlap = summary.get("overlap_analysis", {})
    per_instance = summary.get("per_instance_results", {})
    interval_analysis = summary.get("interval_analysis", {})
    v3_metrics = v3_baseline.get("metrics", {})

    # Build comparison rows
    rows = []
    rows.append({"System": "ORCHESTRATED", **orch_metrics,
                 "trades": summary.get("total_accepted", 0)})
    if v3_metrics:
        rows.append({"System": "v3 Baseline", **v3_metrics,
                     "trades": v3_baseline.get("total_trades", 0)})
    for name, data in baselines.items():
        if "error" not in data:
            rows.append({"System": f"Baseline: {name}", **data.get("metrics", {}),
                         "trades": data.get("total_trades", 0)})
    for iid, data in per_instance.items():
        rows.append({"System": iid, **data.get("metrics", {}),
                     "trades": data.get("total_trades", 0)})

    comparison_html = ""
    for r in rows:
        is_orch = r["System"] == "ORCHESTRATED"
        style = ' style="background:#1a2332;font-weight:bold"' if is_orch else ''
        comparison_html += f"""<tr{style}>
            <td>{r['System']}</td>
            <td>{r.get('trades', 0)}</td>
            <td>{r.get('win_rate', 0):.1f}%</td>
            <td>{r.get('roi', 0):.1f}%</td>
            <td>{r.get('sharpe', 0):.2f}</td>
            <td>${r.get('net_pnl', 0):,.0f}</td>
            <td>${r.get('max_drawdown', 0):,.0f}</td>
            <td>{r.get('profit_factor', 0):.2f}</td>
        </tr>"""

    # Attribution table
    attr_html = ""
    for iid, data in attribution.items():
        m = data.get("metrics", {})
        attr_html += f"""<tr>
            <td>{iid}</td>
            <td>{data.get('algo_name', '')}</td>
            <td>{data.get('ticker', '')}</td>
            <td>{data['trades']}</td>
            <td>{m.get('win_rate', 0):.1f}%</td>
            <td>${m.get('net_pnl', 0):,.0f}</td>
        </tr>"""

    # Interval analysis section
    interval_section = ""
    if interval_analysis:
        by_hour = interval_analysis.get("trades_by_hour", {})
        exit_reasons = interval_analysis.get("exit_reasons", {})
        hour_rows = "".join(
            f"<tr><td>{h}:00 UTC</td><td>{c}</td></tr>"
            for h, c in sorted(by_hour.items())
        )
        exit_rows = "".join(
            f"<tr><td>{r}</td><td>{c}</td></tr>"
            for r, c in exit_reasons.items()
        )
        interval_section = f"""
<div class="section">
<h2>Interval Analysis</h2>
<p>Trade distribution by hour of day and exit reason breakdown.</p>
<div style="display:flex;gap:24px;flex-wrap:wrap">
<div style="flex:1;min-width:200px">
<h3>Trades by Hour (UTC)</h3>
<table><tr><th>Hour</th><th>Trades</th></tr>{hour_rows}</table>
</div>
<div style="flex:1;min-width:200px">
<h3>Exit Reasons</h3>
<table><tr><th>Reason</th><th>Count</th></tr>{exit_rows}</table>
<p>Total exits: {interval_analysis.get('total_exit_events', 0)}</p>
</div>
</div>
</div>"""

    phase2_mode = summary.get("per_instance_results", {})
    mode_label = "Interval Mode" if interval_analysis else "Daily Mode"

    html = f"""<!DOCTYPE html>
<html><head>
<title>Orchestrated Backtest Report - {today}</title>
<style>
body {{ background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; }}
.hero {{ background: linear-gradient(135deg, #161b22, #0d1117); border: 1px solid #30363d; border-radius: 12px; padding: 30px; margin-bottom: 24px; text-align: center; }}
.hero h1 {{ color: #58a6ff; margin: 0; font-size: 28px; }}
.hero .subtitle {{ color: #8b949e; margin-top: 8px; }}
.kpi-strip {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
.kpi {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; flex: 1; min-width: 120px; text-align: center; }}
.kpi .value {{ font-size: 24px; font-weight: bold; color: #58a6ff; }}
.kpi .label {{ color: #8b949e; font-size: 12px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: #161b22; border-radius: 8px; overflow: hidden; }}
th {{ background: #21262d; color: #8b949e; padding: 12px; text-align: left; font-size: 12px; text-transform: uppercase; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; }}
h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
h3 {{ color: #c9d1d9; }}
.section {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
</style></head><body>

<div class="hero">
    <h1>{summary.get('total_accepted', 0)} Trades Orchestrated</h1>
    <div class="subtitle">Multi-Algorithm Orchestration Report ({mode_label}) &mdash; {today}</div>
</div>

<div class="kpi-strip">
    <div class="kpi"><div class="value">{orch_metrics.get('win_rate', 0):.1f}%</div><div class="label">Win Rate</div></div>
    <div class="kpi"><div class="value">{orch_metrics.get('roi', 0):.1f}%</div><div class="label">ROI</div></div>
    <div class="kpi"><div class="value">{orch_metrics.get('sharpe', 0):.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">${orch_metrics.get('net_pnl', 0):,.0f}</div><div class="label">Net P&L</div></div>
    <div class="kpi"><div class="value">${orch_metrics.get('max_drawdown', 0):,.0f}</div><div class="label">Max Drawdown</div></div>
    <div class="kpi"><div class="value">{orch_metrics.get('profit_factor', 0):.2f}</div><div class="label">Profit Factor</div></div>
</div>

<div class="section">
<h2>Comparison: Orchestrated vs Baselines vs Individual</h2>
<table>
<tr><th>System</th><th>Trades</th><th>Win Rate</th><th>ROI</th><th>Sharpe</th><th>Net P&L</th><th>Max DD</th><th>PF</th></tr>
{comparison_html}
</table>
</div>

<div class="section">
<h2>Per-Algo Attribution</h2>
<p>Which algo contributed which trades to the combined portfolio.</p>
<table>
<tr><th>Instance</th><th>Algo</th><th>Ticker</th><th>Trades</th><th>Win Rate</th><th>Net P&L</th></tr>
{attr_html}
</table>
</div>

{interval_section}

<div class="section">
<h2>Overlap Analysis</h2>
<p>How often multiple algos competed for the same slot.</p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Slots</td><td>{overlap.get('total_slots', 0)}</td></tr>
<tr><td>Contested Slots</td><td>{overlap.get('contested_slots', 0)}</td></tr>
<tr><td>Uncontested Slots</td><td>{overlap.get('uncontested_slots', 0)}</td></tr>
<tr><td>Contest Rate</td><td>{overlap.get('contest_rate', 0):.0%}</td></tr>
</table>
</div>

<div class="section" style="text-align:center;color:#8b949e">
Generated with <a href="https://claude.com/claude-code" style="color:#58a6ff">Claude Code</a>
</div>

</body></html>"""

    with open(report_path, "w") as f:
        f.write(html)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="""
Multi-Algorithm Orchestration Backtester

Runs multiple strategy instances in parallel, then applies trigger-based
selection to cherry-pick the best algo per trading slot. Supports:
- Daily mode: one selection per trading date (default)
- Interval mode: 5-minute interval replay with position tracking, exit rules,
  and decaying budget allocation

Supports recursive sub-orchestrators, multi-instance algos, VIX regime triggers,
and baseline comparisons (standalone, equal-weight, v3 cross-ticker).
        """,
        epilog="""
Examples:
  %(prog)s --config scripts/backtesting/configs/orchestration_ndx.yaml
      Run the full orchestrated backtest (daily mode)

  %(prog)s --config scripts/backtesting/configs/orchestration_ndx_interval.yaml
      Run in interval mode (phase2_mode: interval in YAML)

  %(prog)s --config scripts/backtesting/configs/orchestration_ndx.yaml --interval
      Override to interval mode via CLI flag

  %(prog)s --config scripts/backtesting/configs/orchestration_ndx.yaml --dry-run
      Preview instances and triggers without running

  %(prog)s --config scripts/backtesting/configs/orchestration_ndx.yaml --baselines
      Include standalone + equal-weight baseline comparisons

  %(prog)s --config scripts/backtesting/configs/orchestration_ndx.yaml --compare-v3
      Compare orchestrated results against v3 baseline
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", required=True,
        help="Path to orchestration YAML manifest"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without executing backtests"
    )
    parser.add_argument(
        "--interval", action="store_true",
        help="Force interval mode (override phase2_mode in YAML)"
    )
    parser.add_argument(
        "--instance",
        help="Run only this instance ID (e.g., v3:NDX:p95_dte0)"
    )
    parser.add_argument(
        "--group",
        help="Run only instances in this group (e.g., index_0dte)"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Include standalone + equal-weight baseline comparisons"
    )
    parser.add_argument(
        "--compare-v3", action="store_true",
        help="Compare orchestrated results against v3 cross-ticker baseline"
    )
    parser.add_argument(
        "--v3-results-dir", default="results/tiered_portfolio_v3",
        help="Directory containing v3 baseline results (default: results/tiered_portfolio_v3)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory"
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    # Load manifest
    manifest = OrchestrationManifest.load(args.config)
    if args.output_dir:
        manifest.config.output_dir = args.output_dir
    if args.interval:
        manifest.config.phase2_mode = "interval"

    # Run orchestrator
    engine = OrchestratorEngine(manifest, logger)
    summary = engine.run(
        dry_run=args.dry_run,
        filter_instance=args.instance,
        filter_group=args.group,
    )

    if args.dry_run:
        return

    # Save results
    engine.save_results(summary)

    # Compute baselines
    baselines = {}
    if args.baselines:
        from scripts.backtesting.orchestration.baseline import (
            compute_standalone_baseline,
            compute_equal_weight_baseline,
        )

        # Standalone baseline
        per_inst = summary.get("per_instance_results", {})
        standalone = compute_standalone_baseline(per_inst)
        # Aggregate standalone into a single baseline
        all_standalone_trades = []
        for iid, data in per_inst.items():
            # We need the actual trades, not just metrics
            pass
        baselines["standalone"] = {
            "metrics": {},
            "total_trades": sum(d.get("total_trades", 0) for d in per_inst.values()),
            "source": "standalone",
            "per_instance": standalone,
        }

        # Equal-weight baseline (using accepted + rejected trades)
        all_trades = engine.collector.accepted_trades + engine.collector.rejected_trades
        if all_trades:
            baselines["equal_weight"] = compute_equal_weight_baseline(
                all_trades, manifest.config.daily_budget
            )

    # v3 comparison
    v3_baseline = {}
    if args.compare_v3:
        v3_baseline = load_v3_baseline(args.v3_results_dir)
        if "error" in v3_baseline:
            logger.warning(f"v3 baseline: {v3_baseline['error']}")

    # Print comparison
    if v3_baseline or baselines:
        print_comparison(
            summary, v3_baseline,
            summary.get("per_instance_results", {}),
            baselines,
            logger,
        )

    # Generate HTML report
    report_path = generate_html_report(
        summary, v3_baseline, baselines, manifest.config.output_dir
    )
    logger.info(f"\nHTML report: {report_path}")


if __name__ == "__main__":
    main()
