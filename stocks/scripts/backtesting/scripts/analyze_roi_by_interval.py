"""Analyze best achievable ROI per 10-minute interval across all trading days.

For each 10-min window of the trading day, finds the best trade that meets
the orchestration constraints and computes ROI percentiles (P50-P100).
Filters for volume reasonableness: spread width > 0, max_loss capped,
credit per contract reasonable (no near-zero-risk outliers).

Usage:
  python -m scripts.backtesting.scripts.analyze_roi_by_interval \
      --tickers SPX NDX RUT

  python -m scripts.backtesting.scripts.analyze_roi_by_interval \
      --tickers SPX NDX RUT --max-risk 50000 --min-credit 0.30 \
      --max-cr 1.0 --results-dir results/orchestrated_adaptive_budget
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_all_trades(results_dir: str, tickers: list) -> pd.DataFrame:
    """Load all Phase 1 trades for the given tickers from per_instance dirs."""
    all_trades = []
    per_instance_dir = os.path.join(results_dir, "per_instance")
    if not os.path.isdir(per_instance_dir):
        print(f"Error: {per_instance_dir} not found")
        sys.exit(1)

    for instance_dir in sorted(os.listdir(per_instance_dir)):
        trades_path = os.path.join(per_instance_dir, instance_dir, "trades.csv")
        if not os.path.exists(trades_path):
            continue
        df = pd.read_csv(trades_path)

        # Infer ticker from instance dir or ticker column
        if "ticker" in df.columns:
            df = df[df["ticker"].isin(tickers)]
        else:
            matched = None
            for t in tickers:
                if t in instance_dir:
                    matched = t
                    break
            if matched is None:
                continue
            df["ticker"] = matched

        if not df.empty:
            df["instance"] = instance_dir
            all_trades.append(df)

    if not all_trades:
        print(f"No trades found for tickers {tickers}")
        sys.exit(1)

    combined = pd.concat(all_trades, ignore_index=True)
    print(f"Loaded {len(combined):,} trades across {combined['ticker'].nunique()} tickers, "
          f"{combined['instance'].nunique()} instances")
    return combined


def apply_filters(trades: pd.DataFrame, max_risk: float, min_credit: float,
                  max_cr: float) -> pd.DataFrame:
    """Apply constraints and volume reasonableness filters."""
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], format="mixed", utc=True)
    trades["trading_date"] = pd.to_datetime(trades["trading_date"])

    # Compute spread width and per-contract risk
    trades["spread_width"] = (trades["short_strike"] - trades["long_strike"]).abs()
    trades["total_risk"] = trades["max_loss"].abs()
    trades["per_contract_risk"] = trades["total_risk"] / trades["num_contracts"].clip(lower=1)

    # ROI = pnl / total_risk
    trades["roi_pct"] = (trades["pnl"] / trades["total_risk"].clip(lower=1)) * 100

    # CR = initial_credit / (spread_width per share)
    # spread_width is in points, credit is per share
    trades["cr_ratio"] = trades["initial_credit"] / (trades["spread_width"].clip(lower=0.01))

    pre = len(trades)

    # --- Constraint filters ---
    # 1. Max risk per transaction
    trades = trades[trades["total_risk"] <= max_risk]
    # 2. Non-zero risk (exclude degenerate trades)
    trades = trades[trades["total_risk"] > 100]  # at least $100 risk
    # 3. Min credit per contract
    trades = trades[trades["initial_credit"] >= min_credit]
    # 4. Spread width > 0 (real spread, not single leg)
    trades = trades[trades["spread_width"] > 0]
    # 5. CR cap — exclude near-zero-risk outliers that inflate ROI
    trades = trades[trades["cr_ratio"] <= max_cr]
    # 6. Num contracts >= 1
    trades = trades[trades["num_contracts"] >= 1]

    print(f"After filters (max_risk=${max_risk:,.0f}, min_credit=${min_credit:.2f}, "
          f"max_cr={max_cr:.1f}): {len(trades):,} / {pre:,} trades")

    # Extract 10-min interval
    trades["hour_utc"] = trades["entry_time"].dt.hour
    trades["minute_utc"] = trades["entry_time"].dt.minute
    trades["interval_10m"] = (
        trades["hour_utc"].astype(str).str.zfill(2) + ":" +
        (trades["minute_utc"] // 10 * 10).astype(str).str.zfill(2)
    )

    return trades


def analyze_ticker(trades: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """For each 10-min interval: best ROI per day, then percentile stats."""
    tt = trades[trades["ticker"] == ticker].copy()
    if tt.empty:
        return pd.DataFrame()

    # Best ROI trade per (date, interval)
    best = (
        tt.sort_values("roi_pct", ascending=False)
        .groupby(["trading_date", "interval_10m"])
        .first()
        .reset_index()
    )

    # Percentile stats per interval
    def pct_agg(x):
        return pd.Series({
            "days": x["trading_date"].nunique(),
            "trades": len(x),
            "p50_roi": x["roi_pct"].quantile(0.50),
            "p75_roi": x["roi_pct"].quantile(0.75),
            "p90_roi": x["roi_pct"].quantile(0.90),
            "p95_roi": x["roi_pct"].quantile(0.95),
            "p100_roi": x["roi_pct"].max(),
            "avg_roi": x["roi_pct"].mean(),
            "avg_pnl": x["pnl"].mean(),
            "total_pnl": x["pnl"].sum(),
            "win_rate": (x["pnl"] > 0).mean() * 100,
            "avg_credit": x["initial_credit"].mean(),
            "avg_risk": x["total_risk"].mean(),
            "median_cr": x["cr_ratio"].median(),
            "avg_width": x["spread_width"].mean(),
            "avg_contracts": x["num_contracts"].mean(),
            "loss_rate": (x["pnl"] < 0).mean() * 100,
            "avg_loss_when_loss": x.loc[x["pnl"] < 0, "pnl"].mean() if (x["pnl"] < 0).any() else 0,
        })

    stats = best.groupby("interval_10m").apply(pct_agg, include_groups=False).reset_index()
    stats["ticker"] = ticker
    stats = stats.sort_values("interval_10m")
    return stats


def print_report(stats: pd.DataFrame, ticker: str):
    """Print formatted interval report with percentiles."""
    if stats.empty:
        print(f"\n  No data for {ticker}")
        return

    # Market hours: 13:30 - 21:00 UTC
    ms = stats[(stats["interval_10m"] >= "13:30") & (stats["interval_10m"] <= "21:00")].copy()
    if ms.empty:
        ms = stats

    print(f"\n{'='*145}")
    print(f"  {ticker} — Best Achievable ROI per 10-min Interval  |  "
          f"{ms['days'].max():.0f} trading days  |  CR-capped, volume-filtered")
    print(f"{'='*145}")
    print(f"  {'Interval':>8s}  {'Days':>4s}  "
          f"{'P50':>7s}  {'P75':>7s}  {'P90':>7s}  {'P95':>7s}  {'P100':>8s}  "
          f"{'WinR%':>5s}  {'LossR%':>6s}  {'MedCR':>6s}  {'AvgWid':>6s}  "
          f"{'AvgPnL':>8s}  {'TotPnL':>11s}")
    print(f"  {'-'*8}  {'-'*4}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  "
          f"{'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*8}  {'-'*11}")

    for _, r in ms.iterrows():
        print(f"  {r['interval_10m']:>8s}  {r['days']:>4.0f}  "
              f"{r['p50_roi']:>6.1f}%  {r['p75_roi']:>6.1f}%  {r['p90_roi']:>6.1f}%  "
              f"{r['p95_roi']:>6.1f}%  {r['p100_roi']:>7.1f}%  "
              f"{r['win_rate']:>4.0f}%  {r['loss_rate']:>5.1f}%  "
              f"{r['median_cr']:>5.2f}  {r['avg_width']:>5.0f}  "
              f"${r['avg_pnl']:>7,.0f}  ${r['total_pnl']:>9,.0f}")

    # Overall percentile distribution (across all intervals)
    all_p50s = ms["p50_roi"]
    print(f"\n  Distribution of interval P50 ROIs:")
    print(f"    Min={all_p50s.min():.1f}%  P25={all_p50s.quantile(0.25):.1f}%  "
          f"Median={all_p50s.median():.1f}%  P75={all_p50s.quantile(0.75):.1f}%  "
          f"Max={all_p50s.max():.1f}%")

    all_p75s = ms["p75_roi"]
    print(f"  Distribution of interval P75 ROIs:")
    print(f"    Min={all_p75s.min():.1f}%  P25={all_p75s.quantile(0.25):.1f}%  "
          f"Median={all_p75s.median():.1f}%  P75={all_p75s.quantile(0.75):.1f}%  "
          f"Max={all_p75s.max():.1f}%")

    # Top 5 / Bottom 5
    top5 = ms.nlargest(5, "p50_roi")
    bot5 = ms.nsmallest(5, "p50_roi")
    print(f"\n  Top 5 intervals (by P50 ROI):")
    for _, r in top5.iterrows():
        print(f"    {r['interval_10m']} — P50={r['p50_roi']:.1f}%, P75={r['p75_roi']:.1f}%, "
              f"P90={r['p90_roi']:.1f}%, {r['days']:.0f} days, total ${r['total_pnl']:,.0f}")
    print(f"  Bottom 5 intervals (by P50 ROI):")
    for _, r in bot5.iterrows():
        print(f"    {r['interval_10m']} — P50={r['p50_roi']:.1f}%, P75={r['p75_roi']:.1f}%, "
              f"P90={r['p90_roi']:.1f}%, {r['days']:.0f} days, total ${r['total_pnl']:,.0f}")


def suggest_multipliers(all_stats: pd.DataFrame):
    """Analyze combined data and suggest ROI-based multiplier thresholds."""
    print(f"\n{'='*145}")
    print(f"  SUGGESTED MULTIPLIER SCHEDULE")
    print(f"{'='*145}")

    for ticker in sorted(all_stats["ticker"].unique()):
        ts = all_stats[all_stats["ticker"] == ticker]
        ms = ts[(ts["interval_10m"] >= "13:30") & (ts["interval_10m"] <= "21:00")]
        if ms.empty:
            continue

        # Use P50 ROI as the "typical best" for each interval
        p50s = ms["p50_roi"].dropna()
        p75s = ms["p75_roi"].dropna()

        # Compute thresholds from the distribution of interval P50s
        t_p25 = p50s.quantile(0.25)
        t_p50 = p50s.quantile(0.50)
        t_p75 = p50s.quantile(0.75)
        t_p90 = p50s.quantile(0.90)

        print(f"\n  {ticker}:")
        print(f"    Interval P50 ROI distribution: "
              f"P25={t_p25:.1f}%, P50={t_p50:.1f}%, P75={t_p75:.1f}%, P90={t_p90:.1f}%")
        print(f"    Interval P75 ROI distribution: "
              f"P25={p75s.quantile(0.25):.1f}%, P50={p75s.median():.1f}%, "
              f"P75={p75s.quantile(0.75):.1f}%, P90={p75s.quantile(0.90):.1f}%")

        # Count intervals in each tier
        n_flat = len(ms[ms["p50_roi"] < t_p25])
        n_good = len(ms[(ms["p50_roi"] >= t_p25) & (ms["p50_roi"] < t_p50)])
        n_strong = len(ms[(ms["p50_roi"] >= t_p50) & (ms["p50_roi"] < t_p75)])
        n_exc = len(ms[(ms["p50_roi"] >= t_p75) & (ms["p50_roi"] < t_p90)])
        n_spike = len(ms[ms["p50_roi"] >= t_p90])

        print(f"\n    Suggested tiers for {ticker} (based on interval P50 ROI):")
        print(f"      {'Tier':<15s}  {'ROI Range':>15s}  {'Multiplier':>10s}  {'Intervals':>10s}  {'Description'}")
        print(f"      {'─'*15}  {'─'*15}  {'─'*10}  {'─'*10}  {'─'*30}")
        print(f"      {'flat':<15s}  {'< %.1f%%' % t_p25:>15s}  {'1.0x':>10s}  {n_flat:>10d}  Below-average window")
        print(f"      {'good':<15s}  {'%.1f-%.1f%%' % (t_p25, t_p50):>15s}  {'1.5x':>10s}  {n_good:>10d}  Median opportunity")
        print(f"      {'strong':<15s}  {'%.1f-%.1f%%' % (t_p50, t_p75):>15s}  {'2.0x':>10s}  {n_strong:>10d}  Above-average, double dip")
        print(f"      {'exceptional':<15s}  {'%.1f-%.1f%%' % (t_p75, t_p90):>15s}  {'3.0x':>10s}  {n_exc:>10d}  Top quartile, triple dip")
        print(f"      {'spike':<15s}  {'> %.1f%%' % t_p90:>15s}  {'4.0x':>10s}  {n_spike:>10d}  Rare premium, max deploy")

    # Cross-ticker summary
    print(f"\n  {'─'*80}")
    print(f"  Cross-ticker summary (ROI thresholds for unified multiplier schedule):")
    all_market = all_stats[
        (all_stats["interval_10m"] >= "13:30") & (all_stats["interval_10m"] <= "21:00")
    ]
    all_p50 = all_market["p50_roi"].dropna()
    g_p25 = all_p50.quantile(0.25)
    g_p50 = all_p50.quantile(0.50)
    g_p75 = all_p50.quantile(0.75)
    g_p90 = all_p50.quantile(0.90)
    print(f"    Global P50-ROI distribution: P25={g_p25:.1f}%, P50={g_p50:.1f}%, "
          f"P75={g_p75:.1f}%, P90={g_p90:.1f}%")
    print(f"\n    UNIFIED MULTIPLIER SCHEDULE:")
    print(f"      ROI < {g_p25:.0f}%   → 1.0x  (flat)")
    print(f"      ROI {g_p25:.0f}-{g_p50:.0f}%  → 1.5x  (good)")
    print(f"      ROI {g_p50:.0f}-{g_p75:.0f}% → 2.0x  (strong)")
    print(f"      ROI {g_p75:.0f}-{g_p90:.0f}% → 3.0x  (exceptional)")
    print(f"      ROI > {g_p90:.0f}%  → 4.0x  (spike)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best achievable ROI per 10-minute interval with percentiles",
        epilog="""
Examples:
  %(prog)s --tickers SPX NDX RUT
  %(prog)s --tickers NDX --max-risk 50000 --min-credit 0.30 --max-cr 0.80
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", default=["SPX", "NDX", "RUT"],
                        help="Tickers to analyze (default: SPX NDX RUT)")
    parser.add_argument("--results-dir", default="results/orchestrated_adaptive_budget",
                        help="Results directory with per_instance/ subdirs")
    parser.add_argument("--max-risk", type=float, default=50000,
                        help="Max risk per transaction (default: 50000)")
    parser.add_argument("--min-credit", type=float, default=0.30,
                        help="Min credit per contract (default: 0.30)")
    parser.add_argument("--max-cr", type=float, default=0.80,
                        help="Max credit/risk ratio to exclude outliers (default: 0.80)")
    parser.add_argument("--output-csv", default=None,
                        help="Save interval stats to CSV")
    args = parser.parse_args()

    trades = load_all_trades(args.results_dir, args.tickers)
    trades = apply_filters(trades, args.max_risk, args.min_credit, args.max_cr)

    all_stats = []
    for ticker in args.tickers:
        stats = analyze_ticker(trades, ticker)
        if not stats.empty:
            print_report(stats, ticker)
            all_stats.append(stats)

    if all_stats:
        combined = pd.concat(all_stats, ignore_index=True)
        suggest_multipliers(combined)

        if args.output_csv:
            combined.to_csv(args.output_csv, index=False)
            print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
