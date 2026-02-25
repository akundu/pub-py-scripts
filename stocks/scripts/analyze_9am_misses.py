#!/usr/bin/env python3
"""
Analyze 9:30 AM prediction misses to understand root cause.

The opening gap model didn't improve 9:30 AM hit rate (stayed at 88.3%).
This script analyzes the misses to identify patterns and root causes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

def analyze_930am_misses():
    """Analyze misses at 9:30 AM to find patterns."""
    print("="*80)
    print("9:30 AM MISS ANALYSIS")
    print("="*80)
    print()

    # Load detailed backtest results
    results_file = "results/with_gap_model/0dte_detailed_NDX.csv"
    print(f"Loading: {results_file}")
    df = pd.read_csv(results_file)

    # Filter for 9:30 AM predictions (combined model)
    df_930 = df[(df['model'] == 'combined') & (df['hour'] == '9:30')].copy()

    if df_930.empty:
        print("ERROR: No 9:30 AM data found!")
        return

    print(f"Total 9:30 AM predictions: {len(df_930)}")
    print()

    # Identify misses (actual close outside P97 bands)
    df_930['p97_miss'] = (
        (df_930['actual_close'] < df_930['p97_lo']) |
        (df_930['actual_close'] > df_930['p97_hi'])
    )

    n_misses = df_930['p97_miss'].sum()
    miss_rate = n_misses / len(df_930) * 100

    print(f"P97 Misses: {n_misses} / {len(df_930)} ({miss_rate:.1f}%)")
    print(f"P97 Hit Rate: {100 - miss_rate:.1f}%")
    print()

    # Analyze miss characteristics
    misses = df_930[df_930['p97_miss']].copy()

    if misses.empty:
        print("No misses found!")
        return

    # Calculate miss distance (how far outside bands)
    misses['miss_distance_pct'] = misses.apply(lambda row:
        min(
            abs(row['actual_close'] - row['p97_lo']) / row['current_price'] * 100,
            abs(row['actual_close'] - row['p97_hi']) / row['current_price'] * 100
        ), axis=1
    )

    # Calculate midpoint error
    misses['midpoint'] = (misses['p97_lo'] + misses['p97_hi']) / 2
    misses['midpoint_error_pct'] = (misses['actual_close'] - misses['midpoint']) / misses['current_price'] * 100

    # Direction of miss (above or below)
    misses['miss_direction'] = misses.apply(lambda row:
        'above' if row['actual_close'] > row['p97_hi'] else 'below', axis=1
    )

    print("Miss Characteristics:")
    print("-" * 80)
    print(f"Average miss distance: {misses['miss_distance_pct'].mean():.2f}%")
    print(f"Median miss distance: {misses['miss_distance_pct'].median():.2f}%")
    print(f"Max miss distance: {misses['miss_distance_pct'].max():.2f}%")
    print()

    print(f"Miss direction:")
    print(f"  Above bands: {(misses['miss_direction'] == 'above').sum()} ({(misses['miss_direction'] == 'above').sum() / len(misses) * 100:.1f}%)")
    print(f"  Below bands: {(misses['miss_direction'] == 'below').sum()} ({(misses['miss_direction'] == 'below').sum() / len(misses) * 100:.1f}%)")
    print()

    # Analyze midpoint errors
    print("Midpoint Error Analysis:")
    print("-" * 80)
    print(f"Average midpoint error: {misses['midpoint_error_pct'].mean():+.2f}%")
    print(f"Median midpoint error: {misses['midpoint_error_pct'].median():+.2f}%")
    print(f"Midpoint error std dev: {misses['midpoint_error_pct'].std():.2f}%")
    print()

    # Check if there's a systematic bias
    if abs(misses['midpoint_error_pct'].mean()) > 0.2:
        bias_direction = "upward" if misses['midpoint_error_pct'].mean() > 0 else "downward"
        print(f"⚠️  WARNING: Systematic {bias_direction} bias detected!")
        print(f"   Midpoint is consistently {abs(misses['midpoint_error_pct'].mean()):.2f}% {bias_direction}")
        print()

    # Analyze gap characteristics for misses
    if 'gap_pct' in misses.columns:
        print("Gap Analysis for Misses:")
        print("-" * 80)
        print(f"Average gap: {misses['gap_pct'].mean():+.2f}%")
        print(f"Gap range: {misses['gap_pct'].min():+.2f}% to {misses['gap_pct'].max():+.2f}%")

        # Check if misses happen more with certain gap sizes
        large_gap_misses = misses[abs(misses['gap_pct']) > 1.0]
        print(f"Misses with gap > 1%: {len(large_gap_misses)} / {len(misses)} ({len(large_gap_misses)/len(misses)*100:.1f}%)")
        print()

    # Sample of worst misses
    print("Worst 5 Misses:")
    print("-" * 80)
    worst_misses = misses.nlargest(5, 'miss_distance_pct')[
        ['date', 'current_price', 'actual_close', 'p97_lo', 'p97_hi', 'miss_distance_pct', 'midpoint_error_pct']
    ]

    for idx, row in worst_misses.iterrows():
        print(f"\nDate: {row['date']}")
        print(f"  Current: ${row['current_price']:,.0f}")
        print(f"  Actual close: ${row['actual_close']:,.0f}")
        print(f"  P97 band: ${row['p97_lo']:,.0f} - ${row['p97_hi']:,.0f}")
        print(f"  Miss distance: {row['miss_distance_pct']:.2f}%")
        print(f"  Midpoint error: {row['midpoint_error_pct']:+.2f}%")

    print()
    print("="*80)

    # Recommendations based on analysis
    print("\nRECOMMENDATIONS:")
    print("-" * 80)

    if abs(misses['midpoint_error_pct'].mean()) > 0.2:
        print("1. FIX SYSTEMATIC BIAS:")
        bias_dir = "upward" if misses['midpoint_error_pct'].mean() > 0 else "downward"
        print(f"   - Bands are consistently {bias_dir} biased")
        print(f"   - Consider adjusting band midpoint by {abs(misses['midpoint_error_pct'].mean()):.2f}% {bias_dir}")

    if misses['miss_distance_pct'].median() < 0.3:
        print("\n2. SMALL MISS DISTANCES:")
        print("   - Most misses are <0.3% outside bands")
        print("   - A small width increase (5-10%) could catch these")
        print("   - But this suggests calibration issue, not fundamental problem")

    if 'gap_pct' in misses.columns and (misses['gap_pct'].abs() > 1.0).sum() > len(misses) * 0.5:
        print("\n3. LARGE GAP CORRELATION:")
        print("   - Most misses occur with gaps >1%")
        print("   - Gap model may need stronger multipliers for large gaps")

    print()


if __name__ == '__main__':
    analyze_930am_misses()
