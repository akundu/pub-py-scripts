#!/usr/bin/env python3
"""Analyze comparison backtest results."""

import pandas as pd

df = pd.read_csv('comparison_full_10days.csv')

print('='*80)
print('FULL 10-DAY BACKTEST RESULTS')
print('='*80)

# Overall metrics
static = df[df['approach'] == 'STATIC']
dynamic = df[df['approach'] == 'DYNAMIC']

print(f'\nTotal Predictions: {len(static)} per approach')
print(f'\nSTATIC:')
print(f'  Hit Rate: {static["in_range"].mean()*100:.1f}%')
print(f'  Avg Error: ${static["error"].abs().mean():.2f} ({static["error_pct"].mean():.2f}%)')
print(f'  Avg Range Width: ${static["range_width"].mean():.2f} ({static["range_width_pct"].mean():.2f}%)')
print(f'  Failures: {(~static["in_range"]).sum()} out of {len(static)}')

print(f'\nDYNAMIC (with today\'s data):')
print(f'  Hit Rate: {dynamic["in_range"].mean()*100:.1f}%')
print(f'  Avg Error: ${dynamic["error"].abs().mean():.2f} ({dynamic["error_pct"].mean():.2f}%)')
print(f'  Avg Range Width: ${dynamic["range_width"].mean():.2f} ({dynamic["range_width_pct"].mean():.2f}%)')
print(f'  Failures: {(~dynamic["in_range"]).sum()} out of {len(dynamic)}')

print('\n' + '='*80)
print('HIT RATE BY HOUR (Critical for 0DTE)')
print('='*80)
print(f'\n{"Hour":<10} {"STATIC":<15} {"DYNAMIC":<15} {"Winner":<10}')
print('-'*50)

for hour in sorted(static['hour'].unique()):
    s_hour = static[static['hour'] == hour]
    d_hour = dynamic[dynamic['hour'] == hour]
    s_hit = s_hour['in_range'].mean() * 100
    d_hit = d_hour['in_range'].mean() * 100
    winner = 'STATIC' if s_hit > d_hit else ('DYNAMIC' if d_hit > s_hit else 'TIE')
    print(f'{hour:<10.1f} {s_hit:<15.1f} {d_hit:<15.1f} {winner:<10}')

print('\n' + '='*80)
print('RANGE WIDTH BY HOUR (Tighter = Less Capital Risk)')
print('='*80)
print(f'\n{"Hour":<10} {"STATIC %":<15} {"DYNAMIC %":<15} {"Savings":<10}')
print('-'*50)

for hour in sorted(static['hour'].unique()):
    s_hour = static[static['hour'] == hour]
    d_hour = dynamic[dynamic['hour'] == hour]
    s_width = s_hour['range_width_pct'].mean()
    d_width = d_hour['range_width_pct'].mean()
    savings = s_width - d_width
    winner = 'DYNAMIC' if d_width < s_width else 'STATIC'
    print(f'{hour:<10.1f} {s_width:<15.2f} {d_width:<15.2f} {savings:+.2f}%')

# Failure analysis
static_failures = static[~static['in_range']]
dynamic_failures = dynamic[~dynamic['in_range']]

print('\n' + '='*80)
print('FAILURE ANALYSIS (When Price Breached Range)')
print('='*80)

print(f'\nSTATIC Failures: {len(static_failures)} out of {len(static)} ({len(static_failures)/len(static)*100:.1f}%)')
print(f'DYNAMIC Failures: {len(dynamic_failures)} out of {len(dynamic)} ({len(dynamic_failures)/len(dynamic)*100:.1f}%)')

if len(static_failures) > 0:
    print(f'\nSTATIC Breach Details:')
    for _, row in static_failures.iterrows():
        breach = row['error']
        print(f'  {row["date"]} at {row["hour"]:.1f}: ${abs(breach):.2f} breach ({abs(row["error_pct"]):.2f}%)')

if len(dynamic_failures) > 0:
    print(f'\nDYNAMIC Breach Details:')
    for _, row in dynamic_failures.iterrows():
        breach = row['error']
        print(f'  {row["date"]} at {row["hour"]:.1f}: ${abs(breach):.2f} breach ({abs(row["error_pct"]):.2f}%)')

print('\n' + '='*80)
print('RECOMMENDATION FOR 0DTE TRADING')
print('='*80)

static_hit = static["in_range"].mean() * 100
dynamic_hit = dynamic["in_range"].mean() * 100

print(f'\nFor 0DTE credit spreads where breach = total loss:')
print(f'  STATIC Hit Rate:  {static_hit:.1f}% ({100-static_hit:.1f}% failure rate)')
print(f'  DYNAMIC Hit Rate: {dynamic_hit:.1f}% ({100-dynamic_hit:.1f}% failure rate)')

if static_hit > dynamic_hit:
    diff = static_hit - dynamic_hit
    print(f'\n✓ STATIC is BETTER by {diff:.1f} percentage points')
    print(f'  Fewer breaches = fewer catastrophic losses')
elif dynamic_hit > static_hit:
    diff = dynamic_hit - static_hit
    print(f'\n✓ DYNAMIC is BETTER by {diff:.1f} percentage points')
    print(f'  Better adapts to intraday patterns')
else:
    print(f'\n✓ TIE - Both approaches have same hit rate')

# Capital efficiency
static_width = static['range_width_pct'].mean()
dynamic_width = dynamic['range_width_pct'].mean()
width_savings = static_width - dynamic_width

if width_savings > 0:
    print(f'\n✓ DYNAMIC has tighter ranges by {width_savings:.2f}%')
    print(f'  Less capital at risk per trade')
    print(f'  Example: On $25,000 index, saves ${25000 * width_savings/100:.2f} in range width')
else:
    print(f'\n✓ STATIC has tighter ranges by {-width_savings:.2f}%')

print('\n' + '='*80)
print('FINAL VERDICT')
print('='*80)

if static_hit >= 98 and dynamic_hit >= 98:
    print('\nBoth approaches are viable for 0DTE (>98% hit rate)')
    if width_savings > 0:
        print('Recommendation: DYNAMIC for better capital efficiency')
    else:
        print('Recommendation: STATIC for simplicity')
elif static_hit >= 98:
    print('\nRecommendation: STATIC')
    print('Reason: Meets 98% threshold for 0DTE, DYNAMIC does not')
elif dynamic_hit >= 98:
    print('\nRecommendation: DYNAMIC')
    print('Reason: Meets 98% threshold for 0DTE, STATIC does not')
else:
    print(f'\n⚠️  WARNING: Neither approach meets 98% threshold!')
    print(f'Current hit rates insufficient for safe 0DTE trading')
    if static_hit > dynamic_hit:
        print(f'Recommendation: STATIC (better of two, but still risky)')
    else:
        print(f'Recommendation: DYNAMIC (better of two, but still risky)')
    print('\nConsider:')
    print('  - Wider ranges (lower band_width_scale)')
    print('  - Different time windows (avoid early morning)')
    print('  - More training data')
