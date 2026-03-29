"""Generate Roll Cost Report — multi-ticker, with charts and tabbed HTML.

Runs roll_cost_table.py for each ticker, generates charts, and builds a single
tabbed HTML report with playbooks, summary tables, heatmaps, and detailed data.

Usage:
  python scripts/generate_roll_cost_report.py --start 2026-01-01 --end 2026-03-29

  python scripts/generate_roll_cost_report.py --start 2026-01-01 --end 2026-03-29 \\
      --tickers RUT:20 SPX:10 NDX:50 --options-dir ./options_csv_output_full

  python scripts/generate_roll_cost_report.py --start 2025-09-15 --end 2025-11-07 \\
      --tickers SPX:25 --options-dir ./options_csv_output_full_15 \\
      --output-dir ./results/my_roll_analysis
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [("RUT", 20), ("SPX", 10), ("NDX", 50)]
DEFAULT_CHECK_TIMES = [
    "08:30", "09:00", "09:30", "10:00", "10:30",
    "11:00", "11:30", "12:00", "12:30", "12:55",
]
DTES = [1, 2, 3, 5]
ENTRY_PCTS = [100, 75, 50, 25]
TARGET_PCTS = [100, 50, 25, 0]

DTE_COLORS = {1: '#58a6ff', 2: '#3fb950', 3: '#d29922', 5: '#f85149'}
ENTRY_COLORS = {100: '#f85149', 75: '#d29922', 50: '#3fb950', 25: '#58a6ff'}

CHART_STYLE = {
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d', 'font.size': 11,
}


# ── Step 1: Run roll_cost_table.py per ticker ────────────────────────────────

def run_data_generation(tickers, start, end, options_dir, equities_dir, output_dir):
    """Run roll_cost_table.py for each ticker and return CSV paths."""
    script = str(Path(__file__).parent / "roll_cost_table.py")
    csv_paths = {}
    for ticker, width in tickers:
        print(f"\n{'='*60}")
        print(f"  Generating data: {ticker} (width={width})")
        print(f"{'='*60}")
        cmd = [
            sys.executable, script,
            "--ticker", ticker,
            "--start", start, "--end", end,
            "--spread-width", str(width),
            "--options-dir", options_dir,
            "--equities-dir", equities_dir,
            "--output-dir", output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            continue
        # Print summary lines
        for line in result.stdout.split("\n"):
            if any(k in line for k in ["Days analyzed", "Total obs", "Results saved"]):
                print(f"  {line.strip()}")

        csv_name = f"roll_cost_{ticker}_{start}_{end}.csv"
        csv_path = os.path.join(output_dir, csv_name)
        if os.path.exists(csv_path):
            csv_paths[ticker] = csv_path
        else:
            print(f"  WARNING: CSV not found at {csv_path}")
    return csv_paths


# ── Step 2: Generate charts ──────────────────────────────────────────────────

def generate_charts(dfs, tickers_cfg, charts_dir):
    """Generate all charts for each ticker."""
    plt.rcParams.update(CHART_STYLE)

    for ticker, width in tickers_cfg:
        if ticker not in dfs:
            continue
        df = dfs[ticker]
        TIMES = sorted(df['time_pst'].unique())
        t = ticker.lower()

        # Heatmaps 4x4
        for otype in ['call', 'put']:
            fig, axes = plt.subplots(4, 4, figsize=(22, 18), sharey=True)
            fig.suptitle(f'{ticker} ({width}pt) — {otype.upper()}S Heatmaps\n'
                         f'Rows=Entry Breach | Cols=Roll Target',
                         fontsize=16, fontweight='bold', y=0.98)
            for ri, ep in enumerate(ENTRY_PCTS):
                for ci, tp in enumerate(TARGET_PCTS):
                    ax = axes[ri][ci]
                    sub = df[(df['option_type'] == otype) &
                             (df['entry_breach_pct'] == ep) &
                             (df['target_breach_pct'] == tp)]
                    pivot = sub.groupby(['time_pst', 'roll_dte'])['net_roll_cost'] \
                        .mean().unstack().reindex(index=TIMES, columns=DTES)
                    vmax = max(8, np.nanmax(np.abs(pivot.values))) if not pivot.empty else 8
                    ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                              vmin=-vmax, vmax=vmax)
                    ax.set_xticks(range(len(DTES)))
                    ax.set_xticklabels([f'+{d}' for d in DTES], fontsize=8)
                    ax.set_yticks(range(len(TIMES)))
                    ax.set_yticklabels(TIMES if ci == 0 else [], fontsize=8)
                    if ri == 0:
                        ax.set_title('Same' if tp == 100 else
                                     ('ATM' if tp == 0 else f'{tp}%'), fontsize=10)
                    if ci == 0:
                        ax.set_ylabel(f'{ep}%', fontsize=10, fontweight='bold')
                    for i in range(len(TIMES)):
                        for j in range(len(DTES)):
                            v = pivot.values[i, j]
                            if not np.isnan(v):
                                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                                        fontsize=7,
                                        color='white' if abs(v) > 4 else '#c9d1d9')
            fig.tight_layout(rect=[0, 0, 0.96, 0.96])
            fig.savefig(f'{charts_dir}/heatmap_{t}_{otype}s.png',
                        dpi=130, bbox_inches='tight')
            plt.close(fig)

        # Entry breach bars
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, otype in enumerate(['call', 'put']):
            ax = axes[idx]
            x = np.arange(len(DTES))
            w = 0.18
            for i, ep in enumerate(ENTRY_PCTS):
                sub = df[(df['option_type'] == otype) &
                         (df['entry_breach_pct'] == ep) &
                         (df['target_breach_pct'] == 100)]
                means = [sub[sub['roll_dte'] == d]['net_roll_cost'].mean()
                         for d in DTES]
                ax.bar(x + i * w - 1.5 * w, means, w,
                       color=ENTRY_COLORS[ep], label=f'{ep}%', alpha=0.85)
            ax.axhline(0, color='#8b949e', linestyle='--')
            ax.set_xticks(x)
            ax.set_xticklabels([f'DTE+{d}' for d in DTES])
            ax.set_ylabel('$')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(f'{ticker} {otype.upper()}S Same Strikes',
                         fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{charts_dir}/bar_{t}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Line charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        combos = [
            (otype, 100, tp, f'{ticker} {otype.upper()}S 100%→{"Same" if tp==100 else "ATM"}')
            for otype in ['call', 'put'] for tp in [100, 0]
        ]
        for idx, (otype, ep, tp, title) in enumerate(combos):
            ax = axes[idx // 2][idx % 2]
            sub = df[(df['option_type'] == otype) &
                     (df['entry_breach_pct'] == ep) &
                     (df['target_breach_pct'] == tp)]
            for dte in DTES:
                means = sub[sub['roll_dte'] == dte] \
                    .groupby('time_pst')['net_roll_cost'].mean().reindex(TIMES)
                ax.plot(range(len(TIMES)), means.values, marker='o', linewidth=2,
                        color=DTE_COLORS[dte], label=f'DTE+{dte}', markersize=5)
            ax.axhline(0, color='#8b949e', linestyle='--', alpha=0.5)
            ax.set_xticks(range(len(TIMES)))
            ax.set_xticklabels(TIMES, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('$')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{charts_dir}/lines_{t}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Daily time series
        fig, axes = plt.subplots(2, 1, figsize=(18, 10))
        for idx, otype in enumerate(['call', 'put']):
            ax = axes[idx]
            sub = df[(df['option_type'] == otype) &
                     (df['entry_breach_pct'] == 100) &
                     (df['target_breach_pct'] == 100) &
                     (df['time_pst'] == '12:55')]
            for dte in DTES:
                s = sub[sub['roll_dte'] == dte].sort_values('date')
                ax.plot(range(len(s)), s['net_roll_cost'].values,
                        color=DTE_COLORS[dte], label=f'DTE+{dte}',
                        linewidth=1.2, marker='.', markersize=3)
            ax.axhline(0, color='#f85149', linestyle='--', alpha=0.5)
            ax.set_title(f'{ticker} {otype.upper()}S daily@12:55 PST',
                         fontsize=13, fontweight='bold')
            ax.set_ylabel('$')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            dates = s['date'].values
            step = max(1, len(dates) // 12)
            ax.set_xticks(range(0, len(dates), step))
            ax.set_xticklabels(dates[::step], rotation=45, ha='right', fontsize=8)
        fig.tight_layout()
        fig.savefig(f'{charts_dir}/daily_{t}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"  {ticker} charts saved.")


# ── Step 3: Build HTML report ────────────────────────────────────────────────

def _fmt(v, n):
    if v >= 0:
        return f'<span class="cost">${v:.2f}</span> <span class="cnt">({n})</span>'
    return f'<span class="credit">-${abs(v):.2f}</span> <span class="cnt">({n})</span>'


def _ccls(v):
    if v <= -3: return 'sc'
    if v <= -1: return 'c'
    if v <= 1: return 'n'
    if v <= 3: return 'co'
    return 'sco'


def _build_summary_table(df, otype, tp):
    rows = ''
    for ep in ENTRY_PCTS:
        sub = df[(df['option_type'] == otype) &
                 (df['entry_breach_pct'] == ep) &
                 (df['target_breach_pct'] == tp)]
        cells = ''
        for d in DTES:
            s = sub[sub['roll_dte'] == d]['net_roll_cost']
            if len(s):
                cells += f'<td class="{_ccls(s.mean())}">{_fmt(s.mean(), len(s))}</td>'
            else:
                cells += '<td class="na">N/A</td>'
        rows += f'<tr><td class="tc">{ep}%</td>{cells}</tr>\n'
    return rows


def _build_time_table(df, otype, ep, tp, times):
    sub = df[(df['option_type'] == otype) &
             (df['entry_breach_pct'] == ep) &
             (df['target_breach_pct'] == tp)]
    agg = sub.groupby(['time_pst', 'roll_dte'])['net_roll_cost'].agg(['mean', 'count'])
    rows = ''
    for t in times:
        cells = ''
        for d in DTES:
            try:
                r = agg.loc[(t, d)]
                cells += f'<td class="{_ccls(r["mean"])}">{_fmt(r["mean"], int(r["count"]))}</td>'
            except KeyError:
                cells += '<td class="na">N/A</td>'
        rows += f'<tr><td class="tc">{t}</td>{cells}</tr>\n'
    return rows


def _detail_section(df, times):
    h = ''
    for otype in ['call', 'put']:
        color = 'var(--bl)' if otype == 'call' else 'var(--pu)'
        h += f'<h3 style="color:{color};font-size:1.2em;margin-top:20px;">{otype.upper()}S</h3>\n'
        for ep in ENTRY_PCTS:
            h += f'<h4 class="entry-header" style="font-size:1em;">Entry: {ep}%</h4>\n'
            for tp in TARGET_PCTS:
                tl = 'Same Strikes' if tp == 100 else ('ATM' if tp == 0 else f'{tp}%')
                h += f'<p style="color:var(--tm);font-size:0.85em;margin:4px 0;">Target: {tl}</p>\n'
                h += (f'<table class="dt"><thead><tr><th>Time</th>'
                      f'{"".join(f"<th>DTE+{d}</th>" for d in DTES)}'
                      f'</tr></thead><tbody>'
                      f'{_build_time_table(df, otype, ep, tp, times)}'
                      f'</tbody></table>\n')
    return h


def _build_playbook(df, ticker, width, times):
    # Put playbook
    put_rows = ''
    for t in times:
        sub = df[(df['option_type'] == 'put') &
                 (df['entry_breach_pct'] == 100) &
                 (df['target_breach_pct'] == 100) &
                 (df['time_pst'] == t)]
        if sub.empty:
            continue
        best_dte = sub.groupby('roll_dte')['net_roll_cost'].mean().idxmin()
        val = sub[sub['roll_dte'] == best_dte]['net_roll_cost'].mean()
        if val < -1:
            cls = 'sc' if val < -3 else 'c'
            put_rows += (f'<tr><td class="tc">{t}</td><td>DTE+{best_dte}</td>'
                         f'<td class="{cls}"><span class="credit">-${abs(val):.2f}</span></td></tr>\n')

    # Call playbook
    call_good, call_bad = '', ''
    for t in times:
        sub = df[(df['option_type'] == 'call') &
                 (df['entry_breach_pct'] == 100) &
                 (df['target_breach_pct'] == 100) &
                 (df['time_pst'] == t)]
        if sub.empty:
            continue
        best_dte = sub.groupby('roll_dte')['net_roll_cost'].mean().idxmin()
        val = sub[sub['roll_dte'] == best_dte]['net_roll_cost'].mean()
        if val < -1:
            cls = 'sc' if val < -3 else 'c'
            call_good += (f'<tr><td class="tc">{t}</td><td>DTE+{best_dte}</td>'
                          f'<td class="{cls}"><span class="credit">-${abs(val):.2f}</span>'
                          f'</td><td>Roll here</td></tr>\n')
        else:
            cls = 'co' if val > 0 else 'n'
            call_bad += (f'<tr><td class="tc" style="color:var(--tm)">{t}</td>'
                         f'<td>DTE+{best_dte}</td>'
                         f'<td class="{cls}"><span class="{"cost" if val > 0 else "credit"}">'
                         f'{"+" if val > 0 else ""}${abs(val):.2f}</span></td>'
                         f'<td style="color:var(--rd)">Avoid</td></tr>\n')

    # ATM put
    patm = df[(df['option_type'] == 'put') &
              (df['entry_breach_pct'] == 100) &
              (df['target_breach_pct'] == 0)]
    patm_agg = patm.groupby(['time_pst', 'roll_dte'])['net_roll_cost'].mean()
    patm_best = patm_agg.idxmin()
    patm_val = patm_agg[patm_best]

    put_same = df[(df['option_type'] == 'put') &
                  (df['entry_breach_pct'] == 100) &
                  (df['target_breach_pct'] == 100)]['net_roll_cost'].mean()
    call_same = df[(df['option_type'] == 'call') &
                   (df['entry_breach_pct'] == 100) &
                   (df['target_breach_pct'] == 100)]['net_roll_cost'].mean()

    return f'''
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
      <div class="kpi"><div class="v credit">{put_same:+.2f}</div>
        <div class="l">Avg Put Roll (same strikes)</div></div>
      <div class="kpi"><div class="v {"credit" if call_same < 0 else "cost"}">{call_same:+.2f}</div>
        <div class="l">Avg Call Roll (same strikes)</div></div>
    </div>
    <div style="background:var(--bg);border:1px solid var(--pu);border-radius:10px;padding:20px;margin-bottom:18px;">
      <h4 style="color:var(--pu);margin-bottom:10px;">PUT Rolls — Keep Same Strikes</h4>
      <table class="dt"><thead><tr><th>Time (PST)</th><th>Best DTE</th>
        <th>Avg Credit</th></tr></thead><tbody>{put_rows}</tbody></table>
      <p style="color:var(--tm);font-size:0.88em;">ATM exception:
        <b>{patm_best[0]} PST + DTE+{patm_best[1]}</b> =
        <span class="credit">-${abs(patm_val):.2f}</span></p>
    </div>
    <div style="background:var(--bg);border:1px solid var(--bl);border-radius:10px;padding:20px;margin-bottom:18px;">
      <h4 style="color:var(--bl);margin-bottom:10px;">CALL Rolls — DTE+2, Before 11:00 PST</h4>
      <table class="dt"><thead><tr><th>Time</th><th>Best DTE</th><th>Avg</th>
        <th>Action</th></tr></thead><tbody>{call_good}{call_bad}</tbody></table>
    </div>
    <pre style="font-size:0.9em;line-height:1.7;background:var(--bg);padding:14px;border-radius:8px;border:1px solid var(--yl);">
<span style="font-weight:bold;color:var(--yl);">{ticker} Decision Tree ({width}pt spread)</span>
  PUT breached  ->  <span class="credit">DTE+1 or DTE+3, SAME STRIKES</span> (before 11:00 PST)
  CALL breached ->  <span class="credit">DTE+2 ONLY, SAME STRIKES</span> (before 11:00 PST)
  After 11:30 PST -> consider taking the loss on calls</pre>'''


def build_html_report(dfs, tickers_cfg, charts_dir, output_path):
    """Build the tabbed HTML report."""
    all_times = sorted(set().union(*(set(d['time_pst'].unique()) for d in dfs.values())))
    total_obs = sum(len(d) for d in dfs.values())
    all_dates = set()
    for d in dfs.values():
        all_dates.update(d['date'].unique())
    date_min, date_max = min(all_dates), max(all_dates)
    n_days = len(all_dates)

    ticker_labels = ' | '.join(f'{tk} ({w}pt)' for tk, w in tickers_cfg if tk in dfs)

    # Build tabs
    tab_buttons = ''
    tab_contents = ''
    for i, (tk, width) in enumerate(tickers_cfg):
        if tk not in dfs:
            continue
        df = dfs[tk]
        times = sorted(df['time_pst'].unique())
        tl = tk.lower()
        n = df['date'].nunique()
        obs = len(df)
        active = ' active' if i == 0 else ''
        display = 'block' if i == 0 else 'none'

        tab_buttons += (f'<button class="tab-btn{active}" '
                        f'onclick="showTab(\'{tl}\')" '
                        f'id="btn-{tl}">{tk} ({width}pt)</button>\n')

        tab_contents += f'''
        <div class="tab-content" id="tab-{tl}" style="display:{display}">
          <p class="desc">{tk} | {width}-pt spread | {n} days | {obs:,} obs</p>
          <h3 style="color:var(--gn);font-size:1.3em;margin:16px 0 10px;">Roll Playbook</h3>
          {_build_playbook(df, tk, width, times)}
          <h3 style="margin:20px 0 10px;">Summary Tables</h3>
          <h4>CALLS — Same Strikes</h4>
          <table class="sumtbl"><thead><tr><th>Entry</th>{"".join(f"<th>DTE+{d}</th>" for d in DTES)}</tr></thead>
          <tbody>{_build_summary_table(df, 'call', 100)}</tbody></table>
          <h4>PUTS — Same Strikes</h4>
          <table class="sumtbl"><thead><tr><th>Entry</th>{"".join(f"<th>DTE+{d}</th>" for d in DTES)}</tr></thead>
          <tbody>{_build_summary_table(df, 'put', 100)}</tbody></table>
          <h4>CALLS — ATM</h4>
          <table class="sumtbl"><thead><tr><th>Entry</th>{"".join(f"<th>DTE+{d}</th>" for d in DTES)}</tr></thead>
          <tbody>{_build_summary_table(df, 'call', 0)}</tbody></table>
          <h4>PUTS — ATM</h4>
          <table class="sumtbl"><thead><tr><th>Entry</th>{"".join(f"<th>DTE+{d}</th>" for d in DTES)}</tr></thead>
          <tbody>{_build_summary_table(df, 'put', 0)}</tbody></table>
          <h3 style="margin:20px 0 10px;">Charts</h3>
          <div class="cg"><div class="cc"><img src="charts/bar_{tl}.png"></div></div>
          <div class="cg"><div class="cc"><img src="charts/lines_{tl}.png"></div></div>
          <div class="cg"><div class="cc"><img src="charts/daily_{tl}.png"></div></div>
          <h3 style="margin:20px 0 10px;">Heatmaps</h3>
          <div class="cg"><div class="cc"><img src="charts/heatmap_{tl}_calls.png"></div></div>
          <div class="cg"><div class="cc"><img src="charts/heatmap_{tl}_puts.png"></div></div>
          <button class="toggle" onclick="toggle('detail-{tl}')"
            style="background:var(--bl);color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;margin:12px 0;">
            Show All Detailed Tables</button>
          <div id="detail-{tl}" class="toggle-content">{_detail_section(df, times)}</div>
        </div>'''

    html = f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Roll Cost Analysis — Q1 2026</title>
<style>
:root {{ --bg:#0d1117; --sf:#161b22; --bd:#30363d; --tx:#c9d1d9; --tm:#8b949e;
  --bl:#58a6ff; --gn:#3fb950; --yl:#d29922; --rd:#f85149; --pu:#bc8cff; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--tx); line-height:1.6; }}
.ctr {{ max-width:1440px; margin:0 auto; padding:20px; }}
.hero {{ background:linear-gradient(135deg,#1a1e2e,#0d1117 50%,#1a1e2e); border:1px solid var(--bd);
  border-radius:12px; padding:40px; text-align:center; margin-bottom:30px; }}
.hero h1 {{ font-size:2.2em; color:#f0f6fc; }}
.hero .sub {{ font-size:1.1em; color:var(--tm); }}
.hero .dr {{ font-size:0.95em; color:var(--bl); margin-top:8px; }}
.sec {{ background:var(--sf); border:1px solid var(--bd); border-radius:12px; padding:28px; margin-bottom:22px; }}
.kpi {{ background:var(--sf); border:1px solid var(--bd); border-radius:8px; padding:18px; text-align:center; }}
.kpi .v {{ font-size:1.6em; font-weight:bold; }}
.kpi .l {{ font-size:0.82em; color:var(--tm); margin-top:4px; }}
.kpi .v.credit {{ color:var(--gn); }} .kpi .v.cost {{ color:var(--rd); }} .kpi .v.neutral {{ color:var(--yl); }}
.entry-header {{ font-size:1.1em; color:var(--yl); margin:16px 0 6px; border-left:3px solid var(--yl); padding-left:10px; }}
h4 {{ font-size:0.95em; color:var(--tm); margin:12px 0 6px; }}
.desc {{ color:var(--tm); font-size:0.88em; margin-bottom:8px; }}
.dt {{ width:100%; border-collapse:collapse; font-size:0.9em; margin-bottom:14px; }}
.dt th {{ background:#21262d; padding:7px 10px; text-align:center; border:1px solid var(--bd); font-weight:600; }}
.dt td {{ padding:5px 10px; text-align:center; border:1px solid var(--bd); font-family:'SF Mono','Cascadia Code',monospace; }}
.dt .tc {{ font-weight:600; background:#21262d; text-align:left; padding-left:14px; }}
.cost {{ color:var(--rd); font-weight:600; }} .credit {{ color:var(--gn); font-weight:600; }}
.cnt {{ color:var(--tm); font-size:0.82em; }}
.sc {{ background:rgba(63,185,80,0.12); }} .c {{ background:rgba(63,185,80,0.06); }}
.n {{ background:transparent; }} .co {{ background:rgba(248,81,73,0.06); }}
.sco {{ background:rgba(248,81,73,0.12); }} .na {{ color:var(--tm); }}
.cg {{ display:grid; gap:16px; margin:16px 0; }}
.cc {{ background:var(--bg); border:1px solid var(--bd); border-radius:8px; padding:12px; text-align:center; }}
.cc img {{ max-width:100%; height:auto; border-radius:4px; }}
.sumtbl {{ width:100%; border-collapse:collapse; font-size:0.9em; margin:8px 0 16px; }}
.sumtbl th {{ background:#21262d; padding:8px 12px; text-align:center; border:1px solid var(--bd); font-weight:600; }}
.sumtbl td {{ padding:6px 12px; text-align:center; border:1px solid var(--bd); font-family:'SF Mono','Cascadia Code',monospace; }}
.sumtbl .tc {{ font-weight:600; background:#21262d; text-align:left; padding-left:14px; }}
.toggle-content {{ display:none; }} .toggle-content.open {{ display:block; }}
.tab-bar {{ display:flex; gap:4px; margin-bottom:0; }}
.tab-btn {{ background:var(--bg); color:var(--tm); border:1px solid var(--bd); border-bottom:none;
  padding:12px 28px; font-size:1.05em; font-weight:600; cursor:pointer; border-radius:8px 8px 0 0; }}
.tab-btn.active {{ background:var(--sf); color:var(--bl); border-bottom:1px solid var(--sf); }}
.tab-btn:hover {{ color:var(--tx); }}
.tab-body {{ background:var(--sf); border:1px solid var(--bd); border-radius:0 12px 12px 12px; padding:28px; margin-bottom:22px; }}
.how {{ background:linear-gradient(135deg,#1a1e2e,#161b22); border:1px solid var(--bl);
  border-radius:12px; padding:22px; margin-bottom:22px; }}
.how h2 {{ color:var(--bl); margin-bottom:10px; }} .how ul {{ padding-left:18px; }} .how li {{ margin-bottom:6px; }}
.meth {{ background:var(--sf); border:1px solid var(--bd); border-radius:12px; padding:28px; margin-bottom:22px; }}
.meth h2 {{ color:var(--tm); }} .meth table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
.meth td,.meth th {{ padding:7px 10px; border:1px solid var(--bd); text-align:left; }}
.meth th {{ background:#21262d; font-weight:600; width:200px; }}
</style>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-content').forEach(e => e.style.display='none');
  document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-'+id).style.display='block';
  document.getElementById('btn-'+id).classList.add('active');
}}
function toggle(id) {{
  var el=document.getElementById(id); el.classList.toggle('open');
  var btn=el.previousElementSibling;
  btn.textContent=el.classList.contains('open')?btn.textContent.replace('Show','Hide'):btn.textContent.replace('Hide','Show');
}}
</script>
</head><body><div class="ctr">
<div class="hero">
  <h1>Roll Cost Analysis</h1>
  <div class="sub">{ticker_labels}</div>
  <div class="dr">{n_days} trading days | {date_min} &mdash; {date_max} | {total_obs:,} observations | PST</div>
</div>
<div class="how">
  <h2>How to Read</h2>
  <ul>
    <li><b>Entry breach %</b> = how deep ITM the 0DTE spread is. 100% = full max loss.</li>
    <li><b>Roll target</b>: "Same Strikes" = keep ITM. "ATM" = at-the-money.</li>
    <li><span class="credit">Green (negative)</span> = CREDIT. <span class="cost">Red (positive)</span> = COST.</li>
    <li>All times PST. DTE+N from future day's opening snapshot.</li>
  </ul>
</div>
<div class="tab-bar">{tab_buttons}</div>
<div class="tab-body">{tab_contents}</div>
<div class="meth">
  <h2>Methodology</h2>
  <table>
    <tr><th>Date Range</th><td>{date_min} &mdash; {date_max} ({n_days} days)</td></tr>
    <tr><th>Tickers</th><td>{ticker_labels}</td></tr>
    <tr><th>Pricing</th><td>Mid = (bid+ask)/2</td></tr>
    <tr><th>Check Times (PST)</th><td>{", ".join(all_times)}</td></tr>
    <tr><th>Roll DTEs</th><td>+1, +2, +3, +5</td></tr>
    <tr><th>Entry Breach</th><td>100%, 75%, 50%, 25%</td></tr>
    <tr><th>Roll Targets</th><td>100% (same), 50%, 25%, 0% (ATM)</td></tr>
  </table>
</div>
<div style="text-align:center;color:var(--tm);padding:20px;font-size:0.82em;">
  Generated by scripts/generate_roll_cost_report.py | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
</div>
</div></body></html>'''

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"\n  Report saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="""
Generate a multi-ticker roll cost report with charts and tabbed HTML.

Runs roll_cost_table.py for each ticker, generates charts (heatmaps, bar charts,
line charts, daily time series), and builds a single tabbed HTML report with
playbooks, summary tables, and detailed data.
        """,
        epilog="""
Examples:
  %(prog)s --start 2026-01-01 --end 2026-03-29
      Full Q1 with defaults: RUT(20pt), SPX(10pt), NDX(50pt)

  %(prog)s --start 2026-01-01 --end 2026-03-29 --tickers RUT:20 SPX:25

  %(prog)s --start 2025-09-15 --end 2025-11-07 \\
      --tickers SPX:25 NDX:50 --options-dir ./options_csv_output_full_15

  %(prog)s --start 2026-01-01 --end 2026-03-29 \\
      --output-dir ./results/my_roll_report
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers with widths as TICKER:WIDTH "
                             "(default: RUT:20 SPX:10 NDX:50)")
    parser.add_argument("--options-dir", default="./options_csv_output_full",
                        help="Options data directory")
    parser.add_argument("--equities-dir", default="./equities_output",
                        help="Equities data directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results/roll_cost_report)")

    args = parser.parse_args()

    # Parse tickers
    if args.tickers:
        tickers_cfg = []
        for t in args.tickers:
            parts = t.split(":")
            tk = parts[0].upper()
            w = int(parts[1]) if len(parts) > 1 else {
                "SPX": 25, "NDX": 50, "RUT": 20, "DJX": 5, "TQQQ": 2
            }.get(tk, 25)
            tickers_cfg.append((tk, w))
    else:
        tickers_cfg = DEFAULT_TICKERS

    output_dir = args.output_dir or "results/roll_cost_report"
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Step 1: Generate data
    print("\n[1/3] Generating roll cost data...")
    csv_paths = run_data_generation(
        tickers_cfg, args.start, args.end,
        args.options_dir, args.equities_dir, output_dir)

    if not csv_paths:
        print("No data generated. Check data availability.")
        sys.exit(1)

    # Load CSVs
    dfs = {tk: pd.read_csv(p) for tk, p in csv_paths.items()}
    active_tickers = [(tk, w) for tk, w in tickers_cfg if tk in dfs]

    # Step 2: Generate charts
    print("\n[2/3] Generating charts...")
    generate_charts(dfs, active_tickers, charts_dir)

    # Step 3: Build report
    print("\n[3/3] Building HTML report...")
    report_path = os.path.join(output_dir,
                               f"report_roll_cost_{args.start}_{args.end}.html")
    build_html_report(dfs, active_tickers, charts_dir, report_path)

    print(f"\n  Done! Open the report:")
    print(f"  open {report_path}")


if __name__ == "__main__":
    main()
