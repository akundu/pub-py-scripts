"""
build_report.py — HTML report from early-exit sweep results.

Usage:
    python3 build_report.py                     # reads results_v1.tsv + best_config_v1.json
    python3 build_report.py --prefix puts_only  # different prefix
    python3 build_report.py --help

Output: results/dte_comparison/report_autoresearch_early_exit.html
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent.parent
OUT_PATH  = ROOT / "results" / "dte_comparison" / "report_autoresearch_early_exit.html"
STATIC_DIR = Path(os.path.expanduser("~/programs/http-proxy/static/doc/stocks"))


def _color_improve(v: float) -> str:
    if v > 5:   return "color:#3fb950"
    if v > 0:   return "color:#79c0ff"
    if v > -5:  return "color:#d29922"
    return "color:#f85149"


def _color_sharpe(v: float) -> str:
    if v > 3:  return "color:#3fb950"
    if v > 1:  return "color:#79c0ff"
    if v > 0:  return "color:#d29922"
    return "color:#f85149"


def build_html(df: pd.DataFrame, best: dict, prefix: str) -> str:
    n_trials    = len(df)
    n_improve   = int((df["improvement_pct"] > 0).sum())
    best_cfg    = best.get("config", {})
    best_m      = best.get("metrics", {})

    # Top 20 configs by score
    top = df[df["status"] == "keep"].nlargest(20, "score")

    top_rows = ""
    for _, r in top.iterrows():
        imp = r["improvement_pct"]
        sh  = r["ann_sharpe"]
        hsh = r["hold_sharpe"]
        sl_str = "off" if r["stop_loss"] >= 9999 else str(int(r["stop_loss"]))
        top_rows += (
            f"<tr>"
            f"<td>{int(r['trial'])}</td>"
            f"<td>{int(r['profit_target'])}</td>"
            f"<td>{sl_str}</td>"
            f"<td>{r['apply_dtes']}</td>"
            f"<td>{r['apply_tiers']}</td>"
            f"<td>{r['apply_sides']}</td>"
            f"<td style='{_color_sharpe(sh)}'>{sh:.3f}</td>"
            f"<td style='{_color_sharpe(hsh)}'>{hsh:.3f}</td>"
            f"<td style='{_color_improve(imp)}'>{imp:+.1f}%</td>"
            f"<td>{r['score']:.4f}</td>"
            f"</tr>\n"
        )

    # Profit-target distribution (all kept trials)
    kept = df[df["status"] == "keep"]
    if not kept.empty:
        pt_dist = kept.groupby("profit_target")["improvement_pct"].mean().reset_index()
        pt_bars = ""
        for _, r in pt_dist.iterrows():
            v = r["improvement_pct"]
            color = "#3fb950" if v > 0 else "#f85149"
            pt_bars += (
                f"<tr><td>{int(r['profit_target'])}%</td>"
                f"<td style='color:{color}'>{v:+.1f}%</td>"
                f"<td>{int((kept['profit_target'] == r['profit_target']).sum())}</td></tr>\n"
            )
    else:
        pt_bars = "<tr><td colspan=3>No kept trials yet</td></tr>"

    # Best config summary
    bc_pt   = best_cfg.get("profit_target_pct", "?")
    bc_sl   = best_cfg.get("stop_loss_pct", "?")
    bc_dtes = best_cfg.get("apply_dtes", [])
    bc_tiers = best_cfg.get("apply_tiers", [])
    bc_sides = best_cfg.get("apply_sides", [])
    bm_sh   = best_m.get("annualized_sharpe", 0)
    bm_hsh  = best_m.get("hold_sharpe", 0)
    bm_imp  = best_m.get("improvement_pct", 0)
    bm_roi  = best_m.get("annualized_roi_pct", 0)
    bm_dd   = best_m.get("max_drawdown_pct", 0)
    bm_wd   = best_m.get("win_day_pct", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Early Exit Sweep — {prefix}</title>
<style>
body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:24px;max-width:1200px;margin:0 auto}}
h1{{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:8px}}
h2{{color:#79c0ff;margin-top:32px}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
.kpi-card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px 24px;min-width:160px}}
.kpi-val{{font-size:2em;font-weight:bold}}
.kpi-lbl{{color:#8b949e;font-size:0.85em}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}
th{{background:#161b22;color:#8b949e;padding:8px 12px;text-align:left;border-bottom:1px solid #30363d}}
td{{padding:6px 12px;border-bottom:1px solid #21262d}}
tr:hover td{{background:#161b22}}
.best-box{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin:16px 0}}
.best-box code{{color:#79c0ff}}
.footer{{color:#8b949e;font-size:0.85em;margin-top:40px;border-top:1px solid #21262d;padding-top:12px}}
</style>
</head>
<body>
<h1>Early Exit Sweep Results — {prefix}</h1>
<p>Hill-climbing search over profit-target %, stop-loss %, DTE scope, tier scope, and side.
Score = ann_sharpe × (1 - dd_penalty/30%).</p>

<div class="kpi">
  <div class="kpi-card"><div class="kpi-val">{n_trials}</div><div class="kpi-lbl">total trials</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#3fb950">{n_improve}</div><div class="kpi-lbl">configs beating hold</div></div>
  <div class="kpi-card"><div class="kpi-val" style="{_color_sharpe(bm_sh)}">{bm_sh:.3f}</div><div class="kpi-lbl">best ann Sharpe</div></div>
  <div class="kpi-card"><div class="kpi-val" style="{_color_improve(bm_imp)}">{bm_imp:+.1f}%</div><div class="kpi-lbl">vs hold-to-expiry</div></div>
  <div class="kpi-card"><div class="kpi-val">{bc_pt}%</div><div class="kpi-lbl">best profit target</div></div>
</div>

<h2>Best Config Found</h2>
<div class="best-box">
  <table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Profit target</td><td><code>{bc_pt}%</code> credit captured → close</td></tr>
  <tr><td>Stop loss</td><td><code>{'disabled' if bc_sl >= 9999 else f'-{bc_sl}%'}</code></td></tr>
  <tr><td>Apply to DTEs</td><td><code>{bc_dtes}</code></td></tr>
  <tr><td>Apply to tiers</td><td><code>{bc_tiers}</code></td></tr>
  <tr><td>Apply to sides</td><td><code>{bc_sides}</code></td></tr>
  <tr><td>Ann Sharpe (early exit)</td><td style="{_color_sharpe(bm_sh)}">{bm_sh:.3f}</td></tr>
  <tr><td>Ann Sharpe (hold)</td><td style="{_color_sharpe(bm_hsh)}">{bm_hsh:.3f}</td></tr>
  <tr><td>Improvement vs hold</td><td style="{_color_improve(bm_imp)}">{bm_imp:+.1f}%</td></tr>
  <tr><td>Ann ROI</td><td>{bm_roi:.1f}%</td></tr>
  <tr><td>Max drawdown</td><td>{bm_dd:.1f}%</td></tr>
  <tr><td>Win-day %</td><td>{bm_wd:.1f}%</td></tr>
  </table>
</div>

<h2>Top 20 Configs by Score</h2>
<table>
<tr>
  <th>#</th><th>PT%</th><th>SL</th><th>DTEs</th><th>Tiers</th><th>Sides</th>
  <th>Sharpe</th><th>Hold Sh</th><th>Improve</th><th>Score</th>
</tr>
{top_rows}
</table>

<h2>Mean Improvement by Profit Target (all kept configs)</h2>
<table>
<tr><th>Profit target</th><th>Mean improvement vs hold</th><th>N configs</th></tr>
{pt_bars}
</table>

<div class="footer">
  {n_trials} trials · prefix={prefix} · source: autoresearch_early_exit/loop.py
</div>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prefix", default="v1",
                   help="Results file prefix (matches --out-prefix in loop.py). Default: v1")
    args = p.parse_args()

    res_path  = Path(__file__).parent / f"results_{args.prefix}.tsv"
    best_path = Path(__file__).parent / f"best_config_{args.prefix}.json"

    if not res_path.exists():
        print(f"ERROR: {res_path} not found. Run loop.py first.")
        return

    df = pd.read_csv(res_path, sep="\t")
    best = json.loads(best_path.read_text()) if best_path.exists() else {}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(df, best, args.prefix)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Report: {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")

    if STATIC_DIR.is_dir():
        dest = STATIC_DIR / OUT_PATH.name
        shutil.copy2(OUT_PATH, dest)
        print(f"Copied: {dest}")


if __name__ == "__main__":
    main()
