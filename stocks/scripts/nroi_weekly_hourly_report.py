#!/usr/bin/env python3
"""Weekly × hourly × DTE nROI drift report.

Reads a records.parquet produced by nroi_drift_analysis.py and emits a
single interactive HTML page.

See `docs/strategies/nroi_theta_decay_playbook.md` for the cross-script
playbook (regime classification, trade rules, reproduce commands). The
Summary tab in this report computes the same regime classification on the
fly from the records DataFrame.


  * Outer tabs: one per ticker.
  * Inner tabs under each ticker: `All DTEs`, `DTE 0`, `DTE 1`, `DTE 2`,
    `DTE 5`.
  * Each inner subpanel contains:
      1. A raw-data table (week × hour median nROI + All-hours + n_records)
      2. Seven small Chart.js line charts — one per snapshot hour (10–16 ET),
         x = week, y = median nROI for that hour.
      3. One overall line chart — x = week, y = all-hours median per week.

Charts are built lazily on first subpanel activation so the page stays
responsive even with 120 canvas targets (3 tickers × 5 DTEs × 8 charts).
"""

from __future__ import annotations

import argparse
import html
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Data assembly
# ──────────────────────────────────────────────────────────────────────


DEFAULT_HOURS = [9, 10, 11, 12, 13, 14, 15, 16]
DEFAULT_DTES = [0, 1, 2, 5]

# Must match SNAP_MINUTE_OVERRIDE in nroi_drift_analysis.py. Kept locally so
# this module has no import-time dependency on the analyzer.
SNAP_MINUTE_OVERRIDE = {9: 45}


def snap_minute(h: int) -> int:
    return SNAP_MINUTE_OVERRIDE.get(h, 30)


def hour_label(h: int) -> str:
    """e.g. 9 → '09:45 ET' (6:45 PT), 10 → '10:30 ET' (7:30 PT)."""
    return f"{h:02d}:{snap_minute(h):02d} ET"


def build_weekly_hour_data(
    df: pd.DataFrame,
    hours: Optional[List[int]] = None,
    dte_filter: Optional[int] = None,
) -> Dict[str, Dict[str, List]]:
    """Flat structure keyed by ticker. Optional single-DTE filter.

    Returns `{ticker: {weeks, per_hour, all_hours, n_records}}` where:
      * weeks:       list of ISO week-start dates (Monday-anchored), sorted
      * per_hour:    dict {hour_et: [value | None, ...]} aligned to weeks
      * all_hours:   list of overall medians per week, aligned to weeks
      * n_records:   list of record counts per week, aligned to weeks
    """
    if hours is None:
        hours = list(DEFAULT_HOURS)

    ok = df[df["reason"] == "ok"].copy()
    if dte_filter is not None:
        ok = ok[ok["dte"] == dte_filter]
    if ok.empty:
        return {}

    ok["date"] = pd.to_datetime(ok["date"])
    ok["week"] = (
        ok["date"].dt.to_period("W-MON").apply(lambda p: p.start_time.date())
    )

    out: Dict[str, Dict[str, List]] = {}
    for ticker in sorted(ok["ticker"].unique()):
        sub = ok[ok["ticker"] == ticker]
        weeks_sorted = sorted(sub["week"].unique())

        per_hour_med = (
            sub.groupby(["week", "hour_et"])["nroi"].median().unstack("hour_et")
        )
        all_hours_med = sub.groupby("week")["nroi"].median()
        n_records = sub.groupby("week").size()

        def _align(series: pd.Series) -> List:
            vals = []
            for w in weeks_sorted:
                v = series.get(w)
                vals.append(None if pd.isna(v) else round(float(v), 2))
            return vals

        per_hour: Dict[int, List] = {}
        for h in hours:
            if h in per_hour_med.columns:
                per_hour[h] = _align(per_hour_med[h])
            else:
                per_hour[h] = [None] * len(weeks_sorted)

        out[ticker] = {
            "weeks": [w.isoformat() for w in weeks_sorted],
            "per_hour": per_hour,
            "all_hours": _align(all_hours_med),
            "n_records": [int(n_records.get(w, 0)) for w in weeks_sorted],
        }
    return out


def build_weekly_hour_nested(
    df: pd.DataFrame,
    hours: Optional[List[int]] = None,
    dtes: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Dict[str, List]]]:
    """Nested structure: `{ticker: {dte_key: sub_data}}`.

    `dte_key` is `"All"` for the all-DTEs roll-up, then `"DTE N"` for each
    per-DTE slice. Each `sub_data` matches `build_weekly_hour_data`'s shape.
    Tickers with no data for a given DTE still get an entry with empty lists.
    """
    if dtes is None:
        dtes = list(DEFAULT_DTES)
    if hours is None:
        hours = list(DEFAULT_HOURS)

    all_data = build_weekly_hour_data(df, hours=hours, dte_filter=None)
    tickers = sorted(all_data.keys())

    per_dte: Dict[int, Dict[str, Dict[str, List]]] = {}
    for d in dtes:
        per_dte[d] = build_weekly_hour_data(df, hours=hours, dte_filter=d)

    nested: Dict[str, Dict[str, Dict[str, List]]] = {}
    for tk in tickers:
        # Canonical week axis = the "All" tab's weeks (so every subpanel
        # shares the same x-axis even when a DTE has sparse coverage).
        canon_weeks = all_data[tk]["weeks"]
        nested[tk] = {"All": all_data[tk]}
        for d in dtes:
            sub = per_dte[d].get(tk)
            if sub is None:
                # No data for this (ticker, DTE) — fill with None so the
                # subpanel still renders an empty chart rather than crashing.
                sub = {
                    "weeks": canon_weeks,
                    "per_hour": {h: [None] * len(canon_weeks) for h in hours},
                    "all_hours": [None] * len(canon_weeks),
                    "n_records": [0] * len(canon_weeks),
                }
            else:
                sub = _reindex_to_weeks(sub, canon_weeks, hours)
            nested[tk][f"DTE {d}"] = sub
    return nested


def _reindex_to_weeks(
    sub: Dict[str, List], target_weeks: List[str], hours: List[int]
) -> Dict[str, List]:
    """Align a sub-dict to the given target week list, filling gaps with None."""
    src_weeks = sub["weeks"]
    idx = {w: i for i, w in enumerate(src_weeks)}
    out_per_hour = {h: [] for h in hours}
    out_all = []
    out_n = []
    for w in target_weeks:
        i = idx.get(w)
        for h in hours:
            if i is None:
                out_per_hour[h].append(None)
            else:
                out_per_hour[h].append(sub["per_hour"].get(h, [None])[i])
        if i is None:
            out_all.append(None)
            out_n.append(0)
        else:
            out_all.append(sub["all_hours"][i])
            out_n.append(sub["n_records"][i])
    return {
        "weeks": list(target_weeks),
        "per_hour": out_per_hour,
        "all_hours": out_all,
        "n_records": out_n,
    }


# ──────────────────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────────────────


_CSS = """
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1117; color:#c9d1d9;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
  .hero { background:linear-gradient(135deg,#1f6feb 0%,#58a6ff 100%);
          padding:28px 36px; color:#fff; }
  .hero h1 { margin:0; font-size:24px; }
  .hero .sub { margin-top:4px; opacity:.9; font-size:13px; max-width:900px; }
  .tabs { display:flex; flex-wrap:wrap; background:#161b22; padding:0 36px;
          border-bottom:1px solid #30363d; }
  .tab-btn { background:transparent; color:#8b949e; border:0;
             border-bottom:2px solid transparent; padding:12px 20px;
             font-size:14px; font-weight:500; cursor:pointer; }
  .tab-btn.active { color:#58a6ff; border-bottom-color:#58a6ff; }
  .tab-btn:hover { color:#c9d1d9; }
  .dte-tabs { padding:0 36px; background:#0f141b;
              border-bottom:1px solid #30363d; }
  .dte-tab-btn { background:transparent; color:#8b949e; border:0;
                 border-bottom:2px solid transparent; padding:10px 16px;
                 font-size:13px; cursor:pointer; }
  .dte-tab-btn.active { color:#f0883e; border-bottom-color:#f0883e; }
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }
  .dte-panel { display:none; padding:20px 36px 40px; }
  .dte-panel.active { display:block; }
  h2 { color:#c9d1d9; font-size:15px; margin:24px 0 8px; font-weight:500; }
  h2.first { margin-top:0; }
  .sub-note { color:#8b949e; font-size:12px; margin-bottom:12px; }
  .data-table-wrap { max-height:420px; overflow-y:auto;
                     border:1px solid #30363d; border-radius:6px;
                     background:#161b22; margin-bottom:20px; }
  table.data { border-collapse:collapse; width:100%; font-size:12px; }
  table.data th, table.data td { border:1px solid #21262d;
                                 padding:4px 8px; text-align:right; }
  table.data th { background:#0d1117; color:#c9d1d9; position:sticky; top:0;
                  z-index:1; }
  table.data tr:nth-child(even) td { background:#1a1f27; }
  table.data td.week { color:#8b949e; text-align:left; }
  table.data td.nan { color:#30363d; }
  .chart-grid { display:grid;
                grid-template-columns:repeat(auto-fit,minmax(360px,1fr));
                gap:14px; margin-bottom:22px; }
  .chart-cell { background:#161b22; border:1px solid #30363d;
                border-radius:6px; padding:10px 12px; }
  .chart-cell h3 { margin:0 0 6px; font-size:13px; color:#c9d1d9;
                   font-weight:500; }
  .chart-cell .meta { color:#6e7681; font-size:11px; margin-bottom:4px; }
  /* Fixed-height wrappers prevent Chart.js from elongating vertically.
     Taller than the usual 180/260 px defaults so small week-over-week
     movements in nROI are legible. */
  .canvas-box { position:relative; height:320px; width:100%; }
  .canvas-box-overall { position:relative; height:440px; width:100%; }
  .chart-overall-wrap { background:#161b22; border:1px solid #30363d;
                        border-radius:6px; padding:14px 16px; }
  .chart-overall-wrap h3 { margin:0 0 6px; font-size:14px; color:#f0883e;
                           font-weight:600; }
  .trend-legend { color:#6e7681; font-size:11px; margin-top:6px;
                  font-style:italic; }
  .footer { padding:16px 36px; color:#6e7681; font-size:11px;
            border-top:1px solid #30363d; }
"""


def _subpanel_id(ticker: str, dte_key: str) -> str:
    return f"sub-{ticker}-{dte_key.replace(' ', '')}"


def _build_table_html(sub: Dict[str, List], hours: List[int]) -> str:
    weeks = sub["weeks"]
    per_hour = sub["per_hour"]
    all_hours = sub["all_hours"]
    n_records = sub["n_records"]

    parts = ["<table class='data'><thead><tr>"]
    parts.append("<th>week</th>")
    for h in hours:
        parts.append(f"<th>{html.escape(hour_label(h))}</th>")
    parts.append("<th>All_hours</th><th>n</th></tr></thead><tbody>")

    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "<td class='nan'>—</td>"
        return f"<td>{v:.2f}</td>"

    for i, w in enumerate(weeks):
        parts.append("<tr>")
        parts.append(f"<td class='week'>{html.escape(str(w))}</td>")
        for h in hours:
            parts.append(_fmt(per_hour[h][i]))
        parts.append(_fmt(all_hours[i]))
        parts.append(f"<td>{n_records[i]}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_html(
    nested: Dict[str, Dict[str, Dict[str, List]]],
    out_path: Path,
    title: str = "nROI weekly × hourly × DTE — Jan 2025 → present",
    subtitle: str = "",
    hours: Optional[List[int]] = None,
) -> Path:
    if hours is None:
        hours = list(DEFAULT_HOURS)

    tickers = list(nested.keys())
    if not tickers:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "<html><body><p>No data</p></body></html>", encoding="utf-8"
        )
        return out_path

    dte_keys = list(next(iter(nested.values())).keys())  # preserves insertion order

    parts: List[str] = []
    parts.append("<!DOCTYPE html>")
    parts.append("<html lang='en'><head>")
    parts.append("<meta charset='utf-8'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append("<script src='https://cdn.jsdelivr.net/npm/chart.js@4.4.1/"
                 "dist/chart.umd.min.js'></script>")
    parts.append(f"<style>{_CSS}</style>")
    parts.append("</head><body>")

    parts.append("<div class='hero'>")
    parts.append(f"<h1>{html.escape(title)}</h1>")
    if subtitle:
        parts.append(f"<div class='sub'>{html.escape(subtitle)}</div>")
    parts.append("</div>")

    # Ticker tabs
    parts.append("<div class='tabs ticker-tabs'>")
    for i, tk in enumerate(tickers):
        cls = "tab-btn active" if i == 0 else "tab-btn"
        parts.append(
            f"<button class='{cls}' data-target='panel-{tk}'>"
            f"{html.escape(tk)}</button>"
        )
    parts.append("</div>")

    # Per-ticker panels
    for i, tk in enumerate(tickers):
        cls = "tab-panel active" if i == 0 else "tab-panel"
        parts.append(f"<div class='{cls}' id='panel-{tk}'>")

        # DTE subtabs for this ticker
        parts.append(f"<div class='tabs dte-tabs' data-ticker='{tk}'>")
        for j, dk in enumerate(dte_keys):
            cls2 = "dte-tab-btn active" if j == 0 else "dte-tab-btn"
            parts.append(
                f"<button class='{cls2}' "
                f"data-ticker='{tk}' data-dte='{html.escape(dk)}'>"
                f"{html.escape(dk)}</button>"
            )
        parts.append("</div>")

        # Each DTE subpanel
        for j, dk in enumerate(dte_keys):
            cls3 = "dte-panel active" if j == 0 else "dte-panel"
            sub_id = _subpanel_id(tk, dk)
            sub = nested[tk][dk]
            parts.append(f"<div class='{cls3}' id='{sub_id}'>")

            # 1. Per-hour chart grid
            parts.append("<h2 class='first'>Per-hour charts — one point "
                         "per week</h2>")
            parts.append(
                "<div class='sub-note'>Each chart: median nROI at the "
                "snapshot nearest to that hour's target mark (9:45 ET for "
                "the open slot, :30 ET for the rest), week by week. "
                "Hover for values + record count. Dashed line = OLS linear "
                "regression trend.</div>"
            )
            parts.append("<div class='chart-grid'>")
            for h in hours:
                parts.append("<div class='chart-cell'>")
                parts.append(f"<h3>{html.escape(hour_label(h))}</h3>")
                parts.append("<div class='canvas-box'>")
                parts.append(
                    f"<canvas id='chart-{tk}-{dk.replace(' ', '')}-h{h}'>"
                    "</canvas>"
                )
                parts.append("</div>")
                parts.append("</div>")
            parts.append("</div>")

            # 2. Overall chart
            parts.append("<h2>All-hours aggregate — one point per week</h2>")
            parts.append(
                "<div class='sub-note'>Median nROI across every valid "
                "record in the week (hour × tier collapsed). Dashed green "
                "line = OLS linear regression trend.</div>"
            )
            parts.append("<div class='chart-overall-wrap'>")
            parts.append("<h3>All hours (median)</h3>")
            parts.append("<div class='canvas-box-overall'>")
            parts.append(
                f"<canvas id='chart-{tk}-{dk.replace(' ', '')}-all'>"
                "</canvas>"
            )
            parts.append("</div>")
            parts.append("</div>")

            # 3. Raw data table — last, for reference
            parts.append("<h2>Raw data (weekly × hourly median nROI)</h2>")
            parts.append(
                f"<div class='sub-note'>"
                f"{html.escape(tk)} &middot; {html.escape(dk)}. "
                f"Medians taken across all valid records per (week, hour) "
                f"cell. <code>n</code> is the record count in the week. "
                f"Weeks with n &lt; 30 are noisy. Em-dash = no data.</div>"
            )
            parts.append("<div class='data-table-wrap'>")
            parts.append(_build_table_html(sub, hours))
            parts.append("</div>")

            parts.append("</div>")  # dte-panel

        parts.append("</div>")  # ticker tab-panel

    parts.append(
        "<div class='footer'>Generated by scripts/nroi_weekly_hourly_report.py"
        "</div>"
    )

    # JS + data
    payload = json.dumps({"data": nested, "hours": hours, "dte_keys": dte_keys})
    parts.append(
        "<script>\n"
        f"const PAYLOAD = {payload};\n"
        + _JS_BODY
        + "\n</script>"
    )

    parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


_JS_BODY = textwrap.dedent("""
    const DATA = PAYLOAD.data;
    const HOURS = PAYLOAD.hours;
    const DTE_KEYS = PAYLOAD.dte_keys;
    const BUILT = new Set();  // subpanel ids whose charts have been built

    // Ordinary least-squares over (i, values[i]) skipping null values.
    // Returns an array of predicted y values aligned to the full input.
    function regressionLine(values) {
      const pts = [];
      values.forEach((v, i) => {
        if (v !== null && v !== undefined && !Number.isNaN(v)) {
          pts.push([i, v]);
        }
      });
      if (pts.length < 2) return values.map(() => null);
      const n = pts.length;
      let sx=0, sy=0, sxy=0, sxx=0;
      for (const [x, y] of pts) { sx+=x; sy+=y; sxy+=x*y; sxx+=x*x; }
      const denom = n*sxx - sx*sx;
      if (denom === 0) return values.map(() => null);
      const slope = (n*sxy - sx*sy) / denom;
      const intercept = (sy - slope*sx) / n;
      return values.map((_, i) => +(slope*i + intercept).toFixed(3));
    }

    function chartOpts(title, n_records) {
      return {
        responsive: true,
        // Keep maintainAspectRatio false, but we've now wrapped the canvas
        // in a fixed-height parent div so the sizing is stable.
        maintainAspectRatio: false,
        resizeDelay: 80,
        animation: false,
        // Hover/tooltip fires for the nearest x-index no matter where on
        // the plot the cursor is (intersect:false). Crosshair feel.
        interaction: { mode: 'index', intersect: false, axis: 'x' },
        hover:       { mode: 'index', intersect: false, axis: 'x' },
        plugins: {
          legend: {
            display: true,
            labels: {
              color: '#8b949e', boxWidth: 14, font: { size: 10 },
              filter: (item) => true,
            },
            position: 'bottom',
          },
          tooltip: {
            enabled: true,
            mode: 'index',
            intersect: false,
            axis: 'x',
            position: 'nearest',
            backgroundColor: '#161b22',
            titleColor: '#58a6ff',
            bodyColor: '#c9d1d9',
            borderColor: '#30363d',
            borderWidth: 1,
            padding: 10,
            caretSize: 6,
            titleFont: { size: 12, weight: 'bold' },
            bodyFont: { size: 12 },
            filter: (ctx) => ctx.dataset.label !== 'trend',
            callbacks: {
              title: (items) => {
                if (!items.length) return '';
                return 'Week of ' + items[0].label;
              },
              afterTitle: (items) => 'n = ' + (n_records[items[0].dataIndex] || 0),
              label: (ctx) => {
                if (ctx.parsed.y === null || ctx.parsed.y === undefined) {
                  return ctx.dataset.label + ': —';
                }
                return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2);
              },
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#8b949e', autoSkip: true,
                     maxTicksLimit: 14, maxRotation: 45, minRotation: 45,
                     font: { size: 10 } },
            grid: { color: '#30363d33' },
          },
          y: {
            ticks: { color: '#8b949e', font: { size: 10 } },
            grid: { color: '#30363d33' },
            title: { display: true, text: 'nROI (%)', color: '#8b949e',
                     font: { size: 10 } },
            beginAtZero: true,
          },
        },
      };
    }

    function buildSubpanelCharts(ticker, dte_key) {
      const key = ticker + '|' + dte_key;
      if (BUILT.has(key)) return;
      const sub = DATA[ticker][dte_key];
      if (!sub) return;
      const weeks = sub.weeks;
      const n = sub.n_records;
      const dteId = dte_key.replace(' ', '');
      // Per-hour small charts
      HOURS.forEach(h => {
        const id = 'chart-' + ticker + '-' + dteId + '-h' + h;
        const ctx = document.getElementById(id);
        if (!ctx) return;
        const raw = sub.per_hour[h] || [];
        new Chart(ctx, {
          type: 'line',
          data: {
            labels: weeks,
            datasets: [
              {
                label: 'weekly median',
                data: raw,
                borderColor: '#58a6ff',
                backgroundColor: '#58a6ff33',
                borderWidth: 1.4, pointRadius: 1.8, pointHoverRadius: 4,
                tension: 0.15, spanGaps: true, order: 1,
              },
              {
                label: 'trend',
                data: regressionLine(raw),
                borderColor: '#ffa657',
                borderWidth: 1.6,
                borderDash: [6, 4],
                pointRadius: 0,
                pointHoverRadius: 0,
                tension: 0,
                spanGaps: true,
                order: 0,
              },
            ],
          },
          options: chartOpts(h, n),
        });
      });
      // Overall
      const allId = 'chart-' + ticker + '-' + dteId + '-all';
      const allCtx = document.getElementById(allId);
      if (allCtx) {
        const raw = sub.all_hours;
        new Chart(allCtx, {
          type: 'line',
          data: {
            labels: weeks,
            datasets: [
              {
                label: 'weekly median (all hours)',
                data: raw,
                borderColor: '#f0883e',
                backgroundColor: '#f0883e33',
                borderWidth: 2.2, pointRadius: 2.4, pointHoverRadius: 5,
                tension: 0.15, spanGaps: true, fill: true, order: 1,
              },
              {
                label: 'trend',
                data: regressionLine(raw),
                borderColor: '#3fb950',
                borderWidth: 2.0,
                borderDash: [7, 5],
                pointRadius: 0,
                pointHoverRadius: 0,
                tension: 0,
                spanGaps: true,
                order: 0,
              },
            ],
          },
          options: chartOpts('All hours', n),
        });
      }
      BUILT.add(key);
    }

    function activateTicker(ticker) {
      document.querySelectorAll('.tab-btn').forEach(
        b => b.classList.toggle('active', b.dataset.target === 'panel-' + ticker));
      document.querySelectorAll('.tab-panel').forEach(
        p => p.classList.toggle('active', p.id === 'panel-' + ticker));
      // Ensure the currently-active DTE subpanel for this ticker is built
      const activeDteBtn = document.querySelector(
        "#panel-" + ticker + " .dte-tab-btn.active");
      if (activeDteBtn) {
        buildSubpanelCharts(ticker, activeDteBtn.dataset.dte);
      }
    }

    function activateDte(ticker, dte_key) {
      const ddte = dte_key.replace(' ', '');
      document.querySelectorAll(
        "#panel-" + ticker + " .dte-tab-btn").forEach(b =>
        b.classList.toggle('active', b.dataset.dte === dte_key));
      document.querySelectorAll(
        "#panel-" + ticker + " .dte-panel").forEach(p =>
        p.classList.toggle('active', p.id === 'sub-' + ticker + '-' + ddte));
      buildSubpanelCharts(ticker, dte_key);
    }

    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const ticker = btn.dataset.target.replace('panel-', '');
        activateTicker(ticker);
      });
    });
    document.querySelectorAll('.dte-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        activateDte(btn.dataset.ticker, btn.dataset.dte);
      });
    });

    // Build the initial visible subpanel on load
    Object.keys(DATA).forEach((tk, i) => {
      if (i === 0) buildSubpanelCharts(tk, DTE_KEYS[0]);
    });
""")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            Weekly × hourly × DTE line-chart HTML report for nROI drift.

            Reads records.parquet from a prior sweep and emits a single
            interactive HTML page with outer tabs per ticker and inner tabs
            per DTE bucket. Each subpanel shows a raw-data table, seven
            per-hour line charts, and an all-hours aggregate chart.
        """).strip(),
        epilog=textwrap.dedent("""
            Examples:
              %(prog)s --records results/nroi_drift_16mo/raw/records.parquet

              %(prog)s --records results/nroi_drift_16mo/raw/records.parquet \\
                       --out     results/nroi_drift_16mo/hourly_lines.html \\
                       --title   "nROI drift — SPX/RUT/NDX — 16 months"
        """).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--records", type=Path, required=True,
                        help="Path to records.parquet.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HTML path. "
                             "Default: <records-dir>/../hourly_lines.html")
    parser.add_argument("--title",
                        default="nROI weekly × hourly × DTE — Jan 2025 → present")
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--hours",
                        type=lambda s: [int(x) for x in s.split(",")],
                        default=None,
                        help="ET hours to plot. Default: 10,11,12,13,14,15,16.")
    parser.add_argument("--dtes",
                        type=lambda s: [int(x) for x in s.split(",")],
                        default=None,
                        help="DTE buckets to tab. Default: 0,1,2,5.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.records)
    nested = build_weekly_hour_nested(df, hours=args.hours, dtes=args.dtes)
    if not nested:
        print("No OK records found in input parquet; nothing to render.")
        return 1
    out = args.out or (args.records.parent.parent / "hourly_lines.html")
    path = render_html(nested, out, title=args.title, subtitle=args.subtitle,
                       hours=args.hours)
    any_ticker = next(iter(nested.values()))
    weeks = any_ticker["All"]["weeks"]
    print(f"Wrote {path}")
    print(f"  tickers: {', '.join(nested.keys())}")
    print(f"  DTE tabs: {', '.join(next(iter(nested.values())).keys())}")
    print(f"  weeks:   {len(weeks)} ({weeks[0]} → {weeks[-1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
