#!/usr/bin/env python3
"""Chart rendering + HTML report for nROI drift analysis.

Produces:
  <output_dir>/charts/<TICKER>/dte<N>_<tier>_<side>.png  (one per bucket)
  <output_dir>/report.html                               (tabbed dark-theme report)

The HTML layout mirrors results/tiered_portfolio/report.html in style:
  * hero banner
  * KPI strip (# tickers, # dates, coverage)
  * tabs per ticker
  * each tab: DTE panels, each containing tier charts side-by-side
"""

from __future__ import annotations

import html
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402 (after backend)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GRID = "#21262d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"

# Pre-set matplotlib dark style
plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "savefig.facecolor": DARK_BG,
    "savefig.edgecolor": DARK_BG,
})


# ──────────────────────────────────────────────────────────────────────
# Chart rendering
# ──────────────────────────────────────────────────────────────────────


def render_chart(
    df_slice: pd.DataFrame,
    out_path: Path,
    title: str,
    subtitle: str,
) -> Dict[str, float]:
    """Render one chart: x = hour ET, y = nROI, one line per date.

    Returns a small stats dict used for the HTML report (coverage, etc).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Only "ok" rows contribute — but we still want to show coverage gaps.
    ok_mask = df_slice["reason"] == "ok"
    ok_rows = df_slice[ok_mask]
    total_cells = len(df_slice)
    filled_cells = len(ok_rows)

    stats = {
        "filled_cells": filled_cells,
        "total_cells": total_cells,
        "coverage_pct": (100.0 * filled_cells / total_cells) if total_cells else 0.0,
        "n_dates": int(ok_rows["date"].nunique()) if not ok_rows.empty else 0,
        "mean_nroi": float(ok_rows["nroi"].mean()) if not ok_rows.empty else float("nan"),
        "median_nroi": float(ok_rows["nroi"].median()) if not ok_rows.empty else float("nan"),
        "max_nroi": float(ok_rows["nroi"].max()) if not ok_rows.empty else float("nan"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)

    if ok_rows.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=14)
    else:
        # Order dates oldest→newest, viridis gradient
        dates_sorted = sorted(ok_rows["date"].unique())
        norm = Normalize(vmin=0, vmax=max(1, len(dates_sorted) - 1))
        cmap = matplotlib.colormaps.get_cmap("viridis")

        for i, d in enumerate(dates_sorted):
            sub = ok_rows[ok_rows["date"] == d].sort_values("hour_et")
            if sub.empty:
                continue
            color = cmap(norm(i))
            ax.plot(
                sub["hour_et"].to_numpy(),
                sub["nroi"].to_numpy(),
                color=color,
                alpha=0.75,
                linewidth=1.1,
                marker="o",
                markersize=3,
            )

        # Colorbar showing oldest→newest
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.set_yticks([0, len(dates_sorted) - 1])
        cbar.ax.set_yticklabels([dates_sorted[0], dates_sorted[-1]],
                                fontsize=8, color=TEXT)
        cbar.outline.set_edgecolor(BORDER)

    ax.set_xlabel("Hour of day (ET)", color=TEXT)
    ax.set_ylabel("normalized ROI (%)", color=TEXT)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left", pad=12)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes,
            color=MUTED, fontsize=9, ha="left")
    ax.grid(True, which="both")
    ax.spines["top"].set_color(BORDER)
    ax.spines["right"].set_color(BORDER)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=DARK_BG)
    plt.close(fig)
    return stats


# ──────────────────────────────────────────────────────────────────────
# HTML assembly
# ──────────────────────────────────────────────────────────────────────


_CSS = f"""
  body {{ margin: 0; background: {DARK_BG}; color: {TEXT};
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }}
  .hero {{ background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
         padding: 32px 40px; color: #fff; }}
  .hero h1 {{ margin: 0; font-size: 28px; }}
  .hero .sub {{ margin-top: 6px; opacity: 0.9; font-size: 14px; }}
  .kpi {{ display: flex; gap: 16px; padding: 20px 40px;
        background: {PANEL_BG}; border-bottom: 1px solid {BORDER}; flex-wrap: wrap; }}
  .kpi .card {{ background: {DARK_BG}; border: 1px solid {BORDER};
             border-radius: 6px; padding: 10px 16px; min-width: 130px; }}
  .kpi .label {{ color: {MUTED}; font-size: 11px; text-transform: uppercase;
               letter-spacing: 0.05em; }}
  .kpi .value {{ color: {TEXT}; font-size: 20px; font-weight: 600; margin-top: 2px; }}
  .tabs {{ display: flex; padding: 0 40px; background: {PANEL_BG};
         border-bottom: 1px solid {BORDER}; }}
  .tab-btn {{ background: transparent; color: {MUTED}; border: 0;
            border-bottom: 2px solid transparent; padding: 12px 18px;
            font-size: 14px; cursor: pointer; }}
  .tab-btn.active {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
  .tab-panel {{ display: none; padding: 24px 40px; }}
  .tab-panel.active {{ display: block; }}
  .dte-panel {{ margin-bottom: 32px; }}
  .dte-panel h2 {{ font-size: 16px; color: {TEXT};
                 border-bottom: 1px solid {BORDER}; padding-bottom: 8px; }}
  .chart-row {{ display: grid;
              grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
              gap: 14px; }}
  .chart-cell {{ background: {PANEL_BG}; border: 1px solid {BORDER};
               border-radius: 6px; padding: 10px; }}
  .chart-cell img {{ width: 100%; height: auto; display: block; }}
  .chart-cell .caption {{ margin-top: 6px; font-size: 12px; color: {MUTED}; }}
  .footer {{ padding: 20px 40px; color: {MUTED}; font-size: 11px;
           border-top: 1px solid {BORDER}; }}
"""

_JS = """
  function showTab(ticker) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + ticker).classList.add('active');
    document.getElementById('btn-' + ticker).classList.add('active');
  }
"""


def _format_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.1f}%"


def _format_num(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.2f}"


def render_report(df: pd.DataFrame, args) -> Path:
    """Top-level entry — writes charts + index.html and returns the HTML path."""
    output_dir = Path(args.output_dir)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    tickers = list(args.tickers.keys())
    dtes = list(args.dtes)
    tiers = list(args.tiers)
    sides = list(args.sides)

    # Render every (ticker, dte, tier, side) chart and collect stats
    stats_map: Dict[tuple, Dict] = {}
    chart_paths: Dict[tuple, Path] = {}
    for ticker in tickers:
        for dte in dtes:
            for tier in tiers:
                for side in sides:
                    mask = (
                        (df.get("ticker") == ticker)
                        & (df.get("dte") == dte)
                        & (df.get("tier") == tier)
                        & (df.get("side") == side)
                    ) if not df.empty else pd.Series(dtype=bool)
                    slice_df = df[mask] if not df.empty else df
                    fname = f"dte{dte}_{tier}_{side}.png"
                    cpath = charts_dir / ticker / fname
                    title = f"{ticker}  DTE {dte}  {tier}  ({side})"
                    subtitle = (
                        f"p{_PERCENTILE_FOR.get(tier, tier)} of trailing close-to-close; "
                        f"width ≤ {args.tickers[ticker]}"
                    )
                    stats = render_chart(slice_df, cpath, title, subtitle)
                    stats_map[(ticker, dte, tier, side)] = stats
                    chart_paths[(ticker, dte, tier, side)] = cpath

    # KPIs
    total = len(df) if not df.empty else 0
    ok_count = int((df["reason"] == "ok").sum()) if not df.empty else 0
    dates_covered = int(df["date"].nunique()) if not df.empty else 0
    coverage_pct = (100.0 * ok_count / total) if total else 0.0

    # Build HTML
    html_parts: List[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'><head>")
    html_parts.append("<meta charset='utf-8'>")
    html_parts.append("<title>nROI Drift — Credit Spread Tiers</title>")
    html_parts.append(f"<style>{_CSS}</style>")
    html_parts.append("</head><body>")

    html_parts.append("<div class='hero'>")
    html_parts.append("<h1>Normalized ROI Drift — Tiered Credit Spreads</h1>")
    html_parts.append(
        f"<div class='sub'>Window: {args.start} → {args.end} &middot; "
        f"Tickers: {', '.join(tickers)} &middot; "
        f"DTE: {', '.join(map(str, dtes))}</div>"
    )
    html_parts.append("</div>")

    html_parts.append("<div class='kpi'>")
    for label, value in [
        ("Tickers", str(len(tickers))),
        ("Dates", str(dates_covered)),
        ("DTEs", str(len(dtes))),
        ("Tiers", str(len(tiers))),
        ("Records", f"{total:,}"),
        ("Coverage", f"{coverage_pct:.1f}%"),
    ]:
        html_parts.append(
            f"<div class='card'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(value)}</div></div>"
        )
    html_parts.append("</div>")

    # Tabs
    html_parts.append("<div class='tabs'>")
    for i, ticker in enumerate(tickers):
        cls = "tab-btn active" if i == 0 else "tab-btn"
        html_parts.append(
            f"<button class='{cls}' id='btn-{ticker}' "
            f"onclick=\"showTab('{ticker}')\">{html.escape(ticker)}</button>"
        )
    html_parts.append("</div>")

    for i, ticker in enumerate(tickers):
        cls = "tab-panel active" if i == 0 else "tab-panel"
        html_parts.append(f"<div class='{cls}' id='panel-{ticker}'>")
        width_cap = args.tickers[ticker]
        html_parts.append(
            f"<p style='color:{MUTED};margin-top:0'>Width cap: {width_cap}. "
            f"Short strike placed at tier percentile; best-nROI spread within "
            f"<code>width_cap</code> selected per snapshot.</p>"
        )
        for dte in dtes:
            html_parts.append("<div class='dte-panel'>")
            html_parts.append(f"<h2>DTE {dte}</h2>")
            html_parts.append("<div class='chart-row'>")
            for tier in tiers:
                for side in sides:
                    key = (ticker, dte, tier, side)
                    path = chart_paths[key]
                    rel = path.relative_to(output_dir).as_posix()
                    stats = stats_map[key]
                    caption = (
                        f"coverage {stats['filled_cells']}/{stats['total_cells']} "
                        f"({stats['coverage_pct']:.0f}%) &middot; "
                        f"median nROI {_format_num(stats['median_nroi'])} &middot; "
                        f"max {_format_num(stats['max_nroi'])}"
                    )
                    html_parts.append("<div class='chart-cell'>")
                    html_parts.append(
                        f"<img src='{html.escape(rel)}' alt='{html.escape(str(key))}'>"
                    )
                    html_parts.append(
                        f"<div class='caption'>{caption}</div>"
                    )
                    html_parts.append("</div>")
            html_parts.append("</div>")  # chart-row
            html_parts.append("</div>")  # dte-panel
        html_parts.append("</div>")  # tab-panel

    html_parts.append(
        "<div class='footer'>Generated by scripts/nroi_drift_analysis.py &middot; "
        f"data sources: csv_exports/options, options_csv_output_full, equities_output</div>"
    )
    html_parts.append(f"<script>{_JS}</script>")
    html_parts.append("</body></html>")

    html_path = output_dir / "report.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    return html_path


_PERCENTILE_FOR = {"aggressive": 90, "moderate": 95, "conservative": 99}
