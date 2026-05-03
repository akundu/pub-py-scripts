"""
Ticker autocomplete backend for the chart page.

The frontend's clickable symbol header sends `/api/tickers/search?q=...`
as the user types. This module loads the universe of known tickers from
`data/lists/*_symbols.yaml` (sp-500, nasdaq, etfs, dow-jones, etc.) and
ranks them against the query. A small hardcoded `KNOWN_NAMES` map
provides company / index names for the most common matches; tickers
without a known name are still searchable and just render with an empty
right-hand column.

Future enhancement: fall through to Polygon's `/v3/reference/tickers`
for tickers we don't have a name for. Out of scope here — the static
map covers all the indices/ETFs/mega-caps that 95% of autocomplete
queries actually want.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List


# Hardcoded names for indices, common ETFs, and the most-traded equities.
# Intentionally curated rather than fetched dynamically — this is a fast
# path that runs on every keystroke; an external API call per request
# would dominate latency.
KNOWN_NAMES: Dict[str, str] = {
    # Indices (Polygon prefix `I:` is added at chart-load time, so the
    # autocomplete just shows the bare form)
    "NDX":   "NASDAQ 100",
    "SPX":   "S&P 500",
    "RUT":   "Russell 2000",
    "DJX":   "Dow Jones Industrial Average",
    "DJI":   "Dow Jones Industrial Average",
    "VIX":   "CBOE Volatility Index",
    "VIX1D": "CBOE 1-Day Volatility",
    "IXIC":  "NASDAQ Composite",
    "NYA":   "NYSE Composite",
    "TNX":   "10-Year Treasury Yield",
    "FVX":   "5-Year Treasury Yield",

    # Broad-market and leveraged ETFs
    "SPY":   "SPDR S&P 500 ETF",
    "QQQ":   "Invesco QQQ Trust",
    "DIA":   "SPDR Dow Jones Industrial ETF",
    "IWM":   "iShares Russell 2000 ETF",
    "VOO":   "Vanguard S&P 500 ETF",
    "VTI":   "Vanguard Total Stock Market ETF",
    "VEA":   "Vanguard FTSE Developed Markets ETF",
    "VWO":   "Vanguard FTSE Emerging Markets ETF",
    "TQQQ":  "ProShares UltraPro QQQ",
    "SQQQ":  "ProShares UltraPro Short QQQ",
    "FNGU":  "MicroSectors FANG+ 3X Leveraged",
    "GLD":   "SPDR Gold Shares",
    "SLV":   "iShares Silver Trust",
    "TLT":   "iShares 20+ Year Treasury Bond ETF",
    "USO":   "United States Oil Fund",
    "XLE":   "Energy Select Sector SPDR Fund",
    "XLF":   "Financial Select Sector SPDR Fund",
    "XLK":   "Technology Select Sector SPDR Fund",
    "XLV":   "Health Care Select Sector SPDR Fund",
    "ARKK":  "ARK Innovation ETF",

    # Mega-caps
    "AAPL":  "Apple Inc.",
    "MSFT":  "Microsoft Corporation",
    "GOOG":  "Alphabet Inc. Class C",
    "GOOGL": "Alphabet Inc. Class A",
    "AMZN":  "Amazon.com, Inc.",
    "NVDA":  "NVIDIA Corporation",
    "META":  "Meta Platforms, Inc.",
    "TSLA":  "Tesla, Inc.",
    "BRK.B": "Berkshire Hathaway Class B",
    "BRK.A": "Berkshire Hathaway Class A",
    "AVGO":  "Broadcom Inc.",
    "ORCL":  "Oracle Corporation",
    "JPM":   "JPMorgan Chase & Co.",
    "V":     "Visa Inc.",
    "MA":    "Mastercard Incorporated",
    "JNJ":   "Johnson & Johnson",
    "WMT":   "Walmart Inc.",
    "PG":    "Procter & Gamble",
    "HD":    "The Home Depot, Inc.",
    "BAC":   "Bank of America Corp",
    "XOM":   "Exxon Mobil Corporation",
    "CVX":   "Chevron Corporation",
    "ABBV":  "AbbVie Inc.",
    "LLY":   "Eli Lilly and Company",
    "MRK":   "Merck & Co., Inc.",
    "PFE":   "Pfizer Inc.",
    "KO":    "The Coca-Cola Company",
    "PEP":   "PepsiCo, Inc.",
    "DIS":   "The Walt Disney Company",
    "INTC":  "Intel Corporation",
    "AMD":   "Advanced Micro Devices",
    "CSCO":  "Cisco Systems, Inc.",
    "ADBE":  "Adobe Inc.",
    "CRM":   "Salesforce, Inc.",
    "NFLX":  "Netflix, Inc.",
    "ABNB":  "Airbnb, Inc.",
    "PLTR":  "Palantir Technologies",
    "SHOP":  "Shopify Inc.",
    "MU":    "Micron Technology",
    "PYPL":  "PayPal Holdings",
    "SOFI":  "SoFi Technologies",
    "COIN":  "Coinbase Global, Inc.",
    "RBLX":  "Roblox Corporation",
    "UBER":  "Uber Technologies",
    "LYFT":  "Lyft, Inc.",
    "F":     "Ford Motor Company",
    "GM":    "General Motors",
    "T":     "AT&T Inc.",
    "VZ":    "Verizon Communications",
    "BA":    "The Boeing Company",
    "GE":    "General Electric",
    "GS":    "Goldman Sachs Group",
    "MS":    "Morgan Stanley",
    "C":     "Citigroup Inc.",
    "WFC":   "Wells Fargo & Co",
    "BLK":   "BlackRock, Inc.",
    "NOW":   "ServiceNow, Inc.",
    "TT":    "Trane Technologies",
    "TTD":   "The Trade Desk",
    "TTWO":  "Take-Two Interactive",
    "WDAY":  "Workday, Inc.",
    "CART":  "Maplebear Inc. (Instacart)",
}


def _yaml_dir() -> Path:
    """Where the per-list symbol YAMLs live."""
    # __file__ → common/ticker_search.py; lists are a sibling of common.
    return Path(__file__).resolve().parent.parent / "data" / "lists"


@lru_cache(maxsize=1)
def all_tickers(yaml_dir: str | None = None) -> List[str]:
    """Sorted, de-duplicated union of every ticker across the YAML lists.

    Cached for the process lifetime — these files are static between
    rebuilds, so we parse them once. The KNOWN_NAMES keys are merged in
    too so indices like NDX/SPX always show up even if a particular
    YAML list doesn't include them.

    `yaml_dir` is overridable for tests; production callers pass nothing.
    """
    import yaml as _yaml
    base = Path(yaml_dir) if yaml_dir else _yaml_dir()
    seen: set[str] = set()
    if base.exists():
        for path in sorted(base.glob("*.yaml")):
            try:
                data = _yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for sym in (data or {}).get("symbols", []) or []:
                if isinstance(sym, str) and sym.strip():
                    seen.add(sym.strip().upper())
    # Always include the curated KNOWN_NAMES keys so indices/ETFs are
    # findable even if no YAML list names them.
    for k in KNOWN_NAMES:
        seen.add(k)
    return sorted(seen)


def search_tickers(
    query: str,
    limit: int = 10,
    yaml_dir: str | None = None,
) -> List[Dict[str, str]]:
    """Return up to `limit` `{symbol, name}` matches for `query`.

    Ranking (best first):
      1. Exact ticker match
      2. Ticker starts with query
      3. Company name starts with query (or any word in name does)
      4. Substring match — query appears anywhere in ticker or name

    Empty / whitespace queries return []. The match is case-insensitive
    on both sides.
    """
    q = (query or "").strip().upper()
    if not q:
        return []
    tickers = all_tickers(yaml_dir)

    exact: List[tuple[str, str]] = []
    prefix_t: List[tuple[str, str]] = []
    prefix_n: List[tuple[str, str]] = []
    substring: List[tuple[str, str]] = []

    for t in tickers:
        n = KNOWN_NAMES.get(t, "")
        n_up = n.upper()
        if t == q:
            exact.append((t, n))
            continue
        if t.startswith(q):
            prefix_t.append((t, n))
            continue
        # "starts with q" — either the whole name or any of its words
        if n_up.startswith(q) or any(w.startswith(q) for w in n_up.split()):
            prefix_n.append((t, n))
            continue
        if q in t or (n_up and q in n_up):
            substring.append((t, n))

    # Cap each tier so the lower-priority tiers still get a chance to
    # show up when there's a flood of prefix matches (e.g. "A" matches
    # 80+ tickers; we still want "AAPL" near the top).
    ranked = exact + prefix_t[:limit] + prefix_n[:limit] + substring[:limit]
    return [{"symbol": t, "name": n} for t, n in ranked[:limit]]
