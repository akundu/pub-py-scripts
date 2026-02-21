# HD Permits Scraper

Automates downloading public building permits and associated documents (PDFs/images) from multiple municipal portals, and stores normalized metadata locally in SQLite and CSV.

## Features
- Selenium + BeautifulSoup scraper engine with respectful rate limiting
- City-specific scrapers for NYC, Los Angeles, Chicago, plus extensible extras
- Automatic document downloading with organized storage layout
- SQLite metadata database and CSV export
- JSON configuration for endpoints and parameters

## Quick Start

1) Install dependencies (Python 3.9+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure cities in `config/cities.json`.

3) Run a scrape:

```bash
python -m hdp.cli scrape --city nyc --start-date 2024-01-01 --end-date 2024-01-31 --limit 200
```

4) Export CSV:

```bash
python -m hdp.cli export --city nyc --out data/exports/nyc_permits.csv
```

## Storage Layout
- Base data directory: `data/`
- Files saved under: `data/{city}/{YYYY-MM-DD}/{permit_type}/{permit_number}/...`

## Notes
- Headless Chrome is used by default. You may need to install ChromeDriver compatible with your Chrome version.
- Respect robots.txt and site terms. Use `--delay` and `--max-retries` to be polite.

