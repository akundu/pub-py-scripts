import argparse
import csv
import os
import asyncio
from datetime import datetime, timedelta

# Allow running as a script: python hdp/cli.py ...
# Ensure project root is on sys.path so absolute imports like `hdp.config` work
try:
	# If executed as a module, these imports will work
	from hdp.config import load_cities_config
except Exception:
	import sys
	_project_root = os.path.dirname(os.path.dirname(__file__))
	if _project_root not in sys.path:
		sys.path.insert(0, _project_root)
	from hdp.config import load_cities_config


def load_scraper(city_symbol: str):
	from hdp.scrapers import nyc_api, la_api, chicago_api, san_francisco_api, boston_api, seattle_api
	mapping = {
		"nyc": nyc_api.NYCAPIScraper,
		"la": la_api.LAAPIScraper,
		"chicago": chicago_api.ChicagoAPIScraper,
		"san_francisco": san_francisco_api.SFAPIScraper,
		"boston": boston_api.BostonAPIScraper,
		"seattle": seattle_api.SeattleAPIScraper,
	}
	cls = mapping.get(city_symbol)
	if not cls:
		raise SystemExit(f"Unsupported city: {city_symbol}")
	return cls


def _resolve_city_symbols(args: argparse.Namespace, cities_cfg):
	syms = []
	if getattr(args, "cities", None):
		for token in args.cities:
			for s in token.split(','):
				s = s.strip()
				if s:
					syms.append(s)
	if not syms:
		raise SystemExit("No cities provided. Use --cities with one or more symbols (e.g., --cities nyc or --cities nyc la).")
	for s in syms:
		if s not in cities_cfg:
			raise SystemExit(f"City config not found: {s}")
	return syms


def _normalize_dates(args: argparse.Namespace) -> None:
	fmt = "%Y-%m-%d"
	if getattr(args, "days_back", None) is not None:
		end = datetime.today().date()
		start = end - timedelta(days=int(args.days_back))
		args.start_date = start.strftime(fmt)
		args.end_date = end.strftime(fmt)
		return
	# else require provided
	if not getattr(args, "start_date", None) or not getattr(args, "end_date", None):
		raise SystemExit("Provide --days-back, or both --start-date and --end-date (YYYY-MM-DD).")


def cmd_scrape(args: argparse.Namespace) -> None:
	# Lazy imports to avoid requiring heavy deps for --help
	from hdp.db import ensure_db, get_conn, upsert_permit, insert_documents
	from hdp.storage import build_permit_dir, download_file, file_size

	cities_cfg = load_cities_config(os.path.join(os.path.dirname(__file__), "..", "config", "cities.json"))
	city_syms = _resolve_city_symbols(args, cities_cfg)
	ensure_db()
	for city_symbol in city_syms:
		city_cfg = cities_cfg[city_symbol]
		if args.delay is not None:
			city_cfg.rate_limit["delay_seconds"] = float(args.delay)
		Scraper = load_scraper(city_symbol)
		scraper = Scraper(city_cfg, headless=args.headless, debug=args.debug)
		count = 0
		with get_conn() as conn:
			for raw in scraper.search(args.start_date, args.end_date, limit=args.limit):
				permit, docs = scraper.parse_permit(raw)
				permit_id = upsert_permit(conn, permit)
				permit_dir = build_permit_dir(permit["city"], permit.get("issue_date"), permit.get("permit_type", ""), permit["permit_number"])
				stored_docs = []
				for d in docs:
					local = download_file(d.url, permit_dir, d.filename)
					if not local:
						continue
					stored_docs.append({
						"filename": os.path.basename(local),
						"file_ext": os.path.splitext(local)[1].lstrip('.'),
						"file_size_bytes": file_size(local),
						"url": d.url,
					})
				if stored_docs:
					insert_documents(conn, permit_id, stored_docs)
				if args.debug:
					print(f"[DEBUG] {city_symbol} permit={permit.get('permit_number')} dir={permit_dir} docs={len(stored_docs)}")
				count += 1
				if count >= args.limit:
					break
		print(f"Saved {count} permits for {city_symbol}")


async def async_cmd_scrape(args: argparse.Namespace) -> None:
	from hdp.db import ensure_db, upsert_permit_tx_async, insert_documents_tx_async
	from hdp.storage import build_permit_dir, download_file_async, file_size

	cities_cfg = load_cities_config(os.path.join(os.path.dirname(__file__), "..", "config", "cities.json"))
	city_syms = _resolve_city_symbols(args, cities_cfg)
	ensure_db()

	async def scrape_city(city_symbol: str):
		city_cfg = cities_cfg[city_symbol]
		if args.delay is not None:
			city_cfg.rate_limit["delay_seconds"] = float(args.delay)
		Scraper = load_scraper(city_symbol)
		scraper = Scraper(city_cfg, headless=args.headless, debug=args.debug)
		results = await scraper.async_search(args.start_date, args.end_date, limit=args.limit)
		sema = asyncio.Semaphore(args.max_concurrent_downloads)
		count = 0
		for raw in results:
			permit, docs = await scraper.async_parse_permit(raw)
			permit_id = await upsert_permit_tx_async(permit)
			permit_dir = build_permit_dir(permit["city"], permit.get("issue_date"), permit.get("permit_type", ""), permit["permit_number"])

			async def dl(d):
				async with sema:
					return await download_file_async(d.url, permit_dir, d.filename)

			paths = await asyncio.gather(*[dl(d) for d in docs]) if docs else []
			stored_docs = []
			for d, p in zip(docs, paths):
				if not p:
					continue
				stored_docs.append({
					"filename": os.path.basename(p),
					"file_ext": os.path.splitext(p)[1].lstrip('.'),
					"file_size_bytes": file_size(p),
					"url": d.url,
				})
			if stored_docs:
				await insert_documents_tx_async(permit_id, stored_docs)
			if args.debug:
				print(f"[DEBUG] {city_symbol} permit={permit.get('permit_number')} dir={permit_dir} docs={len(stored_docs)}")
			count += 1
			if count >= args.limit:
				break
		print(f"Saved {count} permits for {city_symbol}")

	# Run all requested cities concurrently
	await asyncio.gather(*[scrape_city(sym) for sym in city_syms])


def cmd_export(args: argparse.Namespace) -> None:
	# Lazy imports to avoid requiring dependencies for --help
	from hdp.db import ensure_db, get_conn
	ensure_db()
	with get_conn() as conn, open(args.out, "w", newline="", encoding="utf-8") as f:
		cur = conn.cursor()
		cur.execute("SELECT city, permit_number, issue_date, address, permit_type, project_value, contractor, details_json FROM permits")
		writer = csv.writer(f)
		writer.writerow(["city","permit_number","issue_date","address","permit_type","project_value","contractor","details_json"])
		for row in cur.fetchall():
			writer.writerow(row)
	print(f"Exported CSV to {args.out}")


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="hdp")
	sub = p.add_subparsers(dest="cmd", required=True)
	ps = sub.add_parser("scrape")
	ps.add_argument("--cities", nargs="*", help="One or more city symbols (e.g., nyc la chicago or nyc,la,chicago)")
	ps.add_argument("--start-date", help="YYYY-MM-DD")
	ps.add_argument("--end-date", help="YYYY-MM-DD")
	ps.add_argument("--days-back", dest="days_back", type=int, help="Use today as end-date and N days ago as start-date")
	ps.add_argument("--limit", type=int, default=100)
	ps.add_argument("--headless", action="store_true", default=False, help="Run browser headless")
	ps.add_argument("--delay", type=float, default=None, help="Override per-request delay seconds")
	ps.add_argument("--debug", action="store_true", default=False, help="Save debug HTML to data/debug/<city> and print save locations")
	# Make async default; provide --sync to opt out
	ps.add_argument("--sync", dest="use_async", action="store_false", help="Use synchronous mode")
	ps.add_argument("--max-concurrent-downloads", type=int, default=6)
	ps.set_defaults(func=None, use_async=True)

	pe = sub.add_parser("export")
	pe.add_argument("--out", required=True)
	pe.set_defaults(func=cmd_export)
	return p


def main():
	parser = build_parser()
	args = parser.parse_args()
	# Normalize dates from --days-back or ensure provided
	_normalize_dates(args)
	if args.cmd == "scrape":
		if args.use_async:
			asyncio.run(async_cmd_scrape(args))
		else:
			cmd_scrape(args)
	else:
		args.func(args)


if __name__ == "__main__":
	main()

