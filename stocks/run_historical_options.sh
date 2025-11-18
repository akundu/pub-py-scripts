#!/bin/sh

python scripts/historical_stock_options.py --types stocks_to_track --strike-range-percent 20 --max-days-to-expiry 90 --max-concurrent 3 --executor-type thread --snapshot-max-concurrent 4 --db-path localhost:9101 --fetch-online --continuous --interval-multiplier 0.25
pthon scripts/historical_stock_options.py --types all --strike-range-percent 30 --max-days-to-expiry 120 --max-concurrent 10 --executor-type thread --snapshot-max-concurrent 5 --db-path localhost:9100 --fetch-online --continuous --interval-multiplier 3 --fetch-once-before-wait
python scripts/historical_stock_options.py --types all --strike-range-percent 10 --max-days-to-expiry 90 --max-concurrent 10 --executor-type thread --snapshot-max-concurrent 5 --db-path localhost:9100 --fetch-online --continuous --interval-multiplier 0.5
python scripts/historical_stock_options.py --types stocks_to_track --strike-range-percent 20 --max-days-to-expiry 30 --max-concurrent 8 --executor-type thread --snapshot-max-concurrent 4 --db-path localhost:9101 --fetch-online --continuous --interval-multiplier 0.25

