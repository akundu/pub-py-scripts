#!/bin/sh

 git pull origin; rm options.out.all-stt; python scripts/fetch_options.py --types all -stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier 2  --fetch-once-before-wait   > options.out.all-stt 2>&1

git pull origin; rm options.out.stt; python scripts/fetch_options.py --types stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --snapshot-max-concurrent 20 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier .5  --fetch-once-before-wait  > options.out.stt 2>&1

git pull origin; ulimit -n 65536; python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180 --workers 1 --enable-access-log


git pull origin; python scripts/fetch_options.py --use-csv --symbols I:SPX I:NDX --continuous --executor-type thread --snapshot-max-concurrent 30  --strike-range-percent 100 --refresh-threshold-seconds 30 --interval-multiplier .01  --data-dir ./csv_exports/  --days-ahead 1 --fetch-once-before-wait



exit





 python scripts/fetch_options.py --symbols SPX --continuous --executor-type thread --use-db $QUEST_DB_STRING  --snapshot-max-concurrent 30  --strike-range-percent 100 --refresh-threshold-seconds 30 --interval-multiplier .25


git pull origin; rm options.out.all-stt; python scripts/fetch_options.py --types all -stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier 2  --fetch-once-before-wait  > options.out.all-stt 2>&1


git pull origin; rm options.out.stt; python scripts/fetch_options.py --types stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --snapshot-max-concurrent 20 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier .5  --fetch-once-before-wait  > options.out.stt 2>&1


git pull origin; ulimit -n 65536; python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180 --workers 1 --enable-access-log > db_server.log 2>&1

git pull origin; rm options.out.all-stt; python scripts/fetch_options.py --types all -stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier 2  --fetch-once-before-wait


git pull origin; rm options.out.stt; python scripts/fetch_options.py --types stocks_to_track --continuous --months-ahead 9 --max-concurrent 15 --snapshot-max-concurrent 20 --executor-type process --use-db $QUEST_DB_STRING  --refresh-threshold-seconds 120 --interval-multiplier .5  --fetch-once-before-wait


python scripts/fetch_options.py --use-csv --symbols SPX NDX --continuous --executor-type thread --snapshot-max-concurrent 30  --strike-range-percent 100 --refresh-threshold-seconds 30 --interval-multiplier .01  --data-dir ./csv_exports/  --days-ahead 1 --fetch-once-before-wait


git pull origin; ulimit -n 65536; python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180 --workers 1 --enable-access-log > db_server.log 2>&1

