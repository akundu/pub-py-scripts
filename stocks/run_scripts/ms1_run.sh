#rm db_server.log ;  ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-lev#el WARNING --heartbeat-interval 180  --enable-access-log
#ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9102 --log-level WARNING --heartbe#at-interval 180 --workers 2 --enable-access-log
#ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9102 --log-level WARNING --heartbe#at-interval 180 --workers 2 --enable-access-log
#ulimit -n  65536;  python scripts/stock_display_dashboard.py --symbols AMZN GOOG NFLX CART UBER TQQQ QQQ SPY NVDA TSLA AAPL VOO MU AVGO TSM HOOD --db-server localhost:9102 --debug



ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level INFO --heartbeat-interval 180 --workers 4 --enable-access-log 2>&1 |tee db_server.log
 ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9102 --log-level WARNING --heartbeat-interval 180 --workers 2 --enable-access-log
  ulimit -n  65536;  python scripts/stock_display_dashboard.py --symbols AMZN GOOG NFLX CART UBER TQQQ QQQ SPY NVDA MU AVGO TSM I:VIX1D I:SPX I:NDX --db-server localhost:9102 --debug




python ./run_scripts/covered_call_generation.py --db-conn $QUEST_DB_STRING --type-flag types --type-input all --spread-long-days 180 --spread-long-days-tolerance 30 --max-workers 4 --top-n 50 --output-file ~/Downloads/results.csv --min-vol 25 --market-hours-lookback-seconds 1800 --max-bid-ask-spread 1.5 --max-bid-ask-spread-long 1.5 --db-server-host mm1.kundu.dev


python scripts/option_spread_watcher.py --rules  results/backtest_tight/grid_analysis_tight_successful.csv  --data-dir csv_exports/options/ --max-workers 4 --max-spend 40000 --verbose --db $QUEST_DB_STRING --log-level ERROR --interval 30 --max-spread-width 200


python scripts/fetch_options.py --force-fetch --use-csv --symbols I:SPX I:NDX --force-fetch --continuous --executor-type thread --snapshot-max-concurrent 30  --strike-range-percent 100 --refresh-threshold-seconds 15 --interval-multiplier .01  --data-dir ./csv_exports/  --days-ahead 10 --fetch-once-before-wait --use-market-hours

while true; do date ; python fetch_symbol_data.py NDX --latest --db-path $QUEST_DB_STRING --latest --timezone PST; sleep 10; done



