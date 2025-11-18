#!/bin/sh
ulimit -n 65536; python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9100 --log-level INFO --heartbeat-interval 180 --workers 2 --enable-access-log
ulimit -n 65536; python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9101 --log-level INFO --heartbeat-interval 180 --workers 3 --enable-access-log
ulimit -n 65536; python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9102 --log-level INFO --heartbeat-interval 180 --workers 1 --enable-access-log


ulimit -n 65536; python fetch_all_data.py --types all --max-concurrent 3 --db-path localhost:9002 --executor-type thread --stock-executor-type thread --client-timeout 180 --use-market-hours --fetch-ratios --continuous
ulimit -n 65536; python  scripts/polygon_realtime_streamer.py --types stocks_to_track --feed quotes --db-server localhost:9102 --log-level ERROR --symbols-per-connection 10

python scripts/historical_stock_options.py --types stocks_to_track --strike-range-percent 20 --max-days-to-expiry 30 --max-concurrent 8 --executor-type thread --snapshot-max-concurrent 4 --use-csv --db-path localhost:9100  --fetch-online --continuous --interval-multiplier 0.25
python scripts/historical_stock_options.py --types all --strike-range-percent 10 --max-days-to-expiry 30 --max-concurrent 4 --executor-type thread --snapshot-max-concurrent 4 --use-csv --db-path localhost:9102 --fetch-online --continuous --interval-multiplier 1



##Python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9001 --log-level INFO --heartbeat-interval 180 --workers 1
#Python db_server.py --db-file questdb://stock_user:stock_password@localhost:8812/stock_data --port 9002 --log-level ERROR --heartbeat-interval 60 --workers 4 --questdb-connection-timeout 180
#/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python scripts/polygon_realtime_streamer.py --symbols AAPL MSFT CART TQQQ UBER QQQ NFLX NVDA TSLA SPY AMZN GOOG --feed quotes --db-server localhost:9001 --log-level ERROR --batch-interval 1

 #/opt/homebrew/Cellar/python@3.13/3.13.5/Frameworks/Python.framework/Versions/3.13/Resources/Python.app/Contents/MacOS/Python fetch_all_data.py --types all --max-concurrent 5 --db-path localhost:9002 --executor-type process --stock-executor-type process --client-timeout 180 --latest --use-market-hours --continuous

#ulimit -n  65536;  python scripts/stock_display_dashboard.py --symbols AMZN GOOG NFLX CART UBER TQQQ QQQ SPY NVDA TSLA AAPL --db-server localhost:9001 --log-level ERROR;
