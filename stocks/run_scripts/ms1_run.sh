rm db_server.log ;  ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180  --enable-access-log
ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9102 --log-level WARNING --heartbeat-interval 180 --workers 2 --enable-access-log
ulimit -n 65536;  python db_server.py --db-file $QUEST_DB_STRING --port 9102 --log-level WARNING --heartbeat-interval 180 --workers 2 --enable-access-log
ulimit -n  65536;  python scripts/stock_display_dashboard.py --symbols AMZN GOOG NFLX CART UBER TQQQ QQQ SPY NVDA TSLA AAPL VOO MU AVGO TSM HOOD --db-server localhost:9102 --debug
