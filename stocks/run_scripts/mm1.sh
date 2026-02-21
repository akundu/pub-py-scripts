#!/bin/sh
git pull origin; rm stocks.out ; python fetch_all_data.py --types all --max-concurrent 5 --executor-type process --stock-executor-type process --client-timeout 180 --use-market-hours --data-source polygon  --db-path $QUEST_DB_STRING --continuous --interval-multiplier 3  > stocks.out 2>&1 #--fetch-once-before-wait

git pull origin; while true ; do rm out; ulimit -n 65536; date; time python fetch_all_data.py --types all --max-concurrent 15 --executor-type process --stock-executor-type process --client-timeout 180 --fetch-market-data  --fetch-ratios --data-source polygon  --db-path $QUEST_DB_STRING --start-date  $(date -v-3d +%Y-%m-%d) --end-date $(date -v-0d +%Y-%m-%d)  2>&1 | tee out;  date; sleep 43200; done


 git pull origin; ulimit -n 65536; python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180 --workers 3 --enable-access-log > db_server.log 2>&1 

git pull origin; ulimit -n 65536; python  scripts/polygon_realtime_streamer.py --types all --feed quotes --db-path $QUEST_DB_STRING --symbols-per-connection 30 --log-level ERROR --batch-interval 1


git pull origin; python  scripts/polygon_realtime_streamer.py --symbols I:VIX I:VIX1D I:SPX I:NDX --feed quotes --redis-url redis://lin1_a.kundu.dev:6379 --symbols-per-connection 20 --batch-interval 1 --feed both --poll-only --poll-interval 5 --db-path $QUEST_DB_STRING --use-market-hours

