#!/bin/sh
 git pull origin; ulimit -n 65536; python db_server.py --db-file $QUEST_DB_STRING --port 9100 --log-level WARNING --heartbeat-interval 180 --workers 2 --enable-access-log > db_server.log 2>&1 
