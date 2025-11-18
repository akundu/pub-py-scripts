# Commit History for Currently Modified Files

This document contains the complete commit history for all files that are currently modified in the repository.

## Modified Files

1. `common/common.py`
2. `common/options/options_workers.py`
3. `common/options_utils.py`
4. `common/questdb_db.py`
5. `common/symbol_loader.py`
6. `data/lists/etfs_symbols.yaml`
7. `data/lists/nasdaq_symbols.yaml`
8. `data/lists/nyse_symbols.yaml`
9. `db_server.py`
10. `fetch_all_data.py`
11. `fetch_lists_data.py`
12. `run_scripts/covered_call_generation.sh`
13. `scripts/fetch_options.py`
14. `scripts/options_analyzer.py`
15. `scripts/stocks_analyzer.py`

---

## Commit History by File

### common/common.py
- `e842cd9` Add sensible-price filter and premium-based spread filtering
- `44dad2b` Refactor options_analyzer.py: modularize and deduplicate code
- `0785329` refactor(common): extract reusable timestamp handling functions
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `b34dcdc` better support for parallelization

### common/options/options_workers.py
- `e842cd9` Add sensible-price filter and premium-based spread filtering
- `40cf627` Refactor options_analyzer.py: modularize, deduplicate, and reorganize
- `44dad2b` Refactor options_analyzer.py: modularize and deduplicate code

### common/options_utils.py
- `8d193af` feat: Implement HTML Report Generator V2 with modular architecture
- `44dad2b` Refactor options_analyzer.py: modularize and deduplicate code

### common/questdb_db.py
- `7e15ae2` fix(options_analyzer): correct price change calculation using date-based previous close
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `83628c4` updates on timing handling
- `2f8dac9` better support for bid/ask
- `22d11f5` better timezone handling
- `cb61a84` handling the deprecated warning issues
- `c25db32` reduce the necessity to scan for options tickers for expiration dates & support the ability to cache those
- `eadbacc` cleanup
- `1e991f1` combine shared cache stats into one placek
- `de59fa3` cleaner version of redis v0.1
- `dd85e30` intermediate checkin
- `8dfcb30` redid redis support
- `7e8159e` updates for the first version of redis support
- `3e91aa3` handle deprecations
- `823cdb2` first version of backtesting
- `7ae9683` multiprocess mode support
- `b44cc89` support to show latest price based off of market hours
- `7a2d3db` remove prepared statements
- `bf74e7f` support for access log & remote options save
- `566fa40` first version of options analyzer and supporting financial info
- `7b95077` support for faster table creation & volume & volatility indication for options
- `64e4190` support for db store and retrieve in options mgmt
- `30502f4` first version of interfaces to accept option saves
- `4e4411a` cleanup
- `ba5e465` timezone updates
- `fd6870d` Add QuestDB connection string support and fix write_timestamp issues
- `3447156` clean up db setup on duplicates & setup the latest option
- `10c55f0` optimize  create scripts
- `a39c19b` support for volume & get rid of on_duplicate concerns & optimize insertions of it
- `70f1757` cleanup

### common/symbol_loader.py
- `88c7bd8` feat: enhance covered-call automation and clean up docs
- `7b6a00a` chore: update various files with minor improvements and data updates
- `f2a943c` first version of combined list fetching

### data/lists/etfs_symbols.yaml
- `7b6a00a` chore: update various files with minor improvements and data updates
- `50ee783` update to lists
- `87e3700` list updates
- `566fa40` first version of options analyzer and supporting financial info
- `29e2c0a` list update
- `57922e1` proper batching support
- `b6e2798` removed the last update time
- `4fa8a46` update to symbols present
- `d7d8cbc` update of symbols
- `3ff8817` restructuring
- `8e7d306` support the ability to fetch using pagination
- `07d2fc9` first version of stock fetch

### data/lists/nasdaq_symbols.yaml
- `7b6a00a` chore: update various files with minor improvements and data updates
- `128de85` cleanups
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `71c229b` list update
- `20fa06a` lists updated
- `2f8dac9` better support for bid/ask
- `8dfcb30` redid redis support
- `7e8159e` updates for the first version of redis support
- `50ee783` update to lists
- `87e3700` list updates
- `b44cc89` support to show latest price based off of market hours
- `22ff9ac` updated lists
- `cc6e7fd` update the lists
- `bf74e7f` support for access log & remote options save
- `d5d7923` update the lists & cleanup messaging for help
- `de38986` update to symbols present
- `566fa40` first version of options analyzer and supporting financial info
- `29e2c0a` list update
- `57922e1` proper batching support
- `b6e2798` removed the last update time
- `4fa8a46` update to symbols present
- `d7d8cbc` update of symbols
- `3ff8817` restructuring
- `8e7d306` support the ability to fetch using pagination
- `07d2fc9` first version of stock fetch

### data/lists/nyse_symbols.yaml
- `7b6a00a` chore: update various files with minor improvements and data updates
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `71c229b` list update
- `20fa06a` lists updated
- `2f8dac9` better support for bid/ask
- `e319913` updated lists
- `8dfcb30` redid redis support
- `7e8159e` updates for the first version of redis support
- `50ee783` update to lists
- `87e3700` list updates
- `b44cc89` support to show latest price based off of market hours
- `22ff9ac` updated lists
- `cc6e7fd` update the lists
- `bf74e7f` support for access log & remote options save
- `d5d7923` update the lists & cleanup messaging for help
- `de38986` update to symbols present
- `566fa40` first version of options analyzer and supporting financial info
- `29e2c0a` list update
- `57922e1` proper batching support
- `b6e2798` removed the last update time
- `4fa8a46` update to symbols present
- `d7d8cbc` update of symbols
- `3ff8817` restructuring
- `8e7d306` support the ability to fetch using pagination
- `07d2fc9` first version of stock fetch

### db_server.py
- `7b6a00a` chore: update various files with minor improvements and data updates
- `128de85` cleanups
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `8dfcb30` redid redis support
- `7e8159e` updates for the first version of redis support
- `bf74e7f` support for access log & remote options save
- `566fa40` first version of options analyzer and supporting financial info
- `a39c19b` support for volume & get rid of on_duplicate concerns & optimize insertions of it
- `5ab347e` provide vol data
- `df071e7` feat: enhance real-time stock dashboard and database server capabilities
- `672bc5e` added the realtime  tracking ability
- `489537b` db_server.py - Replace multiprocessing-based manager with native fork() model (ForkingServer) - Bind socket in parent, share across children; ignore SIGPIPE - Parent handles SIGINT/SIGTERM, forwards SIGTERM to children; children handle SIGTERM - Monitor children via waitpid(WNOHANG) and resurrect with backoff (3 deaths/60s => 30s wait) - Stagger child creation; add startup delay after bind - Process-safe PID-prefixed logging via QueueListener/QueueHandler - CLI: add --startup-delay, --child-stagger-ms, --bind-retries, --bind-retry-delay-ms - Remove MultiProcessServer/worker_main and concurrent.futures/threading usage - Use os.cpu_count() for auto workers; keep multiprocessing only for logging Queue - Ensure parent exits cleanly after children on SIGINT
- `a6f58ee` first version of quest_db integration
- `11ff440` Fix ticker previous close prices and add opening price column
- `5d04b35` first version of tsdb support
- `cfafa4e` fix: ensure PostgreSQL database respects command-line log level
- `e672217` feat: add multi-process server with socket sharing
- `6cc9e02` provide apis to get access to db related information & fix pool operations
- `f19b1da` feat: add postgres capabilities
- `a9f33c9` feat: improve Docker container data access and networking
- `57922e1` proper batching support
- `4ab3afd` support for large dumps on remote setups
- `491047d` support for large objects
- `94372a5` add support to  broadcast realtime data
- `62c5e2e` add a websocket handler w/ a heartbeat to feed data outwards for changes being saved to the server
- `3ff8817` restructuring
- `aaa7bb8` move stock_db to common
- `36061f3` support for executing sql
- `39c36a8` first version of execute query
- `6a486f0` add logging

### fetch_all_data.py
- `88c7bd8` feat: enhance covered-call automation and clean up docs
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `b34dcdc` better support for parallelization
- `8dfcb30` redid redis support
- `026f9ff` refetch interval fix
- `bf74e7f` support for access log & remote options save
- `87564e5` support for increasing time to sync on market hours by multipliers
- `566fa40` first version of options analyzer and supporting financial info
- `446f415` support for continuous fetching in historical stock
- `9212937` fetch timing based on current market hrs
- `f2a943c` first version of combined list fetching
- `1200286` clean up fetch_all
- `fd6870d` Add QuestDB connection string support and fix write_timestamp issues
- `3447156` clean up db setup on duplicates & setup the latest option
- `672bc5e` added the realtime  tracking ability
- `489537b` db_server.py - Replace multiprocessing-based manager with native fork() model (ForkingServer) - Bind socket in parent, share across children; ignore SIGPIPE - Parent handles SIGINT/SIGTERM, forwards SIGTERM to children; children handle SIGTERM - Monitor children via waitpid(WNOHANG) and resurrect with backoff (3 deaths/60s => 30s wait) - Stagger child creation; add startup delay after bind - Process-safe PID-prefixed logging via QueueListener/QueueHandler - CLI: add --startup-delay, --child-stagger-ms, --bind-retries, --bind-retry-delay-ms - Remove MultiProcessServer/worker_main and concurrent.futures/threading usage - Use os.cpu_count() for auto workers; keep multiprocessing only for logging Queue - Ensure parent exits cleanly after children on SIGINT
- `f19b1da` feat: add postgres capabilities
- `0f9f059` cleanup
- `feeb2b8` import cleanups
- `e8095f9` change the default
- `f29ee95` further cleanups to merge process & threads and current price & historical data
- `36cc590` fix: Resolve executor and task handling issues in fetch_all_data.py
- `3ed76aa` feat: Add current price functionality with timestamp freshness check
- `57922e1` proper batching support
- `27dd206` support the ability to get multiple store forms at the same time independent of the stocks being downloaded then
- `971d77b` support for splitting out process into a different section
- `472b5ac` support for splitting out process into a different section
- `a2548f0` support for large dumps on remote setups
- `41871c3` remote support
- `e9481d1` Feat(data_fetch): Add configurable time intervals for historical data

### fetch_lists_data.py
- `7b6a00a` chore: update various files with minor improvements and data updates
- `128de85` cleanups
- `566fa40` first version of options analyzer and supporting financial info
- `36cc590` fix: Resolve executor and task handling issues in fetch_all_data.py
- `8ddaeb7` removed the last update time
- `3ff8817` restructuring
- `1946642` support date ranges correctly

### run_scripts/covered_call_generation.sh
- `88c7bd8` feat: enhance covered-call automation and clean up docs
- `e842cd9` Add sensible-price filter and premium-based spread filtering
- `7b6a00a` chore: update various files with minor improvements and data updates
- `ff9cfc7` add the ability to have different options test interval
- `3a4db23` support the first version of grouping cols
- `59d04e5` update time for 2nd 1/2 of day
- `128de85` cleanups
- `54db24f` cleanups & updating values
- `ffc4e58` updated parameters
- `c7fb27c` updated parameters
- `075119e` Update covered call generation script
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `c58a838` stronger support for filter
- `92c066c` cleanups
- `0fca9f8` cleanups
- `7ca0047` file rename
- `6cf4c28` generate covered calls to buy

### scripts/fetch_options.py
- `f3b7b42` add the --fetch-fresh option
- `7b6a00a` chore: update various files with minor improvements and data updates
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `b34dcdc` better support for parallelization
- `2f8dac9` better support for bid/ask
- `5f6b88b` create more ways to fetch IV data
- `8dfcb30` redid redis support
- `26529ea` handling out of market open fetch
- `2e1b75b` sleep time on continuous fixes
- `207eca5` sleep time on continuous fixes
- `bf74e7f` support for access log & remote options save
- `847446e` support for market hours
- `7b95077` support for faster table creation & volume & volatility indication for options
- `c2788e3` clean up command line parsing
- `64e4190` support for db store and retrieve in options mgmt
- `446f415` support for continuous fetching in historical stock
- `f2a943c` first version of combined list fetching
- `4596e10` support for csv caching
- `dc24df7` get the prices of the options
- `7e33666` print the option name
- `d5c1f28` cleanup and reviewing all options inputs
- `808cd9f` cleanups
- `ea57fb6` Introduces a new script, historical_stock_options.py, designed to fetch and analyze stock and options data for a specific historical date using the Polygon.io API.

### scripts/options_analyzer.py
- `e842cd9` Add sensible-price filter and premium-based spread filtering
- `40cf627` Refactor options_analyzer.py: modularize, deduplicate, and reorganize
- `44dad2b` Refactor options_analyzer.py: modularize and deduplicate code
- `7e15ae2` fix(options_analyzer): correct price change calculation using date-based previous close
- `6ee928a` support for refreshing if necessary and cleanups
- `128de85` cleanups
- `26b81aa` Update core modules: db_server, questdb_db, fetch scripts, and evaluation tools
- `b4451b5` added a batch size support for multiprocess mode
- `b34dcdc` better support for parallelization
- `883e50c` multiprocess cleanup
- `83628c4` updates on timing handling
- `2f8dac9` better support for bid/ask
- `d6fe703` make all long fields filterable
- `12d4f0b` black scholes formula support
- `417a5a9` cleansup & simultaneous fetch of spread data
- `1e991f1` combine shared cache stats into one placek
- `de59fa3` cleaner version of redis v0.1
- `8dfcb30` redid redis support
- `7e8159e` updates for the first version of redis support
- `e0786eb` git commit -m "Add calendar spread analysis with long-option-based position sizing
- `7ae9683` multiprocess mode support
- `a6f191e` feat: add performance optimization parameters to options analyzer - Add max_concurrent, batch_size, timestamp_lookback_days, and max_workers parameters to analyze_covered_calls method - Enable memory-efficient batch processing with configurable concurrency limits - Support both asyncio-only and hybrid asyncio+multiprocessing modes - Add CLI arguments for timestamp_lookback_days and max_workers - Include multiprocess statistics reporting when using multiple workers
- `96b6331` end and start date support
- `d06e1f8` csv output feature
- `bff9ed7` upda†e premium computation
- `bf74e7f` support for access log & remote options save
- `d5d7923` update the lists & cleanup messaging for help
- `566fa40` first version of options analyzer and supporting financial info

### scripts/stocks_analyzer.py
- `a023bd0` cleanups
- `8dfcb30` redid redis support
- `50ee783` update to lists
- `a6f191e` feat: add performance optimization parameters to options analyzer - Add max_concurrent, batch_size, timestamp_lookback_days, and max_workers parameters to analyze_covered_calls method - Enable memory-efficient batch processing with configurable concurrency limits - Support both asyncio-only and hybrid asyncio+multiprocessing modes - Add CLI arguments for timestamp_lookback_days and max_workers - Include multiprocess statistics reporting when using multiple workers
- `bea2cac` first version of stocks analyzer

---

## Summary

Total modified files: **15**

These files have been modified and are ready to be committed. Use the `commit_modified_files.sh` script to commit all these files at once, or review the individual changes first with `git diff`.


