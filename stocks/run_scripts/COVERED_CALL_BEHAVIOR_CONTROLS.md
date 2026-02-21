# Covered Call Generation - Behavior Controls Documentation

This document lists all configuration constants, command-line arguments, and behavior-controlling elements in `covered_call_generation.py`.

## Configuration Constants (Top of File)

### Directory and Path Settings
- **`BASE_DIR`**: Automatically calculated as parent directory of script location
- **`TMP_DIR`**: `/tmp` - Temporary directory for intermediate files
- **`STORE_DIR_DEFAULT`**: `/Users/akundu/programs/http-proxy/static/` - Default directory for storing outputs
- **`OUTPUT_DIR_NAME`**: `"stocks_to_buy"` - Name of output directory
- **`ANALYSIS_FILE_BASENAME`**: `"analysis"` - Base name for analysis files
- **`DOWNLOAD_LOC_DEFAULT`**: `Path.home() / "Downloads" / "results.csv"` - Default location for results CSV
- **`GEMINI_PROG`**: `BASE_DIR / "tests" / "gemini_test.py"` - Path to Gemini test program
- **`QUERY_LOC`**: `"questdb://user:password@localhost:8812/stock_data"` - Default database connection string

### Symbol Selection Settings
- **`TYPE_FLAG`**: `"types"` - Default type flag for symbol selection (choices: "symbols" or "types")
- **`TYPE_INPUT`**: `"all"` - Default type input value

### Performance and Processing Settings
- **`MAX_WORKERS`**: `4` - Maximum number of parallel workers
- **`BATCH_SIZE`**: `300` - Batch size for processing
- **`MAX_DAYS`**: `30` - Maximum days to expiry for options
- **`POSITION_SIZE`**: `100000` - Position size in dollars ($100k)

### Gemini Analysis Settings
- **`GEMINI_COOLDOWN_SECONDS`**: `3600` - Cooldown period (1 hour) between Gemini runs
- **`LAST_GEMINI_RUN_FILE`**: `TMP_DIR / "covered_call_last_gemini_run_epoch"` - File tracking last Gemini run time
- **`GEMINI_INSTRUCTION`**: Uses `DEFAULT_GEMINI_INSTRUCTION` from `common.gemini_analysis` module

### Data Freshness Settings
- **`MARKET_HOURS_LOOKBACK_SECONDS`**: `3600` - Lookback period during market hours (1 hour)
- **`OFF_HOURS_LOOKBACK_SECONDS`**: `259200` - Lookback period outside market hours (72 hours / 3 days)

### Filter Thresholds
- **`MIN_PE`**: `1` - Minimum P/E ratio (not currently used in command building)
- **`MIN_VOL`**: `10` - Minimum volume filter
- **`MIN_LONG_PREMIUM`**: `0.5` - Minimum long premium (not currently used in command building)
- **`MIN_PREMIUM`**: `0.25` - Minimum premium (not currently used in command building)
- **`MIN_NET_PREMIUM`**: `1000` - Minimum net premium (not currently used in command building)
- **`CURR_PRICE_MULT`**: `1.01` - Current price multiplier (not currently used in command building)
- **`SENSIBLE_PRICE`**: `0.001` - Sensible price threshold for filtering invalid prices

### Spread Analysis Settings
- **`SPREAD_STRIKE_TOLERANCE`**: `5` - Strike price tolerance for spread matching ($5)
- **`SPREAD_LONG_DAYS`**: `120` - Target days to expiry for long leg of spread
- **`SPREAD_LONG_MIN_DAYS`**: `45` - Minimum days to expiry for long leg
- **`SPREAD_LONG_DAYS_TOLERANCE`**: `60` - Tolerance in days for long leg expiry matching

### Execution Control Settings
- **`REFRESH_RESULTS_BACKGROUND`**: `600` - Background refresh interval in seconds (not currently used)
- **`LOG_LEVEL`**: `"WARNING"` - Default log level
- **`OPTION_TYPE`**: `"both"` - Default option type (call, put, or both)
- **`SORT`**: `"potential_premium"` - Default sort field
- **`DEFAULT_SLEEP_SECONDS`**: `120` - Sleep interval during market hours (2 minutes)
- **`MAX_OFF_HOURS_SLEEP`**: `3600` - Maximum sleep outside market hours (1 hour)
- **`MIN_OFF_HOURS_SLEEP`**: `3600` - Minimum sleep outside market hours (1 hour)

## Command-Line Arguments

### Execution Mode Arguments
- **`--force-gemini`**: Force Gemini analysis to run (bypasses cooldown)
- **`--gemini-only`**: Run only Gemini analysis and exit
- **`--once`**: Run only once and exit (mutually exclusive with `--iterations`)
- **`--iterations N`**: Run N times and exit (mutually exclusive with `--once`)

### Gemini Configuration Arguments
- **`--gemini-input-file`**: Path to Gemini input CSV (default: `DOWNLOAD_LOC_DEFAULT`)
- **`--gemini-output-dir`**: Gemini output directory name (default: `OUTPUT_DIR_NAME`)
- **`--gemini-store-dir`**: Directory to store Gemini outputs (default: `STORE_DIR_DEFAULT`)

### Database and Cache Arguments
- **`--no-cache`**: Forward `--no-cache` flag to `options_analyzer.py`
- **`--db-server-host`**: Database server hostname for cache warmup (default: `"mm.kundu.dev"`)
- **`--db-server-port`**: Database server port for cache warmup (default: `9100`)
- **`--db-conn`**: Database connection string (default: `QUERY_LOC`)

### Output Arguments
- **`--output-file`**: Path to output CSV file (default: `DOWNLOAD_LOC_DEFAULT`)
- **`--html-output-dir`**: Directory for HTML output (default: `TMP_DIR / OUTPUT_DIR_NAME`)

### Options Analyzer Arguments
These arguments are forwarded to `options_analyzer.py`:

- **`--type-flag`**: Type flag: 'symbols' or 'types' (default: `TYPE_FLAG`)
- **`--type-input`**: Type input value (default: `TYPE_INPUT`)
- **`--max-days`**: Maximum days to expiry (default: `MAX_DAYS`)
- **`--batch-size`**: Batch size (default: `BATCH_SIZE`)
- **`--max-workers`**: Maximum workers (default: `MAX_WORKERS`)
- **`--position-size`**: Position size (default: `POSITION_SIZE`)
- **`--no-spread`**: Disable spread analysis (default: spread enabled)
- **`--spread-strike-tolerance`**: Spread strike tolerance (default: `SPREAD_STRIKE_TOLERANCE`)
- **`--spread-long-days`**: Spread long days (default: `SPREAD_LONG_DAYS`)
- **`--spread-long-min-days`**: Spread long minimum days (default: `SPREAD_LONG_MIN_DAYS`)
- **`--spread-long-days-tolerance`**: Spread long days tolerance (default: `SPREAD_LONG_DAYS_TOLERANCE`)
- **`--log-level`**: Log level (default: `LOG_LEVEL`)
- **`--option-type`**: Option type: call, put, or both (default: `OPTION_TYPE`)
- **`--sensible-price`**: Sensible price threshold (default: `SENSIBLE_PRICE`)
- **`--min-vol`**: Minimum volume filter (default: `MIN_VOL`)
- **`--sort`**: Sort field (default: `SORT`)
- **`--top-n`**: Top N results (default: `5`)
- **`--no-stats`**: Disable stats output (default: stats enabled)
- **`--market-hours-lookback-seconds`**: Seconds to look back during market hours (default: `MARKET_HOURS_LOOKBACK_SECONDS`)
- **`--off-hours-lookback-seconds`**: Seconds to look back outside market hours (default: `OFF_HOURS_LOOKBACK_SECONDS`)

## Runtime Behavior Controls

### Loop Execution Behavior
1. **Iteration Control**: 
   - If `--once` is set: runs exactly 1 iteration
   - If `--iterations N` is set: runs exactly N iterations
   - Otherwise: runs continuously until interrupted

2. **Market Hours Detection**:
   - Uses `common_is_market_hours()` to detect if market is open
   - Sleep behavior differs based on market status

3. **Sleep Intervals**:
   - **During Market Hours**: Sleeps for `DEFAULT_SLEEP_SECONDS` (120 seconds / 2 minutes)
   - **Outside Market Hours**: 
     - Calculates time until next market open using `compute_market_transition_times()`
     - Sleeps for `min(MAX_OFF_HOURS_SLEEP, seconds_to_open)`
     - Minimum sleep is `MAX_OFF_HOURS_SLEEP` (3600 seconds / 1 hour)
     - Maximum sleep is `MAX_OFF_HOURS_SLEEP` (3600 seconds / 1 hour)

4. **Cache Warmup Behavior**:
   - After successful `options_analyzer.py` execution:
     - Loads results CSV
     - Cleans duplicate header rows
     - Calculates TTL as `sleep_time / 2.0` (half of sleep interval)
     - Initiates fire-and-forget cache warmup (non-blocking)
     - Uses `warmup_stock_info_cache()` with `wait_timeout=None`

5. **Gemini Analysis Behavior**:
   - Only runs if `--gemini-only` is set OR if `--force-gemini` is set (bypasses cooldown)
   - Respects cooldown period (`GEMINI_COOLDOWN_SECONDS`)
   - Checks `LAST_GEMINI_RUN_FILE` to determine if cooldown has elapsed
   - Market hours status affects Gemini execution (passed to `run_gemini_analysis()`)

6. **Command Building**:
   - Uses `build_options_analyzer_command()` to construct command for `options_analyzer.py`
   - Dynamically builds arguments based on argparse definitions from `common.options.options_args`
   - Handles boolean flags, append actions, and regular arguments appropriately

## Notes

- **Commented Out Code**: Lines 466-508 contain commented-out code that previously:
  - Generated HTML output via `evaluate_covered_calls.py`
  - Ran Gemini analysis after each iteration
  - Copied analysis files to store directory
  - This functionality is currently disabled

- **Error Handling**: 
  - Cache warmup errors are non-fatal (logged but don't stop execution)
  - Gemini analysis errors are logged but execution continues
  - File operations use try/except blocks for graceful failure

- **Dependencies**:
  - Requires `common.market_hours` for market hours detection
  - Requires `common.gemini_analysis` for Gemini functionality
  - Requires `common.cache_warmup` for cache warming
  - Requires `common.options.options_args` for command building
  - Requires `common.symbol_loader` for symbol argument handling




