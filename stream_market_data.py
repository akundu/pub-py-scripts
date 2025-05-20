import argparse
import asyncio
import os
import pandas as pd
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.enums import DataFeed
from stock_db import get_stock_db, StockDBBase, get_default_db_path
import sys
import traceback

# Global variable to hold the DB client session, to be closed on exit
# This is a simple way; for more complex apps, pass it around or use a context manager.
_db_client_instance_for_cleanup: StockDBBase | None = None

async def trade_data_handler(data, stock_db_instance: StockDBBase | None, symbol_type: str, only_log_updates: bool):
    """Asynchronous handler to process and optionally save incoming trade data."""
    # For trades, every trade is an update, so only_log_updates might not be as relevant
    # unless we compare against a last trade price, but instructions are to save trades.
    print(f"Trade for {data.symbol} ({symbol_type}): Price - {data.price}, Size - {data.size} at {data.timestamp}")
    if stock_db_instance:
        try:
            df_data = {
                'timestamp': [pd.to_datetime(data.timestamp, utc=True)],
                'price': [data.price],
                'size': [data.size]
                # 'ask_price', 'ask_size' will be None for trades in DB schema
            }
            trade_df = pd.DataFrame(df_data).set_index('timestamp')
            # Use asyncio.create_task if save_realtime_data is truly async and non-blocking
            # If it involves significant CPU or sync I/O not handled by run_in_executor internally,
            # consider asyncio.to_thread for the client call as well if it becomes a bottleneck.
            # For now, assuming StockDBClient._make_request is efficiently async.
            await stock_db_instance.save_realtime_data(trade_df, data.symbol, data_type="trade")
        except Exception as e:
            print(f"Error saving trade data for {data.symbol} to DB: {e}")
            # traceback.print_exc()

# Closure for quote data handler to manage last_prices
_last_quote_prices = {} # Dictionary to store last prices per symbol {symbol: {bid_price: X, ask_price: Y}}

async def quote_data_handler(data, stock_db_instance: StockDBBase | None, symbol_type: str, only_log_updates: bool):
    """Asynchronous handler to process and optionally save incoming quote data."""
    global _last_quote_prices
    symbol = data.symbol

    current_prices = {
        'bid_price': data.bid_price,
        'ask_price': data.ask_price
    }
    last_symbol_prices = _last_quote_prices.get(symbol, {'bid_price': None, 'ask_price': None})

    prices_have_changed = False
    if current_prices['bid_price'] != last_symbol_prices['bid_price'] or \
       current_prices['ask_price'] != last_symbol_prices['ask_price']:
        prices_have_changed = True

    if not only_log_updates or prices_have_changed:
        print(f"Quote for {symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {data.timestamp}")
        if stock_db_instance:
            try:
                df_data = {
                    'timestamp': [pd.to_datetime(data.timestamp, utc=True)],
                    'price': [data.bid_price], # Using bid_price as the primary 'price' for quotes
                    'size': [data.bid_size],
                    'ask_price': [data.ask_price],
                    'ask_size': [data.ask_size]
                }
                quote_df = pd.DataFrame(df_data).set_index('timestamp')
                await stock_db_instance.save_realtime_data(quote_df, symbol, data_type="quote")
            except Exception as e:
                print(f"Error saving quote data for {symbol} to DB: {e}")
                # traceback.print_exc()
    
    _last_quote_prices[symbol] = current_prices

def setup_and_run_stream(stock_db_instance: StockDBBase | None, symbols: list[str], feed: str, inferred_symbol_type: str, only_log_updates: bool):
    """Sets up and runs the WebSocket stream for a given list of symbols and type."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    wss_client = None

    if not symbols:
        print("No symbols provided to stream.", file=sys.stderr)
        return

    # API Key checks based on inferred_symbol_type
    if inferred_symbol_type == "stock":
        if not api_key or not secret_key:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set for stock data.", file=sys.stderr)
            return # Exit if keys missing for stocks
    elif inferred_symbol_type == "crypto":
        if not api_key or not secret_key:
            print("Info: API keys not found or incomplete. Crypto data can be streamed without keys but with lower rate limits.", file=sys.stderr)

    symbols_str = ", ".join(symbols)
    print(f"Attempting to stream {feed} for {inferred_symbol_type} symbols: {symbols_str}", file=sys.stderr)

    # Define handlers within the scope where stock_db_instance etc. are available
    async def internal_quote_handler(data):
        await quote_data_handler(data, stock_db_instance, inferred_symbol_type, only_log_updates)

    async def internal_trade_handler(data):
        await trade_data_handler(data, stock_db_instance, inferred_symbol_type, only_log_updates)

    try:
        if inferred_symbol_type == "stock":
            wss_client = StockDataStream(api_key, secret_key, feed=DataFeed.SIP) # feed="iex" for free stock data
            if feed == "quotes":
                print(f"Subscribing to stock quotes for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to stock trades for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        elif inferred_symbol_type == "crypto":
            # CryptoDataStream can work without keys (for some exchanges/data)
            wss_client = CryptoDataStream(api_key, secret_key) if api_key and secret_key else CryptoDataStream()
            if feed == "quotes":
                print(f"Subscribing to crypto quotes for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_quotes(internal_quote_handler, *symbols)
            elif feed == "trades":
                print(f"Subscribing to crypto trades for: {symbols_str}", file=sys.stderr)
                wss_client.subscribe_trades(internal_trade_handler, *symbols)
        else:
            print(f"Internal error: Unsupported inferred symbol type: {inferred_symbol_type}", file=sys.stderr)
            return

        if wss_client:
            print(f"Starting WebSocket client for {inferred_symbol_type} symbols: {symbols_str}...", file=sys.stderr)
            wss_client.run() # This is a blocking call
        else:
            # This path should ideally not be hit if logic above is correct
            print(f"WebSocket client for {inferred_symbol_type} was not initialized. This might be due to missing API keys for stock data or an internal error.", file=sys.stderr)

    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in {inferred_symbol_type} stream for {symbols_str}. Stopping stream...", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during {inferred_symbol_type} stream operation for {symbols_str}: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        print(f"Stream processing for {inferred_symbol_type} symbols ({symbols_str}) ended. Cleaning up WebSocket client...", file=sys.stderr)
        if wss_client:
            try:
                # print(f"Attempting to close WebSocket client for {inferred_symbol_type} symbols: {symbols_str}...", file=sys.stderr)
                wss_client.close() # Ensure this is called
                # print(f"WebSocket client for {inferred_symbol_type} symbols ({symbols_str}) closed.", file=sys.stderr)
            except Exception as e_close:
                print(f"Error during WebSocket client close for {inferred_symbol_type} ({symbols_str}): {e_close}", file=sys.stderr)
        # else:
            # print(f"WebSocket client for {inferred_symbol_type} ({symbols_str}) was not initialized, no close needed.", file=sys.stderr)
        print(f"Cleanup complete for {inferred_symbol_type} stream ({symbols_str}).", file=sys.stderr)

async def main():
    global _db_client_instance_for_cleanup
    parser = argparse.ArgumentParser(description="Stream real-time market data and optionally save it to a database.")
    parser.add_argument("symbols", nargs='+',
                        help="One or more stock or crypto symbols (e.g., SPY AAPL, or BTC/USD ETH/USD). Crypto symbols should contain /.")
    parser.add_argument("--feed", choices=["quotes", "trades"], default="quotes",
                        help="The type of data feed to subscribe to (quotes or trades). Default is quotes.")
    
    # Database arguments group
    group = parser.add_mutually_exclusive_group(required=False) # Not strictly required to save data
    group.add_argument("--db-path", type=str, default=None,
                       help="Path to the local database file (SQLite/DuckDB). Extension determines type if --db-type not set.")
    group.add_argument("--remote-db-server", type=str, default=None,
                       help="Address of the remote DB server (e.g., localhost:8080). \
                            If used, --db-type is implicitly 'remote'.")

    parser.add_argument("--db-type", choices=["sqlite", "duckdb"], default="duckdb", # Removed 'remote' here
                        help="Type of local database to use if --db-path is specified (default: duckdb). \
                             Ignored if --remote-db-server is used.")
    
    parser.add_argument('--only-log-updates', dest='only_log_updates', action='store_true',
                        help="Log/save data only if bid/ask prices change (for quotes). Trades always logged/saved if DB is active.")
    parser.add_argument('--log-all-data', dest='only_log_updates', action='store_false',
                        help="Log/save all incoming quote data, regardless of price changes.")
    parser.set_defaults(only_log_updates=True)
    args = parser.parse_args()

    if not args.symbols:
        print("No symbols provided. Exiting.", file=sys.stderr)
        return

    stock_db_instance: StockDBBase | None = None
    db_type_to_use: str
    db_config_to_use: str | None = None

    if args.remote_db_server:
        if args.db_path or (args.db_type and args.db_type != "duckdb"): # duckdb is default, check if user tried to specify sqlite for remote
             print("Warning: --db-path and --db-type are ignored when --remote-db-server is used.", file=sys.stderr)
        db_type_to_use = "remote"
        db_config_to_use = args.remote_db_server
        print(f"Configuring to use remote database server at: {db_config_to_use}", file=sys.stderr)
    elif args.db_path:
        db_type_to_use = args.db_type
        db_config_to_use = args.db_path
        print(f"Configuring to use local database: type='{db_type_to_use}', path='{db_config_to_use}'")
    else:
        # Default local DB if no remote and no specific db-path provided
        # Use args.db_type (defaults to duckdb) and its default path
        db_type_to_use = args.db_type 
        db_config_to_use = get_default_db_path(db_type_to_use)
        print(f"No explicit DB target. Defaulting to local database: type='{db_type_to_use}', path='{db_config_to_use}'")
        # One could choose to not initialize DB at all if no flags are set, by setting db_config_to_use = None here
        # For now, we default to a local duckdb

    if db_config_to_use: # Proceed with DB initialization only if a config is set
        try:
            print(f"Initializing database connection: type='{db_type_to_use}', config='{db_config_to_use}'", file=sys.stderr)
            stock_db_instance = get_stock_db(db_type=db_type_to_use, db_config=db_config_to_use)
            if db_type_to_use == "remote":
                _db_client_instance_for_cleanup = stock_db_instance # Store for cleanup
            print(f"Database connection '{db_config_to_use}' ({db_type_to_use}) initialized successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing database (type: {db_type_to_use}, config: {db_config_to_use}): {e}. Data will not be saved.", file=sys.stderr)
            traceback.print_exc()
            stock_db_instance = None # Ensure it's None if init fails
    else:
        print("No database configured. Market data will be printed to console but not saved.", file=sys.stderr)

    stock_symbols = []
    crypto_symbols = []

    for symbol_arg in args.symbols:
        # Infer type: crypto if contains '/', otherwise stock.
        if "/" in symbol_arg:
            crypto_symbols.append(symbol_arg)
        else:
            stock_symbols.append(symbol_arg)

    tasks_to_run = []
    if stock_symbols:
        print(f"Preparing to stream stocks: { ', '.join(stock_symbols) }", file=sys.stderr)
        # setup_and_run_stream is blocking, so it needs to run in a thread
        tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, stock_symbols, args.feed, "stock", args.only_log_updates))
    
    if crypto_symbols:
        print(f"Preparing to stream cryptos: { ', '.join(crypto_symbols) }", file=sys.stderr)
        tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, crypto_symbols, args.feed, "crypto", args.only_log_updates))

    if not tasks_to_run:
        print("No valid stock or crypto symbols to stream based on input. Exiting.", file=sys.stderr)
        if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
            await _db_client_instance_for_cleanup.close_session() # type: ignore
        return

    print(f"Starting {len(tasks_to_run)} stream(s) concurrently...", file=sys.stderr)
    try:
        await asyncio.gather(*tasks_to_run)
    except Exception as e:
        print(f"Error during asyncio.gather for streams: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        print("All stream tasks have completed or an error occurred.", file=sys.stderr)
        if _db_client_instance_for_cleanup and hasattr(_db_client_instance_for_cleanup, 'close_session'):
            print("Closing remote DB client session...", file=sys.stderr)
            await _db_client_instance_for_cleanup.close_session() # type: ignore
            print("Remote DB client session closed.", file=sys.stderr)

if __name__ == "__main__":
    try:
        print("Starting market data streaming application...", file=sys.stderr)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in __main__. Application is shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"Unhandled exception in main asyncio.run: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Any final cleanup if needed, though client session should be handled in main()
        print("Application shutdown sequence complete.", file=sys.stderr)
        sys.exit(0)
