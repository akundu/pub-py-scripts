import argparse
import asyncio
import os
import pandas as pd
from alpaca.data.live import StockDataStream, CryptoDataStream
from stock_db import get_stock_db, StockDBBase, get_default_db_path
import sys

async def trade_data_handler(data):
    """Asynchronous handler to process incoming trade data."""
    print(f"Trade for {data.symbol}: Price - {data.price}, Size - {data.size} at {data.timestamp}")

def setup_and_run_stream(stock_db_instance: StockDBBase | None, symbols: list[str], feed: str, symbol_type: str, only_log_updates: bool):
    """Sets up and runs the WebSocket stream for a given list of symbols and type."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    wss_client = None

    if not symbols:
        raise ValueError("No symbols provided.")
    
    first_symbol = symbols[0]
    # Infer type: crypto if ends with /USD, otherwise stock.
    symbol_type = "crypto" if "/" in first_symbol else "stock" # Adjusted to contains '/' for broader crypto pair matching based on common patterns. User specified /USD but '/' is more general. If strict /USD is needed, this can be changed back.


    print(f"Inferred symbol type as: {symbol_type} (based on '{first_symbol}')")

    last_prices = {'bid_price': None, 'ask_price': None}
    # Define quote_data_handler as a closure here
    async def quote_data_handler_closure(data):
        """Asynchronous handler to process and save incoming quote data."""
        #print(f"Quote for {data.symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {data.timestamp}", file=sys.stderr)

        # Store last bid/ask prices in a closure variable to track changes
        prices_have_changed = False
        if data.bid_price != last_prices['bid_price'] or data.ask_price != last_prices['ask_price']:
            prices_have_changed = True

        if not only_log_updates or prices_have_changed:
            # Proceed to log/save if only_log_updates is False OR if prices have changed
            print(f"Quote for {data.symbol} ({symbol_type}): Bid - {data.bid_price} (Size: {data.bid_size}), Ask - {data.ask_price} (Size: {data.ask_size}) at {data.timestamp}")
            #print(f"for {data.symbol} Prices have changed: {prices_have_changed} {last_prices} current: bid:{data.bid_price} ask:{data.ask_price}", file=sys.stderr)
            if stock_db_instance:
                try:
                    df_data = {
                        'timestamp': [pd.to_datetime(data.timestamp)],
                        'price': [data.bid_price],
                        'volume': [data.bid_size]
                    }
                    quote_df = pd.DataFrame(df_data).set_index('timestamp')
                    await asyncio.to_thread(stock_db_instance.save_realtime_data, quote_df, data.symbol)
                except Exception as e:
                    print(f"Error saving quote data for {data.symbol} to DB: {e}")

        last_prices['bid_price'] = data.bid_price
        last_prices['ask_price'] = data.ask_price
        return

    # API Key checks based on inferred_symbol_type
    if symbol_type == "stock":
        if not api_key or not secret_key:
            print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set for stock data.")
            return
    elif symbol_type == "crypto":
        if not api_key or not secret_key: # This was a warning for crypto
            print("Warning: API keys not found. Crypto data can be streamed without keys, but with lower rate limits.")

    symbols_str = ", ".join(symbols)
    print(f"Attempting to stream {feed} for {symbol_type} symbols: {symbols_str}")

    try:
        if symbol_type == "stock":
            wss_client = StockDataStream(api_key, secret_key)
            if feed == "quotes":
                print(f"Subscribing to stock quotes for: {symbols_str}")
                wss_client.subscribe_quotes(quote_data_handler_closure, *symbols)
            elif feed == "trades":
                print(f"Subscribing to stock trades for: {symbols_str}")
                wss_client.subscribe_trades(trade_data_handler, *symbols)
        elif symbol_type == "crypto":
            wss_client = CryptoDataStream(api_key, secret_key) if api_key and secret_key else CryptoDataStream()
            if feed == "quotes":
                print(f"Subscribing to crypto quotes for: {symbols_str}")
                wss_client.subscribe_quotes(quote_data_handler_closure, *symbols)
            elif feed == "trades":
                print(f"Subscribing to crypto trades for: {symbols_str}")
                wss_client.subscribe_trades(trade_data_handler, *symbols)
        else:
            # This case should ideally not be reached if inferred_symbol_type is always "stock" or "crypto"
            print(f"Internal error: Unsupported inferred symbol type: {inferred_symbol_type}")
            return

        if wss_client:
            print(f"Starting WebSocket client for {symbol_type} symbols...", file=sys.stderr)
            wss_client.run()
        else:
            print(f"WebSocket client for {symbol_type} was not initialized. This might be due to missing API keys for stock data or an internal error.", file=sys.stderr)
            if symbol_type == "stock" and (not api_key or not secret_key):
                pass # Already handled by the check above
            else:
                raise RuntimeError("WebSocket client for {symbol_type} was not initialized for an unexpected reason.")


    except KeyboardInterrupt:
        print(f"KeyboardInterrupt caught in {symbol_type} stream. Stopping stream...")
    except Exception as e:
        print(f"An error occurred during {symbol_type} stream operation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Stream processing for {symbol_type} ended. Cleaning up WebSocket client...")
        if wss_client:
            try:
                print(f"Attempting to close WebSocket client for {symbol_type}...")
                wss_client.close()
                print(f"WebSocket client for {symbol_type} closed.")
            except Exception as e_close:
                print(f"Error during WebSocket client close for {symbol_type}: {e_close}")
        else:
            print(f"WebSocket client for {symbol_type} was not initialized, no close needed.")
        print(f"Cleanup complete for {symbol_type} stream.")

async def main():
    parser = argparse.ArgumentParser(description="Stream real-time market data and optionally save it to a database.")
    parser.add_argument("symbols", nargs='+',
                        help="One or more stock or crypto symbols (e.g., SPY AAPL, or BTC/USD ETH/USD). Crypto symbols should contain /.")
    parser.add_argument("--feed", choices=["quotes", "trades"], default="quotes",
                        help="The type of data feed to subscribe to (quotes or trades). Default is quotes.")
    parser.add_argument("--db-type", choices=["sqlite", "duckdb"], default="duckdb",
                        help="Database type to use for saving realtime data (default: duckdb). Only used if saving.")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to the database file. If not provided, uses default path for the selected db-type.")
    parser.add_argument('--only-log-updates', dest='only_log_updates', action='store_true',
                        help="Log data only if bid/ask prices change (default behaviour).")
    parser.add_argument('--log-all-data', dest='only_log_updates', action='store_false',
                        help="Log all incoming quote data, regardless of price changes.")
    parser.set_defaults(only_log_updates=True)
    args = parser.parse_args()

    if not args.symbols:
        print("No symbols provided. Exiting.")
        return

    # Initialize DB instance (once)
    stock_db_instance: StockDBBase | None = None
    try:
        actual_db_path = args.db_path
        if actual_db_path is None:
            actual_db_path = get_default_db_path(args.db_type)
        print(f"Initializing database: type='{args.db_type}', path='{actual_db_path}'")
        stock_db_instance = get_stock_db(db_type=args.db_type, db_path=actual_db_path)
        print(f"Database '{actual_db_path}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}. Quote data will not be saved.")
        # stock_db_instance remains None

    stock_symbols = []
    crypto_symbols = []

    for symbol in args.symbols:
        if "/" in symbol:
            crypto_symbols.append(symbol)
        else:
            stock_symbols.append(symbol)

    tasks_to_run = []
    if stock_symbols:
        print(f"Preparing to stream stocks: {', '.join(stock_symbols)}")
        tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, stock_symbols, args.feed, "stock", args.only_log_updates))
    
    if crypto_symbols:
        print(f"Preparing to stream cryptos: {', '.join(crypto_symbols)}")
        tasks_to_run.append(asyncio.to_thread(setup_and_run_stream, stock_db_instance, crypto_symbols, args.feed, "crypto", args.only_log_updates))

    if not tasks_to_run:
        print("No valid stock or crypto symbols to stream based on input. Exiting.")
        return

    print(f"Starting {len(tasks_to_run)} stream(s) concurrently...")
    await asyncio.gather(*tasks_to_run)
    print("All stream tasks have completed.")

if __name__ == "__main__":
    try:
        print("Starting stream application...", file=sys.stderr)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in __main__. Application is shutting down...", file=sys.stderr)
    finally:
        print("Application shutdown complete.", file=sys.stderr)
        sys.exit(0) # Removed to allow natural exit after asyncio.run completes or is interrupted.
