import argparse
import asyncio
import os

from alpaca.data.live import StockDataStream, CryptoDataStream

async def quote_data_handler(data):
    """Asynchronous handler to process incoming quote data."""
    print(f"Quote for {data.symbol}: Bid - {data.bid_price}, Ask - {data.ask_price} at {data.timestamp}")

async def trade_data_handler(data):
    """Asynchronous handler to process incoming trade data."""
    print(f"Trade for {data.symbol}: Price - {data.price}, Size - {data.size} at {data.timestamp}")

def setup_and_run_stream(args):
    """Sets up and runs the WebSocket stream."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_API_SECRET")
    wss_client = None

    if args.type == "stock" and (not api_key or not secret_key):
        print("Error: ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set for stock data.")
        return
    
    if args.type == "crypto" and (not api_key or not secret_key):
        print("Warning: API keys not found. Crypto data can be streamed without keys, but with lower rate limits.")

    symbols_str = ", ".join(args.symbols)
    print(f"Attempting to stream {args.feed} for {args.type} symbols: {symbols_str}")

    try:
        if args.type == "stock":
            wss_client = StockDataStream(api_key, secret_key)
            if args.feed == "quotes":
                print(f"Subscribing to quotes for: {symbols_str}")
                wss_client.subscribe_quotes(quote_data_handler, *args.symbols)
            elif args.feed == "trades":
                print(f"Subscribing to trades for: {symbols_str}")
                wss_client.subscribe_trades(trade_data_handler, *args.symbols)
        elif args.type == "crypto":
            wss_client = CryptoDataStream(api_key, secret_key) if api_key and secret_key else CryptoDataStream()
            if args.feed == "quotes":
                print(f"Subscribing to crypto quotes for: {symbols_str}")
                wss_client.subscribe_quotes(quote_data_handler, *args.symbols)
            elif args.feed == "trades":
                print(f"Subscribing to crypto trades for: {symbols_str}")
                wss_client.subscribe_trades(trade_data_handler, *args.symbols)
        else:
            print(f"Internal error: Unsupported symbol type: {args.type}")
            return

        if wss_client:
            # The .run() method of DataStream is blocking and manages its own event loop activity.
            # It should not be awaited from an already running asyncio event loop.
            # It is called from a synchronous context, or from an async context as if it were synchronous.
            wss_client.run() # Removed await
        else:
            print("Error: WebSocket client was not initialized.")

    except Exception as e:
        print(f"An error occurred during stream operation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stream stopped or an error occurred.")


async def main():
    parser = argparse.ArgumentParser(description="Stream real-time market data for one or more stock/crypto symbols.")
    parser.add_argument("symbols", nargs='+', 
                        help="One or more stock or crypto symbols (e.g., SPY AAPL, or BTC/USD ETH/USD). Crypto symbols should be in XXX/YYY format.")
    parser.add_argument("--type", choices=["stock", "crypto"], default="stock", 
                        help="The type of symbols (stock or crypto). Default is stock. All symbols must be of this type.")
    parser.add_argument("--feed", choices=["quotes", "trades"], default="quotes",
                        help="The type of data feed to subscribe to (quotes or trades). Default is quotes.")
    args = parser.parse_args()
    
    # Since setup_and_run_stream now calls the blocking wss_client.run(),
    # and wss_client.run() handles its own async loop, we call setup_and_run_stream directly.
    # The handlers (quote_data_handler, trade_data_handler) are async and will be scheduled by the SDK's loop.
    #setup_and_run_stream(args)
    await asyncio.to_thread(setup_and_run_stream, args)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stream interrupted by user.") 
