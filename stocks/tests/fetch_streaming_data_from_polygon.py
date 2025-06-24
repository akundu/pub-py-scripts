import os
import asyncio
import pandas as pd
from polygon.websocket import WebSocketClient

# --- Get your API key ---
# It's best practice to use an environment variable for your API key
API_KEY = os.environ.get("POLYGON_API_KEY", "YOUR_API_KEY")

async def stream_stock_data(tickers_to_stream):
    """
    Connects to the Polygon.io WebSocket and streams real-time stock data.
    """
    if API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual Polygon.io API key.")
        return

    print("Connecting to Polygon.io WebSocket...")
    # The WebSocketClient handles the connection and authentication.
    # We specify the 'stocks' market cluster.
    stream = WebSocketClient(
        api_key=API_KEY,
        market="stocks"
    )

    # Define the callback function that will handle incoming messages
    async def handle_msg(msg):
        """
        This function is called for every message received from the stream.
        """
        # The 'msg' object is a list of events
        if isinstance(msg, list):
            for event in msg:
                await handle_single_event(event)
        else:
            await handle_single_event(msg)

    async def handle_single_event(event):
        """
        Handle a single event from the stream.
        """
        try:
            # 'T' for Trade
            if event.event_type == "T":
                trade_time = pd.to_datetime(event.timestamp, unit='ns').tz_localize('UTC').tz_convert('America/New_York')
                print(
                    f"Trade on {event.symbol}: "
                    f"Price: ${event.price:.2f}, "
                    f"Size: {event.size}, "
                    f"Time: {trade_time.strftime('%H:%M:%S.%f')}"
                )

            # 'Q' for Quote
            elif event.event_type == "Q":
                quote_time = pd.to_datetime(event.timestamp, unit='ns').tz_localize('UTC').tz_convert('America/New_York')
                print(
                    f"Quote for {event.symbol}: "
                    f"Bid: ${event.bid_price:.2f}, "
                    f"Ask: ${event.ask_price:.2f}, "
                    f"Time: {quote_time.strftime('%H:%M:%S.%f')}"
                )
                
            # You can add handlers for other event types like aggregates ('A') as well
            # elif event.event_type == "A":
            #     print(f"Aggregate for {event.symbol}: Open: {event.open}, Close: {event.close}")
                
        except Exception as e:
            print(f"Error processing event: {e}")
            print(f"Event data: {event}")

    # Define the tickers you are interested in

    # Subscribe to the desired data feeds for your chosen tickers
    # Use the subscribe method with proper subscription format
    for ticker in tickers_to_stream:
        stream.subscribe(f"T.{ticker}")  # Subscribe to trades
        stream.subscribe(f"Q.{ticker}")  # Subscribe to quotes
    
    # You can also subscribe to minute aggregates
    # for ticker in tickers_to_stream:
    #     stream.subscribe(f"A.{ticker}")  # Subscribe to aggregates

    print(f"Successfully subscribed to Trades and Quotes for: {', '.join(tickers_to_stream)}")
    print("--- Waiting for real-time data... Press Ctrl+C to stop. ---")
    
    # This connects to the stream and will run indefinitely,
    # calling 'handle_msg' for each piece of data received.
    await stream.connect(handle_msg)

if __name__ == '__main__':
    try:
        # The asyncio.run() function starts the asynchronous event loop
        tickers_to_stream = ["CART", "NVDA", "TSLA", "AAPL", "GBTC", "IBIT", "QQQ", "TQQQ"]
        asyncio.run(stream_stock_data(tickers_to_stream))
    except KeyboardInterrupt:
        print("\nStream manually closed by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
