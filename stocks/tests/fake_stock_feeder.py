import asyncio
import argparse
import aiohttp
import random
import datetime
import json
import sys

async def generate_and_send_data(session: aiohttp.ClientSession, server_url: str, ticker: str, interval_seconds: float):
    """Generates fake stock data and sends it to the db_server."""
    base_price = 100.0
    print(f"Starting to feed fake data for {ticker} to {server_url} every ~{interval_seconds:.2f} seconds.")
    
    try:
        while True:
            # Generate a small price fluctuation
            price_change = random.uniform(-0.25, 0.25) # Increased fluctuation slightly
            current_price = round(max(10.0, base_price + price_change), 2) # Ensure price doesn't go too low
            base_price = current_price # Let the price drift

            # Generate bid/ask spread and sizes
            spread = random.uniform(0.01, 0.05)
            bid_price = round(current_price - spread / 2, 2)
            ask_price = round(current_price + spread / 2, 2)
            bid_size = random.randint(1, 50) * 10  # e.g., 100 to 5000 in lots of 10
            ask_size = random.randint(1, 50) * 10

            # Use current UTC timestamp in ISO format
            # Ensure 'Z' at the end for UTC, which pd.to_datetime handles well
            timestamp_dt = datetime.datetime.now(datetime.timezone.utc)
            timestamp_iso_z = timestamp_dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


            data_record = {
                "timestamp": timestamp_iso_z,
                "price": bid_price,      # In db_server, 'price' for quotes often refers to bid_price
                "size": bid_size,
                "ask_price": ask_price,
                "ask_size": ask_size
            }

            payload = {
                "command": "save_realtime_data",
                "params": {
                    "ticker": ticker,
                    "data_type": "quote",    # Sending quote data
                    "index_col": "timestamp", # db_server expects this for creating DataFrame
                    "data": [data_record]    # Data as a list of records
                }
            }

            print(f"[{timestamp_iso_z}] Sending to {ticker}: Bid={bid_price:.2f} (Sz:{bid_size}), Ask={ask_price:.2f} (Sz:{ask_size})")

            try:
                async with session.post(f"{server_url}/db_command", json=payload) as response:
                    if response.status == 200:
                        response_json = await response.json()
                        print(f"Server response: {response_json.get('message', 'OK')}")
                    else:
                        error_text = await response.text()
                        print(f"Error sending data: {response.status} - {error_text}", file=sys.stderr)
            except aiohttp.ClientConnectorError as e:
                print(f"Connection error: {e}", file=sys.stderr)
            except Exception as e:
                print(f"An unexpected error occurred: {e}", file=sys.stderr)

            await asyncio.sleep(interval_seconds * random.uniform(0.8, 1.2)) # Add slight randomness to interval

    except KeyboardInterrupt:
        print(f"\nStopping data feeder for {ticker}.")
    finally:
        print("Feeder shut down.")

async def main():
    parser = argparse.ArgumentParser(description="Fake stock data feeder for db_server.py.")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8080",
        help="URL of the db_server.py (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="FAKE1234",
        help="Ticker symbol to feed data for (default: FAKE1234)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Average interval in seconds between sending data points (default: 2.0)"
    )
    args = parser.parse_args()

    async with aiohttp.ClientSession() as session:
        await generate_and_send_data(session, args.server_url, args.ticker, args.interval)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nFeeder terminated by user.") 