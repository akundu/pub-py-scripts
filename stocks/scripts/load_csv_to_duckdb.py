import argparse
import datetime
from pathlib import Path
import pandas as pd
import sys
import asyncio
from common.stock_db import get_stock_db


async def main():
    parser = argparse.ArgumentParser(description="Load stock data from a given CSV file into a remote stock DB server in chunks.")
    parser.add_argument("symbol", type=str, help="Stock symbol (e.g., GBTC) to associate with the data in the database.")
    parser.add_argument("csv_file", type=str, help="Full path to the CSV file to load.")
    parser.add_argument(
        "--socket-server",
        type=str,
        required=True,
        help="Socket server address (e.g., localhost:8080) for the StockDB server.",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="If set, ignore errors during CSV reading or DB writing and attempt to continue.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of rows to send to the server in each chunk. Default is 100."
    )

    args = parser.parse_args()

    csv_file_path = Path(args.csv_file)

    print(f"--- Stock Data Importer (Remote, Chunked) ---")
    print(f"Target CSV file path: {csv_file_path}")
    print(f"Stock symbol for DB: {args.symbol.upper()}")
    print(f"StockDB server address: {args.socket_server}")
    print(f"Chunk size: {args.chunk_size} rows")
    print(f"Ignore errors mode: {'On' if args.ignore_errors else 'Off'}")
    print(f"---------------------------------------------")

    if not csv_file_path.exists():
        message = f"Error: CSV file not found at {csv_file_path}"
        if args.ignore_errors:
            print(f"Warning: {message}. Skipping processing.")
            return
        else:
            print(message)
            sys.exit(1)

    try:
        df_full = pd.read_csv(csv_file_path) # Read the full CSV first
        
        df_full.columns = [col.lower().strip() for col in df_full.columns]

        if 'timestamp' in df_full.columns:
            df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
        else:
            message = f"Error: 'timestamp' column missing in {csv_file_path}."
            if args.ignore_errors:
                print(f"Warning: {message}. Skipping DB insertion.")
                return
            else:
                print(message)
                sys.exit(1)
        
        expected_columns = ['timestamp', 'price', 'size', 'ask_price', 'ask_size']
        
        missing_cols = [col for col in expected_columns if col not in df_full.columns]
        if missing_cols:
            message = f"Warning: CSV is missing some expected quote columns: {missing_cols}. Found: {list(df_full.columns)}. Proceeding with available data."
            if args.ignore_errors:
                print(message)
            else:
                print(f"Error: CSV is missing expected columns: {missing_cols}. Found: {list(df_full.columns)}")
                sys.exit(1)
        
        if df_full.empty:
            print(f"Info: CSV file {csv_file_path} is empty. Nothing to send.")
            return

    except Exception as e:
        message = f"Error reading or processing CSV file {csv_file_path}: {e}"
        if args.ignore_errors:
            print(f"Warning: {message}. Skipping processing.")
            return
        else:
            print(message)
            sys.exit(1)

    db_client = None
    total_rows_sent = 0
    try:
        print(f"Connecting to StockDB server at {args.socket_server}...")
        db_client = get_stock_db("remote", args.socket_server)
        
        num_chunks = (len(df_full) - 1) // args.chunk_size + 1
        print(f"Preparing to send {len(df_full)} rows in {num_chunks} chunks of up to {args.chunk_size} rows each.")

        for i in range(0, len(df_full), args.chunk_size):
            df_chunk = df_full.iloc[i:i + args.chunk_size]
            chunk_number = (i // args.chunk_size) + 1
            print(f"Sending chunk {chunk_number}/{num_chunks} ({len(df_chunk)} rows) for {args.symbol.upper()} from {csv_file_path} to remote server...")
            try:
                await db_client.save_realtime_data(df_chunk, args.symbol.upper(), data_type="quote")
                total_rows_sent += len(df_chunk)
                print(f"Chunk {chunk_number}/{num_chunks} successfully sent.")
            except Exception as chunk_error:
                message = f"Error sending chunk {chunk_number}/{num_chunks} for {args.symbol.upper()}: {chunk_error}"
                if args.ignore_errors:
                    print(f"Warning: {message}. Skipping this chunk and continuing.")
                    continue # Move to the next chunk
                else:
                    print(message)
                    raise # Re-raise the exception to stop processing if not ignoring errors
        
        print(f"All {num_chunks} chunks processed. Total rows sent successfully: {total_rows_sent} for symbol '{args.symbol.upper()}'.")

    except ConnectionError as e:
        message = f"Error connecting or communicating with StockDB server: {e}"
        if args.ignore_errors:
            print(f"Warning: {message}. Data for {args.symbol.upper()} might not have been fully written.")
        else:
            print(message)
            sys.exit(1)
    except Exception as e:
        message = f"Error during remote DB operation or chunk processing: {e}"
        if args.ignore_errors:
            print(f"Warning: {message}. Data for {args.symbol.upper()} might not have been fully written.")
        else:
            print(message)
            sys.exit(1)
    finally:
        if db_client and hasattr(db_client, 'close_session') and callable(getattr(db_client, 'close_session')):
            try:
                await db_client.close_session()
                print("StockDBClient session closed.")
            except Exception as e:
                print(f"Error closing StockDBClient session: {e}")

if __name__ == "__main__":
    asyncio.run(main())
