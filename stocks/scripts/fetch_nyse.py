import requests
import csv
from io import StringIO
import sys

# --- Configuration ---
# These are the dynamic URLs provided by Nasdaq Trader for daily downloads.
# The 'otherlisted.txt' file contains companies listed on exchanges other than Nasdaq,
# which includes NYSE, NYSE American (AMEX), and NYSE Arca.
NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

def fetch_and_parse_listings(url: str, exchange_name: str) -> list[dict]:
    """
    Fetches the content from a URL and parses the NASDAQ Trader pipe-delimited
    text file format into a list of dictionaries.

    Args:
        url: The public URL of the stock listing file.
        exchange_name: A descriptive name for the exchange (e.g., 'NASDAQ', 'NYSE').

    Returns:
        A list of dictionaries, each representing a listed company.
    """
    try:
        print(f"Fetching data for {exchange_name} from {url}...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # The files use a pipe '|' as a delimiter and the last line is a footer
        # that needs to be removed before parsing.
        content = response.text
        
        # Remove the last line (the footer) which starts with 'File Creation Time'
        lines = content.strip().split('\n')
        if lines and lines[-1].startswith('File Creation Time'):
            lines.pop()

        # Combine lines back into a single string stream for the CSV parser
        data_io = StringIO('\n'.join(lines))

        # Use csv.reader to handle the pipe delimiter. Skip the header row.
        reader = csv.reader(data_io, delimiter='|')
        header = next(reader)
        
        # Define the indices for the key fields we want:
        # Note: The exact column layout differs slightly between the two files,
        # but the first two columns (Symbol and Security Name) are consistent.
        # We will use the common structure for simplicity.
        TICKER_INDEX = 0
        NAME_INDEX = 1

        companies = []
        for row in reader:
            if len(row) > NAME_INDEX:
                companies.append({
                    'ticker': row[TICKER_INDEX].strip(),
                    'name': row[NAME_INDEX].strip(),
                    'exchange_source': exchange_name
                })
        
        print(f"Successfully retrieved {len(companies)} listings for {exchange_name}.")
        return companies

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"An unexpected error occurred during parsing for {exchange_name}: {e}", file=sys.stderr)
        return []


def main():
    """Main function to fetch and combine the NYSE and NASDAQ listings."""
    
    # 1. Fetch NASDAQ listings
    nasdaq_companies = fetch_and_parse_listings(NASDAQ_URL, "NASDAQ")

    # 2. Fetch NYSE/Other listings (which includes NYSE, AMEX, ARCA, etc.)
    # We use 'NYSE Group' as a generalized name for clarity since the file
    # covers multiple New York Stock Exchange venues.
    nyse_group_companies = fetch_and_parse_listings(OTHER_LISTED_URL, "NYSE Group")

    # 3. Combine the lists
    all_companies = nasdaq_companies + nyse_group_companies

    print("\n" + "="*50)
    print(f"TOTAL UNIQUE LISTINGS FOUND: {len(all_companies)}")
    print("="*50)

    # 4. Display a sample of the results
    print("\n--- Sample of Listings (First 10) ---")
    for company in all_companies[:10]:
        print(f"[{company['ticker']}] {company['name']} (Source: {company['exchange_source']})")
    
    # Example: Find a known NYSE company in the combined list
    known_nyse_ticker = 'JPM' # JPMorgan Chase, typically NYSE
    jpm_search = next((c for c in nyse_group_companies if c['ticker'] == known_nyse_ticker), None)
    
    if jpm_search:
        print(f"\nExample known NYSE listing found: [{jpm_search['ticker']}] {jpm_search['name']}")

if __name__ == "__main__":
    # Ensure necessary packages are installed
    try:
        import requests
    except ImportError:
        print("The 'requests' library is required. Install it using: pip install requests", file=sys.stderr)
        sys.exit(1)
        
    main()
