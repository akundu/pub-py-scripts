# Comprehensive Stock Data Fetcher

A comprehensive Python program that uses the Polygon API to fetch detailed stock market information including:

## Features

### 📈 Real-time and Historical Market Data
- **Historical Daily and Minute Bar Data**: OHLCV (Open, High, Low, Close, Volume) data
- **Dividends**: Historical dividend payments and dates
- **Stock Splits**: Historical stock split information
- **Real-time Quotes**: Latest bid/ask prices and sizes
- **Latest Trades**: Most recent trade prices and volumes

### 📊 Options Data
- **Real-time Options Quotes**: Bid, ask, last trade, and volume for options contracts
- **Historical Options Data**: OHLCV data for options
- **Options Chains**: All available options contracts with strike prices and expiration dates
- **Options Greeks**: Delta, gamma, theta, vega, rho (where available)

### 💰 Company Financials
- **Income Statements**: Revenue, cost of goods sold, gross profit, operating expenses, net income
- **Balance Sheets**: Assets, liabilities, equity, cash, accounts receivable
- **Cash Flow Statements**: Operating, investing, and financing cash flows

### 🏢 Company Details
- **Company Profile**: Industry, sector, CEO, number of employees, headquarters, website
- **Market Cap**: Current market capitalization
- **Shares Outstanding**: Number of shares outstanding
- **Exchange Listings**: Where the stock is traded
- **Contact Information**: Phone, address, website

### 📰 News and Events
- **Financial News**: Recent articles related to the company
- **News Metadata**: Publisher information, publication dates, article URLs
- **Keyword Tags**: News article categorization

## Prerequisites

1. **Polygon API Key**: Get a free or paid API key from [Polygon.io](https://polygon.io/)
2. **Python Dependencies**: Install required packages

```bash
pip install polygon-api-client requests pandas aiohttp tabulate
```

Or install from the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Usage

### Setup API Key (Recommended)

Set your Polygon API key as an environment variable:

```bash
export POLYGON_API_KEY=YOUR_POLYGON_API_KEY
```

### Basic Usage

```bash
python comprehensive_stock_data.py AAPL
```

Or use the API key as a command line argument:

```bash
python comprehensive_stock_data.py AAPL --api-key YOUR_POLYGON_API_KEY
```

### Advanced Usage Examples

#### Get data in table format (default: last 180 days)
```bash
python comprehensive_stock_data.py MSFT --format table
```

#### Fetch specific date range
```bash
python comprehensive_stock_data.py GOOGL --start-date 2024-01-01 --end-date 2024-12-31 --save
```

#### Fetch last 60 days using days-back
```bash
python comprehensive_stock_data.py AAPL --days-back 60 --format table
```

#### Get data for a specific month
```bash
python comprehensive_stock_data.py TSLA --start-date 2024-06-01 --end-date 2024-06-30 --no-options
```

#### Get year-to-date data
```bash
python comprehensive_stock_data.py AMZN --start-date 2024-01-01 --no-news
```

#### Limit news articles and specify output directory with custom date range
```bash
python comprehensive_stock_data.py NVDA --start-date 2024-01-01 --max-news 5 --save --output-dir ./stock_data
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `symbol` | Stock symbol to fetch data for | Required |
| `--api-key` | Polygon API key (or set POLYGON_API_KEY env var) | Optional if env var set |
| `--days-back` | Number of days back for historical data | 180 |
| `--start-date` | Start date for historical data (YYYY-MM-DD) | 180 days ago |
| `--end-date` | End date for historical data (YYYY-MM-DD) | Today |
| `--format` | Output format: `json` or `table` | json |
| `--no-options` | Skip options data | False |
| `--no-financials` | Skip financial data | False |
| `--no-news` | Skip news data | False |
| `--max-news` | Maximum number of news articles | 10 |
| `--save` | Save data to JSON file | False |
| `--output-dir` | Output directory for saved files | output |

## Output Format

### JSON Output (Default)
The program outputs comprehensive JSON data with the following structure:

```json
{
  "symbol": "AAPL",
  "timestamp": "2024-01-15T10:30:00",
  "market_data": {
    "success": true,
    "data": {
      "daily_bars": [...],
      "latest_quote": {...},
      "latest_trade": {...},
      "dividends": [...],
      "stock_splits": [...]
    }
  },
  "options_data": {
    "success": true,
    "data": {
      "contracts": [...]
    }
  },
  "company_financials": {
    "success": true,
    "data": {
      "income_statements": [...],
      "balance_sheets": [...],
      "cash_flow_statements": [...]
    }
  },
  "company_details": {
    "success": true,
    "data": {
      "profile": {...}
    }
  },
  "news_and_events": {
    "success": true,
    "data": {
      "financial_news": [...]
    }
  }
}
```

### Table Output
When using `--format table`, the program displays key information in human-readable tables:
- Company profile summary
- Latest market data (quotes and trades)
- Recent news headlines with publication dates

## Data Details

### Market Data
- **Daily Bars**: Date, open, high, low, close, volume, VWAP
- **Real-time Quotes**: Bid, ask, bid size, ask size, timestamp
- **Dividends**: Cash amount, ex-dividend date, pay date, frequency
- **Stock Splits**: Execution date, split ratio

### Options Data
- **Contract Details**: Ticker, strike price, expiration date, contract type
- **Market Data**: Implied volatility, open interest (where available)
- **Greeks**: Delta, gamma, theta, vega, rho (premium tier feature)

### Financial Statements
- **Income Statement**: Revenues, cost of revenue, gross profit, operating expenses, net income
- **Balance Sheet**: Total assets, current assets, liabilities, equity
- **Cash Flow**: Operating, investing, and financing cash flows

### Company Profile
- **Basic Info**: Name, ticker, exchange, market cap, employees
- **Contact**: Address, phone, website
- **Classification**: SIC code, industry description
- **Identifiers**: CIK, FIGI codes

## Error Handling

The program includes robust error handling:
- **API Errors**: Gracefully handles rate limits and API failures
- **Missing Data**: Continues execution if certain data types are unavailable
- **Network Issues**: Includes retry logic for temporary network problems
- **Invalid Symbols**: Provides clear error messages for invalid stock symbols

## Performance Features

- **Concurrent Execution**: Fetches different data types simultaneously for faster performance
- **Rate Limit Handling**: Automatically handles Polygon API rate limits
- **Efficient Data Processing**: Uses async/await patterns for optimal performance
- **Memory Management**: Streams large datasets to avoid memory issues

## File Output

When using the `--save` option, data is saved as:
```
{SYMBOL}_comprehensive_data_{TIMESTAMP}.json
```

Example: `AAPL_comprehensive_data_20240115_103000.json`

## API Rate Limits

- **Free Tier**: 5 requests per minute
- **Paid Tiers**: Higher rate limits based on subscription

The program automatically handles rate limits and will pause execution if limits are exceeded.

## Troubleshooting

### Common Issues

1. **"polygon-api-client not installed"**
   ```bash
   pip install polygon-api-client
   ```

2. **"Invalid API key"**
   - Verify your Polygon API key is correct
   - Check that your API key has the necessary permissions

3. **"No data returned"**
   - Verify the stock symbol is correct
   - Check if the symbol is actively traded
   - Some data may not be available for all symbols

4. **Rate limit errors**
   - The program will automatically wait and retry
   - Consider upgrading to a paid Polygon plan for higher limits

### Getting Help

For issues specific to the Polygon API, consult the [Polygon API Documentation](https://polygon.io/docs/).

## License

This program is provided as-is for educational and research purposes. Please ensure compliance with Polygon.io's terms of service when using their API. 