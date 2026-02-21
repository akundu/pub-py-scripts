#!/usr/bin/env python3
"""
CSV Analysis Script - Analyze covered call spread data using Abacus AI API.

This script loads CSV data and sends it to the Abacus AI API for analysis.
"""

import requests
import pandas as pd
import json
import argparse
import sys
import os
from pathlib import Path

class AbacusOptionsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://api.abacus.ai/v0/chat"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_options(self, csv_path, prompt, model="claude-sonnet-4", temperature=0.5, max_tokens=6000, max_rows=50):
        """Analyze options data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            prompt: Analysis prompt to send to API
            model: Model to use (default: "claude-sonnet-4")
            temperature: Temperature for model (default: 0.5)
            max_tokens: Maximum tokens in response (default: 6000)
            max_rows: Maximum number of rows to include from CSV (default: 50)
            
        Returns:
            Analysis result string
        """
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        csv_content = df.head(max_rows).to_csv(index=False)
        
        # Prepare request
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nData:\n{csv_content}"
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make API call
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    default_prompt = """Analyze this covered call spread data and provide:
1. Top 5 recommendations with option tickers (short and long)
2. Strike prices and expiration dates
3. Risk metrics (delta, volume, P/E)
4. Expected returns and daily premiums"""
    
    parser = argparse.ArgumentParser(
        description="Analyze covered call spread data using Abacus AI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/csv_analysis.py --api-key YOUR_KEY --csv results.csv
  python scripts/csv_analysis.py --api-key YOUR_KEY --csv results.csv --prompt "Analyze risk factors"
  python scripts/csv_analysis.py --api-key YOUR_KEY --csv results.csv --max-rows 100 --temperature 0.7
        """
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        default=None,
        help="Abacus AI API key. If not provided, will use ABACUS_API_KEY environment variable"
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        required=True,
        help="Path to CSV file to analyze"
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=default_prompt,
        help=f"Analysis prompt (default: built-in prompt)"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default="claude-sonnet-4",
        help="Model to use (default: claude-sonnet-4)"
    )
    
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=0.5,
        help="Temperature for model (default: 0.5)"
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=6000,
        help="Maximum tokens in response (default: 6000)"
    )
    
    parser.add_argument(
        '--max-rows',
        type=int,
        default=50,
        help="Maximum number of rows to include from CSV (default: 50)"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output file path (default: print to stdout)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the CSV analysis."""
    args = parse_args()
    
    # Get API key from command line argument or environment variable
    api_key = None
    if args.api_key:
        api_key = args.api_key
        print("Using API key from command line argument", file=sys.stderr)
    else:
        api_key = os.getenv('ABACUS_API_KEY')
        if api_key:
            print("Using API key from ABACUS_API_KEY environment variable", file=sys.stderr)
    
    if not api_key:
        print("Error: API key is required.", file=sys.stderr)
        print("  Option 1: Use --api-key YOUR_KEY on the command line", file=sys.stderr)
        print("  Option 2: Set ABACUS_API_KEY environment variable", file=sys.stderr)
        print("    Example: export ABACUS_API_KEY=YOUR_KEY", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create analyzer
        analyzer = AbacusOptionsAnalyzer(api_key=api_key)
        
        # Run analysis
        print(f"Analyzing CSV file: {args.csv}", file=sys.stderr)
        print(f"Using model: {args.model}", file=sys.stderr)
        print(f"Including up to {args.max_rows} rows from CSV", file=sys.stderr)
        print("", file=sys.stderr)
        
        result = analyzer.analyze_options(
            csv_path=args.csv,
            prompt=args.prompt,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_rows=args.max_rows
        )
        
        # Output result
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Results written to: {output_path.absolute()}", file=sys.stderr)
        else:
            print(result)
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON response from API: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
