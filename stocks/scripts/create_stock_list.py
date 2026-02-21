#!/usr/bin/env python3

import sys
import yaml
from typing import List

def read_symbols_from_stdin() -> List[str]:
    """Read stock symbols from stdin, one per line."""
    symbols = []
    # print("Enter stock symbols (one per line). Press Ctrl+D (Unix) or Ctrl+Z (Windows) when done:")
    for line in sys.stdin:
        symbol = line.strip().upper()
        if symbol:  # Skip empty lines
            symbols.append(symbol)
    return sorted(symbols)  # Sort symbols alphabetically

def create_yaml_file(symbols: List[str], output_file: str):
    """Create a YAML file with the given symbols."""
    data = {
        'type': 'stock-list',
        'symbols': symbols
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_yaml_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    symbols = read_symbols_from_stdin()
    
    if not symbols:
        print("No symbols provided. Exiting.")
        sys.exit(1)
    
    create_yaml_file(symbols, output_file)
    print(f"Successfully created YAML file: {output_file}", file=sys.stderr)

if __name__ == "__main__":
    main() 