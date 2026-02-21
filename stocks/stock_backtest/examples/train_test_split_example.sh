#!/bin/bash

# Example Script: Train/Test Split Backtesting
# 
# This script demonstrates the difference between traditional backtesting
# (using same data for training and testing) and proper train/test split.

echo "=================================================="
echo "Stock Backtesting: Train/Test Split Examples"
echo "=================================================="
echo ""

# ====================
# Example 1: Basic Train/Test Split
# ====================
echo "Example 1: Basic Train/Test Split (REQUIRED)"
echo "-----------------------------------"
echo "Training Period: 2020-01-01 to 2023-12-31 (4 years)"
echo "Testing Period:  2024-01-01 to 2024-10-31 (10 months)"
echo ""
echo "This provides realistic out-of-sample performance metrics"
echo ""
echo "Command:"
echo "python -m stock_backtest.cli.main \\"
echo "  --symbols AAPL \\"
echo "  --strategy markov_int \\"
echo "  --training-start 2020-01-01 \\"
echo "  --training-end 2023-12-31 \\"
echo "  --start 2024-01-01 \\"
echo "  --end 2024-10-31"
echo ""
read -p "Press Enter to run (or Ctrl+C to skip)..."

python -m stock_backtest.cli.main \
  --symbols AAPL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31

echo ""
echo ""

# ====================
# Example 2: Multiple Stocks with Train/Test Split
# ====================
echo "Example 2: Multiple Stocks with Train/Test Split"
echo "-------------------------------------------------"
echo "Testing multiple stocks: AAPL, MSFT, GOOGL"
echo "Using parallel processing for efficiency"
echo ""
echo "Command:"
echo "python -m stock_backtest.cli.main \\"
echo "  --symbols AAPL MSFT GOOGL \\"
echo "  --strategy markov_int \\"
echo "  --training-start 2020-01-01 \\"
echo "  --training-end 2023-12-31 \\"
echo "  --start 2024-01-01 \\"
echo "  --end 2024-10-31 \\"
echo "  --workers 4 \\"
echo "  --output-dir results \\"
echo "  --output-format csv"
echo ""
read -p "Press Enter to run (or Ctrl+C to skip)..."

python -m stock_backtest.cli.main \
  --symbols AAPL MSFT GOOGL \
  --strategy markov_int \
  --training-start 2020-01-01 \
  --training-end 2023-12-31 \
  --start 2024-01-01 \
  --end 2024-10-31 \
  --workers 4 \
  --output-dir results \
  --output-format csv

echo ""
echo ""

# ====================
# Example 3: Using Configuration File
# ====================
echo "Example 3: Using Configuration File"
echo "------------------------------------"
echo "You can also use a YAML configuration file for easier management"
echo ""
echo "Command:"
echo "python -m stock_backtest.cli.main \\"
echo "  --config examples/train_test_split_config.yaml"
echo ""
read -p "Press Enter to run (or Ctrl+C to skip)..."

python -m stock_backtest.cli.main \
  --config examples/train_test_split_config.yaml

echo ""
echo "=================================================="
echo "Examples completed!"
echo "=================================================="
echo ""
echo "Key Takeaways:"
echo "1. Training dates (--training-start and --training-end) are REQUIRED"
echo "2. Training dates MUST be different from prediction dates"
echo "3. Use --start and --end for the prediction/testing period"
echo "4. Training period should NOT overlap with testing period"
echo "5. This prevents data leakage and provides realistic metrics"
echo ""
echo "For more information, see TRAIN_TEST_SPLIT_CHANGES.md"

