#!/usr/bin/env python3
"""
AI-Powered SQL Query Tool for StockQuestDB

This script allows you to query the stock database using natural language.
It uses Google's Gemini AI to convert your English query into SQL, then executes it.
"""

import argparse
import sys
import asyncio
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from common.gemini_sql import generate_and_validate_sql, MODEL_ALIASES, list_available_models, get_model_info
from common.questdb_db import StockQuestDB
from common.stock_db import get_stock_db
from common.logging_utils import get_logger
# Optional retry system - only imported if needed
try:
    from common.ai_sql_retry import QueryRetryManager, RetryConfig, FailureType
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False


def format_output_table(df, max_display_rows: int = 50):
    """Format DataFrame as a readable table."""
    if df.empty:
        return "No results found."
    
    # Limit display rows
    display_df = df.head(max_display_rows)
    
    # Convert to string representation
    result = display_df.to_string()
    
    if len(df) > max_display_rows:
        result += f"\n\n... ({len(df) - max_display_rows} more rows not shown)"
    
    return result


def format_output_csv(df):
    """Format DataFrame as CSV."""
    return df.to_csv(index=False)


def format_output_json(df):
    """Format DataFrame as JSON."""
    import json
    return json.dumps(df.to_dict('records'), indent=2, default=str)


async def execute_query(db_instance: StockQuestDB, sql: str) -> tuple:
    """
    Execute SQL query and return results.
    
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        df = await db_instance.execute_select_sql(sql)
        return df, None
    except Exception as e:
        return None, str(e)


async def main():
    parser = argparse.ArgumentParser(
        description="Query StockQuestDB using natural language with AI-powered SQL generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --natural-query "latest price for AAPL"
  %(prog)s --natural-query "top 10 most traded tickers today" --model pro
  %(prog)s --natural-query "AAPL options expiring in March 2024" --max-rows 500
  %(prog)s --natural-query "daily prices for MSFT last 30 days" --output-format csv --yes
  %(prog)s --list-models
        """
    )
    
    parser.add_argument(
        "--natural-query",
        type=str,
        required=False,
        help="Natural language query description (e.g., 'latest price for AAPL')"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available Gemini models and exit"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="flash",
        choices=list(MODEL_ALIASES.keys()),
        help="Gemini model to use (default: flash)"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum number of rows to return (default: 1000)"
    )
    
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Show the generated SQL query before executing"
    )
    
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and execute query immediately"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Database connection string (default: from environment or questdb://localhost:9009/stock_data)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis cache"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    # Retry system arguments (only shown if available)
    if RETRY_AVAILABLE:
        parser.add_argument(
            "--enable-retry",
            action="store_true",
            help="Enable intelligent retry and self-correction for failed queries (default: disabled)"
        )
        parser.add_argument(
            "--max-retry-attempts",
            type=int,
            default=3,
            help="Maximum number of retry attempts when --enable-retry is used (default: 3)"
        )
        parser.add_argument(
            "--disable-reasoning",
            action="store_true",
            help="Disable AI reasoning for fixes (only retry with same query, for transient errors)"
        )
        parser.add_argument(
            "--disable-zero-result-handling",
            action="store_true",
            help="Don't retry queries that return zero results"
        )
    
    args = parser.parse_args()
    
    # Handle --list-models flag
    if args.list_models:
        print("Available Gemini Models (from API):", file=sys.stdout)
        print("=" * 70, file=sys.stdout)
        print("", file=sys.stdout)
        
        try:
            info = get_model_info()
            api_models = info['models']
            model_dict = info['model_dict']
            alias_mappings = info['alias_mappings']
            descriptions = info['descriptions']
            default = info['default']
            
            # Show all models from API
            print("All Available Models:", file=sys.stdout)
            print("-" * 70, file=sys.stdout)
            for model in api_models:
                # Handle both dict and object formats
                if isinstance(model, dict):
                    model_name = model.get('name', 'Unknown')
                    display_name = model.get('display_name', model_name)
                    description = model.get('description', '')
                    version = model.get('version', '')
                else:
                    model_name = getattr(model, 'name', 'Unknown')
                    display_name = getattr(model, 'display_name', model_name)
                    description = getattr(model, 'description', '')
                    version = getattr(model, 'version', '')
                
                print(f"  Model: {model_name}", file=sys.stdout)
                if display_name and display_name != model_name:
                    print(f"    Display Name: {display_name}", file=sys.stdout)
                if description:
                    print(f"    Description: {description}", file=sys.stdout)
                if version:
                    print(f"    Version: {version}", file=sys.stdout)
                print("", file=sys.stdout)
            
            # Show alias mappings
            if alias_mappings:
                print("=" * 70, file=sys.stdout)
                print("Model Aliases (for use with --model flag):", file=sys.stdout)
                print("-" * 70, file=sys.stdout)
                for alias in info['aliases']:
                    if alias in alias_mappings:
                        model_id = alias_mappings[alias]
                        desc = descriptions.get(alias, "No description available")
                        default_marker = " (default)" if alias == default else ""
                        print(f"  {alias:15} -> {model_id:30} {default_marker}", file=sys.stdout)
                        print(f"    {desc}", file=sys.stdout)
                        print("", file=sys.stdout)
                    else:
                        # Alias not found in API models
                        print(f"  {alias:15} -> (not available in API)", file=sys.stdout)
                        print("", file=sys.stdout)
            
            print("=" * 70, file=sys.stdout)
            print(f"Default model alias: {default}", file=sys.stdout)
            if default in alias_mappings:
                print(f"Default model ID: {alias_mappings[default]}", file=sys.stdout)
            print("", file=sys.stdout)
            print("Usage: Use --model <alias> to select a specific model", file=sys.stdout)
            
        except Exception as e:
            print(f"Error fetching models from Gemini API: {e}", file=sys.stderr)
            print("", file=sys.stderr)
            print("Falling back to hardcoded model list:", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            for alias, model_id in MODEL_ALIASES.items():
                default_marker = " (default)" if alias == default else ""
                print(f"  {alias:15} -> {model_id:30} {default_marker}", file=sys.stderr)
            sys.exit(1)
        
        sys.exit(0)
    
    # Validate that natural-query is provided if not listing models
    if not args.natural_query:
        parser.error("--natural-query is required (or use --list-models to see available models)")
    
    # Setup logging
    logger = get_logger("ai_sql_query", level=args.log_level)
    
    # Validate max_rows
    if args.max_rows > 10000:
        print("Error: --max-rows cannot exceed 10000 for safety.", file=sys.stderr)
        sys.exit(1)
    
    # Determine database connection
    db_config = args.db_path
    if not db_config:
        # Try environment variable
        db_config = os.getenv("QUESTDB_URL") or os.getenv("DB_PATH")
        if not db_config:
            # Default QuestDB connection
            db_config = "questdb://user:password@localhost:9009/stock_data"
    
    # Initialize database
    enable_cache = not args.no_cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0") if enable_cache else None
    
    print("=" * 70, file=sys.stderr)
    print("AI SQL Query Tool", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Natural Query: {args.natural_query}", file=sys.stderr)
    print(f"Model: {args.model} ({MODEL_ALIASES[args.model]})", file=sys.stderr)
    print(f"Max Rows: {args.max_rows}", file=sys.stderr)
    print(f"Database: {db_config}", file=sys.stderr)
    if RETRY_AVAILABLE and args.enable_retry:
        print(f"Retry System: ENABLED (max {args.max_retry_attempts} attempts)", file=sys.stderr)
    else:
        print(f"Retry System: DISABLED", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    try:
        # Generate SQL from natural language
        print("\nGenerating SQL query from natural language...", file=sys.stderr)
        sql = generate_and_validate_sql(
            args.natural_query,
            model_alias=args.model,
            max_rows=args.max_rows
        )
        
        print("\n" + "=" * 70, file=sys.stderr)
        print("Generated SQL:", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(sql, file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        
        # Show SQL if requested
        if args.show_sql:
            print("SQL Query:", file=sys.stdout)
            print(sql, file=sys.stdout)
            print("\n", file=sys.stdout)
        
        # Confirmation prompt (unless --yes)
        if not args.yes:
            response = input("Execute this query? (y/N): ").strip().lower()
            if response != 'y':
                print("Query cancelled.", file=sys.stderr)
                sys.exit(0)
        
        # Initialize database connection
        print("Connecting to database...", file=sys.stderr)
        db_instance = get_stock_db(
            db_type="questdb",
            db_config=db_config,
            log_level=args.log_level,
            enable_cache=enable_cache,
            redis_url=redis_url,
            auto_init=False
        )
        
        # Initialize database
        await db_instance._init_db()
        
        # Execute query (with or without retry system)
        if RETRY_AVAILABLE and args.enable_retry:
            # Use retry system
            print("Executing query with retry system enabled...", file=sys.stderr)
            
            # Create retry configuration
            retry_config = RetryConfig(
                max_attempts=args.max_retry_attempts,
                enable_reasoning=not args.disable_reasoning,
                enable_model_escalation=True,
                enable_zero_result_handling=not args.disable_zero_result_handling,
                initial_model=args.model,
                escalation_model="pro" if args.model != "pro" else "flash",
                log_attempts=True
            )
            
            # Create retry manager
            retry_manager = QueryRetryManager(config=retry_config, logger=logger)
            
            # Create execute function wrapper
            async def execute_with_retry(sql_to_execute: str):
                return await execute_query(db_instance, sql_to_execute)
            
            # Execute with retry
            df, error, attempts = await retry_manager.retry_with_reasoning(
                natural_query=args.natural_query,
                initial_sql=sql,
                execute_fn=execute_with_retry,
                max_rows=args.max_rows,
                initial_model=args.model
            )
            
            # Log retry attempts if any
            if len(attempts) > 1:
                print(f"\nRetry Summary: {len(attempts)} attempt(s) made", file=sys.stderr)
                for attempt in attempts:
                    status = "SUCCESS" if attempt.error is None else "FAILED"
                    print(f"  Attempt {attempt.attempt_number}: {status}", file=sys.stderr)
                    if attempt.error:
                        print(f"    Error: {attempt.error[:200]}", file=sys.stderr)
                    if attempt.result_count is not None:
                        print(f"    Results: {attempt.result_count} rows", file=sys.stderr)
                print("", file=sys.stderr)
        else:
            # Original behavior - no retry
            print("Executing query...", file=sys.stderr)
            df, error = await execute_query(db_instance, sql)
        
        if error:
            print(f"\nError executing query: {error}", file=sys.stderr)
            if RETRY_AVAILABLE and not args.enable_retry:
                print("Hint: Use --enable-retry to enable intelligent retry and self-correction", file=sys.stderr)
            sys.exit(1)
        
        # Ensure we have a valid DataFrame
        if df is None:
            print("\nError: Query execution returned None", file=sys.stderr)
            sys.exit(1)
        
        # Check for zero results (only warn if retry is disabled)
        if len(df) == 0:
            if not (RETRY_AVAILABLE and args.enable_retry):
                print("Warning: Query returned zero results.", file=sys.stderr)
                if RETRY_AVAILABLE:
                    print("Hint: Use --enable-retry to automatically try alternative query interpretations", file=sys.stderr)
        
        # Format and output results
        print("\n" + "=" * 70, file=sys.stderr)
        print("Query Results:", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Rows returned: {len(df)}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        
        if args.output_format == "csv":
            output = format_output_csv(df)
        elif args.output_format == "json":
            output = format_output_json(df)
        else:  # table
            output = format_output_table(df, max_display_rows=100)
        
        print(output, file=sys.stdout)
        
        # Close database connection
        await db_instance.close()
        
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

