"""
Common Gemini AI analysis functionality for option spread analysis.

This module provides functions to run Gemini AI analysis on option spread data,
either from CSV files or pandas DataFrames.
"""

import html as html_escape
import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Get logger - ensure it inherits from root logger to get log level
logger = logging.getLogger(__name__)
# Ensure logger level propagates from root logger
logger.setLevel(logging.NOTSET)  # NOTSET means inherit from parent

# Default Gemini instruction for spread option analysis
DEFAULT_GEMINI_INSTRUCTION = (
    "given the provided file of spread option trades possible, choose the 5 best set "
    "(based on realism of possibility of it happening) of dealing with risk and being "
    "aggressive and being conservative. focus on the intrinsic characteristics of each "
    "spread (strike prices, premiums, days to expiry and theta and delta), the underlying "
    "stock's volatility and market cap, and the reported net_daily_premi as an indicator "
    "of potential theta gain/loss. assume these represent **calendar spreads**, where you "
    "sell the shorter-dated option and buy the longer-dated option of the same type. the "
    "net cost (debit) per share is generally (long leg premium - short leg premium). a "
    "positive net_daily_premi suggests a theoretical daily gain from time decay. also, "
    "use the short_daily_premium in the analysis. make sure to only pick realistic "
    "situations of being able to procure those things. also, give me 3 examples of risky "
    "and 3 examples of conservative choices. Write the responses in a HTML form that I "
    "can save to a .html file. make sure to cover the examples of 3 per put spread and "
    "call spread."
)


def run_gemini_analysis_on_dataframe(
    df: pd.DataFrame,
    instruction: str = DEFAULT_GEMINI_INSTRUCTION,
    gemini_prog: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_alias: str = "gemini-3",
    option_types: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Run Gemini AI analysis on a pandas DataFrame containing option spread data.
    
    Args:
        df: DataFrame with option spread data
        instruction: Custom instruction for Gemini AI (default: DEFAULT_GEMINI_INSTRUCTION)
        gemini_prog: Path to gemini_test.py script (default: tests/gemini_test.py relative to base_dir)
        base_dir: Base directory for resolving gemini_prog path (default: current working directory)
        model_alias: Gemini model alias to use ('flash', 'pro', 'flash-lite', 'gemini-3')
        option_types: List of option types to analyze (default: ['call', 'put'])
    
    Returns:
        Dictionary mapping option_type to generated HTML content string
        Example: {'call': '<html>...', 'put': '<html>...'}
    """
    if option_types is None:
        option_types = ['call', 'put']
    
    if base_dir is None:
        base_dir = Path.cwd()
    
    if gemini_prog is None:
        gemini_prog = base_dir / "tests" / "gemini_test.py"
    
    if not gemini_prog.exists():
        raise FileNotFoundError(f"Gemini program not found: {gemini_prog}")
    
    results = {}
    
    logger.info(f"[GEMINI] Starting analysis for {len(option_types)} option types: {option_types}")
    logger.info(f"[GEMINI] Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
    if 'option_type' in df.columns:
        value_counts = df['option_type'].value_counts()
        logger.info(f"[GEMINI] Option type distribution in input DataFrame:")
        for opt_type_val, count in value_counts.items():
            logger.info(f"[GEMINI]   '{opt_type_val}': {count} rows")
        logger.debug(f"[GEMINI] Option type distribution (dict): {value_counts.to_dict()}")
        
        # Check for case variations
        unique_lower = df['option_type'].str.lower().unique()
        logger.debug(f"[GEMINI] Unique option_type values (lowercase): {unique_lower.tolist()}")
    else:
        logger.warning(f"[GEMINI] WARNING: 'option_type' column NOT found in DataFrame!")
        logger.warning(f"[GEMINI] Available columns: {list(df.columns)}")
    
    def run_single_analysis(opt_type: str) -> tuple[str, str]:
        """Run Gemini analysis for a single option type. Returns (opt_type, html_content_or_error)."""
        logger.info(f"[GEMINI] ===== Processing {opt_type.upper()} analysis (parallel) =====")
        logger.debug(f"[GEMINI] Filtering DataFrame for {opt_type} option type...")
        logger.debug(f"[GEMINI] Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        # Filter DataFrame for this option type
        if 'option_type' in df.columns:
            # Show what unique values exist in option_type column
            unique_option_types = df['option_type'].str.lower().unique()
            logger.info(f"[GEMINI] Unique option_type values in DataFrame: {unique_option_types.tolist()}")
            logger.debug(f"[GEMINI] Looking for option_type == '{opt_type.lower()}' (case-insensitive)")
            
            # Filter with case-insensitive comparison
            df_filtered = df[df['option_type'].str.lower() == opt_type.lower()].copy()
            
            logger.info(f"[GEMINI] Filtered DataFrame: {len(df_filtered)} rows for {opt_type} (out of {len(df)} total rows)")
            logger.debug(f"[GEMINI] Filter mask applied: {df['option_type'].str.lower() == opt_type.lower()}")
            
            # Show sample of what was filtered
            if len(df_filtered) > 0:
                logger.debug(f"[GEMINI] Sample of filtered {opt_type} data (first 3 rows):")
                for idx, row in df_filtered.head(3).iterrows():
                    logger.debug(f"[GEMINI]   Row {idx}: option_type='{row.get('option_type', 'N/A')}', ticker='{row.get('ticker', 'N/A')}'")
            else:
                # Show what option_type values actually exist
                logger.warning(f"[GEMINI] No rows match {opt_type}! Showing option_type value counts:")
                value_counts = df['option_type'].value_counts()
                for opt_val, count in value_counts.items():
                    logger.warning(f"[GEMINI]   '{opt_val}': {count} rows")
        else:
            # If no option_type column, assume all rows are of this type
            logger.warning(f"[GEMINI] No 'option_type' column found, using all {len(df)} rows for {opt_type} analysis")
            logger.warning(f"[GEMINI] DataFrame columns: {list(df.columns)}")
            df_filtered = df.copy()
        
        if len(df_filtered) == 0:
            logger.error(f"[GEMINI] No data found for option type: {opt_type}")
            logger.error(f"[GEMINI] This means the filtered DataFrame is empty. Check option_type column values.")
            error_html = f"<div class='error'>No {opt_type} data available for analysis.</div>"
            logger.debug(f"[GEMINI] Added error placeholder for {opt_type} to results")
            return (opt_type, error_html)
        
        # Create a temporary directory for saving input files for inspection
        # Save to a location that persists for debugging
        debug_temp_dir = Path(tempfile.gettempdir()) / "gemini_analysis_debug"
        debug_temp_dir.mkdir(exist_ok=True)
        
        # Create a timestamped filename for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_file = debug_temp_dir / f"input_{opt_type}_{timestamp}.csv"
        
        # Also save a copy for debugging BEFORE writing to temp file
        try:
            df_filtered.to_csv(debug_file, index=False)
            logger.info(f"[GEMINI] ✓ Saved {opt_type} input CSV for inspection: {debug_file}")
            logger.info(f"[GEMINI]   Debug file location: {debug_file}")
            logger.info(f"[GEMINI]   File size: {debug_file.stat().st_size} bytes, {len(df_filtered)} rows, {len(df_filtered.columns)} columns")
            logger.info(f"[GEMINI]   Columns in saved file: {list(df_filtered.columns)}")
            
            # Show first few rows as a sample
            if len(df_filtered) > 0:
                logger.debug(f"[GEMINI]   First row sample (first 5 columns):")
                first_row = df_filtered.iloc[0]
                sample_cols = list(df_filtered.columns)[:5]
                for col in sample_cols:
                    logger.debug(f"[GEMINI]     {col}: {first_row.get(col, 'N/A')}")
        except Exception as e:
            logger.error(f"[GEMINI] ✗ Could not save debug file: {e}")
            import traceback
            logger.debug(f"[GEMINI] Debug file save exception:\n{traceback.format_exc()}")
        
        # Write DataFrame to temporary CSV file (for subprocess)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            # Write the actual file for subprocess
            df_filtered.to_csv(tmp_path, index=False)
            logger.debug(f"[GEMINI] Wrote {len(df_filtered)} rows to temporary CSV for subprocess: {tmp_path}")
            try:
                # Write DataFrame to CSV
                df_filtered.to_csv(tmp_path, index=False)
                
                # Run Gemini analysis
                command = [
                    sys.executable,
                    str(gemini_prog),
                    "--instruction",
                    instruction,
                    "--file",
                    str(tmp_path),
                    "--model",
                    model_alias,
                ]
                
                logger.info(f"[GEMINI] Running Gemini analysis for {opt_type} options...")
                logger.info(f"[GEMINI] Command: {' '.join(command)}")
                logger.info(f"[GEMINI] Working directory: {base_dir}")
                logger.info(f"[GEMINI] Input file: {tmp_path} ({len(df_filtered)} rows)")
                
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    cwd=str(base_dir),
                    timeout=300  # 5 minute timeout
                )
                
                logger.info(f"[GEMINI] Subprocess completed with return code: {result.returncode} for {opt_type}")
                if result.stderr:
                    logger.debug(f"[GEMINI] {opt_type.upper()} subprocess stderr: {result.stderr[:500]}")  # First 500 chars
                if result.stdout and len(result.stdout) > 0:
                    logger.debug(f"[GEMINI] {opt_type.upper()} subprocess stdout length: {len(result.stdout)} chars")
                
                if result.returncode == 0:
                    # Extract HTML from output (Gemini returns HTML)
                    html_content = result.stdout
                    
                    # Clean up any debug output that might have leaked into stdout
                    # Remove lines that look like debug output (e.g., "--- Querying Model:", "GENERATED RESPONSE", etc.)
                    lines = html_content.split('\n')
                    cleaned_lines = []
                    skip_until_html = True
                    for line in lines:
                        # Skip debug separator lines and debug messages
                        if skip_until_html:
                            # Look for start of HTML content (tags like <html>, <div>, <h1>, etc.)
                            stripped = line.strip()
                            if stripped.startswith('<') or stripped.startswith('```html'):
                                skip_until_html = False
                                # If it's a markdown code block start, skip it
                                if stripped.startswith('```html'):
                                    continue
                                # If it's a markdown code block end, skip it
                                if stripped == '```':
                                    continue
                            else:
                                # Skip debug lines
                                if any(debug_marker in line for debug_marker in [
                                    '--- Querying Model:',
                                    'GENERATED RESPONSE',
                                    '=' * 50,
                                    '=' * 70,
                                ]):
                                    continue
                                # If we hit a non-debug line before HTML, keep it (might be part of content)
                                if line.strip() and not line.strip().startswith('---'):
                                    skip_until_html = False
                        
                        if not skip_until_html:
                            cleaned_lines.append(line)
                    
                    html_content = '\n'.join(cleaned_lines).strip()
                    
                    logger.info(f"[GEMINI] ✓ {opt_type.upper()} analysis completed successfully")
                    logger.debug(f"[GEMINI] {opt_type.upper()} HTML content length: {len(html_content)} characters")
                    # Show first 200 chars of output in debug mode
                    if logger.isEnabledFor(logging.DEBUG):
                        preview = html_content[:200].replace('\n', ' ').strip()
                        logger.debug(f"[GEMINI] {opt_type.upper()} HTML preview: {preview}...")
                    return (opt_type, html_content)
                else:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    
                    # Provide helpful error messages for common issues
                    if "ModuleNotFoundError" in error_msg and "google" in error_msg:
                        error_msg = (
                            "Gemini API package not installed. Please install it with: "
                            "pip install google-genai\n\n"
                            f"Original error: {error_msg}"
                        )
                    elif "GEMINI_API_KEY" in error_msg or "API key" in error_msg.lower():
                        error_msg = (
                            "Gemini API key not configured. Please set the GEMINI_API_KEY "
                            "environment variable.\n\n"
                            f"Original error: {error_msg}"
                        )
                    
                    logger.error(f"[GEMINI] ✗ {opt_type.upper()} analysis failed: {error_msg}")
                    logger.debug(f"[GEMINI] {opt_type.upper()} error details - return code: {result.returncode}")
                    if result.stderr:
                        logger.debug(f"[GEMINI] {opt_type.upper()} stderr: {result.stderr}")
                    if result.stdout:
                        logger.debug(f"[GEMINI] {opt_type.upper()} stdout: {result.stdout[:500]}")
                    
                    # Format error message for HTML display
                    error_html = html_escape.escape(error_msg).replace('\n', '<br>')
                    error_result = (
                        f"<div class='error' style='padding: 20px; background: #fee; "
                        f"border: 1px solid #fcc; border-radius: 4px; margin: 10px 0;'>"
                        f"<strong style='color: #c00;'>Gemini Analysis Error:</strong><br><br>"
                        f"<pre style='white-space: pre-wrap; font-family: monospace; font-size: 12px;'>{error_html}</pre>"
                        f"</div>"
                    )
                    logger.debug(f"[GEMINI] Added error result for {opt_type} to results dictionary")
                    return (opt_type, error_result)
                    
            except subprocess.TimeoutExpired:
                logger.error(f"[GEMINI] ✗ {opt_type.upper()} analysis timed out (exceeded 5 minutes)")
                timeout_error = "<div class='error'>Gemini analysis timed out (exceeded 5 minutes)</div>"
                logger.debug(f"[GEMINI] Added timeout error result for {opt_type} to results dictionary")
                return (opt_type, timeout_error)
            except Exception as e:
                logger.error(f"[GEMINI] ✗ Error running {opt_type.upper()} analysis: {e}")
                import traceback
                logger.debug(f"[GEMINI] {opt_type.upper()} exception traceback:\n{traceback.format_exc()}")
                exception_error = f"<div class='error'>Error running Gemini analysis: {str(e)}</div>"
                logger.debug(f"[GEMINI] Added exception error result for {opt_type} to results dictionary")
                return (opt_type, exception_error)
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                    logger.debug(f"[GEMINI] Cleaned up temporary file: {tmp_path}")
                except Exception:
                    pass
        
        logger.info(f"[GEMINI] ===== Completed {opt_type.upper()} analysis =====")
        return (opt_type, f"<div class='error'>Unexpected error in {opt_type} analysis</div>")
    
    # Run analyses in parallel using ThreadPoolExecutor
    logger.info(f"[GEMINI] Running {len(option_types)} analyses in parallel...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(option_types)) as executor:
        # Submit all analyses
        future_to_opt_type = {executor.submit(run_single_analysis, opt_type): opt_type for opt_type in option_types}
        logger.debug(f"[GEMINI] Submitted {len(future_to_opt_type)} analysis tasks to thread pool")
        
        # Collect results as they complete
        for future in as_completed(future_to_opt_type):
            opt_type = future_to_opt_type[future]
            try:
                result_opt_type, html_content = future.result()
                results[result_opt_type] = html_content
                logger.info(f"[GEMINI] ✓ Collected result for {result_opt_type.upper()} from parallel execution")
            except Exception as e:
                logger.error(f"[GEMINI] ✗ Exception in parallel execution for {opt_type}: {e}")
                import traceback
                logger.debug(f"[GEMINI] {opt_type.upper()} parallel execution exception:\n{traceback.format_exc()}")
                results[opt_type] = f"<div class='error'>Error in parallel execution: {str(e)}</div>"
    
    elapsed_time = time.time() - start_time
    logger.info(f"[GEMINI] All parallel analyses completed in {elapsed_time:.2f} seconds")
    
    logger.info(f"[GEMINI] All analyses complete. Results keys: {list(results.keys())}")
    logger.debug(f"[GEMINI] Final results summary:")
    for opt_type, content in results.items():
        content_type = "HTML" if content.startswith('<') else "Error"
        logger.debug(f"[GEMINI]   - {opt_type}: {content_type} ({len(content)} chars)")
    
    return results


def run_gemini_analysis_on_file(
    input_file: Path,
    output_dir: Optional[Path] = None,
    instruction: str = DEFAULT_GEMINI_INSTRUCTION,
    gemini_prog: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_alias: str = "gemini-3",
    option_types: Optional[list[str]] = None,
    force_run: bool = False,
    market_hours: bool = True,
    cooldown_seconds: int = 3600,
    last_run_file: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Run Gemini AI analysis on a CSV file containing option spread data.
    
    This is the original file-based interface, maintained for backward compatibility.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save output HTML files (optional)
        instruction: Custom instruction for Gemini AI
        gemini_prog: Path to gemini_test.py script
        base_dir: Base directory for resolving paths
        model_alias: Gemini model alias to use ('flash', 'pro', 'flash-lite', 'gemini-3')
        option_types: List of option types to analyze (default: ['call', 'put'])
        force_run: Force analysis even if cooldown hasn't expired
        market_hours: Whether market is currently open (affects whether analysis runs)
        cooldown_seconds: Minimum seconds between runs
        last_run_file: Path to file storing last run timestamp
    
    Returns:
        Dictionary mapping option_type to output file path
    """
    if option_types is None:
        option_types = ['call', 'put']
    
    if base_dir is None:
        base_dir = Path.cwd()
    
    if gemini_prog is None:
        gemini_prog = base_dir / "tests" / "gemini_test.py"
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not gemini_prog.exists():
        raise FileNotFoundError(f"Gemini program not found: {gemini_prog}")
    
    # Check cooldown
    if not force_run:
        if not market_hours:
            logger.info("Skipping Gemini analysis: outside market hours")
            return {}
        
        if last_run_file and last_run_file.exists():
            try:
                last_epoch = int(last_run_file.read_text().strip() or "0")
                current_epoch = int(time.time())
                if current_epoch - last_epoch < cooldown_seconds:
                    remaining = cooldown_seconds - (current_epoch - last_epoch)
                    logger.info(f"Skipping Gemini analysis: ran recently ({remaining}s remaining on cooldown)")
                    return {}
            except (ValueError, Exception) as e:
                logger.warning(f"Error reading last run file: {e}")
    
    # Create temporary directory for subset files
    temp_dir = Path(tempfile.gettempdir())
    analysis_outputs = {}
    
    for opt_type in option_types:
        # Create subset file for this option type
        subset_file = temp_dir / f"{input_file.stem}.{opt_type}.csv"
        
        try:
            # Write subset of CSV file
            df = pd.read_csv(input_file)
            if 'option_type' in df.columns:
                df_subset = df[df['option_type'].str.lower() == opt_type.lower()]
            else:
                # If no option_type column, write all rows
                df_subset = df
            
            if len(df_subset) == 0:
                logger.warning(f"No {opt_type} data in input file")
                continue
            
            df_subset.to_csv(subset_file, index=False)
            
            # Determine output file path
            if output_dir:
                html_output = output_dir / f"analysis.{opt_type}.html"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                html_output = temp_dir / f"analysis.{opt_type}.html"
            
            # Run Gemini analysis
            command = [
                sys.executable,
                str(gemini_prog),
                "--instruction",
                instruction,
                "--file",
                str(subset_file),
                "--model",
                model_alias,
            ]
            
            logger.info(f"Executing Gemini analysis for {opt_type} options...")
            with html_output.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(base_dir),
                    timeout=300
                )
            
            if result.returncode == 0:
                analysis_outputs[opt_type] = html_output
                logger.info(f"Gemini analysis ({opt_type}) completed successfully")
            else:
                logger.error(f"Gemini analysis ({opt_type}) failed with return code {result.returncode}")
                
        except FileNotFoundError:
            logger.error(f"Input file not found for Gemini analysis: {input_file}")
            break
        except subprocess.TimeoutExpired:
            logger.error(f"Gemini analysis timed out for {opt_type}")
        except Exception as e:
            logger.error(f"Error running Gemini analysis for {opt_type}: {e}")
        finally:
            # Clean up subset file
            try:
                if subset_file.exists():
                    subset_file.unlink()
            except Exception:
                pass
    
    # Update last run timestamp
    if last_run_file:
        try:
            last_run_file.write_text(str(int(time.time())), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Error writing last run file: {e}")
    
    return analysis_outputs

