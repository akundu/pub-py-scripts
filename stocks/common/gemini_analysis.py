"""
Common Gemini AI analysis functionality for option spread analysis.

This module provides functions to run Gemini AI analysis on option spread data,
either from CSV files or pandas DataFrames.
"""

import html as html_escape
import html as html_escape
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

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
    model_alias: str = "flash",
    option_types: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Run Gemini AI analysis on a pandas DataFrame containing option spread data.
    
    Args:
        df: DataFrame with option spread data
        instruction: Custom instruction for Gemini AI (default: DEFAULT_GEMINI_INSTRUCTION)
        gemini_prog: Path to gemini_test.py script (default: tests/gemini_test.py relative to base_dir)
        base_dir: Base directory for resolving gemini_prog path (default: current working directory)
        model_alias: Gemini model alias to use ('flash', 'pro', 'flash-lite')
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
    
    # Filter DataFrame by option type and run analysis for each
    for opt_type in option_types:
        # Filter DataFrame for this option type
        if 'option_type' in df.columns:
            df_filtered = df[df['option_type'].str.lower() == opt_type.lower()].copy()
        else:
            # If no option_type column, assume all rows are of this type
            df_filtered = df.copy()
        
        if len(df_filtered) == 0:
            logger.warning(f"No data found for option type: {opt_type}")
            results[opt_type] = f"<div class='error'>No {opt_type} data available for analysis.</div>"
            continue
        
        # Write DataFrame to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            tmp_path = Path(tmp_file.name)
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
                
                logger.info(f"[GEMINI] Subprocess completed with return code: {result.returncode}")
                if result.stderr:
                    logger.debug(f"[GEMINI] Subprocess stderr: {result.stderr[:500]}")  # First 500 chars
                if result.stdout and len(result.stdout) > 0:
                    logger.debug(f"[GEMINI] Subprocess stdout length: {len(result.stdout)} chars")
                
                if result.returncode == 0:
                    # Extract HTML from output (Gemini returns HTML)
                    html_content = result.stdout
                    results[opt_type] = html_content
                    logger.info(f"Gemini analysis completed for {opt_type} options")
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
                    
                    logger.error(f"Gemini analysis failed for {opt_type}: {error_msg}")
                    # Format error message for HTML display
                    error_html = html_escape.escape(error_msg).replace('\n', '<br>')
                    results[opt_type] = (
                        f"<div class='error' style='padding: 20px; background: #fee; "
                        f"border: 1px solid #fcc; border-radius: 4px; margin: 10px 0;'>"
                        f"<strong style='color: #c00;'>Gemini Analysis Error:</strong><br><br>"
                        f"<pre style='white-space: pre-wrap; font-family: monospace; font-size: 12px;'>{error_html}</pre>"
                        f"</div>"
                    )
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Gemini analysis timed out for {opt_type}")
                results[opt_type] = "<div class='error'>Gemini analysis timed out (exceeded 5 minutes)</div>"
            except Exception as e:
                logger.error(f"Error running Gemini analysis for {opt_type}: {e}")
                results[opt_type] = f"<div class='error'>Error running Gemini analysis: {str(e)}</div>"
            finally:
                # Clean up temporary file
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
    
    return results


def run_gemini_analysis_on_file(
    input_file: Path,
    output_dir: Optional[Path] = None,
    instruction: str = DEFAULT_GEMINI_INSTRUCTION,
    gemini_prog: Optional[Path] = None,
    base_dir: Optional[Path] = None,
    model_alias: str = "flash",
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
        model_alias: Gemini model alias to use
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

