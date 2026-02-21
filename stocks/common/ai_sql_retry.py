"""
Modular AI SQL Query Retry and Reasoning System

This module provides intelligent retry and self-correction capabilities for AI-generated SQL queries.
It can be enabled or disabled easily, making it optional for the main query system.
"""

import re
import asyncio
from typing import Optional, Dict, Any, Tuple, List, Callable
from enum import Enum
from dataclasses import dataclass
import google.genai as genai
from google.genai.errors import APIError

from common.gemini_sql import (
    query_gemini_for_sql,
    MODEL_ALIASES,
    validate_sql_readonly,
    validate_sql_tables,
    ensure_limit_clause,
    normalize_sql_for_questdb,
    TABLE_SCHEMAS
)


class FailureType(Enum):
    """Types of failures that can occur during SQL query processing."""
    SQL_GENERATION_ERROR = "sql_generation_error"
    SQL_VALIDATION_ERROR = "sql_validation_error"
    SQL_EXECUTION_ERROR = "sql_execution_error"
    ZERO_RESULTS = "zero_results"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    UNKNOWN = "unknown"


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    sql: str
    error: Optional[str] = None
    failure_type: Optional[FailureType] = None
    result_count: Optional[int] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    enable_reasoning: bool = True
    enable_model_escalation: bool = True
    enable_zero_result_handling: bool = True
    initial_model: str = "flash"
    escalation_model: str = "pro"
    retry_delay: float = 0.5  # seconds between retries
    log_attempts: bool = True


class QueryRetryManager:
    """
    Manages retry logic and AI-powered self-correction for SQL queries.
    
    This class is designed to be modular - it can be used or not used depending on
    whether retry capabilities are desired.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None, logger=None):
        """
        Initialize the retry manager.
        
        Args:
            config: Retry configuration. If None, uses defaults.
            logger: Optional logger instance for logging retry attempts.
        """
        self.config = config or RetryConfig()
        self.logger = logger
        self.attempts: List[RetryAttempt] = []
    
    def _log(self, message: str, level: str = "info"):
        """Log a message if logger is available."""
        if self.logger:
            if level == "debug":
                self.logger.debug(message)
            elif level == "warning":
                self.logger.warning(message)
            elif level == "error":
                self.logger.error(message)
            else:
                self.logger.info(message)
    
    def categorize_failure(self, error: str, result_count: Optional[int] = None) -> FailureType:
        """
        Categorize a failure based on error message or result count.
        
        Args:
            error: Error message (if any)
            result_count: Number of results returned (None if query failed)
            
        Returns:
            FailureType enum value
        """
        if result_count == 0:
            return FailureType.ZERO_RESULTS
        
        if not error:
            return FailureType.UNKNOWN
        
        error_lower = error.lower()
        
        # SQL syntax/execution errors
        if any(phrase in error_lower for phrase in [
            "syntax error", "invalid syntax", "parse error",
            "column", "does not exist", "unknown column",
            "table", "relation", "does not exist"
        ]):
            return FailureType.SQL_EXECUTION_ERROR
        
        # Validation errors
        if any(phrase in error_lower for phrase in [
            "not read-only", "forbidden keyword", "invalid tables"
        ]):
            return FailureType.SQL_VALIDATION_ERROR
        
        # Timeout errors
        if any(phrase in error_lower for phrase in [
            "timeout", "timed out", "query canceled"
        ]):
            return FailureType.TIMEOUT
        
        # Connection errors
        if any(phrase in error_lower for phrase in [
            "connection", "network", "unreachable", "refused"
        ]):
            return FailureType.CONNECTION_ERROR
        
        # Generation errors
        if any(phrase in error_lower for phrase in [
            "failed to generate", "gemini", "api error", "empty response"
        ]):
            return FailureType.SQL_GENERATION_ERROR
        
        return FailureType.UNKNOWN
    
    def should_retry(self, attempt: RetryAttempt) -> bool:
        """
        Determine if we should retry based on the attempt and failure type.
        
        Args:
            attempt: The retry attempt information
            
        Returns:
            True if we should retry, False otherwise
        """
        if attempt.attempt_number >= self.config.max_attempts:
            return False
        
        # Don't retry zero results if handling is disabled
        if attempt.failure_type == FailureType.ZERO_RESULTS and not self.config.enable_zero_result_handling:
            return False
        
        # Always retry validation and execution errors
        if attempt.failure_type in [FailureType.SQL_VALIDATION_ERROR, FailureType.SQL_EXECUTION_ERROR]:
            return True
        
        # Retry generation errors
        if attempt.failure_type == FailureType.SQL_GENERATION_ERROR:
            return True
        
        # Retry zero results if enabled
        if attempt.failure_type == FailureType.ZERO_RESULTS and self.config.enable_zero_result_handling:
            return True
        
        # Retry timeouts and connection errors
        if attempt.failure_type in [FailureType.TIMEOUT, FailureType.CONNECTION_ERROR]:
            return True
        
        return False
    
    async def reason_and_fix_sql(
        self,
        natural_query: str,
        original_sql: str,
        error: Optional[str],
        failure_type: FailureType,
        attempt_number: int,
        model_alias: str,
        max_rows: int
    ) -> str:
        """
        Use AI reasoning to diagnose and fix SQL query issues.
        
        Args:
            natural_query: Original natural language query
            original_sql: The SQL that failed
            error: Error message (if any)
            failure_type: Type of failure
            attempt_number: Current attempt number
            model_alias: Model to use for fixing
            max_rows: Maximum rows to return
            
        Returns:
            Corrected SQL query string
        """
        if not self.config.enable_reasoning:
            # Without reasoning, just return original (shouldn't be called)
            return original_sql
        
        # Build context from previous attempts
        attempt_history = "\n".join([
            f"Attempt {a.attempt_number}: {a.sql[:100]}... "
            f"(Error: {a.error[:100] if a.error else 'None'})"
            for a in self.attempts[-2:]  # Last 2 attempts for context
        ])
        
        # Build diagnostic prompt
        if failure_type == FailureType.ZERO_RESULTS:
            diagnostic_prompt = f"""The SQL query returned zero results, but this might be unexpected.

Original Query: {natural_query}
Generated SQL: {original_sql}

Previous attempts:
{attempt_history if attempt_history else "None"}

Please analyze why this might return zero results and suggest fixes:
1. Check if the table selection is correct (realtime_data vs daily_prices vs hourly_prices)
2. Verify date/timestamp filters are appropriate
3. Check if ticker symbols are correct
4. Consider if the time range is too narrow
5. Verify column names match the schema

Generate a corrected SQL query that addresses these potential issues."""
        else:
            diagnostic_prompt = f"""The SQL query failed with an error. Please diagnose and fix it.

Original Query: {natural_query}
Generated SQL: {original_sql}
Error: {error}

Previous attempts:
{attempt_history if attempt_history else "None"}

Please:
1. Analyze the error message
2. Identify the root cause
3. Generate a corrected SQL query that fixes the issue
4. Ensure the query follows all requirements (read-only, valid tables, proper LIMIT)"""
        
        full_prompt = f"""You are a SQL query expert specializing in QuestDB (PostgreSQL-compatible) queries for stock market data.

## Database Schema:
{TABLE_SCHEMAS}

## Task: Fix a SQL Query

{diagnostic_prompt}

## Requirements:
1. Generate a valid PostgreSQL-compatible SELECT query
2. Choose the appropriate table based on query context
3. Use proper column names from the schema
4. Include appropriate WHERE clauses
5. Include a LIMIT clause with maximum {max_rows} rows
6. Ensure the query is read-only
7. Return ONLY the SQL query, no explanations, no markdown formatting, just the raw SQL

Generate the corrected SQL query now:
"""
        
        try:
            client = genai.Client()
            model_id = MODEL_ALIASES.get(model_alias, MODEL_ALIASES["flash"])
            
            response = client.models.generate_content(
                model=model_id,
                contents=[full_prompt]
            )
            
            if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                raise Exception("Gemini returned empty response for SQL fix")
            
            sql = response.text.strip()
            
            # Clean up SQL
            sql = re.sub(r'^```sql\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
            sql = re.sub(r'^```\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
            sql = re.sub(r'```\s*$', '', sql, flags=re.IGNORECASE | re.MULTILINE)
            sql = sql.strip()
            
            # Apply normalization and validation
            sql = normalize_sql_for_questdb(sql)
            
            is_valid, error_msg = validate_sql_readonly(sql)
            if not is_valid:
                raise ValueError(f"Fixed SQL is not read-only: {error_msg}")
            
            is_valid, error_msg = validate_sql_tables(sql)
            if not is_valid:
                raise ValueError(f"Fixed SQL references invalid tables: {error_msg}")
            
            sql = ensure_limit_clause(sql, max_rows)
            
            self._log(f"AI reasoning generated corrected SQL (attempt {attempt_number})")
            return sql
            
        except Exception as e:
            self._log(f"AI reasoning failed: {e}", "warning")
            raise Exception(f"Failed to generate corrected SQL: {e}")
    
    def get_model_for_attempt(self, attempt_number: int) -> str:
        """
        Get the model to use for a given attempt number.
        Implements model escalation if enabled.
        
        Args:
            attempt_number: Current attempt number (1-indexed)
            
        Returns:
            Model alias to use
        """
        if not self.config.enable_model_escalation:
            return self.config.initial_model
        
        # Escalate to pro model after first attempt if enabled
        if attempt_number > 1:
            return self.config.escalation_model
        
        return self.config.initial_model
    
    async def retry_with_reasoning(
        self,
        natural_query: str,
        initial_sql: str,
        execute_fn: Callable,
        max_rows: int,
        initial_model: str = "flash"
    ) -> Tuple[Any, Optional[str], List[RetryAttempt]]:
        """
        Execute a query with retry and reasoning capabilities.
        
        Args:
            natural_query: Original natural language query
            initial_sql: Initial SQL query to try
            execute_fn: Async function that takes (sql: str) and returns (result, error)
                       Should return (result, None) on success or (None, error_msg) on failure
            max_rows: Maximum rows to return
            initial_model: Initial model alias to use
            
        Returns:
            Tuple of (result, final_error, attempts_list)
            - result: Query result (DataFrame or similar) if successful, None otherwise
            - final_error: Error message if all attempts failed, None if successful
            - attempts_list: List of all retry attempts made
        """
        self.attempts = []
        current_sql = initial_sql
        current_model = initial_model
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            self._log(f"Attempt {attempt_num}/{self.config.max_attempts}: Executing SQL query...")
            
            # Execute the query
            result, error = await execute_fn(current_sql)
            
            # Determine result count if result is available
            result_count = None
            if result is not None:
                try:
                    # Handle pandas DataFrames and similar objects
                    if hasattr(result, 'empty'):
                        # DataFrame-like object
                        result_count = 0 if result.empty else len(result)
                    elif hasattr(result, '__len__'):
                        result_count = len(result)
                    else:
                        # Single value result
                        result_count = 1 if result else 0
                except Exception:
                    result_count = None
            
            # Categorize failure
            failure_type = self.categorize_failure(error or "", result_count)
            
            # Record attempt
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                sql=current_sql,
                error=error,
                failure_type=failure_type,
                result_count=result_count
            )
            self.attempts.append(attempt)
            
            # Check if we succeeded
            if error is None and (result_count is None or result_count > 0 or not self.config.enable_zero_result_handling):
                self._log(f"Query succeeded on attempt {attempt_num}")
                return result, None, self.attempts
            
            # Handle zero results
            if result_count == 0 and self.config.enable_zero_result_handling:
                self._log(f"Zero results on attempt {attempt_num}, checking if retry needed...")
                # For zero results, we'll try to fix the query
                if attempt_num < self.config.max_attempts:
                    try:
                        current_model = self.get_model_for_attempt(attempt_num + 1)
                        current_sql = await self.reason_and_fix_sql(
                            natural_query,
                            current_sql,
                            "Query returned zero results - this may indicate incorrect table selection, date filters, or ticker symbols",
                            failure_type,
                            attempt_num + 1,
                            current_model,
                            max_rows
                        )
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                    except Exception as fix_error:
                        self._log(f"Failed to generate fix for zero results: {fix_error}", "warning")
                        # Continue to next attempt or return
                        if attempt_num < self.config.max_attempts:
                            continue
            
            # Check if we should retry
            if not self.should_retry(attempt):
                break
            
            # Generate fix for errors
            if error and self.config.enable_reasoning:
                try:
                    current_model = self.get_model_for_attempt(attempt_num + 1)
                    current_sql = await self.reason_and_fix_sql(
                        natural_query,
                        current_sql,
                        error,
                        failure_type,
                        attempt_num + 1,
                        current_model,
                        max_rows
                    )
                    self._log(f"Generated corrected SQL for attempt {attempt_num + 1}")
                    await asyncio.sleep(self.config.retry_delay)
                except Exception as fix_error:
                    self._log(f"Failed to generate fix: {fix_error}", "warning")
                    # Try regenerating from scratch
                    if attempt_num < self.config.max_attempts:
                        try:
                            current_model = self.get_model_for_attempt(attempt_num + 1)
                            current_sql = query_gemini_for_sql(natural_query, current_model, max_rows)
                            self._log(f"Regenerated SQL from scratch for attempt {attempt_num + 1}")
                            await asyncio.sleep(self.config.retry_delay)
                        except Exception as regen_error:
                            self._log(f"Failed to regenerate SQL: {regen_error}", "error")
            elif error:
                # No reasoning, but we can retry with same SQL (for transient errors)
                if failure_type in [FailureType.CONNECTION_ERROR, FailureType.TIMEOUT]:
                    self._log(f"Retrying due to {failure_type.value}...")
                    await asyncio.sleep(self.config.retry_delay * attempt_num)  # Exponential backoff
                else:
                    break
        
        # All attempts failed
        final_error = self.attempts[-1].error if self.attempts else "Unknown error"
        if not final_error and self.attempts and self.attempts[-1].result_count == 0:
            final_error = "Query returned zero results after all retry attempts"
        
        self._log(f"All {len(self.attempts)} attempts failed. Final error: {final_error}", "warning")
        return None, final_error, self.attempts

