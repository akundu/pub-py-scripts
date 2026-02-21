"""
Centralized logging utilities for the stock database system.
Provides consistent logging format with timestamps and PIDs across all modules.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def get_logger(name: str, logger: Optional[logging.Logger] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with consistent formatting.
    
    Args:
        name: Logger name (usually module name)
        logger: Existing logger to use, if None creates new one
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    if logger is not None:
        # Ensure provided logger aligns with the desired/ambient level if specified/available
        if level is not None:
            logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        else:
            ambient_level = logging.getLogger().getEffectiveLevel()
            logger.setLevel(ambient_level)
        return logger
    
    # Create new logger
    log = logging.getLogger(name)
    
    # If logger already has handlers, still sync its level to the provided or ambient level
    if log.handlers:
        if level is not None:
            log_level = getattr(logging, level.upper(), logging.INFO)
            log.setLevel(log_level)
        else:
            ambient_level = logging.getLogger().getEffectiveLevel()
            log.setLevel(ambient_level)
        return log
    
    # Set level: if explicit level provided, use it; otherwise inherit root's effective level
    if level is not None:
        log_level = getattr(logging, level.upper(), logging.INFO)
    else:
        log_level = logging.getLogger().getEffectiveLevel()
    log.setLevel(log_level)
    
    # Create formatter with timestamp and PID
    formatter = logging.Formatter(
        '%(asctime)s [PID:%(process)d] [%(name)s] [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    log.addHandler(handler)
    # Prevent duplicate output: do not propagate to root (root would log again with default format)
    log.propagate = False

    return log


def log_info(message: str, logger: Optional[logging.Logger] = None, name: str = "stock_db"):
    """Log info message with consistent formatting."""
    log = get_logger(name, logger)
    log.info(message)


def log_warning(message: str, logger: Optional[logging.Logger] = None, name: str = "stock_db"):
    """Log warning message with consistent formatting."""
    log = get_logger(name, logger)
    log.warning(message)


def log_error(message: str, logger: Optional[logging.Logger] = None, name: str = "stock_db", exc_info: bool = False):
    """Log error message with consistent formatting."""
    log = get_logger(name, logger)
    log.error(message, exc_info=exc_info)


def log_debug(message: str, logger: Optional[logging.Logger] = None, name: str = "stock_db"):
    """Log debug message with consistent formatting."""
    log = get_logger(name, logger)
    log.debug(message)


def legacy_print(message: str, level: str = "INFO", name: str = "stock_db", 
                 logger: Optional[logging.Logger] = None, use_stderr: bool = True):
    """
    Legacy print function replacement that uses proper logging.
    Used to replace print() calls throughout the codebase.
    
    Args:
        message: Message to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        name: Logger name
        logger: Optional existing logger
        use_stderr: If True, uses stderr; if False, uses stdout
    """
    log = get_logger(name, logger, level)
    
    level_upper = level.upper()
    if level_upper == "DEBUG":
        log.debug(message)
    elif level_upper == "WARNING":
        log.warning(message)
    elif level_upper == "ERROR":
        log.error(message)
    else:  # INFO or default
        log.info(message)


# Worker-specific logger for multi-process environments
def get_worker_logger(worker_id: Optional[int] = None, name: str = "stock_db_worker", 
                     level: str = "INFO") -> logging.Logger:
    """
    Get a worker-specific logger for multi-process environments.
    
    Args:
        worker_id: Worker ID for process identification
        name: Base logger name
        level: Log level
        
    Returns:
        Configured logger with worker identification
    """
    if worker_id is not None:
        logger_name = f"{name}_worker_{worker_id}"
    else:
        logger_name = name
        
    return get_logger(logger_name, level=level)
