"""
Logging configuration and formatters for the stock database server.

Provides:
- RequestFormatter: Custom formatter for access logs and general logs
- setup_logging: Basic logging setup
- setup_worker_logging: Worker-specific logging with process identification
- setup_parent_logging_with_queue: Multi-process safe logging with QueueListener
- setup_child_process_logging: Child process logging via QueueHandler
"""

import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from typing import Optional

# Global logging queue (for multi-process safe logging)
log_queue = None
queue_listener = None


class RequestFormatter(logging.Formatter):
    """
    Custom formatter that adapts format based on whether the log record
    contains request-specific fields (access logs) or is a general log message.
    """
    
    access_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] %(client_ip)s - \"%(request_line)s\" %(status_code)s %(response_size)s \"%(user_agent)s\" %(duration_ms)s - %(message)s"
    basic_log_format = "%(asctime)s [PID: %(process)d] [%(levelname)s] - %(message)s"

    def __init__(self):
        super().__init__(fmt=self.basic_log_format, datefmt=None, style='%')

    def format(self, record):
        """Format the log record, choosing format based on available attributes."""
        # Check if request-specific fields are present
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
            # Ensure duration_ms is set (default to 0 if not present)
            if not hasattr(record, 'duration_ms'):
                record.duration_ms = 0
            # Format duration_ms as string with "ms" suffix
            if hasattr(record, 'duration_ms'):
                record.duration_ms = f"{record.duration_ms:.0f}ms"
        else:
            self._style._fmt = self.basic_log_format
        
        # For Python 3.10+ LogRecord.message is already formatted.
        # For older versions, it might not be.
        # The default Formatter.format handles this.
        # We ensure the message attribute exists and is a string.
        if record.args:
            record.msg = record.msg % record.args
            record.args = ()  # Clear args after formatting into msg
        
        # Temporarily store original format string
        original_fmt = self._style._fmt

        # Choose format based on record attributes
        if hasattr(record, 'client_ip'):
            self._style._fmt = self.access_log_format
            # Ensure duration_ms is set (default to 0 if not present)
            if not hasattr(record, 'duration_ms'):
                record.duration_ms = "0ms"
            # Format duration_ms as string with "ms" suffix if it's a number
            elif isinstance(record.duration_ms, (int, float)):
                record.duration_ms = f"{record.duration_ms:.0f}ms"
        else:
            self._style._fmt = self.basic_log_format
        
        # Call superclass format
        result = logging.Formatter.format(self, record)
        
        # Restore original format string
        self._style._fmt = original_fmt
        return result


def setup_logging(log_file: Optional[str] = None, log_level_str: str = "INFO"):
    """
    Configure basic logging to stdout and optionally to a file.
    
    Args:
        log_file: Optional path to log file. If None, logs only to console.
        log_level_str: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Example:
        >>> setup_logging("/var/log/db_server.log", "INFO")
        >>> logger.info("Server started")
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Set root logger level so all child loggers inherit it
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Also set level on the module logger
    logger = logging.getLogger("db_server_logger")
    logger.setLevel(log_level)
    
    # Use the custom formatter
    custom_formatter = RequestFormatter()

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(custom_formatter)
    root_logger.addHandler(console_handler)
    # Also add to module logger if it doesn't have handlers
    if not logger.handlers:
        logger.addHandler(console_handler)

    if log_file:
        # File Handler - Rotate logs, 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(custom_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file} with level {log_level_str.upper()}")
    else:
        logger.info(f"Logging to console with level {log_level_str.upper()}")


def setup_worker_logging(worker_id: int, log_file: Optional[str] = None, log_level_str: str = "INFO"):
    """
    Setup logging for worker processes with process-specific identification.
    
    Each worker gets its own log file (if file logging is enabled) and includes
    the worker ID in all log messages.
    
    Args:
        worker_id: Unique identifier for this worker process
        log_file: Optional base path to log file. Worker ID will be appended.
        log_level_str: Log level as string
        
    Example:
        >>> setup_worker_logging(1, "/var/log/db_server.log", "INFO")
        # Creates /var/log/db_server_worker_1.log
    """
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Create worker-specific logger
    worker_logger = logging.getLogger("db_server_logger")
    worker_logger.setLevel(log_level)
    
    # Clear any existing handlers
    worker_logger.handlers.clear()
    
    # Create custom formatter that includes worker ID
    class WorkerFormatter(RequestFormatter):
        def __init__(self, worker_id: int):
            super().__init__()
            self.worker_id = worker_id
            
        def format(self, record):
            # Add worker ID to the record
            record.worker_id = self.worker_id
            
            # Update format strings to include worker ID
            if hasattr(record, 'client_ip'):
                self._style._fmt = f"%(asctime)s [PID:%(process)d] [Worker-{self.worker_id}] [%(levelname)s] %(client_ip)s - \"%(request_line)s\" %(status_code)s %(response_size)s \"%(user_agent)s\" - %(message)s"
            else:
                self._style._fmt = f"%(asctime)s [PID:%(process)d] [Worker-{self.worker_id}] [%(levelname)s] - %(message)s"
                
            return super(RequestFormatter, self).format(record)
    
    worker_formatter = WorkerFormatter(worker_id)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(worker_formatter)
    worker_logger.addHandler(console_handler)
    
    if log_file:
        # Create worker-specific log file
        log_path = Path(log_file)
        worker_log_file = log_path.parent / f"{log_path.stem}_worker_{worker_id}{log_path.suffix}"
        
        # File Handler - Rotate logs, 5MB per file, keep 5 backups
        file_handler = RotatingFileHandler(worker_log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(worker_formatter)
        worker_logger.addHandler(file_handler)
        
        worker_logger.info(f"Worker {worker_id} logging to file: {worker_log_file} with level {log_level_str.upper()}")
    else:
        worker_logger.info(f"Worker {worker_id} logging to console with level {log_level_str.upper()}")


def setup_parent_logging_with_queue(log_file: Optional[str] = None, log_level_str: str = "INFO"):
    """
    Configure parent process logging with a QueueListener for multi-process safety.
    
    This should be called in the parent process before forking workers. Workers can
    then use the returned queue to send log messages to the parent process.
    
    Args:
        log_file: Optional path to log file
        log_level_str: Log level as string
        
    Returns:
        Tuple of (queue, listener) for use by child processes
        
    Example:
        >>> log_queue, listener = setup_parent_logging_with_queue("/var/log/server.log")
        >>> # In child process:
        >>> setup_child_process_logging(worker_id, log_queue)
    """
    global log_queue, queue_listener

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger = logging.getLogger("db_server_logger")
    logger.setLevel(log_level)

    # Handlers used by the listener
    handlers = []
    formatter = RequestFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Start QueueListener
    from multiprocessing import Queue as MPQueue
    log_queue = MPQueue()
    queue_listener = QueueListener(log_queue, *handlers, respect_handler_level=False)
    queue_listener.start()

    # Parent can also log directly to the same handlers
    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    logger.info(f"Logging initialized with QueueListener. Level={log_level_str.upper()}")
    return log_queue, queue_listener


def setup_child_process_logging(worker_id: int, log_level_str: str = "INFO"):
    """
    Configure child process to send logs to parent's queue via QueueHandler.
    
    This should be called in each worker process after forking. Requires that
    setup_parent_logging_with_queue() was called in the parent first.
    
    Args:
        worker_id: Unique identifier for this worker
        log_level_str: Log level as string
        
    Example:
        >>> # In parent:
        >>> log_queue, listener = setup_parent_logging_with_queue()
        >>> # After fork, in child:
        >>> setup_child_process_logging(worker_id=1)
    """
    global log_queue
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    child_logger = logging.getLogger("db_server_logger")
    child_logger.setLevel(log_level)
    child_logger.handlers.clear()
    if log_queue is not None:
        child_logger.addHandler(QueueHandler(log_queue))
    else:
        # Fallback to console if queue not available
        fallback = logging.StreamHandler()
        fallback.setFormatter(RequestFormatter())
        child_logger.addHandler(fallback)
    child_logger.info(f"Child logger configured for worker {worker_id}")

