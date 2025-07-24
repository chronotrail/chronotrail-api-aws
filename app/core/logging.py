"""
Comprehensive logging configuration for ChronoTrail API.

This module provides structured logging with proper formatting, filtering,
and context management for debugging and monitoring purposes.
"""
import logging
import sys
import os
from typing import Any, Dict, Optional, Union
from datetime import datetime
from uuid import uuid4

import structlog
from structlog.types import FilteringBoundLogger

from app.core.config import settings


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context to log record if available."""
        # Add default values if not present
        if not hasattr(record, 'request_id'):
            record.request_id = getattr(self, '_request_id', 'unknown')
        if not hasattr(record, 'user_id'):
            record.user_id = getattr(self, '_user_id', 'anonymous')
        if not hasattr(record, 'endpoint'):
            record.endpoint = getattr(self, '_endpoint', 'unknown')
        
        return True


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_KEYS = {
        'password', 'token', 'secret', 'key', 'authorization',
        'access_token', 'refresh_token', 'api_key', 'oauth_token',
        'identity_token', 'jwt', 'bearer'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Remove sensitive information from log record."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            # Basic string replacement for common sensitive patterns
            for sensitive_key in self.SENSITIVE_KEYS:
                if sensitive_key in record.msg.lower():
                    # Replace potential token values with [REDACTED]
                    import re
                    pattern = rf'({sensitive_key}["\']?\s*[:=]\s*["\']?)([^"\'\s,}}]+)'
                    record.msg = re.sub(pattern, r'\1[REDACTED]', record.msg, flags=re.IGNORECASE)
        
        return True


def add_request_context(
    logger: FilteringBoundLogger,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None
) -> FilteringBoundLogger:
    """
    Add request context to a logger instance.
    
    Args:
        logger: The logger instance
        request_id: Optional request ID
        user_id: Optional user ID
        endpoint: Optional endpoint path
        
    Returns:
        Logger with bound context
    """
    context = {}
    if request_id:
        context['request_id'] = request_id
    if user_id:
        context['user_id'] = user_id
    if endpoint:
        context['endpoint'] = endpoint
    
    return logger.bind(**context)


def configure_logging(
    log_level: Optional[str] = None,
    enable_json_logs: Optional[bool] = None,
    enable_request_logging: Optional[bool] = None
) -> None:
    """
    Configure comprehensive structured logging for the application.
    
    Args:
        log_level: Optional log level override
        enable_json_logs: Optional JSON logging override
        enable_request_logging: Optional request logging override
    """
    # Determine configuration from settings or parameters
    log_level = log_level or getattr(settings, 'LOG_LEVEL', 'INFO')
    enable_json_logs = enable_json_logs if enable_json_logs is not None else getattr(settings, 'ENABLE_JSON_LOGS', True)
    enable_request_logging = enable_request_logging if enable_request_logging is not None else getattr(settings, 'ENABLE_REQUEST_LOGGING', True)
    
    # Configure processors based on environment
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add JSON renderer for production, console renderer for development
    if enable_json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    log_format = "%(message)s" if enable_json_logs else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create root logger configuration
    logging.basicConfig(
        format=log_format,
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True
    )
    
    # Add filters to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(RequestContextFilter())
    root_logger.addFilter(SecurityFilter())
    
    # Configure specific loggers
    configure_third_party_loggers()
    
    # Log configuration
    logger = get_logger(__name__)
    logger.info(
        "Logging configured",
        log_level=log_level,
        json_logs=enable_json_logs,
        request_logging=enable_request_logging
    )


def configure_third_party_loggers() -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


def log_function_call(
    logger: structlog.stdlib.BoundLogger,
    function_name: str,
    args: Optional[Dict[str, Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    level: str = "debug"
) -> None:
    """
    Log a function call with parameters.
    
    Args:
        logger: Logger instance
        function_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
        level: Log level
    """
    log_method = getattr(logger, level.lower(), logger.debug)
    
    log_data = {
        "function": function_name,
        "call_id": str(uuid4())[:8]
    }
    
    if args:
        log_data["args"] = args
    if kwargs:
        log_data["kwargs"] = kwargs
    
    log_method("Function called", **log_data)


def log_function_result(
    logger: structlog.stdlib.BoundLogger,
    function_name: str,
    result: Any = None,
    execution_time: Optional[float] = None,
    level: str = "debug"
) -> None:
    """
    Log a function result.
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        result: Function result (will be sanitized)
        execution_time: Execution time in seconds
        level: Log level
    """
    log_method = getattr(logger, level.lower(), logger.debug)
    
    log_data = {
        "function": function_name,
        "success": True
    }
    
    if execution_time is not None:
        log_data["execution_time_seconds"] = execution_time
    
    # Sanitize result for logging
    if result is not None:
        if hasattr(result, 'model_dump'):
            # Pydantic model
            log_data["result_type"] = type(result).__name__
        elif isinstance(result, (dict, list)):
            log_data["result_type"] = type(result).__name__
            log_data["result_size"] = len(result)
        else:
            log_data["result_type"] = type(result).__name__
    
    log_method("Function completed", **log_data)


def log_function_error(
    logger: structlog.stdlib.BoundLogger,
    function_name: str,
    error: Exception,
    execution_time: Optional[float] = None,
    level: str = "error"
) -> None:
    """
    Log a function error.
    
    Args:
        logger: Logger instance
        function_name: Name of the function
        error: Exception that occurred
        execution_time: Execution time in seconds
        level: Log level
    """
    log_method = getattr(logger, level.lower(), logger.error)
    
    log_data = {
        "function": function_name,
        "success": False,
        "error_type": type(error).__name__,
        "error_message": str(error)
    }
    
    if execution_time is not None:
        log_data["execution_time_seconds"] = execution_time
    
    log_method("Function failed", **log_data)


class LoggingContext:
    """Context manager for adding logging context."""
    
    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger,
        **context: Any
    ):
        self.logger = logger
        self.context = context
        self.bound_logger = None
    
    def __enter__(self) -> structlog.stdlib.BoundLogger:
        """Enter the context and return bound logger."""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        if exc_type is not None:
            self.bound_logger.error(
                "Exception in logging context",
                exc_type=exc_type.__name__,
                exc_message=str(exc_val)
            )


def with_logging(
    logger: Optional[structlog.stdlib.BoundLogger] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_execution_time: bool = True,
    level: str = "debug"
):
    """
    Decorator to add automatic logging to functions.
    
    Args:
        logger: Optional logger instance
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_execution_time: Whether to log execution time
        level: Log level for function calls
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            
            # Log function call
            if log_args:
                log_function_call(
                    logger,
                    func.__name__,
                    args=args if args else None,
                    kwargs=kwargs if kwargs else None,
                    level=level
                )
            else:
                log_function_call(logger, func.__name__, level=level)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                execution_time = time.time() - start_time if log_execution_time else None
                if log_result:
                    log_function_result(
                        logger,
                        func.__name__,
                        result=result,
                        execution_time=execution_time,
                        level=level
                    )
                else:
                    log_function_result(
                        logger,
                        func.__name__,
                        execution_time=execution_time,
                        level=level
                    )
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time if log_execution_time else None
                log_function_error(
                    logger,
                    func.__name__,
                    error=e,
                    execution_time=execution_time
                )
                raise
        
        return wrapper
    return decorator