"""
ChronoTrail API - Main FastAPI application with comprehensive error handling and logging.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1.api import api_router
from app.aws.exceptions import AWSServiceError, FileProcessingError
from app.core.config import settings
from app.core.exceptions import ChronoTrailError, get_error_handlers
from app.core.logging import configure_logging, get_logger
from app.middleware.error_handling import (
    ErrorHandlingMiddleware,
    PerformanceMonitoringMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger = get_logger(__name__)
    logger.info("ChronoTrail API starting up", version="1.0.0")

    try:
        # Initialize any required services here
        logger.info("Application startup completed successfully")
        yield
    except Exception as e:
        logger.error("Application startup failed", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("ChronoTrail API shutting down")


# Configure logging before creating the app
configure_logging()

# Create FastAPI application
app = FastAPI(
    title="ChronoTrail API",
    description="Personal timeline data storage and natural language querying service",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Get logger for main module
logger = get_logger(__name__)

# Add error handlers
error_handlers = get_error_handlers()
for exception_type, handler in error_handlers.items():
    app.add_exception_handler(exception_type, handler)

# Add middleware in reverse order (last added = first executed)
# Security headers (outermost)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring
app.add_middleware(
    PerformanceMonitoringMiddleware,
    slow_request_threshold=getattr(settings, "SLOW_REQUEST_THRESHOLD", 5.0),
    log_all_requests=getattr(settings, "LOG_ALL_REQUESTS", False),
)

# Error handling middleware
app.add_middleware(
    ErrorHandlingMiddleware,
    include_error_details=getattr(settings, "INCLUDE_ERROR_DETAILS", False),
    log_stack_traces=getattr(settings, "LOG_STACK_TRACES", True),
)

# Request logging (innermost - closest to the application)
app.add_middleware(
    RequestLoggingMiddleware,
    log_request_body=getattr(settings, "LOG_REQUEST_BODY", False),
    log_response_body=getattr(settings, "LOG_RESPONSE_BODY", False),
    log_headers=getattr(settings, "LOG_HEADERS", False),
    exclude_paths=["/health", "/docs", "/openapi.json", "/favicon.ico", "/metrics"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint returning basic API information."""
    logger.info("Root endpoint accessed")
    return {
        "message": "ChronoTrail API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": f"{settings.API_V1_STR}/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint for monitoring."""
    # This could be expanded to include actual metrics
    return {
        "status": "operational",
        "uptime": "unknown",
        "requests_total": "unknown",
        "errors_total": "unknown",
    }


# Log application startup
logger.info(
    "ChronoTrail API initialized",
    version="1.0.0",
    environment=getattr(settings, "ENVIRONMENT", "development"),
    debug=getattr(settings, "DEBUG", False),
)
