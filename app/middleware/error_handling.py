"""
Error handling and request logging middleware for ChronoTrail API.

This module provides middleware for comprehensive request/response logging,
error handling, and performance monitoring.
"""

import time
import traceback
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.exceptions import ChronoTrailError, create_error_response
from app.core.logging import LoggingContext, add_request_context, get_logger
from app.models.schemas import ErrorResponse


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive request and response logging.

    This middleware logs all incoming requests, their processing time,
    response status, and any errors that occur during processing.
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        log_headers: bool = False,
        exclude_paths: Optional[list] = None,
    ):
        """
        Initialize the request logging middleware.

        Args:
            app: ASGI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            log_headers: Whether to log request/response headers
            exclude_paths: List of paths to exclude from logging
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.log_headers = log_headers
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/favicon.ico",
        ]
        self.logger = get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and log relevant information.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint to call

        Returns:
            Response: The response from the application
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid4())

        # Extract user information if available
        user_id = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_id = str(request.state.user.id)

        # Create logger with request context
        request_logger = add_request_context(
            self.logger,
            request_id=request_id,
            user_id=user_id,
            endpoint=request.url.path,
        )

        # Start timing
        start_time = time.time()

        # Log request
        request_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "unknown"),
        }

        if self.log_headers:
            request_data["headers"] = dict(request.headers)

        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Only log first 1000 characters to avoid huge logs
                    request_data["body_preview"] = body.decode("utf-8")[:1000]
                    request_data["body_size"] = len(body)
            except Exception as e:
                request_data["body_error"] = str(e)

        request_logger.info("Request started", **request_data)

        # Store request context in request state
        request.state.request_id = request_id
        request.state.request_logger = request_logger
        request.state.start_time = start_time

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Log response
            response_data = {
                "status_code": response.status_code,
                "processing_time_seconds": processing_time,
                "response_size": (
                    len(response.body) if hasattr(response, "body") else None
                ),
            }

            if self.log_headers and hasattr(response, "headers"):
                response_data["headers"] = dict(response.headers)

            if self.log_response_body and hasattr(response, "body"):
                try:
                    # Only log first 1000 characters to avoid huge logs
                    body_preview = (
                        response.body.decode("utf-8")[:1000] if response.body else None
                    )
                    if body_preview:
                        response_data["body_preview"] = body_preview
                except Exception as e:
                    response_data["body_error"] = str(e)

            # Log based on status code
            if response.status_code >= 500:
                request_logger.error(
                    "Request completed with server error", **response_data
                )
            elif response.status_code >= 400:
                request_logger.warning(
                    "Request completed with client error", **response_data
                )
            else:
                request_logger.info("Request completed successfully", **response_data)

            return response

        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time

            # Log error
            error_data = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time_seconds": processing_time,
                "traceback": traceback.format_exc(),
            }

            request_logger.error("Request failed with exception", **error_data)

            # Re-raise the exception to be handled by error handlers
            raise

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request.

        Args:
            request: The request object

        Returns:
            str: Client IP address
        """
        # Check for forwarded headers (common in load balancers)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling and response formatting.

    This middleware catches unhandled exceptions and converts them to
    properly formatted error responses.
    """

    def __init__(
        self,
        app: ASGIApp,
        include_error_details: bool = False,
        log_stack_traces: bool = True,
    ):
        """
        Initialize the error handling middleware.

        Args:
            app: ASGI application
            include_error_details: Whether to include detailed error info in responses
            log_stack_traces: Whether to log full stack traces
        """
        super().__init__(app)
        self.include_error_details = include_error_details
        self.log_stack_traces = log_stack_traces
        self.logger = get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and handle any unhandled exceptions.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint to call

        Returns:
            Response: The response from the application or error response
        """
        try:
            return await call_next(request)

        except Exception as e:
            # Get request logger if available
            request_logger = getattr(request.state, "request_logger", self.logger)

            # Create error response
            error_response = create_error_response(
                error=e, request=request, include_details=self.include_error_details
            )

            # Determine status code
            if isinstance(e, ChronoTrailError):
                status_code = e.status_code
            else:
                status_code = 500

            # Log the error
            error_context = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "status_code": status_code,
                "request_path": request.url.path,
                "request_method": request.method,
            }

            if self.log_stack_traces:
                error_context["traceback"] = traceback.format_exc()

            if status_code >= 500:
                request_logger.error("Unhandled server error", **error_context)
            else:
                request_logger.warning("Unhandled client error", **error_context)

            # Return formatted error response
            return JSONResponse(
                status_code=status_code, content=error_response.model_dump()
            )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and metrics collection.

    This middleware tracks request processing times, response sizes,
    and other performance metrics.
    """

    def __init__(
        self,
        app: ASGIApp,
        slow_request_threshold: float = 5.0,
        log_all_requests: bool = False,
    ):
        """
        Initialize the performance monitoring middleware.

        Args:
            app: ASGI application
            slow_request_threshold: Threshold in seconds for slow request logging
            log_all_requests: Whether to log performance data for all requests
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.log_all_requests = log_all_requests
        self.logger = get_logger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and monitor performance.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint to call

        Returns:
            Response: The response from the application
        """
        # Start timing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate metrics
        processing_time = time.time() - start_time
        response_size = len(response.body) if hasattr(response, "body") else 0

        # Get request logger if available
        request_logger = getattr(request.state, "request_logger", self.logger)

        # Performance data
        perf_data = {
            "processing_time_seconds": processing_time,
            "response_size_bytes": response_size,
            "status_code": response.status_code,
            "method": request.method,
            "path": request.url.path,
        }

        # Log slow requests or all requests if configured
        if processing_time > self.slow_request_threshold:
            request_logger.warning("Slow request detected", **perf_data)
        elif self.log_all_requests:
            request_logger.info("Request performance", **perf_data)

        # Add performance headers to response
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        response.headers["X-Response-Size"] = str(response_size)

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to responses.

    This middleware adds common security headers to all responses
    to improve the security posture of the API.
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize the security headers middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and add security headers to the response.

        Args:
            request: The incoming request
            call_next: The next middleware/endpoint to call

        Returns:
            Response: The response with added security headers
        """
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        # Remove server header for security
        if "server" in response.headers:
            del response.headers["server"]

        return response
