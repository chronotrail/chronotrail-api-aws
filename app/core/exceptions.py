"""
Comprehensive error handling and exception management for ChronoTrail API.

This module provides centralized error handling, custom exceptions, and structured
error responses for all API endpoints and services.
"""
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.models.schemas import ErrorResponse
from app.aws.exceptions import (
    AWSServiceError,
    S3Error,
    TextractError,
    TranscribeError,
    RekognitionError,
    BedrockError,
    OpenSearchError,
    FileProcessingError
)

# Configure logger
logger = get_logger(__name__)


class ChronoTrailError(Exception):
    """Base exception for ChronoTrail API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        self.error_id = str(uuid4())
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        
        super().__init__(message)


class AuthenticationError(ChronoTrailError):
    """Exception for authentication-related errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "AUTHENTICATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(ChronoTrailError):
    """Exception for authorization-related errors."""
    
    def __init__(
        self,
        message: str = "Access denied",
        error_code: str = "AUTHORIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_403_FORBIDDEN
        )


class ValidationError(ChronoTrailError):
    """Exception for data validation errors."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class ResourceNotFoundError(ChronoTrailError):
    """Exception for resource not found errors."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        error_code: str = "RESOURCE_NOT_FOUND",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_404_NOT_FOUND
        )


class UsageLimitError(ChronoTrailError):
    """Exception for usage limit exceeded errors."""
    
    def __init__(
        self,
        message: str = "Usage limit exceeded",
        error_code: str = "USAGE_LIMIT_EXCEEDED",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS
        )


class SubscriptionError(ChronoTrailError):
    """Exception for subscription-related errors."""
    
    def __init__(
        self,
        message: str = "Subscription error",
        error_code: str = "SUBSCRIPTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_402_PAYMENT_REQUIRED
        )


class FileUploadError(ChronoTrailError):
    """Exception for file upload errors."""
    
    def __init__(
        self,
        message: str = "File upload failed",
        error_code: str = "FILE_UPLOAD_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_400_BAD_REQUEST
        )


class ProcessingError(ChronoTrailError):
    """Exception for content processing errors."""
    
    def __init__(
        self,
        message: str = "Content processing failed",
        error_code: str = "PROCESSING_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )


class DatabaseError(ChronoTrailError):
    """Exception for database-related errors."""
    
    def __init__(
        self,
        message: str = "Database operation failed",
        error_code: str = "DATABASE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class ExternalServiceError(ChronoTrailError):
    """Exception for external service errors."""
    
    def __init__(
        self,
        message: str = "External service error",
        error_code: str = "EXTERNAL_SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = status.HTTP_503_SERVICE_UNAVAILABLE
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code
        )


def create_error_response(
    error: Union[ChronoTrailError, Exception],
    request: Optional[Request] = None,
    include_details: bool = False
) -> ErrorResponse:
    """
    Create a standardized error response from an exception.
    
    Args:
        error: The exception to convert
        request: Optional request object for context
        include_details: Whether to include detailed error information
        
    Returns:
        ErrorResponse: Standardized error response
    """
    # Handle ChronoTrail custom errors
    if isinstance(error, ChronoTrailError):
        error_response = ErrorResponse(
            error=error.error_code,
            message=error.message,
            details=error.details if include_details else None,
            timestamp=error.timestamp
        )
        
        # Log the error with context
        logger.error(
            "ChronoTrail error occurred",
            error_id=error.error_id,
            error_code=error.error_code,
            message=error.message,
            status_code=error.status_code,
            details=error.details,
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle AWS service errors
    elif isinstance(error, AWSServiceError):
        error_response = ErrorResponse(
            error="AWS_SERVICE_ERROR",
            message=f"AWS service error: {error.message}",
            details={
                "service": error.service,
                "operation": error.operation,
                "error_code": error.error_code
            } if include_details else None
        )
        
        # Log AWS service error
        logger.error(
            "AWS service error occurred",
            service=error.service,
            operation=error.operation,
            error_code=error.error_code,
            message=error.message,
            details=error.details,
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle file processing errors
    elif isinstance(error, FileProcessingError):
        error_response = ErrorResponse(
            error="FILE_PROCESSING_ERROR",
            message=f"File processing error: {error.message}",
            details={
                "file_type": error.file_type,
                "processing_type": error.processing_type
            } if include_details else None
        )
        
        # Log file processing error
        logger.error(
            "File processing error occurred",
            file_type=error.file_type,
            processing_type=error.processing_type,
            message=error.message,
            details=error.details,
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle database errors
    elif isinstance(error, SQLAlchemyError):
        if isinstance(error, IntegrityError):
            error_response = ErrorResponse(
                error="DATA_INTEGRITY_ERROR",
                message="Data integrity constraint violation",
                details={"constraint": str(error.orig)} if include_details else None
            )
        elif isinstance(error, OperationalError):
            error_response = ErrorResponse(
                error="DATABASE_OPERATIONAL_ERROR",
                message="Database operational error",
                details={"error": str(error.orig)} if include_details else None
            )
        else:
            error_response = ErrorResponse(
                error="DATABASE_ERROR",
                message="Database operation failed",
                details={"error": str(error)} if include_details else None
            )
        
        # Log database error
        logger.error(
            "Database error occurred",
            error_type=type(error).__name__,
            message=str(error),
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle validation errors
    elif isinstance(error, (ValidationError, RequestValidationError)):
        error_response = ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": error.errors()} if include_details else None
        )
        
        # Log validation error
        logger.warning(
            "Validation error occurred",
            errors=error.errors() if hasattr(error, 'errors') else str(error),
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle HTTP exceptions
    elif isinstance(error, (HTTPException, StarletteHTTPException)):
        error_response = ErrorResponse(
            error="HTTP_ERROR",
            message=error.detail if hasattr(error, 'detail') else str(error),
            details={"status_code": error.status_code} if include_details else None
        )
        
        # Log HTTP error
        logger.warning(
            "HTTP error occurred",
            status_code=error.status_code,
            detail=error.detail if hasattr(error, 'detail') else str(error),
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response
    
    # Handle generic exceptions
    else:
        error_id = str(uuid4())
        error_response = ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            details={"error_id": error_id} if include_details else None
        )
        
        # Log generic error with full traceback
        logger.error(
            "Unexpected error occurred",
            error_id=error_id,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            request_path=request.url.path if request else None,
            request_method=request.method if request else None
        )
        
        return error_response


async def chronotrail_exception_handler(request: Request, exc: ChronoTrailError) -> JSONResponse:
    """
    Exception handler for ChronoTrail custom exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The ChronoTrail exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def aws_service_exception_handler(request: Request, exc: AWSServiceError) -> JSONResponse:
    """
    Exception handler for AWS service exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The AWS service exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    # Map AWS errors to appropriate HTTP status codes
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    if isinstance(exc, (S3Error, TextractError, TranscribeError, RekognitionError, BedrockError)):
        if "access denied" in exc.message.lower() or "unauthorized" in exc.message.lower():
            status_code = status.HTTP_403_FORBIDDEN
        elif "not found" in exc.message.lower():
            status_code = status.HTTP_404_NOT_FOUND
        elif "invalid" in exc.message.lower() or "bad request" in exc.message.lower():
            status_code = status.HTTP_400_BAD_REQUEST
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


async def file_processing_exception_handler(request: Request, exc: FileProcessingError) -> JSONResponse:
    """
    Exception handler for file processing exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The file processing exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """
    Exception handler for database exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The database exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    # Map database errors to appropriate HTTP status codes
    if isinstance(exc, IntegrityError):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, OperationalError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Exception handler for request validation exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The validation exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=True)
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Exception handler for HTTP exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The HTTP exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Exception handler for generic exceptions.
    
    Args:
        request: The request that caused the exception
        exc: The generic exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    error_response = create_error_response(exc, request, include_details=False)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )


def get_error_handlers() -> Dict[Union[int, type], callable]:
    """
    Get all error handlers for the FastAPI application.
    
    Returns:
        Dict: Mapping of exception types/status codes to handlers
    """
    return {
        ChronoTrailError: chronotrail_exception_handler,
        AWSServiceError: aws_service_exception_handler,
        FileProcessingError: file_processing_exception_handler,
        SQLAlchemyError: database_exception_handler,
        RequestValidationError: validation_exception_handler,
        HTTPException: http_exception_handler,
        Exception: generic_exception_handler,
    }