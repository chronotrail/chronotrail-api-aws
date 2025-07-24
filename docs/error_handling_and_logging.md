# Comprehensive Error Handling and Logging System

## Overview

The ChronoTrail API implements a comprehensive error handling and logging system that provides:

- **Structured error responses** for all endpoints
- **Comprehensive logging** for debugging and monitoring
- **Proper exception handling** for AWS service failures
- **Security features** to prevent sensitive data leakage
- **Performance monitoring** and request tracking
- **Centralized error management** with consistent formatting

## Architecture

### Core Components

1. **Custom Exception Classes** (`app/core/exceptions.py`)
2. **Structured Logging** (`app/core/logging.py`)
3. **Error Handling Middleware** (`app/middleware/error_handling.py`)
4. **Standardized Error Responses** (`app/models/schemas.py`)

## Custom Exception Classes

### Base Exception
```python
class ChronoTrailError(Exception):
    """Base exception for ChronoTrail API errors."""
    
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict[str, Any] = None, status_code: int = 500):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.status_code = status_code
        self.error_id = str(uuid4())
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
```

### Specialized Exceptions

- **AuthenticationError** (401) - Invalid or missing authentication
- **AuthorizationError** (403) - Access denied to resource
- **ValidationError** (422) - Data validation failures
- **ResourceNotFoundError** (404) - Resource not found
- **UsageLimitError** (429) - Usage limits exceeded
- **SubscriptionError** (402) - Subscription-related errors
- **FileUploadError** (400) - File upload failures
- **ProcessingError** (422) - Content processing failures
- **DatabaseError** (500) - Database operation failures
- **ExternalServiceError** (503) - External service failures

## Structured Logging

### Configuration
```python
# Configure comprehensive logging
configure_logging(
    log_level="INFO",
    enable_json_logs=True,
    enable_request_logging=True
)
```

### Features

- **JSON-formatted logs** for production environments
- **Console-formatted logs** for development
- **Request context tracking** with unique request IDs
- **User context binding** for user-specific operations
- **Security filtering** to remove sensitive data
- **Performance monitoring** with execution time tracking

### Usage Examples

```python
from app.core.logging import get_logger, LoggingContext, with_logging

logger = get_logger(__name__)

# Basic logging
logger.info("Operation completed", user_id="123", operation="upload")

# Context manager for related operations
with LoggingContext(logger, user_id="123", operation="file_upload") as ctx_logger:
    ctx_logger.info("Starting file validation")
    # ... operation code ...
    ctx_logger.info("File validation completed")

# Decorator for automatic function logging
@with_logging(log_execution_time=True, level="info")
async def process_file(file_data):
    # Function implementation
    return result
```

## Error Response Format

All API errors return a standardized JSON response:

```json
{
    "error": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
        "field": "email",
        "issue": "Invalid email format"
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Response Model
```python
class ErrorResponse(BaseModel):
    error: str = Field(..., max_length=100)
    message: str = Field(..., max_length=500)
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
```

## Middleware Components

### 1. Request Logging Middleware

Logs all incoming requests and responses with:
- Request ID generation
- User context extraction
- Performance timing
- Request/response data (configurable)
- Client IP and user agent tracking

```python
app.add_middleware(
    RequestLoggingMiddleware,
    log_request_body=False,
    log_response_body=False,
    log_headers=False,
    exclude_paths=['/health', '/docs', '/openapi.json']
)
```

### 2. Error Handling Middleware

Catches unhandled exceptions and converts them to standardized error responses:

```python
app.add_middleware(
    ErrorHandlingMiddleware,
    include_error_details=False,  # Set to True for development
    log_stack_traces=True
)
```

### 3. Performance Monitoring Middleware

Tracks request processing times and identifies slow requests:

```python
app.add_middleware(
    PerformanceMonitoringMiddleware,
    slow_request_threshold=5.0,  # seconds
    log_all_requests=False
)
```

### 4. Security Headers Middleware

Adds security headers to all responses:

```python
app.add_middleware(SecurityHeadersMiddleware)
```

## Exception Handlers

### Custom Exception Handlers

The system includes specialized handlers for different exception types:

```python
# ChronoTrail custom exceptions
@app.exception_handler(ChronoTrailError)
async def chronotrail_exception_handler(request: Request, exc: ChronoTrailError):
    # Returns standardized error response with appropriate status code

# AWS service exceptions
@app.exception_handler(AWSServiceError)
async def aws_service_exception_handler(request: Request, exc: AWSServiceError):
    # Maps AWS errors to appropriate HTTP status codes

# Database exceptions
@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    # Handles database-specific errors with proper status codes

# Validation exceptions
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Provides detailed validation error information

# Generic exception handler (fallback)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Handles any unhandled exceptions with generic error response
```

## AWS Service Error Handling

### AWS Exception Classes

Specialized exceptions for different AWS services:

```python
# S3 operations
raise S3Error("Access denied", operation="upload_file", error_code="AccessDenied")

# Textract operations
raise TextractError("Document processing failed", operation="analyze_document")

# Transcribe operations
raise TranscribeError("Audio format not supported", operation="start_transcription_job")

# Rekognition operations
raise RekognitionError("Image analysis failed", operation="detect_labels")

# Bedrock operations
raise BedrockError("Model not available", operation="invoke_model")
```

### Error Mapping

AWS errors are automatically mapped to appropriate HTTP status codes:

- **Access Denied** → 403 Forbidden
- **Not Found** → 404 Not Found
- **Invalid Request** → 400 Bad Request
- **Service Unavailable** → 503 Service Unavailable
- **Rate Limiting** → 429 Too Many Requests

## Security Features

### Sensitive Data Filtering

The logging system automatically filters sensitive information:

```python
class SecurityFilter(logging.Filter):
    SENSITIVE_KEYS = {
        'password', 'token', 'secret', 'key', 'authorization',
        'access_token', 'refresh_token', 'api_key', 'oauth_token',
        'identity_token', 'jwt', 'bearer'
    }
```

### Error Detail Control

Error details can be controlled based on environment:

```python
# Production: Hide detailed error information
error_response = create_error_response(error, include_details=False)

# Development: Include detailed error information
error_response = create_error_response(error, include_details=True)
```

## Usage Examples

### Service Implementation with Error Handling

```python
from app.core.logging import get_logger, LoggingContext, with_logging
from app.core.exceptions import ExternalServiceError, FileUploadError

logger = get_logger(__name__)

class FileService:
    @with_logging(log_execution_time=True, level="info")
    async def upload_file(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
        with LoggingContext(logger, user_id=user_id, operation="file_upload") as ctx_logger:
            try:
                # Validate file
                if not self._validate_file(file):
                    raise FileUploadError(
                        message="Invalid file format",
                        details={"filename": file.filename, "content_type": file.content_type}
                    )
                
                ctx_logger.info("File validation passed", filename=file.filename)
                
                # Upload to S3
                result = await self._upload_to_s3(file)
                
                ctx_logger.info("File upload completed", s3_key=result['s3_key'])
                return result
                
            except S3Error as e:
                ctx_logger.error("S3 upload failed", error=str(e))
                raise ExternalServiceError(
                    message="Failed to upload file to storage",
                    details={"s3_error": str(e)}
                )
            except Exception as e:
                ctx_logger.error("Unexpected error during file upload", error=str(e))
                raise
```

### Endpoint Implementation with Error Handling

```python
from fastapi import APIRouter, HTTPException, Depends
from app.core.exceptions import ValidationError, ResourceNotFoundError

router = APIRouter()

@router.post("/files", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file with comprehensive error handling."""
    
    # Validation
    if not file.filename:
        raise ValidationError(
            message="Filename is required",
            details={"field": "filename"}
        )
    
    try:
        # Process file
        result = await file_service.upload_file(file, current_user.id)
        return FileResponse(**result)
        
    except FileUploadError as e:
        # Re-raise custom exceptions (handled by exception handlers)
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error("Unexpected error in file upload endpoint", error=str(e))
        raise
```

## Configuration

### Environment Variables

```bash
# Logging configuration
LOG_LEVEL=INFO
ENABLE_JSON_LOGS=true
ENABLE_REQUEST_LOGGING=true
LOG_REQUEST_BODY=false
LOG_RESPONSE_BODY=false
LOG_HEADERS=false
LOG_STACK_TRACES=true

# Error handling configuration
INCLUDE_ERROR_DETAILS=false
SLOW_REQUEST_THRESHOLD=5.0
LOG_ALL_REQUESTS=false
```

### Application Configuration

```python
# main.py
from app.core.logging import configure_logging
from app.core.exceptions import get_error_handlers
from app.middleware.error_handling import (
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    PerformanceMonitoringMiddleware,
    SecurityHeadersMiddleware
)

# Configure logging
configure_logging()

# Add error handlers
error_handlers = get_error_handlers()
for exception_type, handler in error_handlers.items():
    app.add_exception_handler(exception_type, handler)

# Add middleware (in reverse order)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(PerformanceMonitoringMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
```

## Monitoring and Debugging

### Log Analysis

Logs are structured in JSON format for easy parsing and analysis:

```json
{
    "request_id": "f2238223-3b70-4d16-b5c8-bb5ad16ee948",
    "user_id": "user_123",
    "endpoint": "/api/v1/files",
    "method": "POST",
    "status_code": 200,
    "processing_time_seconds": 1.234,
    "event": "Request completed successfully",
    "logger": "app.middleware.error_handling",
    "level": "info",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Tracking

Errors include unique error IDs for tracking:

```json
{
    "error_id": "c3e6ed99-440c-43b7-9617-eabe2bc68f5f",
    "error_type": "ValidationError",
    "message": "Invalid email format",
    "request_path": "/api/v1/users",
    "request_method": "POST",
    "event": "Validation error occurred",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Performance Monitoring

Slow requests are automatically logged:

```json
{
    "processing_time_seconds": 6.789,
    "status_code": 200,
    "method": "POST",
    "path": "/api/v1/query",
    "event": "Slow request detected",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## Testing

The error handling system includes comprehensive tests:

```bash
# Run error handling tests
uv run python -m pytest tests/test_error_handling.py -v

# Test specific components
uv run python -m pytest tests/test_error_handling.py::TestChronoTrailExceptions -v
uv run python -m pytest tests/test_error_handling.py::TestExceptionHandlers -v
uv run python -m pytest tests/test_error_handling.py::TestLoggingIntegration -v
```

## Best Practices

### 1. Use Appropriate Exception Types

```python
# Good: Use specific exception types
raise ValidationError("Invalid email format", details={"field": "email"})

# Avoid: Generic exceptions
raise Exception("Something went wrong")
```

### 2. Include Contextual Information

```python
# Good: Include relevant context
raise FileUploadError(
    message="File size exceeds limit",
    details={
        "filename": file.filename,
        "size_mb": file_size_mb,
        "limit_mb": max_size_mb
    }
)
```

### 3. Use Logging Context

```python
# Good: Use context for related operations
with LoggingContext(logger, user_id=user_id, operation="file_processing") as ctx_logger:
    ctx_logger.info("Starting file validation")
    # ... operations ...
    ctx_logger.info("File processing completed")
```

### 4. Handle AWS Errors Appropriately

```python
try:
    result = await s3_client.upload_file(...)
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == 'AccessDenied':
        raise S3Error("Access denied to S3 bucket", error_code=error_code)
    else:
        raise S3Error(f"S3 operation failed: {str(e)}", error_code=error_code)
```

### 5. Log Performance-Critical Operations

```python
@with_logging(log_execution_time=True, level="info")
async def expensive_operation():
    # Implementation
    pass
```

## Conclusion

The comprehensive error handling and logging system provides:

- **Consistent error responses** across all endpoints
- **Detailed logging** for debugging and monitoring
- **Security features** to protect sensitive data
- **Performance monitoring** to identify bottlenecks
- **Proper AWS service error handling** with appropriate status codes
- **Comprehensive testing** to ensure reliability

This system ensures that the ChronoTrail API provides a robust, maintainable, and monitorable service that can handle errors gracefully and provide meaningful feedback to both developers and users.