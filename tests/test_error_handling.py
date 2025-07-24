"""
Tests for comprehensive error handling and logging system.
"""
import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, OperationalError
from pydantic import ValidationError

from app.main import app
from app.core.exceptions import (
    ChronoTrailError,
    AuthenticationError,
    AuthorizationError,
    ValidationError as ChronoValidationError,
    ResourceNotFoundError,
    UsageLimitError,
    SubscriptionError,
    FileUploadError,
    ProcessingError,
    DatabaseError,
    ExternalServiceError,
    create_error_response,
    chronotrail_exception_handler,
    aws_service_exception_handler,
    database_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    generic_exception_handler
)
from app.aws.exceptions import S3Error, TextractError, TranscribeError
from app.models.schemas import ErrorResponse


class TestChronoTrailExceptions:
    """Test ChronoTrail custom exceptions."""
    
    def test_chronotrail_error_base(self):
        """Test base ChronoTrail error."""
        error = ChronoTrailError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"},
            status_code=400
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.details == {"key": "value"}
        assert error.status_code == 400
        assert error.error_id is not None
        assert error.timestamp is not None
    
    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid token")
        
        assert error.message == "Invalid token"
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.status_code == 401
    
    def test_authorization_error(self):
        """Test authorization error."""
        error = AuthorizationError("Access denied")
        
        assert error.message == "Access denied"
        assert error.error_code == "AUTHORIZATION_ERROR"
        assert error.status_code == 403
    
    def test_validation_error(self):
        """Test validation error."""
        error = ChronoValidationError(
            message="Invalid data",
            details={"field": "required"}
        )
        
        assert error.message == "Invalid data"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.status_code == 422
        assert error.details == {"field": "required"}
    
    def test_resource_not_found_error(self):
        """Test resource not found error."""
        error = ResourceNotFoundError("User not found")
        
        assert error.message == "User not found"
        assert error.error_code == "RESOURCE_NOT_FOUND"
        assert error.status_code == 404
    
    def test_usage_limit_error(self):
        """Test usage limit error."""
        error = UsageLimitError("Daily limit exceeded")
        
        assert error.message == "Daily limit exceeded"
        assert error.error_code == "USAGE_LIMIT_EXCEEDED"
        assert error.status_code == 429
    
    def test_subscription_error(self):
        """Test subscription error."""
        error = SubscriptionError("Subscription expired")
        
        assert error.message == "Subscription expired"
        assert error.error_code == "SUBSCRIPTION_ERROR"
        assert error.status_code == 402
    
    def test_file_upload_error(self):
        """Test file upload error."""
        error = FileUploadError("File too large")
        
        assert error.message == "File too large"
        assert error.error_code == "FILE_UPLOAD_ERROR"
        assert error.status_code == 400
    
    def test_processing_error(self):
        """Test processing error."""
        error = ProcessingError("OCR failed")
        
        assert error.message == "OCR failed"
        assert error.error_code == "PROCESSING_ERROR"
        assert error.status_code == 422
    
    def test_database_error(self):
        """Test database error."""
        error = DatabaseError("Connection failed")
        
        assert error.message == "Connection failed"
        assert error.error_code == "DATABASE_ERROR"
        assert error.status_code == 500
    
    def test_external_service_error(self):
        """Test external service error."""
        error = ExternalServiceError("AWS service unavailable")
        
        assert error.message == "AWS service unavailable"
        assert error.error_code == "EXTERNAL_SERVICE_ERROR"
        assert error.status_code == 503


class TestErrorResponseCreation:
    """Test error response creation."""
    
    def test_create_error_response_chronotrail_error(self):
        """Test creating error response from ChronoTrail error."""
        error = AuthenticationError("Invalid token", details={"token": "expired"})
        
        response = create_error_response(error, include_details=True)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "AUTHENTICATION_ERROR"
        assert response.message == "Invalid token"
        assert response.details == {"token": "expired"}
        assert response.timestamp == error.timestamp
    
    def test_create_error_response_aws_error(self):
        """Test creating error response from AWS error."""
        error = S3Error("Access denied", operation="upload_file", error_code="AccessDenied")
        
        response = create_error_response(error, include_details=True)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "AWS_SERVICE_ERROR"
        assert "AWS service error" in response.message
        assert response.details["service"] == "S3"
        assert response.details["operation"] == "upload_file"
        assert response.details["error_code"] == "AccessDenied"
    
    def test_create_error_response_database_error(self):
        """Test creating error response from database error."""
        orig_error = Mock()
        orig_error.__str__ = Mock(return_value="duplicate key value")
        
        error = IntegrityError("statement", "params", orig_error)
        
        response = create_error_response(error, include_details=True)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "DATA_INTEGRITY_ERROR"
        assert response.message == "Data integrity constraint violation"
        assert "constraint" in response.details
    
    def test_create_error_response_http_exception(self):
        """Test creating error response from HTTP exception."""
        error = HTTPException(status_code=400, detail="Bad request")
        
        response = create_error_response(error, include_details=True)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "HTTP_ERROR"
        assert response.message == "Bad request"
        assert response.details["status_code"] == 400
    
    def test_create_error_response_generic_exception(self):
        """Test creating error response from generic exception."""
        error = ValueError("Invalid value")
        
        response = create_error_response(error, include_details=True)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "INTERNAL_SERVER_ERROR"
        assert response.message == "An internal server error occurred"
        assert "error_id" in response.details
    
    def test_create_error_response_without_details(self):
        """Test creating error response without details."""
        error = AuthenticationError("Invalid token", details={"token": "expired"})
        
        response = create_error_response(error, include_details=False)
        
        assert isinstance(response, ErrorResponse)
        assert response.error == "AUTHENTICATION_ERROR"
        assert response.message == "Invalid token"
        assert response.details is None


class TestExceptionHandlers:
    """Test exception handlers."""
    
    @pytest.mark.asyncio
    async def test_chronotrail_exception_handler(self):
        """Test ChronoTrail exception handler."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        error = AuthenticationError("Invalid token")
        
        response = await chronotrail_exception_handler(request, error)
        
        assert response.status_code == 401
        response_data = json.loads(response.body)
        assert response_data["error"] == "AUTHENTICATION_ERROR"
        assert response_data["message"] == "Invalid token"
    
    @pytest.mark.asyncio
    async def test_aws_service_exception_handler(self):
        """Test AWS service exception handler."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "POST"
        
        error = S3Error("Access denied", operation="upload_file")
        
        response = await aws_service_exception_handler(request, error)
        
        assert response.status_code == 403  # Access denied maps to 403
        response_data = json.loads(response.body)
        assert response_data["error"] == "AWS_SERVICE_ERROR"
    
    @pytest.mark.asyncio
    async def test_database_exception_handler(self):
        """Test database exception handler."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "POST"
        
        orig_error = Mock()
        orig_error.__str__ = Mock(return_value="duplicate key")
        error = IntegrityError("statement", "params", orig_error)
        
        response = await database_exception_handler(request, error)
        
        assert response.status_code == 409
        response_data = json.loads(response.body)
        assert response_data["error"] == "DATA_INTEGRITY_ERROR"
    
    @pytest.mark.asyncio
    async def test_http_exception_handler(self):
        """Test HTTP exception handler."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        error = HTTPException(status_code=404, detail="Not found")
        
        response = await http_exception_handler(request, error)
        
        assert response.status_code == 404
        response_data = json.loads(response.body)
        assert response_data["error"] == "HTTP_ERROR"
        assert response_data["message"] == "Not found"
    
    @pytest.mark.asyncio
    async def test_generic_exception_handler(self):
        """Test generic exception handler."""
        request = Mock(spec=Request)
        request.url.path = "/test"
        request.method = "GET"
        
        error = ValueError("Invalid value")
        
        response = await generic_exception_handler(request, error)
        
        assert response.status_code == 500
        response_data = json.loads(response.body)
        assert response_data["error"] == "INTERNAL_SERVER_ERROR"
        assert response_data["message"] == "An internal server error occurred"


class TestErrorHandlingIntegration:
    """Test error handling integration with FastAPI."""
    
    def test_error_handling_middleware_integration(self):
        """Test that error handling middleware is properly integrated."""
        client = TestClient(app)
        
        # Test health endpoint works
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test non-existent endpoint returns 404
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Verify error response format (may be FastAPI default or our custom format)
        error_data = response.json()
        # FastAPI default format has 'detail', our custom format has 'error'
        assert "detail" in error_data or "error" in error_data
    
    def test_validation_error_handling(self):
        """Test validation error handling."""
        client = TestClient(app)
        
        # Test invalid JSON in request body
        response = client.post(
            "/api/v1/locations",
            json={"invalid": "data"},
            headers={"Authorization": "Bearer invalid_token"}
        )
        
        # Should return validation error or authentication error
        assert response.status_code in [401, 422]
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
        assert "timestamp" in error_data


class TestLoggingIntegration:
    """Test logging integration."""
    
    @patch('app.core.logging.get_logger')
    def test_logging_context_manager(self, mock_get_logger):
        """Test logging context manager."""
        from app.core.logging import LoggingContext
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_logger.bind.return_value = mock_logger
        
        with LoggingContext(mock_logger, user_id="test_user", operation="test_op") as ctx_logger:
            assert ctx_logger == mock_logger
            mock_logger.bind.assert_called_once_with(user_id="test_user", operation="test_op")
    
    @patch('app.core.logging.get_logger')
    def test_with_logging_decorator(self, mock_get_logger):
        """Test with_logging decorator."""
        from app.core.logging import with_logging
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @with_logging(logger=mock_logger, log_args=True, log_result=True)
        def test_function(arg1, arg2=None):
            return "test_result"
        
        result = test_function("value1", arg2="value2")
        
        assert result == "test_result"
        # Verify logging calls were made
        assert mock_logger.debug.call_count >= 2  # At least function call and result logs
    
    @patch('app.core.logging.get_logger')
    def test_function_error_logging(self, mock_get_logger):
        """Test function error logging."""
        from app.core.logging import with_logging
        
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        @with_logging(logger=mock_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Verify error was logged
        mock_logger.error.assert_called()


class TestSecurityFeatures:
    """Test security features in logging and error handling."""
    
    def test_sensitive_data_filtering(self):
        """Test that sensitive data is filtered from logs."""
        from app.core.logging import SecurityFilter
        import logging
        
        filter_instance = SecurityFilter()
        
        # Create a log record with sensitive data
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="User login with token: abc123xyz and password: secret123",
            args=(),
            exc_info=None
        )
        
        # Apply filter
        filter_instance.filter(record)
        
        # Verify sensitive data is redacted
        assert "[REDACTED]" in record.msg
        assert "abc123xyz" not in record.msg
        assert "secret123" not in record.msg
    
    def test_error_details_exclusion_in_production(self):
        """Test that error details are excluded in production."""
        error = ChronoTrailError("Test error", details={"sensitive": "data"})
        
        # Test with details excluded (production mode)
        response = create_error_response(error, include_details=False)
        assert response.details is None
        
        # Test with details included (development mode)
        response = create_error_response(error, include_details=True)
        assert response.details == {"sensitive": "data"}


if __name__ == "__main__":
    pytest.main([__file__])