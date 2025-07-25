"""
Custom exceptions for AWS service integrations.
"""

from typing import Any, Dict, Optional


class AWSServiceError(Exception):
    """Base exception for AWS service errors."""

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.service = service
        self.operation = operation
        self.error_code = error_code
        self.details = details or {}

        error_msg = f"{message}"
        if service:
            error_msg = f"{service}: {error_msg}"
        if error_code:
            error_msg = f"{error_msg} (Code: {error_code})"

        super().__init__(error_msg)


class S3Error(AWSServiceError):
    """Exception for S3 service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="S3",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class TextractError(AWSServiceError):
    """Exception for Textract service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="Textract",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class TranscribeError(AWSServiceError):
    """Exception for Transcribe service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="Transcribe",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class RekognitionError(AWSServiceError):
    """Exception for Rekognition service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="Rekognition",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class BedrockError(AWSServiceError):
    """Exception for Bedrock service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="Bedrock",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class OpenSearchError(AWSServiceError):
    """Exception for OpenSearch service errors."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            service="OpenSearch",
            operation=operation,
            error_code=error_code,
            details=details,
        )


class FileProcessingError(Exception):
    """Exception for file processing errors."""

    def __init__(
        self,
        message: str,
        file_type: Optional[str] = None,
        processing_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.file_type = file_type
        self.processing_type = processing_type
        self.details = details or {}

        error_msg = f"{message}"
        if file_type and processing_type:
            error_msg = (
                f"{processing_type} processing error for {file_type}: {error_msg}"
            )
        elif file_type:
            error_msg = f"Processing error for {file_type}: {error_msg}"

        super().__init__(error_msg)
