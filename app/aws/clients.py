"""
AWS client configuration and management for ChronoTrail API.

This module provides centralized AWS client creation with proper credential management,
connection pooling, and retry logic for all AWS services used in the application.
"""
import logging
import json
import time
import random
from functools import lru_cache, wraps
from typing import Dict, Optional, Any, Union, List, Callable, TypeVar

import boto3
import aioboto3
import botocore.config
from botocore.exceptions import ClientError, BotoCoreError, ConnectionError, ConnectTimeoutError

from app.core.config import settings
from app.core.logging import get_logger
from app.aws.exceptions import (
    AWSServiceError, S3Error, TextractError, TranscribeError, 
    RekognitionError, BedrockError, OpenSearchError
)

# Configure logger
logger = get_logger(__name__)

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    'max_attempts': 3,
    'mode': 'standard',
}

# Default connection timeout (in seconds)
DEFAULT_CONNECT_TIMEOUT = 5
DEFAULT_READ_TIMEOUT = 30

# Client configuration with retry and timeout settings
DEFAULT_CONFIG = botocore.config.Config(
    retries=DEFAULT_RETRY_CONFIG,
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=DEFAULT_READ_TIMEOUT,
    max_pool_connections=10,  # Connection pooling
)

# Service-specific configurations
S3_CONFIG = botocore.config.Config(
    retries=DEFAULT_RETRY_CONFIG,
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=60,  # Longer timeout for file operations
    max_pool_connections=20,  # More connections for S3
)

BEDROCK_CONFIG = botocore.config.Config(
    retries={
        'max_attempts': 5,  # More retries for ML services
        'mode': 'adaptive',  # Adaptive retry mode for ML services
    },
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=120,  # Longer timeout for ML operations
    max_pool_connections=5,  # Fewer connections for ML services
)

TEXTRACT_CONFIG = botocore.config.Config(
    retries={
        'max_attempts': 4,
        'mode': 'adaptive',
    },
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=90,  # Longer timeout for OCR operations
    max_pool_connections=5,
)

REKOGNITION_CONFIG = botocore.config.Config(
    retries={
        'max_attempts': 4,
        'mode': 'adaptive',
    },
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=60,  # Longer timeout for image analysis
    max_pool_connections=5,
)

TRANSCRIBE_CONFIG = botocore.config.Config(
    retries={
        'max_attempts': 4,
        'mode': 'adaptive',
    },
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=90,  # Longer timeout for transcription operations
    max_pool_connections=5,
)

OPENSEARCH_CONFIG = botocore.config.Config(
    retries=DEFAULT_RETRY_CONFIG,
    connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    read_timeout=60,  # Longer timeout for search operations
    max_pool_connections=10,
)

# Type variable for generic function return type
T = TypeVar('T')


class AWSClientManager:
    """
    Manages AWS client instances with proper configuration and credential handling.
    
    This class provides methods to create and manage AWS service clients with
    appropriate retry logic, connection pooling, and error handling.
    """
    
    def __init__(self):
        """Initialize the AWS client manager with default configuration."""
        self._session = None
        self._async_session = None
        self._clients = {}
        self._async_clients = {}
        
        # AWS credentials configuration
        self.aws_config = {
            'region_name': settings.AWS_REGION,
        }
        
        # Add explicit credentials if provided in settings
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            self.aws_config.update({
                'aws_access_key_id': settings.AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': settings.AWS_SECRET_ACCESS_KEY,
            })
        
        # Add endpoint URL for LocalStack support
        if settings.AWS_ENDPOINT_URL:
            self.aws_config.update({
                'endpoint_url': settings.AWS_ENDPOINT_URL,
            })
    
    @property
    def session(self) -> boto3.Session:
        """
        Get or create a boto3 session with configured credentials.
        
        Returns:
            boto3.Session: Configured boto3 session
        """
        if self._session is None:
            self._session = boto3.Session(**self.aws_config)
        return self._session
    
    @property
    def async_session(self) -> aioboto3.Session:
        """
        Get or create an aioboto3 session with configured credentials.
        
        Returns:
            aioboto3.Session: Configured aioboto3 session
        """
        if self._async_session is None:
            self._async_session = aioboto3.Session(**self.aws_config)
        return self._async_session
    
    def get_client(
        self, 
        service_name: str, 
        config: Optional[botocore.config.Config] = None,
        **kwargs
    ) -> boto3.client:
        """
        Get a configured boto3 client for the specified AWS service.
        
        Args:
            service_name: Name of the AWS service (e.g., 's3', 'textract')
            config: Optional custom boto3 config
            **kwargs: Additional arguments to pass to boto3.client
            
        Returns:
            boto3.client: Configured boto3 client for the requested service
        """
        client_key = f"{service_name}_{id(config)}_{hash(frozenset(kwargs.items()))}"
        
        if client_key not in self._clients:
            client_config = config or DEFAULT_CONFIG
            try:
                self._clients[client_key] = self.session.client(
                    service_name,
                    config=client_config,
                    **kwargs
                )
                logger.debug(f"Created new {service_name} client")
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Failed to create {service_name} client: {str(e)}")
                raise
                
        return self._clients[client_key]
    
    def get_async_client(
        self, 
        service_name: str, 
        config: Optional[botocore.config.Config] = None,
        **kwargs
    ) -> Callable:
        """
        Get a configured aioboto3 client for the specified AWS service.
        
        Args:
            service_name: Name of the AWS service (e.g., 's3', 'textract')
            config: Optional custom boto3 config
            **kwargs: Additional arguments to pass to aioboto3.client
            
        Returns:
            aioboto3.client: Configured aioboto3 client for the requested service
        """
        client_key = f"{service_name}_async_{id(config)}_{hash(frozenset(kwargs.items()))}"
        
        # For async clients, we return the client factory rather than an instance
        # since aioboto3 clients are async context managers
        client_config = config or DEFAULT_CONFIG
        return lambda: self.async_session.client(
            service_name,
            config=client_config,
            **kwargs
        )
    
    def get_resource(
        self, 
        service_name: str, 
        **kwargs
    ) -> boto3.resource:
        """
        Get a configured boto3 resource for the specified AWS service.
        
        Args:
            service_name: Name of the AWS service (e.g., 's3', 'dynamodb')
            **kwargs: Additional arguments to pass to boto3.resource
            
        Returns:
            boto3.resource: Configured boto3 resource for the requested service
        """
        resource_key = f"{service_name}_resource_{hash(frozenset(kwargs.items()))}"
        
        if resource_key not in self._clients:
            try:
                self._clients[resource_key] = self.session.resource(
                    service_name,
                    **kwargs
                )
                logger.debug(f"Created new {service_name} resource")
            except (ClientError, BotoCoreError) as e:
                logger.error(f"Failed to create {service_name} resource: {str(e)}")
                raise
                
        return self._clients[resource_key]


# Singleton instance of the AWS client manager
aws_client_manager = AWSClientManager()


@lru_cache(maxsize=32)
def get_s3_client(**kwargs) -> boto3.client:
    """
    Get a configured S3 client.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured S3 client
    """
    return aws_client_manager.get_client('s3', config=S3_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_s3_client(**kwargs) -> Callable:
    """
    Get a configured async S3 client.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        aioboto3.client: Configured async S3 client factory
    """
    return aws_client_manager.get_async_client('s3', config=S3_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_textract_client(**kwargs) -> boto3.client:
    """
    Get a configured Textract client.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Textract client
    """
    return aws_client_manager.get_client('textract', config=TEXTRACT_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_transcribe_client(**kwargs) -> boto3.client:
    """
    Get a configured Transcribe client.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Transcribe client
    """
    return aws_client_manager.get_client('transcribe', config=TRANSCRIBE_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_rekognition_client(**kwargs) -> boto3.client:
    """
    Get a configured Rekognition client.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Rekognition client
    """
    return aws_client_manager.get_client('rekognition', **kwargs)


@lru_cache(maxsize=32)
def get_bedrock_client(**kwargs) -> boto3.client:
    """
    Get a configured Bedrock client for LLM inference.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Bedrock runtime client
    """
    return aws_client_manager.get_client('bedrock-runtime', config=BEDROCK_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_bedrock_embeddings_client(**kwargs) -> boto3.client:
    """
    Get a configured Bedrock client for embeddings generation.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Bedrock runtime client for embeddings
    """
    return aws_client_manager.get_client('bedrock-runtime', config=BEDROCK_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_cognito_idp_client(**kwargs) -> boto3.client:
    """
    Get a configured Cognito Identity Provider client.
    
    Args:
        **kwargs: Additional arguments to pass to boto3.client
        
    Returns:
        boto3.client: Configured Cognito Identity Provider client
    """
    return aws_client_manager.get_client('cognito-idp', **kwargs)


@lru_cache(maxsize=32)
def get_opensearch_client(**kwargs) -> Any:
    """
    Get a configured OpenSearch client.
    
    This function creates an OpenSearch client using the opensearch-py library,
    which is not part of boto3. It handles authentication and configuration.
    
    Args:
        **kwargs: Additional arguments to pass to OpenSearch client
        
    Returns:
        Any: Configured OpenSearch client
        
    Raises:
        ImportError: If opensearch-py package is not installed
        OpenSearchError: If client creation fails
    """
    try:
        from opensearchpy import OpenSearch, RequestsHttpConnection, AWSIAM, ConnectionPool
        from opensearchpy.helpers import bulk
        
        # OpenSearch client configuration
        opensearch_config = {
            'hosts': [settings.OPENSEARCH_ENDPOINT],
            'connection_class': RequestsHttpConnection,
            'use_ssl': True,
            'verify_certs': True,
            'timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True,
            'maxsize': 10,  # Connection pool size
            'sniff_on_start': True,  # Discover cluster nodes on startup
            'sniff_on_connection_fail': True,  # Discover cluster nodes on failure
            'sniffer_timeout': 60,  # Sniffer timeout in seconds
        }
        
        # Check if we should use IAM authentication for AWS OpenSearch Service
        if settings.OPENSEARCH_USE_IAM_AUTH:
            # Use AWS IAM credentials for authentication
            opensearch_config.update({
                'http_auth': AWSIAM(
                    region=settings.AWS_REGION,
                    access_key=settings.AWS_ACCESS_KEY_ID,
                    secret_key=settings.AWS_SECRET_ACCESS_KEY
                ),
            })
        # Otherwise use basic authentication if credentials are provided
        elif settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
            opensearch_config.update({
                'http_auth': (settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
            })
        
        # Update with any additional kwargs
        opensearch_config.update(kwargs)
        
        return OpenSearch(**opensearch_config)
    except ImportError:
        logger.error("opensearch-py package is not installed")
        raise ImportError("opensearch-py package is required for OpenSearch integration")
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {str(e)}")
        from app.aws.exceptions import OpenSearchError
        raise OpenSearchError(
            message=f"Failed to create OpenSearch client: {str(e)}",
            operation="get_opensearch_client"
        )


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (ConnectionError, ConnectTimeoutError, Exception)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Advanced retry decorator with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exceptions that should trigger a retry
        
    Returns:
        Decorator function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) reached for {func.__name__}",
                            exc_info=True
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    
                    # Add jitter if enabled (helps prevent thundering herd problem)
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Retry attempt {attempt}/{max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s due to: {str(e)}"
                    )
                    
                    time.sleep(delay)
            
            # This should never be reached due to the raise in the loop
            raise last_exception  # type: ignore
            
        return wrapper
    
    return decorator


def handle_aws_error(
    service_name: Optional[str] = None,
    error_mapping: Optional[Dict[str, type[AWSServiceError]]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Enhanced decorator to handle AWS client errors consistently with service-specific error mapping.
    
    Args:
        service_name: Name of the AWS service for logging
        error_mapping: Mapping of error codes to custom exception types
        
    Returns:
        Wrapped function with error handling
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            service = service_name or func.__module__
            operation = func.__name__
            
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                logger.error(
                    f"AWS ClientError: {error_code} - {error_message}",
                    service=service,
                    operation=operation
                )
                
                # Map to custom exception if provided
                if error_mapping and error_code in error_mapping:
                    raise error_mapping[error_code](
                        message=error_message,
                        operation=operation,
                        error_code=error_code,
                        details={'service': service}
                    )
                
                # Default error handling based on service name
                if service_name == 's3':
                    raise S3Error(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'textract':
                    raise TextractError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'transcribe':
                    raise TranscribeError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'rekognition':
                    raise RekognitionError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'bedrock-runtime':
                    raise BedrockError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                
                # Re-raise original exception if no mapping
                raise
                
            except BotoCoreError as e:
                logger.error(
                    f"AWS BotoCoreError: {str(e)}",
                    service=service,
                    operation=operation
                )
                
                # Map to service-specific exception
                if service_name == 's3':
                    raise S3Error(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'textract':
                    raise TextractError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'transcribe':
                    raise TranscribeError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'rekognition':
                    raise RekognitionError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'bedrock-runtime':
                    raise BedrockError(
                        message=str(e),
                        operation=operation
                    )
                
                # Re-raise original exception if no mapping
                raise
                
            except Exception as e:
                logger.error(
                    f"Unexpected error in AWS operation: {str(e)}",
                    service=service,
                    operation=operation,
                    exc_info=True
                )
                raise
                
        return wrapper
    
    return decorator


def handle_aws_error_async(
    service_name: Optional[str] = None,
    error_mapping: Optional[Dict[str, type[AWSServiceError]]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Enhanced decorator to handle AWS client errors consistently for async functions.
    
    Args:
        service_name: Name of the AWS service for logging
        error_mapping: Mapping of error codes to custom exception types
        
    Returns:
        Wrapped async function with error handling
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            service = service_name or func.__module__
            operation = func.__name__
            
            try:
                return await func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                logger.error(
                    f"AWS ClientError: {error_code} - {error_message}",
                    service=service,
                    operation=operation
                )
                
                # Map to custom exception if provided
                if error_mapping and error_code in error_mapping:
                    raise error_mapping[error_code](
                        message=error_message,
                        operation=operation,
                        error_code=error_code,
                        details={'service': service}
                    )
                
                # Default error handling based on service name
                if service_name == 's3':
                    raise S3Error(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'textract':
                    raise TextractError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'transcribe':
                    raise TranscribeError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'rekognition':
                    raise RekognitionError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                elif service_name == 'bedrock-runtime':
                    raise BedrockError(
                        message=error_message,
                        operation=operation,
                        error_code=error_code
                    )
                
                # Re-raise original exception if no mapping
                raise
                
            except BotoCoreError as e:
                logger.error(
                    f"AWS BotoCoreError: {str(e)}",
                    service=service,
                    operation=operation
                )
                
                # Map to service-specific exception
                if service_name == 's3':
                    raise S3Error(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'textract':
                    raise TextractError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'transcribe':
                    raise TranscribeError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'rekognition':
                    raise RekognitionError(
                        message=str(e),
                        operation=operation
                    )
                elif service_name == 'bedrock-runtime':
                    raise BedrockError(
                        message=str(e),
                        operation=operation
                    )
                
                # Re-raise original exception if no mapping
                raise
                
            except Exception as e:
                logger.error(
                    f"Unexpected error in AWS operation: {str(e)}",
                    service=service,
                    operation=operation,
                    exc_info=True
                )
                raise
                
        return wrapper
    
    return decorator
# Additional client functions for required AWS services

@lru_cache(maxsize=32)
def get_async_textract_client(**kwargs) -> Callable:
    """
    Get a configured async Textract client.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        aioboto3.client: Configured async Textract client factory
    """
    return aws_client_manager.get_async_client('textract', config=TEXTRACT_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_rekognition_client(**kwargs) -> Callable:
    """
    Get a configured async Rekognition client.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        aioboto3.client: Configured async Rekognition client factory
    """
    return aws_client_manager.get_async_client('rekognition', config=REKOGNITION_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_transcribe_client(**kwargs) -> Callable:
    """
    Get a configured async Transcribe client.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        Callable: Configured async Transcribe client factory
    """
    return aws_client_manager.get_async_client('transcribe', config=TRANSCRIBE_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_bedrock_client(**kwargs) -> Callable:
    """
    Get a configured async Bedrock client for LLM inference.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        aioboto3.client: Configured async Bedrock runtime client factory
    """
    return aws_client_manager.get_async_client('bedrock-runtime', config=BEDROCK_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_bedrock_embeddings_client(**kwargs) -> Callable:
    """
    Get a configured async Bedrock client for embeddings generation.
    
    Args:
        **kwargs: Additional arguments to pass to aioboto3.client
        
    Returns:
        aioboto3.client: Configured async Bedrock runtime client factory for embeddings
    """
    return aws_client_manager.get_async_client('bedrock-runtime', config=BEDROCK_CONFIG, **kwargs)


@lru_cache(maxsize=32)
def get_async_opensearch_client(**kwargs) -> Any:
    """
    Get a configured async OpenSearch client.
    
    This function creates an async OpenSearch client using the opensearch-py library.
    Note: The opensearch-py library doesn't have native async support, so this
    returns a client that should be used in a thread pool for async operations.
    
    Args:
        **kwargs: Additional arguments to pass to OpenSearch client
        
    Returns:
        Any: Configured OpenSearch client for async usage
        
    Raises:
        ImportError: If opensearch-py package is not installed
        OpenSearchError: If client creation fails
    """
    try:
        from opensearchpy import AsyncOpenSearch, AsyncHttpConnection, AWSIAM
        
        # OpenSearch client configuration
        opensearch_config = {
            'hosts': [settings.OPENSEARCH_ENDPOINT],
            'connection_class': AsyncHttpConnection,
            'use_ssl': True,
            'verify_certs': True,
            'timeout': 30,
            'max_retries': 3,
            'retry_on_timeout': True,
            'maxsize': 10,  # Connection pool size
            'sniff_on_start': True,
            'sniff_on_connection_fail': True,
            'sniffer_timeout': 60,
        }
        
        # Check if we should use IAM authentication for AWS OpenSearch Service
        if settings.OPENSEARCH_USE_IAM_AUTH:
            # Use AWS IAM credentials for authentication
            opensearch_config.update({
                'http_auth': AWSIAM(
                    region=settings.AWS_REGION,
                    access_key=settings.AWS_ACCESS_KEY_ID,
                    secret_key=settings.AWS_SECRET_ACCESS_KEY
                ),
            })
        # Otherwise use basic authentication if credentials are provided
        elif settings.OPENSEARCH_USERNAME and settings.OPENSEARCH_PASSWORD:
            opensearch_config.update({
                'http_auth': (settings.OPENSEARCH_USERNAME, settings.OPENSEARCH_PASSWORD),
            })
        
        # Update with any additional kwargs
        opensearch_config.update(kwargs)
        
        return AsyncOpenSearch(**opensearch_config)
    except ImportError:
        logger.error("opensearch-py package is not installed or doesn't support async")
        raise ImportError("opensearch-py package with async support is required for async OpenSearch integration")
    except Exception as e:
        logger.error(f"Failed to create async OpenSearch client: {str(e)}")
        raise OpenSearchError(
            message=f"Failed to create async OpenSearch client: {str(e)}",
            operation="get_async_opensearch_client"
        )


# Helper functions for AWS service operations

@handle_aws_error(service_name='s3')
def check_s3_bucket_exists(bucket_name: str) -> bool:
    """
    Check if an S3 bucket exists and is accessible.
    
    Args:
        bucket_name: Name of the S3 bucket to check
        
    Returns:
        bool: True if bucket exists and is accessible, False otherwise
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == '404' or error_code == 'NoSuchBucket':
            return False
        if error_code == '403':
            logger.warning(f"Bucket {bucket_name} exists but access is forbidden")
            return True
        raise


@handle_aws_error(service_name='bedrock-runtime')
def check_bedrock_model_access(model_id: str) -> bool:
    """
    Check if a Bedrock model is accessible.
    
    Args:
        model_id: ID of the Bedrock model to check
        
    Returns:
        bool: True if model is accessible, False otherwise
    """
    try:
        bedrock_client = get_bedrock_client()
        # Use a lightweight operation to check model access
        response = bedrock_client.list_foundation_models(
            byOutputModality='TEXT'
        )
        
        # Check if the model is in the list of available models
        available_models = [model['modelId'] for model in response.get('modelSummaries', [])]
        return model_id in available_models
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AccessDeniedException' or error_code == 'ValidationException':
            return False
        raise


@handle_aws_error(service_name='opensearch')
def check_opensearch_connection() -> bool:
    """
    Check if the OpenSearch endpoint is accessible.
    
    Returns:
        bool: True if OpenSearch is accessible, False otherwise
    """
    try:
        client = get_opensearch_client()
        # Simple ping to check connection
        return client.ping()
    except Exception:
        return False


@with_retry(max_attempts=3, base_delay=1.0, max_delay=5.0)
@handle_aws_error(service_name='textract')
def check_textract_access() -> bool:
    """
    Check if Textract service is accessible.
    
    Returns:
        bool: True if Textract is accessible, False otherwise
    """
    try:
        textract_client = get_textract_client()
        # Call a simple operation that doesn't cost much but verifies access
        response = textract_client.list_document_classification_jobs(MaxResults=1)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AccessDeniedException' or error_code == 'ValidationException':
            return False
        raise


@with_retry(max_attempts=3, base_delay=1.0, max_delay=5.0)
@handle_aws_error(service_name='rekognition')
def check_rekognition_access() -> bool:
    """
    Check if Rekognition service is accessible.
    
    Returns:
        bool: True if Rekognition is accessible, False otherwise
    """
    try:
        rekognition_client = get_rekognition_client()
        # Call a simple operation that doesn't cost much but verifies access
        response = rekognition_client.list_collections(MaxResults=1)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AccessDeniedException' or error_code == 'ValidationException':
            return False
        raise


@with_retry(max_attempts=3, base_delay=1.0, max_delay=5.0)
@handle_aws_error(service_name='transcribe')
def check_transcribe_access() -> bool:
    """
    Check if Transcribe service is accessible.
    
    Returns:
        bool: True if Transcribe is accessible, False otherwise
    """
    try:
        transcribe_client = get_transcribe_client()
        # Call a simple operation that doesn't cost much but verifies access
        response = transcribe_client.list_transcription_jobs(MaxResults=1)
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'AccessDeniedException' or error_code == 'ValidationException':
            return False
        raise


def validate_aws_credentials() -> Dict[str, bool]:
    """
    Validate AWS credentials by checking access to required services.
    
    Returns:
        Dict[str, bool]: Dictionary with service names as keys and access status as values
    """
    results = {}
    
    # Check S3 access
    try:
        results['s3'] = check_s3_bucket_exists(settings.S3_BUCKET_NAME)
    except Exception as e:
        logger.error(f"Error checking S3 access: {str(e)}")
        results['s3'] = False
    
    # Check OpenSearch access
    try:
        results['opensearch'] = check_opensearch_connection()
    except Exception as e:
        logger.error(f"Error checking OpenSearch access: {str(e)}")
        results['opensearch'] = False
    
    # Check Textract access
    try:
        results['textract'] = check_textract_access()
    except Exception as e:
        logger.error(f"Error checking Textract access: {str(e)}")
        results['textract'] = False
    
    # Check Rekognition access
    try:
        results['rekognition'] = check_rekognition_access()
    except Exception as e:
        logger.error(f"Error checking Rekognition access: {str(e)}")
        results['rekognition'] = False
    
    # Check Transcribe access
    try:
        results['transcribe'] = check_transcribe_access()
    except Exception as e:
        logger.error(f"Error checking Transcribe access: {str(e)}")
        results['transcribe'] = False
    
    # Check Bedrock access
    try:
        # Use a common Bedrock model ID for testing
        results['bedrock'] = check_bedrock_model_access('amazon.titan-text-express-v1')
    except Exception as e:
        logger.error(f"Error checking Bedrock access: {str(e)}")
        results['bedrock'] = False
    
    return results