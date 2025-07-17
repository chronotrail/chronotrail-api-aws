"""
Utility functions for AWS service integrations.
"""
import os
import io
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, BinaryIO, Union
from urllib.parse import urlparse

from fastapi import UploadFile
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.aws.clients import get_s3_client, get_async_s3_client
from app.aws.exceptions import S3Error, FileProcessingError

# Configure logger
logger = get_logger(__name__)

# Define allowed file types
ALLOWED_IMAGE_TYPES = [
    'image/jpeg', 
    'image/png', 
    'image/gif', 
    'image/webp', 
    'image/heic'
]

ALLOWED_AUDIO_TYPES = [
    'audio/mpeg',  # MP3
    'audio/mp4',   # M4A
    'audio/wav',   # WAV
    'audio/x-wav', # WAV alternative
    'audio/ogg',   # OGG
    'audio/webm',  # WEBM
    'audio/aac',   # AAC
    'audio/flac'   # FLAC
]

# Maximum file sizes in bytes
MAX_FILE_SIZE_MB = {
    'free': 5,
    'premium': 25,
    'pro': 100
}

def get_max_file_size(subscription_tier: str) -> int:
    """
    Get maximum file size in bytes based on subscription tier.
    
    Args:
        subscription_tier: User's subscription tier ('free', 'premium', 'pro')
        
    Returns:
        int: Maximum file size in bytes
    """
    max_mb = MAX_FILE_SIZE_MB.get(subscription_tier.lower(), MAX_FILE_SIZE_MB['free'])
    return max_mb * 1024 * 1024  # Convert MB to bytes


def generate_s3_key(
    user_id: str,
    file_type: str,
    original_filename: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> str:
    """
    Generate a unique S3 key for file storage.
    
    Args:
        user_id: User ID for isolation
        file_type: Type of file ('photo', 'voice')
        original_filename: Original filename (optional)
        timestamp: Timestamp for the file (defaults to current time)
        
    Returns:
        str: Generated S3 key
    """
    # Use current time if not provided
    if timestamp is None:
        timestamp = datetime.utcnow()
        
    # Format timestamp for path structure
    date_path = timestamp.strftime("%Y/%m/%d")
    
    # Generate a unique identifier
    unique_id = str(uuid.uuid4())
    
    # Extract extension from original filename if available
    extension = ""
    if original_filename:
        _, ext = os.path.splitext(original_filename)
        if ext:
            extension = ext.lower()
    
    # Construct the key
    key = f"{user_id}/{file_type}/{date_path}/{unique_id}{extension}"
    
    return key


@retry(
    retry=retry_if_exception_type((ClientError, S3Error)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def generate_presigned_url(
    bucket_name: str,
    object_key: str,
    expiration: int = 3600,
    http_method: str = 'GET'
) -> str:
    """
    Generate a presigned URL for S3 object access.
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        expiration: URL expiration time in seconds (default: 1 hour)
        http_method: HTTP method for the URL (default: GET)
        
    Returns:
        str: Presigned URL
        
    Raises:
        S3Error: If URL generation fails
    """
    try:
        s3_client = get_s3_client()
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket_name,
                'Key': object_key,
            },
            ExpiresIn=expiration,
            HttpMethod=http_method,
        )
        return url
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Failed to generate presigned URL: {error_code} - {error_message}")
        raise S3Error(
            message=f"Failed to generate presigned URL: {error_message}",
            operation="generate_presigned_url",
            error_code=error_code
        )
    except Exception as e:
        logger.error(f"Unexpected error generating presigned URL: {str(e)}")
        raise S3Error(
            message=f"Unexpected error generating presigned URL: {str(e)}",
            operation="generate_presigned_url"
        )


def is_valid_s3_url(url: str) -> bool:
    """
    Check if a URL is a valid S3 URL.
    
    Args:
        url: URL to check
        
    Returns:
        bool: True if valid S3 URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ('http', 'https') and
            ('.s3.' in parsed.netloc or 's3.amazonaws.com' in parsed.netloc)
        )
    except Exception:
        return False


def extract_bucket_and_key_from_url(url: str) -> Tuple[str, str]:
    """
    Extract bucket name and object key from an S3 URL.
    
    Args:
        url: S3 URL
        
    Returns:
        Tuple[str, str]: Bucket name and object key
        
    Raises:
        ValueError: If URL is not a valid S3 URL
    """
    if not is_valid_s3_url(url):
        raise ValueError("Not a valid S3 URL")
    
    parsed = urlparse(url)
    
    # Handle different S3 URL formats
    if '.s3.amazonaws.com' in parsed.netloc:
        # URL format: https://bucket-name.s3.amazonaws.com/key
        bucket = parsed.netloc.split('.s3.amazonaws.com')[0]
        key = parsed.path.lstrip('/')
    elif 's3.amazonaws.com' in parsed.netloc:
        # URL format: https://s3.amazonaws.com/bucket-name/key
        path_parts = parsed.path.strip('/').split('/', 1)
        if len(path_parts) < 2:
            raise ValueError("Invalid S3 URL format")
        bucket = path_parts[0]
        key = path_parts[1]
    else:
        # URL format: https://bucket-name.s3.region.amazonaws.com/key
        bucket = parsed.netloc.split('.s3.')[0]
        key = parsed.path.lstrip('/')
    
    return bucket, key


@retry(
    retry=retry_if_exception_type((ClientError, S3Error)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def check_file_exists(bucket_name: str, object_key: str) -> bool:
    """
    Check if a file exists in S3.
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Error checking if file exists: {error_code} - {error_message}")
        raise S3Error(
            message=f"Error checking if file exists: {error_message}",
            operation="check_file_exists",
            error_code=error_code
        )
    except Exception as e:
        logger.error(f"Unexpected error checking if file exists: {str(e)}")
        raise S3Error(
            message=f"Unexpected error checking if file exists: {str(e)}",
            operation="check_file_exists"
        )# Additi
onal utility functions for AWS services

@retry(
    retry=retry_if_exception_type((ClientError, S3Error)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def upload_file_to_s3(
    file: UploadFile,
    bucket_name: str,
    object_key: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None
) -> str:
    """
    Upload a file to S3 asynchronously.
    
    Args:
        file: FastAPI UploadFile object
        bucket_name: S3 bucket name
        object_key: S3 object key
        content_type: Content type of the file
        metadata: Optional metadata for the S3 object
        
    Returns:
        str: S3 object URL
        
    Raises:
        S3Error: If upload fails
    """
    try:
        # Read file content
        content = await file.read()
        
        # Reset file pointer for potential future use
        await file.seek(0)
        
        # Determine content type if not provided
        if not content_type:
            content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
        
        # Prepare upload parameters
        upload_params = {
            'Bucket': bucket_name,
            'Key': object_key,
            'Body': content,
            'ContentType': content_type,
        }
        
        # Add metadata if provided
        if metadata:
            upload_params['Metadata'] = metadata
        
        # Get async S3 client
        s3_client_factory = get_async_s3_client()
        
        # Upload file
        async with s3_client_factory() as s3_client:
            await s3_client.upload_fileobj(
                io.BytesIO(content),
                bucket_name,
                object_key,
                ExtraArgs=upload_params
            )
        
        # Generate and return S3 URL
        return f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{object_key}"
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Failed to upload file to S3: {error_code} - {error_message}")
        raise S3Error(
            message=f"Failed to upload file to S3: {error_message}",
            operation="upload_file_to_s3",
            error_code=error_code
        )
    except Exception as e:
        logger.error(f"Unexpected error uploading file to S3: {str(e)}")
        raise S3Error(
            message=f"Unexpected error uploading file to S3: {str(e)}",
            operation="upload_file_to_s3"
        )


@retry(
    retry=retry_if_exception_type((ClientError, S3Error)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def download_file_from_s3(
    bucket_name: str,
    object_key: str
) -> bytes:
    """
    Download a file from S3 asynchronously.
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        
    Returns:
        bytes: File content
        
    Raises:
        S3Error: If download fails
    """
    try:
        # Get async S3 client
        s3_client_factory = get_async_s3_client()
        
        # Create a BytesIO object to store the file content
        file_obj = io.BytesIO()
        
        # Download file
        async with s3_client_factory() as s3_client:
            await s3_client.download_fileobj(
                bucket_name,
                object_key,
                file_obj
            )
        
        # Reset file pointer and return content
        file_obj.seek(0)
        return file_obj.read()
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Failed to download file from S3: {error_code} - {error_message}")
        raise S3Error(
            message=f"Failed to download file from S3: {error_message}",
            operation="download_file_from_s3",
            error_code=error_code
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading file from S3: {str(e)}")
        raise S3Error(
            message=f"Unexpected error downloading file from S3: {str(e)}",
            operation="download_file_from_s3"
        )


@retry(
    retry=retry_if_exception_type((ClientError, S3Error)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def delete_file_from_s3(
    bucket_name: str,
    object_key: str
) -> bool:
    """
    Delete a file from S3 asynchronously.
    
    Args:
        bucket_name: S3 bucket name
        object_key: S3 object key
        
    Returns:
        bool: True if file was deleted successfully
        
    Raises:
        S3Error: If deletion fails
    """
    try:
        # Get async S3 client
        s3_client_factory = get_async_s3_client()
        
        # Delete file
        async with s3_client_factory() as s3_client:
            await s3_client.delete_object(
                Bucket=bucket_name,
                Key=object_key
            )
        
        return True
    
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        error_message = e.response.get('Error', {}).get('Message', str(e))
        logger.error(f"Failed to delete file from S3: {error_code} - {error_message}")
        raise S3Error(
            message=f"Failed to delete file from S3: {error_message}",
            operation="delete_file_from_s3",
            error_code=error_code
        )
    except Exception as e:
        logger.error(f"Unexpected error deleting file from S3: {str(e)}")
        raise S3Error(
            message=f"Unexpected error deleting file from S3: {str(e)}",
            operation="delete_file_from_s3"
        )


def validate_file_type(
    content_type: str,
    allowed_types: List[str]
) -> bool:
    """
    Validate if a file's content type is allowed.
    
    Args:
        content_type: MIME type of the file
        allowed_types: List of allowed MIME types
        
    Returns:
        bool: True if file type is allowed, False otherwise
    """
    return content_type in allowed_types


def validate_file_size(
    file_size: int,
    max_size: int
) -> bool:
    """
    Validate if a file's size is within limits.
    
    Args:
        file_size: Size of the file in bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        bool: True if file size is within limits, False otherwise
    """
    return file_size <= max_size


async def validate_file(
    file: UploadFile,
    allowed_types: List[str],
    max_size_mb: int,
    subscription_tier: str = 'free'
) -> Tuple[bool, Optional[str]]:
    """
    Validate a file's type and size.
    
    Args:
        file: FastAPI UploadFile object
        allowed_types: List of allowed MIME types
        max_size_mb: Maximum allowed size in MB
        subscription_tier: User's subscription tier
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    # Validate file type
    content_type = file.content_type or mimetypes.guess_type(file.filename)[0] or 'application/octet-stream'
    if not validate_file_type(content_type, allowed_types):
        return False, f"File type {content_type} not allowed. Allowed types: {', '.join(allowed_types)}"
    
    # Get file size
    await file.seek(0, os.SEEK_END)
    file_size = file.tell()
    await file.seek(0)
    
    # Get max file size based on subscription tier
    max_size = get_max_file_size(subscription_tier)
    
    # Validate file size
    if not validate_file_size(file_size, max_size):
        return False, f"File size {file_size / (1024 * 1024):.2f} MB exceeds maximum allowed size of {max_size / (1024 * 1024)} MB for {subscription_tier} tier"
    
    return True, None