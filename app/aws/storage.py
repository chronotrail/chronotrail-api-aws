"""
S3 file storage service for ChronoTrail API.

This module provides functionality for uploading, downloading, and managing files in S3,
including file type validation, size limits, and secure URL generation.
"""
import os
import io
import uuid
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, BinaryIO, Union, Any
from uuid import UUID

from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.aws.clients import get_s3_client, get_async_s3_client
from app.aws.utils import (
    generate_s3_key,
    generate_presigned_url,
    check_file_exists,
    upload_file_to_s3,
    download_file_from_s3,
    delete_file_from_s3,
    validate_file,
    ALLOWED_IMAGE_TYPES,
    ALLOWED_AUDIO_TYPES,
)
from app.aws.exceptions import S3Error, FileProcessingError

# Configure logger
logger = get_logger(__name__)


class FileStorageService:
    """
    Service for managing file storage in S3.
    
    This service provides methods for uploading, downloading, and managing files in S3,
    including file type validation, size limits, and secure URL generation.
    """
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize the file storage service.
        
        Args:
            bucket_name: Optional S3 bucket name (defaults to settings.S3_BUCKET_NAME)
        """
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        logger.info(f"Initialized FileStorageService with bucket: {self.bucket_name}")
    
    async def upload_file(
        self,
        file: UploadFile,
        user_id: Union[str, UUID],
        file_type: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, str]] = None,
        custom_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload a file to S3 with validation.
        
        Args:
            file: FastAPI UploadFile object
            user_id: User ID for isolation
            file_type: Type of file ('photo', 'voice')
            timestamp: Optional timestamp for the file
            metadata: Optional metadata for the S3 object
            custom_key: Optional custom S3 key
            
        Returns:
            Dict with file information including S3 key and URL
            
        Raises:
            HTTPException: If file validation fails
            S3Error: If upload fails
        """
        # Convert UUID to string if needed
        if isinstance(user_id, UUID):
            user_id = str(user_id)
        
        # Determine allowed types based on file_type
        if file_type.lower() == 'photo':
            allowed_types = ALLOWED_IMAGE_TYPES
            max_size_mb = settings.MAX_FILE_SIZE_FREE  # Default to free tier
        elif file_type.lower() == 'voice':
            allowed_types = ALLOWED_AUDIO_TYPES
            max_size_mb = settings.MAX_FILE_SIZE_FREE  # Default to free tier
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Validate file
        is_valid, error_message = await validate_file(
            file,
            allowed_types,
            max_size_mb,
            'free'  # Default to free tier
        )
        
        if not is_valid:
            logger.warning(f"File validation failed: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
        
        try:
            # Generate S3 key if not provided
            if not custom_key:
                s3_key = generate_s3_key(
                    user_id=user_id,
                    file_type=file_type,
                    original_filename=file.filename,
                    timestamp=timestamp
                )
            else:
                s3_key = custom_key
            
            # Prepare metadata
            file_metadata = metadata or {}
            file_metadata.update({
                'user_id': str(user_id),
                'file_type': file_type,
                'original_filename': file.filename or 'unknown',
                'timestamp': str(timestamp or datetime.utcnow()),
            })
            
            # Get file size by reading the file content
            content = await file.read()
            file_size = len(content)
            await file.seek(0)
            
            # Upload file
            s3_url = await upload_file_to_s3(
                file=file,
                bucket_name=self.bucket_name,
                object_key=s3_key,
                content_type=file.content_type,
                metadata=file_metadata
            )
            
            # Return file information
            return {
                'file_id': str(uuid.uuid4()),  # Generate a unique ID for the file
                'bucket_name': self.bucket_name,
                's3_key': s3_key,
                's3_url': s3_url,
                'file_type': file_type,
                'original_filename': file.filename,
                'content_type': file.content_type,
                'file_size': file_size,
                'timestamp': timestamp or datetime.utcnow(),
            }
            
        except S3Error as e:
            logger.error(f"S3 error during file upload: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during file upload: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during file upload"
            )
    
    async def download_file(
        self,
        s3_key: str,
    ) -> bytes:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bytes: File content
            
        Raises:
            S3Error: If download fails
            HTTPException: If file doesn't exist
        """
        try:
            # Check if file exists
            exists = check_file_exists(self.bucket_name, s3_key)
            if not exists:
                logger.warning(f"File not found: {s3_key}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # Download file
            file_content = await download_file_from_s3(
                bucket_name=self.bucket_name,
                object_key=s3_key
            )
            
            return file_content
            
        except S3Error as e:
            logger.error(f"S3 error during file download: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download file: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during file download: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during file download"
            )
    
    async def delete_file(
        self,
        s3_key: str,
    ) -> bool:
        """
        Delete a file from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if file was deleted successfully
            
        Raises:
            S3Error: If deletion fails
        """
        try:
            # Delete file
            result = await delete_file_from_s3(
                bucket_name=self.bucket_name,
                object_key=s3_key
            )
            
            return result
            
        except S3Error as e:
            logger.error(f"S3 error during file deletion: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete file: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during file deletion: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during file deletion"
            )
    
    def generate_download_url(
        self,
        s3_key: str,
        expiration: int = 3600,  # 1 hour default
        file_name: Optional[str] = None,
    ) -> str:
        """
        Generate a secure pre-signed URL for file download.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            file_name: Optional file name for Content-Disposition
            
        Returns:
            str: Pre-signed URL
            
        Raises:
            S3Error: If URL generation fails
            HTTPException: If file doesn't exist
        """
        try:
            # Check if file exists
            exists = check_file_exists(self.bucket_name, s3_key)
            if not exists:
                logger.warning(f"File not found: {s3_key}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # Prepare parameters
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
            }
            
            # Add Content-Disposition if file_name is provided
            if file_name:
                params['ResponseContentDisposition'] = f'attachment; filename="{file_name}"'
            
            # Generate URL
            s3_client = get_s3_client()
            url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params=params,
                ExpiresIn=expiration,
                HttpMethod='GET',
            )
            
            return url
            
        except S3Error as e:
            logger.error(f"S3 error during URL generation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate download URL: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during URL generation: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during URL generation"
            )
    
    def generate_upload_url(
        self,
        s3_key: str,
        content_type: str,
        expiration: int = 3600,  # 1 hour default
        max_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a secure pre-signed URL for direct file upload.
        
        Args:
            s3_key: S3 object key
            content_type: Content type of the file
            expiration: URL expiration time in seconds
            max_size: Maximum allowed file size in bytes
            
        Returns:
            Dict with upload URL and fields
            
        Raises:
            S3Error: If URL generation fails
        """
        try:
            # Prepare conditions
            conditions = [
                {'bucket': self.bucket_name},
                {'key': s3_key},
                ['content-length-range', 0, max_size or 10 * 1024 * 1024],  # Default 10MB max
                {'Content-Type': content_type},
            ]
            
            # Generate presigned POST
            s3_client = get_s3_client()
            presigned_post = s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=s3_key,
                Fields={
                    'Content-Type': content_type,
                },
                Conditions=conditions,
                ExpiresIn=expiration,
            )
            
            return presigned_post
            
        except Exception as e:
            logger.error(f"Error generating upload URL: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate upload URL: {str(e)}"
            )
    
    def check_file_exists(self, s3_key: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            bool: True if file exists, False otherwise
        """
        return check_file_exists(self.bucket_name, s3_key)
    
    def get_file_metadata(self, s3_key: str) -> Dict[str, str]:
        """
        Get metadata for a file in S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Dict: File metadata
            
        Raises:
            S3Error: If metadata retrieval fails
            HTTPException: If file doesn't exist
        """
        try:
            # Check if file exists
            exists = check_file_exists(self.bucket_name, s3_key)
            if not exists:
                logger.warning(f"File not found: {s3_key}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="File not found"
                )
            
            # Get metadata
            s3_client = get_s3_client()
            response = s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Extract metadata
            metadata = response.get('Metadata', {})
            content_type = response.get('ContentType')
            content_length = response.get('ContentLength')
            last_modified = response.get('LastModified')
            
            return {
                'metadata': metadata,
                'content_type': content_type,
                'content_length': content_length,
                'last_modified': last_modified.isoformat() if last_modified else None,
            }
            
        except S3Error as e:
            logger.error(f"S3 error during metadata retrieval: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve file metadata: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during metadata retrieval: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred during metadata retrieval"
            )


# Create a singleton instance
file_storage_service = FileStorageService()