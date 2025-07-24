"""
Media retrieval endpoints for accessing stored photos and voice recordings.
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.services.auth import get_current_active_user
from app.models.schemas import User
from app.repositories.media_files import media_file_repository
from app.aws.storage import file_storage_service
from app.core.logging import get_logger

# Configure logger
logger = get_logger(__name__)

router = APIRouter()


@router.get("/{media_id}")
async def get_media_file(
    media_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Retrieve a media file by ID with proper authorization checks.
    
    This endpoint allows users to download their original photos or voice recordings
    that are referenced in query responses. The endpoint enforces user isolation
    and returns appropriate error codes for unauthorized access or missing files.
    
    Args:
        media_id: UUID of the media file to retrieve
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        StreamingResponse: The original media file with appropriate headers
        
    Raises:
        HTTPException 403: If user doesn't have access to the media file
        HTTPException 404: If media file doesn't exist
        HTTPException 500: If file retrieval fails
    """
    try:
        # Get media file record from database with user authorization check
        media_file = await media_file_repository.get_by_user(
            db=db,
            user_id=current_user.id,
            file_id=media_id
        )
        
        # Check if media file exists and belongs to the user (requirement 6.3, 6.4)
        if not media_file:
            # First check if the file exists at all (for different error messages)
            file_exists = await media_file_repository.get(db=db, id=media_id)
            
            if file_exists:
                # File exists but doesn't belong to user - 403 Forbidden
                logger.warning(
                    f"User {current_user.id} attempted to access media file {media_id} "
                    f"belonging to user {file_exists.user_id}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this media file"
                )
            else:
                # File doesn't exist at all - 404 Not Found
                logger.info(f"Media file {media_id} not found")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Media file not found"
                )
        
        # Download file from S3 storage (requirement 6.2)
        try:
            file_content = await file_storage_service.download_file(
                s3_key=media_file.file_path
            )
        except HTTPException as e:
            if e.status_code == 404:
                # File record exists in DB but not in S3 - data inconsistency
                logger.error(
                    f"Media file {media_id} exists in database but not in S3 storage: {media_file.file_path}"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Media file not found in storage"
                )
            else:
                # Other S3 errors
                logger.error(f"Failed to download media file {media_id} from S3: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to retrieve media file"
                )
        
        # Determine content type based on file type and original filename
        content_type = "application/octet-stream"  # Default fallback
        
        if media_file.file_type == "photo":
            # Try to determine image content type from filename
            if media_file.original_filename:
                filename_lower = media_file.original_filename.lower()
                if filename_lower.endswith(('.jpg', '.jpeg')):
                    content_type = "image/jpeg"
                elif filename_lower.endswith('.png'):
                    content_type = "image/png"
                elif filename_lower.endswith('.heic'):
                    content_type = "image/heic"
                elif filename_lower.endswith('.webp'):
                    content_type = "image/webp"
                else:
                    content_type = "image/jpeg"  # Default for photos
            else:
                content_type = "image/jpeg"  # Default for photos
                
        elif media_file.file_type == "voice":
            # Try to determine audio content type from filename
            if media_file.original_filename:
                filename_lower = media_file.original_filename.lower()
                if filename_lower.endswith('.mp3'):
                    content_type = "audio/mpeg"
                elif filename_lower.endswith('.wav'):
                    content_type = "audio/wav"
                elif filename_lower.endswith('.m4a'):
                    content_type = "audio/mp4"
                elif filename_lower.endswith('.aac'):
                    content_type = "audio/aac"
                elif filename_lower.endswith('.ogg'):
                    content_type = "audio/ogg"
                else:
                    content_type = "audio/mpeg"  # Default for voice
            else:
                content_type = "audio/mpeg"  # Default for voice
        
        # Prepare response headers
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(file_content)),
            "Cache-Control": "private, max-age=3600",  # Cache for 1 hour
        }
        
        # Add Content-Disposition header with original filename if available
        if media_file.original_filename:
            headers["Content-Disposition"] = f'inline; filename="{media_file.original_filename}"'
        else:
            # Generate a filename based on file type and timestamp
            timestamp_str = media_file.timestamp.strftime("%Y%m%d_%H%M%S")
            if media_file.file_type == "photo":
                filename = f"photo_{timestamp_str}.jpg"
            else:
                filename = f"voice_{timestamp_str}.mp3"
            headers["Content-Disposition"] = f'inline; filename="{filename}"'
        
        # Log successful access
        logger.info(
            f"User {current_user.id} successfully accessed media file {media_id} "
            f"({media_file.file_type}, {len(file_content)} bytes)"
        )
        
        # Return file as streaming response
        return Response(
            content=file_content,
            media_type=content_type,
            headers=headers
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they already have proper status codes)
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error retrieving media file {media_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the media file"
        )


@router.get("/{media_id}/info")
async def get_media_file_info(
    media_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get metadata information about a media file without downloading it.
    
    This endpoint provides information about a media file including its type,
    size, timestamp, and location data without actually downloading the file content.
    
    Args:
        media_id: UUID of the media file
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict: Media file metadata
        
    Raises:
        HTTPException 403: If user doesn't have access to the media file
        HTTPException 404: If media file doesn't exist
    """
    try:
        # Get media file record from database with user authorization check
        media_file = await media_file_repository.get_by_user(
            db=db,
            user_id=current_user.id,
            file_id=media_id
        )
        
        # Check if media file exists and belongs to the user
        if not media_file:
            # First check if the file exists at all
            file_exists = await media_file_repository.get(db=db, id=media_id)
            
            if file_exists:
                # File exists but doesn't belong to user - 403 Forbidden
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this media file"
                )
            else:
                # File doesn't exist at all - 404 Not Found
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Media file not found"
                )
        
        # Return media file information
        return {
            "id": media_file.id,
            "file_type": media_file.file_type,
            "original_filename": media_file.original_filename,
            "file_size": media_file.file_size,
            "timestamp": media_file.timestamp,
            "location": {
                "longitude": media_file.longitude,
                "latitude": media_file.latitude,
                "address": media_file.address,
                "names": media_file.names
            } if media_file.longitude and media_file.latitude else None,
            "created_at": media_file.created_at
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error retrieving media file info {media_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving media file information"
        )


@router.get("/{media_id}/download-url")
async def get_media_download_url(
    media_id: UUID,
    expiration: int = 3600,  # 1 hour default
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Generate a secure pre-signed URL for direct media file download.
    
    This endpoint generates a time-limited pre-signed URL that allows direct
    download from S3 without going through the API server. Useful for mobile
    apps that want to download large files efficiently.
    
    Args:
        media_id: UUID of the media file
        expiration: URL expiration time in seconds (default: 3600 = 1 hour)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Dict: Pre-signed download URL and metadata
        
    Raises:
        HTTPException 403: If user doesn't have access to the media file
        HTTPException 404: If media file doesn't exist
        HTTPException 400: If expiration time is invalid
    """
    try:
        # Validate expiration parameter
        if expiration < 60 or expiration > 86400:  # 1 minute to 24 hours
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Expiration must be between 60 seconds and 86400 seconds (24 hours)"
            )
        
        # Get media file record from database with user authorization check
        media_file = await media_file_repository.get_by_user(
            db=db,
            user_id=current_user.id,
            file_id=media_id
        )
        
        # Check if media file exists and belongs to the user
        if not media_file:
            # First check if the file exists at all
            file_exists = await media_file_repository.get(db=db, id=media_id)
            
            if file_exists:
                # File exists but doesn't belong to user - 403 Forbidden
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this media file"
                )
            else:
                # File doesn't exist at all - 404 Not Found
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Media file not found"
                )
        
        # Generate pre-signed URL
        try:
            download_url = file_storage_service.generate_download_url(
                s3_key=media_file.file_path,
                expiration=expiration,
                file_name=media_file.original_filename
            )
        except HTTPException as e:
            if e.status_code == 404:
                # File record exists in DB but not in S3
                logger.error(
                    f"Media file {media_id} exists in database but not in S3 storage: {media_file.file_path}"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Media file not found in storage"
                )
            else:
                # Other S3 errors
                raise
        
        # Log URL generation
        logger.info(
            f"User {current_user.id} generated download URL for media file {media_id} "
            f"(expires in {expiration} seconds)"
        )
        
        # Return pre-signed URL and metadata
        return {
            "download_url": download_url,
            "expires_in": expiration,
            "file_info": {
                "id": media_file.id,
                "file_type": media_file.file_type,
                "original_filename": media_file.original_filename,
                "file_size": media_file.file_size,
                "timestamp": media_file.timestamp
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error generating download URL for media file {media_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating download URL"
        )