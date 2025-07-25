"""
Photo upload endpoints for creating and managing photo submissions with OCR and image analysis.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.aws.embedding import embedding_service
from app.aws.image_processing import image_processing_service
from app.aws.storage import file_storage_service
from app.core.logging import get_logger
from app.db.database import get_async_db
from app.middleware.usage_validation import validate_content_usage, validate_file_size
from app.models.schemas import (
    ContentType,
    ErrorResponse,
    FileType,
    FileUploadResponse,
    MediaFile,
    MediaFileCreate,
    User,
)
from app.repositories.media_files import media_file_repository
from app.services.auth import get_current_active_user
from app.services.usage import UsageService

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload Photo",
    description="Upload a photo file with OCR processing and image analysis integration",
    responses={
        201: {
            "description": "Photo uploaded and processed successfully",
            "model": FileUploadResponse,
        },
        400: {"description": "Invalid file or request data", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Usage limits exceeded", "model": ErrorResponse},
        413: {"description": "File too large", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
)
async def upload_photo(
    file: UploadFile = File(..., description="Photo file to upload"),
    timestamp: Optional[str] = Form(
        None, description="ISO timestamp when photo was taken"
    ),
    longitude: Optional[float] = Form(
        None, description="Longitude coordinate", ge=-180, le=180
    ),
    latitude: Optional[float] = Form(
        None, description="Latitude coordinate", ge=-90, le=90
    ),
    address: Optional[str] = Form(
        None, description="Address or location description", max_length=1000
    ),
    names: Optional[str] = Form(
        None, description="Comma-separated location names/tags"
    ),
    current_user: User = Depends(validate_content_usage),
    db: AsyncSession = Depends(get_async_db),
) -> FileUploadResponse:
    """
    Upload and process a photo file.

    This endpoint accepts a photo file with optional location data, validates it,
    stores it in S3, processes it using OCR and image analysis, stores the processed
    content in the vector database, and creates a database record.

    Args:
        file: Photo file to upload
        timestamp: Optional ISO timestamp when photo was taken
        longitude: Optional longitude coordinate
        latitude: Optional latitude coordinate
        address: Optional address or location description
        names: Optional comma-separated location names/tags
        current_user: Current authenticated user (validated for usage limits)
        db: Database session

    Returns:
        FileUploadResponse: Upload confirmation with processing status

    Raises:
        HTTPException: If validation fails, usage limits exceeded, or processing errors occur
    """
    try:
        # Validate file size based on user's subscription tier
        await validate_file_size(file, current_user.subscription_tier, FileType.PHOTO)

        # Parse timestamp if provided
        parsed_timestamp = None
        if timestamp:
            try:
                parsed_timestamp = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
                # Ensure timestamp is not in the future (convert to naive UTC for comparison)
                if parsed_timestamp.replace(tzinfo=None) > datetime.utcnow():
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Timestamp cannot be in the future",
                    )
                # Convert to naive datetime for database storage
                parsed_timestamp = parsed_timestamp.replace(tzinfo=None)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid timestamp format. Use ISO format (e.g., 2024-01-01T12:00:00Z)",
                )
        else:
            parsed_timestamp = datetime.utcnow()

        # Validate location data consistency
        if (longitude is None) != (latitude is None):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both longitude and latitude must be provided together or both omitted",
            )

        # Parse names if provided
        parsed_names = None
        if names:
            parsed_names = [name.strip() for name in names.split(",") if name.strip()]
            if len(parsed_names) > 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Maximum 10 location names/tags allowed",
                )
            for name in parsed_names:
                if len(name) > 100:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Each location name must be 100 characters or less",
                    )

        # Upload file to S3
        try:
            upload_result = await file_storage_service.upload_file(
                file=file,
                user_id=current_user.id,
                file_type="photo",
                timestamp=parsed_timestamp,
            )
        except HTTPException:
            # Re-raise HTTP exceptions from file validation
            raise
        except Exception as e:
            logger.error(f"Failed to upload photo to S3: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to upload photo file",
            )

        # Create MediaFileCreate object for database storage
        media_file_data = MediaFileCreate(
            file_type=FileType.PHOTO,
            original_filename=file.filename,
            timestamp=parsed_timestamp,
            longitude=longitude,
            latitude=latitude,
            address=address,
            names=parsed_names,
        )

        # Create media file record in database
        try:
            db_media_file = await media_file_repository.create(
                db=db,
                obj_in=media_file_data,
                user_id=current_user.id,
                file_path=upload_result["s3_key"],
                file_size=upload_result["file_size"],
            )
        except Exception as e:
            logger.error(f"Failed to create media file record: {str(e)}")
            # Try to clean up uploaded file
            try:
                await file_storage_service.delete_file(upload_result["s3_key"])
            except Exception:
                pass  # Log but don't fail on cleanup error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create media file record",
            )

        # Process image with OCR and image analysis
        processing_status = "completed"
        error_details = None
        processed_text = ""

        try:
            # Reset file pointer for processing
            await file.seek(0)

            # Process the image
            processed_result = await image_processing_service.process_image(
                file=file, original_id=db_media_file.id
            )

            processed_text = processed_result.processed_text
            processing_status = processed_result.processing_status
            error_details = processed_result.error_details

            # Store processed content in vector database if we have meaningful text
            if (
                processed_text
                and processed_text.strip()
                and processed_text.strip()
                != "No text or recognizable content found in image"
            ):
                # Prepare location data for vector storage if provided
                location_data = None
                if longitude is not None and latitude is not None:
                    location_data = {
                        "longitude": longitude,
                        "latitude": latitude,
                        "address": address,
                        "names": parsed_names or [],
                    }

                # Determine content type based on processing result
                content_type = (
                    ContentType.IMAGE_TEXT
                    if processed_result.extracted_text
                    else ContentType.IMAGE_DESC
                )

                # Store in vector database
                try:
                    await embedding_service.process_and_store_text(
                        user_id=current_user.id,
                        content_type=content_type,
                        text=processed_text,
                        source_id=db_media_file.id,
                        timestamp=parsed_timestamp,
                        location=location_data,
                        metadata={
                            "file_type": "photo",
                            "original_filename": file.filename or "unknown",
                        },
                    )
                except Exception as embedding_error:
                    # Log the error but don't fail the request - the file is still stored
                    logger.error(
                        f"Failed to store photo content in vector database: {str(embedding_error)}"
                    )
                    processing_status = "partial"
                    if not error_details:
                        error_details = f"Vector storage failed: {str(embedding_error)}"

        except Exception as processing_error:
            logger.error(f"Failed to process photo: {str(processing_error)}")
            processing_status = "failed"
            error_details = f"Image processing failed: {str(processing_error)}"
            processed_text = "Image processing failed"

        # Increment usage counter
        try:
            await UsageService.increment_content_usage(
                db=db, user_id=current_user.id, content_type="media_files"
            )
        except Exception as e:
            logger.error(f"Failed to increment usage counter: {str(e)}")
            # Don't fail the request for usage counter errors

        # Return success response
        return FileUploadResponse(
            file_id=db_media_file.id,
            message=(
                f"Photo uploaded and processed successfully. {processed_text[:100]}..."
                if len(processed_text) > 100
                else processed_text
            ),
            processing_status=processing_status,
            error_details=error_details if processing_status != "completed" else None,
            file_type=FileType.PHOTO,
            file_size=upload_result["file_size"],
        )

    except HTTPException:
        # Re-raise HTTP exceptions (like usage limit errors, validation errors)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in photo upload: {str(e)}")
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload photo: {str(e)}",
        )
