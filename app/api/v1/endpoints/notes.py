"""
Text notes endpoints for creating and managing text note submissions.
"""
from datetime import datetime
from typing import Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.services.auth import get_current_active_user
from app.services.usage import UsageService
from app.repositories.text_notes import text_note_repository
from app.aws.embedding import embedding_service
from app.models.schemas import (
    TextNote,
    TextNoteCreate,
    User,
    ErrorResponse,
    ContentType
)
from app.middleware.usage_validation import validate_content_usage

router = APIRouter()


@router.post(
    "",
    response_model=TextNote,
    status_code=status.HTTP_201_CREATED,
    summary="Create Text Note",
    description="Submit a text note with optional location data for storage and vector embedding",
    responses={
        201: {"description": "Text note created successfully", "model": TextNote},
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Usage limits exceeded", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def create_text_note(
    note_data: TextNoteCreate,
    current_user: User = Depends(validate_content_usage),
    db: AsyncSession = Depends(get_async_db)
) -> TextNote:
    """
    Create a new text note.
    
    This endpoint accepts text content with optional location data, stores it in the
    relational database, generates vector embeddings, and stores them in the vector
    database for semantic search capabilities.
    
    Args:
        note_data: Text note creation data including content, timestamp, and optional location
        current_user: Current authenticated user (validated for usage limits)
        db: Database session
        
    Returns:
        TextNote: Created text note with all metadata
        
    Raises:
        HTTPException: If validation fails, usage limits exceeded, or processing errors occur
    """
    try:
        # Create text note in relational database
        db_note = await text_note_repository.create(
            db=db,
            obj_in=note_data,
            user_id=current_user.id
        )
        
        # Prepare location data for vector storage if provided
        location_data = None
        if note_data.longitude is not None and note_data.latitude is not None:
            location_data = {
                "longitude": float(note_data.longitude),
                "latitude": float(note_data.latitude),
                "address": note_data.address,
                "names": note_data.names or []
            }
        
        # Store in vector database for semantic search
        try:
            await embedding_service.process_and_store_text(
                user_id=current_user.id,
                content_type=ContentType.NOTE,
                text=note_data.text_content,
                source_id=db_note.id,
                timestamp=note_data.timestamp,
                location=location_data
            )
        except Exception as embedding_error:
            # Log the error but don't fail the request - the note is still stored in relational DB
            # In a production system, you might want to queue this for retry
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to store text note in vector database: {str(embedding_error)}")
        
        # Increment usage counter
        await UsageService.increment_content_usage(
            db=db,
            user_id=current_user.id,
            content_type="text_notes"
        )
        
        # Convert to response model
        return TextNote(
            id=db_note.id,
            user_id=db_note.user_id,
            text_content=db_note.text_content,
            timestamp=db_note.timestamp,
            longitude=db_note.longitude,
            latitude=db_note.latitude,
            address=db_note.address,
            names=db_note.names,
            created_at=db_note.created_at,
            updated_at=db_note.created_at  # Use created_at since there's no updated_at field
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like usage limit errors)
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create text note: {str(e)}"
        )