"""
Query API endpoints for ChronoTrail API.

This module provides endpoints for natural language querying of user timeline data,
with conversation context handling and media reference tracking.
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.database import get_db
from app.models.schemas import QueryRequest, QueryResponse, User
from app.services.auth import get_current_user
from app.services.query import query_processing_service
from app.middleware.usage_validation import validate_query_usage, validate_date_range

# Configure logger
logger = get_logger(__name__)

# Create router
router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query_timeline(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(validate_query_usage)
):
    """
    Process a natural language query against the user's timeline data.
    
    This endpoint accepts a natural language query and returns relevant information
    from the user's timeline, including location visits, text notes, photos, and
    voice recordings. It supports conversation context through session management.
    
    Args:
        query_request: Query request with natural language question
        current_user: Authenticated user
        db: Database session
        _: Query usage validation
        
    Returns:
        QueryResponse with answer, sources, and optional media references
        
    Raises:
        HTTPException: If query processing fails or usage limits are exceeded
    """
    try:
        # Process query using query processing service
        response = await query_processing_service.process_query(
            user=current_user,
            query_request=query_request,
            db=db
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )