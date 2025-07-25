"""
Location visits endpoints for creating, retrieving, and updating location data.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.models.schemas import (
    ErrorResponse,
    LocationVisit,
    LocationVisitCreate,
    LocationVisitsResponse,
    LocationVisitUpdate,
    PaginationParams,
    User,
)
from app.repositories.location_visits import location_visit_repository
from app.services.auth import get_current_active_user
from app.services.usage import UsageService

router = APIRouter()


@router.post(
    "",
    response_model=LocationVisit,
    status_code=status.HTTP_201_CREATED,
    summary="Create Location Visit",
    description="Create a new location visit record with coordinates, address, and optional metadata",
    responses={
        201: {
            "description": "Location visit created successfully",
            "model": LocationVisit,
        },
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Usage limits exceeded", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
    },
)
async def create_location_visit(
    location_data: LocationVisitCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> LocationVisit:
    """
    Create a new location visit.

    Args:
        location_data: Location visit creation data
        current_user: Current authenticated user
        db: Database session

    Returns:
        Created location visit

    Raises:
        HTTPException: If creation fails or usage limits exceeded
    """
    # Check subscription validity and usage limits
    UsageService.check_subscription_validity(current_user)

    # Note: Location visits don't count against daily content limits
    # They are considered metadata/structured data rather than content

    try:
        # Create the location visit
        db_location_visit = await location_visit_repository.create(
            db=db, obj_in=location_data, user_id=current_user.id
        )

        return LocationVisit.model_validate(db_location_visit)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create location visit: {str(e)}",
        )


@router.get(
    "",
    response_model=LocationVisitsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Location Visits",
    description="Retrieve location visits with optional date filtering and pagination",
    responses={
        200: {
            "description": "Location visits retrieved successfully",
            "model": LocationVisitsResponse,
        },
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {
            "description": "Query date range exceeds subscription limits",
            "model": ErrorResponse,
        },
        422: {"description": "Invalid query parameters", "model": ErrorResponse},
    },
)
async def get_location_visits(
    start_date: Optional[datetime] = Query(
        None,
        description="Start date for filtering visits (ISO format)",
        example="2024-01-01T00:00:00Z",
    ),
    end_date: Optional[datetime] = Query(
        None,
        description="End date for filtering visits (ISO format)",
        example="2024-12-31T23:59:59Z",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> LocationVisitsResponse:
    """
    Get location visits for the current user with optional date filtering.

    Args:
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)
        page: Page number for pagination
        page_size: Number of items per page
        current_user: Current authenticated user
        db: Database session

    Returns:
        LocationVisitsResponse with visits and pagination info

    Raises:
        HTTPException: If query parameters are invalid or date range exceeds limits
    """
    # Check subscription validity
    UsageService.check_subscription_validity(current_user)

    # Validate date range
    if start_date and end_date and start_date >= end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_date must be before end_date",
        )

    # Check query date range limits based on subscription tier
    if start_date:
        UsageService.validate_query_date_range(current_user, start_date)

    try:
        # Calculate pagination offset
        skip = (page - 1) * page_size

        # Get location visits with date filtering
        visits = await location_visit_repository.get_by_user_with_date_range(
            db=db,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=page_size,
        )

        # Get total count for pagination
        total_count = await location_visit_repository.count_by_user(
            db=db, user_id=current_user.id, start_date=start_date, end_date=end_date
        )

        # Convert to response models
        visit_models = [LocationVisit.model_validate(visit) for visit in visits]

        return LocationVisitsResponse(
            visits=visit_models, total=total_count, page=page, page_size=page_size
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve location visits: {str(e)}",
        )


@router.put(
    "/{visit_id}",
    response_model=LocationVisit,
    status_code=status.HTTP_200_OK,
    summary="Update Location Visit",
    description="Update a location visit's description and names/tags",
    responses={
        200: {
            "description": "Location visit updated successfully",
            "model": LocationVisit,
        },
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Access denied to this resource", "model": ErrorResponse},
        404: {"description": "Location visit not found", "model": ErrorResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
    },
)
async def update_location_visit(
    visit_id: UUID,
    update_data: LocationVisitUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> LocationVisit:
    """
    Update a location visit's description and names/tags.

    Args:
        visit_id: ID of the location visit to update
        update_data: Update data (description and/or names)
        current_user: Current authenticated user
        db: Database session

    Returns:
        Updated location visit

    Raises:
        HTTPException: If visit not found, access denied, or update fails
    """
    # Check subscription validity
    UsageService.check_subscription_validity(current_user)

    try:
        # Update the location visit (repository handles user ownership validation)
        updated_visit = await location_visit_repository.update_by_id(
            db=db, user_id=current_user.id, visit_id=visit_id, obj_in=update_data
        )

        if not updated_visit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Location visit not found or access denied",
            )

        return LocationVisit.model_validate(updated_visit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update location visit: {str(e)}",
        )


@router.get(
    "/{visit_id}",
    response_model=LocationVisit,
    status_code=status.HTTP_200_OK,
    summary="Get Location Visit by ID",
    description="Retrieve a specific location visit by its ID",
    responses={
        200: {
            "description": "Location visit retrieved successfully",
            "model": LocationVisit,
        },
        401: {"description": "Authentication required", "model": ErrorResponse},
        403: {"description": "Access denied to this resource", "model": ErrorResponse},
        404: {"description": "Location visit not found", "model": ErrorResponse},
    },
)
async def get_location_visit(
    visit_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> LocationVisit:
    """
    Get a specific location visit by ID.

    Args:
        visit_id: ID of the location visit to retrieve
        current_user: Current authenticated user
        db: Database session

    Returns:
        Location visit data

    Raises:
        HTTPException: If visit not found or access denied
    """
    # Check subscription validity
    UsageService.check_subscription_validity(current_user)

    try:
        # Get the location visit (repository handles user ownership validation)
        visit = await location_visit_repository.get_by_user(
            db=db, user_id=current_user.id, visit_id=visit_id
        )

        if not visit:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Location visit not found or access denied",
            )

        return LocationVisit.model_validate(visit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve location visit: {str(e)}",
        )
