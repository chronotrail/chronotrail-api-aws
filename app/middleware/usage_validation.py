"""
Usage validation middleware for enforcing subscription tier limits.
"""

import os
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import get_async_db
from app.models.schemas import FileType, User
from app.services.auth import AuthService, get_current_active_user
from app.services.usage import UsageService

security = HTTPBearer()


class UsageLimitMiddleware:
    """Middleware for validating usage limits based on subscription tiers."""

    # Define which endpoints require which type of usage validation
    CONTENT_ENDPOINTS = {
        "/api/v1/notes": "text_notes",
        "/api/v1/photos": "media_files",
        "/api/v1/voice": "media_files",
    }

    QUERY_ENDPOINTS = {"/api/v1/query"}

    @staticmethod
    async def validate_content_limits(
        request: Request,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_async_db),
    ) -> User:
        """
        Validate content submission limits for protected endpoints.

        Args:
            request: FastAPI request object
            current_user: Current authenticated user
            db: Database session

        Returns:
            User object if validation passes

        Raises:
            HTTPException: If usage limits are exceeded
        """
        # Only validate for POST requests (content creation)
        if request.method != "POST":
            return current_user

        # Check if this endpoint requires content limit validation
        endpoint_path = str(request.url.path)
        content_type = None

        for path, ctype in UsageLimitMiddleware.CONTENT_ENDPOINTS.items():
            if endpoint_path.startswith(path):
                content_type = ctype
                break

        if content_type:
            # Check subscription validity
            UsageService.check_subscription_validity(current_user)

            # Validate daily content limits
            await UsageService.check_daily_content_limit(db, current_user, content_type)

        return current_user

    @staticmethod
    async def validate_query_limits(
        request: Request,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_async_db),
    ) -> User:
        """
        Validate query limits for query endpoints.

        Args:
            request: FastAPI request object
            current_user: Current authenticated user
            db: Database session

        Returns:
            User object if validation passes

        Raises:
            HTTPException: If usage limits are exceeded
        """
        # Only validate for POST requests (query submission)
        if request.method != "POST":
            return current_user

        # Check if this endpoint requires query limit validation
        endpoint_path = str(request.url.path)
        is_query_endpoint = any(
            endpoint_path.startswith(path)
            for path in UsageLimitMiddleware.QUERY_ENDPOINTS
        )

        if is_query_endpoint:
            # Check subscription validity
            UsageService.check_subscription_validity(current_user)

            # Validate daily query limits
            await UsageService.check_daily_query_limit(db, current_user)

        return current_user

    @staticmethod
    async def validate_file_size_limits(
        request: Request, current_user: User = Depends(get_current_active_user)
    ) -> User:
        """
        Validate file size limits for file upload endpoints.

        Args:
            request: FastAPI request object
            current_user: Current authenticated user

        Returns:
            User object if validation passes

        Raises:
            HTTPException: If file size limits are exceeded
        """
        # Only validate for POST requests with file uploads
        if request.method != "POST":
            return current_user

        # Check if this endpoint handles file uploads
        endpoint_path = str(request.url.path)
        is_file_endpoint = any(
            endpoint_path.startswith(path)
            for path in ["/api/v1/photos", "/api/v1/voice"]
        )

        if is_file_endpoint:
            # Check subscription validity
            UsageService.check_subscription_validity(current_user)

            # Note: File size validation will be handled in the endpoint itself
            # when the file is actually received, as we can't access the file
            # content in middleware before it's parsed by FastAPI

        return current_user

    @staticmethod
    async def validate_query_date_range(
        request: Request, current_user: User = Depends(get_current_active_user)
    ) -> User:
        """
        Validate query date range limits based on subscription tier.

        Args:
            request: FastAPI request object
            current_user: Current authenticated user

        Returns:
            User object if validation passes

        Raises:
            HTTPException: If query date range exceeds subscription limits
        """
        # Check subscription validity
        UsageService.check_subscription_validity(current_user)

        # Note: Date range validation will be handled in the query endpoint
        # when the actual query parameters are parsed, as we need access to
        # the request body or query parameters

        return current_user


# Dependency functions for use in endpoints
async def validate_content_usage(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    """Dependency for validating content submission limits."""
    return await UsageLimitMiddleware.validate_content_limits(request, current_user, db)


async def validate_query_usage(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    """Dependency for validating query limits."""
    return await UsageLimitMiddleware.validate_query_limits(request, current_user, db)


async def validate_file_size_usage(
    request: Request, current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency for validating file size limits."""
    return await UsageLimitMiddleware.validate_file_size_limits(request, current_user)


async def validate_query_date_usage(
    request: Request, current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency for validating query date range limits."""
    return await UsageLimitMiddleware.validate_query_date_range(request, current_user)


async def validate_file_size(
    file: UploadFile, subscription_tier: str, file_type: FileType
) -> None:
    """
    Validate file size based on subscription tier and file type.

    Args:
        file: Uploaded file
        subscription_tier: User's subscription tier
        file_type: Type of file being uploaded

    Raises:
        HTTPException: If file size exceeds limits
    """
    # Get tier limits
    tier_limits = UsageService.get_tier_limits(subscription_tier)

    # Get file size by reading the file content
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer

    # Determine max file size based on file type and tier
    if file_type == FileType.PHOTO:
        max_size_mb = tier_limits.get("max_file_size_mb", 5)  # Default to 5MB
    elif file_type == FileType.VOICE:
        max_size_mb = tier_limits.get("max_file_size_mb", 5)  # Default to 5MB
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file_type}",
        )

    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "File too large",
                "message": f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds the limit for {subscription_tier} tier ({max_size_mb}MB). "
                f"Upgrade your subscription for higher limits.",
                "file_size_mb": round(file_size / 1024 / 1024, 1),
                "max_size_mb": max_size_mb,
                "subscription_tier": subscription_tier,
            },
        )


async def validate_date_range(
    start_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
) -> bool:
    """
    Validate query date range based on subscription tier.

    Args:
        start_date: Start date for the query
        current_user: Current authenticated user
        db: Database session

    Returns:
        True if validation passes

    Raises:
        HTTPException: If date range exceeds subscription limits
    """
    # Check subscription validity
    UsageService.check_subscription_validity(current_user)

    # If no start date provided, no need to validate
    if not start_date:
        return True

    # Get tier limits
    tier_limits = UsageService.get_tier_limits(current_user.subscription_tier)

    # Get query history months limit
    query_history_months = tier_limits.get(
        "query_history_months", 3
    )  # Default to 3 months

    # If unlimited (-1), no need to validate
    if query_history_months == -1:
        return True

    # Calculate earliest allowed date
    earliest_allowed = datetime.utcnow() - timedelta(days=query_history_months * 30)

    # Check if start date is within allowed range
    if start_date < earliest_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Date range exceeded",
                "message": f"Your {current_user.subscription_tier} subscription only allows queries up to {query_history_months} months back. "
                f"Upgrade your subscription for access to older data.",
                "allowed_months": query_history_months,
                "subscription_tier": current_user.subscription_tier,
            },
        )

    return True
