"""
Usage validation middleware for enforcing subscription tier limits.
"""
from typing import Callable, Dict, Any
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.services.auth import AuthService, get_current_active_user
from app.services.usage import UsageService
from app.models.schemas import User

security = HTTPBearer()


class UsageLimitMiddleware:
    """Middleware for validating usage limits based on subscription tiers."""
    
    # Define which endpoints require which type of usage validation
    CONTENT_ENDPOINTS = {
        "/api/v1/notes": "text_notes",
        "/api/v1/photos": "media_files", 
        "/api/v1/voice": "media_files"
    }
    
    QUERY_ENDPOINTS = {
        "/api/v1/query"
    }
    
    @staticmethod
    async def validate_content_limits(
        request: Request,
        current_user: User = Depends(get_current_active_user),
        db: AsyncSession = Depends(get_async_db)
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
        db: AsyncSession = Depends(get_async_db)
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
        request: Request,
        current_user: User = Depends(get_current_active_user)
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
        request: Request,
        current_user: User = Depends(get_current_active_user)
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
    db: AsyncSession = Depends(get_async_db)
) -> User:
    """Dependency for validating content submission limits."""
    return await UsageLimitMiddleware.validate_content_limits(request, current_user, db)


async def validate_query_usage(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> User:
    """Dependency for validating query limits."""
    return await UsageLimitMiddleware.validate_query_limits(request, current_user, db)


async def validate_file_size_usage(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency for validating file size limits."""
    return await UsageLimitMiddleware.validate_file_size_limits(request, current_user)


async def validate_query_date_usage(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Dependency for validating query date range limits."""
    return await UsageLimitMiddleware.validate_query_date_range(request, current_user)