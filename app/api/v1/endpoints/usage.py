"""
Usage and subscription endpoints for tracking user limits and subscription details.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.services.auth import get_current_active_user
from app.services.usage import UsageService
from app.models.schemas import User, UsageStats, SubscriptionInfo, ErrorResponse

router = APIRouter()


@router.get(
    "/",
    response_model=UsageStats,
    status_code=status.HTTP_200_OK,
    summary="Get Current Usage Statistics",
    description="Retrieve current daily usage statistics for the authenticated user",
    responses={
        200: {"description": "Usage statistics retrieved successfully", "model": UsageStats},
        401: {"description": "Invalid or expired token", "model": ErrorResponse},
        404: {"description": "User not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def get_usage_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db)
) -> UsageStats:
    """
    Get current usage statistics for the authenticated user.
    
    Returns daily usage counts for text notes, media files, and queries,
    along with the daily limits based on the user's subscription tier.
    
    Args:
        current_user: Current authenticated user from JWT token
        db: Database session
        
    Returns:
        UsageStats with current usage and limits
        
    Raises:
        HTTPException: If user is not found or database error occurs
    """
    try:
        usage_stats = await UsageService.get_usage_stats(db, current_user.id)
        return usage_stats
    except HTTPException:
        # Re-raise HTTP exceptions from the service
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve usage statistics",
                "message": "An internal error occurred while retrieving your usage statistics. Please try again later."
            }
        )


@router.get(
    "/subscription",
    response_model=SubscriptionInfo,
    status_code=status.HTTP_200_OK,
    summary="Get Subscription Details",
    description="Retrieve subscription information and limits for the authenticated user",
    responses={
        200: {"description": "Subscription information retrieved successfully", "model": SubscriptionInfo},
        401: {"description": "Invalid or expired token", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def get_subscription_info(
    current_user: User = Depends(get_current_active_user)
) -> SubscriptionInfo:
    """
    Get subscription information and limits for the authenticated user.
    
    Returns subscription tier, expiration date, daily limits, and other
    subscription-related information.
    
    Args:
        current_user: Current authenticated user from JWT token
        
    Returns:
        SubscriptionInfo with subscription details and limits
        
    Raises:
        HTTPException: If an error occurs while retrieving subscription info
    """
    try:
        # Check subscription validity and get effective tier
        UsageService.check_subscription_validity(current_user)
        subscription_info = UsageService.get_subscription_info(current_user)
        return subscription_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to retrieve subscription information",
                "message": "An internal error occurred while retrieving your subscription information. Please try again later."
            }
        )