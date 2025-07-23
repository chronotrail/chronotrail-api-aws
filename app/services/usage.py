"""
Usage tracking and enforcement service for subscription tier management.
"""
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Any
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import DailyUsage as DBDailyUsage, User as DBUser
from app.models.schemas import User, UsageStats, SubscriptionInfo


class UsageService:
    """Service for tracking and enforcing usage limits based on subscription tiers."""
    
    @staticmethod
    async def get_or_create_daily_usage(
        db: AsyncSession, 
        user_id: UUID, 
        usage_date: date = None
    ) -> DBDailyUsage:
        """
        Get or create daily usage record for a user.
        
        Args:
            db: Database session
            user_id: User ID
            usage_date: Date for usage tracking (defaults to today)
            
        Returns:
            DailyUsage database object
        """
        if usage_date is None:
            usage_date = date.today()
        
        # Try to get existing record
        stmt = select(DBDailyUsage).where(
            DBDailyUsage.user_id == user_id,
            DBDailyUsage.usage_date == usage_date
        )
        
        result = await db.execute(stmt)
        daily_usage = result.scalars().first()
        
        if not daily_usage:
            # Create new record
            daily_usage = DBDailyUsage(
                user_id=user_id,
                usage_date=usage_date,
                text_notes_count=0,
                media_files_count=0,
                queries_count=0
            )
            db.add(daily_usage)
            await db.commit()
            await db.refresh(daily_usage)
        
        return daily_usage
    
    @staticmethod
    def get_tier_limits(subscription_tier: str) -> Dict[str, Any]:
        """
        Get usage limits for a subscription tier.
        
        Args:
            subscription_tier: Subscription tier ('free', 'premium', 'pro')
            
        Returns:
            Dictionary containing tier limits
        """
        return settings.TIER_LIMITS.get(subscription_tier, settings.TIER_LIMITS["free"])
    
    @staticmethod
    async def check_daily_content_limit(
        db: AsyncSession, 
        user: User, 
        content_type: str = "content"
    ) -> bool:
        """
        Check if user has reached their daily content submission limit.
        
        Args:
            db: Database session
            user: User object
            content_type: Type of content ('text_notes' or 'media_files')
            
        Returns:
            True if user can submit content, False otherwise
            
        Raises:
            HTTPException: If daily limit is exceeded
        """
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        daily_limit = tier_limits["daily_content_limit"]
        
        # Unlimited for pro tier
        if daily_limit == -1:
            return True
        
        # Get today's usage
        daily_usage = await UsageService.get_or_create_daily_usage(db, user.id)
        
        # Calculate total content submissions for today
        total_content = daily_usage.text_notes_count + daily_usage.media_files_count
        
        if total_content >= daily_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Daily content limit exceeded",
                    "message": f"You have reached your daily limit of {daily_limit} content submissions. "
                              f"Upgrade your subscription for higher limits.",
                    "current_usage": total_content,
                    "daily_limit": daily_limit,
                    "subscription_tier": user.subscription_tier
                }
            )
        
        return True
    
    @staticmethod
    async def check_daily_query_limit(db: AsyncSession, user: User) -> bool:
        """
        Check if user has reached their daily query limit.
        
        Args:
            db: Database session
            user: User object
            
        Returns:
            True if user can make queries, False otherwise
            
        Raises:
            HTTPException: If daily limit is exceeded
        """
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        daily_limit = tier_limits["daily_query_limit"]
        
        # Unlimited for pro tier
        if daily_limit == -1:
            return True
        
        # Get today's usage
        daily_usage = await UsageService.get_or_create_daily_usage(db, user.id)
        
        if daily_usage.queries_count >= daily_limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Daily query limit exceeded",
                    "message": f"You have reached your daily limit of {daily_limit} queries. "
                              f"Upgrade your subscription for higher limits.",
                    "current_usage": daily_usage.queries_count,
                    "daily_limit": daily_limit,
                    "subscription_tier": user.subscription_tier
                }
            )
        
        return True
    
    @staticmethod
    async def increment_content_usage(
        db: AsyncSession, 
        user_id: UUID, 
        content_type: str
    ) -> None:
        """
        Increment content usage counter for a user.
        
        Args:
            db: Database session
            user_id: User ID
            content_type: Type of content ('text_notes' or 'media_files')
        """
        daily_usage = await UsageService.get_or_create_daily_usage(db, user_id)
        
        if content_type == "text_notes":
            daily_usage.text_notes_count += 1
        elif content_type == "media_files":
            daily_usage.media_files_count += 1
        else:
            raise ValueError(f"Invalid content type: {content_type}")
        
        await db.commit()
    
    @staticmethod
    async def increment_query_usage(db: AsyncSession, user_id: UUID) -> None:
        """
        Increment query usage counter for a user.
        
        Args:
            db: Database session
            user_id: User ID
        """
        daily_usage = await UsageService.get_or_create_daily_usage(db, user_id)
        daily_usage.queries_count += 1
        await db.commit()
    
    @staticmethod
    async def get_usage_stats(db: AsyncSession, user_id: UUID) -> UsageStats:
        """
        Get current usage statistics for a user.
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            UsageStats object with current usage and limits
        """
        # Get user to determine tier
        stmt = select(DBUser).where(DBUser.id == user_id)
        result = await db.execute(stmt)
        user = result.scalars().first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get today's usage
        daily_usage = await UsageService.get_or_create_daily_usage(db, user_id)
        
        # Get tier limits
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        
        return UsageStats(
            date=daily_usage.usage_date,
            text_notes_count=daily_usage.text_notes_count,
            media_files_count=daily_usage.media_files_count,
            queries_count=daily_usage.queries_count,
            daily_limits={
                "content_limit": tier_limits["daily_content_limit"],
                "query_limit": tier_limits["daily_query_limit"],
                "total_content_used": daily_usage.text_notes_count + daily_usage.media_files_count
            }
        )
    
    @staticmethod
    def get_subscription_info(user: User) -> SubscriptionInfo:
        """
        Get subscription information and limits for a user.
        
        Args:
            user: User object
            
        Returns:
            SubscriptionInfo object with subscription details and limits
        """
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        
        return SubscriptionInfo(
            tier=user.subscription_tier,
            expires_at=user.subscription_expires_at,
            daily_limits={
                "content_limit": tier_limits["daily_content_limit"],
                "query_limit": tier_limits["daily_query_limit"]
            },
            query_history_months=tier_limits["query_history_months"],
            max_file_size_mb=tier_limits["max_file_size_mb"],
            max_storage_mb=tier_limits["max_storage_mb"]
        )
    
    @staticmethod
    def validate_query_date_range(user: User, start_date: datetime = None) -> bool:
        """
        Validate if user can query data from a specific date range based on subscription tier.
        
        Args:
            user: User object
            start_date: Start date for the query (defaults to checking current limits)
            
        Returns:
            True if query is allowed
            
        Raises:
            HTTPException: If query date range exceeds subscription limits
        """
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        history_months = tier_limits["query_history_months"]
        
        # Unlimited for pro tier
        if history_months == -1:
            return True
        
        # If no start_date provided, just return True (no specific date range to validate)
        if start_date is None:
            return True
        
        # Calculate the earliest allowed date
        earliest_allowed = datetime.utcnow() - timedelta(days=history_months * 30)  # Approximate months to days
        
        if start_date < earliest_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Query date range exceeds subscription limits",
                    "message": f"Your {user.subscription_tier} subscription allows querying data from the last "
                              f"{history_months} months only. Upgrade your subscription to access older data.",
                    "earliest_allowed_date": earliest_allowed.isoformat(),
                    "requested_date": start_date.isoformat(),
                    "subscription_tier": user.subscription_tier,
                    "history_limit_months": history_months
                }
            )
        
        return True
    
    @staticmethod
    def validate_file_size(user: User, file_size_mb: float) -> bool:
        """
        Validate if file size is within subscription tier limits.
        
        Args:
            user: User object
            file_size_mb: File size in megabytes
            
        Returns:
            True if file size is allowed
            
        Raises:
            HTTPException: If file size exceeds subscription limits
        """
        tier_limits = UsageService.get_tier_limits(user.subscription_tier)
        max_file_size = tier_limits["max_file_size_mb"]
        
        if file_size_mb > max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "error": "File size exceeds subscription limits",
                    "message": f"Your {user.subscription_tier} subscription allows files up to "
                              f"{max_file_size}MB. This file is {file_size_mb:.2f}MB. "
                              f"Upgrade your subscription for larger file limits.",
                    "file_size_mb": file_size_mb,
                    "max_allowed_mb": max_file_size,
                    "subscription_tier": user.subscription_tier
                }
            )
        
        return True
    
    @staticmethod
    def check_subscription_validity(user: User) -> User:
        """
        Check if user's subscription is still valid and update tier if expired.
        
        Args:
            user: User object
            
        Returns:
            User object (potentially with updated tier)
        """
        # Check if subscription has expired for premium/pro users
        if (user.subscription_tier in ["premium", "pro"] and 
            user.subscription_expires_at and 
            user.subscription_expires_at < datetime.utcnow()):
            
            # Note: In a real implementation, you might want to update the user's tier
            # in the database here, but for now we'll just treat them as free tier
            # for usage calculations while keeping their original tier in the user object
            pass
        
        return user
    
    @staticmethod
    def get_effective_subscription_tier(user: User) -> str:
        """
        Get the effective subscription tier considering expiration.
        
        Args:
            user: User object
            
        Returns:
            Effective subscription tier string
        """
        # Check if subscription has expired
        if (user.subscription_tier in ["premium", "pro"] and 
            user.subscription_expires_at and 
            user.subscription_expires_at < datetime.utcnow()):
            return "free"  # Downgrade to free if expired
        
        return user.subscription_tier


# Create a singleton instance
usage_service = UsageService()