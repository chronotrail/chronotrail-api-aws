"""
Simplified tests for usage tracking and enforcement service.
"""
import pytest
from datetime import datetime, date, timedelta
from uuid import uuid4

from fastapi import HTTPException, status

from app.services.usage import UsageService
from app.models.schemas import User, SubscriptionTier, OAuthProvider, UsageStats, SubscriptionInfo


@pytest.fixture
def free_user():
    """Create a free tier user for testing."""
    return User(
        id=uuid4(),
        email="free@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="free_user_123",
        display_name="Free User",
        subscription_tier=SubscriptionTier.FREE,
        subscription_expires_at=None,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@pytest.fixture
def premium_user():
    """Create a premium tier user for testing."""
    return User(
        id=uuid4(),
        email="premium@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="premium_user_123",
        display_name="Premium User",
        subscription_tier=SubscriptionTier.PREMIUM,
        subscription_expires_at=datetime.now() + timedelta(days=30),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@pytest.fixture
def pro_user():
    """Create a pro tier user for testing."""
    return User(
        id=uuid4(),
        email="pro@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="pro_user_123",
        display_name="Pro User",
        subscription_tier=SubscriptionTier.PRO,
        subscription_expires_at=datetime.now() + timedelta(days=365),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


@pytest.fixture
def expired_premium_user():
    """Create an expired premium user for testing."""
    return User(
        id=uuid4(),
        email="expired@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="expired_user_123",
        display_name="Expired User",
        subscription_tier=SubscriptionTier.PREMIUM,
        subscription_expires_at=datetime.now() - timedelta(days=1),
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


class TestUsageServiceLogic:
    """Test cases for UsageService logic methods that don't require database."""
    
    def test_get_tier_limits(self):
        """Test getting tier limits for different subscription tiers."""
        # Test free tier
        free_limits = UsageService.get_tier_limits("free")
        assert free_limits["daily_content_limit"] == 3
        assert free_limits["daily_query_limit"] == 10
        assert free_limits["query_history_months"] == 3
        
        # Test premium tier
        premium_limits = UsageService.get_tier_limits("premium")
        assert premium_limits["daily_content_limit"] == 10
        assert premium_limits["daily_query_limit"] == 50
        assert premium_limits["query_history_months"] == 24
        
        # Test pro tier
        pro_limits = UsageService.get_tier_limits("pro")
        assert pro_limits["daily_content_limit"] == -1  # Unlimited
        assert pro_limits["daily_query_limit"] == -1    # Unlimited
        assert pro_limits["query_history_months"] == -1 # Unlimited
        
        # Test invalid tier (should default to free)
        invalid_limits = UsageService.get_tier_limits("invalid")
        assert invalid_limits == free_limits
    
    def test_get_subscription_info(self, premium_user):
        """Test getting subscription information."""
        info = UsageService.get_subscription_info(premium_user)
        
        assert isinstance(info, SubscriptionInfo)
        assert info.tier == SubscriptionTier.PREMIUM
        assert info.expires_at == premium_user.subscription_expires_at
        assert info.daily_limits["content_limit"] == 10
        assert info.daily_limits["query_limit"] == 50
        assert info.query_history_months == 24
        assert info.max_file_size_mb == 25
        assert info.max_storage_mb == 1024
    
    def test_validate_query_date_range_within_limit(self, free_user):
        """Test query date range validation within limits."""
        # Query from 2 months ago (within 3-month limit for free tier)
        query_date = datetime.now() - timedelta(days=60)
        
        result = UsageService.validate_query_date_range(free_user, query_date)
        assert result is True
    
    def test_validate_query_date_range_exceeds_limit(self, free_user):
        """Test query date range validation exceeding limits."""
        # Query from 4 months ago (exceeds 3-month limit for free tier)
        query_date = datetime.now() - timedelta(days=120)
        
        with pytest.raises(HTTPException) as exc_info:
            UsageService.validate_query_date_range(free_user, query_date)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Query date range exceeds subscription limits" in str(exc_info.value.detail)
    
    def test_validate_query_date_range_pro_unlimited(self, pro_user):
        """Test query date range validation for pro user (unlimited)."""
        # Query from 2 years ago
        query_date = datetime.now() - timedelta(days=730)
        
        result = UsageService.validate_query_date_range(pro_user, query_date)
        assert result is True
    
    def test_validate_query_date_range_no_date(self, free_user):
        """Test query date range validation with no specific date."""
        result = UsageService.validate_query_date_range(free_user, None)
        assert result is True
    
    def test_validate_file_size_within_limit(self, free_user):
        """Test file size validation within limits."""
        result = UsageService.validate_file_size(free_user, 3.0)  # 3MB for free tier (limit is 5MB)
        assert result is True
    
    def test_validate_file_size_exceeds_limit(self, free_user):
        """Test file size validation exceeding limits."""
        with pytest.raises(HTTPException) as exc_info:
            UsageService.validate_file_size(free_user, 10.0)  # 10MB exceeds 5MB limit for free tier
        
        assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "File size exceeds subscription limits" in str(exc_info.value.detail)
    
    def test_check_subscription_validity_active(self, premium_user):
        """Test subscription validity check for active subscription."""
        result = UsageService.check_subscription_validity(premium_user)
        assert result == premium_user
    
    def test_check_subscription_validity_expired(self, expired_premium_user):
        """Test subscription validity check for expired subscription."""
        result = UsageService.check_subscription_validity(expired_premium_user)
        # Should still return the user object (tier handling is done elsewhere)
        assert result == expired_premium_user
    
    def test_get_effective_subscription_tier_active(self, premium_user):
        """Test getting effective subscription tier for active subscription."""
        tier = UsageService.get_effective_subscription_tier(premium_user)
        assert tier == "premium"
    
    def test_get_effective_subscription_tier_expired(self, expired_premium_user):
        """Test getting effective subscription tier for expired subscription."""
        tier = UsageService.get_effective_subscription_tier(expired_premium_user)
        assert tier == "free"  # Should downgrade to free
    
    def test_get_effective_subscription_tier_free(self, free_user):
        """Test getting effective subscription tier for free user."""
        tier = UsageService.get_effective_subscription_tier(free_user)
        assert tier == "free"
    
    def test_validate_file_size_premium_user(self, premium_user):
        """Test file size validation for premium user."""
        # Should allow up to 25MB for premium
        result = UsageService.validate_file_size(premium_user, 20.0)
        assert result is True
        
        # Should reject files over 25MB
        with pytest.raises(HTTPException) as exc_info:
            UsageService.validate_file_size(premium_user, 30.0)
        
        assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_validate_file_size_pro_user(self, pro_user):
        """Test file size validation for pro user."""
        # Should allow up to 100MB for pro
        result = UsageService.validate_file_size(pro_user, 90.0)
        assert result is True
        
        # Should reject files over 100MB
        with pytest.raises(HTTPException) as exc_info:
            UsageService.validate_file_size(pro_user, 150.0)
        
        assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    
    def test_validate_query_date_range_premium_user(self, premium_user):
        """Test query date range validation for premium user."""
        # Should allow queries from 2 years ago (within 24-month limit)
        query_date = datetime.now() - timedelta(days=600)  # ~20 months
        result = UsageService.validate_query_date_range(premium_user, query_date)
        assert result is True
        
        # Should reject queries from 3 years ago (exceeds 24-month limit)
        query_date = datetime.now() - timedelta(days=1100)  # ~36 months
        with pytest.raises(HTTPException) as exc_info:
            UsageService.validate_query_date_range(premium_user, query_date)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Query date range exceeds subscription limits" in str(exc_info.value.detail)