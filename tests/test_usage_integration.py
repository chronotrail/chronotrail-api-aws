"""
Integration tests for usage service with other components.
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from app.services.usage import UsageService
from app.services.auth import AuthService
from app.models.schemas import User, SubscriptionTier, OAuthProvider


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id=uuid4(),
        email="test@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="test_user_123",
        display_name="Test User",
        subscription_tier=SubscriptionTier.FREE,
        subscription_expires_at=None,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


class TestUsageIntegration:
    """Integration tests for usage service."""
    
    def test_usage_service_imports_correctly(self):
        """Test that the usage service can be imported and instantiated."""
        assert UsageService is not None
        
        # Test that static methods can be called
        limits = UsageService.get_tier_limits("free")
        assert limits is not None
        assert "daily_content_limit" in limits
    
    def test_usage_service_with_user_object(self, sample_user):
        """Test that usage service works with user objects from auth service."""
        # Test subscription info
        info = UsageService.get_subscription_info(sample_user)
        assert info.tier == SubscriptionTier.FREE
        
        # Test file size validation
        result = UsageService.validate_file_size(sample_user, 3.0)
        assert result is True
        
        # Test query date range validation
        recent_date = datetime.now() - timedelta(days=30)
        result = UsageService.validate_query_date_range(sample_user, recent_date)
        assert result is True
    
    def test_tier_hierarchy_consistency(self):
        """Test that tier limits are consistent across different tiers."""
        free_limits = UsageService.get_tier_limits("free")
        premium_limits = UsageService.get_tier_limits("premium")
        pro_limits = UsageService.get_tier_limits("pro")
        
        # Premium should have higher limits than free
        assert premium_limits["daily_content_limit"] > free_limits["daily_content_limit"]
        assert premium_limits["daily_query_limit"] > free_limits["daily_query_limit"]
        assert premium_limits["query_history_months"] > free_limits["query_history_months"]
        assert premium_limits["max_file_size_mb"] > free_limits["max_file_size_mb"]
        assert premium_limits["max_storage_mb"] > free_limits["max_storage_mb"]
        
        # Pro should have unlimited or higher limits than premium
        assert pro_limits["daily_content_limit"] == -1 or pro_limits["daily_content_limit"] > premium_limits["daily_content_limit"]
        assert pro_limits["daily_query_limit"] == -1 or pro_limits["daily_query_limit"] > premium_limits["daily_query_limit"]
        assert pro_limits["query_history_months"] == -1 or pro_limits["query_history_months"] > premium_limits["query_history_months"]
        assert pro_limits["max_file_size_mb"] > premium_limits["max_file_size_mb"]
        assert pro_limits["max_storage_mb"] > premium_limits["max_storage_mb"]
    
    def test_subscription_expiration_handling(self):
        """Test that subscription expiration is handled correctly."""
        # Create expired premium user
        expired_user = User(
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
        
        # Should return free tier limits for expired premium user
        effective_tier = UsageService.get_effective_subscription_tier(expired_user)
        assert effective_tier == "free"
        
        # Active premium user should return premium tier
        active_user = User(
            id=uuid4(),
            email="active@example.com",
            oauth_provider=OAuthProvider.GOOGLE,
            oauth_subject="active_user_123",
            display_name="Active User",
            subscription_tier=SubscriptionTier.PREMIUM,
            subscription_expires_at=datetime.now() + timedelta(days=30),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        effective_tier = UsageService.get_effective_subscription_tier(active_user)
        assert effective_tier == "premium"