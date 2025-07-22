"""
Tests for usage and subscription endpoints.
"""
import pytest
from datetime import datetime, date, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.services.usage import UsageService
from app.services.auth import get_current_active_user
from app.db.database import get_async_db
from app.models.schemas import User, SubscriptionTier, OAuthProvider, UsageStats, SubscriptionInfo


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
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
        subscription_expires_at=datetime.utcnow() + timedelta(days=30),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def mock_usage_stats():
    """Mock usage statistics."""
    return UsageStats(
        date=date.today(),
        text_notes_count=2,
        media_files_count=1,
        queries_count=5,
        daily_limits={
            "content_limit": 3,
            "query_limit": 10,
            "total_content_used": 3
        }
    )


@pytest.fixture
def mock_subscription_info():
    """Mock subscription information."""
    return SubscriptionInfo(
        tier=SubscriptionTier.FREE,
        expires_at=None,
        daily_limits={
            "content_limit": 3,
            "query_limit": 10
        },
        query_history_months=3,
        max_file_size_mb=5,
        max_storage_mb=100
    )


class TestUsageEndpoints:
    """Test cases for usage and subscription endpoints."""
    
    def test_get_usage_stats_success(self, client, free_user, mock_usage_stats, mock_db):
        """Test successful retrieval of usage statistics."""
        # Override dependencies
        def mock_get_current_user():
            return free_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        # Mock usage service
        with patch.object(UsageService, 'get_usage_stats', return_value=mock_usage_stats):
            # Make request
            response = client.get(
                "/api/v1/usage/",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            
            assert data["date"] == str(date.today())
            assert data["text_notes_count"] == 2
            assert data["media_files_count"] == 1
            assert data["queries_count"] == 5
            assert data["daily_limits"]["content_limit"] == 3
            assert data["daily_limits"]["query_limit"] == 10
            assert data["daily_limits"]["total_content_used"] == 3
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_usage_stats_user_not_found(self, client, free_user, mock_db):
        """Test usage statistics retrieval when user is not found."""
        # Override dependencies
        def mock_get_current_user():
            return free_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        # Mock service to raise HTTPException
        with patch.object(UsageService, 'get_usage_stats', side_effect=HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )):
            # Make request
            response = client.get(
                "/api/v1/usage/",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            # Verify response
            assert response.status_code == 404
            data = response.json()
            assert "User not found" in str(data["detail"])
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_usage_stats_internal_error(self, client, free_user, mock_db):
        """Test usage statistics retrieval with internal error."""
        # Override dependencies
        def mock_get_current_user():
            return free_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        # Mock service to raise generic exception
        with patch.object(UsageService, 'get_usage_stats', side_effect=Exception("Database error")):
            # Make request
            response = client.get(
                "/api/v1/usage/",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            # Verify response
            assert response.status_code == 500
            data = response.json()
            assert data["detail"]["error"] == "Failed to retrieve usage statistics"
            assert "internal error occurred" in data["detail"]["message"]
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_usage_stats_unauthorized(self, client):
        """Test usage statistics retrieval without authentication."""
        response = client.get("/api/v1/usage/")
        
        # Should return 401 or 403 depending on auth implementation
        assert response.status_code in [401, 403]
    
    def test_get_subscription_info_success(self, client, free_user, mock_subscription_info):
        """Test successful retrieval of subscription information."""
        # Override dependencies
        def mock_get_current_user():
            return free_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        
        # Mock subscription service
        with patch.object(UsageService, 'check_subscription_validity', return_value=free_user):
            with patch.object(UsageService, 'get_subscription_info', return_value=mock_subscription_info):
                # Make request
                response = client.get(
                    "/api/v1/usage/subscription",
                    headers={"Authorization": "Bearer fake_token"}
                )
                
                # Verify response
                assert response.status_code == 200
                data = response.json()
                
                assert data["tier"] == "free"
                assert data["expires_at"] is None
                assert data["daily_limits"]["content_limit"] == 3
                assert data["daily_limits"]["query_limit"] == 10
                assert data["query_history_months"] == 3
                assert data["max_file_size_mb"] == 5
                assert data["max_storage_mb"] == 100
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_subscription_info_premium_user(self, client, premium_user):
        """Test subscription information retrieval for premium user."""
        # Override dependencies
        def mock_get_current_user():
            return premium_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        
        # Mock subscription service
        premium_subscription_info = SubscriptionInfo(
            tier=SubscriptionTier.PREMIUM,
            expires_at=premium_user.subscription_expires_at,
            daily_limits={
                "content_limit": 10,
                "query_limit": 50
            },
            query_history_months=24,
            max_file_size_mb=25,
            max_storage_mb=1024
        )
        
        with patch.object(UsageService, 'check_subscription_validity', return_value=premium_user):
            with patch.object(UsageService, 'get_subscription_info', return_value=premium_subscription_info):
                # Make request
                response = client.get(
                    "/api/v1/usage/subscription",
                    headers={"Authorization": "Bearer fake_token"}
                )
                
                # Verify response
                assert response.status_code == 200
                data = response.json()
                
                assert data["tier"] == "premium"
                assert data["expires_at"] is not None
                assert data["daily_limits"]["content_limit"] == 10
                assert data["daily_limits"]["query_limit"] == 50
                assert data["query_history_months"] == 24
                assert data["max_file_size_mb"] == 25
                assert data["max_storage_mb"] == 1024
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_subscription_info_internal_error(self, client, free_user):
        """Test subscription information retrieval with internal error."""
        # Override dependencies
        def mock_get_current_user():
            return free_user
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        
        # Mock service to raise generic exception
        with patch.object(UsageService, 'check_subscription_validity', side_effect=Exception("Service error")):
            # Make request
            response = client.get(
                "/api/v1/usage/subscription",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            # Verify response
            assert response.status_code == 500
            data = response.json()
            assert data["detail"]["error"] == "Failed to retrieve subscription information"
            assert "internal error occurred" in data["detail"]["message"]
        
        # Clean up
        app.dependency_overrides.clear()
    
    def test_get_subscription_info_unauthorized(self, client):
        """Test subscription information retrieval without authentication."""
        response = client.get("/api/v1/usage/subscription")
        
        # Should return 401 or 403 depending on auth implementation
        assert response.status_code in [401, 403]


class TestUsageMiddleware:
    """Test cases for usage validation middleware."""
    
    @pytest.mark.asyncio
    async def test_validate_content_limits_within_limit(self, mock_db, free_user):
        """Test content limit validation when user is within limits."""
        from app.middleware.usage_validation import UsageLimitMiddleware
        from fastapi import Request
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/notes"
        
        # Mock usage service
        with patch.object(UsageService, 'check_subscription_validity', return_value=free_user):
            with patch.object(UsageService, 'check_daily_content_limit', return_value=True):
                result = await UsageLimitMiddleware.validate_content_limits(
                    mock_request, free_user, mock_db
                )
                
                assert result == free_user
    
    @pytest.mark.asyncio
    async def test_validate_content_limits_exceeded(self, mock_db, free_user):
        """Test content limit validation when user has exceeded limits."""
        from app.middleware.usage_validation import UsageLimitMiddleware
        from fastapi import Request
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/notes"
        
        # Mock usage service to raise exception
        with patch.object(UsageService, 'check_subscription_validity', return_value=free_user):
            with patch.object(UsageService, 'check_daily_content_limit', 
                            side_effect=HTTPException(status_code=429, detail="Limit exceeded")):
                with pytest.raises(HTTPException) as exc_info:
                    await UsageLimitMiddleware.validate_content_limits(
                        mock_request, free_user, mock_db
                    )
                
                assert exc_info.value.status_code == 429
    
    @pytest.mark.asyncio
    async def test_validate_content_limits_get_request(self, mock_db, free_user):
        """Test content limit validation for GET request (should skip validation)."""
        from app.middleware.usage_validation import UsageLimitMiddleware
        from fastapi import Request
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/notes"
        
        result = await UsageLimitMiddleware.validate_content_limits(
            mock_request, free_user, mock_db
        )
        
        # Should return user without validation
        assert result == free_user
    
    @pytest.mark.asyncio
    async def test_validate_query_limits_within_limit(self, mock_db, free_user):
        """Test query limit validation when user is within limits."""
        from app.middleware.usage_validation import UsageLimitMiddleware
        from fastapi import Request
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/query"
        
        # Mock usage service
        with patch.object(UsageService, 'check_subscription_validity', return_value=free_user):
            with patch.object(UsageService, 'check_daily_query_limit', return_value=True):
                result = await UsageLimitMiddleware.validate_query_limits(
                    mock_request, free_user, mock_db
                )
                
                assert result == free_user
    
    @pytest.mark.asyncio
    async def test_validate_query_limits_exceeded(self, mock_db, free_user):
        """Test query limit validation when user has exceeded limits."""
        from app.middleware.usage_validation import UsageLimitMiddleware
        from fastapi import Request
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/query"
        
        # Mock usage service to raise exception
        with patch.object(UsageService, 'check_subscription_validity', return_value=free_user):
            with patch.object(UsageService, 'check_daily_query_limit', 
                            side_effect=HTTPException(status_code=429, detail="Query limit exceeded")):
                with pytest.raises(HTTPException) as exc_info:
                    await UsageLimitMiddleware.validate_query_limits(
                        mock_request, free_user, mock_db
                    )
                
                assert exc_info.value.status_code == 429