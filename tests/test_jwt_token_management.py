"""
Tests for JWT token management functionality.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from jose import jwt

from app.services.auth import AuthService, get_current_user, get_current_active_user, require_subscription_tier
from app.models.schemas import User, TokenResponse, RefreshTokenRequest
from app.core.config import settings
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials


@pytest.fixture
def sample_user():
    """Create a sample user for testing."""
    return User(
        id=uuid4(),
        email="test@example.com",
        oauth_provider="google",
        oauth_subject="google_123",
        display_name="Test User",
        profile_picture_url="https://example.com/avatar.jpg",
        subscription_tier="free",
        subscription_expires_at=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def premium_user():
    """Create a premium user for testing."""
    return User(
        id=uuid4(),
        email="premium@example.com",
        oauth_provider="google",
        oauth_subject="google_456",
        display_name="Premium User",
        profile_picture_url="https://example.com/premium.jpg",
        subscription_tier="premium",
        subscription_expires_at=datetime.utcnow() + timedelta(days=30),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


class TestJWTTokenGeneration:
    """Test JWT token generation functionality."""
    
    def test_create_tokens_for_user(self, sample_user):
        """Test creating access and refresh tokens for a user."""
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token
        assert token_response.refresh_token
        assert token_response.token_type == "bearer"
        assert token_response.expires_in == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        
        # Verify access token contains correct claims
        access_payload = jwt.decode(
            token_response.access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert access_payload["sub"] == str(sample_user.id)
        assert access_payload["email"] == sample_user.email
        assert access_payload["tier"] == sample_user.subscription_tier
        assert access_payload["oauth_provider"] == sample_user.oauth_provider
        assert access_payload["type"] == "access"
        
        # Verify refresh token contains correct claims
        refresh_payload = jwt.decode(
            token_response.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert refresh_payload["sub"] == str(sample_user.id)
        assert refresh_payload["type"] == "refresh"
        assert "jti" in refresh_payload  # Unique token ID
    
    def test_create_tokens_for_premium_user(self, premium_user):
        """Test creating tokens for premium user includes correct tier."""
        token_response = AuthService.create_tokens_for_user(premium_user)
        
        access_payload = jwt.decode(
            token_response.access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert access_payload["tier"] == "premium"
    
    def test_access_token_expiration(self, sample_user):
        """Test that access token has correct expiration."""
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        access_payload = jwt.decode(
            token_response.access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Check expiration is approximately correct (within 1 minute tolerance)
        expected_exp = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        actual_exp = datetime.utcfromtimestamp(access_payload["exp"])
        
        assert abs((expected_exp - actual_exp).total_seconds()) < 60
    
    def test_refresh_token_expiration(self, sample_user):
        """Test that refresh token has correct expiration."""
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        refresh_payload = jwt.decode(
            token_response.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Check expiration is approximately correct (within 1 hour tolerance)
        expected_exp = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        actual_exp = datetime.utcfromtimestamp(refresh_payload["exp"])
        
        assert abs((expected_exp - actual_exp).total_seconds()) < 3600


class TestJWTTokenValidation:
    """Test JWT token validation functionality."""
    
    def test_decode_valid_token(self, sample_user):
        """Test decoding a valid token."""
        token_response = AuthService.create_tokens_for_user(sample_user)
        payload = AuthService.decode_token(token_response.access_token)
        
        assert payload["sub"] == str(sample_user.id)
        assert payload["email"] == sample_user.email
        assert payload["type"] == "access"
    
    def test_decode_expired_token(self, sample_user):
        """Test decoding an expired token raises appropriate exception."""
        # Create token with past expiration
        expired_data = {
            "sub": str(sample_user.id),
            "email": sample_user.email,
            "type": "access",
            "exp": datetime.utcnow() - timedelta(minutes=1)
        }
        
        expired_token = jwt.encode(
            expired_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        with pytest.raises(HTTPException) as exc_info:
            AuthService.decode_token(expired_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in exc_info.value.detail.lower()
    
    def test_decode_invalid_token(self):
        """Test decoding an invalid token raises appropriate exception."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(HTTPException) as exc_info:
            AuthService.decode_token(invalid_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid" in exc_info.value.detail.lower()
    
    def test_decode_token_wrong_secret(self, sample_user):
        """Test decoding token with wrong secret raises exception."""
        # Create token with different secret
        token_data = {
            "sub": str(sample_user.id),
            "email": sample_user.email,
            "type": "access"
        }
        
        wrong_secret_token = jwt.encode(
            token_data,
            "wrong-secret",
            algorithm=settings.ALGORITHM
        )
        
        with pytest.raises(HTTPException) as exc_info:
            AuthService.decode_token(wrong_secret_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestRefreshTokenFunctionality:
    """Test refresh token functionality."""
    
    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, sample_user):
        """Test successful refresh token usage."""
        # Mock database session and user lookup
        mock_db = AsyncMock()
        
        with patch('app.services.auth.get_user_by_id', return_value=sample_user):
            token_response = AuthService.create_tokens_for_user(sample_user)
            
            new_token_response = await AuthService.refresh_access_token(
                mock_db,
                token_response.refresh_token
            )
            
            assert isinstance(new_token_response, TokenResponse)
            assert new_token_response.access_token
            assert new_token_response.token_type == "bearer"
            assert new_token_response.expires_in == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            assert new_token_response.refresh_token is None  # Only new access token returned
            
            # Verify new access token has correct claims
            new_payload = jwt.decode(
                new_token_response.access_token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            assert new_payload["sub"] == str(sample_user.id)
            assert new_payload["email"] == sample_user.email
            assert new_payload["tier"] == sample_user.subscription_tier
            assert new_payload["type"] == "access"
    
    @pytest.mark.asyncio
    async def test_refresh_with_access_token_fails(self, sample_user):
        """Test that using access token for refresh fails."""
        mock_db = AsyncMock()
        
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        with pytest.raises(HTTPException) as exc_info:
            await AuthService.refresh_access_token(mock_db, token_response.access_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid token type" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_refresh_with_nonexistent_user(self, sample_user):
        """Test refresh token with user that no longer exists."""
        mock_db = AsyncMock()
        
        with patch('app.services.auth.get_user_by_id', return_value=None):
            token_response = AuthService.create_tokens_for_user(sample_user)
            
            with pytest.raises(HTTPException) as exc_info:
                await AuthService.refresh_access_token(mock_db, token_response.refresh_token)
            
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
            assert "user not found" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_refresh_with_expired_refresh_token(self, sample_user):
        """Test refresh with expired refresh token."""
        mock_db = AsyncMock()
        
        # Create expired refresh token
        expired_refresh_data = {
            "sub": str(sample_user.id),
            "type": "refresh",
            "jti": "test_jti",
            "exp": datetime.utcnow() - timedelta(days=1)
        }
        
        expired_refresh_token = jwt.encode(
            expired_refresh_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await AuthService.refresh_access_token(mock_db, expired_refresh_token)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in exc_info.value.detail.lower()


class TestAuthenticationMiddleware:
    """Test authentication middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self, sample_user):
        """Test successful user authentication via middleware."""
        mock_db = AsyncMock()
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token_response.access_token
        )
        
        with patch('app.services.auth.get_user_by_id', return_value=sample_user):
            authenticated_user = await get_current_user(credentials, mock_db)
            
            assert authenticated_user.id == sample_user.id
            assert authenticated_user.email == sample_user.email
    
    @pytest.mark.asyncio
    async def test_get_current_user_with_refresh_token_fails(self, sample_user):
        """Test that using refresh token for authentication fails."""
        mock_db = AsyncMock()
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token_response.refresh_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "invalid token type" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, sample_user):
        """Test authentication with expired access token."""
        mock_db = AsyncMock()
        
        # Create expired access token
        expired_data = {
            "sub": str(sample_user.id),
            "email": sample_user.email,
            "type": "access",
            "exp": datetime.utcnow() - timedelta(minutes=1)
        }
        
        expired_token = jwt.encode(
            expired_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=expired_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "expired" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_get_current_active_user_success(self, sample_user):
        """Test getting current active user."""
        active_user = await get_current_active_user(sample_user)
        assert active_user.id == sample_user.id
    
    @pytest.mark.asyncio
    async def test_get_current_active_user_expired_subscription(self, premium_user):
        """Test active user check with expired subscription."""
        # Set subscription as expired
        premium_user.subscription_expires_at = datetime.utcnow() - timedelta(days=1)
        
        # Should still return user (usage service handles limits)
        active_user = await get_current_active_user(premium_user)
        assert active_user.id == premium_user.id


class TestSubscriptionTierRequirements:
    """Test subscription tier requirement functionality."""
    
    @pytest.mark.asyncio
    async def test_require_free_tier_with_free_user(self, sample_user):
        """Test free tier requirement with free user."""
        check_tier = require_subscription_tier("free")
        result = await check_tier(sample_user)
        assert result.id == sample_user.id
    
    @pytest.mark.asyncio
    async def test_require_premium_tier_with_premium_user(self, premium_user):
        """Test premium tier requirement with premium user."""
        check_tier = require_subscription_tier("premium")
        result = await check_tier(premium_user)
        assert result.id == premium_user.id
    
    @pytest.mark.asyncio
    async def test_require_premium_tier_with_free_user_fails(self, sample_user):
        """Test premium tier requirement with free user fails."""
        check_tier = require_subscription_tier("premium")
        
        with pytest.raises(HTTPException) as exc_info:
            await check_tier(sample_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "premium subscription" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_require_pro_tier_with_free_user_fails(self, sample_user):
        """Test pro tier requirement with free user fails."""
        check_tier = require_subscription_tier("pro")
        
        with pytest.raises(HTTPException) as exc_info:
            await check_tier(sample_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "pro subscription" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_tier_hierarchy(self):
        """Test that higher tiers can access lower tier features."""
        pro_user = User(
            id=uuid4(),
            email="pro@example.com",
            oauth_provider="google",
            oauth_subject="google_789",
            display_name="Pro User",
            subscription_tier="pro",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Pro user should be able to access premium features
        check_premium = require_subscription_tier("premium")
        result = await check_premium(pro_user)
        assert result.id == pro_user.id
        
        # Pro user should be able to access free features
        check_free = require_subscription_tier("free")
        result = await check_free(pro_user)
        assert result.id == pro_user.id


class TestTokenSecurity:
    """Test token security features."""
    
    def test_refresh_token_has_unique_jti(self, sample_user):
        """Test that refresh tokens have unique JTI for revocation."""
        token1 = AuthService.create_tokens_for_user(sample_user)
        token2 = AuthService.create_tokens_for_user(sample_user)
        
        payload1 = jwt.decode(
            token1.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        payload2 = jwt.decode(
            token2.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        assert payload1["jti"] != payload2["jti"]
    
    def test_access_token_includes_all_required_claims(self, sample_user):
        """Test that access tokens include all required claims."""
        token_response = AuthService.create_tokens_for_user(sample_user)
        
        payload = jwt.decode(
            token_response.access_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        required_claims = ["sub", "email", "tier", "oauth_provider", "type", "exp"]
        for claim in required_claims:
            assert claim in payload
    
    @pytest.mark.asyncio
    async def test_token_type_validation(self, sample_user):
        """Test that token type is properly validated."""
        # Create token without type claim
        invalid_data = {
            "sub": str(sample_user.id),
            "email": sample_user.email
        }
        
        invalid_token = jwt.encode(
            invalid_data,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=invalid_token
        )
        
        mock_db = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED