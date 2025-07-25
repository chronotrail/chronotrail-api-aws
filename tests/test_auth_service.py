"""
Tests for the OAuth authentication services.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import httpx
import pytest
from fastapi import HTTPException
from jose import jwt

from app.models.schemas import OAuthProvider, User, UserCreate, UserUpdate
from app.services.auth import AuthService


@pytest.fixture
def mock_google_user_info():
    """Mock Google OAuth user info response."""
    return {
        "sub": "123456789",
        "email": "test@example.com",
        "email_verified": True,
        "name": "Test User",
        "given_name": "Test",
        "family_name": "User",
        "picture": "https://example.com/photo.jpg",
    }


@pytest.fixture
def mock_apple_user_info():
    """Mock Apple Sign-In user info response."""
    return {
        "sub": "001234.abcdef1234567890",
        "email": "test@privaterelay.appleid.com",
        "email_verified": True,
        "is_private_email": True,
    }


@pytest.fixture
def mock_apple_jwk():
    """Mock Apple JWK for token verification."""
    return {
        "kty": "RSA",
        "kid": "test_key_id",
        "use": "sig",
        "alg": "RS256",
        "n": "sample_modulus",
        "e": "AQAB",
    }


@pytest.fixture
def mock_user():
    """Mock user object."""
    return User(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        email="test@example.com",
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="123456789",
        display_name="Test User",
        profile_picture_url="https://example.com/photo.jpg",
        subscription_tier="free",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_verify_google_token_success(mock_google_user_info):
    """Test successful Google token verification."""
    # Mock the httpx client response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_google_user_info

    # Mock the AsyncClient context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await AuthService.verify_google_token("fake_token")

        # Verify the result contains the expected user info
        assert result == mock_google_user_info
        assert result["sub"] == "123456789"
        assert result["email"] == "test@example.com"


@pytest.mark.asyncio
async def test_verify_google_token_invalid():
    """Test Google token verification with invalid token."""
    # Mock the httpx client response for invalid token
    mock_response = MagicMock()
    mock_response.status_code = 401

    # Mock the AsyncClient context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(HTTPException) as exc_info:
            await AuthService.verify_google_token("invalid_token")

        # Verify the exception details
        assert exc_info.value.status_code == 401
        assert "Invalid Google OAuth token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_google_token_connection_error():
    """Test Google token verification with connection error."""
    # Mock the AsyncClient to raise a RequestError
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.side_effect = httpx.RequestError(
        "Connection error"
    )

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(HTTPException) as exc_info:
            await AuthService.verify_google_token("fake_token")

        # Verify the exception details
        assert exc_info.value.status_code == 503
        assert "Could not connect to Google OAuth service" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_apple_token_success(mock_apple_user_info, mock_apple_jwk):
    """Test successful Apple token verification."""
    # Create a mock Apple token
    token_header = {"kid": "test_key_id", "alg": "RS256"}
    token_payload = {
        "iss": "https://appleid.apple.com",
        "sub": mock_apple_user_info["sub"],
        "email": mock_apple_user_info["email"],
        "email_verified": True,
        "is_private_email": True,
        "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
    }

    # Mock the httpx client response for Apple keys
    mock_keys_response = MagicMock()
    mock_keys_response.status_code = 200
    mock_keys_response.json.return_value = {"keys": [mock_apple_jwk]}

    # Mock the AsyncClient context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_keys_response

    # Mock jwt functions
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch("jose.jwt.get_unverified_header", return_value=token_header),
        patch("jose.jwt.decode", return_value=token_payload),
    ):

        result = await AuthService.verify_apple_token("fake_apple_token")

        # Verify the result contains the expected user info
        assert result["sub"] == mock_apple_user_info["sub"]
        assert result["email"] == mock_apple_user_info["email"]


@pytest.mark.asyncio
async def test_verify_apple_token_invalid_format():
    """Test Apple token verification with invalid token format."""
    # Mock jwt.get_unverified_header to return a header without kid
    with patch("jose.jwt.get_unverified_header", return_value={}):
        with pytest.raises(HTTPException) as exc_info:
            await AuthService.verify_apple_token("invalid_token")

        # Verify the exception details
        assert exc_info.value.status_code == 401
        assert "Invalid Apple token format" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_apple_token_key_not_found(mock_apple_jwk):
    """Test Apple token verification when key is not found."""
    # Mock the httpx client response for Apple keys
    mock_keys_response = MagicMock()
    mock_keys_response.status_code = 200
    mock_keys_response.json.return_value = {"keys": [mock_apple_jwk]}

    # Mock the AsyncClient context manager
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.get.return_value = mock_keys_response

    # Mock jwt.get_unverified_header to return a header with non-matching kid
    with (
        patch("httpx.AsyncClient", return_value=mock_client),
        patch(
            "jose.jwt.get_unverified_header",
            return_value={"kid": "non_matching_key_id"},
        ),
    ):

        with pytest.raises(HTTPException) as exc_info:
            await AuthService.verify_apple_token("fake_apple_token")

        # Verify the exception details
        assert exc_info.value.status_code == 401
        assert "Could not find matching Apple public key" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_or_create_user_existing(mock_google_user_info, mock_user):
    """Test getting an existing user."""
    # Mock database functions
    mock_db = AsyncMock()

    with patch("app.services.auth.get_user_by_oauth", return_value=mock_user):
        result = await AuthService.get_or_create_user(
            mock_db, "google", mock_google_user_info
        )

        # Verify the result is the existing user
        assert result.id == mock_user.id
        assert result.email == mock_user.email
        assert result.oauth_provider == mock_user.oauth_provider
        assert result.oauth_subject == mock_user.oauth_subject


@pytest.mark.asyncio
async def test_get_or_create_user_update(mock_google_user_info, mock_user):
    """Test updating an existing user with new information."""
    # Mock database functions
    mock_db = AsyncMock()

    # Create a modified user with updated information
    updated_user = mock_user.model_copy()
    updated_user.display_name = "Updated Name"

    # Mock the user with outdated information
    outdated_user = mock_user.model_copy()
    outdated_user.display_name = "Old Name"

    with (
        patch("app.services.auth.get_user_by_oauth", return_value=outdated_user),
        patch("app.services.auth.update_user", return_value=updated_user),
    ):

        # Update the mock_google_user_info to have a different name
        updated_info = mock_google_user_info.copy()
        updated_info["name"] = "Updated Name"

        result = await AuthService.get_or_create_user(mock_db, "google", updated_info)

        # Verify the result is the updated user
        assert result.id == updated_user.id
        assert result.display_name == "Updated Name"


@pytest.mark.asyncio
async def test_get_or_create_user_new(mock_google_user_info):
    """Test creating a new user."""
    # Mock database functions
    mock_db = AsyncMock()

    # Create a new user to be returned by create_user
    new_user = User(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        email=mock_google_user_info["email"],
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject=mock_google_user_info["sub"],
        display_name=mock_google_user_info["name"],
        profile_picture_url=mock_google_user_info["picture"],
        subscription_tier="free",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    with (
        patch("app.services.auth.get_user_by_oauth", return_value=None),
        patch("app.services.auth.create_user", return_value=new_user),
    ):

        result = await AuthService.get_or_create_user(
            mock_db, "google", mock_google_user_info
        )

        # Verify the result is the new user
        assert result.id == new_user.id
        assert result.email == mock_google_user_info["email"]
        assert result.oauth_provider == "google"
        assert result.oauth_subject == mock_google_user_info["sub"]
        assert result.display_name == mock_google_user_info["name"]


@pytest.mark.asyncio
async def test_get_or_create_user_missing_info():
    """Test handling missing OAuth information."""
    # Mock database functions
    mock_db = AsyncMock()

    # Create OAuth data with missing fields
    incomplete_data = {"name": "Test User"}

    with pytest.raises(HTTPException) as exc_info:
        await AuthService.get_or_create_user(mock_db, "google", incomplete_data)

    # Verify the exception details
    assert exc_info.value.status_code == 400
    assert "Missing required OAuth user information" in exc_info.value.detail


def test_create_access_token():
    """Test creating a JWT access token."""
    # Test data
    user_id = "12345678-1234-5678-1234-567812345678"
    user_data = {"sub": user_id, "email": "test@example.com", "tier": "free"}

    # Create token with explicit expiration
    expires = timedelta(minutes=15)

    with patch("jose.jwt.encode", return_value="mock_token"):
        token = AuthService.create_access_token(user_data, expires)
        assert token == "mock_token"


def test_decode_token_valid():
    """Test decoding a valid JWT token."""
    # Mock token payload
    token_payload = {
        "sub": "12345678-1234-5678-1234-567812345678",
        "email": "test@example.com",
        "tier": "free",
        "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
    }

    with patch("jose.jwt.decode", return_value=token_payload):
        result = AuthService.decode_token("valid_token")
        assert result == token_payload


def test_decode_token_expired():
    """Test decoding an expired JWT token."""
    with patch("jose.jwt.decode", side_effect=jwt.ExpiredSignatureError):
        with pytest.raises(HTTPException) as exc_info:
            AuthService.decode_token("expired_token")

        # Verify the exception details
        assert exc_info.value.status_code == 401
        assert "Token has expired" in exc_info.value.detail


def test_decode_token_invalid():
    """Test decoding an invalid JWT token."""
    with patch("jose.jwt.decode", side_effect=jwt.JWTError):
        with pytest.raises(HTTPException) as exc_info:
            AuthService.decode_token("invalid_token")

        # Verify the exception details
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail


def test_ensure_user_access_allowed():
    """Test ensuring user access is allowed."""
    user_id = "12345678-1234-5678-1234-567812345678"
    resource_user_id = "12345678-1234-5678-1234-567812345678"

    result = AuthService.ensure_user_access(user_id, resource_user_id)
    assert result is True


def test_ensure_user_access_denied():
    """Test ensuring user access is denied."""
    user_id = "12345678-1234-5678-1234-567812345678"
    resource_user_id = "87654321-8765-4321-8765-432187654321"

    with pytest.raises(HTTPException) as exc_info:
        AuthService.ensure_user_access(user_id, resource_user_id)

    # Verify the exception details
    assert exc_info.value.status_code == 403
    assert "Access denied to this resource" in exc_info.value.detail
