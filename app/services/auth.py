"""
Authentication services for OAuth providers integration and JWT token management.
"""

import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from jose.constants import ALGORITHMS
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.database import (
    create_user,
    get_async_db,
    get_user_by_id,
    get_user_by_oauth,
    update_user,
)
from app.models.schemas import (
    RefreshTokenRequest,
    TokenResponse,
    User,
    UserCreate,
    UserUpdate,
)

# Google OAuth verification URL
GOOGLE_OAUTH_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"

# Apple Sign-In keys URL
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"


class AuthService:
    """Service for handling OAuth authentication and user management."""

    @staticmethod
    async def verify_google_token(token: str) -> Dict[str, Any]:
        """
        Verify a Google OAuth token and return user information.

        Args:
            token: The Google OAuth ID token

        Returns:
            Dict containing user information from Google

        Raises:
            HTTPException: If token verification fails
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    GOOGLE_OAUTH_USERINFO_URL,
                    headers={"Authorization": f"Bearer {token}"},
                )

                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid Google OAuth token",
                    )

                user_info = response.json()

                # Validate required fields
                if not user_info.get("sub") or not user_info.get("email"):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Missing required user information from Google",
                    )

                return user_info

        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not connect to Google OAuth service",
            )

    @staticmethod
    async def verify_apple_token(token: str) -> Dict[str, Any]:
        """
        Verify an Apple Sign-In token and return user information.

        Args:
            token: The Apple Sign-In identity token (JWT)

        Returns:
            Dict containing user information from Apple

        Raises:
            HTTPException: If token verification fails
        """
        try:
            # Get Apple's public keys
            async with httpx.AsyncClient() as client:
                response = await client.get(APPLE_KEYS_URL)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Could not retrieve Apple public keys",
                    )

                keys_data = response.json()

            # Decode the token header to get the key ID
            token_headers = jwt.get_unverified_header(token)
            key_id = token_headers.get("kid")

            if not key_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Apple token format",
                )

            # Find the matching public key
            public_key = None
            for key in keys_data.get("keys", []):
                if key.get("kid") == key_id:
                    # Use the JWK directly with jose
                    public_key = key
                    break

            if not public_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not find matching Apple public key",
                )

            # Verify the token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=settings.APPLE_CLIENT_ID,  # This should be added to settings
                options={"verify_exp": True},
            )

            # Extract user information
            user_info = {
                "sub": payload.get("sub"),
                "email": payload.get("email"),
                # Apple tokens might not include name information
                "name": payload.get("name", ""),
                "email_verified": payload.get("email_verified", False),
            }

            # Validate required fields
            if not user_info.get("sub"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing required user information from Apple",
                )

            return user_info

        except jwt.JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid Apple token: {str(e)}",
            )
        except httpx.RequestError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not connect to Apple authentication service",
            )

    @staticmethod
    async def get_or_create_user(
        db: AsyncSession, oauth_provider: str, oauth_data: Dict[str, Any]
    ) -> User:
        """
        Get an existing user or create a new one based on OAuth data.

        Args:
            db: Database session
            oauth_provider: The OAuth provider ('google' or 'apple')
            oauth_data: User data from the OAuth provider

        Returns:
            User object
        """
        # Extract common fields
        oauth_subject = oauth_data.get("sub")
        email = oauth_data.get("email")

        if not oauth_subject or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required OAuth user information",
            )

        # Try to find existing user
        existing_user = await get_user_by_oauth(db, oauth_provider, oauth_subject)

        if existing_user:
            # Update user information if needed
            display_name = oauth_data.get("name") or oauth_data.get("given_name", "")
            profile_picture = oauth_data.get("picture", "")

            # Check if we need to update the user
            if (display_name and display_name != existing_user.display_name) or (
                profile_picture and profile_picture != existing_user.profile_picture_url
            ):

                user_update = UserUpdate(
                    display_name=display_name or existing_user.display_name,
                    profile_picture_url=profile_picture
                    or existing_user.profile_picture_url,
                )

                updated_user = await update_user(db, existing_user.id, user_update)
                return updated_user

            return existing_user

        # Create new user
        display_name = oauth_data.get("name") or oauth_data.get("given_name", "")
        profile_picture = oauth_data.get("picture", "")

        new_user = UserCreate(
            email=email,
            oauth_provider=oauth_provider,
            oauth_subject=oauth_subject,
            display_name=display_name,
            profile_picture_url=profile_picture,
            subscription_tier="free",  # Default tier for new users
        )

        created_user = await create_user(db, new_user)
        return created_user

    @staticmethod
    def create_access_token(
        data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            data: Data to encode in the token
            expires_delta: Token expiration time

        Returns:
            JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )

        return encoded_jwt

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """
        Decode and verify a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid
        """
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )

    @staticmethod
    def create_tokens_for_user(user: User) -> TokenResponse:
        """
        Create access and refresh tokens for a user.

        Args:
            user: User object

        Returns:
            TokenResponse with access and refresh tokens
        """
        # Create access token with user claims and subscription tier
        access_token_data = {
            "sub": str(user.id),
            "email": user.email,
            "tier": user.subscription_tier,
            "oauth_provider": user.oauth_provider,
            "type": "access",
        }

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = AuthService.create_access_token(
            data=access_token_data, expires_delta=access_token_expires
        )

        # Create refresh token with longer expiration
        refresh_token_data = {
            "sub": str(user.id),
            "type": "refresh",
            "jti": secrets.token_urlsafe(32),  # Unique token ID for revocation
        }

        refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        refresh_token = AuthService.create_access_token(
            data=refresh_token_data, expires_delta=refresh_token_expires
        )

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=int(access_token_expires.total_seconds()),
            refresh_token=refresh_token,
        )

    @staticmethod
    async def refresh_access_token(
        db: AsyncSession, refresh_token: str
    ) -> TokenResponse:
        """
        Create a new access token using a refresh token.

        Args:
            db: Database session
            refresh_token: Valid refresh token

        Returns:
            TokenResponse with new access token

        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            # Decode and validate refresh token
            payload = AuthService.decode_token(refresh_token)

            # Verify it's a refresh token
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                )

            # Get current user data
            user = await get_user_by_id(db, user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
                )

            # Create new access token with current user data
            access_token_data = {
                "sub": str(user.id),
                "email": user.email,
                "tier": user.subscription_tier,
                "oauth_provider": user.oauth_provider,
                "type": "access",
            }

            access_token_expires = timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
            access_token = AuthService.create_access_token(
                data=access_token_data, expires_delta=access_token_expires
            )

            return TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=int(access_token_expires.total_seconds()),
            )

        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired",
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

    @staticmethod
    def ensure_user_access(user_id: str, resource_user_id: str) -> bool:
        """
        Ensure a user has access to a resource.

        Args:
            user_id: The authenticated user's ID
            resource_user_id: The user ID associated with the resource

        Returns:
            True if access is allowed

        Raises:
            HTTPException: If access is denied
        """
        if user_id != resource_user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resource",
            )
        return True


# JWT Token validation middleware
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_db),
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        Current authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Decode the token
        payload = AuthService.decode_token(credentials.credentials)

        # Verify it's an access token
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )

        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
            )

        # Get user from database
        user = await get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
            )

        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to get the current active user.
    Can be extended to check for user status, subscription validity, etc.

    Args:
        current_user: Current authenticated user

    Returns:
        Current active user

    Raises:
        HTTPException: If user is inactive
    """
    # Check if subscription has expired for premium/pro users
    if (
        current_user.subscription_tier in ["premium", "pro"]
        and current_user.subscription_expires_at
        and current_user.subscription_expires_at < datetime.utcnow()
    ):

        # Could downgrade to free tier or raise an exception
        # For now, we'll allow access but the usage service should handle limits
        pass

    return current_user


def require_subscription_tier(required_tier: str):
    """
    Dependency factory to require a specific subscription tier.

    Args:
        required_tier: Required subscription tier ('free', 'premium', 'pro')

    Returns:
        Dependency function
    """

    async def check_tier(current_user: User = Depends(get_current_active_user)) -> User:
        tier_hierarchy = {"free": 0, "premium": 1, "pro": 2}

        user_tier_level = tier_hierarchy.get(current_user.subscription_tier, 0)
        required_tier_level = tier_hierarchy.get(required_tier, 0)

        if user_tier_level < required_tier_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {required_tier} subscription or higher",
            )

        return current_user

    return check_tier
