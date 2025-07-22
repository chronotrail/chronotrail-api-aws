"""
Authentication endpoints for OAuth providers and JWT token management.
"""
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.services.auth import AuthService, get_current_active_user
from app.models.schemas import (
    TokenResponse, 
    RefreshTokenRequest, 
    User, 
    ErrorResponse,
    OAuthProvider,
    GoogleOAuthRequest,
    AppleOAuthRequest
)

router = APIRouter()


@router.post(
    "/google",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Google OAuth Sign-In",
    description="Authenticate user with Google OAuth token and return JWT tokens",
    responses={
        200: {"description": "Authentication successful", "model": TokenResponse},
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Invalid Google OAuth token", "model": ErrorResponse},
        503: {"description": "Google OAuth service unavailable", "model": ErrorResponse}
    }
)
async def google_oauth_signin(
    request_data: GoogleOAuthRequest,
    db: AsyncSession = Depends(get_async_db)
) -> TokenResponse:
    """
    Authenticate user with Google OAuth token.
    
    Args:
        request_data: Google OAuth request containing access token
        db: Database session
        
    Returns:
        TokenResponse with access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    # Verify Google OAuth token and get user info
    google_user_info = await AuthService.verify_google_token(request_data.access_token)
    
    # Get or create user in our database
    user = await AuthService.get_or_create_user(
        db=db,
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_data=google_user_info
    )
    
    # Create JWT tokens for the user
    tokens = AuthService.create_tokens_for_user(user)
    
    return tokens


@router.post(
    "/apple",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Apple Sign-In",
    description="Authenticate user with Apple Sign-In token and return JWT tokens",
    responses={
        200: {"description": "Authentication successful", "model": TokenResponse},
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Invalid Apple Sign-In token", "model": ErrorResponse},
        503: {"description": "Apple authentication service unavailable", "model": ErrorResponse}
    }
)
async def apple_signin(
    request_data: AppleOAuthRequest,
    db: AsyncSession = Depends(get_async_db)
) -> TokenResponse:
    """
    Authenticate user with Apple Sign-In token.
    
    Args:
        request_data: Apple OAuth request containing identity token
        db: Database session
        
    Returns:
        TokenResponse with access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    # Verify Apple Sign-In token and get user info
    apple_user_info = await AuthService.verify_apple_token(request_data.identity_token)
    
    # Get or create user in our database
    user = await AuthService.get_or_create_user(
        db=db,
        oauth_provider=OAuthProvider.APPLE,
        oauth_data=apple_user_info
    )
    
    # Create JWT tokens for the user
    tokens = AuthService.create_tokens_for_user(user)
    
    return tokens


@router.post(
    "/refresh",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Refresh JWT Token",
    description="Generate a new access token using a valid refresh token",
    responses={
        200: {"description": "Token refresh successful", "model": TokenResponse},
        400: {"description": "Invalid request data", "model": ErrorResponse},
        401: {"description": "Invalid or expired refresh token", "model": ErrorResponse}
    }
)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_async_db)
) -> TokenResponse:
    """
    Generate a new access token using a refresh token.
    
    Args:
        refresh_request: Request containing the refresh token
        db: Database session
        
    Returns:
        TokenResponse with new access token
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    # Use the auth service to refresh the token
    new_tokens = await AuthService.refresh_access_token(
        db=db,
        refresh_token=refresh_request.refresh_token
    )
    
    return new_tokens


@router.get(
    "/me",
    response_model=User,
    status_code=status.HTTP_200_OK,
    summary="Get Current User Profile",
    description="Retrieve the current authenticated user's profile information",
    responses={
        200: {"description": "User profile retrieved successfully", "model": User},
        401: {"description": "Invalid or expired token", "model": ErrorResponse}
    }
)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get the current authenticated user's profile.
    
    Args:
        current_user: Current authenticated user from JWT token
        
    Returns:
        User profile information
    """
    return current_user