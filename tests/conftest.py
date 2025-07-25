"""
Pytest configuration and fixtures for testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app as main_app
from app.models.database import DailyUsage as DailyUsageModel
from app.models.database import LocationVisit as LocationVisitModel
from app.models.database import MediaFile as MediaFileModel
from app.models.database import QuerySession as QuerySessionModel
from app.models.database import TextNote as TextNoteModel
from app.models.database import User as UserModel
from app.models.schemas import SubscriptionTier, User

# Create a test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Create a test engine and session
test_engine = create_async_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}, poolclass=NullPool
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=test_engine, class_=AsyncSession
)


# Override the get_db dependency
async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get a test database session."""
    async with TestingSessionLocal() as session:
        yield session


# Create a test app with overridden dependencies
@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI app."""
    main_app.dependency_overrides[get_db] = override_get_db
    return main_app


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client for the app."""
    from httpx import ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def mock_db() -> AsyncGenerator[AsyncSession, None]:
    """Create a mock database session."""
    # For integration tests, we'll mock the database operations
    # rather than creating actual tables due to PostgreSQL-specific types
    mock_session = AsyncMock(spec=AsyncSession)
    yield mock_session


@pytest.fixture
def mock_auth_user() -> Dict[str, Any]:
    """Create a mock authenticated user."""
    user_id = uuid4()
    return {
        "user": User(
            id=user_id,
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="test_subject",
            subscription_tier=SubscriptionTier.FREE,
            display_name="Test User",
        ),
        "token": "test_token",
        "token_type": "bearer",
    }


# Override dependencies
@pytest.fixture(autouse=True)
def mock_dependencies(
    app: FastAPI, mock_auth_user: Dict[str, Any], mock_db: AsyncSession
) -> None:
    """Mock all necessary dependencies."""
    from app.db.database import get_async_db
    from app.services.auth import get_current_active_user, get_current_user

    def mock_current_user():
        return mock_auth_user["user"]

    def mock_current_active_user():
        return mock_auth_user["user"]

    def mock_get_db():
        return mock_db

    app.dependency_overrides[get_current_user] = mock_current_user
    app.dependency_overrides[get_current_active_user] = mock_current_active_user
    app.dependency_overrides[get_async_db] = mock_get_db
    yield
    app.dependency_overrides.clear()
