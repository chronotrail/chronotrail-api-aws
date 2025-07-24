"""
Pytest configuration and fixtures for testing.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.database import get_db, Base
from app.main import app as main_app
from app.models.schemas import User, SubscriptionTier
from app.models.database import (
    User as UserModel,
    LocationVisit as LocationVisitModel,
    TextNote as TextNoteModel,
    MediaFile as MediaFileModel,
    QuerySession as QuerySessionModel,
    DailyUsage as DailyUsageModel
)


# Create a test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Create a test engine and session
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=NullPool
)
TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
    class_=AsyncSession
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
            display_name="Test User"
        ),
        "token": "test_token",
        "token_type": "bearer"
    }


# Override dependencies
@pytest.fixture(autouse=True)
def mock_dependencies(app: FastAPI, mock_auth_user: Dict[str, Any], mock_db: AsyncSession) -> None:
    """Mock all necessary dependencies."""
    from app.services.auth import get_current_user, get_current_active_user
    from app.db.database import get_async_db
    
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