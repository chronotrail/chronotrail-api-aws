"""
Test configuration and fixtures.
"""
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, DECIMAL
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.sql import func
from uuid import uuid4

from app.db.base import Base


# Test database URL - using SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


# Test-specific models that work with SQLite (no ARRAY support)
class TestUser(Base):
    """Test user model for SQLite compatibility."""
    
    __tablename__ = "test_users"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    oauth_provider = Column(String(50), nullable=False)
    oauth_subject = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    profile_picture_url = Column(Text, nullable=True)
    subscription_tier = Column(String(50), nullable=False, default="free")
    subscription_expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now())


class TestLocationVisit(Base):
    """Test location visit model for SQLite compatibility."""
    
    __tablename__ = "test_location_visits"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("test_users.id", ondelete="CASCADE"), nullable=False)
    longitude = Column(DECIMAL(10, 8), nullable=False)
    latitude = Column(DECIMAL(11, 8), nullable=False)
    address = Column(Text, nullable=True)
    names = Column(Text, nullable=True)  # JSON string instead of ARRAY for SQLite
    visit_time = Column(DateTime, nullable=False)
    duration = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=func.now())
    updated_at = Column(DateTime, nullable=False, default=func.now())


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def async_engine():
    """Create async engine for testing."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        },
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    await engine.dispose()


@pytest_asyncio.fixture
async def async_db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async_session = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()