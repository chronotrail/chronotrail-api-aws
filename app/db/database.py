"""
Database configuration and connection management
"""
from typing import Optional
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.base import Base
from app.models.schemas import UserCreate, UserUpdate

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create AsyncEngine for async operations
async_engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create AsyncSessionLocal class
AsyncSessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=async_engine, 
    class_=AsyncSession
)

def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Dependency to get async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# Database CRUD operations for User model
async def get_user_by_oauth(db: AsyncSession, oauth_provider: str, oauth_subject: str):
    """Get user by OAuth provider and subject."""
    from app.models.database import User
    
    result = await db.execute(
        select(User).where(
            User.oauth_provider == oauth_provider,
            User.oauth_subject == oauth_subject
        )
    )
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str):
    """Get user by ID."""
    from app.models.database import User
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, user_create: UserCreate):
    """Create a new user."""
    from app.models.database import User
    
    db_user = User(
        email=user_create.email,
        oauth_provider=user_create.oauth_provider,
        oauth_subject=user_create.oauth_subject,
        display_name=user_create.display_name,
        profile_picture_url=user_create.profile_picture_url,
        subscription_tier=user_create.subscription_tier,
        subscription_expires_at=user_create.subscription_expires_at
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user


async def update_user(db: AsyncSession, user_id: UUID, user_update: UserUpdate):
    """Update an existing user."""
    from app.models.database import User
    
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    db_user = result.scalar_one_or_none()
    
    if not db_user:
        return None
    
    # Update fields that are provided
    if user_update.display_name is not None:
        db_user.display_name = user_update.display_name
    if user_update.profile_picture_url is not None:
        db_user.profile_picture_url = user_update.profile_picture_url
    if user_update.subscription_tier is not None:
        db_user.subscription_tier = user_update.subscription_tier
    if user_update.subscription_expires_at is not None:
        db_user.subscription_expires_at = user_update.subscription_expires_at
    
    await db.commit()
    await db.refresh(db_user)
    return db_user