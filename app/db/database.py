"""
Database configuration and connection management
"""
from typing import Optional
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.models.schemas import UserCreate, UserUpdate, User

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

# Create Base class for models
Base = declarative_base()


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


async def get_user_by_oauth(db: AsyncSession, oauth_provider: str, oauth_subject: str) -> Optional[User]:
    """
    Get a user by OAuth provider and subject.
    
    Args:
        db: Database session
        oauth_provider: OAuth provider name ('google' or 'apple')
        oauth_subject: OAuth subject ID
        
    Returns:
        User object if found, None otherwise
    """
    from app.models.database import User as DBUser
    
    stmt = select(DBUser).where(
        DBUser.oauth_provider == oauth_provider,
        DBUser.oauth_subject == oauth_subject
    )
    
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    if user:
        return User.model_validate(user)
    
    return None


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> Optional[User]:
    """
    Get a user by ID.
    
    Args:
        db: Database session
        user_id: User ID
        
    Returns:
        User object if found, None otherwise
    """
    from app.models.database import User as DBUser
    
    stmt = select(DBUser).where(DBUser.id == user_id)
    
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    if user:
        return User.model_validate(user)
    
    return None


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Get a user by email.
    
    Args:
        db: Database session
        email: User email
        
    Returns:
        User object if found, None otherwise
    """
    from app.models.database import User as DBUser
    
    stmt = select(DBUser).where(DBUser.email == email)
    
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    if user:
        return User.model_validate(user)
    
    return None


async def create_user(db: AsyncSession, user_create: UserCreate) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        user_create: User creation data
        
    Returns:
        Created user object
    """
    from app.models.database import User as DBUser
    
    db_user = DBUser(
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
    
    return User.model_validate(db_user)


async def update_user(db: AsyncSession, user_id: UUID, user_update: UserUpdate) -> Optional[User]:
    """
    Update a user.
    
    Args:
        db: Database session
        user_id: User ID
        user_update: User update data
        
    Returns:
        Updated user object if found, None otherwise
    """
    from app.models.database import User as DBUser
    
    stmt = select(DBUser).where(DBUser.id == user_id)
    
    result = await db.execute(stmt)
    db_user = result.scalars().first()
    
    if not db_user:
        return None
    
    # Update fields if provided
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
    
    return User.model_validate(db_user)