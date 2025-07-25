"""
Unit tests for repository base class and patterns.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError
from app.models.database import User
from app.models.schemas import UserCreate, UserUpdate
from app.repositories.base import BaseRepository


class MockRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """Mock repository for testing base functionality."""

    def __init__(self):
        super().__init__(User)

    async def create(self, db: AsyncSession, *, obj_in: UserCreate, **kwargs) -> User:
        """Create a new user."""
        try:
            user = User(**obj_in.model_dump(), **kwargs)
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise DatabaseError(f"Failed to create entity: {str(e)}")

    async def get(self, db: AsyncSession, id) -> Optional[User]:
        """Get a user by ID."""
        try:
            return await db.get(User, id)
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get entity by ID: {str(e)}")

    async def get_multi(
        self, db: AsyncSession, *, skip: int = 0, limit: int = 100, filters=None
    ) -> List[User]:
        """Get multiple users."""
        try:
            from sqlalchemy import select

            stmt = select(User).offset(skip).limit(limit)
            result = await db.execute(stmt)
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to get entities: {str(e)}")

    async def update(
        self, db: AsyncSession, *, db_obj: User, obj_in: UserUpdate
    ) -> User:
        """Update a user."""
        try:
            update_data = obj_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_obj, field, value)
            await db.commit()
            await db.refresh(db_obj)
            return db_obj
        except SQLAlchemyError as e:
            await db.rollback()
            raise DatabaseError(f"Failed to update entity: {str(e)}")

    async def delete(self, db: AsyncSession, *, id) -> Optional[User]:
        """Delete a user."""
        try:
            user = await self.get(db, id)
            if user:
                await db.delete(user)
                await db.commit()
            return user
        except SQLAlchemyError as e:
            await db.rollback()
            raise DatabaseError(f"Failed to delete entity: {str(e)}")

    async def count(self, db: AsyncSession, filters=None) -> int:
        """Count users."""
        try:
            from sqlalchemy import func, select

            stmt = select(func.count(User.id))
            result = await db.execute(stmt)
            return result.scalar()
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to count entities: {str(e)}")


class TestBaseRepository:
    """Test base repository functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self):
        """Create a mock repository instance."""
        return MockRepository()

    @pytest.fixture
    def sample_user_create(self):
        """Create a sample user create schema for testing."""
        return UserCreate(
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
            subscription_tier="free",
        )

    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
            subscription_tier="free",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    async def test_create_success(self, repository, mock_db, sample_user_create):
        """Test successful entity creation."""
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.create(mock_db, obj_in=sample_user_create)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert isinstance(result, User)
        assert result.email == "test@example.com"

    async def test_create_database_error(self, repository, mock_db, sample_user_create):
        """Test create with database error."""
        mock_db.add = Mock()
        mock_db.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = AsyncMock()

        with pytest.raises(DatabaseError) as exc_info:
            await repository.create(mock_db, obj_in=sample_user_create)

        assert "Failed to create entity" in str(exc_info.value)
        mock_db.rollback.assert_called_once()

    async def test_get_by_id_success(self, repository, mock_db, sample_user):
        """Test successful get by ID."""
        mock_db.get = AsyncMock(return_value=sample_user)

        result = await repository.get(mock_db, sample_user.id)

        mock_db.get.assert_called_once_with(User, sample_user.id)
        assert result == sample_user

    async def test_get_by_id_not_found(self, repository, mock_db):
        """Test get by ID when entity not found."""
        user_id = uuid4()
        mock_db.get = AsyncMock(return_value=None)

        result = await repository.get(mock_db, user_id)

        mock_db.get.assert_called_once_with(User, user_id)
        assert result is None

    async def test_get_by_id_database_error(self, repository, mock_db):
        """Test get by ID with database error."""
        user_id = uuid4()
        mock_db.get = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        with pytest.raises(DatabaseError) as exc_info:
            await repository.get(mock_db, user_id)

        assert "Failed to get entity by ID" in str(exc_info.value)

    async def test_update_success(self, repository, mock_db, sample_user):
        """Test successful entity update."""
        update_data = UserUpdate(display_name="Updated Name")
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.update(
            mock_db, db_obj=sample_user, obj_in=update_data
        )

        assert sample_user.display_name == "Updated Name"
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(sample_user)
        assert result == sample_user

    async def test_update_database_error(self, repository, mock_db, sample_user):
        """Test update with database error."""
        update_data = UserUpdate(display_name="Updated Name")
        mock_db.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = AsyncMock()

        with pytest.raises(DatabaseError) as exc_info:
            await repository.update(mock_db, db_obj=sample_user, obj_in=update_data)

        assert "Failed to update entity" in str(exc_info.value)
        mock_db.rollback.assert_called_once()

    async def test_delete_success(self, repository, mock_db, sample_user):
        """Test successful entity deletion."""
        mock_db.get = AsyncMock(return_value=sample_user)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        result = await repository.delete(mock_db, id=sample_user.id)

        mock_db.delete.assert_called_once_with(sample_user)
        mock_db.commit.assert_called_once()
        assert result == sample_user

    async def test_delete_not_found(self, repository, mock_db):
        """Test delete when entity not found."""
        user_id = uuid4()
        mock_db.get = AsyncMock(return_value=None)

        result = await repository.delete(mock_db, id=user_id)

        assert result is None

    async def test_delete_database_error(self, repository, mock_db, sample_user):
        """Test delete with database error."""
        mock_db.get = AsyncMock(return_value=sample_user)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = AsyncMock()

        with pytest.raises(DatabaseError) as exc_info:
            await repository.delete(mock_db, id=sample_user.id)

        assert "Failed to delete entity" in str(exc_info.value)
        mock_db.rollback.assert_called_once()

    async def test_get_multi_success(self, repository, mock_db):
        """Test successful get multiple entities."""
        users = [
            User(
                id=uuid4(),
                email="user1@example.com",
                oauth_provider="google",
                oauth_subject="1",
            ),
            User(
                id=uuid4(),
                email="user2@example.com",
                oauth_provider="google",
                oauth_subject="2",
            ),
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = users
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_multi(mock_db, skip=0, limit=10)

        assert len(result) == 2
        assert result == users
        mock_db.execute.assert_called_once()

    async def test_get_multi_database_error(self, repository, mock_db):
        """Test get multiple with database error."""
        mock_db.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        with pytest.raises(DatabaseError) as exc_info:
            await repository.get_multi(mock_db)

        assert "Failed to get entities" in str(exc_info.value)

    async def test_count_success(self, repository, mock_db):
        """Test successful count operation."""
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.count(mock_db)

        assert result == 5
        mock_db.execute.assert_called_once()

    async def test_count_database_error(self, repository, mock_db):
        """Test count with database error."""
        mock_db.execute = AsyncMock(side_effect=SQLAlchemyError("Database error"))

        with pytest.raises(DatabaseError) as exc_info:
            await repository.count(mock_db)

        assert "Failed to count entities" in str(exc_info.value)


class TestRepositoryPatterns:
    """Test common repository patterns and utilities."""

    def test_repository_inheritance(self):
        """Test that repository inheritance works correctly."""
        from app.repositories.location_visits import LocationVisitRepository
        from app.repositories.text_notes import TextNoteRepository

        # All repositories should inherit from BaseRepository
        assert issubclass(LocationVisitRepository, BaseRepository)
        assert issubclass(TextNoteRepository, BaseRepository)

    def test_model_type_safety(self):
        """Test that repository is type-safe with its model."""
        repository = MockRepository()

        # Repository should be typed to work with User model
        assert repository.model == User

        # Should be able to create instances with correct type
        user = User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
        )

        # This should work without type errors
        assert isinstance(user, repository.model)

    def test_abstract_base_class(self):
        """Test that BaseRepository is properly abstract."""
        # Should not be able to instantiate BaseRepository directly
        with pytest.raises(TypeError):
            BaseRepository(User)
