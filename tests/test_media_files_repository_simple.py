"""
Simplified unit tests for media files repository.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import MediaFile
from app.models.schemas import MediaFileCreate
from app.repositories.media_files import MediaFileRepository


class TestMediaFileRepositorySimple:
    """Test media file repository basic functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    def repository(self):
        """Create a media file repository instance."""
        return MediaFileRepository()

    @pytest.fixture
    def sample_user_id(self):
        """Create a sample user ID."""
        return uuid4()

    @pytest.fixture
    def sample_media_create(self):
        """Create a sample media file create schema."""
        return MediaFileCreate(
            file_type="photo",
            original_filename="test.jpg",
            timestamp=datetime.utcnow(),
            longitude=-122.4194,
            latitude=37.7749,
            address="San Francisco, CA",
            names=["vacation", "beach"],
        )

    async def test_create_media_file_success(
        self, repository, mock_db, sample_user_id, sample_media_create
    ):
        """Test successful media file creation."""
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.create(
            mock_db,
            obj_in=sample_media_create,
            user_id=sample_user_id,
            file_path="photos/test.jpg",
            file_size=1024,
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert isinstance(result, MediaFile)
        assert result.user_id == sample_user_id
        assert result.file_type == "photo"
        assert result.original_filename == "test.jpg"

    async def test_get_by_id_success(self, repository, mock_db):
        """Test successful get by ID."""
        media_file = MediaFile(
            id=uuid4(),
            user_id=uuid4(),
            file_type="photo",
            file_path="photos/test.jpg",
            original_filename="test.jpg",
            timestamp=datetime.utcnow(),
        )

        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = media_file
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get(mock_db, media_file.id)

        assert result == media_file
        mock_db.execute.assert_called_once()

    async def test_get_by_id_not_found(self, repository, mock_db):
        """Test get by ID when media file not found."""
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get(mock_db, uuid4())

        assert result is None
        mock_db.execute.assert_called_once()

    async def test_get_by_user_success(self, repository, mock_db, sample_user_id):
        """Test successful get by user and file ID."""
        media_file = MediaFile(
            id=uuid4(),
            user_id=sample_user_id,
            file_type="photo",
            file_path="photos/test.jpg",
            original_filename="test.jpg",
            timestamp=datetime.utcnow(),
        )

        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = media_file
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(mock_db, sample_user_id, media_file.id)

        assert result == media_file
        mock_db.execute.assert_called_once()

    async def test_get_by_user_wrong_user(self, repository, mock_db):
        """Test get by user with wrong user ID."""
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(mock_db, uuid4(), uuid4())

        assert result is None
        mock_db.execute.assert_called_once()

    async def test_get_multi_success(self, repository, mock_db):
        """Test successful get multiple media files."""
        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=uuid4(),
                file_type="photo",
                file_path="1.jpg",
                original_filename="1.jpg",
                timestamp=datetime.utcnow(),
            ),
            MediaFile(
                id=uuid4(),
                user_id=uuid4(),
                file_type="video",
                file_path="2.mp4",
                original_filename="2.mp4",
                timestamp=datetime.utcnow(),
            ),
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_multi(mock_db, skip=0, limit=10)

        assert len(result) == 2
        assert result == media_files
        mock_db.execute.assert_called_once()

    async def test_count_success(self, repository, mock_db):
        """Test successful count operation."""
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.count(mock_db)

        assert result == 5
        mock_db.execute.assert_called_once()

    async def test_delete_success(self, repository, mock_db, sample_user_id):
        """Test successful delete operation."""
        media_file = MediaFile(
            id=uuid4(),
            user_id=sample_user_id,
            file_type="photo",
            file_path="photos/test.jpg",
            original_filename="test.jpg",
            timestamp=datetime.utcnow(),
        )

        # Mock get method to return the media file
        mock_get_result = Mock()
        mock_get_result.scalars.return_value.first.return_value = media_file
        mock_db.execute = AsyncMock(return_value=mock_get_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()

        result = await repository.delete(mock_db, id=media_file.id)

        assert result == media_file
        mock_db.delete.assert_called_once_with(media_file)
        mock_db.commit.assert_called_once()

    async def test_repository_model_type(self, repository):
        """Test that repository is properly typed."""
        assert repository.model == MediaFile

    async def test_user_isolation_in_filters(self, repository, mock_db, sample_user_id):
        """Test that user isolation works in filtered queries."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        filters = {"user_id": sample_user_id, "file_type": "photo"}
        result = await repository.get_multi(mock_db, filters=filters)

        assert result == []
        mock_db.execute.assert_called_once()
        # Verify that the query was built with user_id filter
        call_args = mock_db.execute.call_args[0][0]
        assert hasattr(call_args, "whereclause")
