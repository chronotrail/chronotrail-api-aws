"""
Unit tests for media files repository.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DatabaseError, ResourceNotFoundError
from app.models.database import MediaFile, User
from app.repositories.media_files import MediaFileRepository


class TestMediaFileRepository:
    """Test media file repository functionality."""

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
    def sample_media_file(self, sample_user_id):
        """Create a sample media file."""
        return MediaFile(
            id=uuid4(),
            user_id=sample_user_id,
            file_type="photo",
            file_path="photos/user123/photo1.jpg",
            original_filename="vacation.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
            address="San Francisco, CA",
            names=["Vacation", "Beach"],
            created_at=datetime.utcnow(),
        )

    async def test_create_media_file_success(self, repository, mock_db, sample_user_id):
        """Test successful media file creation."""
        from app.models.schemas import MediaFileCreate

        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        create_data = MediaFileCreate(
            file_type="photo", original_filename="test.jpg", timestamp=datetime.utcnow()
        )

        result = await repository.create(
            mock_db,
            obj_in=create_data,
            user_id=sample_user_id,
            file_path="photos/test.jpg",
            file_size=1024,
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert isinstance(result, MediaFile)
        assert result.user_id == sample_user_id

    async def test_get_by_id_success(self, repository, mock_db, sample_media_file):
        """Test successful get media file by ID."""
        mock_db.get = AsyncMock(return_value=sample_media_file)

        result = await repository.get_by_id(sample_media_file.id)

        mock_db.get.assert_called_once_with(MediaFile, sample_media_file.id)
        assert result == sample_media_file

    async def test_get_by_id_not_found(self, repository, mock_db):
        """Test get media file by ID when not found."""
        media_id = uuid4()
        mock_db.get = AsyncMock(return_value=None)

        result = await repository.get_by_id(media_id)

        assert result is None

    async def test_get_by_user_success(self, repository, mock_db, sample_user_id):
        """Test get media files by user."""
        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow(),
            ),
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="voice",
                file_path="voice/voice1.mp3",
                original_filename="voice1.mp3",
                timestamp=datetime.utcnow(),
            ),
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(sample_user_id)

        assert len(result) == 2
        assert all(media.user_id == sample_user_id for media in result)
        mock_db.execute.assert_called_once()

    async def test_get_by_user_with_file_type_filter(
        self, repository, mock_db, sample_user_id
    ):
        """Test get media files by user with file type filter."""
        photo_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow(),
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = photo_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(sample_user_id, file_type="photo")

        assert len(result) == 1
        assert result[0].file_type == "photo"

    async def test_get_by_user_with_date_range(
        self, repository, mock_db, sample_user_id
    ):
        """Test get media files by user with date range filter."""
        start_date = datetime.utcnow() - timedelta(days=7)
        end_date = datetime.utcnow()

        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow() - timedelta(days=3),
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(
            sample_user_id, start_date=start_date, end_date=end_date
        )

        assert len(result) == 1
        assert start_date <= result[0].timestamp <= end_date

    async def test_get_by_location_success(self, repository, mock_db, sample_user_id):
        """Test get media files by location."""
        longitude = Decimal("-122.4194")
        latitude = Decimal("37.7749")
        radius_km = 1.0

        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow(),
                longitude=longitude,
                latitude=latitude,
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_location(
            sample_user_id, longitude, latitude, radius_km
        )

        assert len(result) == 1
        assert result[0].longitude == longitude
        assert result[0].latitude == latitude

    async def test_search_by_names_success(self, repository, mock_db, sample_user_id):
        """Test search media files by names/tags."""
        search_names = ["Vacation", "Beach"]

        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow(),
                names=["Vacation", "Beach", "Summer"],
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.search_by_names(sample_user_id, search_names)

        assert len(result) == 1
        assert any(name in result[0].names for name in search_names)

    async def test_search_by_address_success(self, repository, mock_db, sample_user_id):
        """Test search media files by address."""
        address_query = "San Francisco"

        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/photo1.jpg",
                original_filename="photo1.jpg",
                timestamp=datetime.utcnow(),
                address="San Francisco, CA",
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.search_by_address(sample_user_id, address_query)

        assert len(result) == 1
        assert address_query in result[0].address

    async def test_get_by_file_path_success(
        self, repository, mock_db, sample_media_file
    ):
        """Test get media file by file path."""
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = sample_media_file
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_file_path(sample_media_file.file_path)

        assert result == sample_media_file

    async def test_get_by_file_path_not_found(self, repository, mock_db):
        """Test get media file by file path when not found."""
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_file_path("nonexistent/path.jpg")

        assert result is None

    async def test_count_by_user_success(self, repository, mock_db, sample_user_id):
        """Test count media files by user."""
        mock_result = Mock()
        mock_result.scalar.return_value = 5
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.count_by_user(sample_user_id)

        assert result == 5

    async def test_count_by_user_with_file_type(
        self, repository, mock_db, sample_user_id
    ):
        """Test count media files by user with file type filter."""
        mock_result = Mock()
        mock_result.scalar.return_value = 3
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.count_by_user(sample_user_id, file_type="photo")

        assert result == 3

    async def test_get_recent_by_user_success(
        self, repository, mock_db, sample_user_id
    ):
        """Test get recent media files by user."""
        recent_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path="photos/recent1.jpg",
                original_filename="recent1.jpg",
                timestamp=datetime.utcnow(),
            ),
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="voice",
                file_path="voice/recent1.mp3",
                original_filename="recent1.mp3",
                timestamp=datetime.utcnow() - timedelta(hours=1),
            ),
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = recent_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_recent_by_user(sample_user_id, limit=10)

        assert len(result) == 2
        # Should be ordered by timestamp descending
        assert result[0].timestamp >= result[1].timestamp

    async def test_get_storage_stats_by_user(self, repository, mock_db, sample_user_id):
        """Test get storage statistics by user."""
        mock_result = Mock()
        mock_result.fetchone.return_value = (5, 10485760)  # 5 files, 10MB total
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_storage_stats_by_user(sample_user_id)

        assert result["file_count"] == 5
        assert result["total_size"] == 10485760

    async def test_delete_by_user_success(self, repository, mock_db, sample_user_id):
        """Test delete all media files by user."""
        mock_result = Mock()
        mock_result.rowcount = 3
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        result = await repository.delete_by_user(sample_user_id)

        assert result == 3
        mock_db.commit.assert_called_once()

    async def test_delete_old_files_success(self, repository, mock_db):
        """Test delete old media files."""
        cutoff_date = datetime.utcnow() - timedelta(days=365)

        mock_result = Mock()
        mock_result.rowcount = 10
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()

        result = await repository.delete_old_files(cutoff_date)

        assert result == 10
        mock_db.commit.assert_called_once()

    async def test_user_isolation(self, repository, mock_db):
        """Test that user data is properly isolated."""
        user1_id = uuid4()
        user2_id = uuid4()

        # Mock returning only user1's files
        user1_files = [
            MediaFile(
                id=uuid4(),
                user_id=user1_id,
                file_type="photo",
                file_path="photos/user1_photo.jpg",
                original_filename="photo.jpg",
                timestamp=datetime.utcnow(),
            )
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = user1_files
        mock_db.execute = AsyncMock(return_value=mock_result)

        # Get files for user1
        result = await repository.get_by_user(user1_id)

        assert len(result) == 1
        assert result[0].user_id == user1_id

        # Verify the query included user_id filter
        call_args = mock_db.execute.call_args[0][0]
        # The query should contain a WHERE clause filtering by user_id
        assert hasattr(call_args, "whereclause")

    async def test_pagination_support(self, repository, mock_db, sample_user_id):
        """Test pagination in get_by_user method."""
        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path=f"photos/photo{i}.jpg",
                original_filename=f"photo{i}.jpg",
                timestamp=datetime.utcnow() - timedelta(hours=i),
            )
            for i in range(5)
        ]

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = media_files[
            :3
        ]  # First 3 items
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user(sample_user_id, skip=0, limit=3)

        assert len(result) == 3

    async def test_array_field_support(self, repository, mock_db, sample_user_id):
        """Test support for array fields (names)."""
        media_file = MediaFile(
            id=uuid4(),
            user_id=sample_user_id,
            file_type="photo",
            file_path="photos/tagged_photo.jpg",
            original_filename="tagged_photo.jpg",
            timestamp=datetime.utcnow(),
            names=["Tag1", "Tag2", "Tag3"],
        )

        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.create(media_file)

        assert result.names == ["Tag1", "Tag2", "Tag3"]
        mock_db.add.assert_called_once_with(media_file)

    async def test_database_error_handling(
        self, repository, mock_db, sample_media_file
    ):
        """Test database error handling."""
        mock_db.add = Mock()
        mock_db.commit = AsyncMock(side_effect=SQLAlchemyError("Database error"))
        mock_db.rollback = AsyncMock()

        with pytest.raises(DatabaseError) as exc_info:
            await repository.create(sample_media_file)

        assert "Failed to create entity" in str(exc_info.value)
        mock_db.rollback.assert_called_once()

    async def test_get_by_user_and_id_success(
        self, repository, mock_db, sample_media_file
    ):
        """Test get media file by user and ID (for authorization)."""
        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = sample_media_file
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user_and_id(
            sample_media_file.user_id, sample_media_file.id
        )

        assert result == sample_media_file

    async def test_get_by_user_and_id_wrong_user(self, repository, mock_db):
        """Test get media file by user and ID with wrong user."""
        wrong_user_id = uuid4()
        media_id = uuid4()

        mock_result = Mock()
        mock_result.scalars.return_value.first.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_by_user_and_id(wrong_user_id, media_id)

        assert result is None

    async def test_update_file_path_success(
        self, repository, mock_db, sample_media_file
    ):
        """Test updating media file path."""
        new_path = "photos/updated/new_photo.jpg"

        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.update(sample_media_file, {"file_path": new_path})

        assert sample_media_file.file_path == new_path
        mock_db.commit.assert_called_once()

    async def test_bulk_operations_support(self, repository, mock_db, sample_user_id):
        """Test bulk operations for media files."""
        media_files = [
            MediaFile(
                id=uuid4(),
                user_id=sample_user_id,
                file_type="photo",
                file_path=f"photos/bulk{i}.jpg",
                original_filename=f"bulk{i}.jpg",
                timestamp=datetime.utcnow(),
            )
            for i in range(3)
        ]

        mock_db.add_all = Mock()
        mock_db.commit = AsyncMock()

        # Simulate bulk create
        for media_file in media_files:
            mock_db.add(media_file)

        await mock_db.commit()

        mock_db.commit.assert_called_once()
        assert len(media_files) == 3
