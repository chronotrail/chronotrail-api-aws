"""
Simple tests for media retrieval endpoints focusing on core functionality.
"""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from app.db.database import get_async_db
from app.main import app
from app.models.database import MediaFile as DBMediaFile
from app.models.schemas import FileType, User
from app.services.auth import get_current_active_user


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        oauth_provider="google",
        oauth_subject="test_subject",
        subscription_tier="free",
        display_name="Test User",
    )


@pytest.fixture
def test_media_file(test_user):
    """Create a test media file."""
    return DBMediaFile(
        id=uuid.uuid4(),
        user_id=test_user.id,
        file_type="photo",
        file_path="users/test-user/photos/2024/01/test-photo.jpg",
        original_filename="test-photo.jpg",
        file_size=1024000,  # 1MB
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        longitude=None,
        latitude=None,
        address=None,
        names=None,
        created_at=datetime(2024, 1, 15, 10, 30, 0),
    )


class TestMediaEndpointsCore:
    """Core tests for media endpoints functionality."""

    def test_get_media_file_success(self, test_user, test_media_file):
        """Test successful media file retrieval with proper mocking."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        # Mock repository and storage service
        mock_repository = Mock()
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)

        mock_storage_service = Mock()
        test_file_content = b"fake_image_data"
        mock_storage_service.download_file = AsyncMock(return_value=test_file_content)

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                with patch(
                    "app.api.v1.endpoints.media.file_storage_service",
                    mock_storage_service,
                ):
                    client = TestClient(app)
                    response = client.get(f"/api/v1/media/{test_media_file.id}")

                    # Verify response
                    assert response.status_code == status.HTTP_200_OK
                    assert response.content == test_file_content
                    assert response.headers["content-type"] == "image/jpeg"
                    assert (
                        'filename="test-photo.jpg"'
                        in response.headers["content-disposition"]
                    )

                    # Verify repository was called
                    mock_repository.get_by_user.assert_called_once()

                    # Verify storage service was called
                    mock_storage_service.download_file.assert_called_once_with(
                        s3_key=test_media_file.file_path
                    )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_get_media_file_not_found(self, test_user):
        """Test media file not found scenario."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        # Mock repository to return None (file not found)
        mock_repository = Mock()
        mock_repository.get_by_user = AsyncMock(return_value=None)
        mock_repository.get = AsyncMock(return_value=None)

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                client = TestClient(app)
                non_existent_id = uuid.uuid4()
                response = client.get(f"/api/v1/media/{non_existent_id}")

                # Verify response
                assert response.status_code == status.HTTP_404_NOT_FOUND
                assert response.json()["message"] == "Media file not found"
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_get_media_file_unauthorized(self, test_user, test_media_file):
        """Test unauthorized access to media file."""
        # Create another user
        other_user = User(
            id=uuid.uuid4(),
            email="other@example.com",
            oauth_provider="google",
            oauth_subject="other_subject",
            subscription_tier="free",
            display_name="Other User",
        )

        # Mock dependencies
        async def mock_get_user():
            return test_user  # Current user

        async def mock_get_db():
            return Mock()

        # Mock repository - file doesn't belong to current user
        mock_repository = Mock()
        mock_repository.get_by_user = AsyncMock(
            return_value=None
        )  # No file for current user

        # File exists but belongs to other user
        other_user_file = test_media_file
        other_user_file.user_id = other_user.id
        mock_repository.get = AsyncMock(return_value=other_user_file)

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                client = TestClient(app)
                response = client.get(f"/api/v1/media/{test_media_file.id}")

                # Verify response
                assert response.status_code == status.HTTP_403_FORBIDDEN
                assert response.json()["message"] == "Access denied to this media file"
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_get_media_file_info_success(self, test_user, test_media_file):
        """Test successful media file info retrieval."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        # Mock repository
        mock_repository = Mock()
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                client = TestClient(app)
                response = client.get(f"/api/v1/media/{test_media_file.id}/info")

                # Verify response
                assert response.status_code == status.HTTP_200_OK
                data = response.json()

                assert data["id"] == str(test_media_file.id)
                assert data["file_type"] == "photo"
                assert data["original_filename"] == "test-photo.jpg"
                assert data["file_size"] == 1024000
                assert data["location"] is None  # No location data in test file
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_get_download_url_success(self, test_user, test_media_file):
        """Test successful download URL generation."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        # Mock repository and storage service
        mock_repository = Mock()
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)

        mock_storage_service = Mock()
        test_url = "https://s3.amazonaws.com/bucket/signed-url"
        mock_storage_service.generate_download_url = Mock(return_value=test_url)

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                with patch(
                    "app.api.v1.endpoints.media.file_storage_service",
                    mock_storage_service,
                ):
                    client = TestClient(app)
                    response = client.get(
                        f"/api/v1/media/{test_media_file.id}/download-url"
                    )

                    # Verify response
                    assert response.status_code == status.HTTP_200_OK
                    data = response.json()

                    assert data["download_url"] == test_url
                    assert data["expires_in"] == 3600  # Default expiration
                    assert data["file_info"]["id"] == str(test_media_file.id)
                    assert data["file_info"]["file_type"] == "photo"

                    # Verify storage service was called
                    mock_storage_service.generate_download_url.assert_called_once_with(
                        s3_key=test_media_file.file_path,
                        expiration=3600,
                        file_name=test_media_file.original_filename,
                    )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_get_download_url_invalid_expiration(self, test_user, test_media_file):
        """Test download URL generation with invalid expiration."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            client = TestClient(app)

            # Test expiration too short
            response = client.get(
                f"/api/v1/media/{test_media_file.id}/download-url?expiration=30"
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert (
                "Expiration must be between 60 seconds and 86400 seconds"
                in response.json()["message"]
            )

            # Test expiration too long
            response = client.get(
                f"/api/v1/media/{test_media_file.id}/download-url?expiration=90000"
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert (
                "Expiration must be between 60 seconds and 86400 seconds"
                in response.json()["message"]
            )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()


class TestContentTypeDetection:
    """Test content type detection for different file formats."""

    def test_photo_content_types(self, test_user):
        """Test content type detection for different photo formats."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        mock_repository = Mock()
        mock_storage_service = Mock()
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                with patch(
                    "app.api.v1.endpoints.media.file_storage_service",
                    mock_storage_service,
                ):
                    client = TestClient(app)

                    # Test different photo extensions
                    test_cases = [
                        ("photo.jpg", "image/jpeg"),
                        ("photo.png", "image/png"),
                        ("photo.heic", "image/heic"),
                        ("photo.webp", "image/webp"),
                        (None, "image/jpeg"),  # Default for photos
                    ]

                    for filename, expected_content_type in test_cases:
                        # Create test file with specific extension
                        test_file = DBMediaFile(
                            id=uuid.uuid4(),
                            user_id=test_user.id,
                            file_type="photo",
                            file_path="test/path",
                            original_filename=filename,
                            file_size=1024,
                            timestamp=datetime.now(),
                        )

                        mock_repository.get_by_user = AsyncMock(return_value=test_file)

                        response = client.get(f"/api/v1/media/{test_file.id}")

                        assert response.status_code == status.HTTP_200_OK
                        assert response.headers["content-type"] == expected_content_type
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()

    def test_voice_content_types(self, test_user):
        """Test content type detection for different voice formats."""

        # Mock dependencies
        async def mock_get_user():
            return test_user

        async def mock_get_db():
            return Mock()

        mock_repository = Mock()
        mock_storage_service = Mock()
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")

        # Override dependencies
        app.dependency_overrides[get_current_active_user] = mock_get_user
        app.dependency_overrides[get_async_db] = mock_get_db

        try:
            with patch(
                "app.api.v1.endpoints.media.media_file_repository", mock_repository
            ):
                with patch(
                    "app.api.v1.endpoints.media.file_storage_service",
                    mock_storage_service,
                ):
                    client = TestClient(app)

                    # Test different voice extensions
                    test_cases = [
                        ("voice.mp3", "audio/mpeg"),
                        ("voice.wav", "audio/wav"),
                        ("voice.m4a", "audio/mp4"),
                        ("voice.aac", "audio/aac"),
                        ("voice.ogg", "audio/ogg"),
                        (None, "audio/mpeg"),  # Default for voice
                    ]

                    for filename, expected_content_type in test_cases:
                        # Create test file with specific extension
                        test_file = DBMediaFile(
                            id=uuid.uuid4(),
                            user_id=test_user.id,
                            file_type="voice",
                            file_path="test/path",
                            original_filename=filename,
                            file_size=1024,
                            timestamp=datetime.now(),
                        )

                        mock_repository.get_by_user = AsyncMock(return_value=test_file)

                        response = client.get(f"/api/v1/media/{test_file.id}")

                        assert response.status_code == status.HTTP_200_OK
                        assert response.headers["content-type"] == expected_content_type
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()
