"""
Tests for media retrieval endpoints.
"""
import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from uuid import UUID

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import User, FileType
from app.models.database import MediaFile as DBMediaFile
from app.api.v1.endpoints import media
from app.services.auth import get_current_active_user
from app.db.database import get_async_db


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        oauth_provider="google",
        oauth_subject="test_subject",
        subscription_tier="free",
        display_name="Test User"
    )


@pytest.fixture
def other_user():
    """Create another test user for authorization tests."""
    return User(
        id=uuid.uuid4(),
        email="other@example.com",
        oauth_provider="google",
        oauth_subject="other_subject",
        subscription_tier="free",
        display_name="Other User"
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
        created_at=datetime(2024, 1, 15, 10, 30, 0)
    )


@pytest.fixture
def test_voice_file(test_user):
    """Create a test voice file."""
    return DBMediaFile(
        id=uuid.uuid4(),
        user_id=test_user.id,
        file_type="voice",
        file_path="users/test-user/voice/2024/01/test-voice.mp3",
        original_filename="test-voice.mp3",
        file_size=512000,  # 512KB
        timestamp=datetime(2024, 1, 15, 14, 45, 0),
        longitude=-122.4194,
        latitude=37.7749,
        address="San Francisco, CA",
        names=["Meeting notes", "Important"],
        created_at=datetime(2024, 1, 15, 14, 45, 0)
    )


class TestGetMediaFile:
    """Test cases for GET /media/{media_id} endpoint."""
    
    def test_get_media_file_success_photo(self, test_user, test_media_file):
        """Test successful retrieval of a photo file (requirement 6.2)."""
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
            with patch('app.api.v1.endpoints.media.media_file_repository', mock_repository):
                with patch('app.api.v1.endpoints.media.file_storage_service', mock_storage_service):
                    client = TestClient(app)
                    response = client.get(f"/api/v1/media/{test_media_file.id}")
                    
                    # Verify response
                    assert response.status_code == status.HTTP_200_OK
                    assert response.content == test_file_content
                    assert response.headers["content-type"] == "image/jpeg"
                    assert response.headers["content-length"] == str(len(test_file_content))
                    assert 'filename="test-photo.jpg"' in response.headers["content-disposition"]
                    
                    # Verify repository was called
                    mock_repository.get_by_user.assert_called_once()
                    
                    # Verify storage service was called
                    mock_storage_service.download_file.assert_called_once_with(
                        s3_key=test_media_file.file_path
                    )
        finally:
            # Clean up dependency overrides
            app.dependency_overrides.clear()
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_get_media_file_success_voice(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_voice_file
    ):
        """Test successful retrieval of a voice file (requirement 6.2)."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_voice_file)
        
        # Mock file content
        test_file_content = b"fake_audio_data"
        mock_storage_service.download_file = AsyncMock(return_value=test_file_content)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_voice_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        assert response.content == test_file_content
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["content-length"] == str(len(test_file_content))
        assert 'filename="test-voice.mp3"' in response.headers["content-disposition"]
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_media_file_not_found(
        self, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test retrieval of non-existent media file (requirement 6.4)."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=None)
        mock_repository.get = AsyncMock(return_value=None)
        
        # Make request
        client = TestClient(app)
        non_existent_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/media/{non_existent_id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["detail"] == "Media file not found"
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_media_file_unauthorized_access(
        self, 
        mock_repository, 
        mock_get_user,
        test_user,
        other_user,
        test_media_file
    ):
        """Test unauthorized access to another user's media file (requirement 6.3)."""
        # Setup mocks - user tries to access other user's file
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=None)  # No file for current user
        
        # Mock that file exists but belongs to other user
        other_user_file = test_media_file
        other_user_file.user_id = other_user.id
        mock_repository.get = AsyncMock(return_value=other_user_file)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Access denied to this media file"
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_get_media_file_storage_not_found(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_media_file
    ):
        """Test when file exists in DB but not in storage."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)
        
        # Mock storage service to return 404
        from fastapi import HTTPException
        mock_storage_service.download_file = AsyncMock(
            side_effect=HTTPException(status_code=404, detail="File not found")
        )
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["detail"] == "Media file not found in storage"
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_get_media_file_storage_error(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_media_file
    ):
        """Test when storage service returns an error."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)
        
        # Mock storage service to return 500 error
        from fastapi import HTTPException
        mock_storage_service.download_file = AsyncMock(
            side_effect=HTTPException(status_code=500, detail="Storage error")
        )
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()["detail"] == "Failed to retrieve media file"
    
    def test_get_media_file_no_auth(self):
        """Test accessing media file without authentication."""
        client = TestClient(app)
        media_id = uuid.uuid4()
        response = client.get(f"/api/v1/media/{media_id}")
        
        # Should return 401 Unauthorized due to missing auth
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGetMediaFileInfo:
    """Test cases for GET /media/{media_id}/info endpoint."""
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_media_file_info_success(
        self, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_voice_file
    ):
        """Test successful retrieval of media file info."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_voice_file)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_voice_file.id}/info",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["id"] == str(test_voice_file.id)
        assert data["file_type"] == "voice"
        assert data["original_filename"] == "test-voice.mp3"
        assert data["file_size"] == 512000
        assert data["location"]["longitude"] == -122.4194
        assert data["location"]["latitude"] == 37.7749
        assert data["location"]["address"] == "San Francisco, CA"
        assert data["location"]["names"] == ["Meeting notes", "Important"]
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_media_file_info_no_location(
        self, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_media_file
    ):
        """Test media file info for file without location data."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}/info",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["location"] is None
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_media_file_info_not_found(
        self, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test info retrieval for non-existent media file."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=None)
        mock_repository.get = AsyncMock(return_value=None)
        
        # Make request
        client = TestClient(app)
        non_existent_id = uuid.uuid4()
        response = client.get(
            f"/api/v1/media/{non_existent_id}/info",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert response.json()["detail"] == "Media file not found"


class TestGetMediaDownloadUrl:
    """Test cases for GET /media/{media_id}/download-url endpoint."""
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_get_download_url_success(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_media_file
    ):
        """Test successful generation of download URL."""
        # Setup mocks
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=test_media_file)
        
        test_url = "https://s3.amazonaws.com/bucket/signed-url"
        mock_storage_service.generate_download_url = Mock(return_value=test_url)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}/download-url?expiration=7200",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["download_url"] == test_url
        assert data["expires_in"] == 7200
        assert data["file_info"]["id"] == str(test_media_file.id)
        assert data["file_info"]["file_type"] == "photo"
        assert data["file_info"]["original_filename"] == "test-photo.jpg"
        
        # Verify storage service was called with correct parameters
        mock_storage_service.generate_download_url.assert_called_once_with(
            s3_key=test_media_file.file_path,
            expiration=7200,
            file_name=test_media_file.original_filename
        )
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_download_url_invalid_expiration(
        self, 
        mock_repository, 
        mock_get_user,
        test_user,
        test_media_file
    ):
        """Test download URL generation with invalid expiration."""
        # Setup mocks
        mock_get_user.return_value = test_user
        
        # Make request with invalid expiration (too short)
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}/download-url?expiration=30",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Expiration must be between 60 seconds and 86400 seconds" in response.json()["detail"]
        
        # Make request with invalid expiration (too long)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}/download-url?expiration=90000",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Expiration must be between 60 seconds and 86400 seconds" in response.json()["detail"]
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    def test_get_download_url_unauthorized(
        self, 
        mock_repository, 
        mock_get_user,
        test_user,
        other_user,
        test_media_file
    ):
        """Test download URL generation for unauthorized access."""
        # Setup mocks - user tries to access other user's file
        mock_get_user.return_value = test_user
        mock_repository.get_by_user = AsyncMock(return_value=None)
        
        # Mock that file exists but belongs to other user
        other_user_file = test_media_file
        other_user_file.user_id = other_user.id
        mock_repository.get = AsyncMock(return_value=other_user_file)
        
        # Make request
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_media_file.id}/download-url",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        # Verify response
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Access denied to this media file"


class TestContentTypeDetection:
    """Test cases for content type detection based on file extensions."""
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_photo_content_types(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test content type detection for different photo formats."""
        mock_get_user.return_value = test_user
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")
        
        client = TestClient(app)
        
        # Test different photo extensions
        test_cases = [
            ("photo.jpg", "image/jpeg"),
            ("photo.jpeg", "image/jpeg"),
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
                timestamp=datetime.now()
            )
            
            mock_repository.get_by_user = AsyncMock(return_value=test_file)
            
            response = client.get(
                f"/api/v1/media/{test_file.id}",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == expected_content_type
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_voice_content_types(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test content type detection for different voice formats."""
        mock_get_user.return_value = test_user
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")
        
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
                timestamp=datetime.now()
            )
            
            mock_repository.get_by_user = AsyncMock(return_value=test_file)
            
            response = client.get(
                f"/api/v1/media/{test_file.id}",
                headers={"Authorization": "Bearer fake_token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == expected_content_type


class TestFilenameGeneration:
    """Test cases for filename generation when original filename is missing."""
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_generated_filename_photo(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test filename generation for photos without original filename."""
        mock_get_user.return_value = test_user
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")
        
        # Create test file without original filename
        test_file = DBMediaFile(
            id=uuid.uuid4(),
            user_id=test_user.id,
            file_type="photo",
            file_path="test/path",
            original_filename=None,
            file_size=1024,
            timestamp=datetime(2024, 1, 15, 10, 30, 45)
        )
        
        mock_repository.get_by_user = AsyncMock(return_value=test_file)
        
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert 'filename="photo_20240115_103045.jpg"' in response.headers["content-disposition"]
    
    @patch('app.api.v1.endpoints.media.get_current_active_user')
    @patch('app.api.v1.endpoints.media.media_file_repository')
    @patch('app.api.v1.endpoints.media.file_storage_service')
    def test_generated_filename_voice(
        self, 
        mock_storage_service, 
        mock_repository, 
        mock_get_user,
        test_user
    ):
        """Test filename generation for voice files without original filename."""
        mock_get_user.return_value = test_user
        mock_storage_service.download_file = AsyncMock(return_value=b"fake_data")
        
        # Create test file without original filename
        test_file = DBMediaFile(
            id=uuid.uuid4(),
            user_id=test_user.id,
            file_type="voice",
            file_path="test/path",
            original_filename=None,
            file_size=1024,
            timestamp=datetime(2024, 1, 15, 14, 45, 30)
        )
        
        mock_repository.get_by_user = AsyncMock(return_value=test_file)
        
        client = TestClient(app)
        response = client.get(
            f"/api/v1/media/{test_file.id}",
            headers={"Authorization": "Bearer fake_token"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert 'filename="voice_20240115_144530.mp3"' in response.headers["content-disposition"]