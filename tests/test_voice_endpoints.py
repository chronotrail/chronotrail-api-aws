"""
Tests for voice upload endpoints.
"""

import io
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.main import app
from app.models.database import MediaFile as DBMediaFile
from app.models.schemas import FileType, SubscriptionTier, User
from app.services.auth import get_current_active_user


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=uuid4(),
        email="test@example.com",
        oauth_provider="google",
        oauth_subject="test_subject",
        display_name="Test User",
        subscription_tier=SubscriptionTier.FREE,
        subscription_expires_at=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def premium_user():
    """Create a premium test user."""
    return User(
        id=uuid4(),
        email="premium@example.com",
        oauth_provider="google",
        oauth_subject="premium_subject",
        display_name="Premium User",
        subscription_tier=SubscriptionTier.PREMIUM,
        subscription_expires_at=datetime.utcnow() + timedelta(days=30),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


@pytest.fixture
def test_audio_file():
    """Create a test audio file."""
    # Create a simple test audio file (mock WAV data)
    audio_data = b"RIFF" + b"\x00" * 44 + b"fake audio data" * 100
    audio_bytes = io.BytesIO(audio_data)

    return ("test_audio.wav", audio_bytes, "audio/wav")


@pytest.fixture
def large_audio_file():
    """Create a large test audio file (>5MB)."""
    # Create a large test audio file
    audio_data = b"RIFF" + b"\x00" * 44 + b"fake audio data" * 100000
    audio_bytes = io.BytesIO(audio_data)

    return ("large_audio.wav", audio_bytes, "audio/wav")


class TestVoiceUpload:
    """Test cases for voice upload endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)

    def test_upload_voice_success(self, client, mock_db, test_user, test_audio_file):
        """Test successful voice upload."""
        # Mock database creation
        db_media_file = DBMediaFile(
            id=uuid4(),
            user_id=test_user.id,
            file_type="voice",
            file_path="test/path/audio.wav",
            original_filename="test_audio.wav",
            file_size=1024000,
            timestamp=datetime.utcnow(),
        )

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
            patch(
                "app.repositories.media_files.media_file_repository.create",
                return_value=db_media_file,
            ) as mock_create,
            patch(
                "app.aws.transcription.transcription_service.process_voice"
            ) as mock_transcription,
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_123",
            ) as mock_embedding,
            patch(
                "app.services.usage.UsageService.increment_content_usage",
                return_value=None,
            ) as mock_increment,
        ):

            # Mock storage service
            mock_storage.return_value = {
                "file_id": str(uuid4()),
                "s3_key": "test/path/audio.wav",
                "file_size": 1024000,
                "timestamp": datetime.utcnow(),
            }

            # Mock transcription processing
            from app.models.schemas import ContentType, ProcessedVoice

            mock_transcription_result = ProcessedVoice(
                original_id=db_media_file.id,
                processed_text="This is a test voice recording with some spoken content",
                content_type=ContentType.VOICE_TRANSCRIPT,
                processing_status="completed",
                transcript="This is a test voice recording with some spoken content",
                confidence_score=0.95,
                duration_seconds=30.5,
                error_details=None,
            )
            mock_transcription.return_value = mock_transcription_result

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice",
                files={"file": (filename, file_content, content_type)},
                data={
                    "timestamp": "2024-01-01T12:00:00Z",
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA",
                    "names": "Golden Gate Park,Park",
                },
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["file_type"] == "voice"
        assert data["processing_status"] == "completed"
        assert "This is a test voice recording" in data["message"]
        assert data["file_size"] == 1024000

        # Verify mocks were called
        mock_storage.assert_called_once()
        mock_create.assert_called_once()
        mock_transcription.assert_called_once()
        mock_embedding.assert_called_once()
        mock_increment.assert_called_once()

    def test_upload_voice_file_too_large(
        self, client, mock_db, test_user, large_audio_file
    ):
        """Test voice upload with file too large."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.api.v1.endpoints.voice.validate_file_size"
            ) as mock_validate_file_size,
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
        ):

            # Mock file size validation to raise exception
            from fastapi import HTTPException

            mock_validate_file_size.side_effect = HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large",
            )

            filename, file_content, content_type = large_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "File too large" in response.json()["detail"]

    def test_upload_voice_invalid_timestamp(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with invalid timestamp."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
        ):

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice",
                files={"file": (filename, file_content, content_type)},
                data={"timestamp": "invalid-timestamp"},
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid timestamp format" in response.json()["detail"]

    def test_upload_voice_future_timestamp(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with future timestamp."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
        ):

            filename, file_content, content_type = test_audio_file
            future_time = (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z"

            response = client.post(
                "/api/v1/voice",
                files={"file": (filename, file_content, content_type)},
                data={"timestamp": future_time},
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Timestamp cannot be in the future" in response.json()["detail"]

    def test_upload_voice_invalid_location(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with invalid location data."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
        ):

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice",
                files={"file": (filename, file_content, content_type)},
                data={"longitude": -122.4194},
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert (
            "Both longitude and latitude must be provided together"
            in response.json()["detail"]
        )

    def test_upload_voice_too_many_names(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with too many location names."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
        ):

            filename, file_content, content_type = test_audio_file
            many_names = ",".join(
                [f"name{i}" for i in range(15)]
            )  # 15 names (limit is 10)

            response = client.post(
                "/api/v1/voice",
                files={"file": (filename, file_content, content_type)},
                data={"names": many_names},
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Maximum 10 location names/tags allowed" in response.json()["detail"]

    def test_upload_voice_transcription_failure(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with transcription failure."""
        # Mock database creation
        db_media_file = DBMediaFile(
            id=uuid4(),
            user_id=test_user.id,
            file_type="voice",
            file_path="test/path/audio.wav",
            original_filename="test_audio.wav",
            file_size=1024000,
            timestamp=datetime.utcnow(),
        )

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
            patch(
                "app.repositories.media_files.media_file_repository.create",
                return_value=db_media_file,
            ) as mock_create,
            patch(
                "app.aws.transcription.transcription_service.process_voice"
            ) as mock_transcription,
            patch(
                "app.services.usage.UsageService.increment_content_usage",
                return_value=None,
            ) as mock_increment,
        ):

            # Mock storage service
            mock_storage.return_value = {
                "file_id": str(uuid4()),
                "s3_key": "test/path/audio.wav",
                "file_size": 1024000,
                "timestamp": datetime.utcnow(),
            }

            # Mock transcription failure
            mock_transcription.side_effect = Exception("Transcription failed")

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["processing_status"] == "failed"
        assert "Voice processing failed" in data["message"]
        assert data["error_details"] is not None

        # Verify storage and database operations still completed
        mock_storage.assert_called_once()
        mock_create.assert_called_once()
        mock_increment.assert_called_once()

    def test_upload_voice_storage_failure(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with storage failure."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
        ):

            # Mock storage service failure
            mock_storage.side_effect = Exception("Storage failed")

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to upload voice file" in response.json()["detail"]

    def test_upload_voice_database_failure(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with database failure and cleanup."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
            patch(
                "app.repositories.media_files.media_file_repository.create"
            ) as mock_create,
            patch("app.aws.storage.file_storage_service.delete_file") as mock_delete,
        ):

            # Mock storage service success
            mock_storage.return_value = {
                "file_id": str(uuid4()),
                "s3_key": "test/path/audio.wav",
                "file_size": 1024000,
                "timestamp": datetime.utcnow(),
            }

            # Mock database failure
            mock_create.side_effect = Exception("Database failed")

            # Mock cleanup
            mock_delete.return_value = True

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to create media file record" in response.json()["detail"]

        # Verify cleanup was attempted
        mock_delete.assert_called_once()

    def test_upload_voice_no_meaningful_content(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload with no meaningful content transcribed."""
        # Mock database creation
        db_media_file = DBMediaFile(
            id=uuid4(),
            user_id=test_user.id,
            file_type="voice",
            file_path="test/path/audio.wav",
            original_filename="test_audio.wav",
            file_size=1024000,
            timestamp=datetime.utcnow(),
        )

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit",
                return_value=None,
            ),
            patch(
                "app.middleware.usage_validation.validate_file_size", return_value=None
            ),
            patch("app.aws.storage.file_storage_service.upload_file") as mock_storage,
            patch(
                "app.repositories.media_files.media_file_repository.create",
                return_value=db_media_file,
            ) as mock_create,
            patch(
                "app.aws.transcription.transcription_service.process_voice"
            ) as mock_transcription,
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text"
            ) as mock_embedding,
            patch(
                "app.services.usage.UsageService.increment_content_usage",
                return_value=None,
            ) as mock_increment,
        ):

            # Mock storage service
            mock_storage.return_value = {
                "file_id": str(uuid4()),
                "s3_key": "test/path/audio.wav",
                "file_size": 1024000,
                "timestamp": datetime.utcnow(),
            }

            # Mock transcription with no meaningful content
            from app.models.schemas import ContentType, ProcessedVoice

            mock_transcription_result = ProcessedVoice(
                original_id=db_media_file.id,
                processed_text="Audio quality too poor for transcription",
                content_type=ContentType.VOICE_TRANSCRIPT,
                processing_status="completed",
                transcript="Audio quality too poor for transcription",
                confidence_score=0.1,
                duration_seconds=5.0,
                error_details=None,
            )
            mock_transcription.return_value = mock_transcription_result

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["processing_status"] == "completed"
        assert "Audio quality too poor for transcription" in data["message"]

        # Verify embedding service was not called for poor quality content
        mock_embedding.assert_not_called()

    def test_upload_voice_no_authentication(self, client, test_audio_file):
        """Test voice upload without authentication."""
        filename, file_content, content_type = test_audio_file

        response = client.post(
            "/api/v1/voice", files={"file": (filename, file_content, content_type)}
        )

        # Should return 401 Unauthorized (but might be 403 due to middleware)
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]

    def test_upload_voice_usage_limit_exceeded(
        self, client, mock_db, test_user, test_audio_file
    ):
        """Test voice upload when usage limits are exceeded."""

        # Override dependencies
        def mock_get_current_user():
            return test_user

        def mock_get_db():
            return mock_db

        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db

        with (
            patch(
                "app.services.usage.UsageService.check_subscription_validity",
                return_value=test_user,
            ),
            patch(
                "app.middleware.usage_validation.UsageService.check_daily_content_limit"
            ) as mock_check_limit,
        ):

            # Mock usage validation to raise exception
            from fastapi import HTTPException

            mock_check_limit.side_effect = HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Daily content limit exceeded",
            )

            filename, file_content, content_type = test_audio_file

            response = client.post(
                "/api/v1/voice", files={"file": (filename, file_content, content_type)}
            )

        # Clean up overrides
        app.dependency_overrides.clear()

        # Assertions
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Daily content limit exceeded" in response.json()["detail"]


class TestVoiceUploadValidation:
    """Test cases for voice upload validation."""

    def test_validate_file_size_free_tier(self):
        """Test file size validation for free tier."""
        from app.middleware.usage_validation import validate_file_size
        from app.models.schemas import FileType

        # Mock file with size > 5MB
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=b"x" * (6 * 1024 * 1024))  # 6MB
        mock_file.seek = AsyncMock()

        # Should raise exception for free tier
        with pytest.raises(Exception):  # HTTPException
            import asyncio

            asyncio.run(validate_file_size(mock_file, "free", FileType.VOICE))

    def test_validate_file_size_premium_tier(self):
        """Test file size validation for premium tier."""
        from app.middleware.usage_validation import validate_file_size
        from app.models.schemas import FileType

        # Mock file with size < 25MB
        mock_file = Mock()
        mock_file.read = AsyncMock(return_value=b"x" * (20 * 1024 * 1024))  # 20MB
        mock_file.seek = AsyncMock()

        # Should not raise exception for premium tier
        import asyncio

        try:
            asyncio.run(validate_file_size(mock_file, "premium", FileType.VOICE))
        except Exception:
            pytest.fail("validate_file_size raised exception unexpectedly")
