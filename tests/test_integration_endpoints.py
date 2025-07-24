"""
Integration tests for ChronoTrail API endpoints.

This module contains comprehensive integration tests that test complete request/response
cycles for all endpoints, file upload and processing workflows, and authentication
and authorization flows.
"""
import io
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

import pytest
from fastapi import status
from httpx import AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import (
    User, SubscriptionTier, OAuthProvider, FileType, ContentType,
    LocationVisitCreate, TextNoteCreate, QueryRequest
)
from app.models.database import (
    User as DBUser,
    LocationVisit as DBLocationVisit,
    TextNote as DBTextNote,
    MediaFile as DBMediaFile,
    QuerySession as DBQuerySession,
    DailyUsage as DBDailyUsage
)


class TestAuthenticationIntegration:
    """Integration tests for authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_google_oauth_signin_complete_flow(self, client: AsyncClient, mock_db: AsyncSession):
        """Test complete Google OAuth sign-in flow."""
        # Mock Google OAuth verification
        mock_google_user = {
            "sub": "google_123456",
            "email": "test@example.com",
            "name": "Test User",
            "picture": "https://example.com/avatar.jpg"
        }
        
        # Mock database user creation
        db_user = DBUser(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123456",
            display_name="Test User",
            profile_picture_url="https://example.com/avatar.jpg",
            subscription_tier="free",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with patch("app.services.auth.AuthService.verify_google_token", return_value=mock_google_user), \
             patch("app.services.auth.AuthService.get_or_create_user", return_value=db_user), \
             patch("app.services.auth.AuthService.create_tokens_for_user") as mock_create_tokens:
            
            # Mock token creation
            mock_create_tokens.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600
            }
            
            # Make request
            response = await client.post(
                "/api/v1/auth/google",
                json={"access_token": "google_oauth_token"}
            )
            
            # Verify response
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["access_token"] == "test_access_token"
            assert data["refresh_token"] == "test_refresh_token"
            assert data["token_type"] == "bearer"
            assert data["expires_in"] == 3600
    
    @pytest.mark.asyncio
    async def test_apple_signin_complete_flow(self, client: AsyncClient, mock_db: AsyncSession):
        """Test complete Apple Sign-In flow."""
        # Mock Apple OAuth verification
        mock_apple_user = {
            "sub": "apple_123456",
            "email": "test@example.com",
            "name": "Test User"
        }
        
        # Mock database user creation
        db_user = DBUser(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="apple",
            oauth_subject="apple_123456",
            display_name="Test User",
            subscription_tier="free",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with patch("app.services.auth.AuthService.verify_apple_token", return_value=mock_apple_user), \
             patch("app.services.auth.AuthService.get_or_create_user", return_value=db_user), \
             patch("app.services.auth.AuthService.create_tokens_for_user") as mock_create_tokens:
            
            # Mock token creation
            mock_create_tokens.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600
            }
            
            # Make request
            response = await client.post(
                "/api/v1/auth/apple",
                json={"identity_token": "apple_identity_token"}
            )
            
            # Verify response
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["access_token"] == "test_access_token"
            assert data["refresh_token"] == "test_refresh_token"
            assert data["token_type"] == "bearer"
            assert data["expires_in"] == 3600
    
    @pytest.mark.asyncio
    async def test_refresh_token_complete_flow(self, client: AsyncClient, mock_db: AsyncSession):
        """Test complete token refresh flow."""
        with patch("app.services.auth.AuthService.refresh_access_token") as mock_refresh:
            # Mock token refresh
            mock_refresh.return_value = {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "token_type": "bearer",
                "expires_in": 3600
            }
            
            # Make request
            response = await client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": "old_refresh_token"}
            )
            
            # Verify response
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["access_token"] == "new_access_token"
            assert data["refresh_token"] == "new_refresh_token"
    
    @pytest.mark.asyncio
    async def test_get_current_user_profile(self, client: AsyncClient, mock_auth_user: Dict[str, Any]):
        """Test getting current user profile."""
        response = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == mock_auth_user["user"].email
        assert data["display_name"] == mock_auth_user["user"].display_name


class TestLocationVisitsIntegration:
    """Integration tests for location visits endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_location_visit_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete location visit creation flow."""
        # Mock database creation
        visit_id = uuid4()
        db_visit = DBLocationVisit(
            id=visit_id,
            user_id=mock_auth_user["user"].id,
            longitude=-122.4194,
            latitude=37.7749,
            address="San Francisco, CA",
            names=["Golden Gate Park"],
            visit_time=datetime.utcnow() - timedelta(hours=1),
            duration=120,
            description="Beautiful park visit",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with patch("app.repositories.location_visits.location_visit_repository.create", return_value=db_visit):
            response = await client.post(
                "/api/v1/locations",
                json={
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA",
                    "names": ["Golden Gate Park"],
                    "visit_time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "duration": 120,
                    "description": "Beautiful park visit"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["id"] == str(visit_id)
            assert data["address"] == "San Francisco, CA"
            assert data["names"] == ["Golden Gate Park"]
            assert data["duration"] == 120
    
    @pytest.mark.asyncio
    async def test_get_location_visits_with_pagination(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test location visits retrieval with pagination."""
        # Mock database query
        visits = [
            DBLocationVisit(
                id=uuid4(),
                user_id=mock_auth_user["user"].id,
                longitude=-122.4194,
                latitude=37.7749,
                address=f"Location {i}",
                visit_time=datetime.utcnow() - timedelta(hours=i),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ) for i in range(5)
        ]
        
        with patch("app.repositories.location_visits.location_visit_repository.get_by_user_with_date_range", return_value=visits), \
             patch("app.repositories.location_visits.location_visit_repository.count_by_user", return_value=25):
            
            response = await client.get(
                "/api/v1/locations?page=2&page_size=5",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total"] == 25
            assert data["page"] == 2
            assert data["page_size"] == 5
            assert len(data["visits"]) == 5
    
    @pytest.mark.asyncio
    async def test_update_location_visit_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete location visit update flow."""
        visit_id = uuid4()
        updated_visit = DBLocationVisit(
            id=visit_id,
            user_id=mock_auth_user["user"].id,
            longitude=-122.4194,
            latitude=37.7749,
            address="San Francisco, CA",
            names=["Updated Park", "New Name"],
            visit_time=datetime.utcnow() - timedelta(hours=1),
            description="Updated description",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with patch("app.repositories.location_visits.location_visit_repository.update_by_id", return_value=updated_visit):
            response = await client.put(
                f"/api/v1/locations/{visit_id}",
                json={
                    "description": "Updated description",
                    "names": ["Updated Park", "New Name"]
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["description"] == "Updated description"
            assert data["names"] == ["Updated Park", "New Name"]


class TestTextNotesIntegration:
    """Integration tests for text notes endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_text_note_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete text note creation flow including vector embedding."""
        # Mock database creation
        note_id = uuid4()
        db_note = DBTextNote(
            id=note_id,
            user_id=mock_auth_user["user"].id,
            text_content="This is a test note about my day",
            timestamp=datetime.utcnow(),
            longitude=-122.4194,
            latitude=37.7749,
            address="San Francisco, CA",
            names=["Coffee Shop"],
            created_at=datetime.utcnow()
        )
        
        with patch("app.repositories.text_notes.text_note_repository.create", return_value=db_note), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123") as mock_embedding, \
             patch("app.services.usage.UsageService.increment_content_usage") as mock_increment:
            
            response = await client.post(
                "/api/v1/notes",
                json={
                    "text_content": "This is a test note about my day",
                    "timestamp": datetime.utcnow().isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA",
                    "names": ["Coffee Shop"]
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["id"] == str(note_id)
            assert data["text_content"] == "This is a test note about my day"
            assert data["address"] == "San Francisco, CA"
            assert data["names"] == ["Coffee Shop"]
            
            # Verify vector embedding was called
            mock_embedding.assert_called_once()
            mock_increment.assert_called_once()


class TestPhotoUploadIntegration:
    """Integration tests for photo upload endpoints."""
    
    def create_test_image(self, size=(100, 100), format='JPEG'):
        """Create a test image file."""
        image = Image.new('RGB', size, color='red')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format=format)
        img_bytes.seek(0)
        return img_bytes
    
    @pytest.mark.asyncio
    async def test_photo_upload_complete_processing_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete photo upload and processing flow."""
        # Create test image
        test_image = self.create_test_image()
        
        # Mock database creation
        media_id = uuid4()
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="test_image.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        # Mock processing result
        from app.models.schemas import ProcessedPhoto
        processed_result = ProcessedPhoto(
            original_id=media_id,
            processed_text="Test image with extracted text",
            content_type="image_text",
            processing_status="completed",
            extracted_text="Some extracted text",
            image_description="A test image",
            detected_objects=["object1", "object2"]
        )
        
        with patch("app.aws.storage.file_storage_service.upload_file") as mock_storage, \
             patch("app.repositories.media_files.media_file_repository.create", return_value=db_media), \
             patch("app.aws.image_processing.image_processing_service.process_image", return_value=processed_result), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123") as mock_embedding, \
             patch("app.services.usage.UsageService.increment_content_usage") as mock_increment:
            
            # Mock storage service
            mock_storage.return_value = {
                'file_id': str(media_id),
                's3_key': 'test/path/image.jpg',
                'file_size': 1024000,
                'timestamp': datetime.utcnow()
            }
            
            response = await client.post(
                "/api/v1/photos",
                files={"file": ("test_image.jpg", test_image, "image/jpeg")},
                data={
                    "timestamp": datetime.utcnow().isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA",
                    "names": "Golden Gate Park,Park"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["file_type"] == "photo"
            assert data["processing_status"] == "completed"
            assert "Test image with extracted text" in data["message"]
            
            # Verify all processing steps were called
            mock_storage.assert_called_once()
            mock_embedding.assert_called_once()
            mock_increment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_photo_upload_processing_failure_handling(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test photo upload with processing failure handling."""
        test_image = self.create_test_image()
        
        # Mock database creation
        media_id = uuid4()
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="test_image.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        with patch("app.aws.storage.file_storage_service.upload_file") as mock_storage, \
             patch("app.repositories.media_files.media_file_repository.create", return_value=db_media), \
             patch("app.aws.image_processing.image_processing_service.process_image", side_effect=Exception("Processing failed")), \
             patch("app.services.usage.UsageService.increment_content_usage") as mock_increment:
            
            # Mock storage service
            mock_storage.return_value = {
                'file_id': str(media_id),
                's3_key': 'test/path/image.jpg',
                'file_size': 1024000,
                'timestamp': datetime.utcnow()
            }
            
            response = await client.post(
                "/api/v1/photos",
                files={"file": ("test_image.jpg", test_image, "image/jpeg")},
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            # Should still succeed but with failed processing status
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["processing_status"] == "failed"
            assert "Image processing failed" in data["message"]
            assert data["error_details"] is not None
            
            # Verify file was still stored and usage incremented
            mock_storage.assert_called_once()
            mock_increment.assert_called_once()


class TestVoiceUploadIntegration:
    """Integration tests for voice upload endpoints."""
    
    def create_test_audio(self):
        """Create a test audio file."""
        # Create a simple audio file mock
        audio_data = b"fake_audio_data" * 1000  # Simulate audio data
        return io.BytesIO(audio_data)
    
    @pytest.mark.asyncio
    async def test_voice_upload_complete_processing_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete voice upload and transcription flow."""
        test_audio = self.create_test_audio()
        
        # Mock database creation
        media_id = uuid4()
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="voice",
            file_path="test/path/audio.mp3",
            original_filename="test_audio.mp3",
            file_size=50000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        # Mock transcription result
        from app.models.schemas import ProcessedVoice
        transcription_result = ProcessedVoice(
            original_id=media_id,
            processed_text="This is a transcribed voice note about my day",
            content_type="voice_transcript",
            processing_status="completed",
            transcript="This is a transcribed voice note about my day",
            confidence_score=0.95,
            duration_seconds=30.5
        )
        
        with patch("app.aws.storage.file_storage_service.upload_file") as mock_storage, \
             patch("app.repositories.media_files.media_file_repository.create", return_value=db_media), \
             patch("app.aws.transcription.transcription_service.process_voice", return_value=transcription_result), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123") as mock_embedding, \
             patch("app.services.usage.UsageService.increment_content_usage") as mock_increment:
            
            # Mock storage service
            mock_storage.return_value = {
                'file_id': str(media_id),
                's3_key': 'test/path/audio.mp3',
                'file_size': 50000,
                'timestamp': datetime.utcnow()
            }
            
            response = await client.post(
                "/api/v1/voice",
                files={"file": ("test_audio.mp3", test_audio, "audio/mpeg")},
                data={
                    "timestamp": datetime.utcnow().isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "San Francisco, CA",
                    "names": "Coffee Shop"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["file_type"] == "voice"
            assert data["processing_status"] == "completed"
            assert "This is a transcribed voice note" in data["message"]
            
            # Verify all processing steps were called
            mock_storage.assert_called_once()
            mock_embedding.assert_called_once()
            mock_increment.assert_called_once()


class TestQueryIntegration:
    """Integration tests for query endpoints."""
    
    @pytest.mark.asyncio
    async def test_query_complete_processing_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete query processing flow with LLM integration."""
        # Mock query processing service
        from app.models.schemas import QueryResponse, MediaReference
        
        mock_response = QueryResponse(
            answer="Based on your timeline, you visited Golden Gate Park yesterday and took some photos.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Golden Gate Park visit",
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                    "relevance_score": 0.95
                }
            ],
            media_references=[
                MediaReference(
                    media_id=uuid4(),
                    media_type="photo",
                    description="Photo from Golden Gate Park",
                    timestamp=datetime.utcnow() - timedelta(days=1),
                    location={"address": "Golden Gate Park, San Francisco"}
                )
            ],
            session_id="session_123"
        )
        
        with patch("app.services.query.query_processing_service.process_query", return_value=mock_response) as mock_process:
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "What did I do yesterday?",
                    "session_id": "session_123"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "Golden Gate Park" in data["answer"]
            assert len(data["sources"]) == 1
            assert len(data["media_references"]) == 1
            assert data["session_id"] == "session_123"
            
            # Verify query processing was called
            mock_process.assert_called_once()


class TestMediaRetrievalIntegration:
    """Integration tests for media retrieval endpoints."""
    
    @pytest.mark.asyncio
    async def test_media_file_retrieval_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete media file retrieval flow."""
        media_id = uuid4()
        
        # Mock database media file
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="test_image.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        # Mock file content
        fake_image_data = b"fake_image_data" * 1000
        
        with patch("app.repositories.media_files.media_file_repository.get_by_user", return_value=db_media), \
             patch("app.aws.storage.file_storage_service.download_file", return_value=fake_image_data):
            
            response = await client.get(
                f"/api/v1/media/{media_id}",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "image/jpeg"
            assert response.headers["content-length"] == str(len(fake_image_data))
            assert response.content == fake_image_data
    
    @pytest.mark.asyncio
    async def test_media_file_info_retrieval(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test media file info retrieval."""
        media_id = uuid4()
        
        # Mock database media file
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="test_image.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            longitude=-122.4194,
            latitude=37.7749,
            address="San Francisco, CA",
            names=["Golden Gate Park"],
            created_at=datetime.utcnow()
        )
        
        with patch("app.repositories.media_files.media_file_repository.get_by_user", return_value=db_media):
            response = await client.get(
                f"/api/v1/media/{media_id}/info",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["id"] == str(media_id)
            assert data["file_type"] == "photo"
            assert data["original_filename"] == "test_image.jpg"
            assert data["file_size"] == 1024000
            assert data["location"]["address"] == "San Francisco, CA"
            assert data["location"]["names"] == ["Golden Gate Park"]
    
    @pytest.mark.asyncio
    async def test_media_download_url_generation(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test pre-signed download URL generation."""
        media_id = uuid4()
        
        # Mock database media file
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="test_image.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        mock_download_url = "https://s3.amazonaws.com/bucket/test/path/image.jpg?signature=abc123"
        
        with patch("app.repositories.media_files.media_file_repository.get_by_user", return_value=db_media), \
             patch("app.aws.storage.file_storage_service.generate_download_url", return_value=mock_download_url):
            
            response = await client.get(
                f"/api/v1/media/{media_id}/download-url?expiration=1800",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["download_url"] == mock_download_url
            assert data["expires_in"] == 1800
            assert data["file_info"]["id"] == str(media_id)
            assert data["file_info"]["file_type"] == "photo"


class TestUsageAndSubscriptionIntegration:
    """Integration tests for usage and subscription endpoints."""
    
    @pytest.mark.asyncio
    async def test_usage_stats_retrieval_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete usage statistics retrieval flow."""
        from app.models.schemas import UsageStats
        
        mock_usage_stats = UsageStats(
            date=datetime.utcnow().date(),
            text_notes_count=2,
            media_files_count=1,
            queries_count=5,
            daily_limits={
                "content_limit": 3,
                "query_limit": 10,
                "total_content_used": 3
            }
        )
        
        with patch("app.services.usage.UsageService.get_usage_stats", return_value=mock_usage_stats):
            response = await client.get(
                "/api/v1/usage",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["text_notes_count"] == 2
            assert data["media_files_count"] == 1
            assert data["queries_count"] == 5
            assert data["daily_limits"]["content_limit"] == 3
    
    @pytest.mark.asyncio
    async def test_subscription_info_retrieval_complete_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete subscription information retrieval flow."""
        from app.models.schemas import SubscriptionInfo
        
        mock_subscription_info = SubscriptionInfo(
            tier=SubscriptionTier.FREE,
            expires_at=None,
            daily_limits={
                "content_limit": 3,
                "query_limit": 10
            },
            query_history_months=3,
            max_file_size_mb=5,
            max_storage_mb=100
        )
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=mock_auth_user["user"]), \
             patch("app.services.usage.UsageService.get_subscription_info", return_value=mock_subscription_info):
            
            response = await client.get(
                "/api/v1/usage/subscription",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["tier"] == "free"
            assert data["expires_at"] is None
            assert data["daily_limits"]["content_limit"] == 3
            assert data["query_history_months"] == 3


class TestAuthorizationIntegration:
    """Integration tests for authorization and access control."""
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_to_protected_endpoints(self, client: AsyncClient):
        """Test that protected endpoints require authentication."""
        endpoints_to_test = [
            ("GET", "/api/v1/auth/me"),
            ("GET", "/api/v1/usage"),
            ("GET", "/api/v1/usage/subscription"),
            ("POST", "/api/v1/locations"),
            ("GET", "/api/v1/locations"),
            ("POST", "/api/v1/notes"),
            ("POST", "/api/v1/photos"),
            ("POST", "/api/v1/voice"),
            ("POST", "/api/v1/query"),
            ("GET", f"/api/v1/media/{uuid4()}"),
        ]
        
        for method, endpoint in endpoints_to_test:
            if method == "GET":
                response = await client.get(endpoint)
            elif method == "POST":
                response = await client.post(endpoint, json={})
            
            # Should return 401 Unauthorized or 403 Forbidden
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED, 
                status.HTTP_403_FORBIDDEN
            ], f"Endpoint {method} {endpoint} should require authentication"
    
    @pytest.mark.asyncio
    async def test_cross_user_data_access_prevention(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession
    ):
        """Test that users cannot access other users' data."""
        user1_id = uuid4()
        user2_id = uuid4()
        
        # Create two different users
        user1 = User(
            id=user1_id,
            email="user1@example.com",
            oauth_provider=OAuthProvider.GOOGLE,
            oauth_subject="user1_subject",
            subscription_tier=SubscriptionTier.FREE
        )
        
        user2 = User(
            id=user2_id,
            email="user2@example.com",
            oauth_provider=OAuthProvider.GOOGLE,
            oauth_subject="user2_subject",
            subscription_tier=SubscriptionTier.FREE
        )
        
        # Mock media file belonging to user2
        media_id = uuid4()
        user2_media = DBMediaFile(
            id=media_id,
            user_id=user2_id,
            file_type="photo",
            file_path="user2/photo.jpg",
            original_filename="photo.jpg",
            file_size=1000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        # Mock repository to return the file (exists) but not for user1
        with patch("app.repositories.media_files.media_file_repository.get_by_user", return_value=None), \
             patch("app.repositories.media_files.media_file_repository.get", return_value=user2_media), \
             patch("app.services.auth.get_current_active_user", return_value=user1):
            
            response = await client.get(
                f"/api/v1/media/{media_id}",
                headers={"Authorization": "Bearer user1_token"}
            )
            
            # Should return 403 Forbidden (file exists but belongs to different user)
            assert response.status_code == status.HTTP_403_FORBIDDEN
            assert "Access denied" in response.json()["detail"]


class TestFileUploadWorkflows:
    """Integration tests for complete file upload workflows."""
    
    @pytest.mark.asyncio
    async def test_photo_upload_with_usage_limit_enforcement(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test photo upload with usage limit enforcement."""
        test_image = io.BytesIO(b"fake_image_data")
        
        # Mock usage limit exceeded
        from fastapi import HTTPException
        with patch("app.middleware.usage_validation.UsageService.check_daily_content_limit", 
                   side_effect=HTTPException(status_code=429, detail="Daily limit exceeded")):
            
            response = await client.post(
                "/api/v1/photos",
                files={"file": ("test.jpg", test_image, "image/jpeg")},
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "Daily limit exceeded" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_voice_upload_with_file_size_validation(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test voice upload with file size validation."""
        # Create large audio file mock
        large_audio = io.BytesIO(b"fake_audio_data" * 10000)  # Large file
        
        # Mock file size validation failure
        from fastapi import HTTPException
        with patch("app.middleware.usage_validation.validate_file_size", 
                   side_effect=HTTPException(status_code=413, detail="File too large")):
            
            response = await client.post(
                "/api/v1/voice",
                files={"file": ("large_audio.mp3", large_audio, "audio/mpeg")},
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            assert "File too large" in response.json()["detail"]


class TestEndToEndWorkflows:
    """Integration tests for complete end-to-end user workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_user_journey_data_submission_to_query(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test complete user journey from data submission to querying."""
        
        # Step 1: Create location visit
        visit_id = uuid4()
        db_visit = DBLocationVisit(
            id=visit_id,
            user_id=mock_auth_user["user"].id,
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco",
            names=["Golden Gate Park"],
            visit_time=datetime.utcnow() - timedelta(hours=2),
            duration=120,
            description="Beautiful park visit",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        with patch("app.repositories.location_visits.location_visit_repository.create", return_value=db_visit):
            location_response = await client.post(
                "/api/v1/locations",
                json={
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco",
                    "names": ["Golden Gate Park"],
                    "visit_time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    "duration": 120,
                    "description": "Beautiful park visit"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert location_response.status_code == status.HTTP_201_CREATED
        
        # Step 2: Submit text note
        note_id = uuid4()
        db_note = DBTextNote(
            id=note_id,
            user_id=mock_auth_user["user"].id,
            text_content="Had a great time at the park today, saw beautiful flowers",
            timestamp=datetime.utcnow() - timedelta(hours=1),
            longitude=-122.4194,
            latitude=37.7749,
            address="Golden Gate Park, San Francisco",
            created_at=datetime.utcnow()
        )
        
        with patch("app.repositories.text_notes.text_note_repository.create", return_value=db_note), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123"), \
             patch("app.services.usage.UsageService.increment_content_usage"):
            
            note_response = await client.post(
                "/api/v1/notes",
                json={
                    "text_content": "Had a great time at the park today, saw beautiful flowers",
                    "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Golden Gate Park, San Francisco"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert note_response.status_code == status.HTTP_201_CREATED
        
        # Step 3: Query the data
        from app.models.schemas import QueryResponse
        
        mock_query_response = QueryResponse(
            answer="You visited Golden Gate Park today and had a great time seeing beautiful flowers. You spent about 2 hours there.",
            sources=[
                {
                    "type": "location_visit",
                    "content": "Golden Gate Park visit - Beautiful park visit",
                    "timestamp": datetime.utcnow() - timedelta(hours=2),
                    "relevance_score": 0.95
                },
                {
                    "type": "text_note",
                    "content": "Had a great time at the park today, saw beautiful flowers",
                    "timestamp": datetime.utcnow() - timedelta(hours=1),
                    "relevance_score": 0.90
                }
            ],
            media_references=[],
            session_id="session_123"
        )
        
        with patch("app.services.query.query_processing_service.process_query", return_value=mock_query_response):
            query_response = await client.post(
                "/api/v1/query",
                json={
                    "query": "What did I do at the park today?",
                    "session_id": "session_123"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert query_response.status_code == status.HTTP_200_OK
            data = query_response.json()
            assert "Golden Gate Park" in data["answer"]
            assert "beautiful flowers" in data["answer"]
            assert len(data["sources"]) == 2
            assert data["session_id"] == "session_123"
    
    @pytest.mark.asyncio
    async def test_conversational_media_retrieval_flow(
        self, 
        client: AsyncClient, 
        mock_db: AsyncSession,
        mock_auth_user: Dict[str, Any]
    ):
        """Test conversational media retrieval flow."""
        
        # Step 1: Upload photo
        test_image = io.BytesIO(b"fake_image_data")
        media_id = uuid4()
        
        db_media = DBMediaFile(
            id=media_id,
            user_id=mock_auth_user["user"].id,
            file_type="photo",
            file_path="test/path/image.jpg",
            original_filename="park_photo.jpg",
            file_size=1024000,
            timestamp=datetime.utcnow(),
            created_at=datetime.utcnow()
        )
        
        from app.models.schemas import ProcessedPhoto
        processed_result = ProcessedPhoto(
            original_id=media_id,
            processed_text="Beautiful flowers in Golden Gate Park",
            content_type="image_desc",
            processing_status="completed",
            image_description="Beautiful flowers in Golden Gate Park"
        )
        
        with patch("app.aws.storage.file_storage_service.upload_file") as mock_storage, \
             patch("app.repositories.media_files.media_file_repository.create", return_value=db_media), \
             patch("app.aws.image_processing.image_processing_service.process_image", return_value=processed_result), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123"), \
             patch("app.services.usage.UsageService.increment_content_usage"):
            
            mock_storage.return_value = {
                'file_id': str(media_id),
                's3_key': 'test/path/image.jpg',
                'file_size': 1024000,
                'timestamp': datetime.utcnow()
            }
            
            upload_response = await client.post(
                "/api/v1/photos",
                files={"file": ("park_photo.jpg", test_image, "image/jpeg")},
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert upload_response.status_code == status.HTTP_201_CREATED
        
        # Step 2: Query for photos
        from app.models.schemas import QueryResponse, MediaReference
        
        mock_query_response = QueryResponse(
            answer="You took a photo of beautiful flowers in Golden Gate Park. Would you like to see the photo?",
            sources=[
                {
                    "type": "image_desc",
                    "content": "Beautiful flowers in Golden Gate Park",
                    "timestamp": datetime.utcnow(),
                    "relevance_score": 0.95
                }
            ],
            media_references=[
                MediaReference(
                    media_id=media_id,
                    media_type="photo",
                    description="Photo of beautiful flowers in Golden Gate Park",
                    timestamp=datetime.utcnow(),
                    location={"address": "Golden Gate Park, San Francisco"}
                )
            ],
            session_id="session_456"
        )
        
        with patch("app.services.query.query_processing_service.process_query", return_value=mock_query_response):
            query_response = await client.post(
                "/api/v1/query",
                json={
                    "query": "Show me photos of flowers",
                    "session_id": "session_456"
                },
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert query_response.status_code == status.HTTP_200_OK
            data = query_response.json()
            assert len(data["media_references"]) == 1
            assert data["media_references"][0]["media_id"] == str(media_id)
            assert data["media_references"][0]["media_type"] == "photo"
        
        # Step 3: Retrieve the referenced media
        fake_image_data = b"fake_image_data" * 1000
        
        with patch("app.repositories.media_files.media_file_repository.get_by_user", return_value=db_media), \
             patch("app.aws.storage.file_storage_service.download_file", return_value=fake_image_data):
            
            media_response = await client.get(
                f"/api/v1/media/{media_id}",
                headers={"Authorization": f"Bearer {mock_auth_user['token']}"}
            )
            
            assert media_response.status_code == status.HTTP_200_OK
            assert media_response.headers["content-type"] == "image/jpeg"
            assert media_response.content == fake_image_data