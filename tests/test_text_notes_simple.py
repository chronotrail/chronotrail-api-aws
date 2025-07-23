"""
Simple tests for text notes endpoints.
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth import get_current_active_user
from app.db.database import get_async_db
from app.models.schemas import User, SubscriptionTier


class TestTextNotesSimple:
    """Simple test cases for text notes endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_user(self):
        """Create a sample user for testing."""
        return User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
            display_name="Test User",
            subscription_tier=SubscriptionTier.FREE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def test_create_text_note_unauthenticated(self, client):
        """Test text note creation without authentication."""
        response = client.post(
            "/api/v1/notes",
            json={
                "text_content": "Test note",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "authenticated" in response.json()["detail"].lower()
    
    def test_create_text_note_success_mocked(self, client, sample_user):
        """Test successful text note creation with full mocking."""
        # Mock database note
        from app.models.database import TextNote as DBTextNote
        from decimal import Decimal
        
        mock_db_note = DBTextNote(
            id=uuid4(),
            user_id=sample_user.id,
            text_content="Test note content",
            timestamp=datetime.utcnow(),
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
            address="Test Address",
            names=["Test Location"],
            created_at=datetime.utcnow()
        )
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return AsyncMock()
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity"), \
             patch("app.middleware.usage_validation.UsageService.check_daily_content_limit"), \
             patch("app.repositories.text_notes.text_note_repository.create", return_value=mock_db_note), \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123"), \
             patch("app.services.usage.UsageService.increment_content_usage"):
            
            response = client.post(
                "/api/v1/notes",
                json={
                    "text_content": "Test note content",
                    "timestamp": datetime.utcnow().isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Test Address",
                    "names": ["Test Location"]
                }
            )
        
        # Clean up overrides
        app.dependency_overrides.clear()
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        # Verify response structure
        assert "id" in data
        assert data["text_content"] == "Test note content"
        assert data["user_id"] == str(sample_user.id)
        assert data["longitude"] == "-122.4194"
        assert data["latitude"] == "37.7749"
        assert data["address"] == "Test Address"
        assert data["names"] == ["Test Location"]
        assert "created_at" in data
    
    def test_create_text_note_validation_error(self, client, sample_user):
        """Test text note creation with validation error."""
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return AsyncMock()
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        response = client.post(
            "/api/v1/notes",
            json={
                "text_content": "",  # Empty content should fail validation
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Clean up overrides
        app.dependency_overrides.clear()
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY