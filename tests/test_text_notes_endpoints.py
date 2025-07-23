"""
Tests for text notes endpoints.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.services.auth import get_current_active_user
from app.db.database import get_async_db
from app.models.database import TextNote as DBTextNote
from app.models.schemas import SubscriptionTier, ContentType, User


class TestTextNotesEndpoints:
    """Test cases for text notes endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock(spec=AsyncSession)
    
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
    
    @pytest.fixture
    def valid_text_note_data(self) -> dict:
        """Valid text note creation data."""
        return {
            "text_content": "This is a test note about my day at the park.",
            "timestamp": datetime.utcnow().isoformat(),
            "longitude": -122.4194,
            "latitude": 37.7749,
            "address": "Golden Gate Park, San Francisco, CA",
            "names": ["Golden Gate Park", "Park"]
        }
    
    @pytest.fixture
    def minimal_text_note_data(self) -> dict:
        """Minimal valid text note creation data."""
        return {
            "text_content": "Simple note without location.",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def test_create_text_note_success(
        self,
        client,
        mock_db,
        sample_user,
        valid_text_note_data
    ):
        """Test successful text note creation."""
        # Create mock database text note
        mock_db_note = DBTextNote(
            id=uuid4(),
            user_id=sample_user.id,
            text_content=valid_text_note_data["text_content"],
            timestamp=datetime.fromisoformat(valid_text_note_data["timestamp"]),
            longitude=Decimal(str(valid_text_note_data["longitude"])),
            latitude=Decimal(str(valid_text_note_data["latitude"])),
            address=valid_text_note_data["address"],
            names=valid_text_note_data["names"],
            created_at=datetime.utcnow()
        )
        
        # Mock the database operations properly
        mock_result = AsyncMock()
        mock_scalars = AsyncMock()
        mock_scalars.first.return_value = None  # No existing daily usage
        mock_result.scalars.return_value = mock_scalars
        
        # Make execute return the mock_result directly (not as a coroutine)
        async def mock_execute(*args, **kwargs):
            return mock_result
        mock_db.execute = mock_execute
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.middleware.usage_validation.UsageService.check_daily_content_limit", return_value=None), \
             patch("app.repositories.text_notes.text_note_repository.create", return_value=mock_db_note) as mock_create, \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123") as mock_embedding, \
             patch("app.services.usage.UsageService.increment_content_usage", return_value=None) as mock_increment:
            
            response = client.post(
                "/api/v1/notes",
                json=valid_text_note_data
            )
        
        # Clean up overrides
        app.dependency_overrides.clear()
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        # Verify response structure
        assert "id" in data
        assert data["text_content"] == valid_text_note_data["text_content"]
        assert data["user_id"] == str(sample_user.id)
        assert data["longitude"] == str(valid_text_note_data["longitude"])
        assert data["latitude"] == str(valid_text_note_data["latitude"])
        assert data["address"] == valid_text_note_data["address"]
        assert data["names"] == valid_text_note_data["names"]
        assert "created_at" in data
        assert "updated_at" in data
        
        # Verify repository was called
        mock_create.assert_called_once()
        
        # Verify embedding service was called
        mock_embedding.assert_called_once()
        
        # Verify usage was incremented
        mock_increment.assert_called_once_with(db=mock_db, user_id=sample_user.id, content_type="text_notes")
    
    def test_create_text_note_minimal_data(
        self,
        client,
        mock_db,
        sample_user,
        minimal_text_note_data
    ):
        """Test text note creation with minimal data (no location)."""
        # Create mock database text note
        mock_db_note = DBTextNote(
            id=uuid4(),
            user_id=sample_user.id,
            text_content=minimal_text_note_data["text_content"],
            timestamp=datetime.fromisoformat(minimal_text_note_data["timestamp"]),
            longitude=None,
            latitude=None,
            address=None,
            names=None,
            created_at=datetime.utcnow()
        )
        
        # Mock the database operations properly
        mock_result = AsyncMock()
        mock_scalars = AsyncMock()
        mock_scalars.first.return_value = None  # No existing daily usage
        mock_result.scalars.return_value = mock_scalars
        
        # Make execute return the mock_result directly (not as a coroutine)
        async def mock_execute(*args, **kwargs):
            return mock_result
        mock_db.execute = mock_execute
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        with patch("app.services.usage.UsageService.check_subscription_validity", return_value=sample_user), \
             patch("app.middleware.usage_validation.UsageService.check_daily_content_limit", return_value=None), \
             patch("app.repositories.text_notes.text_note_repository.create", return_value=mock_db_note) as mock_create, \
             patch("app.aws.embedding.embedding_service.process_and_store_text", return_value="doc_123") as mock_embedding, \
             patch("app.services.usage.UsageService.increment_content_usage", return_value=None) as mock_increment:
            
            response = client.post(
                "/api/v1/notes",
                json=minimal_text_note_data
            )
        
        # Clean up overrides
        app.dependency_overrides.clear()
        
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        
        # Verify response structure
        assert data["text_content"] == minimal_text_note_data["text_content"]
        assert data["longitude"] is None
        assert data["latitude"] is None
        assert data["address"] is None
        assert data["names"] is None
        
        # Verify embedding service was called with no location
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        assert call_args[1]["location"] is None
    
    def test_create_text_note_unauthenticated(
        self,
        client,
        valid_text_note_data
    ):
        """Test text note creation without authentication."""
        response = client.post(
            "/api/v1/notes",
            json=valid_text_note_data
        )
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_create_text_note_validation_error_empty_content(
        self,
        client,
        mock_db,
        sample_user
    ):
        """Test text note creation with empty content."""
        invalid_data = {
            "text_content": "",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Override dependencies
        def mock_get_current_user():
            return sample_user
        
        def mock_get_db():
            return mock_db
        
        app.dependency_overrides[get_current_active_user] = mock_get_current_user
        app.dependency_overrides[get_async_db] = mock_get_db
        
        # Mock the usage service methods to avoid database issues
        with patch("app.services.usage.UsageService.get_or_create_daily_usage") as mock_get_usage, \
             patch("app.services.usage.UsageService.check_daily_content_limit") as mock_check_limit:
            
            # Mock daily usage object
            mock_daily_usage = AsyncMock()
            mock_daily_usage.text_notes_count = 0
            mock_get_usage.return_value = mock_daily_usage
            mock_check_limit.return_value = None
            
            response = client.post(
                "/api/v1/notes",
                json=invalid_data
            )
        
        # Clean up overrides
        app.dependency_overrides.clear()
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY