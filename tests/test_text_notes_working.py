"""
Working tests for text notes endpoints using proper mocking patterns.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.db.database import get_async_db
from app.main import app
from app.models.database import TextNote as DBTextNote
from app.models.schemas import SubscriptionTier, User
from app.services.auth import get_current_active_user


class TestTextNotesWorking:
    """Working test cases for text notes endpoints."""

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
            updated_at=datetime.utcnow(),
        )

    @pytest.fixture
    def mock_db_note(self, sample_user):
        """Create a mock database text note."""
        return DBTextNote(
            id=uuid4(),
            user_id=sample_user.id,
            text_content="Test note content",
            timestamp=datetime.utcnow(),
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
            address="Test Address",
            names=["Test Location"],
            created_at=datetime.utcnow(),
        )

    def test_create_text_note_unauthenticated(self, client):
        """Test text note creation without authentication."""
        response = client.post(
            "/api/v1/notes",
            json={
                "text_content": "Test note",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "authenticated" in response.json()["detail"].lower()

    def test_create_text_note_success(self, client, sample_user, mock_db_note):
        """Test successful text note creation."""
        from app.middleware.usage_validation import validate_content_usage

        # Override dependencies - including the usage validation
        def mock_validate_content_usage():
            return sample_user

        def mock_get_db():
            return AsyncMock()

        app.dependency_overrides[validate_content_usage] = mock_validate_content_usage
        app.dependency_overrides[get_async_db] = mock_get_db

        # Mock all the services directly
        with (
            patch(
                "app.repositories.text_notes.text_note_repository.create",
                return_value=mock_db_note,
            ),
            patch(
                "app.aws.embedding.embedding_service.process_and_store_text",
                return_value="doc_123",
            ),
            patch("app.services.usage.UsageService.increment_content_usage"),
        ):

            response = client.post(
                "/api/v1/notes",
                json={
                    "text_content": "Test note content",
                    "timestamp": datetime.utcnow().isoformat(),
                    "longitude": -122.4194,
                    "latitude": 37.7749,
                    "address": "Test Address",
                    "names": ["Test Location"],
                },
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
        from app.middleware.usage_validation import validate_content_usage

        # Override dependencies - including the usage validation
        def mock_validate_content_usage():
            return sample_user

        def mock_get_db():
            return AsyncMock()

        app.dependency_overrides[validate_content_usage] = mock_validate_content_usage
        app.dependency_overrides[get_async_db] = mock_get_db

        response = client.post(
            "/api/v1/notes",
            json={
                "text_content": "",  # Empty content should fail validation
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Clean up overrides
        app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
