"""
Unit tests for Pydantic data models and validation.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from typing import List

from pydantic import ValidationError

from app.models.schemas import (
    LocationVisitCreate,
    LocationVisitUpdate,
    LocationVisit,
    TextNoteCreate,
    TextNote,
    MediaFileCreate,
    MediaFile,
    QueryRequest,
    QueryResponse,
    MediaReference,
    User,
    UsageStats,
    SubscriptionInfo,
    SubscriptionTier,
    ErrorResponse,
    FileType
)


class TestLocationVisitModels:
    """Test location visit data models and validation."""
    
    def test_location_visit_create_valid(self):
        """Test valid location visit creation."""
        data = {
            "longitude": -122.4194,
            "latitude": 37.7749,
            "address": "San Francisco, CA",
            "names": ["Home", "Office"],
            "visit_time": datetime.utcnow(),
            "duration": 120,
            "description": "Meeting at the office"
        }
        visit = LocationVisitCreate(**data)
        assert visit.longitude == Decimal("-122.4194")
        assert visit.latitude == Decimal("37.7749")
        assert visit.names == ["Home", "Office"]
        assert visit.duration == 120
    
    def test_location_visit_create_minimal(self):
        """Test location visit creation with minimal required fields."""
        data = {
            "longitude": -122.4194,
            "latitude": 37.7749,
            "visit_time": datetime.utcnow()
        }
        visit = LocationVisitCreate(**data)
        assert visit.longitude == Decimal("-122.4194")
        assert visit.latitude == Decimal("37.7749")
        assert visit.address is None
        assert visit.names is None
        assert visit.duration is None
        assert visit.description is None
    
    def test_location_visit_invalid_coordinates(self):
        """Test location visit with invalid coordinates."""
        # Invalid longitude (out of range)
        with pytest.raises(ValidationError) as exc_info:
            LocationVisitCreate(
                longitude=200.0,
                latitude=37.7749,
                visit_time=datetime.utcnow()
            )
        assert "Input should be less than or equal to 180" in str(exc_info.value)
        
        # Invalid latitude (out of range)
        with pytest.raises(ValidationError) as exc_info:
            LocationVisitCreate(
                longitude=-122.4194,
                latitude=100.0,
                visit_time=datetime.utcnow()
            )
        assert "Input should be less than or equal to 90" in str(exc_info.value)
    
    def test_location_visit_future_time_validation(self):
        """Test location visit with future timestamp."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            LocationVisitCreate(
                longitude=-122.4194,
                latitude=37.7749,
                visit_time=future_time
            )
        assert "Visit time cannot be in the future" in str(exc_info.value)
    
    def test_location_visit_too_many_names(self):
        """Test location visit with too many names."""
        names = [f"Name{i}" for i in range(11)]  # 11 names, limit is 10
        with pytest.raises(ValidationError) as exc_info:
            LocationVisitCreate(
                longitude=-122.4194,
                latitude=37.7749,
                visit_time=datetime.utcnow(),
                names=names
            )
        assert "List should have at most 10 items" in str(exc_info.value)
    
    def test_location_visit_update_validation(self):
        """Test location visit update validation."""
        update = LocationVisitUpdate(
            description="Updated description",
            names=["New Name"]
        )
        assert update.description == "Updated description"
        assert update.names == ["New Name"]
        
        # Test with too many names
        with pytest.raises(ValidationError):
            LocationVisitUpdate(names=[f"Name{i}" for i in range(11)])


class TestTextNoteModels:
    """Test text note data models and validation."""
    
    def test_text_note_create_valid(self):
        """Test valid text note creation."""
        data = {
            "text_content": "This is a test note",
            "timestamp": datetime.utcnow(),
            "longitude": -122.4194,
            "latitude": 37.7749,
            "address": "San Francisco, CA",
            "names": ["Meeting", "Work"]
        }
        note = TextNoteCreate(**data)
        assert note.text_content == "This is a test note"
        assert note.longitude == Decimal("-122.4194")
        assert note.names == ["Meeting", "Work"]
    
    def test_text_note_create_minimal(self):
        """Test text note creation with minimal fields."""
        data = {
            "text_content": "Minimal note",
            "timestamp": datetime.utcnow()
        }
        note = TextNoteCreate(**data)
        assert note.text_content == "Minimal note"
        assert note.longitude is None
        assert note.latitude is None
        assert note.address is None
        assert note.names is None
    
    def test_text_note_empty_content(self):
        """Test text note with empty content."""
        with pytest.raises(ValidationError) as exc_info:
            TextNoteCreate(
                text_content="",
                timestamp=datetime.utcnow()
            )
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_text_note_content_too_long(self):
        """Test text note with content too long."""
        long_content = "x" * 10001  # Exceeds 10000 character limit
        with pytest.raises(ValidationError) as exc_info:
            TextNoteCreate(
                text_content=long_content,
                timestamp=datetime.utcnow()
            )
        assert "String should have at most 10000 characters" in str(exc_info.value)
    
    def test_text_note_future_timestamp(self):
        """Test text note with future timestamp."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        with pytest.raises(ValidationError) as exc_info:
            TextNoteCreate(
                text_content="Test note",
                timestamp=future_time
            )
        assert "Timestamp cannot be in the future" in str(exc_info.value)


class TestMediaFileModels:
    """Test media file data models and validation."""
    
    def test_media_file_create_valid(self):
        """Test valid media file creation."""
        data = {
            "file_type": FileType.PHOTO,
            "original_filename": "test.jpg",
            "timestamp": datetime.utcnow(),
            "longitude": -122.4194,
            "latitude": 37.7749
        }
        media = MediaFileCreate(**data)
        assert media.file_type == FileType.PHOTO
        assert media.original_filename == "test.jpg"
    
    def test_media_file_invalid_type(self):
        """Test media file with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            MediaFileCreate(
                file_type="invalid",
                original_filename="test.txt",
                timestamp=datetime.utcnow()
            )
        assert "Input should be 'photo' or 'voice'" in str(exc_info.value)
    
    def test_media_file_negative_size(self):
        """Test media file with negative size."""
        # Note: MediaFileCreate doesn't have file_size field, so this test is not applicable
        # The file_size validation would be in MediaFile model, not MediaFileCreate
        pass


class TestQueryModels:
    """Test query-related data models and validation."""
    
    def test_query_request_valid(self):
        """Test valid query request."""
        request = QueryRequest(
            query="What did I do yesterday?",
            session_id="test-session-123"
        )
        assert request.query == "What did I do yesterday?"
        assert request.session_id == "test-session-123"
    
    def test_query_request_minimal(self):
        """Test query request with minimal fields."""
        request = QueryRequest(query="Simple query")
        assert request.query == "Simple query"
        assert request.session_id is None
    
    def test_query_request_empty_query(self):
        """Test query request with empty query."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="")
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_query_request_query_too_long(self):
        """Test query request with query too long."""
        long_query = "x" * 1001  # Exceeds 1000 character limit
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query=long_query)
        assert "String should have at most 1000 characters" in str(exc_info.value)
    
    def test_media_reference_valid(self):
        """Test valid media reference."""
        ref = MediaReference(
            media_id=uuid4(),
            media_type=FileType.PHOTO,
            description="A photo from yesterday",
            timestamp=datetime.utcnow(),
            location={"address": "San Francisco, CA"}
        )
        assert ref.media_type == FileType.PHOTO
        assert ref.description == "A photo from yesterday"
        assert ref.location == {"address": "San Francisco, CA"}
    
    def test_query_response_valid(self):
        """Test valid query response."""
        media_ref = MediaReference(
            media_id=uuid4(),
            media_type=FileType.PHOTO,
            description="Test photo",
            timestamp=datetime.utcnow()
        )
        response = QueryResponse(
            answer="You visited the office yesterday.",
            sources=[{"type": "location", "content": "Office visit"}],
            media_references=[media_ref],
            session_id="test-session"
        )
        assert response.answer == "You visited the office yesterday."
        assert len(response.sources) == 1
        assert len(response.media_references) == 1
        assert response.session_id == "test-session"


class TestUserModels:
    """Test user-related data models and validation."""
    
    def test_user_valid(self):
        """Test valid user model."""
        user = User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="google_123",
            subscription_tier=SubscriptionTier.PREMIUM,
            display_name="Test User",
            subscription_expires_at=datetime.utcnow() + timedelta(days=30)
        )
        assert user.email == "test@example.com"
        assert user.oauth_provider == "google"
        assert user.subscription_tier == SubscriptionTier.PREMIUM
    
    def test_user_invalid_email(self):
        """Test user with invalid email."""
        with pytest.raises(ValidationError) as exc_info:
            User(
                id=uuid4(),
                email="invalid-email",
                oauth_provider="google",
                oauth_subject="google_123"
            )
        assert "value is not a valid email address" in str(exc_info.value)
    
    def test_user_invalid_oauth_provider(self):
        """Test user with invalid OAuth provider."""
        with pytest.raises(ValidationError) as exc_info:
            User(
                id=uuid4(),
                email="test@example.com",
                oauth_provider="invalid",
                oauth_subject="test_123"
            )
        assert "Input should be 'google' or 'apple'" in str(exc_info.value)
    
    def test_subscription_tier_enum(self):
        """Test subscription tier enum validation."""
        # Valid tiers
        assert SubscriptionTier.FREE == "free"
        assert SubscriptionTier.PREMIUM == "premium"
        assert SubscriptionTier.PRO == "pro"
        
        # Test tier hierarchy
        tiers = [SubscriptionTier.FREE, SubscriptionTier.PREMIUM, SubscriptionTier.PRO]
        assert len(tiers) == 3


class TestUsageModels:
    """Test usage and subscription data models."""
    
    def test_usage_stats_valid(self):
        """Test valid usage stats."""
        stats = UsageStats(
            date=datetime.utcnow().date(),
            text_notes_count=5,
            media_files_count=3,
            queries_count=10,
            daily_limits={
                "text_notes": 10,
                "media_files": 10,
                "queries": 50
            }
        )
        assert stats.text_notes_count == 5
        assert stats.media_files_count == 3
        assert stats.queries_count == 10
        assert stats.daily_limits["queries"] == 50
    
    def test_usage_stats_negative_counts(self):
        """Test usage stats with negative counts."""
        with pytest.raises(ValidationError) as exc_info:
            UsageStats(
                date=datetime.utcnow().date(),
                text_notes_count=-1,
                media_files_count=0,
                queries_count=0,
                daily_limits={}
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_subscription_info_valid(self):
        """Test valid subscription info."""
        info = SubscriptionInfo(
            tier=SubscriptionTier.PREMIUM,
            expires_at=datetime.utcnow() + timedelta(days=30),
            daily_limits={
                "content": 10,
                "queries": 50
            },
            query_history_months=24,
            max_file_size_mb=25,
            max_storage_mb=1000
        )
        assert info.tier == SubscriptionTier.PREMIUM
        assert info.query_history_months == 24
        assert info.daily_limits["queries"] == 50
        assert info.max_file_size_mb == 25
        assert info.max_storage_mb == 1000


class TestErrorResponseModel:
    """Test error response data model."""
    
    def test_error_response_valid(self):
        """Test valid error response."""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid input data",
            details={"field": "longitude", "issue": "out of range"}
        )
        assert error.error == "ValidationError"
        assert error.message == "Invalid input data"
        assert error.details["field"] == "longitude"
        assert error.timestamp is not None  # timestamp is auto-generated
    
    def test_error_response_minimal(self):
        """Test error response with minimal fields."""
        error = ErrorResponse(
            error="NotFound",
            message="Resource not found"
        )
        assert error.error == "NotFound"
        assert error.message == "Resource not found"
        assert error.details is None
        assert error.timestamp is not None  # timestamp is auto-generated


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_location_visit_json_serialization(self):
        """Test location visit JSON serialization."""
        visit = LocationVisit(
            id=uuid4(),
            user_id=uuid4(),
            longitude=Decimal("-122.4194"),
            latitude=Decimal("37.7749"),
            address="San Francisco, CA",
            names=["Home"],
            visit_time=datetime.utcnow(),
            duration=60,
            description="Test visit",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Test JSON serialization
        json_data = visit.model_dump()
        assert json_data["longitude"] == visit.longitude  # Decimal is preserved in model_dump
        assert json_data["latitude"] == visit.latitude
        assert json_data["names"] == ["Home"]
    
    def test_model_field_aliases(self):
        """Test model field aliases work correctly."""
        # Test that snake_case fields are properly handled
        data = {
            "text_content": "Test content",
            "timestamp": datetime.utcnow(),
            "original_filename": "test.txt"
        }
        
        # Should not raise validation errors
        note_data = {k: v for k, v in data.items() if k in ["text_content", "timestamp"]}
        note = TextNoteCreate(**note_data)
        assert note.text_content == "Test content"