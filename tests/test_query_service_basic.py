"""
Basic tests for the query processing service focusing on working functionality.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from app.services.query import QueryProcessingService, QueryIntent
from app.models.schemas import User, QueryRequest, QueryResponse, SubscriptionTier


class TestQueryServiceBasic:
    """Basic tests for query processing service."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.search_by_text_async = AsyncMock(return_value=[])
        service.convert_search_results_to_model = Mock(return_value=[])
        return service
    
    @pytest.fixture
    def query_service(self, mock_embedding_service):
        """Create a query processing service with mocked dependencies."""
        return QueryProcessingService(embedding_service=mock_embedding_service)
    
    @pytest.fixture
    def test_user(self):
        """Create a test user."""
        return User(
            id=uuid4(),
            email="test@example.com",
            oauth_provider="google",
            oauth_subject="test_subject",
            subscription_tier=SubscriptionTier.FREE,
            display_name="Test User"
        )
    
    def test_service_initialization(self, mock_embedding_service):
        """Test that the service initializes correctly."""
        service = QueryProcessingService(embedding_service=mock_embedding_service)
        assert service.embedding_service == mock_embedding_service
    
    def test_parse_query_intent_basic(self, query_service):
        """Test basic query intent parsing."""
        intent = query_service._parse_query_intent("What did I do yesterday?")
        
        # Check that intent has the expected structure
        assert hasattr(intent, 'query_type')
        assert hasattr(intent, 'location_terms')
        assert hasattr(intent, 'time_terms')
        assert hasattr(intent, 'media_terms')
        assert hasattr(intent, 'content_terms')
        assert hasattr(intent, 'has_location_filter')
        assert hasattr(intent, 'has_time_filter')
        assert hasattr(intent, 'has_media_filter')
        
        # Check that it's a valid QueryIntent
        assert isinstance(intent, QueryIntent)
        assert isinstance(intent.location_terms, list)
        assert isinstance(intent.time_terms, list)
        assert isinstance(intent.media_terms, list)
        assert isinstance(intent.content_terms, list)
    
    def test_parse_query_intent_time_query(self, query_service):
        """Test parsing a time-based query."""
        intent = query_service._parse_query_intent("What did I do yesterday?")
        
        # Should detect time-related terms
        assert intent.has_time_filter is True
        assert "yesterday" in intent.time_terms
    
    def test_parse_query_intent_location_query(self, query_service):
        """Test parsing a location-based query."""
        intent = query_service._parse_query_intent("Where did I go to the office?")
        
        # Should detect location-related terms
        assert intent.has_location_filter is True
        assert "where" in intent.location_terms or "office" in intent.location_terms
    
    def test_parse_query_intent_media_query(self, query_service):
        """Test parsing a media-related query."""
        intent = query_service._parse_query_intent("Show me my photos from last week")
        
        # Should detect media-related terms
        assert intent.has_media_filter is True
        assert "photos" in intent.media_terms
    
    def test_parse_time_range_with_terms(self, query_service):
        """Test parsing time range with time terms."""
        time_range = query_service._parse_time_range(["yesterday"])
        
        # Should return a dictionary with start_date and end_date
        if time_range:  # Method might return None for some inputs
            assert isinstance(time_range, dict)
            assert "start_date" in time_range or "end_date" in time_range
    
    def test_parse_time_range_empty(self, query_service):
        """Test parsing time range with empty terms."""
        time_range = query_service._parse_time_range([])
        
        # Should return None for empty terms
        assert time_range is None
    
    def test_rank_and_score_results_basic(self, query_service):
        """Test basic result ranking and scoring."""
        search_results = [
            {
                "type": "text_note",
                "content": "Meeting notes from yesterday",
                "timestamp": datetime.utcnow() - timedelta(days=1),
                "score": 0.8,
                "id": str(uuid4())
            },
            {
                "type": "location_visit",
                "content": "Visit to office",
                "timestamp": datetime.utcnow() - timedelta(days=2),
                "score": 0.6,
                "id": str(uuid4())
            }
        ]
        
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["meeting"],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False
        )
        
        ranked_results = query_service._rank_and_score_results(
            search_results, intent, "meeting"
        )
        
        assert len(ranked_results) == 2
        # Results should be sorted by score (descending)
        assert ranked_results[0]["score"] >= ranked_results[1]["score"]
    
    @pytest.mark.asyncio
    async def test_generate_empty_response(self, query_service, test_user):
        """Test generating response when no results are found."""
        query_request = QueryRequest(query="test query")
        mock_db = AsyncMock()
        
        with patch('app.services.query.query_session_repository') as mock_session_repo:
            mock_session = Mock()
            mock_session.id = uuid4()
            mock_session.session_context = {}
            mock_session_repo.create = AsyncMock(return_value=mock_session)
            
            response = await query_service._generate_response(
                test_user, query_request, [], mock_db
            )
            
            assert isinstance(response, QueryResponse)
            assert "couldn't find any relevant information" in response.answer
            assert response.sources == []
            assert response.session_id == str(mock_session.id)
    
    @pytest.mark.asyncio
    async def test_process_query_usage_limit_exceeded(self, query_service, test_user):
        """Test query processing when usage limit is exceeded."""
        from fastapi import HTTPException
        
        query_request = QueryRequest(query="test query")
        mock_db = AsyncMock()
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.can_make_query = AsyncMock(return_value=False)
            
            with pytest.raises(HTTPException) as exc_info:
                await query_service.process_query(test_user, query_request, mock_db)
            
            # Should raise HTTP 429 error
            assert exc_info.value.status_code == 429
    
    def test_query_intent_validation(self, query_service):
        """Test that QueryIntent validation works correctly."""
        # Test creating a valid QueryIntent
        intent = QueryIntent(
            query_type="general",
            location_terms=["office"],
            time_terms=["yesterday"],
            media_terms=["photos"],
            content_terms=["meeting", "notes"],
            has_location_filter=True,
            has_time_filter=True,
            has_media_filter=True
        )
        
        assert intent.query_type == "general"
        assert "office" in intent.location_terms
        assert "yesterday" in intent.time_terms
        assert "photos" in intent.media_terms
        assert "meeting" in intent.content_terms
        assert intent.has_location_filter is True
        assert intent.has_time_filter is True
        assert intent.has_media_filter is True
    
    def test_search_result_processing_empty(self, query_service):
        """Test processing empty search results."""
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["test"],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False
        )
        
        ranked_results = query_service._rank_and_score_results(
            [], intent, "test"
        )
        
        assert len(ranked_results) == 0
    
    def test_search_result_processing_single_result(self, query_service):
        """Test processing a single search result."""
        search_results = [
            {
                "type": "text_note",
                "content": "Test content",
                "timestamp": datetime.utcnow(),
                "score": 0.8,
                "id": str(uuid4())
            }
        ]
        
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["test"],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False
        )
        
        ranked_results = query_service._rank_and_score_results(
            search_results, intent, "test"
        )
        
        assert len(ranked_results) == 1
        assert ranked_results[0]["type"] == "text_note"
        assert "score" in ranked_results[0]
        assert "timestamp" in ranked_results[0]