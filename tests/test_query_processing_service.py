"""
Tests for the query processing service.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4, UUID

from app.services.query import (
    QueryProcessingService, QueryIntent, QueryFilter, SearchResult,
    DEFAULT_VECTOR_SEARCH_K, DEFAULT_MIN_SCORE, DEFAULT_MAX_RESULTS
)
from app.models.schemas import (
    User, QueryRequest, QueryResponse, ContentType, VectorSearchResult,
    SubscriptionTier, FileType, MediaReference
)
from app.aws.embedding import EmbeddingService


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service.search_by_text_async = AsyncMock()
    service.convert_search_results_to_model = Mock()
    return service


@pytest.fixture
def query_service(mock_embedding_service):
    """Create a query processing service with mocked dependencies."""
    return QueryProcessingService(embedding_service=mock_embedding_service)


@pytest.fixture
def test_user():
    """Create a test user."""
    return User(
        id=uuid4(),
        email="test@example.com",
        oauth_provider="google",
        oauth_subject="test_subject",
        subscription_tier=SubscriptionTier.FREE,
        display_name="Test User"
    )


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    return Mock()


class TestQueryIntent:
    """Test query intent parsing."""
    
    def test_parse_general_query(self, query_service):
        """Test parsing a general query."""
        query = "What did I do last week?"
        intent = query_service._parse_query_intent(query)
        
        assert intent.query_type == "time"
        assert "last week" in intent.time_terms
        assert intent.has_time_filter is True
        assert intent.has_location_filter is False
        assert intent.has_media_filter is False
    
    def test_parse_location_query(self, query_service):
        """Test parsing a location-based query."""
        query = "What did I do at the coffee shop?"
        intent = query_service._parse_query_intent(query)
        
        assert intent.query_type == "location"
        assert "coffee" in intent.location_terms or "shop" in intent.location_terms
        assert intent.has_location_filter is True
        assert intent.has_time_filter is False
    
    def test_parse_media_query(self, query_service):
        """Test parsing a media-related query."""
        query = "Show me photos from yesterday"
        intent = query_service._parse_query_intent(query)
        
        assert intent.query_type == "media"
        assert "photos" in intent.media_terms
        assert "yesterday" in intent.time_terms
        assert intent.has_media_filter is True
        assert intent.has_time_filter is True
    
    def test_parse_hybrid_query(self, query_service):
        """Test parsing a hybrid query with location and time."""
        query = "What did I do at the gym last Monday?"
        intent = query_service._parse_query_intent(query)
        
        assert intent.query_type == "hybrid"
        assert "gym" in intent.location_terms
        assert "monday" in intent.time_terms
        assert intent.has_location_filter is True
        assert intent.has_time_filter is True
    
    def test_parse_content_terms(self, query_service):
        """Test extraction of content terms."""
        query = "Find my notes about the important meeting"
        intent = query_service._parse_query_intent(query)
        
        assert "notes" in intent.content_terms
        assert "important" in intent.content_terms
        assert "meeting" in intent.content_terms
        # Stop words should be filtered out
        assert "the" not in intent.content_terms
        assert "my" not in intent.content_terms


class TestTimeRangeParsing:
    """Test time range parsing functionality."""
    
    def test_parse_today(self, query_service):
        """Test parsing 'today' time range."""
        time_range = query_service._parse_time_range(["today"])
        
        assert time_range is not None
        assert "start_date" in time_range
        assert "end_date" in time_range
        
        # Should be today's date range
        now = datetime.utcnow()
        assert time_range["start_date"].date() == now.date()
        assert time_range["end_date"].date() == now.date()
    
    def test_parse_yesterday(self, query_service):
        """Test parsing 'yesterday' time range."""
        time_range = query_service._parse_time_range(["yesterday"])
        
        assert time_range is not None
        yesterday = datetime.utcnow() - timedelta(days=1)
        assert time_range["start_date"].date() == yesterday.date()
        assert time_range["end_date"].date() == yesterday.date()
    
    def test_parse_last_week(self, query_service):
        """Test parsing 'last week' time range."""
        time_range = query_service._parse_time_range(["last week"])
        
        assert time_range is not None
        now = datetime.utcnow()
        expected_start = now - timedelta(days=7)
        
        # Should be approximately 7 days ago
        assert abs((time_range["start_date"] - expected_start).total_seconds()) < 3600
        assert time_range["end_date"] <= now
    
    def test_parse_this_month(self, query_service):
        """Test parsing 'this month' time range."""
        time_range = query_service._parse_time_range(["this month"])
        
        assert time_range is not None
        now = datetime.utcnow()
        
        # Should start from first day of current month
        assert time_range["start_date"].day == 1
        assert time_range["start_date"].month == now.month
        assert time_range["end_date"] <= now
    
    def test_parse_empty_time_terms(self, query_service):
        """Test parsing empty time terms."""
        time_range = query_service._parse_time_range([])
        assert time_range is None


class TestQueryFilter:
    """Test query filter creation."""
    
    @pytest.mark.asyncio
    async def test_create_basic_filter(self, query_service, test_user, mock_db):
        """Test creating a basic query filter."""
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
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.get_query_date_range.return_value = None
            
            query_filter = await query_service._create_query_filter(test_user, intent, mock_db)
            
            assert isinstance(query_filter, QueryFilter)
            assert query_filter.min_score == DEFAULT_MIN_SCORE
            assert query_filter.max_results == DEFAULT_MAX_RESULTS
    
    @pytest.mark.asyncio
    async def test_create_filter_with_media_terms(self, query_service, test_user, mock_db):
        """Test creating filter with media terms."""
        intent = QueryIntent(
            query_type="media",
            location_terms=[],
            time_terms=[],
            media_terms=["photo", "voice"],
            content_terms=[],
            has_media_filter=True
        )
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.get_query_date_range.return_value = None
            
            query_filter = await query_service._create_query_filter(test_user, intent, mock_db)
            
            assert query_filter.content_types is not None
            assert ContentType.IMAGE_TEXT in query_filter.content_types
            assert ContentType.IMAGE_DESC in query_filter.content_types
            assert ContentType.VOICE_TRANSCRIPT in query_filter.content_types
    
    @pytest.mark.asyncio
    async def test_create_filter_with_location_terms(self, query_service, test_user, mock_db):
        """Test creating filter with location terms."""
        intent = QueryIntent(
            query_type="location",
            location_terms=["coffee shop", "downtown"],
            time_terms=[],
            media_terms=[],
            content_terms=[],
            has_location_filter=True
        )
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.get_query_date_range.return_value = None
            
            query_filter = await query_service._create_query_filter(test_user, intent, mock_db)
            
            assert query_filter.location_filter is not None
            assert "names" in query_filter.location_filter
            assert "coffee shop" in query_filter.location_filter["names"]
            assert "downtown" in query_filter.location_filter["names"]
    
    @pytest.mark.asyncio
    async def test_create_filter_with_subscription_limits(self, query_service, test_user, mock_db):
        """Test creating filter with subscription-based date limits."""
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=[],
        )
        
        # Mock subscription date range
        start_date = datetime.utcnow() - timedelta(days=90)
        end_date = datetime.utcnow()
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.get_query_date_range.return_value = {
                "start_date": start_date,
                "end_date": end_date
            }
            
            query_filter = await query_service._create_query_filter(test_user, intent, mock_db)
            
            assert query_filter.date_range is not None
            assert query_filter.date_range["start_date"] == start_date
            assert query_filter.date_range["end_date"] == end_date


class TestVectorSearch:
    """Test vector search functionality."""
    
    @pytest.mark.asyncio
    async def test_vector_search_success(self, query_service, mock_embedding_service, test_user, mock_db):
        """Test successful vector search."""
        # Mock vector search results
        mock_results = [
            {
                "id": "doc1",
                "score": 0.9,
                "content_type": "note",
                "content_text": "Test content",
                "timestamp": datetime.utcnow().isoformat(),
                "source_id": str(uuid4()),
                "location": None,
                "metadata": None
            }
        ]
        
        mock_vector_results = [
            VectorSearchResult(
                content_id="doc1",
                content_type=ContentType.NOTE,
                content_text="Test content",
                timestamp=datetime.utcnow(),
                score=0.9,
                source_id=uuid4()
            )
        ]
        
        mock_embedding_service.search_by_text_async.return_value = mock_results
        mock_embedding_service.convert_search_results_to_model.return_value = mock_vector_results
        
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["test"]
        )
        
        query_filter = QueryFilter()
        
        search_result = await query_service._perform_search(
            test_user.id, "test query", intent, query_filter, mock_db
        )
        
        assert isinstance(search_result, SearchResult)
        assert len(search_result.vector_results) == 1
        assert search_result.vector_results[0].content_text == "Test content"
        assert search_result.result_type == "vector"
        assert search_result.combined_score > 0
    
    @pytest.mark.asyncio
    async def test_vector_search_failure(self, query_service, mock_embedding_service, test_user, mock_db):
        """Test vector search failure handling."""
        # Mock search failure
        mock_embedding_service.search_by_text_async.side_effect = Exception("Search failed")
        
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["test"]
        )
        
        query_filter = QueryFilter()
        
        with patch('app.services.query.location_visit_repository') as mock_location_repo:
            mock_location_repo.search_by_names.return_value = []
            
            with patch('app.services.query.text_note_repository') as mock_text_repo:
                mock_text_repo.search_by_names.return_value = []
                mock_text_repo.search_by_content.return_value = []
                
                with patch('app.services.query.media_file_repository') as mock_media_repo:
                    mock_media_repo.search_by_names.return_value = []
                    
                    search_result = await query_service._perform_search(
                        test_user.id, "test query", intent, query_filter, mock_db
                    )
                    
                    # Should handle failure gracefully
                    assert isinstance(search_result, SearchResult)
                    assert len(search_result.vector_results) == 0
                    assert search_result.result_type == "vector"


class TestStructuredSearch:
    """Test structured data search functionality."""
    
    @pytest.mark.asyncio
    async def test_location_search(self, query_service, test_user, mock_db):
        """Test location-based structured search."""
        intent = QueryIntent(
            query_type="location",
            location_terms=["coffee shop"],
            time_terms=[],
            media_terms=[],
            content_terms=[],
            has_location_filter=True
        )
        
        query_filter = QueryFilter()
        
        # Mock location visit
        mock_visit = Mock()
        mock_visit.id = uuid4()
        mock_visit.address = "123 Coffee St"
        mock_visit.visit_time = datetime.utcnow()
        mock_visit.longitude = Decimal("1.0")
        mock_visit.latitude = Decimal("2.0")
        mock_visit.names = ["coffee shop"]
        mock_visit.description = "Great coffee"
        
        with patch('app.services.query.location_visit_repository') as mock_repo:
            mock_repo.search_by_names.return_value = [mock_visit]
            
            results = await query_service._search_structured_data(
                test_user.id, intent, query_filter, mock_db
            )
            
            assert len(results) == 1
            assert results[0]["type"] == "location_visit"
            assert results[0]["location"]["address"] == "123 Coffee St"
            assert results[0]["score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_media_search(self, query_service, test_user, mock_db):
        """Test media-based structured search."""
        intent = QueryIntent(
            query_type="media",
            location_terms=[],
            time_terms=[],
            media_terms=["photo"],
            content_terms=[],
            has_media_filter=True
        )
        
        query_filter = QueryFilter()
        
        # Mock media file
        mock_media = Mock()
        mock_media.id = uuid4()
        mock_media.file_type = FileType.PHOTO
        mock_media.original_filename = "test.jpg"
        mock_media.timestamp = datetime.utcnow()
        mock_media.file_path = "/path/to/file"
        mock_media.longitude = None
        mock_media.latitude = None
        mock_media.address = None
        mock_media.names = None
        
        with patch('app.services.query.media_file_repository') as mock_repo:
            mock_repo.search_by_names.return_value = []
            mock_repo.get_by_user_with_date_range.return_value = []
            mock_repo.get_by_file_type.return_value = [mock_media]
            
            results = await query_service._search_structured_data(
                test_user.id, intent, query_filter, mock_db
            )
            
            assert len(results) == 1
            assert results[0]["type"] == "media_file"
            assert results[0]["file_type"] == FileType.PHOTO
            assert results[0]["score"] == 0.85


class TestResultRanking:
    """Test result ranking and scoring."""
    
    def test_rank_vector_results(self, query_service):
        """Test ranking of vector search results."""
        vector_results = [
            VectorSearchResult(
                content_id="doc1",
                content_type=ContentType.NOTE,
                content_text="Important meeting notes",
                timestamp=datetime.utcnow(),
                score=0.8,
                source_id=uuid4()
            ),
            VectorSearchResult(
                content_id="doc2",
                content_type=ContentType.NOTE,
                content_text="Random thoughts",
                timestamp=datetime.utcnow() - timedelta(days=1),
                score=0.9,
                source_id=uuid4()
            )
        ]
        
        search_results = SearchResult(
            vector_results=vector_results,
            structured_results=[],
            combined_score=0.85,
            result_type="vector"
        )
        
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=["meeting", "important"]
        )
        
        ranked = query_service._rank_and_score_results(
            search_results, intent, "important meeting"
        )
        
        assert len(ranked) == 2
        # First result should have higher score due to content match boost
        assert ranked[0]["score"] > ranked[1]["score"]
        assert "meeting" in ranked[0]["content"].lower()
    
    def test_rank_location_boost(self, query_service):
        """Test location-based score boosting."""
        structured_results = [
            {
                "type": "location_visit",
                "content": "Visited coffee shop",
                "timestamp": datetime.utcnow(),
                "score": 0.8,
                "location": {"address": "Coffee Shop"}
            }
        ]
        
        search_results = SearchResult(
            vector_results=[],
            structured_results=structured_results,
            combined_score=0.8,
            result_type="structured"
        )
        
        intent = QueryIntent(
            query_type="location",
            location_terms=["coffee"],
            time_terms=[],
            media_terms=[],
            content_terms=[]
        )
        
        ranked = query_service._rank_and_score_results(
            search_results, intent, "coffee shop"
        )
        
        assert len(ranked) == 1
        # Score should be boosted for location query
        assert ranked[0]["score"] > 0.8
    
    def test_rank_time_boost(self, query_service):
        """Test time-based score boosting."""
        recent_time = datetime.utcnow() - timedelta(days=2)
        old_time = datetime.utcnow() - timedelta(days=60)
        
        structured_results = [
            {
                "type": "text_note",
                "content": "Recent note",
                "timestamp": recent_time,
                "score": 0.7
            },
            {
                "type": "text_note", 
                "content": "Old note",
                "timestamp": old_time,
                "score": 0.8
            }
        ]
        
        search_results = SearchResult(
            vector_results=[],
            structured_results=structured_results,
            combined_score=0.75,
            result_type="structured"
        )
        
        intent = QueryIntent(
            query_type="time",
            location_terms=[],
            time_terms=["recent"],
            media_terms=[],
            content_terms=[]
        )
        
        ranked = query_service._rank_and_score_results(
            search_results, intent, "recent notes"
        )
        
        assert len(ranked) == 2
        # Recent result should be ranked higher due to time boost
        assert ranked[0]["timestamp"] == recent_time
        assert ranked[0]["score"] > ranked[1]["score"]


class TestResponseGeneration:
    """Test response generation."""
    
    @pytest.mark.asyncio
    async def test_generate_empty_response(self, query_service, test_user, mock_db):
        """Test generating response with no results."""
        query_request = QueryRequest(query="test query")
        
        with patch('app.services.query.query_session_repository') as mock_session_repo:
            mock_session = Mock()
            mock_session.id = uuid4()
            mock_session_repo.create.return_value = mock_session
            
            response = await query_service._generate_response(
                test_user, query_request, [], mock_db
            )
            
            assert isinstance(response, QueryResponse)
            assert "couldn't find" in response.answer.lower()
            assert len(response.sources) == 0
            assert response.media_references is None
            assert response.session_id == str(mock_session.id)
    
    @pytest.mark.asyncio
    async def test_generate_response_with_results(self, query_service, test_user, mock_db):
        """Test generating response with search results."""
        query_request = QueryRequest(query="test query")
        
        ranked_results = [
            {
                "type": "text_note",
                "content": "Test note content",
                "timestamp": datetime.utcnow(),
                "score": 0.9,
                "id": str(uuid4())
            },
            {
                "type": "media_file",
                "content": "Photo file: test.jpg",
                "timestamp": datetime.utcnow(),
                "score": 0.8,
                "file_type": FileType.PHOTO,
                "id": str(uuid4())
            }
        ]
        
        with patch('app.services.query.query_session_repository') as mock_session_repo:
            mock_session = Mock()
            mock_session.id = uuid4()
            mock_session_repo.create.return_value = mock_session
            
            response = await query_service._generate_response(
                test_user, query_request, ranked_results, mock_db
            )
            
            assert isinstance(response, QueryResponse)
            assert "2 relevant items" in response.answer
            assert len(response.sources) == 2
            assert response.media_references is not None
            assert len(response.media_references) == 1
            assert response.media_references[0].media_type == FileType.PHOTO
    
    @pytest.mark.asyncio
    async def test_session_management(self, query_service, test_user, mock_db):
        """Test query session management."""
        existing_session_id = str(uuid4())
        query_request = QueryRequest(query="test query", session_id=existing_session_id)
        
        with patch('app.services.query.query_session_repository') as mock_session_repo:
            mock_session = Mock()
            mock_session.id = UUID(existing_session_id)
            mock_session.session_context = {"query_count": 1}
            mock_session_repo.get_by_user.return_value = mock_session
            mock_session_repo.update_context.return_value = mock_session
            
            response = await query_service._generate_response(
                test_user, query_request, [], mock_db
            )
            
            assert response.session_id == existing_session_id
            mock_session_repo.update_context.assert_called_once()


class TestFullQueryProcessing:
    """Test full query processing workflow."""
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, query_service, mock_embedding_service, test_user, mock_db):
        """Test successful query processing."""
        query_request = QueryRequest(query="What did I do yesterday?")
        
        # Mock all dependencies
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.can_make_query.return_value = True
            mock_usage.get_query_date_range.return_value = None
            mock_usage.increment_query_usage.return_value = None
            
            with patch('app.services.query.query_session_repository') as mock_session_repo:
                mock_session = Mock()
                mock_session.id = uuid4()
                mock_session_repo.create.return_value = mock_session
                
                # Mock vector search
                mock_embedding_service.search_by_text_async.return_value = []
                mock_embedding_service.convert_search_results_to_model.return_value = []
                
                # Mock structured search
                with patch('app.services.query.location_visit_repository') as mock_location_repo:
                    mock_location_repo.search_by_names.return_value = []
                    
                    with patch('app.services.query.text_note_repository') as mock_text_repo:
                        mock_text_repo.search_by_names.return_value = []
                        mock_text_repo.search_by_content.return_value = []
                        
                        with patch('app.services.query.media_file_repository') as mock_media_repo:
                            mock_media_repo.search_by_names.return_value = []
                            
                            response = await query_service.process_query(
                                test_user, query_request, mock_db
                            )
                            
                            assert isinstance(response, QueryResponse)
                            assert response.session_id is not None
                            mock_usage.can_make_query.assert_called_once_with(test_user.id, mock_db)
                            mock_usage.increment_query_usage.assert_called_once_with(test_user.id, mock_db)
    
    @pytest.mark.asyncio
    async def test_process_query_usage_limit_exceeded(self, query_service, test_user, mock_db):
        """Test query processing when usage limit is exceeded."""
        query_request = QueryRequest(query="test query")
        
        with patch('app.services.query.usage_service') as mock_usage:
            mock_usage.can_make_query.return_value = False
            
            with pytest.raises(Exception) as exc_info:
                await query_service.process_query(test_user, query_request, mock_db)
            
            # Should raise HTTP 429 error
            assert "429" in str(exc_info.value) or "limit exceeded" in str(exc_info.value).lower()