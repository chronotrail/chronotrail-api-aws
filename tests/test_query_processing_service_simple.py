"""
Simplified tests for the query processing service focusing on core functionality.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest

from app.models.schemas import (
    ContentType,
    FileType,
    MediaReference,
    QueryRequest,
    QueryResponse,
    SubscriptionTier,
    User,
    VectorSearchResult,
)
from app.services.query import QueryFilter, QueryIntent, QueryProcessingService


class TestQueryProcessingServiceSimple:
    """Simplified tests for query processing service."""

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
            display_name="Test User",
        )

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return AsyncMock()

    def test_parse_query_intent_basic(self, query_service):
        """Test basic query intent parsing."""
        # Test location query
        intent = query_service._parse_query_intent("Where did I go yesterday?")
        assert intent.query_type in ["time", "hybrid"]  # "yesterday" is a time term
        assert intent.has_time_filter is True

        # Test time query
        intent = query_service._parse_query_intent("What did I do last week?")
        assert intent.query_type in ["time", "hybrid"]
        assert intent.has_time_filter is True

        # Test general query
        intent = query_service._parse_query_intent("Tell me about my notes")
        assert intent.query_type in ["general", "hybrid"]

    def test_parse_time_range_today(self, query_service):
        """Test parsing 'today' time range."""
        now = datetime.utcnow()
        result = query_service._parse_time_range(["today"])

        assert result is not None
        start_time = result["start_date"]
        end_time = result["end_date"]

        assert start_time.date() == now.date()
        assert end_time.date() == now.date()
        assert start_time.hour == 0
        assert end_time.hour == 23

    def test_parse_time_range_yesterday(self, query_service):
        """Test parsing 'yesterday' time range."""
        yesterday = datetime.utcnow() - timedelta(days=1)
        result = query_service._parse_time_range(["yesterday"])

        assert result is not None
        start_time = result["start_date"]
        end_time = result["end_date"]

        assert start_time.date() == yesterday.date()
        assert end_time.date() == yesterday.date()

    def test_parse_time_range_last_week(self, query_service):
        """Test parsing 'last week' time range."""
        now = datetime.utcnow()
        result = query_service._parse_time_range(["last week"])

        assert result is not None
        start_time = result["start_date"]
        end_time = result["end_date"]

        # Should be roughly 7 days ago
        days_diff = (now - start_time).days
        assert 6 <= days_diff <= 8  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_create_basic_filter(self, query_service, test_user, mock_db):
        """Test creating a basic query filter."""
        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=[],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False,
        )

        with patch("app.services.query.usage_service") as mock_usage:
            mock_usage.get_query_date_range = AsyncMock(return_value=None)

            query_filter = await query_service._create_query_filter(
                test_user, intent, mock_db
            )

            assert isinstance(query_filter, QueryFilter)
            assert query_filter.min_score == 0.7  # DEFAULT_MIN_SCORE
            assert query_filter.max_results == 10  # DEFAULT_MAX_RESULTS

    def test_rank_and_score_results_basic(self, query_service):
        """Test basic result ranking and scoring."""
        search_results = [
            {
                "type": "text_note",
                "content": "Meeting notes from yesterday",
                "timestamp": datetime.utcnow() - timedelta(days=1),
                "score": 0.8,
                "id": str(uuid4()),
            },
            {
                "type": "location_visit",
                "content": "Visit to office",
                "timestamp": datetime.utcnow() - timedelta(days=2),
                "score": 0.7,  # Increased to be above RELEVANCE_THRESHOLD (0.65)
                "id": str(uuid4()),
            },
        ]

        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=[],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False,
        )

        # Create a mock SearchResult since _rank_and_score_results expects SearchResult, not list
        from app.models.schemas import VectorSearchResult
        from app.services.query import SearchResult

        search_result = SearchResult(
            vector_results=[],
            structured_results=search_results,
            combined_score=0.7,
            result_type="structured",
        )

        ranked_results = query_service._rank_and_score_results(
            search_result, intent, "meeting"
        )

        assert len(ranked_results) == 2
        # Results should be sorted by score (descending)
        assert ranked_results[0]["score"] >= ranked_results[1]["score"]

    def test_rank_results_with_time_boost(self, query_service):
        """Test result ranking with time-based boosting."""
        recent_time = datetime.utcnow() - timedelta(days=1)
        old_time = datetime.utcnow() - timedelta(days=30)

        search_results = [
            {
                "type": "text_note",
                "content": "Old note",
                "timestamp": old_time,
                "score": 0.75,  # Reduced to make boost more effective
                "id": str(uuid4()),
            },
            {
                "type": "text_note",
                "content": "Recent note",
                "timestamp": recent_time,
                "score": 0.7,
                "id": str(uuid4()),
            },
        ]

        intent = QueryIntent(
            query_type="time",
            location_terms=[],
            time_terms=["recent"],
            media_terms=[],
            content_terms=["recent"],
            has_location_filter=False,
            has_time_filter=True,
            has_media_filter=False,
        )

        # Create a mock SearchResult since _rank_and_score_results expects SearchResult, not list
        from app.services.query import SearchResult

        search_result = SearchResult(
            vector_results=[],
            structured_results=search_results,
            combined_score=0.8,
            result_type="structured",
        )

        ranked_results = query_service._rank_and_score_results(
            search_result, intent, "recent"
        )

        # Recent result should be boosted and ranked higher
        assert ranked_results[0]["content"] == "Recent note"

    @pytest.mark.asyncio
    async def test_generate_empty_response(self, query_service, test_user, mock_db):
        """Test generating response when no results are found."""
        query_request = QueryRequest(query="test query")

        with patch("app.services.query.query_session_repository") as mock_session_repo:
            mock_session = Mock()
            mock_session.id = uuid4()
            mock_session_repo.create = AsyncMock(return_value=mock_session)

            response = await query_service._generate_response(
                test_user, query_request, [], mock_db
            )

            assert isinstance(response, QueryResponse)
            assert "couldn't find any relevant information" in response.answer
            assert response.sources == []
            assert response.session_id == str(mock_session.id)

    @pytest.mark.asyncio
    async def test_generate_response_with_results(
        self, query_service, test_user, mock_db
    ):
        """Test generating response with search results."""
        query_request = QueryRequest(query="test query")

        ranked_results = [
            {
                "type": "text_note",
                "content": "Test note content",
                "timestamp": datetime.utcnow(),
                "score": 0.9,
                "id": str(uuid4()),
            }
        ]

        with patch("app.services.query.query_session_repository") as mock_session_repo:
            mock_session = Mock()
            mock_session.id = uuid4()
            mock_session.session_context = {}  # Make it a dict instead of Mock
            mock_session_repo.create = AsyncMock(return_value=mock_session)
            mock_session_repo.update_context = AsyncMock()  # Add this mock

            response = await query_service._generate_response(
                test_user, query_request, ranked_results, mock_db
            )

            assert isinstance(response, QueryResponse)
            assert "I found 1 relevant item" in response.answer
            assert len(response.sources) == 1
            assert response.sources[0]["type"] == "text_note"

    @pytest.mark.asyncio
    async def test_process_query_basic_success(
        self, query_service, mock_embedding_service, test_user, mock_db
    ):
        """Test basic successful query processing."""
        query_request = QueryRequest(query="What are my notes?")

        # Mock all dependencies with proper async mocks
        with patch("app.services.query.usage_service") as mock_usage:
            mock_usage.can_make_query = AsyncMock(return_value=True)
            mock_usage.get_query_date_range = AsyncMock(return_value=None)
            mock_usage.increment_query_usage = AsyncMock()

            with patch(
                "app.services.query.query_session_repository"
            ) as mock_session_repo:
                mock_session = Mock()
                mock_session.id = uuid4()
                mock_session_repo.create = AsyncMock(return_value=mock_session)

                # Mock repository searches
                with patch(
                    "app.services.query.location_visit_repository"
                ) as mock_location_repo:
                    mock_location_repo.search_by_names = AsyncMock(return_value=[])

                    with patch(
                        "app.services.query.text_note_repository"
                    ) as mock_text_repo:
                        mock_text_repo.search_by_names = AsyncMock(return_value=[])
                        mock_text_repo.search_by_content = AsyncMock(return_value=[])

                        with patch(
                            "app.services.query.media_file_repository"
                        ) as mock_media_repo:
                            mock_media_repo.search_by_names = AsyncMock(return_value=[])

                            response = await query_service.process_query(
                                test_user, query_request, mock_db
                            )

                            assert isinstance(response, QueryResponse)
                            assert response.session_id is not None
                            mock_usage.can_make_query.assert_called_once()
                            mock_usage.increment_query_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_query_usage_limit_exceeded(
        self, query_service, test_user, mock_db
    ):
        """Test query processing when usage limit is exceeded."""
        from fastapi import HTTPException

        query_request = QueryRequest(query="test query")

        with patch("app.services.query.usage_service") as mock_usage:
            mock_usage.can_make_query = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await query_service.process_query(test_user, query_request, mock_db)

            # Should raise HTTP 429 error
            assert exc_info.value.status_code == 429

    def test_service_initialization(self, mock_embedding_service):
        """Test that the service initializes correctly."""
        service = QueryProcessingService(embedding_service=mock_embedding_service)
        assert service.embedding_service == mock_embedding_service

    def test_query_intent_structure(self, query_service):
        """Test that QueryIntent has the expected structure."""
        intent = query_service._parse_query_intent("test query")

        assert hasattr(intent, "query_type")
        assert hasattr(intent, "has_time_filter")
        assert hasattr(intent, "time_terms")
        assert hasattr(intent, "location_terms")
        assert hasattr(intent, "media_terms")

        assert isinstance(intent.time_terms, list)
        assert isinstance(intent.location_terms, list)
        assert isinstance(intent.media_terms, list)

    def test_search_result_processing(self, query_service):
        """Test that search results are processed correctly."""
        search_results = [
            {
                "type": "text_note",
                "content": "Test content",
                "timestamp": datetime.utcnow(),
                "score": 0.8,
                "id": str(uuid4()),
            }
        ]

        intent = QueryIntent(
            query_type="general",
            location_terms=[],
            time_terms=[],
            media_terms=[],
            content_terms=[],
            has_location_filter=False,
            has_time_filter=False,
            has_media_filter=False,
        )

        # Create a mock SearchResult since _rank_and_score_results expects SearchResult, not list
        from app.services.query import SearchResult

        search_result = SearchResult(
            vector_results=[],
            structured_results=search_results,
            combined_score=0.8,
            result_type="structured",
        )

        ranked_results = query_service._rank_and_score_results(
            search_result, intent, "test"
        )

        assert len(ranked_results) == 1
        assert ranked_results[0]["type"] == "text_note"
        assert "score" in ranked_results[0]
        assert "timestamp" in ranked_results[0]
