"""
Tests for the query API endpoints.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import status
from httpx import AsyncClient

from app.models.schemas import QueryRequest, QueryResponse, SubscriptionTier, User
from app.services.query import QueryProcessingService


@pytest.fixture
def mock_query_service():
    """Create a mock query processing service."""
    service = Mock(spec=QueryProcessingService)
    service.process_query = AsyncMock()
    return service


@pytest.mark.asyncio
class TestQueryEndpoint:
    """Test the query API endpoint."""

    @pytest.mark.asyncio
    async def test_query_timeline_success(
        self, client: AsyncClient, mock_auth_user, mock_db
    ):
        """Test successful query processing."""
        # Mock the query processing service
        with patch(
            "app.api.v1.endpoints.query.query_processing_service"
        ) as mock_service:
            # Mock response
            mock_response = QueryResponse(
                answer="This is a test response.",
                sources=[{"type": "text_note", "content": "Test content"}],
                media_references=None,
                session_id=str(uuid4()),
            )
            mock_service.process_query.return_value = mock_response

            # Mock usage validation
            with patch(
                "app.api.v1.endpoints.query.validate_query_usage", return_value=True
            ):
                # Make request
                response = await client.post(
                    "/api/v1/query/",
                    json={"query": "What did I do yesterday?"},
                    headers={"Authorization": f"Bearer {mock_auth_user.token}"},
                )

                # Check response
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["answer"] == "This is a test response."
                assert len(data["sources"]) == 1
                assert data["media_references"] is None
                assert "session_id" in data

                # Check that the service was called with the correct parameters
                mock_service.process_query.assert_called_once()
                args, kwargs = mock_service.process_query.call_args
                assert kwargs["query_request"].query == "What did I do yesterday?"

    @pytest.mark.asyncio
    async def test_query_timeline_with_session(
        self, client: AsyncClient, mock_auth_user, mock_db
    ):
        """Test query processing with session ID."""
        # Mock the query processing service
        with patch(
            "app.api.v1.endpoints.query.query_processing_service"
        ) as mock_service:
            # Mock response
            session_id = str(uuid4())
            mock_response = QueryResponse(
                answer="This is a test response with session.",
                sources=[{"type": "text_note", "content": "Test content"}],
                media_references=None,
                session_id=session_id,
            )
            mock_service.process_query.return_value = mock_response

            # Mock usage validation
            with patch(
                "app.api.v1.endpoints.query.validate_query_usage", return_value=True
            ):
                # Make request
                response = await client.post(
                    "/api/v1/query/",
                    json={
                        "query": "What did I do yesterday?",
                        "session_id": session_id,
                    },
                    headers={"Authorization": f"Bearer {mock_auth_user.token}"},
                )

                # Check response
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["answer"] == "This is a test response with session."
                assert data["session_id"] == session_id

                # Check that the service was called with the correct parameters
                mock_service.process_query.assert_called_once()
                args, kwargs = mock_service.process_query.call_args
                assert kwargs["query_request"].query == "What did I do yesterday?"
                assert kwargs["query_request"].session_id == session_id

    @pytest.mark.asyncio
    async def test_query_timeline_usage_limit_exceeded(
        self, client: AsyncClient, mock_auth_user, mock_db
    ):
        """Test query processing when usage limit is exceeded."""
        # Mock usage validation to raise an exception
        with patch("app.api.v1.endpoints.query.validate_query_usage") as mock_validate:
            mock_validate.side_effect = Exception("Usage limit exceeded")

            # Make request
            response = await client.post(
                "/api/v1/query/",
                json={"query": "What did I do yesterday?"},
                headers={"Authorization": f"Bearer {mock_auth_user.token}"},
            )

            # Check response
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_query_timeline_unauthorized(self, client: AsyncClient):
        """Test query processing without authentication."""
        # Make request without authentication
        response = await client.post(
            "/api/v1/query/", json={"query": "What did I do yesterday?"}
        )

        # Check response
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_query_timeline_service_error(
        self, client: AsyncClient, mock_auth_user, mock_db
    ):
        """Test query processing when service raises an error."""
        # Mock the query processing service to raise an exception
        with patch(
            "app.api.v1.endpoints.query.query_processing_service"
        ) as mock_service:
            mock_service.process_query.side_effect = Exception("Service error")

            # Mock usage validation
            with patch(
                "app.api.v1.endpoints.query.validate_query_usage", return_value=True
            ):
                # Make request
                response = await client.post(
                    "/api/v1/query/",
                    json={"query": "What did I do yesterday?"},
                    headers={"Authorization": f"Bearer {mock_auth_user.token}"},
                )

                # Check response
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                data = response.json()
                assert "detail" in data
                assert data["detail"] == "Failed to process query"
