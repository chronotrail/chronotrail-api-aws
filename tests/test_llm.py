"""
Tests for the LLM service.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest

from app.aws.exceptions import BedrockError
from app.aws.llm import DEFAULT_MODEL_ID, FALLBACK_MODEL_ID, LLMService
from app.models.schemas import FileType


@pytest.fixture
def llm_service():
    """Create an LLM service for testing."""
    return LLMService()


@pytest.fixture
def mock_context_items():
    """Create mock context items for testing."""
    return [
        {
            "type": "location_visit",
            "content": "Visited Coffee Shop",
            "timestamp": datetime.utcnow(),
            "location": {"address": "123 Coffee St", "names": ["Coffee Shop"]},
            "description": "Had a great latte",
        },
        {
            "type": "text_note",
            "content": "Remember to buy groceries",
            "timestamp": datetime.utcnow(),
            "id": str(uuid4()),
        },
        {
            "type": "media_file",
            "content": "Photo of sunset",
            "timestamp": datetime.utcnow(),
            "file_type": FileType.PHOTO,
            "id": str(uuid4()),
        },
    ]


class TestLLMService:
    """Test the LLM service."""

    def test_init(self):
        """Test initialization with default and custom model IDs."""
        # Test with default model ID
        service = LLMService()
        assert service.model_id == DEFAULT_MODEL_ID

        # Test with custom model ID
        custom_model = "custom.model-id"
        service = LLMService(model_id=custom_model)
        assert service.model_id == custom_model

    def test_format_context_for_claude(self, llm_service, mock_context_items):
        """Test formatting context for Claude prompt."""
        formatted_context = llm_service._format_context_for_claude(mock_context_items)

        # Check that all items are included
        assert "Location Visit" in formatted_context
        assert "123 Coffee St" in formatted_context
        assert "Had a great latte" in formatted_context
        assert "Text Note" in formatted_context
        assert "Remember to buy groceries" in formatted_context
        assert "Photo" in formatted_context
        assert "Photo of sunset" in formatted_context

        # Check formatting
        assert formatted_context.startswith("1. Location Visit")

        # Test with empty context
        empty_context = llm_service._format_context_for_claude([])
        assert empty_context == "No relevant information found."

    def test_create_claude_prompt(self, llm_service, mock_context_items):
        """Test creating a Claude prompt."""
        query = "What did I do today?"
        prompt = llm_service._create_claude_prompt(query, mock_context_items)

        # Check that the prompt contains all required sections
        assert "<instructions>" in prompt
        assert "</instructions>" in prompt
        assert "<context>" in prompt
        assert "</context>" in prompt
        assert "<question>" in prompt
        assert "</question>" in prompt
        assert "<answer>" in prompt

        # Check that the query is included
        assert query in prompt

        # Check that context is included
        assert "Location Visit" in prompt
        assert "Text Note" in prompt
        assert "Photo" in prompt

    def test_create_anthropic_payload(self, llm_service):
        """Test creating an Anthropic payload."""
        prompt = "Test prompt"
        payload = llm_service._create_anthropic_payload(prompt)

        assert payload["prompt"] == prompt
        assert payload["max_tokens_to_sample"] > 0
        assert 0 <= payload["temperature"] <= 1
        assert 0 <= payload["top_p"] <= 1
        assert "</answer>" in payload["stop_sequences"]

    def test_create_amazon_payload(self, llm_service):
        """Test creating an Amazon payload."""
        prompt = "Test prompt"
        payload = llm_service._create_amazon_payload(prompt)

        assert payload["inputText"] == prompt
        assert payload["textGenerationConfig"]["maxTokenCount"] > 0
        assert 0 <= payload["textGenerationConfig"]["temperature"] <= 1
        assert 0 <= payload["textGenerationConfig"]["topP"] <= 1
        assert "</answer>" in payload["textGenerationConfig"]["stopSequences"]

    @patch("app.aws.llm.get_bedrock_client")
    def test_generate_response_anthropic(
        self, mock_get_client, llm_service, mock_context_items
    ):
        """Test generating a response with an Anthropic model."""
        # Mock the Bedrock client
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock the response
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(
            {"completion": "This is a test response."}
        )
        mock_client.invoke_model.return_value = mock_response

        # Test generating a response
        query = "What did I do today?"
        response = llm_service.generate_response(query, mock_context_items)

        # Check that the response is correct
        assert response == "This is a test response."

        # Check that the client was called with the correct parameters
        mock_client.invoke_model.assert_called_once()
        args, kwargs = mock_client.invoke_model.call_args
        assert kwargs["modelId"] == DEFAULT_MODEL_ID
        assert kwargs["contentType"] == "application/json"
        assert kwargs["accept"] == "application/json"
        assert "body" in kwargs

    @patch("app.aws.llm.get_bedrock_client")
    def test_generate_response_amazon(self, mock_get_client, mock_context_items):
        """Test generating a response with an Amazon model."""
        # Create an LLM service with an Amazon model
        service = LLMService(model_id="amazon.titan-text-express-v1")

        # Mock the Bedrock client
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock the response
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(
            {"results": [{"outputText": "This is a test response."}]}
        )
        mock_client.invoke_model.return_value = mock_response

        # Test generating a response
        query = "What did I do today?"
        response = service.generate_response(query, mock_context_items)

        # Check that the response is correct
        assert response == "This is a test response."

        # Check that the client was called with the correct parameters
        mock_client.invoke_model.assert_called_once()
        args, kwargs = mock_client.invoke_model.call_args
        assert kwargs["modelId"] == "amazon.titan-text-express-v1"
        assert kwargs["contentType"] == "application/json"
        assert kwargs["accept"] == "application/json"
        assert "body" in kwargs

    @patch("app.aws.llm.get_bedrock_client")
    def test_generate_response_error(
        self, mock_get_client, llm_service, mock_context_items
    ):
        """Test error handling when generating a response."""
        # Mock the Bedrock client to raise an exception
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Test error")

        # Test generating a response with error
        query = "What did I do today?"
        response = llm_service.generate_response(query, mock_context_items)

        # Check that a fallback response was generated
        assert "I found" in response
        assert "location visits" in response
        assert "text notes" in response
        assert "photos" in response

    @patch("app.aws.llm.get_bedrock_client")
    def test_generate_response_fallback_model(
        self, mock_get_client, llm_service, mock_context_items
    ):
        """Test fallback to another model when primary model fails."""
        # Mock the Bedrock client to raise an exception for the first call
        # and succeed for the second call (fallback model)
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # First call raises exception, second call (fallback model) succeeds
        mock_response = {"body": MagicMock()}
        mock_response["body"].read.return_value = json.dumps(
            {"completion": "This is a fallback model response."}
        )
        mock_client.invoke_model.side_effect = [Exception("Test error"), mock_response]

        # Test generating a response
        query = "What did I do today?"

        # This should use the fallback model
        response = llm_service.generate_response(query, mock_context_items)

        # Check that the fallback model response was used
        assert response == "This is a fallback model response."

    def test_generate_fallback_response(self, llm_service, mock_context_items):
        """Test generating a fallback response."""
        # Test with location query
        query = "Where did I go today?"
        response = llm_service._generate_fallback_response(query, mock_context_items)

        assert "I found 3 relevant items" in response
        assert "location visits" in response
        assert "text notes" in response
        assert "photos" in response
        assert "123 Coffee St" in response

        # Test with time query
        query = "When did I take notes?"
        response = llm_service._generate_fallback_response(query, mock_context_items)

        assert "I found 3 relevant items" in response
        assert "occurred on" in response

        # Test with empty context
        response = llm_service._generate_fallback_response(query, [])
        assert "couldn't find any relevant information" in response


@pytest.mark.asyncio
class TestLLMServiceAsync:
    """Test the async methods of the LLM service."""

    @pytest.mark.asyncio
    @patch("app.aws.llm.get_async_bedrock_client")
    async def test_generate_response_async(
        self, mock_get_client, llm_service, mock_context_items
    ):
        """Test generating a response asynchronously."""
        # Mock the async Bedrock client
        mock_client = AsyncMock()

        # Mock the response with async read method
        mock_body = AsyncMock()
        mock_body.read.return_value = json.dumps(
            {"completion": "This is an async test response."}
        )

        mock_response = {"body": mock_body}
        mock_client.invoke_model.return_value = mock_response

        # Mock the context manager factory to return an async context manager
        async def mock_context_manager():
            return mock_client

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_client
        mock_cm.__aexit__.return_value = None

        # Mock get_async_bedrock_client to return a function that returns the context manager
        mock_get_client.return_value = lambda: mock_cm

        # Test generating a response asynchronously
        query = "What did I do today?"
        response = await llm_service.generate_response_async(query, mock_context_items)

        # Check that the response is correct
        assert response == "This is an async test response."

        # Check that the client was called with the correct parameters
        mock_client.invoke_model.assert_called_once()
        args, kwargs = mock_client.invoke_model.call_args
        assert kwargs["modelId"] == DEFAULT_MODEL_ID
        assert kwargs["contentType"] == "application/json"
        assert kwargs["accept"] == "application/json"
        assert "body" in kwargs

    @pytest.mark.asyncio
    @patch("app.aws.llm.get_async_bedrock_client")
    async def test_generate_response_async_error(
        self, mock_get_client, llm_service, mock_context_items
    ):
        """Test error handling when generating a response asynchronously."""
        # Mock the async Bedrock client to raise an exception
        mock_client_cm = MagicMock()
        mock_client = MagicMock()
        mock_client_cm.__aenter__.return_value = mock_client
        mock_get_client.return_value = mock_client_cm
        mock_client.invoke_model.side_effect = Exception("Test error")

        # Test generating a response asynchronously with error
        query = "What did I do today?"
        response = await llm_service.generate_response_async(query, mock_context_items)

        # Check that a fallback response was generated
        assert "I found" in response
        assert "location visits" in response
        assert "text notes" in response
        assert "photos" in response
