"""
Tests for embedding and vector storage service.
"""
import os
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from datetime import datetime
from uuid import UUID, uuid4

from botocore.exceptions import ClientError
from botocore.stub import Stubber
from opensearchpy import OpenSearch

from app.aws.embedding import (
    EmbeddingService,
    embedding_service,
    EMBEDDING_MODEL_ID,
    EMBEDDING_DIMENSION,
    OPENSEARCH_INDEX_NAME,
    OPENSEARCH_MAPPING
)
from app.aws.exceptions import BedrockError, OpenSearchError
from app.models.schemas import ContentType, VectorSearchResult


class TestEmbeddingService(unittest.TestCase):
    """Test embedding and vector storage service."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test instance with a unique index name for testing
        self.test_index = f"test-index-{uuid4()}"
        self.service = EmbeddingService(index_name=self.test_index)
        
        # Sample data for testing
        self.sample_text = "This is a test document for embedding generation."
        self.sample_embedding = [0.1] * EMBEDDING_DIMENSION
        self.sample_user_id = str(uuid4())
        self.sample_content_type = ContentType.NOTE
        self.sample_timestamp = datetime.utcnow()
        self.sample_source_id = str(uuid4())
        self.sample_location = {
            "longitude": 10.0,
            "latitude": 20.0,
            "address": "123 Test St",
            "names": ["Home", "Office"]
        }
        self.sample_metadata = {
            "file_type": "text",
            "original_filename": "test.txt"
        }
    
    @patch("app.aws.embedding.get_bedrock_embeddings_client")
    def test_generate_embeddings(self, mock_get_bedrock_client):
        """Test embedding generation."""
        # Set up mock
        mock_client = MagicMock()
        mock_response = {
            "body": MagicMock()
        }
        mock_response["body"].read.return_value = json.dumps({
            "embedding": self.sample_embedding
        })
        mock_client.invoke_model.return_value = mock_response
        mock_get_bedrock_client.return_value = mock_client
        
        # Generate embeddings
        embeddings = self.service.generate_embeddings(self.sample_text)
        
        # Verify embeddings
        self.assertEqual(len(embeddings), EMBEDDING_DIMENSION)
        self.assertEqual(embeddings, self.sample_embedding)
        
        # Verify client was called correctly
        mock_client.invoke_model.assert_called_once_with(
            modelId=EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": self.sample_text})
        )
    
    @patch("app.aws.embedding.get_bedrock_embeddings_client")
    def test_generate_embeddings_error(self, mock_get_bedrock_client):
        """Test embedding generation error handling."""
        # Set up mock to raise error
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ModelError", "Message": "Model error"}},
            "InvokeModel"
        )
        mock_get_bedrock_client.return_value = mock_client
        
        # Verify error is raised
        with self.assertRaises(BedrockError):
            self.service.generate_embeddings(self.sample_text)
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_ensure_index_exists_new(self, mock_get_opensearch_client):
        """Test index creation when it doesn't exist."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = False
        mock_get_opensearch_client.return_value = mock_client
        
        # Ensure index exists
        result = self.service.ensure_index_exists()
        
        # Verify result
        self.assertTrue(result)
        
        # Verify client was called correctly
        mock_client.indices.exists.assert_called_once_with(index=self.test_index)
        mock_client.indices.create.assert_called_once_with(
            index=self.test_index,
            body=OPENSEARCH_MAPPING
        )
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_ensure_index_exists_already(self, mock_get_opensearch_client):
        """Test index check when it already exists."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.indices.exists.return_value = True
        mock_get_opensearch_client.return_value = mock_client
        
        # Ensure index exists
        result = self.service.ensure_index_exists()
        
        # Verify result
        self.assertTrue(result)
        
        # Verify client was called correctly
        mock_client.indices.exists.assert_called_once_with(index=self.test_index)
        mock_client.indices.create.assert_not_called()
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_index_document(self, mock_get_opensearch_client):
        """Test document indexing."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.index.return_value = {"_id": "test-id", "result": "created"}
        mock_get_opensearch_client.return_value = mock_client
        
        # Mock ensure_index_exists
        self.service.ensure_index_exists = MagicMock(return_value=True)
        
        # Index document
        doc_id = self.service.index_document(
            user_id=self.sample_user_id,
            content_type=self.sample_content_type,
            content_text=self.sample_text,
            content_vector=self.sample_embedding,
            timestamp=self.sample_timestamp,
            source_id=self.sample_source_id,
            location=self.sample_location,
            metadata=self.sample_metadata
        )
        
        # Verify document ID is returned
        self.assertIsNotNone(doc_id)
        self.assertTrue(isinstance(doc_id, str))
        
        # Verify ensure_index_exists was called
        self.service.ensure_index_exists.assert_called_once()
        
        # Verify client was called correctly
        mock_client.index.assert_called_once()
        call_args = mock_client.index.call_args[1]
        self.assertEqual(call_args["index"], self.test_index)
        self.assertTrue("body" in call_args)
        self.assertEqual(call_args["refresh"], True)
        
        # Verify document body
        body = call_args["body"]
        self.assertEqual(body["user_id"], self.sample_user_id)
        self.assertEqual(body["content_type"], self.sample_content_type.value)
        self.assertEqual(body["content_text"], self.sample_text)
        self.assertEqual(body["content_vector"], self.sample_embedding)
        self.assertEqual(body["source_id"], self.sample_source_id)
        self.assertEqual(body["location"], self.sample_location)
        self.assertEqual(body["metadata"], self.sample_metadata)
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_search_by_vector(self, mock_get_opensearch_client):
        """Test vector search."""
        # Set up mock
        mock_client = MagicMock()
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 0.95,
                        "_source": {
                            "content_type": "note",
                            "content_text": "Test note",
                            "timestamp": self.sample_timestamp.isoformat(),
                            "source_id": self.sample_source_id,
                            "location": self.sample_location,
                            "metadata": self.sample_metadata
                        }
                    },
                    {
                        "_id": "doc2",
                        "_score": 0.85,
                        "_source": {
                            "content_type": "location_desc",
                            "content_text": "Test location",
                            "timestamp": self.sample_timestamp.isoformat(),
                            "source_id": str(uuid4()),
                            "location": self.sample_location,
                            "metadata": None
                        }
                    }
                ]
            }
        }
        mock_client.search.return_value = mock_response
        mock_get_opensearch_client.return_value = mock_client
        
        # Search by vector
        results = self.service.search_by_vector(
            user_id=self.sample_user_id,
            vector=self.sample_embedding,
            k=5,
            content_types=[ContentType.NOTE, ContentType.LOCATION_DESC],
            min_score=0.8
        )
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "doc1")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["content_type"], "note")
        self.assertEqual(results[0]["content_text"], "Test note")
        
        # Verify client was called correctly
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        self.assertEqual(call_args["index"], self.test_index)
        
        # Verify query structure
        query = call_args["body"]["query"]["bool"]
        self.assertTrue("must" in query)
        self.assertTrue("should" in query)
        
        # Verify user isolation
        user_term = next((term for term in query["must"] if "term" in term and "user_id" in term["term"]), None)
        self.assertIsNotNone(user_term)
        self.assertEqual(user_term["term"]["user_id"], self.sample_user_id)
        
        # Verify content type filter
        content_filter = next((term for term in query["must"] if "terms" in term and "content_type" in term["terms"]), None)
        self.assertIsNotNone(content_filter)
        self.assertEqual(set(content_filter["terms"]["content_type"]), {"note", "location_desc"})
    
    @patch("app.aws.embedding.EmbeddingService.generate_embeddings")
    @patch("app.aws.embedding.EmbeddingService.search_by_vector")
    def test_search_by_text(self, mock_search_by_vector, mock_generate_embeddings):
        """Test text search."""
        # Set up mocks
        mock_generate_embeddings.return_value = self.sample_embedding
        mock_search_by_vector.return_value = [{"id": "doc1", "score": 0.95}]
        
        # Search by text
        results = self.service.search_by_text(
            user_id=self.sample_user_id,
            text=self.sample_text,
            k=5,
            content_types=[ContentType.NOTE]
        )
        
        # Verify results
        self.assertEqual(results, [{"id": "doc1", "score": 0.95}])
        
        # Verify mocks were called correctly
        mock_generate_embeddings.assert_called_once_with(self.sample_text)
        mock_search_by_vector.assert_called_once_with(
            user_id=self.sample_user_id,
            vector=self.sample_embedding,
            k=5,
            content_types=[ContentType.NOTE],
            date_range=None,
            location_filter=None
        )
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_delete_document(self, mock_get_opensearch_client):
        """Test document deletion."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.delete.return_value = {"result": "deleted"}
        mock_get_opensearch_client.return_value = mock_client
        
        # Delete document
        result = self.service.delete_document("doc1")
        
        # Verify result
        self.assertTrue(result)
        
        # Verify client was called correctly
        mock_client.delete.assert_called_once_with(
            index=self.test_index,
            id="doc1",
            refresh=True
        )
    
    @patch("app.aws.embedding.get_opensearch_client")
    def test_delete_user_documents(self, mock_get_opensearch_client):
        """Test user documents deletion."""
        # Set up mock
        mock_client = MagicMock()
        mock_client.delete_by_query.return_value = {"deleted": 5}
        mock_get_opensearch_client.return_value = mock_client
        
        # Delete user documents
        result = self.service.delete_user_documents(self.sample_user_id)
        
        # Verify result
        self.assertEqual(result, 5)
        
        # Verify client was called correctly
        mock_client.delete_by_query.assert_called_once()
        call_args = mock_client.delete_by_query.call_args[1]
        self.assertEqual(call_args["index"], self.test_index)
        self.assertEqual(call_args["refresh"], True)
        
        # Verify query
        query = call_args["body"]["query"]["term"]
        self.assertEqual(query["user_id"], self.sample_user_id)
    
    def test_convert_search_results_to_model(self):
        """Test conversion of search results to models."""
        # Sample search results
        search_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "content_type": "note",
                "content_text": "Test note",
                "timestamp": self.sample_timestamp.isoformat(),
                "source_id": self.sample_source_id,
                "location": self.sample_location
            },
            {
                "id": "doc2",
                "score": 0.85,
                "content_type": "location_desc",
                "content_text": "Test location",
                "timestamp": self.sample_timestamp.isoformat(),
                "source_id": None,
                "location": None
            }
        ]
        
        # Convert to models
        models = self.service.convert_search_results_to_model(search_results)
        
        # Verify models
        self.assertEqual(len(models), 2)
        self.assertIsInstance(models[0], VectorSearchResult)
        self.assertEqual(models[0].content_id, "doc1")
        self.assertEqual(models[0].content_type, "note")
        self.assertEqual(models[0].content_text, "Test note")
        self.assertEqual(models[0].score, 0.95)
        self.assertEqual(str(models[0].source_id), self.sample_source_id)
        self.assertEqual(models[0].location, self.sample_location)
        
        self.assertIsInstance(models[1], VectorSearchResult)
        self.assertEqual(models[1].content_id, "doc2")
        self.assertEqual(models[1].content_type, "location_desc")
        self.assertEqual(models[1].content_text, "Test location")
        self.assertEqual(models[1].score, 0.85)
        self.assertIsNone(models[1].source_id)
        self.assertIsNone(models[1].location)


@pytest.mark.asyncio
class TestAsyncEmbeddingService:
    """Tests for async embedding service methods."""
    
    @pytest.fixture
    def service(self):
        """Create a test service instance."""
        test_index = f"test-index-{uuid4()}"
        return EmbeddingService(index_name=test_index)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {
            "text": "This is a test document for embedding generation.",
            "embedding": [0.1] * EMBEDDING_DIMENSION,
            "user_id": str(uuid4()),
            "content_type": ContentType.NOTE,
            "timestamp": datetime.utcnow(),
            "source_id": str(uuid4()),
            "location": {
                "longitude": 10.0,
                "latitude": 20.0,
                "address": "123 Test St",
                "names": ["Home", "Office"]
            },
            "metadata": {
                "file_type": "text",
                "original_filename": "test.txt"
            }
        }
    
    @patch("app.aws.embedding.EmbeddingService.generate_embeddings")
    async def test_generate_embeddings_async(self, mock_generate_embeddings, service, sample_data):
        """Test async embedding generation."""
        # Set up mock
        mock_generate_embeddings.return_value = sample_data["embedding"]
        
        # Generate embeddings
        embeddings = await service.generate_embeddings_async(sample_data["text"])
        
        # Verify embeddings
        assert len(embeddings) == EMBEDDING_DIMENSION
        assert embeddings == sample_data["embedding"]
        
        # Verify mock was called correctly
        mock_generate_embeddings.assert_called_once_with(sample_data["text"])
    
    @patch("app.aws.embedding.EmbeddingService.index_document")
    async def test_index_document_async(self, mock_index_document, service, sample_data):
        """Test async document indexing."""
        # Set up mock
        mock_index_document.return_value = "doc1"
        
        # Index document
        doc_id = await service.index_document_async(
            user_id=sample_data["user_id"],
            content_type=sample_data["content_type"],
            content_text=sample_data["text"],
            content_vector=sample_data["embedding"],
            timestamp=sample_data["timestamp"],
            source_id=sample_data["source_id"],
            location=sample_data["location"],
            metadata=sample_data["metadata"]
        )
        
        # Verify document ID
        assert doc_id == "doc1"
        
        # Verify mock was called correctly
        mock_index_document.assert_called_once_with(
            user_id=sample_data["user_id"],
            content_type=sample_data["content_type"],
            content_text=sample_data["text"],
            content_vector=sample_data["embedding"],
            timestamp=sample_data["timestamp"],
            source_id=sample_data["source_id"],
            location=sample_data["location"],
            metadata=sample_data["metadata"]
        )
    
    @patch("app.aws.embedding.EmbeddingService.search_by_vector")
    async def test_search_by_vector_async(self, mock_search_by_vector, service, sample_data):
        """Test async vector search."""
        # Set up mock
        mock_search_by_vector.return_value = [{"id": "doc1", "score": 0.95}]
        
        # Search by vector
        results = await service.search_by_vector_async(
            user_id=sample_data["user_id"],
            vector=sample_data["embedding"],
            k=5,
            content_types=[sample_data["content_type"]]
        )
        
        # Verify results
        assert results == [{"id": "doc1", "score": 0.95}]
        
        # Verify mock was called correctly
        mock_search_by_vector.assert_called_once_with(
            user_id=sample_data["user_id"],
            vector=sample_data["embedding"],
            k=5,
            content_types=[sample_data["content_type"]],
            min_score=0.7,
            date_range=None,
            location_filter=None
        )
    
    @patch("app.aws.embedding.EmbeddingService.generate_embeddings_async")
    @patch("app.aws.embedding.EmbeddingService.search_by_vector_async")
    async def test_search_by_text_async(self, mock_search_by_vector_async, mock_generate_embeddings_async, service, sample_data):
        """Test async text search."""
        # Set up mocks
        mock_generate_embeddings_async.return_value = sample_data["embedding"]
        mock_search_by_vector_async.return_value = [{"id": "doc1", "score": 0.95}]
        
        # Search by text
        results = await service.search_by_text_async(
            user_id=sample_data["user_id"],
            text=sample_data["text"],
            k=5,
            content_types=[sample_data["content_type"]]
        )
        
        # Verify results
        assert results == [{"id": "doc1", "score": 0.95}]
        
        # Verify mocks were called correctly
        mock_generate_embeddings_async.assert_called_once_with(sample_data["text"])
        mock_search_by_vector_async.assert_called_once_with(
            user_id=sample_data["user_id"],
            vector=sample_data["embedding"],
            k=5,
            content_types=[sample_data["content_type"]],
            date_range=None,
            location_filter=None
        )
    
    @patch("app.aws.embedding.EmbeddingService.generate_embeddings_async")
    @patch("app.aws.embedding.EmbeddingService.index_document_async")
    async def test_process_and_store_text(self, mock_index_document_async, mock_generate_embeddings_async, service, sample_data):
        """Test text processing and storage."""
        # Set up mocks
        mock_generate_embeddings_async.return_value = sample_data["embedding"]
        mock_index_document_async.return_value = "doc1"
        
        # Process and store text
        doc_id = await service.process_and_store_text(
            user_id=sample_data["user_id"],
            content_type=sample_data["content_type"],
            text=sample_data["text"],
            source_id=sample_data["source_id"],
            timestamp=sample_data["timestamp"],
            location=sample_data["location"],
            metadata=sample_data["metadata"]
        )
        
        # Verify document ID
        assert doc_id == "doc1"
        
        # Verify mocks were called correctly
        mock_generate_embeddings_async.assert_called_once_with(sample_data["text"])
        mock_index_document_async.assert_called_once_with(
            user_id=sample_data["user_id"],
            content_type=sample_data["content_type"],
            content_text=sample_data["text"],
            content_vector=sample_data["embedding"],
            timestamp=sample_data["timestamp"],
            source_id=sample_data["source_id"],
            location=sample_data["location"],
            metadata=sample_data["metadata"]
        )


if __name__ == "__main__":
    unittest.main()